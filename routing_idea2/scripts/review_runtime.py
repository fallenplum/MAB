"""Review only disjoint real runtime qualification; never E1 scientific acceptance."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[1]
CODE_ROOT=ROOT
sys.path.insert(0,str(ROOT/'src'))
from budget_router.inference import ModelSpec,configuration_fingerprint,_validated_usage
from budget_router.preparation import read_jsonl,validate_success_rows,file_sha,load_inputs
from budget_router.scoring import score_answer
from budget_router.execution import ExecutionStore,validate_job
from budget_router.member_execution import MemberStore,validate_member_job,validate_resample_groups,MEMBER_PROMPT_VERSION

TECHNICAL_RULE_VERSION='format_cap_screen_v1'
TECHNICAL_RULE={
    'minimum_jobs':4,
    'minimum_parsed_fraction':.5,
    'minimum_parsed_per_represented_task':1,
    'maximum_completion_cap_fraction':.5,
    'accuracy_is_not_an_acceptance_criterion':True,
}


def _project_path(path,root):
    value=Path(path)
    value=value if value.is_absolute() else root/value
    value=value.resolve()
    if not value.is_relative_to(root.resolve()):raise ValueError('Qualification source/output escaped project root')
    return value


def _code_hashes():
    paths=['scripts/review_runtime.py','src/budget_router/inference.py','src/budget_router/execution.py',
           'src/budget_router/member_execution.py','src/budget_router/scoring.py','src/budget_router/preparation.py']
    return [{'path':name,'sha256':file_sha(CODE_ROOT/name)} for name in paths]


def review(jobs_path,outputs_path,config_path,receipt_path):
    root=Path(ROOT).resolve()
    jobs_path=_project_path(jobs_path,root)
    outputs_path=_project_path(outputs_path,root)
    config_path=_project_path(config_path,root)
    receipt_path=_project_path(receipt_path,root)
    spec=ModelSpec(**json.loads(config_path.read_text('utf-8-sig')))
    spec.validate()
    jobs=read_jsonl(jobs_path)
    rows=validate_success_rows(outputs_path)
    if len(jobs)<TECHNICAL_RULE['minimum_jobs'] or any(job.get('split')!='qualification' for job in jobs):
        raise ValueError('At least four disjoint qualification jobs are required')
    ids=[job['job_id'] for job in jobs]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate qualification job IDs')
    if any(row['job_id'] not in set(ids) for row in rows):raise ValueError('Unrelated call in qualification output ledger')
    byid={row['job_id']:row for row in rows if row['status']=='success'}
    if set(byid)!=set(ids):raise ValueError('Incomplete qualification outputs')
    manifest,assignments,questions,_=load_inputs(root)
    qualification_ids={row['question_id'] for row in assignments if row['split']=='qualification'}
    if not qualification_ids:raise ValueError('No disjoint qualification partition in audited source manifest')
    member_jobs=all(job.get('prompt_version')==MEMBER_PROMPT_VERSION for job in jobs)
    if any(job.get('prompt_version')==MEMBER_PROMPT_VERSION for job in jobs) and not member_jobs:
        raise ValueError('Do not mix member and aggregator qualification protocols')
    if member_jobs:validate_resample_groups(jobs)
    store=MemberStore(outputs_path) if member_jobs else ExecutionStore(outputs_path)
    for job in jobs:
        qid=job['question_id']
        if qid not in qualification_ids:raise ValueError('Question is not assigned to the disjoint qualification split')
        q=questions[qid]
        if (job['task'],job.get('subtask'))!=(q['task'],q.get('subtask')):
            raise ValueError('Qualification job/source task mismatch')
        if member_jobs:
            validate_member_job(job,spec)
            question_hash=job['source_question_sha256']
        else:
            validate_job(job,spec,require_token_policy=True)
            question_hash=job['source_hashes']['question_sha256']
            if not job['messages'][1]['content'].startswith('Original problem:\n'+q['question']+'\n\nAnonymous candidate responses:\n'):
                raise ValueError('Qualification aggregation prompt differs from the original source question')
        if question_hash!=hashlib.sha256(q['question'].encode()).hexdigest():
            raise ValueError('Qualification source question hash mismatch')
        if store._cached(job,spec) is None:raise ValueError('Qualification output/job identity mismatch')
    parsed={'gsm8k':0,'bbh':0}
    counts={'gsm8k':0,'bbh':0}
    caps=tokens=0
    seconds=0.
    peaks=[]
    for job in jobs:
        row=byid[job['job_id']]
        result=row['result']
        if result.get('model_identity_verified') is not True or (
            result.get('model_id'),result.get('model_revision'),result.get('seed'),result.get('backend'))!=(
                spec.model_id,spec.revision,job['seed'],spec.backend):
            raise ValueError('Qualification actual served model/seed identity mismatch')
        if not isinstance(result.get('text'),str) or not result['text'].strip():raise ValueError('Empty generated output')
        usage=_validated_usage(result)
        if usage is None or not 0<usage['prompt_tokens']<=spec.max_input_tokens or not 0<usage['completion_tokens']<=spec.max_new_tokens:
            raise ValueError('Actual token usage missing or outside locked caps')
        duration=result.get('wall_time_s')
        if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<=0:
            raise ValueError('Invalid measured generation runtime')
        reached=usage['completion_tokens']>=spec.max_new_tokens
        if result.get('completion_reached_cap') is not None and result['completion_reached_cap'] is not reached:
            raise ValueError('Completion-cap metadata disagrees with actual token usage')
        task=job['task']
        q=questions[job['question_id']]
        # Only parseability is used. Numerical correctness is not a qualification criterion.
        parsed[task]+=int(score_answer(task,result['text'],q['reference'],subtask=q.get('subtask')).parsed)
        counts[task]+=1
        caps+=int(reached)
        tokens+=usage['completion_tokens']
        seconds+=duration
        if result.get('peak_cuda_bytes') is not None:
            peak=result['peak_cuda_bytes']
            if isinstance(peak,bool) or not isinstance(peak,int) or peak<0:raise ValueError('Invalid measured peak memory')
            peaks.append(peak)
    failed=[row for row in rows if row['status']!='success']
    failure_usage={'prompt_tokens':0,'completion_tokens':0}
    for row in failed:
        usage=_validated_usage(row.get('actual_usage'))
        if usage is None:raise ValueError('Qualification failed-attempt cost remains unknown')
        for key in failure_usage:failure_usage[key]+=usage[key]
    n=len(jobs)
    enough=(all(parsed[task]>=1 for task,count in counts.items() if count)
            and sum(parsed.values())>=n/2 and caps<=n/2)
    status='PASS' if enough else 'NEEDS_GENERATION_PROTOCOL_REVIEW'
    receipt={'status':status,'scope':'technical_runtime_only_not_E0_or_E1_completion',
        'model_fingerprint':configuration_fingerprint(spec),'model_id':spec.model_id,'model_revision':spec.revision,
        'jobs_path':jobs_path.relative_to(root).as_posix(),'outputs_path':outputs_path.relative_to(root).as_posix(),
        'model_config_path':config_path.relative_to(root).as_posix(),
        'jobs_sha256':file_sha(jobs_path),'outputs_sha256':file_sha(outputs_path),'model_config_sha256':file_sha(config_path),
        'split_manifest_sha256':file_sha(root/'data/derived/split_manifest.json'),
        'qualification_question_ids':sorted({job['question_id'] for job in jobs}),
        'qualification_job_ids_sha256':hashlib.sha256(json.dumps(ids,ensure_ascii=False).encode()).hexdigest(),
        'source_code':_code_hashes(),'jobs':n,'question_count':len({job['question_id'] for job in jobs}),
        'parsed_outputs_by_task':parsed,'outputs_by_task':counts,'completion_cap_reached':caps,
        'completion_tokens':tokens,'generation_seconds':seconds,'measured_output_tokens_per_s':tokens/seconds,
        'peak_cuda_bytes':max(peaks) if peaks else None,'failed_attempt_count':len(failed),
        'failed_attempt_actual_usage':failure_usage,'accuracy_was_not_an_acceptance_criterion':True,
        'technical_rule_version':TECHNICAL_RULE_VERSION,'technical_rule':TECHNICAL_RULE,
        'member_modes':sorted({job['member_mode'] for job in jobs}) if member_jobs else [],
        'reviewed_at_unix':time.time()}
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text('utf-8-sig'))
        ignore={'reviewed_at_unix'}
        if {k:v for k,v in old.items() if k not in ignore}=={k:v for k,v in receipt.items() if k not in ignore}:
            print(json.dumps(old,ensure_ascii=False),flush=True)
            return 0 if old['status']=='PASS' else 3
        if (root/'protocol/E0_LOCK.json').exists():raise ValueError('Locked runtime receipt changed; explicit protocol version reconciliation required')
    temporary=receipt_path.with_name(receipt_path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(receipt_path)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    return 0 if status=='PASS' else 3


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--jobs',type=Path,required=True)
    parser.add_argument('--outputs',type=Path,required=True)
    parser.add_argument('--model-config',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    return review(args.jobs,args.outputs,args.model_config,args.receipt)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status':'blocked','error_type':type(error).__name__,
                          'reason_code':'runtime_qualification_validation_failed','scientific_completion':False}),file=sys.stderr)
        raise SystemExit(2) from None
