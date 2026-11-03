"""Evidence-backed E0 lock; missing real assets fail closed."""
from pathlib import Path
from dataclasses import replace
from contextlib import contextmanager
import json,time,hashlib,os
from .preparation import load_inputs,select_records,collect_members,file_sha,read_spec
from .inference import ModelSpec,configuration_fingerprint,validate_local_model_source
from .scoring import score_answer,SCORER_VERSION
from .execution import AGGREGATION_PROMPT_VERSION


def verify_qualification(root,path,spec,*,role,expected_jobs_path,expected_outputs_path,expected_config_path,expected_alias=None):
    from .preparation import read_jsonl,validate_success_rows
    from .execution import ExecutionStore,validate_job
    from .member_execution import MemberStore,validate_member_job,MEMBER_PROMPT_VERSION
    root=Path(root).resolve();path=Path(path)
    receipt=json.loads(path.read_text('utf-8-sig'))
    if role not in {'member','aggregator'}:raise ValueError('Explicit qualification role required')
    if receipt.get('status')!='PASS' or receipt.get('model_fingerprint')!=configuration_fingerprint(spec):
        raise ValueError('Actual reviewed runtime qualification required for exact model spec')
    required_n=8 if role=='member' else 4
    if (receipt.get('scope')!='technical_runtime_only_not_E0_or_E1_completion' or
        receipt.get('jobs')!=required_n or receipt.get('question_count')!=required_n):
        raise ValueError('Incomplete role-specific runtime qualification scope/count')
    expected_paths={'jobs':Path(expected_jobs_path),'outputs':Path(expected_outputs_path),'model_config':Path(expected_config_path)}
    bound={}
    for key in ('jobs','outputs','model_config'):
        source=receipt.get(key+'_path')
        if not source:raise ValueError('Qualification receipt needs source paths and hashes')
        target=Path(source);target=target if target.is_absolute() else root/target
        expected=expected_paths[key];expected=expected if expected.is_absolute() else root/expected
        if (not target.resolve().is_relative_to(root) or target.resolve()!=expected.resolve() or
            file_sha(target)!=receipt.get(key+'_sha256')):
            raise ValueError('Qualification source hash/path mismatch')
        bound[key]=target
    if receipt.get('split_manifest_sha256')!=file_sha(root/'data/derived/split_manifest.json'):
        raise ValueError('Qualification belongs to another split manifest')
    required_code={'scripts/review_runtime.py','src/budget_router/inference.py','src/budget_router/execution.py',
                   'src/budget_router/member_execution.py','src/budget_router/scoring.py','src/budget_router/preparation.py'}
    code=receipt.get('source_code',[])
    if len(code)!=len(required_code) or {x.get('path') for x in code}!=required_code:
        raise ValueError('Qualification current reviewer/source-code inventory required')
    for item in code:
        target=(root/item['path']).resolve()
        if not target.is_relative_to(root) or file_sha(target)!=item.get('sha256'):
            raise ValueError('Qualification source-code hash changed')
    _,assignments,questions,_=load_inputs(root)
    expected_rows=[r for r in assignments if r['split']=='qualification' and (role=='member' or r['task']=='bbh')]
    expected_ids={r['question_id'] for r in expected_rows}
    expected_counts={'gsm8k':4 if role=='member' else 0,'bbh':4}
    if len(expected_rows)!=required_n or len(expected_ids)!=required_n:
        raise ValueError('Frozen qualification partition is incomplete')
    jobs=read_jsonl(bound['jobs']);outputs=validate_success_rows(bound['outputs'])
    if len(jobs)!=required_n or {x['question_id'] for x in jobs}!=expected_ids or len({x['job_id'] for x in jobs})!=required_n:
        raise ValueError('Qualification jobs are not the exact role-specific disjoint partition')
    if (receipt.get('qualification_question_ids')!=sorted(expected_ids) or receipt.get('outputs_by_task')!=expected_counts):
        raise ValueError('Qualification task/question coverage mismatch')
    ids={j['job_id'] for j in jobs}
    if any(row.get('job_id') not in ids for row in outputs):
        raise ValueError('Qualification outputs include unrelated calls')
    if {row['job_id'] for row in outputs if row['status']=='success'}!=ids:
        raise ValueError('Qualification successful output coverage incomplete')
    store=MemberStore(bound['outputs']) if role=='member' else ExecutionStore(bound['outputs'])
    observed={'gsm8k':0,'bbh':0}
    for job in jobs:
        q=questions[job['question_id']]
        if job.get('split')!='qualification' or (job.get('task'),job.get('subtask'))!=(q['task'],q.get('subtask')):
            raise ValueError('Qualification split/task differs from current source')
        expected_question_sha=hashlib.sha256(q['question'].encode()).hexdigest()
        if role=='member':
            validate_member_job(job,spec)
            if (job.get('model_alias')!=expected_alias or job.get('sample_index')!=0 or
                job.get('sampling_control',False) is not False or job.get('seed')!=42):
                raise ValueError('Qualification member role/seed does not match base generation protocol')
            question_sha=job['source_question_sha256']
        else:
            validate_job(job,spec,require_token_policy=True)
            question_sha=job['source_hashes']['question_sha256']
        if question_sha!=expected_question_sha:raise ValueError('Qualification question content changed')
        if store._cached(job,spec) is None:raise ValueError('Qualification real output/job identity mismatch')
        observed[job['task']]+=1
    if observed!=expected_counts:raise ValueError('Qualification per-task coverage incomplete')
    return {'path':path.relative_to(root).as_posix(),'sha256':file_sha(path),'role':role,'jobs':required_n,
            'model_fingerprint':configuration_fingerprint(spec)}

@contextmanager
def _freeze_guard(root):
    path=Path(root)/'protocol/E0_FREEZE.guard.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if stream.seek(0,os.SEEK_END)==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:raise ValueError('Another E0 freeze transaction is active') from None
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

def _atomic_bytes(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.atomic_tmp')
    try:
        with pending.open('wb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(pending,path)
    finally:pending.unlink(missing_ok=True)

def _transaction_paths(root):
    root=Path(root).resolve()
    values={
      'config':root/'configs/experiment_v1.json',
      'destination':root/'protocol/E0_LOCK.json',
      'backup':root/'configs/experiment_v1.json.pre_e0_backup',
      'config_pending':root/'configs/experiment_v1.json.pending',
      'lock_pending':root/'protocol/E0_LOCK.json.pending',
      'journal':root/'protocol/.E0_COMMIT.transaction.json'}
    if any(not p.resolve().is_relative_to(root) for p in values.values()):
        raise ValueError('E0 transaction path escaped project root')
    return values

def _clean_transaction(paths):
    for name in ('backup','config_pending','lock_pending','journal'):
        paths[name].unlink(missing_ok=True)

def _recover_e0_transaction(root):
    paths=_transaction_paths(root)
    if not paths['journal'].exists():
        # No config replacement happens before this journal is committed.
        _clean_transaction(paths)
        return
    journal=json.loads(paths['journal'].read_text('utf-8'))
    if paths['destination'].exists():
        if (file_sha(paths['destination'])!=journal['lock_sha256'] or
            file_sha(paths['config'])!=journal['new_config_sha256']):
            raise ValueError('Incomplete E0 commit conflicts with final files; explicit reconciliation required')
        _clean_transaction(paths)
        return
    if not paths['backup'].is_file() or file_sha(paths['backup'])!=journal['old_config_sha256']:
        raise ValueError('Interrupted E0 commit has no verified rollback backup')
    if file_sha(paths['config']) not in {journal['old_config_sha256'],journal['new_config_sha256']}:
        raise ValueError('Config changed outside interrupted E0 transaction; preserve it for review')
    _atomic_bytes(paths['config'],paths['backup'].read_bytes())
    _clean_transaction(paths)

def _commit_e0_pair(root,old_config,proposed_config,lock):
    paths=_transaction_paths(root)
    if paths['destination'].exists():raise ValueError('Refusing to replace an existing E0 lock')
    if paths['config'].read_bytes()!=old_config:raise ValueError('Experiment config changed during E0 preflight')
    lock_bytes=json.dumps(lock,ensure_ascii=False,indent=2).encode('utf-8')
    journal={'version':1,'old_config_sha256':hashlib.sha256(old_config).hexdigest(),
             'new_config_sha256':hashlib.sha256(proposed_config).hexdigest(),
             'lock_sha256':hashlib.sha256(lock_bytes).hexdigest()}
    try:
        _atomic_bytes(paths['lock_pending'],lock_bytes)
        _atomic_bytes(paths['backup'],old_config)
        _atomic_bytes(paths['journal'],json.dumps(journal,sort_keys=True).encode('utf-8'))
        _atomic_bytes(paths['config_pending'],proposed_config)
        os.replace(paths['config_pending'],paths['config'])
        os.replace(paths['lock_pending'],paths['destination'])
    except Exception:
        if paths['journal'].exists():_recover_e0_transaction(root)
        else:_clean_transaction(paths)
        raise
    _clean_transaction(paths)

def freeze_e0(root):
    root=Path(root).resolve()
    with _freeze_guard(root):
        _recover_e0_transaction(root)
        return _freeze_e0_guarded(root)

def _freeze_e0_guarded(root):
    root=Path(root);manifest,records,questions,pool=load_inputs(root)
    _,e1,_,_=select_records(root,'e1')
    answers,_=collect_members(root,'e1',e1,pool)
    if len(answers)!=300 or any(len(x)!=6 for x in answers.values()):raise ValueError('Complete real member coverage required')
    config_path=root/'configs/experiment_v1.json';old_config_bytes=config_path.read_bytes();config=json.loads(old_config_bytes.decode('utf-8-sig'))
    if config.get('scoring',{}).get('version')!=SCORER_VERSION:
        raise ValueError('Experiment scoring version differs from current scorer')
    # Validate only non-sealed references before expensive weight provenance checks.
    # Shared input loading is unchanged; this loop never accesses sealed questions.
    checked_reference_splits=('qualification','e1_selection_calibration','e1_diagnostic_evaluation',
                              'online_calibration','online_stream')
    reference_counts={split:0 for split in checked_reference_splits};deferred_reference_count=0
    for record in records:
        split=record['split']
        if split=='sealed_test':
            deferred_reference_count+=1
            continue
        if split not in reference_counts:raise ValueError('Unknown split in E0 reference preflight')
        q=questions[record['question_id']]
        result=score_answer(q['task'],q['reference'],q['reference'],subtask=q['subtask'])
        if not result.parsed or result.score!=1:raise ValueError('Official reference fails locked scorer roundtrip')
        reference_counts[split]+=1
    reference_preflight={'schema':'e0_nonsealed_reference_preflight_v1','status':'PASS_NON_SEALED_REFERENCE_ROUNDTRIP',
      'validated_splits':list(checked_reference_splits),'validated_questions_by_split':reference_counts,
      'validated_questions':sum(reference_counts.values()),'deferred_splits':['sealed_test'],
      'deferred_questions':deferred_reference_count,'sealed_questions_accessed_by_preflight':False,
      'deferred_validation_stage':'released_final_grading_after_final_release_and_frozen_decisions',
      'scope':'Reference format roundtrip only; shared source loading unchanged; no model prediction scoring.'}
    agg=ModelSpec(**json.loads((root/'configs/aggregator_v1.json').read_text('utf-8-sig')))
    qualifications=[];models=[];member_specs=[]
    for entry in pool:
        spec=read_spec(root,entry);validate_local_model_source(spec);member_specs.append(spec)
        qualifications.append(verify_qualification(root,root/'artifacts/qualification'/f"member_{entry['alias'].lower()}.json",spec,
            role='member',expected_alias=entry['alias'],
            expected_jobs_path=root/'data/jobs/qualification'/f"members_{entry['alias'].lower()}.jsonl",
            expected_outputs_path=root/'results/raw/qualification'/f"members_{entry['alias'].lower()}.jsonl",
            expected_config_path=root/entry['config']))
        models.append({'alias':entry['alias'],'model_id':spec.model_id,'revision':spec.revision,
                       'fingerprint':configuration_fingerprint(spec),
                       'sampling_control_fingerprint':configuration_fingerprint(replace(spec,temperature=.7,top_p=.95))})
    validate_local_model_source(agg)
    qualifications.append(verify_qualification(root,root/'artifacts/qualification/aggregator.json',agg,
        role='aggregator',expected_jobs_path=root/'data/jobs/qualification/historical_runtime_qualification.jsonl',
        expected_outputs_path=root/'results/raw/qualification/aggregator.jsonl',
        expected_config_path=root/'configs/aggregator_v1.json'))
    if config['n_models']!=6 or config['k']!=3 or config['split']['e1_development']['total']!=300:
        raise ValueError('Original experiment scale changed')
    required_sources=['src/budget_router/inference.py','src/budget_router/execution.py',
      'src/budget_router/member_execution.py','src/budget_router/scoring.py',
      'src/budget_router/diagnostics.py','src/budget_router/preparation.py','src/budget_router/e1_controls.py',
      'src/budget_router/protocol_lock.py','scripts/review_runtime.py','scripts/freeze_e0.py',
      'scripts/run_local_e1.py','scripts/prepare_jobs.py','scripts/prepare_e1_controls.py',
      'scripts/run_members.py','scripts/run_aggregation.py','scripts/evaluate_real_e1.py',
      'src/budget_router/runtime_memory.py']
    config['pool'].update(model_ids=[x['model_id'] for x in models],model_revisions=[x['revision'] for x in models],
        cache_revision='local_frozen_generation_v1',
        selection_rule='Six publicly accessible candidate aliases fixed from cache coverage and model-family diversity before E1; both tasks regenerated with exact pinned revisions and common NF4 protocol; historical cache only audit.')
    config['split']['split_manifest_sha256']=file_sha(root/'data/derived/split_manifest.json')
    config['aggregator'].update(qualification_status='PASS_TECHNICAL_RUNTIME',prompt_revision=AGGREGATION_PROMPT_VERSION)
    member_caps={(s.max_input_tokens,s.max_new_tokens) for s in member_specs}
    if len(member_caps)!=1:raise ValueError('Common preregistered member token caps required')
    member_input,member_output=next(iter(member_caps))
    service_tokens=3*(member_input+member_output)+agg.max_input_tokens+agg.max_new_tokens
    audit_tokens=member_input+member_output+agg.max_input_tokens+agg.max_new_tokens
    config['budget'].update(input_cap_tokens=member_input,member_max_output_tokens=member_output,
       aggregator_input_cap_tokens=agg.max_input_tokens,aggregator_max_output_tokens=agg.max_new_tokens,
       total_execution_budget={'executions':(600+32)*4+300*2,'tokens':(600+32)*service_tokens+300*audit_tokens},
       cost_weights={'executions':1.0,'labels':1.0,'tokens':0.001},
       member_call_cap=1,aggregator_call_cap=1,
       resource_cap_derivation='600 normal services + at most32 paid warmup services + at most300 one-replacement audits; worstcase percall input/output caps; label-budget further restricts audits',
       scalar_cost_interpretation='engineering resource units: one call + one feedback label + 0.001 logical input/output tokens; not USD; physical GPU seconds reported separately')
    config['status']='E0_LOCKED_REAL_MEMBERS_AND_RUNTIME_PASS'
    config['pending_locks']=[]
    config['resource_decision']['status']='LOCAL_RUNTIME_QUALIFIED'
    proposed_config=json.dumps(config,ensure_ascii=False,indent=2).encode('utf-8')
    lock={'status':'PASS','stage':'E0','scientific_completion':False,'N':6,'K':3,'combinations':20,
      'e1_questions':300,'formal_member_answer_source':'BOTH_TASKS_REGENERATED_PINNED_NF4_NO_HISTORICAL_ANSWERS',
      'pool_config_sha256':file_sha(root/'configs/pool_v1.json'),
      'experiment_config_sha256':hashlib.sha256(proposed_config).hexdigest(),
      'split_manifest_sha256':file_sha(root/'data/derived/split_manifest.json'),
      'aggregator_fingerprint':configuration_fingerprint(agg),'models':models,
      'qualification_receipts':qualifications,'cache_revision':'local_frozen_generation_v1',
      'scorer_version':config['scoring']['version'],
      'reference_preflight':reference_preflight,
      'source_code':[{'path':x,'sha256':file_sha(root/x)} for x in required_sources],
      'member_outputs':[{'path':f"results/raw/e1/members_{e['alias'].lower()}.jsonl",'sha256':file_sha(root/'results/raw/e1'/f"members_{e['alias'].lower()}.jsonl")} for e in pool]}
    destination=root/'protocol/E0_LOCK.json'
    if destination.exists():
        previous=json.loads(destination.read_text('utf-8-sig'))
        if {k:v for k,v in previous.items() if k!='locked_at_unix'}!=lock:
            raise ValueError('Existing E0 lock differs; reconcile before changing any formal protocol')
        if file_sha(config_path)!=lock['experiment_config_sha256']:raise ValueError('Existing frozen config bytes changed')
        return previous
    lock['locked_at_unix']=time.time()
    _commit_e0_pair(root,old_config_bytes,proposed_config,lock)
    return lock
