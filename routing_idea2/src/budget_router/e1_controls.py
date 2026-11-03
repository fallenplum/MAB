"""Calibration-only model selection and three real, fresh independent-sample controls."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import math
import numpy as np
from .preparation import select_records,collect_members,read_spec,immutable_jsonl,file_sha,validate_success_rows,read_jsonl
from .member_execution import make_member_job,validate_member_job,validate_resample_groups
from .inference import ModelSpec,configuration_fingerprint,_validated_usage
from .scoring import score_answer


def _select_calibration_model(pool,records,questions,answers):
    cal=[row for row in records if row['split']=='e1_selection_calibration']
    if len(cal)!=100 or len({row['question_id'] for row in cal})!=100:
        raise ValueError('Only full unique 100-question E1 selection split may choose strongest model')
    if any(sum(row['task']==task for row in cal)!=50 for task in ('gsm8k','bbh')):
        raise ValueError('Strongest model selection requires exactly 50 calibration questions per task')
    scores={}
    for entry in pool:
        per_task={}
        for task in ('gsm8k','bbh'):
            group=[row for row in cal if row['task']==task]
            per_task[task]=float(np.mean([score_answer(task,answers[row['question_id']][entry['alias']],
                questions[row['question_id']]['reference'],subtask=row['subtask']).score for row in group]))
        scores[entry['alias']]={'macro':sum(per_task.values())/2,'per_task':per_task}
    # The diagnostic-evaluation rows/answers/references never enter this ranking.
    strong=max(sorted(pool,key=lambda entry:entry['index']),key=lambda entry:scores[entry['alias']]['macro'])
    return strong,scores,cal


def select_strong_model(root):
    root=Path(root)
    _,records,questions,pool=select_records(root,'e1')
    answers,_=collect_members(root,'e1',records,pool)
    strong,scores,cal=_select_calibration_model(pool,records,questions,answers)
    receipt={'status':'SELECTED_ON_E1_CALIBRATION_ONLY','model_alias':strong['alias'],'model_index':strong['index'],
      'selection_rule':'highest_equal_task_macro_individual_score_on_100_calibration_questions; tie_break_pool_index',
      'selection_question_ids':[row['question_id'] for row in cal],'diagnostic_evaluation_scores_used':False,
      'split_manifest_sha256':file_sha(root/'data/derived/split_manifest.json'),
      'pool_config_sha256':file_sha(root/'configs/pool_v1.json'),'calibration_member_scores':scores,
      'source_member_outputs':[{'path':f"results/raw/e1/members_{entry['alias'].lower()}.jsonl",
         'sha256':file_sha(root/'results/raw/e1'/f"members_{entry['alias'].lower()}.jsonl")} for entry in pool]}
    path=root/'artifacts/audit/e1_strong_model_selection.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(receipt,ensure_ascii=False,indent=2)
    if path.exists() and path.read_text('utf-8')!=text:raise ValueError('Existing calibration selection/source evidence differs')
    if not path.exists():path.write_text(text,encoding='utf-8')
    return strong,receipt


def _sample_seed(question_id,index):
    return int(hashlib.sha256(f'independent-control-v1:{question_id}:{index}'.encode()).hexdigest()[:8],16)


def prepare_strong_samples(root,*,qualification=False):
    root=Path(root)
    strong,_=select_strong_model(root)
    stage='qualification' if qualification else 'e1'
    _,records,questions,_=select_records(root,stage)
    if not qualification:records=[row for row in records if row['split']=='e1_diagnostic_evaluation']
    spec=read_spec(root,strong,sampling=True)
    config=root/'configs/e1_strong_sampling.json'
    text=json.dumps(asdict(spec),ensure_ascii=False,indent=2)
    if config.exists() and config.read_text('utf-8')!=text:raise ValueError('Strong sampling config changed')
    if not config.exists():config.write_text(text,encoding='utf-8')
    jobs=[]
    for row in records:
        q=questions[row['question_id']]
        for index in range(3):
            jobs.append(make_member_job(question_id=q['question_id'],task=q['task'],subtask=q['subtask'],
                question=q['question'],model_alias=strong['alias'],seed=_sample_seed(q['question_id'],index),
                model_spec=spec,split=row['split'],sample_index=index,sampling_control=True))
    validate_resample_groups(jobs)
    return immutable_jsonl(root/'data/jobs'/stage/'strong_samples.jsonl',jobs)


def _validate_sample_result(row,expected,spec):
    for field in ('job_id','question_id','task','subtask','split','model_alias','sample_index','seed',
                  'model_fingerprint','member_mode','sampling_control','prompt_version','source_question_sha256'):
        if row.get(field)!=expected[field]:raise ValueError('Strong sample output/job identity or sampling protocol mismatch')
    result=row['result']
    if (result.get('model_id'),result.get('model_revision'),result.get('backend'),result.get('seed'))!=(
        spec.model_id,spec.revision,spec.backend,expected['seed']) or result.get('model_identity_verified') is not True:
        raise ValueError('Strong sample actual served model/seed mismatch')
    usage=_validated_usage(result)
    if usage is None or not 0<usage['prompt_tokens']<=spec.max_input_tokens or not 0<=usage['completion_tokens']<=spec.max_new_tokens:
        raise ValueError('Strong sample actual generation usage is unknown or invalid')
    if not isinstance(result.get('text'),str) or not isinstance(result.get('wall_time_s'),(int,float)) or not math.isfinite(result['wall_time_s']) or result['wall_time_s']<0:
        raise ValueError('Invalid actual strong sample output/runtime')
    return usage


def prepare_strong_aggregation(root):
    from transformers import AutoTokenizer
    from .execution import make_job
    root=Path(root)
    strong,_=select_strong_model(root)
    _,records,questions,_=select_records(root,'e1')
    records=[row for row in records if row['split']=='e1_diagnostic_evaluation']
    if len(records)!=200:raise ValueError('All 200 independent E1 evaluation questions are required')
    sampling_spec=read_spec(root,strong,sampling=True)
    jobs_by_key={}
    for job in read_jsonl(root/'data/jobs/e1/strong_samples.jsonl'):
        validate_member_job(job,sampling_spec)
        key=(job['question_id'],job['sample_index'])
        if key in jobs_by_key:raise ValueError('Duplicated planned strong-control sample')
        jobs_by_key[key]=job
    required={(row['question_id'],index) for row in records for index in range(3)}
    if set(jobs_by_key)!=required:raise ValueError('Strong control requires exactly three planned calls on each E1 evaluation question')
    expected_by_id={}
    for row in records:
        q=questions[row['question_id']]
        for index in range(3):
            expected=make_member_job(question_id=q['question_id'],task=q['task'],subtask=q['subtask'],question=q['question'],
                model_alias=strong['alias'],seed=_sample_seed(q['question_id'],index),model_spec=sampling_spec,
                split=row['split'],sample_index=index,sampling_control=True)
            if jobs_by_key[(q['question_id'],index)]!=expected:
                raise ValueError('Strong-control planned job differs from pinned question/sample/seed protocol')
            expected_by_id[expected['job_id']]=expected
    validate_resample_groups(list(jobs_by_key.values()))
    source=root/'results/raw/e1/strong_samples.jsonl'
    source_sha=file_sha(source)
    sampled={};provenance={};seen=set()
    for row in validate_success_rows(source):
        expected=expected_by_id.get(row['job_id'])
        if expected is None:raise ValueError('Unexpected call in strong-control source ledger')
        if row['status']!='success':continue
        usage=_validate_sample_result(row,expected,sampling_spec)
        if row['job_id'] in seen:raise ValueError('Duplicated successful independent sample')
        seen.add(row['job_id'])
        qid,index=row['question_id'],row['sample_index']
        key=f"{strong['alias']}::sample{index}"
        sampled.setdefault(qid,{})[key]=row['result']['text']
        provenance.setdefault(qid,{})[key]={'job_id':row['job_id'],'record_sha256':row['record_sha256'],
            'sample_index':index,'generation_seed':row['seed'],'model_id':sampling_spec.model_id,
            'revision':sampling_spec.revision,'model_fingerprint':row['model_fingerprint'],
            'prompt_tokens':usage['prompt_tokens'],'completion_tokens':usage['completion_tokens'],
            'wall_time_s':row['result']['wall_time_s'],
            'completion_reached_cap':row['result'].get('completion_reached_cap'),
            'cost_is_original_full_member_generation_not_clipped_candidate':True}
    if seen!=set(expected_by_id):raise ValueError('Missing one or more real independent strong-control sample calls')
    spec=ModelSpec(**json.loads((root/'configs/aggregator_v1.json').read_text('utf-8-sig')))
    tokenizer=AutoTokenizer.from_pretrained(spec.model_path,local_files_only=True,trust_remote_code=False)
    selection_sha=file_sha(root/'artifacts/audit/e1_strong_model_selection.json')
    jobs=[]
    for row in records:
        q=questions[row['question_id']];qid=q['question_id']
        p={'control':'strong_model_resample','selection_receipt_sha256':selection_sha,
           'selected_from_calibration_sha256':selection_sha,'source_samples_store_path':'results/raw/e1/strong_samples.jsonl',
           'source_samples_store_sha256':source_sha,'samples':provenance[qid]}
        jobs.append(make_job(question_id=qid,task=q['task'],subtask=q['subtask'],question=q['question'],answers=sampled[qid],
            seed=42,model_spec=spec,split=row['split'],tokenizer=tokenizer,member_provenance=p))
    return immutable_jsonl(root/'data/jobs/e1/strong_aggregations.jsonl',jobs)
