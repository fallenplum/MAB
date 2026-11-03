"""Label-free immutable jobs for pinned, real model executions.

References are read only by the explicit calibration selection function, never
serialized into any generation job. E1 and final-release guards are explicit.
"""
from pathlib import Path
from dataclasses import asdict,replace
from itertools import combinations
import hashlib,json,unicodedata,math
from .inference import ModelSpec,configuration_fingerprint,_validated_usage


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_inputs(root):
    root=Path(root)
    manifest_path=root/'data/derived/split_manifest.json'
    manifest=json.loads(manifest_path.read_text('utf-8-sig'))
    audit=json.loads((root/'artifacts/audit/split_audit.json').read_text('utf-8-sig'))
    if audit.get('manifest_sha256')!=file_sha(manifest_path):raise ValueError('Split manifest/audit hash mismatch')
    for source in manifest['source_files']:
        path=(root/source['path']).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha(path)!=source['sha256']:
            raise ValueError('Split source asset changed or escaped project root')
        if source.get('receipt_path') and file_sha(root/source['receipt_path'])!=source['receipt_sha256']:
            raise ValueError('Split source receipt changed')
    records=manifest['records']
    if len({x['question_id'] for x in records})!=len(records):raise ValueError('Duplicated split IDs')
    questions={}
    for name in ('available_records.jsonl','gsm8k_questions.jsonl'):
        for row in read_jsonl(root/'data/derived'/name):
            qid=row['question_id']
            if qid in questions:raise ValueError('Duplicate source question ID')
            questions[qid]=row
    for row in records:
        q=questions[row['question_id']]
        if (q['task'],q['subtask'],q['source_split'])!=(row['task'],row['subtask'],row['source_split']):
            raise ValueError('Split/source mismatch')
        canonical=' '.join(unicodedata.normalize('NFKC',q['question']).split())
        if hashlib.sha256(canonical.encode()).hexdigest()!=manifest['canonical_question_fingerprints'][row['question_id']]:
            raise ValueError('Canonical question content changed')
    if len(set(manifest['canonical_question_fingerprints'].values()))!=len(records):
        raise ValueError('Canonical question fingerprints are not globally disjoint')
    pool=json.loads((root/'configs/pool_v1.json').read_text('utf-8-sig'))['models']
    if len(pool)!=6 or [x['index'] for x in pool]!=list(range(6)):raise ValueError('Frozen six-model order required')
    return manifest,records,questions,pool


def select_records(root,stage):
    manifest,records,questions,pool=load_inputs(root)
    stage_splits={'qualification':{'qualification'},'e1':{'e1_selection_calibration','e1_diagnostic_evaluation'},
                  'online_calibration':{'online_calibration'},'online_stream':{'online_stream'}}
    if stage not in stage_splits:raise ValueError('Unknown stage; sealed test requires separate release gate')
    if stage.startswith('online_'):
        gate=json.loads((Path(root)/'results/e1/e1_diagnostics.json').read_text('utf-8-sig'))
        if gate.get('status')!='PASS':
            raise ValueError('E1 PASS required before preparing later-stage outputs')
        evidence=gate.get('evidence',{})
        aggregator=ModelSpec(**json.loads((Path(root)/'configs/aggregator_v1.json').read_text('utf-8-sig')))
        aggregator.validate()
        if (evidence.get('split_manifest_sha256')!=file_sha(Path(root)/'data/derived/split_manifest.json') or
            evidence.get('aggregator_revision')!=aggregator.revision or
            evidence.get('aggregator_fingerprint')!=configuration_fingerprint(aggregator)):
            raise ValueError('E1 PASS belongs to another split/aggregator configuration')
        lock_path=Path(root)/'protocol/E0_LOCK.json'
        if evidence.get('e0_lock_sha256')!=file_sha(lock_path):raise ValueError('E1 PASS belongs to another E0 lock')
        lock=json.loads(lock_path.read_text('utf-8-sig'))
        if lock.get('pool_config_sha256')!=file_sha(Path(root)/'configs/pool_v1.json'):
            raise ValueError('E0 locked pool changed')
    selected=[x for x in records if x['split'] in stage_splits[stage]]
    required={'qualification':8,'e1':300,'online_calibration':200,'online_stream':600}[stage]
    if len(selected)!=required:raise ValueError('Full-scale split cardinality mismatch')
    quotas={'qualification':{'qualification':4},
            'e1':{'e1_selection_calibration':50,'e1_diagnostic_evaluation':100},
            'online_calibration':{'online_calibration':100},
            'online_stream':{'online_stream':300}}[stage]
    for task in ('gsm8k','bbh'):
        for split,per_task in quotas.items():
            if sum(r['task']==task and r['split']==split for r in selected)!=per_task:
                raise ValueError('Frozen per-task/per-split cardinality mismatch')
    if any(r['task'] not in {'gsm8k','bbh'} for r in selected):
        raise ValueError('Unexpected task in frozen split')
    return manifest,selected,questions,pool


def read_spec(root,entry,*,sampling=False):
    spec=ModelSpec(**json.loads((Path(root)/entry['config']).read_text('utf-8-sig')))
    if (spec.model_id,spec.revision)!=(entry['model_id'],entry['revision']):raise ValueError('Pool/config mismatch')
    spec.validate()
    if configuration_fingerprint(spec)!=entry.get('fingerprint'):
        raise ValueError('Pool/base configuration fingerprint mismatch')
    if sampling:spec=replace(spec,temperature=0.7,top_p=0.95)
    spec.validate();return spec


def immutable_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in rows)
    if path.exists():
        if path.read_text('utf-8')!=content:raise ValueError('Existing immutable job batch differs; never overwrite provenance')
    else:path.write_text(content,encoding='utf-8',newline='\n')
    return {'path':str(path),'jobs':len(rows),'sha256':file_sha(path)}


def validate_success_rows(path):
    rows=read_jsonl(path);seen=set()
    for row in rows:
        digest=row.get('record_sha256')
        plain={k:v for k,v in row.items() if k!='record_sha256'}
        expected=hashlib.sha256(json.dumps(plain,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        if digest!=expected:raise ValueError('Output record hash mismatch')
        if row.get('cost_status')=='unknown_must_reconcile':raise ValueError('Unreconciled failed call')
        if row.get('status') not in {'success','failure'}:
            raise ValueError('Unrecognized execution record status')
        if row['status']=='failure' and (row.get('cost_status')!='recorded_actual_usage' or
                                         _validated_usage(row.get('actual_usage')) is None):
            raise ValueError('Failed call requires reconciled actual usage')
        if row['status']=='success':
            if row.get('cost_status')!='recorded_actual_usage':
                raise ValueError('Successful call requires recorded actual usage')
            if row['job_id'] in seen:raise ValueError('Duplicated successful call')
            seen.add(row['job_id'])
            if not row['result'].get('model_identity_verified'):raise ValueError('Unverified served model')
    return rows


def prepare_members(root,stage,*,only_alias=None):
    from .member_execution import make_member_job
    root=Path(root);_,records,questions,pool=select_records(root,stage)
    receipts=[]
    for entry in pool:
        if only_alias is not None and entry['alias']!=only_alias:continue
        spec=read_spec(root,entry);jobs=[]
        for row in records:
            q=questions[row['question_id']]
            jobs.append(make_member_job(question_id=q['question_id'],task=q['task'],subtask=q['subtask'],
                question=q['question'],model_alias=entry['alias'],seed=42,model_spec=spec,split=row['split'],sample_index=0))
        receipts.append(immutable_jsonl(root/'data/jobs'/stage/f"members_{entry['alias'].lower()}.jsonl",jobs))
    if not receipts:raise ValueError('Pool alias not found')
    return receipts


def collect_members(root,stage,records,pool):
    from .member_execution import make_member_job
    root=Path(root);_,_,questions,_=load_inputs(root);record_map={r['question_id']:r for r in records}
    required=set(record_map);answers={qid:{} for qid in required}
    provenance={qid:{} for qid in required}
    for entry in pool:
        spec=read_spec(root,entry)
        rows=validate_success_rows(root/'results/raw'/stage/f"members_{entry['alias'].lower()}.jsonl")
        for row in rows:
            if row['status']!='success':continue
            qid=row['question_id']
            if qid not in required:raise ValueError('Unexpected question in stage member store')
            if row['model_alias']!=entry['alias'] or row['model_fingerprint']!=configuration_fingerprint(spec) or row['sample_index']!=0:
                raise ValueError('Member stage/model/sampling mismatch')
            source=questions[qid];assigned=record_map[qid]
            expected=make_member_job(question_id=qid,task=source['task'],subtask=source['subtask'],question=source['question'],
                model_alias=entry['alias'],seed=42,model_spec=spec,split=assigned['split'],sample_index=0)
            if row['job_id']!=expected['job_id']:raise ValueError('Member output belongs to a different question/prompt/protocol')
            for field in ('question_id','task','subtask','split','model_alias','sample_index','seed','model_fingerprint',
                          'member_mode','prompt_version','source_question_sha256'):
                if row.get(field)!=expected[field]:raise ValueError('Member output stage/identity metadata mismatch')
            result=row['result']
            if (result.get('model_id'),result.get('model_revision'),result.get('seed'),result.get('backend'))!=(spec.model_id,spec.revision,42,spec.backend):
                raise ValueError('Member actual served-model identity mismatch')
            usage=_validated_usage(result)
            if (usage is None or usage['prompt_tokens']>spec.max_input_tokens or
                usage['completion_tokens']>spec.max_new_tokens):
                raise ValueError('Member actual usage is missing or outside locked limits')
            if (not isinstance(result.get('text'),str) or not result['text'].strip() or
                not isinstance(result.get('wall_time_s'),(int,float)) or
                isinstance(result.get('wall_time_s'),bool) or not math.isfinite(result['wall_time_s']) or
                result['wall_time_s']<0):
                raise ValueError('Member output text/runtime invalid')
            if entry['alias'] in answers[qid]:raise ValueError('Multiple member answers for a fixed question')
            answers[qid][entry['alias']]=row['result']['text']
            provenance[qid][entry['alias']]={'job_id':row['job_id'],'record_sha256':row['record_sha256'],
                'model_id':spec.model_id,'revision':spec.revision,'model_fingerprint':row['model_fingerprint']}
    if any(len(x)!=6 for x in answers.values()):raise ValueError('All six real members on every question required')
    return answers,provenance


def prepare_aggregations(root,stage,*,qualification_from_historical=False):
    from transformers import AutoTokenizer
    from .execution import make_job
    root=Path(root);_,records,questions,pool=select_records(root,stage)
    spec=ModelSpec(**json.loads((root/'configs/aggregator_v1.json').read_text('utf-8-sig')))
    tokenizer=AutoTokenizer.from_pretrained(spec.model_path,local_files_only=True,trust_remote_code=False)
    if qualification_from_historical:
        if stage!='qualification':raise ValueError('Historical cache cannot supply formal experiment answers')
        records=[r for r in records if r['task']=='bbh']
        answers={};provenance={}
        for r in records:
            qid=r['question_id'];q=questions[qid];answers[qid]={};provenance[qid]={}
            for entry in pool[:3]:
                old=q['member_outputs'][entry['historical_cache_alias']]
                answers[qid][entry['alias']]=old['raw_output']
                provenance[qid][entry['alias']]={'source_file':old['source_file'],
                    'historical_checkpoint_revision':'UNKNOWN_NOT_ASSIGNED_CURRENT_MODEL_REVISION',
                    'usage':'runtime_qualification_only_not_formal_research'}
        teams=[tuple(x['alias'] for x in pool[:3])]
    else:
        answers,provenance=collect_members(root,stage,records,pool)
        teams=list(combinations([x['alias'] for x in pool],3))
    jobs=[]
    for row in records:
        q=questions[row['question_id']];qid=q['question_id']
        for team in teams:
            jobs.append(make_job(question_id=qid,task=q['task'],subtask=q['subtask'],question=q['question'],
                answers={x:answers[qid][x] for x in team},seed=42,model_spec=spec,split=row['split'],tokenizer=tokenizer,
                member_provenance={x:provenance[qid][x] for x in team}))
    name='historical_runtime_qualification.jsonl' if qualification_from_historical else 'aggregations.jsonl'
    return immutable_jsonl(root/'data/jobs'/stage/name,jobs)


def prepare_alone(root):
    from transformers import AutoTokenizer
    from .execution import make_alone_job
    root=Path(root);_,records,questions,_=select_records(root,'e1')
    spec=ModelSpec(**json.loads((root/'configs/aggregator_v1.json').read_text('utf-8-sig')))
    tokenizer=AutoTokenizer.from_pretrained(spec.model_path,local_files_only=True,trust_remote_code=False)
    jobs=[]
    for r in records:
        if r['split']!='e1_diagnostic_evaluation':continue
        q=questions[r['question_id']]
        jobs.append(make_alone_job(question_id=q['question_id'],task=q['task'],subtask=q['subtask'],question=q['question'],
            seed=42,model_spec=spec,split=r['split'],tokenizer=tokenizer))
    return immutable_jsonl(root/'data/jobs/e1/aggregator_alone.jsonl',jobs)
