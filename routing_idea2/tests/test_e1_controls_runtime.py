"""Synthetic protocol tests only; never model qualification or scientific results."""
from dataclasses import asdict,replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from budget_router import e1_controls as controls
from budget_router.execution import _digest,validate_job
from budget_router.inference import ModelSpec,GenerationResult,configuration_fingerprint
from budget_router.member_execution import make_member_job,MemberStore,validate_resample_groups
from budget_router import preparation

REV='b'*40


def member_spec():
    return ModelSpec('fixture/member',REV,model_path='fixture_local_member',quantization='nf4',
                     temperature=.7,top_p=.95,max_new_tokens=2048,enable_thinking=False)


def records_and_questions():
    rows=[];questions={}
    for split,count in [('e1_selection_calibration',50),('e1_diagnostic_evaluation',100)]:
        for task in ('gsm8k','bbh'):
            for i in range(count):
                qid=f'{split}_{task}_{i}'
                row=dict(question_id=qid,task=task,subtask='causal_judgement' if task=='bbh' else None,split=split)
                rows.append(row)
                questions[qid]={**row,'question':f'Original {qid}?','reference':'Yes' if task=='bbh' else '4'}
    return rows,questions


def test_strongest_model_never_uses_evaluation_correctness_and_ties_use_pool_index():
    records,questions=records_and_questions()
    pool=[{'index':i,'alias':f'm{i}'} for i in range(6)]
    answers={}
    for row in records:
        correct='Yes' if row['task']=='bbh' else '4'
        wrong='No' if row['task']=='bbh' else '5'
        answers[row['question_id']]={entry['alias']:(correct if entry['index']==0 else wrong) for entry in pool}
    chosen,_,_=controls._select_calibration_model(list(reversed(pool)),records,questions,answers)
    assert chosen['index']==0
    for row in records:
        if row['split']=='e1_diagnostic_evaluation':
            # Deliberately reverse evaluation ranking; it must not change selection.
            answers[row['question_id']]={entry['alias']:('Yes' if row['task']=='bbh' else '4') if entry['index']==5 else ('No' if row['task']=='bbh' else '5') for entry in pool}
    assert controls._select_calibration_model(pool,records,questions,answers)[0]['index']==0
    bad=[dict(row,task='gsm8k') if row['split']=='e1_selection_calibration' else row for row in records]
    with pytest.raises(ValueError,match='50 calibration'):controls._select_calibration_model(pool,bad,questions,answers)


class Tokenizer:
    def __init__(self,name):self.name_or_path=name;self.init_kwargs={}
    def encode(self,text,**kwargs):return [ord(c) for c in text]
    def decode(self,ids,**kwargs):return ''.join(chr(i) for i in ids)


def test_strong_control_requires_three_actual_calls_and_preserves_unclipped_costs(tmp_path,monkeypatch):
    records,questions=records_and_questions()
    strong={'alias':'m0','index':0};ms=member_spec()
    for path in ['configs','artifacts/audit','results/raw/e1']:(tmp_path/path).mkdir(parents=True,exist_ok=True)
    selection=tmp_path/'artifacts/audit/e1_strong_model_selection.json';selection.write_text('{}',encoding='utf-8')
    monkeypatch.setattr(controls,'select_strong_model',lambda root:(strong,{}))
    monkeypatch.setattr(controls,'select_records',lambda root,stage:({},records,questions,[strong]))
    monkeypatch.setattr(controls,'read_spec',lambda root,entry,sampling=False:ms)
    receipt=controls.prepare_strong_samples(tmp_path)
    jobs=preparation.read_jsonl(receipt['path'])
    assert len(jobs)==600 and all(job['sampling_control'] for job in jobs)
    validate_resample_groups(jobs)
    output=tmp_path/'results/raw/e1/strong_samples.jsonl'
    rows=[]
    for job in jobs:
        row={key:job[key] for key in ('job_id','question_id','task','subtask','split','model_alias','sample_index','seed','model_fingerprint','member_mode','sampling_control','prompt_version','source_question_sha256')}
        row.update(status='success',cost_status='recorded_actual_usage',attempt_id=job['job_id'],result={
            'text':'x'*1200+' Final answer: 4','prompt_tokens':99,'completion_tokens':900,'wall_time_s':.2,
            'model_id':ms.model_id,'model_revision':ms.revision,'seed':job['seed'],'backend':ms.backend,
            'model_identity_verified':True,'completion_reached_cap':False})
        row['record_sha256']=_digest(row);rows.append(row)
    output.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    agg=ModelSpec('fixture/aggregator','c'*40,model_path=str(tmp_path/'models/agg'),quantization='nf4')
    (tmp_path/'configs/aggregator_v1.json').write_text(json.dumps(asdict(agg)),encoding='utf-8')
    auto=SimpleNamespace(from_pretrained=lambda name,**kwargs:Tokenizer(name))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=auto))
    made=controls.prepare_strong_aggregation(tmp_path)
    generated=preparation.read_jsonl(made['path'])
    assert len(generated)==200
    first=generated[0];validate_job(first,agg,require_token_policy=True)
    assert len(first['members'])==3 and all('::sample' in alias for alias in first['members'])
    assert first['member_provenance']['selected_from_calibration_sha256']==preparation.file_sha(selection)
    assert all(value['completion_tokens']==900 for value in first['member_provenance']['samples'].values())
    assert all(value['retained_source_tokens']==512 for value in first['candidate_statistics'].values())
    wrong=dict(rows[0]);wrong['result']=dict(wrong['result'],seed=42)
    with pytest.raises(ValueError,match='actual served model/seed'):controls._validate_sample_result(wrong,jobs[0],ms)
    output.write_text(''.join(json.dumps(row)+'\n' for row in rows[:-1]),encoding='utf-8')
    with pytest.raises(ValueError,match='Missing one or more real'):controls.prepare_strong_aggregation(tmp_path)


def load_review():
    path=Path(__file__).resolve().parents[1]/'scripts/review_runtime.py'
    spec=importlib.util.spec_from_file_location('runtime_review_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def runtime_fixture(tmp_path,monkeypatch,*,parsed=(True,True,True,True),caps=(False,False,False,False),misassigned=False):
    module=load_review();monkeypatch.setattr(module,'ROOT',tmp_path)
    ms=member_spec()
    questions={};assignments=[];jobs=[]
    for i,task in enumerate(('gsm8k','bbh','gsm8k','bbh')):
        qid=f'qual_{i}';subtask='causal_judgement' if task=='bbh' else None
        question=f'Qualification original {i} with \u2028 and \u0085?'
        questions[qid]={'question_id':qid,'question':question,'task':task,'subtask':subtask,'reference':'4' if task=='gsm8k' else 'Yes'}
        assignments.append({'question_id':qid,'split':'e1_diagnostic_evaluation' if misassigned and i==0 else 'qualification'})
        jobs.append(make_member_job(question_id=qid,task=task,subtask=subtask,question=question,
            model_alias='m0',seed=42,model_spec=ms,split='qualification'))
    monkeypatch.setattr(module,'load_inputs',lambda root:({},assignments,questions,[]))
    (tmp_path/'data/derived').mkdir(parents=True)
    (tmp_path/'data/derived/split_manifest.json').write_text('{}',encoding='utf-8')
    jobs_path=tmp_path/'jobs.jsonl';jobs_path.write_text(''.join(json.dumps(job,ensure_ascii=False)+'\n' for job in jobs),encoding='utf-8')
    config=tmp_path/'model.json';config.write_text(json.dumps(asdict(ms)),encoding='utf-8')
    outputs=tmp_path/'raw.jsonl';store=MemberStore(outputs)
    class Backend:
        spec=ms
        count=0
        def generate(self,messages,seed):
            i=self.count;self.count+=1
            # All final answers are deliberately WRONG but safely parseable.
            text=('Final answer: 999' if i%2==0 else 'Final answer: No') if parsed[i] else 'Only unfinished reasoning here'
            return GenerationResult(text,20,2048 if caps[i] else 20,.1,ms.model_id,ms.revision,seed,ms.backend,
                                    model_identity_verified=True,completion_reached_cap=caps[i])
    backend=Backend()
    for job in jobs:assert store.run(backend,job)[0]['status']=='success'
    return module,(jobs_path,outputs,config,tmp_path/'receipt.json')


def test_runtime_is_format_only_and_receipt_binds_all_sources(tmp_path,monkeypatch,capsys):
    module,args=runtime_fixture(tmp_path,monkeypatch)
    assert module.review(*args)==0
    receipt=json.loads(args[-1].read_text('utf-8'))
    assert receipt['status']=='PASS' and receipt['accuracy_was_not_an_acceptance_criterion']
    assert receipt['technical_rule']['minimum_jobs']==4
    for key,path in [('jobs',args[0]),('outputs',args[1]),('model_config',args[2])]:
        assert (tmp_path/receipt[key+'_path']).resolve()==path.resolve()
        assert receipt[key+'_sha256']==preparation.file_sha(path)
    assert receipt['split_manifest_sha256']==preparation.file_sha(tmp_path/'data/derived/split_manifest.json')
    assert len(receipt['source_code'])>=6
    old=args[-1].read_bytes();assert module.review(*args)==0;assert args[-1].read_bytes()==old
    capsys.readouterr()


def test_runtime_cannot_relabel_an_e1_question_as_qualification(tmp_path,monkeypatch):
    module,args=runtime_fixture(tmp_path,monkeypatch,misassigned=True)
    with pytest.raises(ValueError,match='not assigned'):module.review(*args)


@pytest.mark.parametrize('parsed,caps,expected',[
    ((True,True,False,False),(False,False,True,True),0),
    ((True,False,True,False),(False,False,False,False),3),
    ((True,True,True,True),(True,True,True,False),3),
])
def test_declared_parse_and_cap_rules_only(tmp_path,monkeypatch,parsed,caps,expected,capsys):
    module,args=runtime_fixture(tmp_path,monkeypatch,parsed=parsed,caps=caps)
    assert module.review(*args)==expected
    capsys.readouterr()


def test_runtime_minimum_four_and_job_hash_are_required(tmp_path,monkeypatch):
    module,args=runtime_fixture(tmp_path,monkeypatch)
    jobs=preparation.read_jsonl(args[0]);args[0].write_text(''.join(json.dumps(job)+'\n' for job in jobs[:3]),encoding='utf-8')
    with pytest.raises(ValueError,match='At least four'):module.review(*args)
    args[0].write_text(''.join(json.dumps(job)+'\n' for job in jobs),encoding='utf-8')
    jobs[0]['question']='tampered'
    args[0].write_text(''.join(json.dumps(job)+'\n' for job in jobs),encoding='utf-8')
    with pytest.raises(ValueError,match='hash'):module.review(*args)


def test_prepare_alone_accepts_current_tokenizer_keyword(tmp_path,monkeypatch):
    agg=ModelSpec('fixture/aggregator','c'*40,model_path=str(tmp_path/'models/agg'),quantization='nf4')
    (tmp_path/'configs').mkdir();(tmp_path/'configs/aggregator_v1.json').write_text(json.dumps(asdict(agg)),encoding='utf-8')
    records,questions=records_and_questions();selected=[row for row in records if row['split']=='e1_diagnostic_evaluation'][:2]
    monkeypatch.setattr(preparation,'select_records',lambda root,stage:({},selected,questions,[]))
    auto=SimpleNamespace(from_pretrained=lambda name,**kwargs:Tokenizer(name))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=auto))
    result=preparation.prepare_alone(tmp_path)
    for job in preparation.read_jsonl(result['path']):
        validate_job(job,agg,require_token_policy=True)
        assert job['members']==[] and job['member_provenance']['control']=='aggregator_alone'
