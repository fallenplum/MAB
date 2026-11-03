"""Software mocks only: no model loading, downloads, paid API or research scores."""
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from budget_router.inference import (ModelSpec,GenerationResult,GenerationFailure,OpenAICompatibleBackend,
                                    configuration_fingerprint,validate_local_model_source)
from budget_router.execution import (ExecutionStore,make_job,validate_job,head_tail_candidate,_digest)

REV="a"*40

class CharTokenizer:
    name_or_path="fixture/model"
    init_kwargs={"revision":REV}
    def encode(self,text,**kwargs):return [ord(x) for x in text]
    def decode(self,tokens,**kwargs):return "".join(chr(x) for x in tokens)


def spec():return ModelSpec("fixture/model",REV,quantization="nf4")


def job(split="qualification",tokenizer=None,**kwargs):
    args=dict(question_id="fixture_q",task="gsm8k",question="What is 2+2?",
        answers={"model-A":"Reasoning A. Final answer: 4","model-B":"Reasoning B. Final answer: 5"},
        seed=42,model_spec=spec(),split=split,tokenizer=tokenizer)
    args.update(kwargs)
    return make_job(**args)

class FakeBackend:
    def __init__(self,model_spec=None,error=None):self.spec=model_spec or spec();self.calls=0;self.error=error
    def generate(self,messages,seed):
        self.calls+=1
        if self.error is not None:raise self.error
        return GenerationResult("Final answer: 4",20,4,.01,self.spec.model_id,self.spec.revision,seed,
                                self.spec.backend,model_identity_verified=True,completion_reached_cap=False)


def test_revision_nf4_and_credentials_are_explicit():
    for revision in ("main","latest","pending","some_branch"):
        with pytest.raises(ValueError):replace(spec(),revision=revision).validate()
    with pytest.raises(ValueError,match="CUDA"):replace(spec(),device="cpu").validate()
    with pytest.raises(ValueError):ModelSpec("api/model",REV,backend="openai_compatible",base_url="https://user:secret@example.test/v1").validate()


def test_head_tail_keeps_source_budget_and_charges_marker_separately():
    answer="a"*128+"x"*900+"z"*384
    rendered,stats=head_tail_candidate(answer,CharTokenizer())
    assert rendered.startswith("a"*128) and rendered.endswith("z"*384)
    assert "omitted 900 source tokens" in rendered
    assert stats["retained_source_tokens"]==512
    assert stats["rendered_tokens"]>512 and stats["marker_tokens"]>0
    assert stats["original_member_generation_cost_unchanged"] is True


def test_job_binds_middle_source_content_policy_and_full_original_cost():
    answers={"model-A":"a"*128+"x"*900+"z"*384}
    provenance={"model-A":{"completion_tokens":1412,"source_revision":REV}}
    first=job(answers=answers,tokenizer=CharTokenizer(),member_provenance=provenance)
    second=job(answers={"model-A":"a"*128+"y"*900+"z"*384},tokenizer=CharTokenizer(),member_provenance=provenance)
    assert first["messages"]==second["messages"]
    assert first["job_id"]!=second["job_id"]
    assert first["member_provenance"]["model-A"]["completion_tokens"]==1412
    assert "model-A" not in first["messages"][1]["content"]
    validate_job(first,spec(),require_token_policy=True)
    with pytest.raises(ValueError,match="tokenizer-driven"):validate_job(job(),spec(),require_token_policy=True)


def test_hash_and_model_config_tampering_stops_before_generation(tmp_path):
    backend=FakeBackend();j=job();j["messages"][1]["content"]+=" Secret gold: 4"
    with pytest.raises(ValueError,match="hash"):ExecutionStore(tmp_path/"raw.jsonl").run(backend,j)
    assert backend.calls==0
    with pytest.raises(ValueError,match="configuration"):
        ExecutionStore(tmp_path/"raw.jsonl").run(FakeBackend(replace(spec(),max_new_tokens=128)),job())


def test_labels_never_enter_job_provenance_or_prompt():
    with pytest.raises(ValueError,match="Labels"):
        job(member_provenance={"nested":{"gold_answer":"4"}})
    with pytest.raises(ValueError,match="Labels"):
        job(member_provenance={"api_key":"secret"})
    j=job();j["score"]=1;j["job_id"]=_digest({k:v for k,v in j.items() if k!="job_id"})
    with pytest.raises(ValueError,match="schema"):validate_job(j)


def test_resume_reuses_exact_success_without_duplicate_costs(tmp_path):
    target=tmp_path/"raw.jsonl";backend=FakeBackend();j=job(tokenizer=CharTokenizer())
    row,cached=ExecutionStore(target).run(backend,j)
    assert not cached and row["cost_status"]=="recorded_actual_usage"
    second,cached=ExecutionStore(target).run(backend,j)
    assert cached and backend.calls==1 and second==row
    assert len(target.read_text('utf-8').splitlines())==1
    assert second["candidate_token_policy"]==j["candidate_token_policy"]


def test_unknown_cost_is_null_and_retry_requires_reconciliation(tmp_path):
    token="do-not-log-SECRET-token"
    backend=FakeBackend(error=RuntimeError(token));target=tmp_path/"raw.jsonl"
    row,_=ExecutionStore(target).run(backend,job())
    assert row["actual_usage"] is None and row["cost_status"]=="unknown_must_reconcile"
    assert token not in target.read_text('utf-8')
    with pytest.raises(RuntimeError,match="unknown cost"):
        ExecutionStore(target).run(backend,job())
    assert backend.calls==1


def test_known_failed_response_usage_is_preserved(tmp_path):
    failure=GenerationFailure("served_model_identity_unverified",usage={"prompt_tokens":12,"completion_tokens":3})
    row,_=ExecutionStore(tmp_path/"raw.jsonl").run(FakeBackend(error=failure),job())
    assert row["status"]=="failure" and row["actual_usage"]["prompt_tokens"]==12
    assert row["cost_status"]=="recorded_actual_usage"


def test_success_record_mutation_is_not_a_cache_hit(tmp_path):
    target=tmp_path/"raw.jsonl";ExecutionStore(target).run(FakeBackend(),job())
    row=json.loads(target.read_text('utf-8'));row['result']['text']='tampered'
    target.write_text(json.dumps(row)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match="integrity"):ExecutionStore(target)


def test_local_model_receipt_checks_repo_revision_inventory_and_actual_bytes(tmp_path):
    directory=tmp_path/'models'/'fixture--model';directory.mkdir(parents=True)
    files=[]
    for name in ('config.json','tokenizer_config.json','tokenizer.json','model.safetensors'):
        f=directory/name;f.write_text('{}',encoding='utf-8')
        files.append({'path':f.relative_to(tmp_path).as_posix(),'bytes':f.stat().st_size,
                      'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    receipt=tmp_path/'receipt.json'
    receipt.write_text(json.dumps({'repo_id':spec().model_id,'revision':REV,'path':str(directory),
                                  'files':files,'completed_at_unix':1}),encoding='utf-8')
    local=replace(spec(),model_path=str(directory))
    validate_local_model_source(local,receipt)
    (directory/'model.safetensors').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match="bytes"):validate_local_model_source(local,receipt)


def test_service_must_attest_model_revision_and_real_usage(monkeypatch):
    api=ModelSpec('fixture/model',REV,backend='openai_compatible',base_url='http://localhost:1234/v1')
    payload={'model':'fixture/model','model_revision':REV,'usage':{'prompt_tokens':10,'completion_tokens':2},
             'choices':[{'message':{'content':'Final answer: 4'},'finish_reason':'stop'}]}
    response=SimpleNamespace(raise_for_status=lambda:None,json=lambda:payload)
    monkeypatch.setitem(sys.modules,'requests',SimpleNamespace(post=lambda *a,**k:response))
    result=OpenAICompatibleBackend(api).generate([{'role':'user','content':'question'}],42)
    assert result.model_identity_verified and result.prompt_tokens==10
    payload.pop('model_revision')
    with pytest.raises(GenerationFailure) as exc:OpenAICompatibleBackend(api).generate([],42)
    assert exc.value.usage['prompt_tokens']==10
    payload['model_revision']=REV;payload.pop('usage')
    with pytest.raises(GenerationFailure) as exc:OpenAICompatibleBackend(api).generate([],42)
    assert exc.value.usage is None


def load_cli():
    path=Path(__file__).resolve().parents[1]/'scripts/run_aggregation.py'
    module_spec=importlib.util.spec_from_file_location('aggregation_cli_under_test',path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    return module


def test_formal_cli_requires_matching_pass_receipt(tmp_path):
    cli=load_cli();receipt=tmp_path/'qualification.json'
    with pytest.raises(ValueError):cli._read_qualification(None,spec())
    receipt.write_text(json.dumps({'status':'PASS','model_fingerprint':configuration_fingerprint(spec())}),encoding='utf-8')
    assert cli._read_qualification(receipt,spec())['status']=='PASS'


def test_cached_formal_batch_does_not_load_a_model(tmp_path,monkeypatch,capsys):
    cli=load_cli();j=job(split='e1_diagnostic_evaluation',tokenizer=CharTokenizer(),question="Legal Unicode \u2028 separator \u0085 in one question?")
    output=tmp_path/'raw.jsonl';ExecutionStore(output).run(FakeBackend(),j)
    jobs=tmp_path/'jobs.jsonl';jobs.write_text(json.dumps(j,ensure_ascii=False)+'\n',encoding='utf-8')
    cfg=tmp_path/'spec.json';cfg.write_text(json.dumps(asdict(spec())),encoding='utf-8')
    receipt=tmp_path/'qualification.json';receipt.write_text(json.dumps({'status':'PASS','model_fingerprint':configuration_fingerprint(spec())}),encoding='utf-8')
    monkeypatch.setattr(cli,'make_backend',lambda *a:pytest.fail('Cached batch loaded model'))
    monkeypatch.setattr(sys,'argv',['run_aggregation','--jobs',str(jobs),'--model-config',str(cfg),'--output',str(output),'--qualification-receipt',str(receipt)])
    cli.main()
    assert 'already_recorded' in capsys.readouterr().out


from budget_router.execution import make_alone_job,ALONE_CANDIDATE_TEXT
from budget_router.member_execution import (MemberStore,make_member_job,validate_member_job,validate_resample_groups)


def member_spec(model_id='fixture/member',revision=REV,**kwargs):
    return replace(spec(),model_id=model_id,revision=revision,model_path='software_fixture_local_path',**kwargs)


def member_job(model_spec=None,**kwargs):
    values=dict(question_id='member_q',task='gsm8k',question='Question with legal \u2028 and \u0085 separators?',
                model_alias='member0',seed=42,model_spec=model_spec or member_spec(),split='qualification',sample_index=0)
    values.update(kwargs)
    return make_member_job(**values)


def test_aggregator_alone_has_no_fake_candidate_and_same_prompt():
    alone=make_alone_job(question_id='alone_q',task='gsm8k',question='What is 2+2?',seed=42,
                         model_spec=spec(),split='e1_diagnostic_evaluation')
    validate_job(alone,spec(),require_token_policy=True)
    assert alone['members']==[] and alone['candidate_order']==[]
    assert alone['candidate_statistics']=={} and alone['source_hashes']['member_answers_sha256']=={}
    assert alone['messages'][0]==job()['messages'][0]
    assert alone['messages'][1]['content'].endswith(ALONE_CANDIDATE_TEXT)
    with pytest.raises(ValueError):job(answers={})


def test_member_jobs_are_question_only_and_have_independent_sampling_protocol():
    ms=member_spec();j=member_job(ms)
    validate_member_job(j,ms)
    assert len(j['messages'])==1 and j['messages'][0]['role']=='user'
    assert j['messages'][0]['content'].endswith(j['question'])
    bad=dict(j,gold_answer='4');bad['job_id']=_digest({k:v for k,v in bad.items() if k!='job_id'})
    with pytest.raises(ValueError,match='schema'):validate_member_job(bad,ms)
    control=member_spec(temperature=.7,top_p=.95)
    samples=[member_job(control,sample_index=i,seed=100+i,sampling_control=True) for i in range(3)]
    validate_resample_groups(samples)
    with pytest.raises(ValueError,match='3 distinct'):validate_resample_groups(samples[:2])
    samples[2]=member_job(control,sample_index=2,seed=101,sampling_control=True)
    with pytest.raises(ValueError,match='3 distinct'):validate_resample_groups(samples)
    with pytest.raises(ValueError):member_job(member_spec(temperature=.7,top_p=1.0))


def test_member_store_has_required_fields_and_resumes_without_aggregation_disguise(tmp_path):
    ms=member_spec();backend=FakeBackend(ms);j=member_job(ms)
    target=tmp_path/'members.jsonl';row,cached=MemberStore(target).run(backend,j)
    assert not cached and row['status']=='success'
    assert all(key in row for key in ('question_id','task','subtask','split','model_alias','sample_index','seed','model_fingerprint','result'))
    assert row['member_mode']=='independent_deterministic'
    assert row['scoring_isolated_outside_generation_runner'] is True
    assert row['result']['completion_reached_cap'] is False
    row2,cached=MemberStore(target).run(backend,j)
    assert cached and backend.calls==1 and row2==row
    assert 'candidate_token_policy' not in row


def test_member_unknown_cost_remains_unknown_and_stops_retry(tmp_path):
    ms=member_spec();backend=FakeBackend(ms,error=RuntimeError('secret-do-not-echo'))
    target=tmp_path/'members.jsonl';row,_=MemberStore(target).run(backend,member_job(ms))
    assert row['actual_usage'] is None and row['cost_status']=='unknown_must_reconcile'
    assert 'secret-do-not-echo' not in target.read_text('utf-8')
    with pytest.raises(RuntimeError,match='unknown cost'):MemberStore(target).run(backend,member_job(ms))
    assert backend.calls==1


def load_member_cli():
    path=Path(__file__).resolve().parents[1]/'scripts/run_members.py'
    module_spec=importlib.util.spec_from_file_location('member_cli_under_test',path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    return module


def test_member_cli_serializes_models_and_preserves_unicode_jsonl(tmp_path,monkeypatch,capsys):
    import weakref
    cli=load_member_cli()
    monkeypatch.setattr(cli,'ROOT',tmp_path)  # Keep scheduler audits inside this software fixture.
    first=member_spec('fixture/first');second=member_spec('fixture/second','b'*40)
    specs=[first,second];all_jobs=[];arguments=['run_members']
    for i,ms in enumerate(specs):
        j=member_job(ms,model_alias=f'm{i}',split='e1_selection_calibration')
        all_jobs.append(j)
        cfg=tmp_path/f'spec{i}.json';cfg.write_text(json.dumps(asdict(ms)),encoding='utf-8')
        receipt=tmp_path/f'pass{i}.json';receipt.write_text(json.dumps({'status':'PASS','model_fingerprint':configuration_fingerprint(ms)}),encoding='utf-8')
        arguments+=['--model-config',str(cfg),'--qualification-receipt',str(receipt)]
    path=tmp_path/'jobs.jsonl';path.write_text('\n'.join(json.dumps(j,ensure_ascii=False) for j in all_jobs)+'\n',encoding='utf-8')
    out=tmp_path/'members.jsonl';arguments+=['--jobs',str(path),'--output',str(out)]
    loaded=[];weak=[]
    def factory(ms):
        assert not weak or weak[-1]() is None,'Previous model is still resident'
        backend=FakeBackend(ms);loaded.append(ms.model_id);weak.append(weakref.ref(backend));return backend
    monkeypatch.setattr(cli,'make_backend',factory)
    monkeypatch.setattr(sys,'argv',arguments)
    cli.main()
    assert loaded==['fixture/first','fixture/second']
    assert len(out.read_text('utf-8').split('\n'))==3
    assert 'member_generation_batch_complete' in capsys.readouterr().out
    monkeypatch.setattr(cli,'make_backend',lambda *a:pytest.fail('resume loaded a model'))
    cli.main()
    assert '"cache_hit": true' in capsys.readouterr().out



def test_candidate_cap_cannot_use_a_different_models_tokenizer():
    wrong=CharTokenizer();wrong.name_or_path='another/model'
    j=job(tokenizer=wrong)
    with pytest.raises(ValueError,match='tokenizer identity'):validate_job(j,spec(),require_token_policy=True)



def test_new_sampled_member_and_strong_role_share_fingerprint_but_not_jobs():
    ms=member_spec(temperature=.7,top_p=.95,max_new_tokens=2048,enable_thinking=False)
    ordinary=member_job(ms)
    control=member_job(ms,sampling_control=True,sample_index=0,seed=314)
    assert ordinary['member_mode']=='independent_sampled'
    assert control['member_mode']=='strong_model_resample'
    assert ordinary['model_fingerprint']==control['model_fingerprint']
    assert ordinary['messages']==control['messages'] and ordinary['job_id']!=control['job_id']
    assert len(ordinary['messages'])==1 and ordinary['messages'][0]['role']=='user'
    with pytest.raises(ValueError,match='fresh'):member_job(ms,sampling_control=True,seed=42)
    validate_member_job(ordinary,ms);validate_member_job(control,ms)


def test_enable_thinking_forwarding_and_aggregator_fingerprint_compatibility():
    from budget_router.inference import chat_template_text
    class Template:
        def apply_chat_template(self,messages,**kwargs):return kwargs
    base=spec()
    old_dict=asdict(base);old_dict.pop('enable_thinking')
    expected=hashlib.sha256(json.dumps(old_dict,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    assert configuration_fingerprint(base)==expected
    assert 'enable_thinking' not in chat_template_text(Template(),[],base)
    revised=replace(base,enable_thinking=False)
    assert chat_template_text(Template(),[],revised)['enable_thinking'] is False
    assert configuration_fingerprint(revised)!=expected
