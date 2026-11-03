"""Scheduler housekeeping tests with fake CUDA only; no GPU/model execution."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import random
import sys
import pytest
from budget_router import runtime_memory as rm
from budget_router import execution
from budget_router.execution import ExecutionStore
from budget_router.member_execution import MemberStore
from budget_router.inference import GenerationFailure
from test_real_execution_protocol import (FakeBackend,job,spec,member_job,member_spec,load_cli,load_member_cli)


class FakeCuda:
 def __init__(self,callback=None,error=None):self.reserved=500;self.allocated=100;self.calls=0;self.callback=callback;self.error=error
 def is_available(self):return True
 def memory_allocated(self):return self.allocated
 def memory_reserved(self):return self.reserved
 def empty_cache(self):
  self.calls+=1
  if self.callback:self.callback()
  if self.error:raise self.error
  self.reserved=120


class CudaBackend(FakeBackend):
 def __init__(self,model_spec=None,error=None,cuda=None):
  super().__init__(model_spec,error);self.generating=False;self.torch=SimpleNamespace(cuda=cuda or FakeCuda())
 def generate(self,messages,seed):
  self.generating=True
  try:return super().generate(messages,seed)
  finally:self.generating=False


def test_cleanup_is_after_generate_return_raw_append_fsync_and_store_lock_release(tmp_path,monkeypatch):
 output=tmp_path/'raw.jsonl';backend=CudaBackend();events=[];real_fsync=execution.os.fsync
 def fsync(fd):
  result=real_fsync(fd)
  try:
   if output.exists() and __import__('os').fstat(fd).st_ino==output.stat().st_ino:events.append('raw_fsync')
  except OSError:pass
  return result
 monkeypatch.setattr(execution.os,'fsync',fsync)
 row,cached=ExecutionStore(output).run(backend,job());before=output.read_bytes();original=deepcopy(row);rng=random.getstate()
 def safe_boundary():
  assert not backend.generating and output.read_bytes()==before and 'raw_fsync' in events
  assert not output.with_name(output.name+'.lock').exists()
  events.append('cleanup')
 backend.torch.cuda.callback=safe_boundary
 event=rm.cleanup_after_committed_success(backend,row,cached=cached,project_root=tmp_path,output_path=output,
  runner_name='run_aggregation.py',runner_source_sha256='a'*64)
 assert row==original and output.read_bytes()==before and random.getstate()==rng
 assert events.index('raw_fsync')<events.index('cleanup')
 assert event['allocator_before']=={'allocated_bytes':100,'reserved_bytes':500}
 assert event['allocator_after']=={'allocated_bytes':100,'reserved_bytes':120}
 assert event['cleanup_performed'] and event['total_housekeeping_wall_time_s']>=event['cleanup_wall_time_s']>=0
 audit=json.loads((tmp_path/event['audit_path']).read_text())
 assert audit['generation_record_sha256']==row['record_sha256'] and audit['generation_result_or_reward_modified'] is False
 assert audit['retry_permitted'] is False and 'PyTorch' in audit['memory_scope']
 assert 'Final answer' not in json.dumps(audit) and 'fixture_q' not in json.dumps(audit)


@pytest.mark.parametrize('cached,status',[(True,'success'),(False,'failure'),(True,'failure')])
def test_cached_or_failed_rows_do_not_even_access_torch(tmp_path,cached,status):
 class Poison:
  @property
  def torch(self):raise AssertionError('Cached/failure accessed CUDA')
 assert rm.cleanup_after_committed_success(Poison(),{'status':status,'cost_status':'unknown_must_reconcile'},cached=cached,
  project_root=tmp_path,output_path=tmp_path/'raw.jsonl',runner_name='test.py') is None
 assert not (tmp_path/'artifacts').exists()


def test_unsettled_or_modified_success_is_rejected_before_cuda(tmp_path):
 backend=CudaBackend();row={'status':'success','cost_status':'unknown_must_reconcile'}
 with pytest.raises(rm.RuntimeMemoryHousekeepingError):rm.cleanup_after_committed_success(backend,row,cached=False,project_root=tmp_path,output_path=tmp_path/'raw.jsonl',runner_name='test.py')
 row.update(cost_status='recorded_actual_usage',record_sha256='0'*64)
 with pytest.raises(rm.RuntimeMemoryHousekeepingError):rm.cleanup_after_committed_success(backend,row,cached=False,project_root=tmp_path,output_path=tmp_path/'raw.jsonl',runner_name='test.py')
 assert backend.torch.cuda.calls==0


def test_cleanup_failure_is_sanitized_and_never_retries_an_already_successful_job(tmp_path):
 backend=CudaBackend(cuda=FakeCuda(error=RuntimeError('PRIVATE_BACKEND_SECRET')));output=tmp_path/'raw.jsonl'
 row,cached=ExecutionStore(output).run(backend,job());before=output.read_bytes()
 with pytest.raises(rm.RuntimeMemoryHousekeepingError) as caught:
  rm.cleanup_after_committed_success(backend,row,cached=cached,project_root=tmp_path,output_path=output,runner_name='test.py')
 assert 'PRIVATE_BACKEND_SECRET' not in str(caught.value)
 text=(tmp_path/'artifacts/audit/runtime_memory_housekeeping.jsonl').read_text()
 assert 'PRIVATE_BACKEND_SECRET' not in text and json.loads(text)['error_type']=='RuntimeError'
 again,cached=ExecutionStore(output).run(backend,job())
 assert cached and backend.calls==1 and output.read_bytes()==before and again==row


def cli_fixture(tmp_path,monkeypatch,role,*,failure=False,cached=False):
 cli=load_member_cli() if role=='member' else load_cli();monkeypatch.setattr(cli,'ROOT',tmp_path)
 ms=member_spec() if role=='member' else spec();factory=member_job if role=='member' else job
 jobs=[factory(ms,question_id='one') if role=='member' else factory(question_id='one'),
       factory(ms,question_id='two') if role=='member' else factory(question_id='two')]
 jp=tmp_path/'jobs.jsonl';jp.write_text('\n'.join(json.dumps(j) for j in jobs)+'\n')
 cfg=tmp_path/'spec.json';cfg.write_text(json.dumps(asdict(ms)));output=tmp_path/'outputs.jsonl'
 backend=CudaBackend(ms,error=GenerationFailure('unknown_usage') if failure else None)
 if cached:
  store=MemberStore(output) if role=='member' else ExecutionStore(output)
  for j in jobs:store.run(FakeBackend(ms),j)
  monkeypatch.setattr(cli,'make_backend',lambda *_:pytest.fail('Cached batch loaded a model'))
  monkeypatch.setattr(cli,'cleanup_after_committed_success',lambda *a,**k:pytest.fail('Cached row called housekeeping'))
  monkeypatch.setattr(cli,'cleanup_after_backend_initialization',lambda *a,**k:pytest.fail('Cached batch called initialization housekeeping'))
 else:monkeypatch.setattr(cli,'make_backend',lambda *_:backend)
 monkeypatch.setattr(sys,'argv',[role,'--jobs',str(jp),'--model-config',str(cfg),'--output',str(output),'--qualification','--max-jobs','2'])
 return cli,backend,output


@pytest.mark.parametrize('role',['member','aggregation'])
def test_runners_only_housekeep_fresh_durable_success_and_separate_batch_overhead(tmp_path,monkeypatch,role):
 cli,backend,output=cli_fixture(tmp_path,monkeypatch,role)
 cli.main();rows=[json.loads(line) for line in output.read_text().splitlines()]
 audit=[json.loads(line) for line in (tmp_path/'artifacts/audit/runtime_memory_housekeeping.jsonl').read_text().splitlines()]
 assert backend.calls==2 and len(rows)==2 and len(audit)==3
 assert [r['phase'] for r in audit]==['post_backend_initialization','post_persisted_success','post_persisted_success']
 assert audit[0]['job_id'] is None and audit[0]['generation_record_sha256'] is None
 assert all(r['status']=='success' and 'runtime_memory_housekeeping' not in r and r['result']['wall_time_s']==.01 for r in rows)
 batch=json.loads(output.with_name(output.name+'.batch.json').read_text());stats=batch['runtime_memory_housekeeping']
 assert stats['cuda_empty_cache_calls']==3 and stats['fresh_success_records_observed']==2 and stats['post_backend_initialization_records']==1
 assert stats['total_housekeeping_wall_time_s']>=stats['cleanup_wall_time_s']>=0
 assert len(batch['runner_source_sha256'])==64


@pytest.mark.parametrize('role',['member','aggregation'])
def test_cached_batches_never_load_backend_or_touch_cuda(tmp_path,monkeypatch,role):
 cli,backend,output=cli_fixture(tmp_path,monkeypatch,role,cached=True);cli.main()
 assert not (tmp_path/'artifacts/audit/runtime_memory_housekeeping.jsonl').exists()
 assert len(output.read_text().splitlines())==2


@pytest.mark.parametrize('role',['member','aggregation'])
def test_unknown_failure_still_stops_before_second_call_without_success_cleanup_or_retry(tmp_path,monkeypatch,role):
 cli,backend,output=cli_fixture(tmp_path,monkeypatch,role,failure=True)
 with pytest.raises(RuntimeError,match='Unknown'):cli.main()
 assert backend.calls==1 and len(output.read_text().splitlines())==1
 row=json.loads(output.read_text());assert row['cost_status']=='unknown_must_reconcile' and row['status']=='failure'
 audit=[json.loads(line) for line in (tmp_path/'artifacts/audit/runtime_memory_housekeeping.jsonl').read_text().splitlines()]
 assert len(audit)==1 and audit[0]['phase']=='post_backend_initialization'


def test_core_qualification_source_bytes_remain_identical_and_old_runners_are_preserved():
 root=Path(__file__).resolve().parents[1]
 latest=root/'artifacts/private/runtime_memory_recovery/LATEST_PATCH_DIRECTORY.txt'
 if not latest.is_file():pytest.skip('Operator source-lineage archive remains local-only')
 folder=Path(latest.read_text(encoding='utf8'))
 if not (folder/'change_manifest.json').is_file():pytest.skip('Operator archive path is not mounted here')
 manifest=json.loads((folder/'change_manifest.json').read_text(encoding='utf8'))
 for name,expected in manifest['qualified_math_sources_before'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
 for name in ('scripts/run_members.py','scripts/run_aggregation.py'):
  old=manifest['previous'][name]
  assert hashlib.sha256(Path(old['backup']).read_bytes()).hexdigest()==old['sha256']
  assert hashlib.sha256((root/name).read_bytes()).hexdigest()!=old['sha256']


def test_initialization_boundary_has_no_job_and_precedes_first_generate(tmp_path):
 backend=CudaBackend();rng=random.getstate();model_spec=backend.spec;output=tmp_path/'raw.jsonl';events=[]
 def at_init():
  assert backend.calls==0 and not backend.generating and not output.exists()
  events.append('init_cleanup')
 backend.torch.cuda.callback=at_init
 event=rm.cleanup_after_backend_initialization(backend,project_root=tmp_path,output_path=output,runner_name='test.py')
 assert event['phase']=='post_backend_initialization' and backend.spec is model_spec and random.getstate()==rng
 backend.torch.cuda.callback=lambda:events.append('success_cleanup')
 row,cached=ExecutionStore(output).run(backend,job())
 rm.cleanup_after_committed_success(backend,row,cached=cached,project_root=tmp_path,output_path=output,runner_name='test.py')
 assert events==['init_cleanup','success_cleanup'] and backend.calls==1
 audit=[json.loads(line) for line in (tmp_path/'artifacts/audit/runtime_memory_housekeeping.jsonl').read_text().splitlines()]
 assert audit[0]['job_id'] is None and audit[0]['phase']=='post_backend_initialization' and audit[0]['generation_record_sha256'] is None
 assert audit[1]['phase']=='post_persisted_success' and audit[1]['generation_record_sha256']==row['record_sha256']


@pytest.mark.parametrize('role',['member','aggregation'])
def test_constructor_failure_never_claims_initialization_cleanup_or_generates(tmp_path,monkeypatch,role):
 from budget_router.runtime_diagnostics import BackendInitializationError
 cli,backend,output=cli_fixture(tmp_path,monkeypatch,role)
 def failed(_):raise ValueError('Synthetic initialization failure')
 monkeypatch.setattr(cli,'make_backend',failed)
 monkeypatch.setattr(cli,'cleanup_after_backend_initialization',lambda *a,**k:pytest.fail('Failed constructor claimed successful init cleanup'))
 with pytest.raises(BackendInitializationError):cli.main()
 assert backend.calls==0 and not (tmp_path/'artifacts/audit/runtime_memory_housekeeping.jsonl').exists()
 assert not output.exists() or not output.read_bytes()


def test_initialization_cleanup_failure_stops_before_any_generation(tmp_path):
 backend=CudaBackend(cuda=FakeCuda(error=RuntimeError('PRIVATE_INIT_SECRET')));output=tmp_path/'raw.jsonl'
 with pytest.raises(rm.RuntimeMemoryHousekeepingError,match='before the first generation'):
  rm.cleanup_after_backend_initialization(backend,project_root=tmp_path,output_path=output,runner_name='test.py')
 assert backend.calls==0 and not output.exists()
 text=(tmp_path/'artifacts/audit/runtime_memory_housekeeping.jsonl').read_text()
 assert 'PRIVATE_INIT_SECRET' not in text and json.loads(text)['status']=='HOUSEKEEPING_FAILED_BEFORE_FIRST_GENERATION'
