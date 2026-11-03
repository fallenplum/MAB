"""Full authorized experiment orchestration; real gates precede every next stage.

This supervisor follows the already running E0/E1 process, never competes with
it. Checkpoints and receipts are immutable. Unknown inference costs stop for
review. GitHub sync remains a separately verified connector action after final
acceptance; process startup is never reported as research completion.
"""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,subprocess,sys,time,uuid,math
from contextlib import contextmanager
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from budget_router.stage_validation import read_json,immutable_json,artifact_reference,verified_reference,canonical_digest,input_identity
from budget_router.recovery_runtime import atomic_json,validated_recovery_owner,validated_recovery_completion,RecoveryAuthorizationError
PYTHON=ROOT/".venv/Scripts/python.exe"
OUT=ROOT/"results/full_v1"
SOURCE_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class ReviewRequired(RuntimeError):pass


def state(step,**fields):
 path=ROOT/"FULL_RUN_STATUS.json";old=read_json(path) if path.exists() else {}
 old.update(schema="full_experiment_supervisor_status_v1",master_pid=os.getpid(),
  process_create_time=psutil.Process().create_time(),master_script=str(Path(__file__).resolve()),
  master_source_sha256=SOURCE_SHA,master_running=True,overall="IN_PROGRESS",step=step,
  updated_at_unix=time.time(),scientific_completion=False,workflow_complete=False,github_upload="WAITING_FOR_FINAL_ACCEPTANCE")
 old.update(fields);atomic_json(path,old)
 print(json.dumps({"event":"full_pipeline_state","step":step,**fields}),flush=True)


def _pulse():
 path=ROOT/"FULL_RUN_STATUS.json"
 if path.exists():
  d=read_json(path);d["updated_at_unix"]=time.time();atomic_json(path,d)


def child(script,*args):
 script_path=ROOT/"scripts"/script
 if not script_path.is_file():raise ReviewRequired("Required experiment entrypoint missing: "+script)
 stamp=time.strftime("%Y%m%d_%H%M%S");logs=ROOT/"logs";logs.mkdir(exist_ok=True)
 log=logs/("full_"+script.replace(".py","")+"_"+stamp+"_"+uuid.uuid4().hex[:6]+".log")
 command=[str(PYTHON),"-X","utf8","-u",str(script_path),*[str(x) for x in args]]
 with log.open("ab") as stream:
  proc=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT)
  status=read_json(ROOT/"FULL_RUN_STATUS.json");status.update(active_child_pid=proc.pid,active_script=script,
   active_child_create_time=psutil.Process(proc.pid).create_time(),active_log=str(log),last_command=command,child_ownership="full_master_child")
  atomic_json(ROOT/"FULL_RUN_STATUS.json",status)
  while True:
   try:proc.wait(timeout=20);break
   except subprocess.TimeoutExpired:_pulse()
 status=read_json(ROOT/"FULL_RUN_STATUS.json");status.update(active_child_pid=None,active_script=None,
  last_child_script=script,last_child_exit=proc.returncode,updated_at_unix=time.time());atomic_json(ROOT/"FULL_RUN_STATUS.json",status)
 if proc.returncode!=0:raise ReviewRequired("Required stage process failed: "+script+"; private log "+str(log))
 return log


def _legacy_active(record):
 from show_status import observe_process
 lock=read_json(ROOT/'logs/local_e1_supervisor.lock') if (ROOT/'logs/local_e1_supervisor.lock').exists() else {}
 if record.get('supervisor_running') is not True or lock.get('pid')!=record.get('supervisor_pid'):return False
 started=lock.get('started_at_unix')
 if isinstance(started,bool) or not isinstance(started,(int,float)) or not math.isfinite(started):return False
 observation=observe_process(record.get('supervisor_pid'),ROOT/'scripts/run_local_e1.py',recorded_start=started,recorded_update=record.get('updated_at_unix'))
 return observation.get('identity_verified') is True and observation.get('execution_active') is True


def _legacy_owner_gone(record):
 pid=record.get('supervisor_pid')
 if isinstance(pid,bool) or not isinstance(pid,int) or pid<=0:return False
 try:
  process=psutil.Process(pid)
  lock=read_json(ROOT/'logs/local_e1_supervisor.lock') if (ROOT/'logs/local_e1_supervisor.lock').exists() else {}
  if lock.get('pid')!=pid or not isinstance(lock.get('started_at_unix'),(int,float)):return False
  return abs(process.create_time()-lock['started_at_unix'])>2
 except psutil.NoSuchProcess:return True
 except (psutil.AccessDenied,PermissionError):return False



def _recovery_record(record):
 return isinstance(record,dict) and (any(key in record for key in ('recovery_authorization','recovery_owner','recovery_completion','recovery_phase')) or
  Path(str(record.get('supervisor_script',''))).name=='resume_e1_aggregation.py')


def _recovery_snapshot(record,*,completion=False):
 validator=validated_recovery_completion if completion else validated_recovery_owner
 try:return validator(ROOT,record)
 except (RecoveryAuthorizationError,OSError,ValueError,TypeError,KeyError,psutil.Error) as error:
  raise ReviewRequired('Authorized recovery owner or completion evidence failed validation') from error


def _upstream_active(record):
 if not _recovery_record(record):return _legacy_active(record)
 completed=record.get('supervisor_running') is not True
 observed=_recovery_snapshot(record,completion=completed)
 if completed:return False
 return observed.get('identity_verified') is True and observed.get('execution_active') is True


def _upstream_identity_snapshot(record,known=None):
 if not _recovery_record(record):return _legacy_identity_snapshot(record,known)
 observed=_recovery_snapshot(record,completion=record.get('supervisor_running') is not True)
 pid=observed.get('pid')
 if observed.get('identity_verified') is not True:
  raise psutil.NoSuchProcess(pid)
 if observed.get('execution_active') is not True:raise ReviewRequired('Verified recovery owner is not executing')
 process=psutil.Process(pid);birth=process.create_time();command=process.cmdline()
 if abs(birth-observed['create_time'])>=.01 or command!=observed['argv']:
  raise ReviewRequired('Recovery owner identity changed during observation')
 identity={'pid':pid,'create_time':birth,'argv':command,'authorization_sha256':observed['authorization_sha256'],'recovery_owner':True}
 if known is not None and known!=identity:raise ReviewRequired('Recovery exact owner or authorization changed')
 return process,identity


def _upstream_owner_gone(record):
 if not _recovery_record(record):return _legacy_owner_gone(record)
 return _recovery_snapshot(record,completion=True).get('owner_gone') is True


def _normal_upstream_completion(record,matrix_dir):
 if not _recovery_record(record):return _normal_legacy_completion(record,matrix_dir)
 evidence=_recovery_snapshot(record,completion=True)
 return (evidence.get('completion_verified') is True and evidence.get('original_worker_exit_verified') is True and
  evidence.get('e1_status')=='PASS' and _normal_legacy_completion(record,matrix_dir))


def _wait_recovery_natural_exit(record,expected_e1,known=None):
 evidence=_recovery_snapshot(record,completion=True)
 if (evidence.get('completion_verified') is not True or evidence.get('original_worker_exit_verified') is not True or
  evidence.get('e1_status')!=expected_e1):raise ReviewRequired('Recovery completion outcome changed')
 _settled_before_resume()
 if evidence.get('owner_gone') is True:return
 deadline=time.monotonic()+10.
 try:
  process,identity=_upstream_identity_snapshot(record,known)
  while True:
   current=read_json(ROOT/'RUN_STATUS.json')
   if current!=record:raise ReviewRequired('Recovery terminal record changed while waiting for owner exit')
   if process.create_time()!=identity['create_time'] or process.cmdline()!=identity['argv']:
    raise ReviewRequired('Recovery terminal owner identity changed')
   remaining=deadline-time.monotonic()
   if remaining<=0:raise ReviewRequired('Verified completed recovery owner did not naturally exit within bounded grace')
   state('WAITING_VERIFIED_RECOVERY_NATURAL_EXIT',recovery_owner_pid=identity['pid'],
    upstream_owner_kind='validated_recovery_owner',recovery_authorization=record['recovery_authorization'],
    active_child_pid=None,active_script=None,child_ownership='authorized_recovery_e1_supervisor')
   try:code=process.wait(timeout=min(1.,remaining));break
   except psutil.TimeoutExpired:continue
  if code!=0:raise ReviewRequired('Recovery owner did not provide an observed zero exit')
  if read_json(ROOT/'RUN_STATUS.json')!=record:raise ReviewRequired('Recovery terminal record changed during natural exit')
  final=_recovery_snapshot(record,completion=True)
  if final.get('e1_status')!=expected_e1:raise ReviewRequired('Recovery completion changed during natural exit')
  _settled_before_resume()
 except psutil.NoSuchProcess:
  raise ReviewRequired('Recovery owner disappeared before a zero exit was observed; verify terminal artifacts and explicitly resume') from None
 except (psutil.AccessDenied,PermissionError):raise ReviewRequired('Recovery owner cannot be inspected during completion grace') from None


def _wait_upstream_normal_exit(record,matrix_dir,known=None):
 if not _recovery_record(record):return _wait_legacy_normal_exit(record,matrix_dir,known)
 if not _normal_upstream_completion(record,matrix_dir):raise ReviewRequired('Recovery has no verified normal matrix completion')
 _wait_recovery_natural_exit(record,'PASS',known)


def _script_from_command(process):
 cwd=Path(process.cwd())
 for value in process.cmdline():
  if isinstance(value,str) and value.lower().endswith('.py'):
   return (Path(value) if Path(value).is_absolute() else cwd/value).resolve()
 return None


def _owner_gone(record):
 if (isinstance(record.get('pid'),bool) or not isinstance(record.get('pid'),int) or record['pid']<=0
  or not isinstance(record.get('create_time'),(int,float)) or isinstance(record.get('create_time'),bool)
  or not math.isfinite(record['create_time']) or record['create_time']<=0 or not isinstance(record.get('token'),str) or not record['token']
  or record.get('script')!=str(Path(__file__).resolve())):raise ReviewRequired('Unverifiable existing master lease; no takeover')
 try:
  process=psutil.Process(record['pid'])
  if abs(process.create_time()-record['create_time'])>=.01:return True
  if _script_from_command(process)!=Path(__file__).resolve():raise ReviewRequired('Master lease PID identity ambiguous')
  return False
 except psutil.NoSuchProcess:return True
 except (psutil.AccessDenied,PermissionError):raise ReviewRequired('Master lease owner cannot be inspected') from None


def _owned_previous_child(record):
 pid=record.get('active_child_pid')
 if not pid:return None
 if record.get('child_ownership') in {'upstream_e1_supervisor','authorized_recovery_e1_supervisor'}:
  legacy=read_json(ROOT/'RUN_STATUS.json') if (ROOT/'RUN_STATUS.json').exists() else {}
  if _upstream_active(legacy):return 'FOLLOW_LEGACY'
  if _recovery_record(legacy):
   completed=_recovery_snapshot(legacy,completion=True)
   if completed.get('completion_verified') is True:return 'FOLLOW_RECOVERY_TERMINAL'
  raise ReviewRequired('Previous upstream child ownership no longer verifies; inspect its outcome')
 created=record.get('active_child_create_time');script=record.get('active_script');command=record.get('last_command')
 if not isinstance(script,str) or Path(script).name!=script or not isinstance(command,list) or not command or isinstance(created,bool) or not isinstance(created,(int,float)) or not math.isfinite(created) or created<=0:
  raise ReviewRequired('Previous active child lacks complete PID/birth/argv identity')
 try:
  process=psutil.Process(pid)
  if abs(process.create_time()-created)>=.01 or _script_from_command(process)!=(ROOT/'scripts'/script).resolve() or process.cmdline()!=command:
   raise ReviewRequired('Previous active child identity changed; do not relaunch')
  return process
 except psutil.NoSuchProcess:raise ReviewRequired('Previous active child exited without a recorded exit status; review before resuming') from None
 except (psutil.AccessDenied,PermissionError):raise ReviewRequired('Previous active child cannot be inspected') from None


def _settled_before_resume():
 for path in (ROOT/'results/raw').rglob('*.jsonl.lock'):
  raise ReviewRequired('Unsettled paid-call lock requires operator review; no inference launch: '+str(path))
 from budget_router.execution import _digest
 for path in (ROOT/'results/raw').rglob('*.jsonl'):
  with path.open(encoding='utf-8-sig') as stream:
   for line in stream:
    if not line.strip():continue
    try:row=json.loads(line)
    except ValueError:raise ReviewRequired('Uncommitted/corrupt raw store; no automatic retry') from None
    if row.get('record_sha256')!=_digest({k:v for k,v in row.items() if k!='record_sha256'}) or row.get('cost_status')!='recorded_actual_usage':
     raise ReviewRequired('Unknown raw attempt cost/integrity requires reconciliation; no automatic retry')


@contextmanager
def _master_guard():
 path=ROOT/'logs/full_experiment_supervisor_guard.lock';path.parent.mkdir(parents=True,exist_ok=True)
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
  except OSError:raise ReviewRequired('Another master starter is reviewing ownership') from None
  try:yield
  finally:
   if os.name=='nt':stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
   else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def _acquire_master_lease(resume=False):
 lock=ROOT/'logs/full_experiment_supervisor.lock';lock.parent.mkdir(parents=True,exist_ok=True)
 previous=read_json(ROOT/'FULL_RUN_STATUS.json') if (ROOT/'FULL_RUN_STATUS.json').exists() else {}
 # A live previous master is not adoptable, even if its lease was accidentally removed.
 if previous.get('master_running') is True and previous.get('master_pid')!=os.getpid():
  candidate={'pid':previous.get('master_pid'),'create_time':previous.get('process_create_time'),
   'script':previous.get('master_script'),'token':'status-owner'}
  if not _owner_gone(candidate):raise ReviewRequired('A verified existing master is still alive')
 child_process=None
 if lock.exists():
  if not resume:raise ReviewRequired('Existing master lease requires explicit --resume and identity review')
  old=read_json(lock)
  if not _owner_gone(old):raise ReviewRequired('A verified existing master is still alive')
 if previous.get('active_child_pid'):
  if not resume:raise ReviewRequired('An existing active child requires --resume; never launch another GPU job')
  child_process=_owned_previous_child(previous)
 if resume and child_process is None and not _upstream_active(read_json(ROOT/'RUN_STATUS.json') if (ROOT/'RUN_STATUS.json').exists() else {}):_settled_before_resume()
 if lock.exists():
  folder=ROOT/'artifacts/private/master_recovery';folder.mkdir(parents=True,exist_ok=True)
  os.rename(lock,folder/('verified_dead_master_'+uuid.uuid4().hex+'.json'))
 token=uuid.uuid4().hex;fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 try:
  os.write(fd,json.dumps({'pid':os.getpid(),'create_time':psutil.Process().create_time(),'token':token,
   'script':str(Path(__file__).resolve()),'source_sha256':SOURCE_SHA}).encode());os.fsync(fd)
 finally:os.close(fd)
 return lock,token,previous,child_process


@contextmanager
def master_lease(resume=False):
 with _master_guard():lock,token,previous,child_process=_acquire_master_lease(resume)
 try:yield previous,child_process
 finally:
  with _master_guard():
   if lock.exists() and read_json(lock).get('token')==token:lock.unlink()


def follow_previous_child(previous,process):
 if process is None or process in ('FOLLOW_LEGACY','FOLLOW_RECOVERY_TERMINAL'):return
 state('FOLLOWING_PREVIOUS_VERIFIED_CHILD_NO_RELAUNCH',active_child_pid=previous['active_child_pid'],
  active_child_create_time=previous['active_child_create_time'],active_script=previous['active_script'],
  last_command=previous['last_command'],child_ownership='adopted_verified_orphan')
 while True:
  try:code=process.wait(timeout=20);break
  except psutil.TimeoutExpired:_pulse()
 status=read_json(ROOT/'FULL_RUN_STATUS.json');status.update(active_child_pid=None,active_script=None,
  last_child_script=previous['active_script'],last_child_exit=code,updated_at_unix=time.time());atomic_json(ROOT/'FULL_RUN_STATUS.json',status)
 if code!=0:raise ReviewRequired('Previous child did not provide a verified zero exit; no automatic retry')
 _settled_before_resume()


def _legacy_identity_snapshot(record,known=None):
 pid=record.get('supervisor_pid')
 if isinstance(pid,bool) or not isinstance(pid,int) or pid<=0:raise ReviewRequired('Legacy owner PID is unverifiable')
 process=psutil.Process(pid);birth=process.create_time();command=process.cmdline()
 if process.status() in (psutil.STATUS_STOPPED,psutil.STATUS_ZOMBIE):raise ReviewRequired('Legacy owner is paused or not executing; no completion grace')
 if _script_from_command(process)!=(ROOT/'scripts/run_local_e1.py').resolve() or '--continue-online-after-e1-pass' not in command:
  raise ReviewRequired('Legacy owner argv/script does not match completed online generation')
 if known is None:
  path=ROOT/'logs/local_e1_supervisor.lock';lock=read_json(path) if path.exists() else {}
  started=lock.get('started_at_unix')
  if (lock.get('pid')!=pid or isinstance(started,bool) or not isinstance(started,(int,float))
   or not math.isfinite(started) or abs(birth-started)>2):raise ReviewRequired('No verified legacy owner birth/lock binding for completion grace')
 elif known.get('pid')!=pid or known.get('create_time')!=birth or known.get('argv')!=command:
  raise ReviewRequired('Legacy exact PID/birth/argv changed during completion grace')
 return process,{'pid':pid,'create_time':birth,'argv':command}


def _normal_legacy_completion(record,matrix_dir):
 finished=record.get('supervisor_finished_at_unix');receipt=matrix_dir/'matrix_preparation_receipt.json'
 return (record.get('supervisor_running') is False and record.get('E0')=='PASS' and record.get('E1')=='PASS'
  and record.get('overall')=='PARTIAL' and record.get('step')=='REAL_ONLINE_MATRICES_READY'
  and record.get('E2')=='REAL_MATRICES_READY_NOT_E2_VALIDATION' and record.get('e2_executed') is False
  and record.get('scientific_completion') is False and record.get('active_child_pid') is None and record.get('active_script') is None
  and record.get('last_child_script')=='prepare_online_matrices.py' and record.get('last_child_exit')==0
  and not isinstance(finished,bool) and isinstance(finished,(int,float)) and math.isfinite(finished) and finished>0
  and record.get('matrix_preparation_receipt')==receipt.relative_to(ROOT).as_posix()
  and record.get('matrix_preparation_receipt_sha256')==artifact_reference(receipt)['sha256'])


def _wait_legacy_normal_exit(record,matrix_dir,known=None):
 """Bounded read-only grace for a verified normal completion; never signal/retry.

 If an owner vanishes before a zero exit is observed, stop for review. After
 independently checking its completed status, matrices and settled ledgers,
 an explicit master --resume can use the normal already-gone-owner path;
 it never reruns E0/E1 inference or removes another process's execution lock.
 """
 from budget_router.online_continuation import verify_matrix_receipt
 if not _normal_legacy_completion(record,matrix_dir):raise ReviewRequired('Upstream is paused or identity-unverifiable, not proven exited; do not advance')
 _settled_before_resume();deadline=time.monotonic()+10.
 try:
  process,identity=_legacy_identity_snapshot(record,known)
  while True:
   current=read_json(ROOT/'RUN_STATUS.json');result=read_json(ROOT/'results/e1/e1_diagnostics.json')
   verify_matrix_receipt(ROOT,matrix_dir)
   if current!=record or result.get('status')!='PASS' or not _normal_legacy_completion(current,matrix_dir):
    raise ReviewRequired('Legacy completion evidence changed during natural-exit grace')
   process,_=_legacy_identity_snapshot(current,identity)
   remaining=deadline-time.monotonic()
   if remaining<=0:raise ReviewRequired('Verified completed legacy did not naturally exit within the bounded grace period')
   state('WAITING_VERIFIED_LEGACY_NATURAL_EXIT',legacy_supervisor_pid=identity['pid'],legacy_step=current['step'],
    active_child_pid=None,active_script=None,child_ownership='upstream_e1_supervisor')
   try:
    code=process.wait(timeout=min(1.,remaining))
   except psutil.TimeoutExpired:continue
   if code!=0:raise ReviewRequired('Legacy final exit is nonzero or unknown; no automatic advance')
   if read_json(ROOT/'RUN_STATUS.json')!=record or read_json(ROOT/'results/e1/e1_diagnostics.json').get('status')!='PASS':
    raise ReviewRequired('Legacy completion changed while its owner exited')
   verify_matrix_receipt(ROOT,matrix_dir);_settled_before_resume();return
 except psutil.NoSuchProcess:
  raise ReviewRequired('Legacy owner disappeared before a zero exit was observed; independently verify completed status/matrices/settled costs, then explicitly --resume; never relaunch E0/E1 or clear paid locks') from None
 except (psutil.AccessDenied,PermissionError):raise ReviewRequired('Legacy owner identity cannot be inspected during completion grace') from None


def wait_real_matrices():
 from budget_router.online_continuation import verify_matrix_receipt
 known_owner=None
 while True:
  path=ROOT/"RUN_STATUS.json";legacy=read_json(path) if path.exists() else {}
  active=_upstream_active(legacy)
  if active:
   try:_,known_owner=_upstream_identity_snapshot(legacy,known_owner)
   except psutil.NoSuchProcess:pass
  e1_path=ROOT/"results/e1/e1_diagnostics.json"
  if e1_path.exists():
   result=read_json(e1_path)
   if result.get("status")=="STOP_E1":
    if _recovery_record(legacy):
     if not legacy.get('recovery_completion'):
      if not active:raise ReviewRequired('Recovery STOP_E1 has no verified completion')
      state('WAITING_RECOVERY_STOP_E1_TERMINAL',recovery_owner_pid=legacy.get('supervisor_pid'),
       upstream_owner_kind='validated_recovery_owner',recovery_authorization=legacy.get('recovery_authorization'),
       active_child_pid=legacy.get('active_child_pid'),active_script=legacy.get('active_script'),child_ownership='authorized_recovery_e1_supervisor')
      time.sleep(30);continue
     _wait_recovery_natural_exit(legacy,'STOP_E1',known_owner)
    return "STOP_E1"
   if result.get("status")=="PASS":
    start_second_download_if_needed()
    matrix_dir=ROOT/"data/derived/online_matrices_v1"
    receipt=matrix_dir/"matrix_preparation_receipt.json"
    if receipt.exists():
     verify_matrix_receipt(ROOT,matrix_dir)
     if not active:
      if not _normal_upstream_completion(legacy,matrix_dir):raise ReviewRequired("Upstream is paused or identity-unverifiable, not proven exited; do not advance")
      _settled_before_resume()
      if _upstream_owner_gone(legacy):return "PASS"
      _wait_upstream_normal_exit(legacy,matrix_dir,known_owner);return "PASS"
  if not active:
   raise ReviewRequired("The upstream E0/E1 supervisor is not active and has no accepted complete outcome; preserve and review actual costs before resuming")
  ownership={'child_ownership':'authorized_recovery_e1_supervisor','upstream_owner_kind':'validated_recovery_owner',
   'recovery_owner_pid':legacy.get('supervisor_pid'),'recovery_authorization':legacy.get('recovery_authorization'),
   'legacy_supervisor_pid':None} if _recovery_record(legacy) else {'child_ownership':'upstream_e1_supervisor','legacy_supervisor_pid':legacy.get('supervisor_pid')}
  state("WAITING_E0_E1_REAL_ASSETS",E0=legacy.get("E0"),E1=legacy.get("E1"),legacy_step=legacy.get("step"),
   active_child_pid=legacy.get("active_child_pid"),active_script=legacy.get("active_script"),**ownership)
  time.sleep(30)


def _inputs():
 d=ROOT/"data/derived/online_matrices_v1"
 return input_identity(ROOT,online_matrix=d/"online_stream.npz",online_manifest=d/"online_stream.manifest.json",
  calibration_matrix=d/"online_calibration.npz",calibration_manifest=d/"online_calibration.manifest.json",config_path=ROOT/"configs/experiment_v1.json")


def matrix_args():
 d=ROOT/"data/derived/online_matrices_v1"
 return ["--online-matrix",d/"online_stream.npz","--online-manifest",d/"online_stream.manifest.json",
  "--calibration-matrix",d/"online_calibration.npz","--calibration-manifest",d/"online_calibration.manifest.json"]


def _resume(directory):return ["--resume"] if Path(directory).exists() else []


def accept_legitimate_e1_stop():
 from run_local_e1 import verify_e0_after_child
 from evaluate_real_e1 import assemble_e1
 verify_e0_after_child();original_path=ROOT/'results/e1/e1_diagnostics.json';original=read_json(original_path)
 verified=assemble_e1(ROOT)
 if verified!=original or verified.get('status')!='STOP_E1':raise ReviewRequired('E1 stop does not reproduce exactly')
 physical=verified.get('physical_diagnostic_cost',{})
 if physical.get('all_costs_reconciled') is not True or physical.get('unique_calls',0)<8800:raise ReviewRequired('E1 stop lacks full real controls and reconciled actual costs')
 replay=OUT/'stop_e1_replay'/(artifact_reference(original_path)['sha256']+'.json');immutable_json(replay,verified)
 receipt={'schema':'final_experiment_acceptance_v1','status':'LEGITIMATE_STOP_E1_VERIFIED',
  'research_complete':False,'workflow_execution_complete':True,'ready_for_github_sync':True,
  'outcome':'Complete real E1 did not pass the preregistered opportunity gate; E2-E5 are not executed or claimed',
  'e1_result':artifact_reference(original_path),'independent_replay':artifact_reference(replay),
  'E0':'PASS','E1':'STOP_E1','E2_E5':'NOT_RUN_PREREGISTERED_STOP','master_source_sha256':SOURCE_SHA}
 immutable_json(ROOT/'FINAL_ACCEPTANCE.json',receipt);return receipt


def _admit_gpu():
 from run_local_e1 import gpu_snapshot
 count=0
 while count<2:
  sample=gpu_snapshot();okay=(sample["free_mib"]>=6144 and sample["utilization_percent"]<=20
   and not sample["other_python_gpu_pids"] and sample["host_available_bytes"]>=5*1024**3)
  count=count+1 if okay else 0
  state("WAITING_FINAL_OR_SECOND_AGGREGATOR_GPU_ADMISSION",gpu=sample,consecutive_ready_samples=count)
  if count<2:time.sleep(30)


def ensure_second_aggregator():
 state("E5_SECOND_AGGREGATOR_ASSET_AND_RUNTIME_QUALIFICATION")
 wait_second_download()
 child("prepare_second_aggregator.py")
 _admit_gpu()
 child("run_aggregation.py","--jobs",ROOT/"data/jobs/qualification/aggregator_mistral_v03.jsonl",
  "--model-config",ROOT/"configs/aggregator_mistral_v03.json","--output",ROOT/"results/raw/qualification/aggregator_mistral_v03.jsonl",
  "--qualification","--max-jobs","4")
 child("review_runtime.py","--jobs",ROOT/"data/jobs/qualification/aggregator_mistral_v03.jsonl",
  "--model-config",ROOT/"configs/aggregator_mistral_v03.json","--outputs",ROOT/"results/raw/qualification/aggregator_mistral_v03.jsonl",
  "--receipt",ROOT/"artifacts/qualification/aggregator_mistral_v03.json")


def start_second_download_if_needed():
 """Download assets while the original supervisor continues real GPU work."""
 completed=ROOT/"artifacts/audit/aggregator_second_model_download.json"
 if completed.is_file():
  d=read_json(completed)
  if d.get("completed_at_unix") and d.get("repo_id")=="mistralai/Mistral-7B-Instruct-v0.3" and d.get("revision")=="c170c708c41dac9275d15a8fff4eca08d52bab71":return
 launch=ROOT/"artifacts/audit/background_second_aggregator_download.json"
 if launch.is_file():
  info=read_json(launch)
  try:
   proc=psutil.Process(info["pid"])
   if abs(proc.create_time()-info["create_time"])<.01 and str(ROOT/"scripts/download_second_aggregator.py") in proc.cmdline():return
  except psutil.Error:pass
 log=ROOT/"logs/second_aggregator_asset_download.log";log.parent.mkdir(exist_ok=True)
 with log.open("ab") as output:
  proc=subprocess.Popen([str(PYTHON),"-X","utf8","-u",str(ROOT/"scripts/download_second_aggregator.py")],cwd=ROOT,
   stdout=output,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
 atomic_json(launch,{"pid":proc.pid,"create_time":psutil.Process(proc.pid).create_time(),"script":str(ROOT/"scripts/download_second_aggregator.py"),"log":str(log),"research_complete":False})


def wait_second_download():
 start_second_download_if_needed()
 target=ROOT/"artifacts/audit/aggregator_second_model_download.json"
 while not target.exists():
  launch=read_json(ROOT/"artifacts/audit/background_second_aggregator_download.json")
  try:
   p=psutil.Process(launch["pid"])
   if abs(p.create_time()-launch["create_time"])>=.01:raise ReviewRequired("Second download PID changed")
  except psutil.NoSuchProcess:raise ReviewRequired("Second aggregator download exited without a completed source receipt")
  _pulse();time.sleep(30)
 # The actual backend still rehashes every model file before loading.
 value=read_json(target)
 if not value.get("completed_at_unix") or not value.get("files"):raise ReviewRequired("Incomplete second-model source receipt")


def run_restricted_stages():
 from budget_router.stage_validation import validate_e2_gate,validate_e3_gate
 from budget_router.neighbor_validation import validate_e4_gate
 from budget_router.flex_validation import validate_e4_full_gate
 from budget_router.e5_validation import validate_e5_pre_final_gate
 inputs=_inputs();common=matrix_args();gates={}
 e2=OUT/"e2";e2gate=e2/"e2_gate.json"
 state("E2_FULL_FIVE_BASELINES",E2="RUNNING")
 if not e2gate.exists():
  child("run_online.py","--stage","e2",*common,"--output-dir",e2,*_resume(e2))
  child("evaluate_e2.py","--runs-dir",e2,"--output-gate",e2gate)
 validate_e2_gate(e2gate,inputs);gates["E2"]=artifact_reference(e2gate)
 e3=OUT/"e3";e3gate=e3/"e3_gate.json"
 state("E3_FULL_PAID_FEEDBACK_MECHANISMS",E2="TECHNICAL_PASS",E3="RUNNING")
 if not e3gate.exists():
  child("run_online.py","--stage","e3",*common,"--e2-gate",e2gate,"--output-dir",e3,*_resume(e3))
  child("evaluate_e3.py","--runs-dir",e3,"--output-gate",e3gate)
 validate_e3_gate(e3gate,inputs);gates["E3"]=artifact_reference(e3gate)
 e4=OUT/"e4/mandatory";mandatory=e4/"e4_mandatory_gate.json"
 state("E4_MANDATORY_CES_AND_C2MABV",E3="TECHNICAL_PASS",E4="RUNNING")
 if not mandatory.exists():
  child("run_neighbors.py",*common,"--e3-gate",e3gate,"--output-dir",e4,*_resume(e4))
  child("evaluate_e4.py","--directory",e4,"--gate",mandatory)
 validate_e4_gate(mandatory,inputs)
 flex=OUT/"e4/flex_reference";flexgate=flex/"flex_reference_gate.json";fullgate=OUT/"e4/e4_full_gate.json"
 state("E4_EXPLICITLY_PRICED_FLEX_COVERAGE_REFERENCE",E4="MANDATORY_NEIGHBORS_PASS_FLEX_RUNNING")
 if not fullgate.exists():
  child("run_flex_reference.py",*common,"--e3-gate",e3gate,"--mandatory-e4-gate",mandatory,"--output-dir",flex,*_resume(flex))
  child("evaluate_flex_reference.py","--grid-dir",flex,"--flex-gate",flexgate,"--e4-full-gate",fullgate)
 validate_e4_full_gate(fullgate,inputs);gates["E4"]=artifact_reference(fullgate)
 e5=OUT/"e5";e5gate=e5/"e5_pre_final_gate.json"
 state("E5_COMPLETE_PREREGISTERED_ABLATIONS",E4="FULL_TECHNICAL_PASS",E5="RUNNING_PRE_FINAL")
 if not e5gate.exists():
  child("run_e5_ablations.py",*common,"--e4-full-gate",fullgate,"--output-dir",e5,"--output-gate",e5gate,*_resume(e5))
 validate_e5_pre_final_gate(e5gate,inputs);gates["E5_PRE_FINAL"]=artifact_reference(e5gate)
 return gates


def prepare_final_release(gates):
 from budget_router.frozen_evaluation import validate_preparation_release,validate_final_release,_read_frozen_decisions
 pre=ROOT/"protocol/FINAL_FEATURE_PREPARATION_RELEASE.json"
 if not pre.exists():
  child("build_final_release.py","--kind","preparation","--e5-pre-final-gate",gates["E5_PRE_FINAL"]["path"],
   "--e4-full-gate",gates["E4"]["path"],"--output",pre)
 validate_preparation_release(pre)
 features=ROOT/"data/derived/features_final_v1"
 if not features.exists():child("prepare_final_features.py","--preparation-release",pre,"--output",features)
 release=ROOT/"protocol/FINAL_RELEASE.json"
 if not release.exists():
  child("build_final_release.py","--kind","complete","--preparation-release",pre,
   "--features",features/"final_features_v1.npz","--feature-manifest",features/"sealed_features_manifest.json","--output",release)
 validated=validate_final_release(release)
 decisions=OUT/"final/frozen_decisions"
 if not decisions.exists():child("evaluate_frozen_policies.py","freeze","--release",release,"--output",decisions)
 _read_frozen_decisions(decisions,validated)
 return release,decisions


def run_all_final_conditions(release,decisions):
 conditions=read_json(release)["evaluation_conditions"]
 state("FINAL_ALL_FROZEN_CHOICES_RECORDED_GENERATING_REAL_CACHES",E5="PRE_FINAL_TECHNICAL_PASS",FINAL="REAL_GENERATION_RUNNING")
 child("prepare_final_jobs.py","--kind","members","--release",release,"--decisions-dir",decisions)
 seeds=sorted({c["generation_seed"] for c in conditions});pool=read_json(ROOT/"configs/pool_v1.json")["models"]
 for seed in seeds:
  for entry in pool:
   manifest=ROOT/f"data/jobs/final/generation{seed}/members_{entry['alias'].lower()}.manifest.json"
   state("FINAL_REAL_MEMBER_GENERATION",final_generation_seed=seed,active_model=entry["alias"])
   child("run_final_generation.py","--jobs-manifest",manifest)
 evaluations=[]
 for condition in conditions:
  name=condition["condition_id"]
  state("FINAL_REAL_FROZEN_AGGREGATION",final_condition=name)
  child("prepare_final_jobs.py","--kind","aggregations","--release",release,"--decisions-dir",decisions,"--condition",name)
  child("run_final_generation.py","--jobs-manifest",ROOT/f"data/jobs/final/{name}/aggregations.manifest.json")
  child("prepare_final_matrices.py","--release",release,"--decisions-dir",decisions,"--condition",name,
   "--output-dir",ROOT/"results/raw/final/matrices")
  manifest=ROOT/f"results/raw/final/matrices/{name}/manifest.json"
  if not manifest.is_file():raise ReviewRequired('Expected atomic final matrix bundle manifest.json is absent')
  matrix=verified_reference(read_json(manifest)['matrix'])
  if matrix!=(manifest.parent/'matrix.npz').resolve():raise ReviewRequired('Final matrix bundle references a different matrix')
  output=OUT/f"final/{name}.evaluation.json"
  if not output.exists():
   child("evaluate_frozen_policies.py","evaluate","--release",release,"--decisions",decisions,
    "--matrix",matrix,"--manifest",manifest,"--output",output)
  evaluation=read_json(output)
  if evaluation.get("status")!="EVALUATED_FROZEN_FINAL_POLICIES" or evaluation.get("condition_id")!=name:
   raise ReviewRequired("Final independent evaluation incomplete or wrong condition")
  verified_reference(evaluation["matrix"]);verified_reference(evaluation["matrix_manifest"])
  evaluations.append(artifact_reference(output))
 return evaluations


def physical_model_accounting(*,project_root=None,write_receipt=True):
 root=Path(project_root).resolve() if project_root is not None else ROOT
 from budget_router.execution import _digest
 from budget_router.inference import _validated_usage
 pool=read_json(root/"configs/pool_v1.json")["models"];aliases=[e["alias"].lower() for e in pool]
 paths=[root/"results/raw/qualification/aggregator.jsonl",root/"results/raw/qualification/aggregator_mistral_v03.jsonl"]
 paths += [root/f"results/raw/qualification/members_{alias}.jsonl" for alias in aliases]
 for stage in ("e1","online_calibration","online_stream"):
  paths.extend(root/f"results/raw/{stage}/members_{alias}.jsonl" for alias in aliases)
  paths.append(root/f"results/raw/{stage}/aggregations.jsonl")
 paths.extend(root/f"results/raw/e1/{name}.jsonl" for name in ("aggregator_alone","strong_samples","strong_aggregations"))
 release=read_json(root/"protocol/FINAL_RELEASE.json")
 seeds=sorted({c["generation_seed"] for c in release["evaluation_conditions"]})
 paths.extend(root/f"results/raw/final/generation{seed}/members_{alias}.jsonl" for seed in seeds for alias in aliases)
 paths.extend(root/f"results/raw/final/{c['condition_id']}/aggregations.jsonl" for c in release["evaluation_conditions"])
 attempts={};refs=[];totals={"calls":0,"prompt_tokens":0,"completion_tokens":0,"model_call_wall_time_s":0.,"failed_calls":0}
 for path in paths:
  if not path.is_file() or path.with_name(path.name+".lock").exists():raise ReviewRequired("Missing or unsettled real model store: "+str(path))
  refs.append(artifact_reference(path))
  for line in path.read_text("utf8").split("\n"):
   if not line.strip():continue
   row=json.loads(line);digest=row.get("record_sha256")
   if digest!=_digest({k:v for k,v in row.items() if k!="record_sha256"}):raise ReviewRequired("Physical model call checksum mismatch")
   attempt=row.get("attempt_id")
   if not attempt:raise ReviewRequired("Model call has no unique physical attempt identity")
   if attempt in attempts:
    if attempts[attempt]!=digest:raise ReviewRequired("Conflicting physical call attempt identity")
    continue
   attempts[attempt]=digest
   if row.get("cost_status")!="recorded_actual_usage" or row.get("status") not in ("success","failure"):
    raise ReviewRequired("Unknown actual model call cost/status")
   result=row.get("result",{}) if row["status"]=="success" else row.get("actual_usage")
   usage=_validated_usage(result)
   wall=row.get("result",{}).get("wall_time_s") if row["status"]=="success" else row.get("failure_wall_time_s")
   if usage is None or isinstance(wall,bool) or not isinstance(wall,(int,float)) or not __import__("math").isfinite(wall) or wall<0:
    raise ReviewRequired("Unreconciled actual physical model tokens/time")
   totals["calls"]+=1;totals["prompt_tokens"]+=usage["prompt_tokens"];totals["completion_tokens"]+=usage["completion_tokens"]
   totals["model_call_wall_time_s"]+=wall;totals["failed_calls"]+=int(row["status"]=="failure")
 if totals["calls"]<49256:raise ReviewRequired("Full original real call grid and qualifications are incomplete")
 value={"schema":"complete_physical_model_call_accounting_v1","totals":totals,"source_stores":refs,
  "unique_physical_attempts":len(attempts),"attempt_identity_sha256":canonical_digest(attempts),
  "cross_condition_member_reuse_counted_once":True,"physical_is_not_sum_of_logical_policy_services":True,
  "scope":"Recorded real model attempts only, including failures. Model loading, source SHA checks, CUDA cache housekeeping, controller CPU, grading and wall waiting remain separate measurements; model wall seconds are not GPU kernel seconds.",
  "research_complete":False}
 if not write_receipt:return value
 path=(OUT if project_root is None else root/"results/full_v1")/"complete_physical_model_costs.json";immutable_json(path,value);return artifact_reference(path)


def accept_full_experiment(gates,release,decisions,evaluations):
 from run_local_e1 import verify_e0_after_child
 from budget_router.frozen_evaluation import evaluate_final_matrix
 from budget_router.e5_validation import validate_e5_pre_final_gate
 verify_e0_after_child();validate_e5_pre_final_gate(gates["E5_PRE_FINAL"]["path"],_inputs())
 from build_delivery import validate_final_evaluation_set,compute_final_statistics,validate_final_grading_measurement
 conditions=read_json(release)["evaluation_conditions"]
 if len(evaluations)!=4 or len(conditions)!=4:raise ReviewRequired("All four predeclared final conditions are required")
 evaluated=[]
 for reference in evaluations:
  result=read_json(verified_reference(reference));evaluated.append(result)
  replay=evaluate_final_matrix(release,decisions,verified_reference(result["matrix"]),verified_reference(result["matrix_manifest"]))
  if replay!=result:raise ReviewRequired("Final evaluation does not exactly reproduce from immutable choices and real matrices")
  validate_final_grading_measurement(result)
  groups=result.get("summaries",[])
  frozen=read_json(release)["policy_groups"]
  expected={(group["method"],entry["seed"]) for group in frozen for entry in group["checkpoints"]}
  if len(groups)!=1080 or {(g.get("method"),g.get("seed")) for g in groups}!=expected:
   raise ReviewRequired("Final evaluation must contain all216 method/budget/delay groups and all five states")
  for group in groups:
   rows=group.get("evaluation_question_rows",[])
   if group.get("n_questions")!=200 or len(rows)!=200 or len({r["question_id"] for r in rows})!=200:
    raise ReviewRequired("Final evaluation lacks complete independent question support")
   if {task:sum(r["task"]==task for r in rows) for task in ("gsm8k","bbh")}!={"gsm8k":100,"bbh":100}:
    raise ReviewRequired("Final task balance changed")
  if result.get("research_complete") is not False:raise ReviewRequired("A final condition must not self-claim whole-project completion")
 validate_final_evaluation_set(read_json(release),evaluated)
 paired=compute_final_statistics(release,evaluations,decisions)
 paired_path=OUT/'final/complete_paired_statistics.json';immutable_json(paired_path,paired)
 physical=physical_model_accounting()
 value={"schema":"final_experiment_acceptance_v1","status":"FULL_REQUIRED_EXPERIMENTS_VERIFIED",
  "E0":"PASS","E1":"PASS","E2_E3_E4":"TECHNICAL_PASS_SCIENTIFIC_VERDICTS_RETAINED",
  "E5":"PRE_FINAL_ABLATIONS_AND_ALL_FOUR_FINAL_CONDITIONS_VERIFIED", "research_complete":True,
  "workflow_execution_complete":True,"ready_for_github_sync":True,"github_sync_complete":False,
  "e1_result":artifact_reference(ROOT/"results/e1/e1_diagnostics.json"),"stage_gates":gates,"release":artifact_reference(release),"decisions":artifact_reference(Path(decisions)/"decision_receipt.json"),
  "final_evaluations":evaluations,"physical_model_accounting":physical,"master_source_sha256":SOURCE_SHA,
  "final_paired_statistics":artifact_reference(paired_path),"analysis_sources":{name:artifact_reference(ROOT/name) for name in ("scripts/build_delivery.py","scripts/run_full_experiment.py")},
  "claim_boundary":"Complete execution/acceptance does not imply a positive algorithmic result, native neighbour-system reproduction, identified causal interactions, USD pricing or measured GPU kernel time."}
 immutable_json(ROOT/"FINAL_ACCEPTANCE.json",value);return value


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--resume",action="store_true");p.add_argument("--dry-run",action="store_true")
 args=p.parse_args(argv)
 if args.dry_run:
  result={"mode":"DRY_RUN_NO_MODEL_CALLS","sequence":["follow_real_E0_E1","E2","E3","E4_mandatory","Flex_reference","E5_pre_final",
   "second_aggregator_qualification","preparation_release","final_features_only","final_release","freeze_choices","real_all_final_conditions","independent_final_acceptance","GitHub_sync_by_authorized_connector"],
   "original_scale":{"E1":300,"calibration":200,"online":600,"final":200,"N":6,"K":3,"teams":20,"algorithm_seeds":5,"frozen_final_groups":216,"generation_conditions":4},"research_complete":False}
  print(json.dumps(result));return result
 intent=read_json(ROOT/"FULL_RUN_INTENT.json")
 if intent.get("github_write_authorized") is not True or intent.get("workspace")!=str(ROOT):raise ReviewRequired("Explicit full-run intent mismatch")
 if (ROOT/'FINAL_ACCEPTANCE.json').exists():
  from build_delivery import validate_acceptance
  return validate_acceptance(ROOT)['acceptance']
 with master_lease(args.resume) as (previous,previous_child):
  power=None
  if os.name=='nt':
   power=ctypes.WinDLL('kernel32').SetThreadExecutionState;power.argtypes=[ctypes.c_uint];power.restype=ctypes.c_uint
   power(0x80000001)
  try:
   follow_previous_child(previous,previous_child)
   state("FOLLOWING_EXISTING_REAL_E0_E1_SUPERVISOR",active_child_pid=None,active_script=None)
   outcome=wait_real_matrices()
   if outcome=="STOP_E1":acceptance=accept_legitimate_e1_stop()
   else:
    gates=run_restricted_stages();ensure_second_aggregator()
    state("FINAL_PREPARATION_RELEASE_AND_QUESTION_ONLY_FEATURES")
    release,decisions=prepare_final_release(gates)
    evaluations=run_all_final_conditions(release,decisions)
    state("INDEPENDENT_FULL_EXPERIMENT_ACCEPTANCE")
    acceptance=accept_full_experiment(gates,release,decisions,evaluations)
   state("EXPERIMENTS_VERIFIED_AWAITING_GITHUB_SYNC",overall=acceptance["status"],master_running=False,
    scientific_completion=acceptance["research_complete"],workflow_execution_complete=True,workflow_complete=False,
    final_acceptance=artifact_reference(ROOT/"FINAL_ACCEPTANCE.json"),active_child_pid=None,active_script=None,
    github_upload="READY_FOR_AUTHORIZED_CONNECTOR_PUSH_AFTER_DELIVERY_REVIEW")
   return acceptance
  except BaseException as error:
   state("STOPPED_FOR_REVIEW",overall="STOPPED_FOR_REVIEW",master_running=False,error_type=type(error).__name__,
    error_summary=str(error),scientific_completion=False,workflow_complete=False)
   raise
 
  finally:
   if power is not None:power(0x80000000)



if __name__=="__main__":main()
