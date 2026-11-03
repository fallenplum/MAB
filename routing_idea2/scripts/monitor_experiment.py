"""Low-memory, read-only experiment observer; never executes or retries model jobs.

Only monitor-owned files under artifacts/monitor are written. Ledger counters
are incremental observations, not stage acceptance or research-completion proof.
"""
from pathlib import Path
from collections import Counter
from contextlib import contextmanager
import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
import uuid
import sys
import tempfile

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from budget_router.recovery_runtime import atomic_json as recovery_atomic_json,validated_recovery_owner,validated_recovery_completion,RecoveryAuthorizationError
STAGES = ('qualification', 'e1', 'online_calibration', 'online_stream', 'final')
MAX_LINE_BYTES = 2 * 1024 * 1024
SCHEMA = 'experiment_monitor_v1'
RECENT_WALL_SAMPLES = 128
WALL_BOOTSTRAP_BYTES = 2 * 1024 * 1024
ACTIVE_RUN_STATUSES = {'IN_PROGRESS', 'IN_PROGRESS_TECHNICAL_RECOVERY', 'RUNNING', 'WAITING'}
TERMINAL_RUN_STATUSES = {'FULL_REQUIRED_EXPERIMENTS_VERIFIED': True, 'LEGITIMATE_STOP_E1_VERIFIED': False}


def atomic_json(path, value):
    return recovery_atomic_json(path,value)


@contextmanager
def open_shared_read(path):
    """Read-only handle which permits the writer to rename/delete on Windows.

    Snapshots may disappear or change; the existing double-read identity checks
    remain responsible for treating those races as unverified observations.
    """
    if os.name != 'nt':
        with Path(path).open('rb') as stream:
            yield stream
        return
    import ctypes
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(Path(path)), 0x80000000, 0x1 | 0x2 | 0x4,
                                None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    # fdopen owns the native handle after open_osfhandle succeeds.
    try:
        stream = os.fdopen(descriptor, 'rb')
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        yield stream


def read_bytes_shared(path):
    with open_shared_read(path) as stream:
        return stream.read()


def read_json(path):
    try:
        return json.loads(read_bytes_shared(path).decode('utf-8-sig'), parse_constant=reject_constant)
    except FileNotFoundError:
        return None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def reject_constant(value):
    raise ValueError('Non-finite JSON constant')


def stage_glob(stage,suffix):
    return '*/*'+suffix if stage=='final' else '*'+suffix


def valid_ledger_key(parts):
    if len(parts)<4 or parts[:2]!=['results','raw'] or parts[2] not in STAGES:return False
    expected=5 if parts[2]=='final' else 4
    return len(parts)==expected and all(x not in ('','..','.') for x in parts) and parts[-1].endswith('.jsonl')


def load_status_support(root):
    spec = importlib.util.spec_from_file_location('live_status_for_monitor_' + uuid.uuid4().hex,
                                                Path(__file__).with_name('show_status.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = Path(root).resolve()
    return module


class RecoveryObservationError(ValueError):
    pass


def recovery_record(record):
    return isinstance(record,dict) and (any(key in record for key in ('recovery_authorization','recovery_owner','recovery_completion','recovery_phase')) or
        Path(str(record.get('supervisor_script',''))).name=='resume_e1_aggregation.py')


def observe_recovery_run(root,record,process_factory):
    active=record.get('supervisor_running') is True
    validator=validated_recovery_owner if active else validated_recovery_completion
    try:
        observed=validator(root,record,process_factory=process_factory)
    except (RecoveryAuthorizationError,OSError,ValueError,TypeError,KeyError,psutil.Error) as error:
        raise RecoveryObservationError('Recovery owner or terminal authorization could not be verified') from error
    if active and (observed.get('identity_verified') is not True or observed.get('execution_active') is not True):
        raise RecoveryObservationError('Recovery owner is not verified executing')
    if not active and (observed.get('completion_verified') is not True or observed.get('original_worker_exit_verified') is not True or
        observed.get('e1_status')!=record.get('E1')):
        raise RecoveryObservationError('Recovery completion does not match terminal E1 status')
    owner_gone=observed.get('owner_gone') is True
    if not active and not owner_gone and (observed.get('identity_verified') is not True or observed.get('execution_active') is not True):
        raise RecoveryObservationError('Recovery terminal owner remains present but unverified or paused')
    if owner_gone and observed.get('identity_verified') is True:
        raise RecoveryObservationError('Historical recovery identity cannot be reported as a live identity')
    result=dict(record)
    result.update(recorded_status=record.get('overall'),recorded_supervisor_running=record.get('supervisor_running'),
        effective_status=record.get('overall'),recovery_owner_observation_verified=True,
        recovery_completion_verified=observed.get('completion_verified') is True,
        original_worker_exit_verified=observed.get('original_worker_exit_verified') is True,
        historical_identity_verified=observed.get('historical_identity_verified') is True)
    process=dict(observed.get('process_observation',{}))
    process.update(pid=observed.get('pid'),observed_create_time=observed.get('create_time'),
        alive=not owner_gone,identity_verified=observed.get('identity_verified') is True,
        execution_active=observed.get('execution_active') is True,
        status='VERIFIED_RECOVERY_OWNER_RUNNING' if active else 'VERIFIED_RECOVERY_TERMINAL_OWNER_GONE' if owner_gone else 'VERIFIED_RECOVERY_TERMINAL_OWNER_RUNNING')
    result['process_observation']=process
    result['recovery_evidence']={'authorization':record.get('recovery_authorization'),'owner':record.get('recovery_owner'),
        'completion':record.get('recovery_completion'),'phase':record.get('recovery_phase')}
    return result


class LedgerCounters:
    """Seek only newly committed LF-delimited rows; never retain generated text."""
    def __init__(self, root, saved=None):
        self.root = Path(root).resolve()
        self.files = {}
        for key, entry in (saved or {}).items():
            parts = key.split('/')
            numbers = ('offset', 'invalid_records', 'unknown_cost_records', 'partial_tail_bytes')
            if (not valid_ledger_key(parts) or not isinstance(entry, dict) or
                    any(isinstance(entry.get(n), bool) or not isinstance(entry.get(n), int) or entry[n] < 0 for n in numbers) or
                    'file_id' not in entry or (entry['file_id'] is not None and (not isinstance(entry['file_id'], list) or len(entry['file_id']) != 2 or not all(isinstance(x, int) for x in entry['file_id']))) or
                    not isinstance(entry.get('status_counts'), dict) or
                    any(k not in ('success', 'failure', 'unknown') or isinstance(v, bool) or not isinstance(v, int) or v < 0
                        for k, v in entry['status_counts'].items()) or
                    not isinstance(entry.get('issues'), list) or not all(isinstance(x, str) for x in entry['issues'])):
                raise ValueError('Invalid ledger cursor cache')
            history=entry.get('recent_success_wall_time_s',[])
            if not isinstance(history,list) or len(history)>RECENT_WALL_SAMPLES or any(not finite_number(x) or x<0 for x in history):raise ValueError('Invalid recent wall cursor')
            self.files[key] = dict(entry)
            self.files[key].setdefault('recent_success_wall_time_s',[])
            self.files[key].setdefault('latest_success_metadata',None)
            self.files[key].setdefault('wall_history_initialized',False)
        self.bytes_read_this_poll = 0

    @staticmethod
    def fresh(identity):
        return {'file_id': identity, 'offset': 0, 'status_counts': {},
                'invalid_records': 0, 'unknown_cost_records': 0,
                'partial_tail_bytes': 0, 'issues': [], 'present': True,
                'recent_success_wall_time_s': [], 'latest_success_metadata': None, 'wall_history_initialized': True}

    @staticmethod
    def observe_success_wall(entry,row):
        result=row.get('result',{})
        if not isinstance(result,dict):return
        wall=result.get('wall_time_s')
        if not finite_number(wall) or wall<0:return
        entry['recent_success_wall_time_s']=(entry.get('recent_success_wall_time_s',[])+[float(wall)])[-RECENT_WALL_SAMPLES:]
        metadata={'wall_time_s':float(wall)}
        for key in ('model_id','model_revision'):
            if isinstance(result.get(key),str):metadata[key]=result[key][:240]
        for key in ('model_alias','model_fingerprint','split'):
            if isinstance(row.get(key),str):metadata[key]=row[key][:240]
        if finite_number(row.get('started_at_unix')):metadata['started_at_unix']=row['started_at_unix']
        if row.get('question_id') is not None:
            metadata['question_anonymous']='sha256:'+hashlib.sha256(('monitor-question-v1:'+str(row['question_id'])).encode()).hexdigest()
        entry['latest_success_metadata']=metadata

    def bootstrap_recent_wall(self,path,entry):
        # Old cursor versions had counts but no timing window. Read at most one
        # bounded committed tail on upgrade, then resume normal incremental IO.
        end=entry['offset'];start=max(0,end-WALL_BOOTSTRAP_BYTES)
        with path.open('rb') as stream:
            stream.seek(start);data=stream.read(end-start)
        self.bytes_read_this_poll+=len(data)
        if start:data=data.partition(b'\n')[2]
        for line in data.split(b'\n'):
            if not line.strip():continue
            try:
                row=json.loads(line.decode('utf-8-sig'),parse_constant=reject_constant)
                checksum=row.pop('record_sha256',None)
                if checksum==digest(row) and row.get('status')=='success':self.observe_success_wall(entry,row)
            except (UnicodeError,ValueError,TypeError,AttributeError):pass
        entry['wall_history_initialized']=True

    def scan(self):
        self.bytes_read_this_poll = 0
        present = set()
        for stage in STAGES:
            for path in sorted((self.root / 'results/raw' / stage).glob(stage_glob(stage,'.jsonl'))):
                key = path.relative_to(self.root).as_posix()
                present.add(key)
                if not path.resolve().is_relative_to(self.root):
                    entry = self.files.setdefault(key, self.fresh(None))
                    entry['issues'] = sorted(set(entry['issues']) | {'LEDGER_PATH_OUTSIDE_ROOT'})
                    continue
                stat = path.stat()
                identity = [stat.st_dev, stat.st_ino]
                previous = self.files.get(key)
                reason = None
                if previous is not None:
                    if previous['file_id'] != identity:
                        reason = 'LEDGER_REPLACED'
                    elif stat.st_size < previous['offset']:
                        reason = 'LEDGER_TRUNCATED'
                if previous is None or reason:
                    entry = self.fresh(identity)
                    if previous:
                        entry['issues'] = sorted(set(previous.get('issues', [])) | {reason})
                    self.files[key] = entry
                else:
                    entry = previous
                entry['present'] = True
                if entry.get('wall_history_initialized') is not True:
                    self.bootstrap_recent_wall(path,entry)
                entry['partial_tail_bytes'] = 0
                with path.open('rb') as stream:
                    stream.seek(entry['offset'])
                    while True:
                        start = stream.tell()
                        line = stream.readline(MAX_LINE_BYTES + 1)
                        self.bytes_read_this_poll += len(line)
                        if not line:
                            break
                        if len(line) > MAX_LINE_BYTES:
                            entry['partial_tail_bytes'] = max(0, stat.st_size - start)
                            entry['issues'] = sorted(set(entry['issues']) | {'OVERSIZED_LEDGER_ROW'})
                            break
                        if not line.endswith(b'\n'):
                            entry['partial_tail_bytes'] = len(line)
                            break
                        entry['offset'] = stream.tell()
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line.decode('utf-8-sig'), parse_constant=reject_constant)
                            if not isinstance(row, dict):
                                raise ValueError('Ledger row is not an object')
                            checksum = row.pop('record_sha256', None)
                            if checksum != digest(row):
                                raise ValueError('Ledger checksum mismatch')
                            status = row.get('status', 'unknown')
                            if status not in ('success', 'failure'):
                                status = 'unknown'
                                entry['issues'] = sorted(set(entry['issues']) | {'UNKNOWN_RECORD_STATUS'})
                            entry['status_counts'][status] = entry['status_counts'].get(status, 0) + 1
                            if status=='success':self.observe_success_wall(entry,row)
                            if row.get('cost_status') == 'unknown_must_reconcile':
                                entry['unknown_cost_records'] += 1
                        except (UnicodeError, ValueError, TypeError):
                            entry['invalid_records'] += 1
                            entry['issues'] = sorted(set(entry['issues']) | {'INVALID_LEDGER_RECORD'})
                entry['observed_size'] = stat.st_size
        for key, entry in self.files.items():
            if key not in present:
                entry['present'] = False
                entry['issues'] = sorted(set(entry['issues']) | {'LEDGER_REMOVED'})
        summary = {}
        issues = []
        for stage in STAGES:
            counts = Counter()
            invalid = unknown_cost = partial = files = 0
            for key, entry in sorted(self.files.items()):
                if key.split('/')[2] != stage:
                    continue
                for code in entry.get('issues', []):
                    issues.append({'code': code, 'ledger': key})
                if not entry.get('present', False):
                    continue
                files += 1
                counts.update(entry['status_counts'])
                invalid += entry['invalid_records']
                unknown_cost += entry['unknown_cost_records']
                partial += int(entry['partial_tail_bytes'] > 0)
                if entry['unknown_cost_records']:
                    issues.append({'code': 'UNKNOWN_FAILED_CALL_COST', 'ledger': key})
            summary[stage] = {'status_counts': dict(sorted(counts.items())), 'files': files,
                              'invalid_records': invalid, 'unknown_cost_records': unknown_cost,
                              'incomplete_tail_files': partial}
        return summary, issues



def finite_number(value):
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)


def recent_wall_distribution(entry):
    samples=entry.get('recent_success_wall_time_s',[]) if entry else []
    if not samples:return {'samples':0,'min_wall_time_s':None,'p50_wall_time_s':None,'max_wall_time_s':None,'p95_wall_time_s':None,'review_after_s':600.,'scope':'same output ledger; successful committed actual calls only'}
    ordered=sorted(samples);position=(len(ordered)-1)*.95;lower=int(position);upper=min(lower+1,len(ordered)-1)
    p95=ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)
    middle=(len(ordered)-1)/2;lo=int(middle);hi=min(lo+1,len(ordered)-1)
    median=ordered[lo]+(ordered[hi]-ordered[lo])*(middle-lo)
    return {'samples':len(samples),'min_wall_time_s':float(ordered[0]),'p50_wall_time_s':float(median),'max_wall_time_s':float(ordered[-1]),'p95_wall_time_s':float(p95),'review_after_s':max(600.,5.*p95),
            'scope':'last at most 128 successful actual call durations in this same output ledger; not a failure deadline'}


def option_values(command,name):
    values=[]
    for i,arg in enumerate(command):
        if arg==name and i+1<len(command):values.append(command[i+1])
        elif isinstance(arg,str) and arg.startswith(name+'='):values.append(arg[len(name)+1:])
    return values


def command_path(value,cwd):
    return (Path(value) if Path(value).is_absolute() else Path(cwd)/value).resolve()


def compact_run(run):
    if run is None:return None
    result={key:run.get(key) for key in ('overall','recorded_status','effective_status','step','active_model',
        'active_script','active_child_pid','supervisor_pid','supervisor_running','recorded_supervisor_running',
        'updated_at_unix','gpu_min_free_mib','E0','E1','E2','E3','E4','E5','master_pid','master_script','process_create_time',
        'master_source_sha256','master_running','workflow_execution_complete','scientific_completion',
        'supervisor_script','supervisor_create_time','recovery_authorization','recovery_owner','recovery_phase','recovery_completion',
        'recovery_evidence','recovery_owner_observation_verified','recovery_completion_verified','original_worker_exit_verified','historical_identity_verified',
        'final_acceptance','terminal_acceptance_verified','independent_acceptance')}
    result['process_observation']=run.get('process_observation',{})
    return result


def sha256_file(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


_VALIDATION_WRITE_DIRECTORY = None
_VALIDATION_AUDIT_INSTALLED = False


def validation_write_audit(event,args):
    """Guard audited Python writes on this verifier path; not an OS sandbox."""
    allowed=_VALIDATION_WRITE_DIRECTORY
    if allowed is None:return
    targets=()
    if event=='open':
        path,mode,flags=args
        writing=(isinstance(mode,str) and any(value in mode for value in ('w','a','x','+')))
        writing=writing or isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing:targets=(path,)
    elif event in ('os.mkdir','os.remove','os.rmdir','os.truncate','shutil.rmtree','os.chmod','os.chown','os.utime'):
        targets=(args[0],)
    elif event in ('os.rename','os.link','os.symlink'):
        targets=args[:2]
    for target in targets:
        if not isinstance(target,(str,bytes,os.PathLike)):
            raise PermissionError('Independent acceptance may not mutate an unbound file descriptor')
        # dill/torch probe file classes through the OS null device; no file/cache is persisted.
        if event=='open' and os.fsdecode(target).casefold()==os.devnull.casefold():continue
        path=Path(os.fsdecode(target)).resolve()
        if not path.is_relative_to(allowed):
            raise PermissionError('Independent acceptance may write only observer-owned scratch, never scientific evidence or external cache')


@contextmanager
def validation_readonly(root):
    global _VALIDATION_WRITE_DIRECTORY,_VALIDATION_AUDIT_INSTALLED
    if _VALIDATION_WRITE_DIRECTORY is not None:raise RuntimeError('Nested terminal validation is not permitted')
    root=Path(root).resolve();directory=root/'artifacts/monitor';scratch=directory/'validation_scratch'
    if directory.resolve()!=directory or scratch.resolve()!=scratch:
        raise ValueError('Observer validation directory/scratch may not redirect through a symlink or junction')
    scratch.mkdir(parents=True,exist_ok=True)
    if directory.resolve()!=directory or scratch.resolve()!=scratch:
        raise ValueError('Observer validation directory/scratch changed its resolved ownership')
    if not _VALIDATION_AUDIT_INSTALLED:
        sys.addaudithook(validation_write_audit);_VALIDATION_AUDIT_INSTALLED=True
    bytecode=sys.dont_write_bytecode;previous_temp=tempfile.tempdir
    environment={key:os.environ.get(key) for key in ('TMP','TEMP')}
    try:
        sys.dont_write_bytecode=True;tempfile.tempdir=str(scratch)
        os.environ.update(TMP=str(scratch),TEMP=str(scratch));_VALIDATION_WRITE_DIRECTORY=scratch
        yield
    finally:
        _VALIDATION_WRITE_DIRECTORY=None
        sys.dont_write_bytecode=bytecode;tempfile.tempdir=previous_temp
        for key,value in environment.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def validate_final_acceptance_readonly(root):
    """Existing scientific verifier only; never call build_delivery or its CLI.

    FULL acceptance may replay CPU Flex fits, all frozen evaluations/bootstrap
    and raw-store accounting. It performs no LLM/GPU generation. Complete
    scientific inputs are read and replayed without committing evidence. The caller enforces a local
    write guard for missing-store fallbacks; CPU temporary files are confined to
    observer scratch. No result is cached: another poll/restart revalidates.
    """
    spec=importlib.util.spec_from_file_location('monitor_acceptance_'+uuid.uuid4().hex,
        Path(__file__).with_name('build_delivery.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.validate_acceptance(Path(root))


def verify_terminal_acceptance(root,record):
    root=Path(root).resolve();outcome=record.get('overall');complete=TERMINAL_RUN_STATUSES.get(outcome)
    if (outcome not in TERMINAL_RUN_STATUSES or record.get('schema')!='full_experiment_supervisor_status_v1'
        or record.get('step')!='EXPERIMENTS_VERIFIED_AWAITING_GITHUB_SYNC' or record.get('master_running') is not False
        or record.get('workflow_execution_complete') is not True or record.get('scientific_completion') is not complete
        or record.get('active_child_pid') is not None or record.get('active_script') is not None):
        raise ValueError('Only a normal child-free master terminal declaration may request independent acceptance')
    expected=root/'FINAL_ACCEPTANCE.json';reference=record.get('final_acceptance')
    if (not isinstance(reference,dict) or not isinstance(reference.get('path'),str)
        or not Path(reference['path']).is_absolute() or Path(reference['path']).resolve()!=expected
        or isinstance(reference.get('bytes'),bool) or not isinstance(reference.get('bytes'),int)
        or reference['bytes']<0):raise ValueError('Final acceptance must bind the exact local acceptance artifact')
    master_sha=sha256_file(root/'scripts/run_full_experiment.py')
    if record.get('master_source_sha256')!=master_sha:raise ValueError('Terminal master source changed')
    acceptance_sha=sha256_file(expected)
    if reference.get('sha256')!=acceptance_sha or reference['bytes']!=expected.stat().st_size:
        raise ValueError('Terminal acceptance artifact changed')
    acceptance=read_json(expected)
    if (not isinstance(acceptance,dict) or acceptance.get('status')!=outcome or acceptance.get('research_complete') is not complete
        or acceptance.get('master_source_sha256')!=master_sha):raise ValueError('Terminal and acceptance declarations disagree')
    cpu=time.process_time();wall=time.perf_counter()
    with validation_readonly(root):
        context=validate_final_acceptance_readonly(root)
    elapsed={'process_time_s':time.process_time()-cpu,'wall_time_s':time.perf_counter()-wall}
    if (not isinstance(context,dict) or context.get('acceptance')!=acceptance
        or context.get('acceptance_ref')!={'path':str(expected),'sha256':acceptance_sha,'bytes':reference['bytes']}):
        raise ValueError('Independent acceptance verifier did not return this bound acceptance')
    if (sha256_file(expected)!=acceptance_sha or sha256_file(root/'scripts/run_full_experiment.py')!=master_sha
        or read_json(root/'FULL_RUN_STATUS.json')!=record):raise ValueError('Terminal evidence changed during independent verification')
    return {'status':'INDEPENDENT_FINAL_ACCEPTANCE_VERIFIED','outcome':outcome,'research_complete':complete,
        'workflow_execution_complete':True,'acceptance_sha256':acceptance_sha,'master_source_sha256':master_sha,
        'verified_at_unix':time.time(),'validator':'scripts/build_delivery.py:validate_acceptance',
        'verification_cost':elapsed,'verification_device':'CPU_ONLY_NO_LM_GENERATION','result_cached':False,
        'scientific_writes_permitted':False,'observer_scratch_writes_permitted':True,'external_cache_writes_permitted':False,
        'write_guard_scope':'Python audited filesystem operations on the reviewed verifier path; not an OS sandbox',
        'scope':'Independent complete acceptance replay; FULL retains original scientific verdicts. A legitimate E1 stop verifies the prescribed stopping workflow and does not claim E2-E5/final research completion.'}


def master_pid_absent(pid,factory):
    try:
        (factory or psutil.Process)(pid)
    except psutil.NoSuchProcess:
        return True
    except (psutil.AccessDenied,PermissionError):
        return False
    return False


def observe_full_run(root,support,factory):
    path=Path(root)/'FULL_RUN_STATUS.json'
    record=read_json(path)
    if record is None:return None,[]
    expected=(Path(root)/'scripts/run_full_experiment.py').resolve()
    if (not isinstance(record,dict) or isinstance(record.get('master_pid'),bool) or not isinstance(record.get('master_pid'),int)
        or record['master_pid']<=0 or not finite_number(record.get('process_create_time')) or record['process_create_time']<=0
        or not isinstance(record.get('master_script'),str) or not Path(record['master_script']).is_absolute()
        or Path(record['master_script']).resolve()!=expected or not finite_number(record.get('updated_at_unix'))):
        return {'effective_status':'UNVERIFIED_MASTER_IDENTITY','process_observation':{'identity_verified':False}},[{'code':'FULL_RUN_STATUS_IDENTITY_METADATA_INVALID'}]
    adapted=dict(record,supervisor_pid=record['master_pid'],supervisor_running=record.get('overall') in ACTIVE_RUN_STATUSES,
        terminal_acceptance_verified=False,independent_acceptance=None)
    view=support.effective_record(adapted,pid_field='supervisor_pid',status_field='overall',script='run_full_experiment.py',
        active_statuses=ACTIVE_RUN_STATUSES,recorded_start=record['process_create_time'],process_factory=factory)
    observed=view.get('process_observation',{})
    if observed.get('identity_verified') and (not finite_number(observed.get('observed_create_time')) or abs(observed['observed_create_time']-record['process_create_time'])>=.01):
        observed.update(identity_verified=False,status='MASTER_PROCESS_CREATE_TIME_MISMATCH');view['supervisor_running']=False
        view.update(overall='UNVERIFIED_MASTER_IDENTITY',effective_status='UNVERIFIED_MASTER_IDENTITY')
    if observed.get('identity_verified'):
        try:
            process=(factory or psutil.Process)(record['master_pid']);cwd=process.cwd()
            argv_scripts=[command_path(x,cwd) for x in process.cmdline() if isinstance(x,str) and x.lower().endswith('.py')]
            if not argv_scripts or argv_scripts[0]!=expected:raise ValueError('Master argv script mismatch')
        except (OSError,ValueError,TypeError,psutil.Error):
            observed.update(identity_verified=False,status='MASTER_ARGV_SCRIPT_UNVERIFIED');view['supervisor_running']=False
            view.update(overall='UNVERIFIED_MASTER_IDENTITY',effective_status='UNVERIFIED_MASTER_IDENTITY')
    if (record.get('overall') in TERMINAL_RUN_STATUSES and observed.get('status')=='PROCESS_NOT_FOUND'
        and observed.get('alive') is False):
        try:
            if not master_pid_absent(record['master_pid'],factory):raise ValueError('Terminal owner PID is still present or cannot be inspected')
            proof=verify_terminal_acceptance(root,record)
            if not master_pid_absent(record['master_pid'],factory):raise ValueError('Terminal owner PID became present or unverifiable during validation')
            view.update(terminal_acceptance_verified=True,independent_acceptance=proof)
            observed.update(status='EXPECTED_TERMINAL_EXIT',identity_verified=False,execution_active=False)
            return view,[]
        except Exception as error:
            # Never retain arbitrary verifier messages that may contain private paths/text.
            view.update(terminal_acceptance_verified=False,effective_status='TERMINAL_ACCEPTANCE_UNVERIFIED')
            return view,[{'code':'FULL_RUN_TERMINAL_ACCEPTANCE_REQUIRES_AUDIT','error_type':type(error).__name__}]
    if not observed.get('identity_verified'):
        return view,[{'code':'FULL_RUN_MASTER_IDENTITY_OR_LIVENESS_UNVERIFIED'}]
    return view,[]


def model_config_metadata(command,cwd,root):
    output=[]
    for value in option_values(command,'--model-config'):
        path=command_path(value,cwd)
        if not path.is_relative_to((Path(root)/'configs').resolve()) or path.suffix.lower()!='.json' or path.stat().st_size>128*1024:
            raise ValueError('Model config outside the narrow metadata scope')
        raw=path.read_bytes();record=json.loads(raw.decode('utf-8-sig'),parse_constant=reject_constant)
        if not isinstance(record,dict):raise ValueError('Invalid model config metadata')
        item={'config':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()}
        for key in ('model_id','revision','backend','max_input_tokens','max_new_tokens'):
            value=record.get(key)
            if isinstance(value,str):item[key]=value[:240]
            elif finite_number(value):item[key]=value
        output.append(item)
    return {'identity_status':'SINGLE_CONFIG_PATH_IN_VERIFIED_WORKER_ARGV' if len(output)==1 else 'MULTIPLE_OR_MISSING_CONFIGS_CURRENT_MODEL_UNDETERMINED',
            'configurations':output,
            'scope':'Observed argv path and current config metadata only; loaded runtime weights are not inspected. Last successful model identity is a separate observation.'}


def final_manifest_metadata(command,cwd,root,expected_output):
    # Only inspect the label-free manifest and config, never jobs/questions/gold.
    values=option_values(command,'--jobs-manifest')
    if len(values)!=1:raise ValueError('Final worker needs exactly one manifest')
    path=command_path(values[0],cwd)
    if not path.is_relative_to((root/'data/jobs/final').resolve()) or path.suffix!='.json' or path.stat().st_size>256*1024:
        raise ValueError('Final manifest outside narrow metadata scope')
    raw=path.read_bytes();manifest=json.loads(raw.decode('utf-8-sig'),parse_constant=reject_constant)
    if (not isinstance(manifest,dict) or manifest.get('schema')!='released_final_jobs_v1'
            or manifest.get('split')!='sealed_test' or manifest.get('gold_excluded_from_model_inputs') is not True
            or manifest.get('research_complete') is not False):raise ValueError('Unexpected final metadata schema')
    if command_path(manifest.get('output',''),cwd)!=expected_output:raise ValueError('Final worker manifest output does not match lock')
    reference=manifest.get('model_config',{})
    if not isinstance(reference,dict) or not isinstance(reference.get('path'),str):raise ValueError('Final config reference missing')
    model=model_config_metadata(['--model-config',reference['path']],cwd,root)
    if len(model['configurations'])!=1 or model['configurations'][0]['sha256']!=reference.get('sha256'):
        raise ValueError('Final config reference changed')
    model['identity_status']='SINGLE_CONFIG_REFERENCED_BY_VERIFIED_WORKER_MANIFEST'
    model['scope']='Observed label-free manifest/config binding only; model weights, jobs, release acceptance and scientific completion are not inspected.'
    model['manifest']={'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()}
    return model


def observe_inflight(root,counters,support,factory,now):
    root=Path(root).resolve();records=[];issues=[];factory=factory or psutil.Process
    for stage in STAGES:
        for path in sorted((root/'results/raw'/stage).glob(stage_glob(stage,'.jsonl.lock'))):
            key=path.relative_to(root).as_posix();source=key[:-5];entry=counters.files.get(source,{})
            item={'source':source,'stage':stage,'paid_call_lock':{'path':key},'active_inflight_age_s':None,
                  'current_question_anonymous':None,'current_question_status':'NOT_RECORDED_IN_LOCK_NO_GUESS',
                  'recent_success_wall':recent_wall_distribution(entry),'last_success_metadata':entry.get('latest_success_metadata'),
                  'owner_identity_verified':False,'long_call_review_required':False,
                  'interpretation':'A long held paid-call lock requires review; liveness and GPU utilization do not prove progress. No failure, deadlock, retry or restart is inferred.'}
            try:
                if not path.resolve().is_relative_to(root):raise ValueError('Lock escaped root')
                before=read_bytes_shared(path);lock=json.loads(before.decode('utf-8-sig'),parse_constant=reject_constant)
                pid=lock.get('pid');created=lock.get('created_at_unix')
                if isinstance(pid,bool) or not isinstance(pid,int) or pid<=0 or not finite_number(created) or created<=0 or created>now+1:
                    raise ValueError('Invalid inflight metadata')
                item['paid_call_lock'].update(pid=pid,created_at_unix=created)
                process=factory(pid);birth=process.create_time();command=process.cmdline();cwd=process.cwd()
                item['paid_call_lock']['observed_process_create_time']=birth
                if not finite_number(birth) or birth>created+.01:raise ValueError('PID newer than inflight call')
                scripts=[command_path(x,cwd) for x in command if isinstance(x,str) and x.lower().endswith('.py')]
                supported={root/'scripts/run_members.py',root/'scripts/run_aggregation.py',root/'scripts/run_final_generation.py'}
                if not scripts or scripts[0] not in supported:raise ValueError('Unknown worker role')
                final_worker=scripts[0].name=='run_final_generation.py'
                model=None
                if final_worker:
                    if stage!='final':raise ValueError('Final worker outside final ledger scope')
                    model=final_manifest_metadata(command,cwd,root,(root/source).resolve())
                else:
                    outputs=option_values(command,'--output')
                    if len(outputs)!=1 or command_path(outputs[0],cwd)!=(root/source).resolve():raise ValueError('Worker output does not match lock')
                observation=support.observe_process(pid,scripts[0],recorded_update=created,process_factory=factory)
                item['process_observation']=observation
                if not observation.get('identity_verified'):raise ValueError('Worker identity unverified')
                try:after=read_bytes_shared(path)
                except FileNotFoundError:continue  # A committed call can finish during this read-only probe.
                if before!=after:raise ValueError('Inflight lock changed during probe')
                item['owner_identity_verified']=True;item['active_inflight_age_s']=max(0.,now-created)
                item['worker_script']=scripts[0].name
                try:item['model']=model if final_worker else model_config_metadata(command,cwd,root)
                except (OSError,ValueError,TypeError):
                    item['model']={'identity_status':'MODEL_CONFIG_METADATA_REQUIRES_AUDIT'}
                    issues.append({'code':'INFLIGHT_MODEL_METADATA_REQUIRES_AUDIT','source':source})
                if observation.get('execution_active') is False:
                    issues.append({'code':'INFLIGHT_WORKER_SUSPENDED_REQUIRES_REVIEW','source':source})
                if item['active_inflight_age_s']>item['recent_success_wall']['review_after_s']:
                    item['long_call_review_required']=True
                    issues.append({'code':'LONG_CALL_REQUIRES_REVIEW','source':source,'owner_pid':pid,
                                   'review_after_s':item['recent_success_wall']['review_after_s']})
            except FileNotFoundError:
                if not path.exists():continue
                item['audit_status']='INFLIGHT_LOCK_IDENTITY_OR_METADATA_UNVERIFIED'
                issues.append({'code':'INFLIGHT_LOCK_REQUIRES_IDENTITY_AUDIT','source':source})
            except (OSError,ValueError,TypeError,AttributeError,psutil.Error):
                item['audit_status']='INFLIGHT_LOCK_IDENTITY_OR_METADATA_UNVERIFIED'
                issues.append({'code':'INFLIGHT_LOCK_REQUIRES_IDENTITY_AUDIT','source':source})
            records.append(item)
    if sum(x['owner_identity_verified'] for x in records)>1:issues.append({'code':'MULTIPLE_VERIFIED_INFLIGHT_CALLS_REQUIRE_REVIEW'})
    return records,issues

class MonitorLockError(RuntimeError):
    pass


@contextmanager
def lease_guard(path):
    """Short native mutex serializes stale-lock recovery by concurrent starters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise MonitorLockError('Another monitor is acquiring its lease') from error
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class MonitorLease:
    def __init__(self, directory, root, process_factory=None):
        self.directory = Path(directory).resolve()
        self.root = Path(root).resolve()
        if not self.directory.is_relative_to(self.root):
            raise MonitorLockError('Monitor output must remain inside its experiment root')
        self.path = self.directory / 'monitor.lock'
        self.guard = self.directory / 'monitor_guard.lock'
        self.factory = process_factory or psutil.Process
        self.record = {'schema': SCHEMA, 'pid': os.getpid(),
                       'process_create_time': psutil.Process().create_time(),
                       'root': str(self.root), 'script': str(Path(__file__).resolve()),
                       'owner_token': uuid.uuid4().hex}

    def __enter__(self):
        with lease_guard(self.guard):
            if self.path.exists():
                old = read_json(self.path)
                if (not isinstance(old, dict) or old.get('schema') != SCHEMA or
                        old.get('root') != str(self.root) or old.get('script') != str(Path(__file__).resolve()) or
                        isinstance(old.get('pid'), bool) or not isinstance(old.get('pid'), int) or old['pid'] <= 0 or
                        isinstance(old.get('process_create_time'), bool) or not isinstance(old.get('process_create_time'), (int, float)) or
                        not math.isfinite(old['process_create_time']) or old['process_create_time'] <= 0):
                    raise MonitorLockError('Invalid existing monitor lease; operator review required')
                try:
                    owner = self.factory(old['pid'])
                    created = owner.create_time()
                except psutil.NoSuchProcess:
                    owner = None
                except (psutil.AccessDenied, PermissionError) as error:
                    raise MonitorLockError('Cannot verify existing monitor owner') from error
                if owner is not None and abs(created - old['process_create_time']) < 0.01:
                    raise MonitorLockError('An existing monitor owner is still alive')
                # A missing PID or a different birth time proves the old owner is gone.
                retained = self.directory / ('retired_monitor_lease_' + uuid.uuid4().hex + '.json')
                self.path.replace(retained)
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            try:
                os.write(descriptor, json.dumps(self.record, ensure_ascii=False).encode('utf-8'))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with lease_guard(self.guard):
            current = read_json(self.path)
            if current and current.get('owner_token') == self.record['owner_token']:
                self.path.unlink()


def gpu_probe():
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, timeout=8, check=True)
        free, utilization = map(int, result.stdout.strip().split(','))
        return {'free_mib': free, 'utilization_percent': utilization,
                'does_not_prove_model_progress': True}
    except Exception:
        return {'status': 'UNAVAILABLE'}


class ExperimentMonitor:
    def __init__(self, root=ROOT, process_factory=None, gpu_reader=None):
        self.root = Path(root).resolve()
        self.directory = self.root / 'artifacts/monitor'
        self.support = load_status_support(self.root)
        self.factory = process_factory
        self.gpu_reader = gpu_reader or gpu_probe
        self.cache_issue = False
        try:
            saved = read_json(self.directory / 'cursors.json') or {}
            if saved and (saved.get('schema') != SCHEMA or not isinstance(saved.get('files'), dict)):
                raise ValueError('Invalid cursor schema')
            self.counters = LedgerCounters(self.root, saved.get('files'))
            self.last_event = saved.get('last_event_id')
        except (OSError, ValueError, TypeError):
            self.cache_issue = True
            self.counters = LedgerCounters(self.root)
            self.last_event = None

    def poll(self):
        now = time.time()
        issues = []
        try:
            recorded = read_json(self.root / 'RUN_STATUS.json')
            if recovery_record(recorded):
                run=observe_recovery_run(self.root,recorded,self.factory)
            else:
                lock = read_json(self.root / 'logs/local_e1_supervisor.lock') or {}
                start = lock.get('started_at_unix') if recorded and lock.get('pid') == recorded.get('supervisor_pid') else None
                run = self.support.effective_record(recorded, pid_field='supervisor_pid', status_field='overall',
                    script='run_local_e1.py', active_statuses={'IN_PROGRESS', 'IN_PROGRESS_TECHNICAL_RECOVERY'},
                    recorded_start=start, process_factory=self.factory)
        except RecoveryObservationError:
            run=None
            issues.append({'code':'RECOVERY_OWNER_AUTHORIZATION_OR_STATE_UNVERIFIED'})
        except (OSError, ValueError, TypeError, KeyError):
            run = None
            issues.append({'code': 'INVALID_RUN_STATUS_OR_OWNER_LOCK'})
        if run is None:
            issues.append({'code': 'RUN_STATUS_UNAVAILABLE'})
            run = {}
        legacy_run=run
        full_run=None
        try:
            full_run,full_issues=observe_full_run(self.root,self.support,self.factory)
            issues.extend(full_issues)
            if full_run and (full_run.get('process_observation',{}).get('identity_verified') or full_run.get('terminal_acceptance_verified') is True):
                run=full_run
                old_observation=legacy_run.get('process_observation',{})
                old_expected=legacy_run.get('recorded_supervisor_running') is True or legacy_run.get('recorded_status') in ACTIVE_RUN_STATUSES
                if old_expected and not old_observation.get('identity_verified'):issues.append({'code':'LEGACY_E1_SUPERVISOR_IDENTITY_OR_LIVENESS_UNVERIFIED'})
                elif old_expected and old_observation.get('execution_active') is False:issues.append({'code':'LEGACY_E1_SUPERVISOR_SUSPENDED'})
        except (OSError,ValueError,TypeError,KeyError,psutil.Error):issues.append({'code':'FULL_RUN_STATUS_UNREADABLE_REQUIRES_AUDIT'})
        observation = run.get('process_observation', {})
        expected_active = run.get('recorded_supervisor_running') is True or run.get('recorded_status') in {'IN_PROGRESS', 'IN_PROGRESS_TECHNICAL_RECOVERY'}
        if expected_active and not observation.get('identity_verified'):
            issues.append({'code': 'SUPERVISOR_IDENTITY_OR_LIVENESS_UNVERIFIED'})
        if expected_active and observation.get('execution_active') is False:
            issues.append({'code': 'SUPERVISOR_SUSPENDED'})
        updated = run.get('updated_at_unix')
        if expected_active and isinstance(updated, (int, float)) and now - updated > 180:
            issues.append({'code': 'SUPERVISOR_HEARTBEAT_STALE'})
        child = None
        script = run.get('active_script')
        if isinstance(script, str) and script.endswith('.py') and Path(script).name == script and run.get('active_child_pid'):
            child = self.support.observe_process(run['active_child_pid'], self.root / 'scripts' / script,
                recorded_update=run.get('updated_at_unix'), process_factory=self.factory)
            if not child.get('identity_verified'):
                issues.append({'code': 'ACTIVE_CHILD_IDENTITY_OR_LIVENESS_UNVERIFIED'})
            elif child.get('execution_active') is False:
                issues.append({'code': 'ACTIVE_CHILD_SUSPENDED'})
        counts, ledger_issues = self.counters.scan()
        issues.extend(ledger_issues)
        inflight,inflight_issues=observe_inflight(self.root,self.counters,self.support,self.factory,now)
        issues.extend(inflight_issues)
        if any(value['incomplete_tail_files'] for value in counts.values()) and not observation.get('execution_active'):
            issues.append({'code': 'PARTIAL_TAIL_WITHOUT_ACTIVE_SUPERVISOR'})
        if self.cache_issue:
            issues.append({'code': 'MONITOR_CURSOR_CACHE_INVALID_RECOUNTED'})
        e1 = None
        try:
            receipt = read_json(self.root / 'results/e1/e1_diagnostics.json')
            if receipt is not None:
                e1 = {key: receipt.get(key) for key in ('status', 'n_total_e1', 'n_combinations')}
        except (OSError, ValueError, TypeError, AttributeError):
            issues.append({'code': 'E1_RECEIPT_UNREADABLE'})
        compact=compact_run(run)
        terminal=run.get('independent_acceptance') if run is full_run and run.get('terminal_acceptance_verified') is True else None
        snapshot = {'schema': SCHEMA, 'heartbeat_at_unix': now,
                    'monitor': {'pid': os.getpid(), 'process_create_time': psutil.Process().create_time()},
                    'status': 'ATTENTION_REQUIRED' if issues else 'VERIFIED_TERMINAL' if terminal else 'MONITORING',
                    'run': compact, 'active_child_observation': child,
                    'selected_run_source':'FULL_RUN_STATUS.json' if run is full_run else 'RUN_STATUS.json',
                    'full_run':compact_run(full_run),'legacy_e1_run':compact_run(legacy_run),
                    'active_inflight':inflight,
                    'active_inflight_max_age_s':max((x['active_inflight_age_s'] for x in inflight if x['active_inflight_age_s'] is not None),default=None),
                    'actual_model_calls': counts, 'ledger_bytes_read_this_poll': self.counters.bytes_read_this_poll,
                    'issues': issues, 'e1_result': e1,
                    'terminal_acceptance_verified': terminal is not None,'independent_acceptance':terminal,
                    'scientific_completion': False,
                    'reported_scientific_completion': run.get('scientific_completion'),
                    'completion_scope': 'Monitor observations never replace independent final research acceptance.',
                    'permissions': {'model_execution': False, 'inference_retry': False, 'github_write': False},
                    'gpu_observation': self.gpu_reader()}
        try:
            memory = psutil.virtual_memory()
            snapshot['host_observation'] = {'available_bytes': memory.available,
                'total_bytes': memory.total, 'disk_free_bytes': psutil.disk_usage(str(self.root)).free}
        except (OSError, psutil.Error):
            snapshot['host_observation'] = {'status': 'UNAVAILABLE'}
        event_view = {'run': {key: compact.get(key) for key in compact if key != 'updated_at_unix'},
                      'active_child': child, 'calls': counts, 'issues': issues, 'e1_result': e1,
                      'inflight':[{k:v for k,v in item.items() if k!='active_inflight_age_s'} for item in inflight],
                      'selected_run_source':snapshot['selected_run_source'],
                      'full_run':None if full_run is None else {k:v for k,v in compact_run(full_run).items() if k!='updated_at_unix'},
                      'legacy_e1_run':{k:v for k,v in compact_run(legacy_run).items() if k!='updated_at_unix'}}
        event_id = digest(event_view)
        self.directory.mkdir(parents=True, exist_ok=True)
        if event_id != self.last_event:
            with (self.directory / 'events.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps({'event_id': event_id, 'observed_at_unix': now, **event_view},
                                        ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            self.last_event = event_id
        atomic_json(self.directory / 'cursors.json', {'schema': SCHEMA, 'files': self.counters.files,
                                                     'last_event_id': self.last_event})
        atomic_json(self.directory / 'current_status.json', snapshot)
        return snapshot


    def record_observation_error(self, error_type):
        """Keep the heartbeat fresh even when a source observation fails."""
        try:
            previous = read_json(self.directory / 'current_status.json') or {}
        except (OSError, ValueError, TypeError):
            previous = {}
        now = time.time()
        snapshot = {key: previous[key] for key in ('run','actual_model_calls','e1_result','full_run','legacy_e1_run','selected_run_source','active_inflight','active_inflight_max_age_s') if key in previous}
        snapshot.update(schema=SCHEMA, heartbeat_at_unix=now,
            monitor={'pid': os.getpid(), 'process_create_time': psutil.Process().create_time()},
            status='MONITOR_OBSERVATION_ERROR',
            issues=[{'code': 'MONITOR_OBSERVATION_FAILED', 'error_type': error_type}],
            counts_are_last_successful_observation=True,
            last_successful_heartbeat_at_unix=previous.get('last_successful_heartbeat_at_unix', previous.get('heartbeat_at_unix')),
            scientific_completion=False,
            permissions={'model_execution': False, 'inference_retry': False, 'github_write': False})
        fingerprint = digest({'monitor_error': error_type})
        self.directory.mkdir(parents=True, exist_ok=True)
        if fingerprint != self.last_event:
            with (self.directory / 'events.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps({'event_id': fingerprint, 'observed_at_unix': now,
                    'event': 'MONITOR_OBSERVATION_FAILED', 'error_type': error_type}) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            self.last_event = fingerprint
        atomic_json(self.directory / 'current_status.json', snapshot)
        return snapshot


def positive_interval(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 1:
        raise argparse.ArgumentTypeError('Monitoring interval must be finite and at least one second')
    return seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--interval-seconds', type=positive_interval, default=30)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    observer = ExperimentMonitor(root)
    with MonitorLease(observer.directory, root):
        print(json.dumps({'event': 'monitor_started', 'pid': os.getpid(), 'root': str(root),
                          'interval_seconds': args.interval_seconds, 'scientific_completion': False}), flush=True)
        while True:
            started = time.monotonic();status=None
            try:
                status = observer.poll()
                print(json.dumps({'event': 'monitor_heartbeat', 'at': status['heartbeat_at_unix'],
                                  'status': status['status'], 'issues': len(status['issues'])}), flush=True)
            except Exception as error:
                # Never echo ledger text, model output, arbitrary exception text or credentials.
                print(json.dumps({'event': 'monitor_observation_error', 'error_type': type(error).__name__,
                                  'at': time.time(), 'scientific_completion': False}), flush=True)
                try:
                    observer.record_observation_error(type(error).__name__)
                except Exception as write_error:
                    print(json.dumps({'event': 'monitor_state_write_error',
                                      'error_type': type(write_error).__name__}), flush=True)
                if args.once:
                    return 2
            if isinstance(status,dict) and status.get('terminal_acceptance_verified') is True and not status.get('issues'):
                print(json.dumps({'event':'monitor_finished_after_independent_acceptance',
                    'outcome':status['independent_acceptance']['outcome'],
                    'acceptance_sha256':status['independent_acceptance']['acceptance_sha256'],
                    'research_complete':status['independent_acceptance']['research_complete']}),flush=True)
                return 0
            if args.once:
                return 0
            time.sleep(max(0, args.interval_seconds - (time.monotonic() - started)))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
