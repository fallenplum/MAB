"""Audited scheduling recovery; never generates, scores, or changes frozen sources."""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
import hashlib
import json
import math
import os
import time
import uuid


class RecoveryAuthorizationError(RuntimeError):
    pass


FROZEN_SOURCES = (
    'src/budget_router/inference.py', 'src/budget_router/execution.py',
    'src/budget_router/member_execution.py', 'src/budget_router/scoring.py',
    'src/budget_router/diagnostics.py', 'src/budget_router/preparation.py',
    'src/budget_router/e1_controls.py', 'src/budget_router/protocol_lock.py',
    'scripts/review_runtime.py', 'scripts/freeze_e0.py', 'scripts/run_local_e1.py',
    'scripts/prepare_jobs.py', 'scripts/prepare_e1_controls.py', 'scripts/run_members.py',
    'scripts/run_aggregation.py', 'scripts/evaluate_real_e1.py', 'src/budget_router/runtime_memory.py',
)
RECOVERY_SOURCES = ('scripts/resume_e1_aggregation.py', 'src/budget_router/recovery_runtime.py')
TERMINAL_KEYS = ('overall', 'step', 'E0', 'E1', 'E2', 'e2_executed',
    'scientific_completion', 'supervisor_running', 'supervisor_finished_at_unix',
    'active_child_pid', 'active_script', 'last_child_script', 'last_child_exit',
    'matrix_preparation_receipt', 'matrix_preparation_receipt_sha256',
    'supervisor_pid', 'supervisor_create_time', 'supervisor_script',
    'recovery_owner_argv_sha256', 'recovery_authorization', 'recovery_owner',
    'recovery_settlement', 'recovery_phase')


def _authorization_guard(function):
    from functools import wraps
    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except RecoveryAuthorizationError:
            raise
        except Exception as error:
            raise RecoveryAuthorizationError('Invalid recovery evidence: ' + type(error).__name__) from error
    return guarded


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def argv_digest(argv):
    return hashlib.sha256(json.dumps(argv, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def read_json(path):
    def reject(value):
        raise RecoveryAuthorizationError('Non-finite JSON value')
    return json.loads(Path(path).read_text('utf-8-sig'), parse_constant=reject)


def atomic_json(path, value, *, replace=None, sleep=None, max_attempts=8):
    """Unique fsynced state temporary; bounded retries for Windows 5/32/33 only.

    These errors can also indicate persistent access problems. Exhaustion raises;
    this function neither retries model calls nor authorizes stage advancement.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 16:
        raise ValueError('A bounded state-write retry count is required')
    replace = os.replace if replace is None else replace
    sleep = time.sleep if sleep is None else sleep
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    try:
        with pending.open('x', encoding='utf8', newline='\n') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        for index in range(max_attempts):
            try:
                replace(pending, path)
                return
            except OSError as error:
                if getattr(error, 'winerror', None) not in {5, 32, 33} or index + 1 == max_attempts:
                    raise
                sleep(min(.05 * 2 ** index, 1.))
    finally:
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            # Do not mask the original replace error. Never remove the target.
            pass


def immutable_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise RecoveryAuthorizationError('Existing recovery evidence differs')
        return
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def artifact_reference(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha_file(path), 'bytes': path.stat().st_size}


def _inside(root, path):
    root = Path(root).resolve(); path = Path(path)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise RecoveryAuthorizationError('Recovery artifact escaped project root')
    return path


def verified_reference(root, reference, *, expected=None):
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256', 'bytes'}
        or not isinstance(reference.get('sha256'), str) or len(reference['sha256']) != 64
        or any(c not in '0123456789abcdef' for c in reference['sha256'])
        or isinstance(reference.get('bytes'), bool) or not isinstance(reference.get('bytes'), int)):
        raise RecoveryAuthorizationError('Recovery artifact reference needs exact path/SHA/bytes')
    path = _inside(root, reference['path'])
    if expected is not None and path != _inside(root, expected):
        raise RecoveryAuthorizationError('Recovery artifact uses a noncanonical path')
    if not path.is_file() or path.stat().st_size != reference['bytes'] or sha_file(path) != reference['sha256']:
        raise RecoveryAuthorizationError('Recovery artifact hash or size changed')
    return path


def _positive(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _process_factory(factory):
    if factory is None:
        import psutil
        return psutil.Process
    return factory


def _script(argv, cwd):
    scripts = [Path(value) for value in argv if isinstance(value, str) and value.lower().endswith('.py')]
    if len(scripts) != 1:
        raise RecoveryAuthorizationError('Exactly one actual Python script identity required')
    return (scripts[0] if scripts[0].is_absolute() else Path(cwd) / scripts[0]).resolve()


def _process_record(value):
    if (not isinstance(value, dict) or isinstance(value.get('pid'), bool)
        or not isinstance(value.get('pid'), int) or value['pid'] <= 0
        or not _positive(value.get('create_time')) or not isinstance(value.get('argv'), list)
        or not value['argv'] or any(not isinstance(v, str) for v in value['argv'])
        or value.get('argv_sha256') != argv_digest(value['argv'])):
        raise RecoveryAuthorizationError('Exact process PID/birth/compact argv binding required')
    return value


def _flag(argv, flag):
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise RecoveryAuthorizationError('Original batch argv flag binding missing or ambiguous')
    return argv[argv.index(flag) + 1]


@_authorization_guard
def validate_recovery_authorization(root, reference_or_path):
    root = Path(root).resolve()
    reference = reference_or_path if isinstance(reference_or_path, dict) else artifact_reference(_inside(root, reference_or_path))
    path = verified_reference(root, reference)
    auth = read_json(path)
    if (auth.get('schema') != 'e1_aggregation_recovery_authorization_v1'
        or auth.get('scope') != 'scheduling_recovery_only_no_math_change'
        or auth.get('operational_overlay') != 'run_local_e1.atomic_json_only'
        or auth.get('software_test_only') is not False
        or auth.get('continue_online_after_e1_pass') is not True
        or auth.get('gpu_min_free_mib') != 6144):
        raise RecoveryAuthorizationError('Explicit original-scope recovery authorization required')
    lock_path = verified_reference(root, auth['original_e0'], expected='protocol/E0_LOCK.json')
    lock = read_json(lock_path)
    if (lock.get('status'), lock.get('stage'), lock.get('N'), lock.get('K'), lock.get('e1_questions')) != ('PASS', 'E0', 6, 3, 300):
        raise RecoveryAuthorizationError('Original full-scale E0 PASS required')
    frozen = lock.get('source_code', [])
    if (len(frozen) != len(FROZEN_SOURCES) or {v.get('path') for v in frozen} != set(FROZEN_SOURCES)
        or auth.get('e0_sources_sha256') != canonical_digest(frozen)):
        raise RecoveryAuthorizationError('Original seventeen-source E0 closure changed')
    for item in frozen:
        if sha_file(_inside(root, item['path'])) != item.get('sha256'):
            raise RecoveryAuthorizationError('An original frozen E0 source changed')
    for field, relative in (('experiment_config_sha256', 'configs/experiment_v1.json'),
                            ('pool_config_sha256', 'configs/pool_v1.json'),
                            ('split_manifest_sha256', 'data/derived/split_manifest.json')):
        if sha_file(root / relative) != lock.get(field):
            raise RecoveryAuthorizationError('Original E0 configuration or split changed')
    for field, count in (('qualification_receipts', 7), ('member_outputs', 6)):
        items = lock.get(field, [])
        if len(items) != count:
            raise RecoveryAuthorizationError('Original E0 real assets inventory incomplete')
        for item in items:
            if sha_file(_inside(root, item['path'])) != item.get('sha256'):
                raise RecoveryAuthorizationError('Original E0 real asset changed')
    refs = auth.get('recovery_sources', [])
    if len(refs) != 2 or {_inside(root, r.get('path', '')) for r in refs} != {root / s for s in RECOVERY_SOURCES}:
        raise RecoveryAuthorizationError('Exact two-source recovery closure required')
    for ref in refs:
        verified_reference(root, ref)
    if sha_file(root / RECOVERY_SOURCES[1]) != sha_file(__file__):
        raise RecoveryAuthorizationError('Loaded recovery validator differs from authorized source')
    batch = auth['active_batch']
    if batch.get('expected_jobs') != 6000:
        raise RecoveryAuthorizationError('All six-thousand original aggregation jobs required')
    for key, relative in (('jobs', 'data/jobs/e1/aggregations.jsonl'),
                          ('model_config', 'configs/aggregator_v1.json'),
                          ('qualification', 'artifacts/qualification/aggregator.json')):
        verified_reference(root, batch[key], expected=relative)
    if _inside(root, batch['output']) != root / 'results/raw/e1/aggregations.jsonl':
        raise RecoveryAuthorizationError('Original aggregation output path changed')
    qitem = next((v for v in lock['qualification_receipts'] if v['path'] == 'artifacts/qualification/aggregator.json'), None)
    if qitem is None or qitem['sha256'] != batch['qualification']['sha256']:
        raise RecoveryAuthorizationError('Aggregator qualification is not the original E0 receipt')
    original = [_process_record(auth[key]) for key in ('original_wrapper', 'original_worker')]
    if original[0]['pid'] == original[1]['pid']:
        raise RecoveryAuthorizationError('Wrapper and actual worker must have distinct identities')
    for record in original:
        if _script(record['argv'], root) != root / 'scripts/run_aggregation.py':
            raise RecoveryAuthorizationError('Original worker script changed')
        for flag, key in (('--jobs', 'jobs'), ('--model-config', 'model_config'), ('--qualification-receipt', 'qualification')):
            if _inside(root, _flag(record['argv'], flag)) != _inside(root, batch[key]['path']):
                raise RecoveryAuthorizationError('Original worker argv asset changed')
        if _inside(root, _flag(record['argv'], '--output')) != _inside(root, batch['output']):
            raise RecoveryAuthorizationError('Original worker output argv changed')
        if '--qualification' in record['argv'] or '--max-jobs' in record['argv']:
            raise RecoveryAuthorizationError('A truncated or qualification batch cannot be adopted')
    old = auth['original_supervisor']
    if isinstance(old.get('pid'), bool) or not isinstance(old.get('pid'), int) or old['pid'] <= 0 or not _positive(old.get('create_time')):
        raise RecoveryAuthorizationError('Original supervisor exact PID/birth required')
    helper = auth['helper']
    if (_inside(root, helper.get('script', '')) != root / RECOVERY_SOURCES[0]
        or not isinstance(helper.get('arguments'), list) or any(not isinstance(v, str) for v in helper['arguments'])
        or not isinstance(helper.get('allowed_executables'), list) or not helper['allowed_executables']):
        raise RecoveryAuthorizationError('Actual recovery entrypoint and argv authorization required')
    arguments = helper['arguments']
    if (_inside(root, _flag(arguments, '--authorization')) != path
        or arguments.count('--continue-online-after-e1-pass') != 1
        or _flag(arguments, '--gpu-min-free-mib') != '6144'):
        raise RecoveryAuthorizationError('Recovery helper exact authorized flags changed')
    return {'authorization': auth, 'reference': reference, 'e0': lock, 'path': path}


def observed_process(record, *, process_factory=None, expected_script=None):
    record = _process_record(record); factory = _process_factory(process_factory)
    try:
        process = factory(record['pid'])
        if abs(process.create_time() - record['create_time']) >= .01 or process.cmdline() != record['argv']:
            raise RecoveryAuthorizationError('Process PID/birth/argv identity changed')
        script = _script(process.cmdline(), process.cwd())
        if expected_script is not None and script != Path(expected_script).resolve():
            raise RecoveryAuthorizationError('Actual process script changed')
        if str(process.status()).lower() in {'stopped', 'zombie', 'dead'} or not process.is_running():
            raise RecoveryAuthorizationError('Authorized process is not actively executing')
        return process
    except RecoveryAuthorizationError:
        raise
    except Exception as error:
        raise RecoveryAuthorizationError('Authorized process cannot be inspected: ' + type(error).__name__) from error


def assert_original_supervisor_gone(auth, *, process_factory=None):
    import psutil
    old = auth['original_supervisor']
    try:
        process = _process_factory(process_factory)(old['pid'])
        if abs(process.create_time() - old['create_time']) < .01:
            raise RecoveryAuthorizationError('Original supervisor remains alive; no concurrent recovery')
    except psutil.NoSuchProcess:
        return
    except RecoveryAuthorizationError:
        raise
    except Exception as error:
        raise RecoveryAuthorizationError('Original supervisor death cannot be established') from error


def validate_helper_process(root, auth, process):
    argv = process.cmdline(); helper = auth['helper']; script = _script(argv, process.cwd())
    if script != _inside(root, helper['script']):
        raise RecoveryAuthorizationError('Recovery owner has a different actual script')
    index = next(i for i, value in enumerate(argv) if str(value).lower().endswith('.py'))
    if argv[index + 1:] != helper['arguments']:
        raise RecoveryAuthorizationError('Recovery owner actual argv differs from authorization')
    if Path(process.exe()).resolve() not in {Path(p).resolve() for p in helper['allowed_executables']}:
        raise RecoveryAuthorizationError('Recovery owner interpreter differs from authorization')
    return {'pid': process.pid, 'create_time': process.create_time(), 'argv': argv,
            'argv_sha256': argv_digest(argv), 'script': str(script), 'executable': str(Path(process.exe()).resolve())}


def validate_settled_costs(root):
    from .execution import _digest
    root = Path(root)
    if next((root / 'results/raw').rglob('*.jsonl.lock'), None) is not None:
        raise RecoveryAuthorizationError('A paid-call lock remains; no stage advancement')
    for path in (root / 'results/raw').rglob('*.jsonl'):
        for line in path.open(encoding='utf-8-sig'):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('record_sha256') != _digest({k: v for k, v in row.items() if k != 'record_sha256'}) or row.get('cost_status') != 'recorded_actual_usage':
                raise RecoveryAuthorizationError('Raw attempt integrity or actual cost is unresolved')
            from .inference import _validated_usage
            value = row.get('result', {}) if row.get('status') == 'success' else row.get('actual_usage')
            wall = row.get('result', {}).get('wall_time_s') if row.get('status') == 'success' else row.get('failure_wall_time_s')
            if (row.get('status') not in {'success', 'failure'} or _validated_usage(value) is None
                or isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0):
                raise RecoveryAuthorizationError('Actual call usage or wall time remains unknown')


@_authorization_guard
def validate_original_aggregation(root, auth):
    """Pure CPU identity/cost checks. Never calls a backend, tokenizer or scorer."""
    from .inference import ModelSpec, _validated_usage
    from .execution import ExecutionStore, validate_job
    from .preparation import read_jsonl, validate_success_rows
    root = Path(root); batch = auth['active_batch']; validate_settled_costs(root)
    jobs_path = verified_reference(root, batch['jobs']); output = _inside(root, batch['output'])
    if not output.is_file():
        raise RecoveryAuthorizationError('Original aggregation output is missing')
    before = artifact_reference(output); jobs = read_jsonl(jobs_path)
    if len(jobs) != 6000 or len({j.get('job_id') for j in jobs}) != 6000:
        raise RecoveryAuthorizationError('Exact six-thousand original jobs required')
    spec = ModelSpec(**read_json(verified_reference(root, batch['model_config']))); spec.validate()
    ids = {validate_job(j, spec, require_token_policy=True) for j in jobs}
    assignments = read_json(root / 'data/derived/split_manifest.json')['records']
    selected = {r['question_id']: r for r in assignments if r['split'] in {'e1_selection_calibration', 'e1_diagnostic_evaluation'}}
    pool = read_json(root / 'configs/pool_v1.json')['models']; aliases = {e['alias'] for e in pool}
    if len(selected) != 300 or len(aliases) != 6:
        raise RecoveryAuthorizationError('Original three-hundred question/six-model inventory changed')
    expected = {(qid, tuple(sorted(team))) for qid in selected for team in combinations(aliases, 3)}
    cells = {(j['question_id'], tuple(j['members'])) for j in jobs}
    if cells != expected:
        raise RecoveryAuthorizationError('Original aggregation grid coverage changed')
    for job in jobs:
        record = selected[job['question_id']]
        if (job['task'], job.get('subtask'), job['split']) != (record['task'], record.get('subtask'), record['split']):
            raise RecoveryAuthorizationError('Original job split/task identity changed')
    rows = validate_success_rows(output); store = ExecutionStore(output)
    if store.unreconciled or set(store.completed) != ids or any(r.get('job_id') not in ids for r in rows):
        raise RecoveryAuthorizationError('Original output coverage or actual failed cost unresolved')
    for job in jobs:
        row = store._cached(job, spec)
        if row is None:
            raise RecoveryAuthorizationError('Original aggregation is not completely cached')
        result = row['result']; usage = _validated_usage(result)
        wall = result.get('wall_time_s')
        if (usage is None or usage['prompt_tokens'] > spec.max_input_tokens
            or usage['completion_tokens'] > spec.max_new_tokens
            or isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0
            or not isinstance(result.get('text'), str) or result.get('model_identity_verified') is not True
            or result.get('seed') != job['seed'] or result.get('backend') != spec.backend):
            raise RecoveryAuthorizationError('Original output served identity or measured usage invalid')
    validate_settled_costs(root)
    if before != artifact_reference(output):
        raise RecoveryAuthorizationError('Original ledger changed during settlement validation')
    return {'schema': 'original_e1_aggregation_validation_v1', 'expected_jobs': 6000,
        'cached_successes': 6000, 'unknown_cost_attempts': 0, 'paid_locks_remaining': 0,
        'jobs': batch['jobs'], 'model_config': batch['model_config'], 'qualification': batch['qualification'],
        'output': before, 'generation_calls_performed_by_validation': 0, 'scorer_calls': 0}


class ReadOnlyProcessHandle:
    """Retain QUERY_LIMITED_INFORMATION|SYNCHRONIZE; no terminate/suspend rights."""
    def __init__(self, pid):
        if os.name != 'nt':
            raise RecoveryAuthorizationError('Production adoption requires Windows retained query handles')
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes; self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]; self.kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]; self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]; self.kernel.GetExitCodeProcess.restype = wintypes.BOOL
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        self.kernel.GetProcessTimes.restype = wintypes.BOOL
        self.handle = self.kernel.OpenProcess(0x100000 | 0x1000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        values = [wintypes.FILETIME() for _ in range(4)]
        if not self.kernel.GetProcessTimes(self.handle, *(ctypes.byref(v) for v in values)):
            self.close(); raise ctypes.WinError(ctypes.get_last_error())
        self.create_time = ((values[0].dwHighDateTime << 32) | values[0].dwLowDateTime) / 10000000. - 11644473600.

    def wait(self, timeout_ms=10000):
        from ctypes import wintypes
        value = self.kernel.WaitForSingleObject(self.handle, int(timeout_ms))
        if value == 258:
            return None
        if value != 0:
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, self.ctypes.byref(code)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        return int(code.value)

    def close(self):
        if getattr(self, 'handle', None):
            self.kernel.CloseHandle(self.handle); self.handle = None


def acquire_original_handles(root, auth, *, process_factory=None, handle_factory=None):
    assert_original_supervisor_gone(auth, process_factory=process_factory)
    handles = {}; factory = ReadOnlyProcessHandle if handle_factory is None else handle_factory
    try:
        for name in ('original_wrapper', 'original_worker'):
            record = auth[name]; observed_process(record, process_factory=process_factory,
                expected_script=Path(root) / 'scripts/run_aggregation.py')
            handle = factory(record['pid']); handles[name] = handle
            if abs(handle.create_time - record['create_time']) >= .01:
                raise RecoveryAuthorizationError('Retained process handle birth differs')
            observed_process(record, process_factory=process_factory,
                expected_script=Path(root) / 'scripts/run_aggregation.py')
        actual = observed_process(auth['original_worker'], process_factory=process_factory)
        if actual.ppid() != auth['original_wrapper']['pid']:
            raise RecoveryAuthorizationError('Original wrapper/worker relationship changed')
        return handles
    except BaseException:
        for handle in handles.values():
            handle.close()
        raise


def _validated_owner_artifact(root, status, loaded):
    owner_path = verified_reference(root, status.get('recovery_owner'))
    owner = read_json(owner_path); auth = loaded['authorization']; record = _process_record(owner)
    if (owner.get('schema') != 'e1_recovery_owner_v1'
        or owner.get('authorization') != loaded['reference']
        or owner.get('script') != str(_inside(root, auth['helper']['script']))
        or owner.get('recovery_sources') != auth['recovery_sources']
        or owner.get('original_worker') != auth['original_worker'] or owner.get('original_wrapper') != auth['original_wrapper']
        or owner.get('original_handles_acquired_and_birth_verified') is not True
        or status.get('recovery_authorization') != loaded['reference']
        or status.get('supervisor_pid') != record['pid']
        or status.get('supervisor_create_time') != record['create_time']
        or status.get('supervisor_script') != owner['script']
        or status.get('recovery_owner_argv_sha256') != owner['argv_sha256']):
        raise RecoveryAuthorizationError('Status is not bound to the real recovery owner artifact')
    argv = owner['argv']; index = next(i for i, v in enumerate(argv) if str(v).lower().endswith('.py'))
    if (_script(argv, root) != _inside(root, auth['helper']['script'])
        or argv[index + 1:] != auth['helper']['arguments']
        or Path(owner.get('executable', '')).resolve() not in {Path(v).resolve() for v in auth['helper']['allowed_executables']}
        or _inside(root, owner.get('lease_path', '')) != Path(root).resolve() / 'logs/e1_aggregation_recovery_supervisor.lock'
        or not isinstance(owner.get('lease_token'), str) or not owner['lease_token']):
        raise RecoveryAuthorizationError('Historical recovery owner argv/executable/lease differs from authorization')
    return owner


@_authorization_guard
def validated_recovery_owner(root, run_status, *, process_factory=None):
    loaded = validate_recovery_authorization(root, run_status.get('recovery_authorization'))
    owner = _validated_owner_artifact(root, run_status, loaded)
    process = observed_process(owner, process_factory=process_factory, expected_script=owner['script'])
    validate_helper_process(root, loaded['authorization'], process)
    lease = read_json(_inside(root, owner['lease_path']))
    if lease.get('token') != owner.get('lease_token') or lease.get('pid') != owner['pid'] or lease.get('create_time') != owner['create_time']:
        raise RecoveryAuthorizationError('Recovery owner exclusive lease no longer matches')
    if run_status.get('supervisor_running') is not True:
        raise RecoveryAuthorizationError('Recovery owner is not declaring active supervision')
    observation = {'exists': True, 'identity_verified': True, 'execution_active': True,
        'pid': owner['pid'], 'create_time': owner['create_time'], 'script': owner['script']}
    return {'identity_verified': True, 'execution_active': True, 'pid': owner['pid'],
        'create_time': owner['create_time'], 'argv': owner['argv'], 'script': owner['script'],
        'authorization_sha256': loaded['reference']['sha256'], 'process_observation': observation,
        'recovery_evidence': {'authorization': loaded['reference'], 'owner': run_status['recovery_owner'],
                              'phase': run_status.get('recovery_phase')}}


@_authorization_guard
def validated_recovery_completion(root, run_status, *, process_factory=None):
    import psutil
    loaded = validate_recovery_authorization(root, run_status.get('recovery_authorization'))
    owner = _validated_owner_artifact(root, run_status, loaded); auth = loaded['authorization']
    completion_path = verified_reference(root, run_status.get('recovery_completion'))
    completion = read_json(completion_path)
    if (completion.get('schema') != 'e1_recovery_completion_v1'
        or completion.get('authorization') != loaded['reference']
        or completion.get('owner') != run_status['recovery_owner']
        or completion.get('terminal_status') != {key: run_status.get(key) for key in TERMINAL_KEYS}
        or run_status.get('supervisor_running') is not False
        or run_status.get('E0') != 'PASS' or run_status.get('E1') not in {'PASS', 'STOP_E1'}
        or run_status.get('active_child_pid') is not None or run_status.get('active_script') is not None
        or run_status.get('scientific_completion') is not False
        or not _positive(run_status.get('supervisor_finished_at_unix'))):
        raise RecoveryAuthorizationError('Complete original E1 recovery terminal evidence required')
    settlement = read_json(verified_reference(root, completion['settlement']))
    if run_status.get('recovery_settlement') != completion['settlement']:
        raise RecoveryAuthorizationError('Terminal status reused another settlement')
    if settlement.get('schema') != 'e1_original_aggregation_settlement_v1' or settlement.get('authorization') != loaded['reference'] or settlement.get('owner') != run_status['recovery_owner']:
        raise RecoveryAuthorizationError('Original aggregation settlement provenance differs')
    for name in ('original_wrapper', 'original_worker'):
        proof = settlement.get('process_exits', {}).get(name, {})
        if proof.get('identity') != auth[name] or proof.get('exit_code') != 0 or proof.get('observed_with_retained_query_synchronize_handle') is not True:
            raise RecoveryAuthorizationError('Original wrapper/worker observed zero exits required')
    if settlement.get('aggregation') != validate_original_aggregation(root, auth):
        raise RecoveryAuthorizationError('Original six-thousand cached settlement changed')
    diagnostics = read_json(verified_reference(root, completion['e1_diagnostics'], expected='results/e1/e1_diagnostics.json'))
    if diagnostics.get('status') != run_status['E1'] or diagnostics.get('evidence', {}).get('e0_lock_sha256') != auth['original_e0']['sha256']:
        raise RecoveryAuthorizationError('E1 scientific receipt differs from original E0/terminal outcome')
    if run_status['E1'] == 'PASS':
        from .online_continuation import verify_matrix_receipt
        matrix_dir = Path(root) / 'data/derived/online_matrices_v1'
        matrix = verified_reference(root, completion.get('matrix_receipt'), expected=matrix_dir / 'matrix_preparation_receipt.json')
        if (run_status.get('overall') != 'PARTIAL' or run_status.get('step') != 'REAL_ONLINE_MATRICES_READY'
            or run_status.get('E2') != 'REAL_MATRICES_READY_NOT_E2_VALIDATION'
            or run_status.get('e2_executed') is not False
            or run_status.get('last_child_script') != 'prepare_online_matrices.py' or run_status.get('last_child_exit') != 0
            or run_status.get('matrix_preparation_receipt') != matrix.relative_to(root).as_posix()
            or run_status.get('matrix_preparation_receipt_sha256') != completion['matrix_receipt']['sha256']):
            raise RecoveryAuthorizationError('PASS recovery did not complete the original real-matrix continuation')
        verify_matrix_receipt(root, matrix_dir)
    elif (run_status.get('overall') != 'STOP_E1' or completion.get('matrix_receipt') is not None
          or run_status.get('step') != 'E1_DIAGNOSTICS_RECORDED'
          or run_status.get('E2') != 'STOPPED_OR_GATED'
          or run_status.get('last_child_script') != 'evaluate_real_e1.py'
          or run_status.get('last_child_exit') not in (0, 2) or run_status.get('e2_executed') is True):
        raise RecoveryAuthorizationError('Legitimate E1 STOP cannot claim later-stage matrices or skip original evaluation')
    validate_settled_costs(root)
    owner_gone = False; current_identity = False
    try:
        process = _process_factory(process_factory)(owner['pid'])
        if abs(process.create_time() - owner['create_time']) >= .01:
            owner_gone = True
        else:
            observed_process(owner, process_factory=process_factory, expected_script=owner['script'])
            validate_helper_process(root, auth, process); current_identity = True
    except psutil.NoSuchProcess:
        owner_gone = True
    except RecoveryAuthorizationError:
        raise
    except Exception as error:
        raise RecoveryAuthorizationError('Terminal recovery owner identity is unverifiable') from error
    observation = {'exists': not owner_gone, 'identity_verified': current_identity,
        'execution_active': current_identity, 'pid': owner['pid'],
        'create_time': owner['create_time'], 'script': owner['script']}
    return {'completion_verified': True, 'original_worker_exit_verified': True,
        'identity_verified': current_identity, 'historical_identity_verified': True,
        'execution_active': current_identity, 'owner_gone': owner_gone, 'pid': owner['pid'],
        'create_time': owner['create_time'], 'argv': owner['argv'], 'script': owner['script'],
        'authorization_sha256': loaded['reference']['sha256'], 'e1_status': run_status['E1'],
        'completion': run_status['recovery_completion'], 'process_observation': observation,
        'recovery_evidence': {'authorization': loaded['reference'], 'owner': run_status['recovery_owner'],
                              'phase': run_status.get('recovery_phase')}}
