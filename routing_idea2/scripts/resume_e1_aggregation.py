"""Explicitly adopt the authorized original E1 worker; never rerun E0 or its batch.

The only overlay on the unchanged, E0-bound supervisor is its RUN_STATUS writer.
Production CLI requires the external authorization artifact and original 6144 MiB
threshold. Software tests inject process/handle stubs; CLI exposes no bypass.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from budget_router import recovery_runtime as rr


def _load_frozen(root):
    path = root / 'scripts/run_local_e1.py'
    spec = importlib.util.spec_from_file_location('e0_bound_local_e1_recovery_base', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _write_lease(path, identity, token):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {'pid': identity['pid'], 'create_time': identity['create_time'], 'token': token,
             'script': identity['script'], 'argv_sha256': identity['argv_sha256']}
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False); stream.flush(); os.fsync(stream.fileno())


def _lease_token(path):
    try:
        return rr.read_json(path).get('token')
    except (OSError, ValueError, TypeError):
        return None


def _release_owned_leases(paths, token):
    for path in reversed(paths):
        if _lease_token(path) == token:
            path.unlink(missing_ok=True)


def _acquire_supervisor_leases(root, identity, token):
    # Share the frozen entrypoint's O_EXCL fence as well as the recovery lease.
    paths = [root / 'logs/local_e1_supervisor.lock', root / 'logs/e1_aggregation_recovery_supervisor.lock']
    owned = []
    try:
        for path in paths:
            try:
                _write_lease(path, identity, token)
            except FileExistsError:
                # Pre-existing files are never ours, even if a token matches.
                raise
            except BaseException:
                # A write/fsync can fail after this call created its own token.
                if _lease_token(path) == token:
                    owned.append(path)
                raise
            else:
                owned.append(path)
        return owned
    except BaseException:
        _release_owned_leases(owned, token)
        raise


def wait_original_processes(auth, handles, *, heartbeat, process_factory=None):
    exits = {}
    while len(exits) != 2:
        for name in ('original_worker', 'original_wrapper'):
            if name in exits:
                continue
            code = handles[name].wait(timeout_ms=10000)
            if code is None:
                # Identity was checked around OpenProcess, including kernel birth.
                # The retained object cannot become a reused PID; do not lose a
                # normal exit racing with a second PID lookup.
                continue
            exits[name] = {'identity': auth[name], 'exit_code': code,
                'observed_with_retained_query_synchronize_handle': True}
            if code != 0:
                raise rr.RecoveryAuthorizationError('Original process exited nonzero; no retry or new GPU job')
        if len(exits) != 2:
            heartbeat()
    return exits


def continue_frozen_tail(root, frozen, *, continue_online_after_e1_pass):
    """Exactly the original tail after the first aggregations batch: alone first."""
    root = Path(root)
    frozen.wait_gpu(); frozen.state('E1_FULL_REAL_AGGREGATION', E1='IN_PROGRESS', active_batch='aggregator_alone')
    frozen.run('run_aggregation.py', '--jobs', root / 'data/jobs/e1/aggregator_alone.jsonl',
        '--model-config', root / 'configs/aggregator_v1.json', '--output', root / 'results/raw/e1/aggregator_alone.jsonl',
        '--qualification-receipt', root / 'artifacts/qualification/aggregator.json')
    frozen.run('prepare_e1_controls.py', '--kind', 'samples')
    selection = rr.read_json(root / 'artifacts/audit/e1_strong_model_selection.json')
    strong = selection['model_alias'].lower()
    frozen.wait_gpu(); frozen.state('E1_STRONG_MODEL_THREE_INDEPENDENT_SAMPLES', active_model=selection['model_alias'])
    frozen.run('run_members.py', '--jobs', root / 'data/jobs/e1/strong_samples.jsonl',
        '--model-config', root / 'configs/e1_strong_sampling.json', '--output', root / 'results/raw/e1/strong_samples.jsonl',
        '--qualification-receipt', root / 'artifacts/qualification' / ('member_' + strong + '.json'))
    frozen.run('prepare_e1_controls.py', '--kind', 'aggregate')
    frozen.wait_gpu(); frozen.state('E1_REAL_STRONG_SAMPLE_AGGREGATION')
    frozen.run('run_aggregation.py', '--jobs', root / 'data/jobs/e1/strong_aggregations.jsonl',
        '--model-config', root / 'configs/aggregator_v1.json', '--output', root / 'results/raw/e1/strong_aggregations.jsonl',
        '--qualification-receipt', root / 'artifacts/qualification/aggregator.json')
    frozen.state('E1_SCORING_COMPLETE_REAL_ARTIFACTS')
    result = frozen.evaluate_e1_receipt(); outcome = result['status']
    if outcome not in {'PASS', 'STOP_E1'}:
        raise rr.RecoveryAuthorizationError('Partial E1 evidence cannot complete recovery')
    frozen.state('E1_DIAGNOSTICS_RECORDED', E0='PASS', E1=outcome,
        E2='READY_FOR_E2_REAL_ASSETS' if outcome == 'PASS' else 'STOPPED_OR_GATED',
        E3='GATED_BY_E2', E4='GATED_BY_E3', E5='GATED_BY_E4')
    if continue_online_after_e1_pass and outcome == 'PASS':
        from budget_router.online_continuation import continue_online_assets
        continue_online_assets(root, run_script=frozen.run, gpu_admission=frozen.wait_gpu, update_state=frozen.state)
    return outcome


def execute_recovery(root, authorization_path, *, continue_online_after_e1_pass, gpu_min_free_mib,
                     process_factory=None, handle_factory=None, frozen_module=None):
    root = Path(root).resolve()
    loaded = rr.validate_recovery_authorization(root, authorization_path); auth = loaded['authorization']
    if continue_online_after_e1_pass is not auth['continue_online_after_e1_pass'] or gpu_min_free_mib != auth['gpu_min_free_mib']:
        raise rr.RecoveryAuthorizationError('Actual recovery CLI protocol differs from authorization')
    factory = rr._process_factory(process_factory)
    actual = factory(os.getpid()); identity = rr.validate_helper_process(root, auth, actual)
    token = uuid.uuid4().hex; lease = root / 'logs/e1_aggregation_recovery_supervisor.lock'
    handles = {}; owned_leases = []; frozen = None; original_writer = None
    metadata = {}; control = {'phase': 'WAITING_ORIGINAL_AGGREGATION'}
    evidence = root / 'artifacts/private/e1_aggregation_recovery' / loaded['reference']['sha256'] / token
    owner_reference = None; settlement_reference = None
    try:
        # Retain actual query/synchronize handles before issuing any status/state changes.
        handles = rr.acquire_original_handles(root, auth, process_factory=process_factory, handle_factory=handle_factory)
        owned_leases = _acquire_supervisor_leases(root, identity, token); evidence.mkdir(parents=True)
        if (root / 'RUN_STATUS.json').is_file():
            prior = evidence / 'prior_RUN_STATUS.json'; prior.write_bytes((root / 'RUN_STATUS.json').read_bytes())
        owner = {'schema': 'e1_recovery_owner_v1', **identity, 'authorization': loaded['reference'],
            'recovery_sources': auth['recovery_sources'], 'lease_path': str(lease), 'lease_token': token,
            'original_wrapper': auth['original_wrapper'], 'original_worker': auth['original_worker'],
            'original_handles_acquired_and_birth_verified': True, 'created_at_unix': time.time()}
        rr.immutable_json(evidence / 'owner.json', owner); owner_reference = rr.artifact_reference(evidence / 'owner.json')
        metadata = {'supervisor_pid': identity['pid'], 'supervisor_create_time': identity['create_time'],
            'supervisor_script': identity['script'], 'recovery_owner_argv_sha256': identity['argv_sha256'],
            'recovery_authorization': loaded['reference'], 'recovery_owner': owner_reference}
        frozen = _load_frozen(root) if frozen_module is None else frozen_module
        original_writer = frozen.atomic_json
        frozen.GPU_MIN_FREE_MIB = gpu_min_free_mib

        def decorate(value):
            value = dict(value); value.update(metadata, recovery_phase=control['phase'])
            if settlement_reference is not None:
                value['recovery_settlement'] = settlement_reference
            return value

        def writer(path, value):
            if Path(path).resolve() == root / 'RUN_STATUS.json':
                rr.atomic_json(path, decorate(value))
            else:
                original_writer(path, value)

        frozen.atomic_json = writer
        frozen.state('E1_RECOVERY_WAITING_EXISTING_AGGREGATION', E0='PASS', E1='IN_PROGRESS',
            active_batch='aggregations', active_child_pid=auth['original_worker']['pid'], active_script='run_aggregation.py',
            active_child_wrapper_pid=auth['original_wrapper']['pid'],
            active_child_create_time=auth['original_worker']['create_time'],
            failure_type=None, failure_summary=None)

        def heartbeat():
            # Heartbeats never infer completion or start another process.
            current = rr.read_json(root / 'RUN_STATUS.json'); current['updated_at_unix'] = time.time()
            writer(root / 'RUN_STATUS.json', current)

        exits = wait_original_processes(auth, handles, heartbeat=heartbeat, process_factory=process_factory)
        aggregation = rr.validate_original_aggregation(root, auth)
        settlement = {'schema': 'e1_original_aggregation_settlement_v1', 'authorization': loaded['reference'],
            'owner': owner_reference, 'process_exits': exits, 'aggregation': aggregation,
            'settled_at_unix': time.time(), 'model_calls_started_by_adoption': 0}
        rr.immutable_json(evidence / 'settlement.json', settlement)
        settlement_reference = rr.artifact_reference(evidence / 'settlement.json')
        control['phase'] = 'CONTINUING_FROZEN_E1_TAIL'
        frozen.state('E1_RECOVERY_ORIGINAL_AGGREGATION_SETTLED', E0='PASS', E1='IN_PROGRESS',
            active_child_pid=None, active_script=None, active_child_wrapper_pid=None)
        # Revalidate E0 and the authorization after waiting; no source revision is implicit.
        rr.validate_recovery_authorization(root, loaded['reference'])
        frozen.verify_e0_after_child()
        outcome = continue_frozen_tail(root, frozen, continue_online_after_e1_pass=continue_online_after_e1_pass)
        rr.validate_settled_costs(root)
        control['phase'] = 'TERMINAL_E1_PASS' if outcome == 'PASS' else 'TERMINAL_E1_STOP'
        final = rr.read_json(root / 'RUN_STATUS.json')
        final.update(overall='STOP_E1' if outcome == 'STOP_E1' else 'PARTIAL',
            supervisor_finished_at_unix=time.time(), supervisor_running=False, active_child_pid=None, active_script=None)
        final = decorate(final)
        matrix = root / 'data/derived/online_matrices_v1/matrix_preparation_receipt.json'
        completion = {'schema': 'e1_recovery_completion_v1', 'authorization': loaded['reference'],
            'owner': owner_reference, 'settlement': settlement_reference,
            'e1_diagnostics': rr.artifact_reference(root / 'results/e1/e1_diagnostics.json'),
            'matrix_receipt': rr.artifact_reference(matrix) if outcome == 'PASS' else None,
            'terminal_status': {key: final.get(key) for key in rr.TERMINAL_KEYS},
            'completed_at_unix': time.time(), 'research_complete': False,
            'owner_future_exit_code_is_not_claimed': True}
        rr.immutable_json(evidence / 'completion.json', completion)
        final['recovery_completion'] = rr.artifact_reference(evidence / 'completion.json')
        # Independent CPU revalidation precedes publishing the terminal state.
        rr.validated_recovery_completion(root, final, process_factory=process_factory)
        writer(root / 'RUN_STATUS.json', final)
        return final
    except BaseException as error:
        if owner_reference is not None:
            try:
                current = rr.read_json(root / 'RUN_STATUS.json') if (root / 'RUN_STATUS.json').exists() else {}
                control['phase'] = 'STOPPED_FOR_REVIEW'
                current.update(metadata, recovery_phase=control['phase'], overall='STOPPED_FOR_REVIEW',
                    supervisor_running=False, scientific_completion=False, failure_type=type(error).__name__,
                    failure_summary='Recovery stopped; inspect private supervisor log; no automatic retry.')
                # Preserve active child metadata. Do not terminate an unfinished call.
                rr.atomic_json(root / 'RUN_STATUS.json', current)
            except Exception:
                pass
        raise
    finally:
        if frozen is not None and original_writer is not None:
            frozen.atomic_json = original_writer
        for handle in handles.values():
            handle.close()
        _release_owned_leases(owned_leases, token)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--continue-online-after-e1-pass', action='store_true', required=True)
    parser.add_argument('--gpu-min-free-mib', type=int, required=True, choices=[6144])
    args = parser.parse_args(argv)
    return execute_recovery(ROOT, args.authorization,
        continue_online_after_e1_pass=args.continue_online_after_e1_pass,
        gpu_min_free_mib=args.gpu_min_free_mib)


if __name__ == '__main__':
    main()
