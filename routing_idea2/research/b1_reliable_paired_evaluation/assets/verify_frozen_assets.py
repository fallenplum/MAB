"""Audit B1 actor reproducibility and application-level hidden-data boundary."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import builtins
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=r'E:\语言模型组合路由')
    args = parser.parse_args()
    root = Path(args.root)
    work = root / 'research/b1_reliable_paired_evaluation'
    audit_dir = work / 'private/assets/replay_boundary_audit' / str(time.time_ns())
    source = Path(__file__).with_name('freeze_assets.py')
    spec = importlib.util.spec_from_file_location('b1_freeze_assets', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen = json.loads((work / 'private/assets/FREEZE_RECEIPT.json').read_text(encoding='utf8'))
    assert module.sha(source) == frozen['script_sha256'], 'Frozen actor implementation changed'
    state_path = root / 'research/conditional_gain_20261106/private/FITTED_CANDIDATE_STATES.pkl'
    assert module.sha(state_path) == frozen['router']['source_sha256'], 'Historical fitted state changed'
    original_open = builtins.open
    original_io_open = __import__('io').open
    access = []

    def guarded_open(path, *positional, **keywords):
        name = str(path).replace('\\', '/').lower()
        # Deny the real withheld files and any fresh/sealed score payload.
        if ('selector_validation_outcomes' in name or
            ('fresh' in name or 'sealed' in name) and
            any(token in name for token in ('outcome', '.npz', '.jsonl'))):
            raise PermissionError('B1 actor attempted hidden reward or future payload access')
        access.append(name)
        return original_open(path, *positional, **keywords)

    def guarded_io_open(path, *positional, **keywords):
        name = str(path).replace('\\', '/').lower()
        if ('selector_validation_outcomes' in name or
            ('fresh' in name or 'sealed' in name) and
            any(token in name for token in ('outcome', '.npz', '.jsonl'))):
            raise PermissionError('B1 actor attempted hidden reward or future payload access')
        access.append(name)
        return original_io_open(path, *positional, **keywords)

    started = time.perf_counter()
    builtins.open = guarded_open
    __import__('io').open = guarded_io_open
    old_argv = sys.argv
    sys.argv = [str(source), '--root', str(root), '--work', str(audit_dir)]
    try:
        module.main()
    finally:
        builtins.open = original_open
        __import__('io').open = original_io_open
        sys.argv = old_argv
    errors = {}
    for name in ('FROZEN_INPUTS.npz', 'FROZEN_MODEL.npz', 'TRAIN_OOF.npz'):
        first = work / 'private/assets' / name
        second = audit_dir / 'private/assets' / name
        with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                assert np.array_equal(a[key], b[key], equal_nan=False), (name, key)
        errors[name] = 'identical arrays with future payloads access-denied'
    with np.load(work / 'private/assets/FROZEN_INPUTS.npz', allow_pickle=False) as a:
        assert not any(key in a.files for key in ('Y', 'd', 'D', 'aggregate_scores'))
        assert np.all(a['m'][~a['active']] == 0)
        assert np.all(a['cost_prediction'][~a['active']] == 0)
        assert np.all(a['vhat'][~a['active']] == 0)
        assert np.all(a['rawmom'][~a['active']] == 0)
        assert np.all(a['vhat'][a['active']] >= 1e-6)
        assert np.all(a['predicted_aggregator_calls'][a['active']] == 6)
        assert np.all(np.isin(a['predicted_member_calls'][a['active']], (12, 15, 18)))
    result = dict(status='PASS', arrays_equal=errors,
        hidden_validation_rewards_or_costs_opened=0, fresh_or_sealed_payloads_opened=0,
        boundary='Python builtins.open and io.open path guard during complete actor rebuild',
        limitation='Application-level access audit; no claim of OS-wide process confinement',
        file_access_attempts=len(access), unique_access_paths=len(set(access)),
        checks=['same-action structural zero', 'reward-free actor artifact schema',
                'three-seed member execution union', 'two distinct combinations: six aggregator calls',
                'raw-repeat/mean Ridge equivalence retained from freeze receipt'],
        wall_seconds=time.perf_counter() - started)
    module.put(work / 'public/assets/ASSET_REPLAY_AUDIT.json', result)
    module.put(work / 'private/assets/ACCESS_REPLAY_LOG.json', dict(result=result, access=access))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
