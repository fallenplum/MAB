"""Shared frozen E9 identities, folds and immutable artifact operations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

ASSETS = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
E8 = ASSETS / 'research/e8_decision_focused_combination_learning'
NS = ASSETS / 'research/m4_noise_aware_set_learning'
ROOT = ASSETS / 'research/e9_consistent_cost_sensitive_learning'
sys.path[:0] = [str(E8 / 'public/qualification'), str(NS / 'private/runtime/noise')]
from e8_assets import fit_features, inner_splits

MULTIPLIERS = (.1, 1., 10.)
PERMUTATION_SEEDS = (20261114091, 20261114092, 20261114093)
MEMBERS = ('Qwen3-8B', 'Qwen2.5-Coder-7B-Instruct', 'Granite-3.3-8b-instruct',
           'DeepHermes-3-Llama-3-8B-Preview', 'DeepSeek-R1-Distill-Qwen-7B',
           'Cogito-v1-preview-llama-8B')
TEAMS = tuple(itertools.combinations(range(6), 3))
ACTION_KEYS = tuple(tuple(sorted(MEMBERS[i] for i in team)) for team in TEAMS)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()) + '\n', encoding='utf-8')


def archive(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_sources(lock_name):
    lock = read(ROOT / 'private' / lock_name)
    if sha(ROOT / 'public/E9_PROTOCOL.json') != lock['protocol_sha256']:
        raise AssertionError('Frozen E9 protocol changed')
    for name, digest in lock['sources'].items():
        if sha(Path(__file__).parent / name) != digest:
            raise AssertionError('Frozen E9 source changed ' + name)
    return lock


def subset(data, rows):
    rows = np.asarray(rows, dtype=int)
    return {key: value[rows].copy() for key, value in data.items()
            if isinstance(value, np.ndarray) and value.ndim and len(value) == len(data['ids'])}


def fold_packs(fold):
    folder = E8 / f'private/qualification/fold_packs/fold{fold}'
    history_path, evaluation_path = folder / 'history480.npz', folder / 'evaluation_inputs_only120.npz'
    manifest_path = E8 / 'private/qualification/FOLD_PACKS_MANIFEST.json'
    protocol_path = E8 / 'public/E8_PROTOCOL.json'
    expected = read(ROOT / 'private/PARENT_SNAPSHOTS.json')['e8_decision_focused_combination_learning']
    if sha(protocol_path) != expected['public/E8_PROTOCOL.json']:
        raise AssertionError('Original E8 protocol changed')
    if sha(manifest_path) != read(protocol_path)['data']['fold_pack_manifest_sha256']:
        raise AssertionError('Original fold-pack manifest changed')
    spec = read(manifest_path)['folds'][fold]
    if sha(history_path) != spec['history']['sha256'] or sha(evaluation_path) != spec['evaluation_inputs']['sha256']:
        raise AssertionError('Original fold pack changed')
    history, evaluation = archive(history_path), archive(evaluation_path)
    from csce_operators import validate_training_inputs
    validate_training_inputs(history, evaluation)
    log_path = E8 / f'private/training/fold{fold}/M0_tuned/FIT_LOG.json'
    if sha(log_path) != expected[f'private/training/fold{fold}/M0_tuned/FIT_LOG.json']:
        raise AssertionError('Original A selection log changed')
    log = read(log_path)
    if len(log['candidates']) != 3 or any(len(item['fits']) != 3 for item in log['candidates']):
        raise AssertionError('Three original candidates and three inner folds required')
    saved = [(np.array(item['fitting_rows']), np.array(item['held_rows']))
             for item in log['candidates'][0]['fits']]
    recreated = inner_splits(history, np.arange(len(history['ids'])), 20261113 + fold)
    for (a, b), (c, d) in zip(saved, recreated):
        if not np.array_equal(a, c) or not np.array_equal(b, d):
            raise AssertionError('E8 inner split differs')
    for candidate in log['candidates']:
        for (a, b), item in zip(saved, candidate['fits']):
            if not np.array_equal(a, item['fitting_rows']) or not np.array_equal(b, item['held_rows']):
                raise AssertionError('E8 grid candidates used different split')
    return history, evaluation, saved, dict(history_sha256=sha(history_path),
        evaluation_sha256=sha(evaluation_path), old_log_sha256=sha(E8 / f'private/training/fold{fold}/M0_tuned/FIT_LOG.json'))


def permute_region(region, seed):
    rng = np.random.default_rng(seed)
    mapping = np.arange(len(region['ids']))
    for label in sorted(set(zip(region['task'].tolist(), region['sub'].tolist()))):
        rows = np.flatnonzero((region['task'] == label[0]) & (region['sub'] == label[1]))
        mapping[rows] = rng.permutation(rows)
    result = {key: value.copy() for key, value in region.items()}
    result['Y'] = region['Y'][mapping].copy()
    return result, mapping
