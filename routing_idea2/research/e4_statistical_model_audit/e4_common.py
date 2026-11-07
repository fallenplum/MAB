"""Frozen E3 dependencies and E4-only derived artifacts."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
WORK = ROOT / 'research/e4_statistical_model_audit'
PUBLIC = WORK / 'public'
PRIVATE = WORK / 'private'
E3 = ROOT / 'research/decision_acquisition_e3/public'
sys.path.insert(0, str(E3))
from assets import load_assets, make_splits, TEAMS, SEEDS, Env, algorithm_seed
from predictors import ModelTemplate as LegacyTemplate, Posterior as LegacyPosterior


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def semantic(data):
    return np.where(data['task'] == 'gsm8k', 'gsm8k', data['sub'])


def load_train():
    train, references = load_assets(ROOT)
    return train, references


def e4_splits(train):
    splits = make_splits(train)
    for fold in splits['folds']:
        rng = np.random.default_rng(np.random.SeedSequence([20261107, fold['fold'], 460360]))
        reference = fold['references'][120]
        smallest = []
        for task in ('gsm8k', 'bbh'):
            fixed = reference[train['task'][reference] == task]
            remaining = fold['nested'][120]
            remaining = remaining[(train['task'][remaining] == task) & ~np.isin(remaining, fixed)]
            smallest.extend(np.r_[fixed, rng.permutation(remaining)[:3]])
        fold['nested'][60] = np.sort(np.asarray(smallest, dtype=int))
        extra = np.setdiff1d(fold['history'], fold['nested'][240])
        selected = np.r_[rng.permutation(extra[train['task'][extra] == 'gsm8k'])[:60],
                         rng.permutation(extra[train['task'][extra] == 'bbh'])[:60]]
        fold['nested'][360] = np.sort(np.r_[fold['nested'][240], selected])
        for size in (60, 120, 240, 360, 480):
            rows = fold['nested'][size]
            assert len(rows) == size and np.sum(train['task'][rows] == 'gsm8k') == size//2
            assert set(reference) <= set(rows)
        assert all(set(fold['nested'][a]) <= set(fold['nested'][b]) for a, b in ((60,120),(120,240),(240,360),(360,480)))
        # Controlled B audit. Twenty logging-policy support questions precede a
        # fixed disjoint 100-question evaluator. All noise variants see the same labels.
        stream = fold['stream']
        support_gsm = rng.permutation(stream[train['task'][stream] == 'gsm8k'])[:10]
        names = sorted(set(train['sub'][stream][train['task'][stream] == 'bbh']))
        names = [names[j] for j in rng.permutation(len(names))[:10]]
        support_bbh = [int(rng.choice(stream[(train['task'][stream] == 'bbh') & (train['sub'][stream] == name)])) for name in names]
        fold['audit_support'] = np.sort(np.r_[support_gsm, support_bbh])
        fold['audit_query'] = np.setdiff1d(stream, fold['audit_support'])
        fold['audit_orders'] = {seed: order[np.isin(order, fold['audit_support'])] for seed, order in fold['orders'].items()}
        assert len(fold['audit_support']) == 20 and len(fold['audit_query']) == 100
        assert len(set(semantic(train)[fold['audit_query']]) - {'gsm8k'}) == 27
    splits['metadata']['e4_nested_seed'] = 20261107
    splits['metadata']['e4_sizes'] = [60,120,240,360,480]
    splits['metadata']['b_controlled_roles'] = '480 historical,20 fixed logging-support,100 untouched query per fold'
    return splits
