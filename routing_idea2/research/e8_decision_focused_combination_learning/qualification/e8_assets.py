"""Historical train600-only data and literal original direct20 feature recipe."""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

ROOT = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
ALPHA_RAW = 100.
FOLD_SHA256 = '6b433dcd2d8e6cd9938d2dd2d97261e18c3ae3d18325d7a04c6590e4d0600d50'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_argmax(values):
    """Preserve original numpy argmax, including exact lexicographic ties."""
    return np.argmax(np.asarray(values), axis=-1)


def outer_splits(root=ROOT):
    path = Path(root) / 'research/conditional_gain_20261106/private/FOLDS_PRIVATE.npz'
    if sha(path) != FOLD_SHA256:
        raise ValueError('Original five-fold artifact changed')
    with np.load(path, allow_pickle=False) as archive:
        splits = [(archive[f'fold{j}_train'].astype(int), archive[f'fold{j}_holdout'].astype(int))
                  for j in range(5)]
    held_all = []
    for history, held in splits:
        if len(history) != 480 or len(held) != 120 or set(history) & set(held):
            raise ValueError('Invalid historical fold')
        held_all.extend(held)
    if len(set(held_all)) != 600:
        raise ValueError('Outer holdouts must partition the original 600 questions')
    return splits


def load_historical(root=ROOT):
    sys.path.insert(0, str(Path(root) / 'research/decision_acquisition_e3/public'))
    from assets import load_assets
    data, refs = load_assets(root)
    return data, refs, outer_splits(root)


def inner_splits(data, rows, seed):
    """Three question-level folds, semantic stratification used for splitting only."""
    rows = np.asarray(rows, dtype=int)
    labels = np.array([str(t) + '/' + str(s) for t, s in zip(data['task'][rows], data['sub'][rows])])
    split = StratifiedKFold(3, shuffle=True, random_state=int(seed))
    return [(rows[fit], rows[held]) for fit, held in split.split(rows, labels)]


@dataclass
class FeatureMap:
    pca: PCA
    scaler: StandardScaler

    def phi(self, embeddings):
        projected = self.pca.transform(np.asarray(embeddings, dtype=float))
        standardized = self.scaler.transform(projected)
        return np.column_stack((np.ones(len(standardized)), standardized))

    def arrays(self):
        return dict(pca_mean=self.pca.mean_, pca_components=self.pca.components_,
                    scaler_mean=self.scaler.mean_, scaler_scale=self.scaler.scale_)


def fit_features(embeddings):
    embeddings = np.asarray(embeddings, dtype=float)
    pca = PCA(16, svd_solver='full').fit(embeddings)
    scaler = StandardScaler().fit(pca.transform(embeddings))
    return FeatureMap(pca, scaler)


def ridge_weights(phi, rewards, raw_alpha=ALPHA_RAW):
    rewards = np.asarray(rewards, dtype=float)
    means = rewards.mean(axis=1) if rewards.ndim == 3 else rewards
    if phi.shape[0] != means.shape[0] or means.shape[1] != 20:
        raise ValueError('Ridge needs one row per question and 20 action mean labels')
    u, s, vt = np.linalg.svd(phi, full_matrices=False)
    return (vt.T * (s / (s * s + float(raw_alpha) / 3))) @ (u.T @ means)


def matched_mean_loss_lambda(n_questions, raw_alpha=ALPHA_RAW):
    """Mean over questions and20 outputs; also M3's mean pair loss /20**2."""
    return float(raw_alpha) / (3 * int(n_questions) * 20)
