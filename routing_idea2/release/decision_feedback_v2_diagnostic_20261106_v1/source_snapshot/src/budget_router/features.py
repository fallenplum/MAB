"""Frozen question projection and standard second-order fixed-size subset features.

The encoder is external and frozen: this module consumes embeddings, never labels.
Fixed K makes the basis rank deficient; ridge coefficients are predictive parameters,
not identified main effects or causal model-pair contributions.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from sklearn.decomposition import PCA


def _finite_array(value, *, ndim=None):
    result = np.asarray(value, dtype=float)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"Expected {ndim} dimensions, got {result.ndim}")
    if not np.isfinite(result).all():
        raise ValueError("Non-finite features are not permitted")
    return result


def enumerate_subsets(n_models=6, k=3):
    if not (isinstance(n_models, int) and isinstance(k, int) and 1 <= k <= n_models):
        raise ValueError("Require integer 1 <= k <= n_models")
    return tuple(combinations(range(n_models), k))


def canonical_subset(subset, n_models, k=None):
    raw = tuple(subset)
    if any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) for i in raw):
        raise ValueError("Model identifiers must be integer indices")
    result = tuple(sorted(int(i) for i in raw))
    if len(set(result)) != len(result) or any(i < 0 or i >= n_models for i in result):
        raise ValueError("Subset contains duplicate or out-of-pool model identifiers")
    if k is not None and len(result) != k:
        raise ValueError("Incorrect subset size")
    return result


def single_member_replacements(subset, n_models=6):
    base = canonical_subset(subset, n_models)
    members = set(base)
    return tuple(sorted({tuple(sorted((members - {old}) | {new}))
                         for old in base for new in range(n_models) if new not in members}))


class FrozenPCAFeatures:
    """Fit PCA exactly once on explicitly designated train embeddings, prepend 1.

    fit() rejects other split names and repeated fitting. The caller remains
    responsible for supplying embeddings from the recorded frozen encoder and
    for question-level disjoint splits. fit_ids supports provenance checks.
    """
    def __init__(self, n_components=16, *, encoder_id="external-frozen-encoder"):
        self.n_components = int(n_components)
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        self.encoder_id = str(encoder_id)
        self._pca = None
        self.fit_ids = ()
        self.fit_split = None

    def fit(self, train_embeddings, *, split="train", sample_ids=None):
        if split != "train":
            raise ValueError("PCA may be fitted only on the train split")
        if self._pca is not None:
            raise RuntimeError("Frozen PCA cannot be refitted")
        data = _finite_array(train_embeddings, ndim=2)
        if self.n_components > min(data.shape):
            raise ValueError("PCA dimension exceeds train samples or embedding dimensions")
        ids = tuple(sample_ids) if sample_ids is not None else ()
        if ids and (len(ids) != len(data) or len(set(ids)) != len(ids)):
            raise ValueError("Training question IDs must align and be unique")
        projection = PCA(n_components=self.n_components, svd_solver="full")
        projection.fit(data)
        self._pca = projection
        self.fit_ids = ids
        self.fit_split = split
        return self

    @property
    def pca_mean(self):
        if self._pca is None:
            raise RuntimeError("PCA has not been fitted")
        return self._pca.mean_.copy()

    @property
    def feature_dim(self):
        return self.n_components + 1

    def transform(self, embeddings):
        if self._pca is None:
            raise RuntimeError("PCA has not been fitted")
        data = _finite_array(embeddings)
        single = data.ndim == 1
        if single:
            data = data[None, :]
        if data.ndim != 2:
            raise ValueError("Embeddings must have one or two dimensions")
        reduced = self._pca.transform(data)
        result = np.column_stack((np.ones(len(data)), reduced))
        return result[0] if single else result


class SubsetFeatureMap:
    """psi(x,S)=kron(phi(x), [1,z_i,z_i*z_j for i<j])."""
    def __init__(self, n_models=6, k=3):
        self.n_models = n_models
        self.k = k
        self.subsets = enumerate_subsets(n_models, k)
        self.pairs = tuple(combinations(range(n_models), 2))
        self.basis_dim = 1 + n_models + len(self.pairs)
        self._basis = np.stack([self.basis(s) for s in self.subsets])

    def basis(self, subset):
        subset = canonical_subset(subset, self.n_models, self.k)
        z = np.zeros(self.n_models)
        z[list(subset)] = 1.0
        return np.r_[1.0, z, [z[i] * z[j] for i, j in self.pairs]]

    def transform(self, phi, subsets=None):
        phi = _finite_array(phi, ndim=1)
        basis = self._basis if subsets is None else np.stack([self.basis(s) for s in subsets])
        return np.einsum("p,ab->apb", phi, basis).reshape(len(basis), -1)

    def feature(self, phi, subset):
        return np.kron(_finite_array(phi, ndim=1), self.basis(subset))
