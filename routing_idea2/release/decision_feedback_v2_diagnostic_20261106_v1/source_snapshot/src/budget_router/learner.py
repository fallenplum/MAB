"""Shared original-label ridge regression and question-block bootstrap.

V=lambda*I+sum(psi*psi.T/sigma2), b=sum(psi*r/sigma2).
A difference is never an additional training observation. Design uncertainty
and Gaussian samples are working-model quantities, not coverage guarantees.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve


class FeedbackNotArrived(ValueError):
    pass


@dataclass(frozen=True)
class OriginalLabel:
    label_id: str
    question_id: str
    features: np.ndarray
    value: float
    available_at: int
    kind: str = "original"


def cold_start_limit(total_label_budget, requested=32):
    if not isinstance(total_label_budget, int) or total_label_budget < 0:
        raise ValueError("Label budget must be a nonnegative integer")
    if not isinstance(requested, int) or requested < 0:
        raise ValueError("Requested initialization must be a nonnegative integer")
    return min(32, requested, total_label_budget // 5)


def question_block_bootstrap_weights(question_ids, n_bootstrap, rng):
    """Return label weights; every label of a question shares one block count."""
    question_ids = tuple(question_ids)
    unique = tuple(dict.fromkeys(question_ids))
    if not unique or n_bootstrap < 2:
        raise ValueError("Need question blocks and at least two bootstrap draws")
    locations = {q: i for i, q in enumerate(unique)}
    counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=n_bootstrap)
    return counts[:, [locations[q] for q in question_ids]]


class SharedRidgeRegressor:
    def __init__(self, feature_dim, *, ridge=1.0, noise_variance=1.0,
                 bootstrap_min_labels=12, bootstrap_min_questions=4, prior_mean=None):
        self.feature_dim = int(feature_dim)
        self.ridge = float(ridge)
        self.noise_variance = float(noise_variance)
        if self.feature_dim <= 0 or not np.isfinite([self.ridge, self.noise_variance]).all() or min(self.ridge, self.noise_variance) <= 0:
            raise ValueError("Feature dimension, ridge and noise variance must be positive")
        if bootstrap_min_labels < 2 or bootstrap_min_questions < 2:
            raise ValueError("Bootstrap requires a nontrivial label/question startup gate")
        self.bootstrap_min_labels = int(bootstrap_min_labels)
        self.bootstrap_min_questions = int(bootstrap_min_questions)
        self._inverse = np.eye(self.feature_dim) / self.ridge
        self._design = np.eye(self.feature_dim) * self.ridge
        prior = np.zeros(self.feature_dim) if prior_mean is None else np.asarray(prior_mean, dtype=float)
        if prior.shape != (self.feature_dim,) or not np.isfinite(prior).all():
            raise ValueError("Invalid explicit ridge prior mean")
        self.prior_mean = prior.copy()
        self.prior_mean.flags.writeable = False
        self._rhs = self.ridge * self.prior_mean.copy()
        self._labels = {}

    @property
    def label_count(self):
        return len(self._labels)

    @property
    def question_count(self):
        return len({v.question_id for v in self._labels.values()})

    @property
    def labels(self):
        return tuple(OriginalLabel(v.label_id, v.question_id, v.features.copy(), v.value, v.available_at, v.kind) for v in self._labels.values())

    @property
    def design(self):
        return self._design.copy()

    @property
    def design_inverse(self):
        return self._inverse.copy()

    @property
    def theta(self):
        return self._inverse @ self._rhs

    def add_label(self, label, *, now):
        """Add an arrived original label once; identical replay returns False.

        Pending labels raise without mutation. Conflicting reuse of a label ID
        and pseudo/difference observations are rejected. Calling code controls
        authorized label delivery; the learner has no access to evaluator data.
        """
        if not isinstance(label.available_at, (int, np.integer)) or not isinstance(now, (int, np.integer)) or min(label.available_at, now) < 0:
            raise ValueError("Feedback availability and current time must be nonnegative integer ticks")
        if label.kind != "original":
            raise ValueError("Only original labels may enter the shared regression")
        if not label.label_id or not label.question_id:
            raise ValueError("Original label and question IDs are required")
        if label.available_at > now:
            raise FeedbackNotArrived("Feedback is not available at the current decision time")
        x = np.asarray(label.features, dtype=float)
        y = float(label.value)
        if x.shape != (self.feature_dim,) or not np.isfinite(x).all() or not np.isfinite(y) or not 0 <= y <= 1:
            raise ValueError("Invalid original quality label or feature vector")
        old = self._labels.get(label.label_id)
        if old is not None:
            if old.question_id != label.question_id or old.value != y or old.available_at != label.available_at or not np.array_equal(old.features, x):
                raise ValueError("Conflicting reuse of an original label ID")
            return False
        x = x.copy()
        x.flags.writeable = False
        stored = OriginalLabel(str(label.label_id), str(label.question_id), x, y,
                               int(label.available_at), label.kind)
        u = self._inverse @ x
        self._inverse -= np.outer(u, u) / (self.noise_variance + x @ u)
        self._inverse = (self._inverse + self._inverse.T) * 0.5
        self._design += np.outer(x, x) / self.noise_variance
        self._rhs += x * y / self.noise_variance
        self._labels[label.label_id] = stored
        return True

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.shape[-1] != self.feature_dim or not np.isfinite(x).all():
            raise ValueError("Invalid prediction features")
        return x @ self.theta

    def working_variance(self, directions):
        x = np.asarray(directions, dtype=float)
        if x.shape[-1] != self.feature_dim or not np.isfinite(x).all():
            raise ValueError("Invalid variance directions")
        return np.maximum(0.0, np.einsum("...i,ij,...j->...", x, self._inverse, x))

    def bootstrap_coefficients(self, n_bootstrap=24, *, rng):
        """Return (B,d) coefficients, or None before BOTH startup gates.

        Resample questions, not individual labels. This is an uncertainty
        approximation; fixed-K parameters remain non-identifiable.
        """
        if self.label_count < self.bootstrap_min_labels or self.question_count < self.bootstrap_min_questions:
            return None
        records = self.labels
        x = np.stack([v.features for v in records])
        y = np.array([v.value for v in records]) - x @ self.prior_mean
        weights = question_block_bootstrap_weights([v.question_id for v in records], n_bootstrap, rng)
        draws = []
        for weight in weights:
            active = weight > 0
            factor = np.sqrt(weight[active] / self.noise_variance)
            a = x[active] * factor[:, None]
            b = y[active] * factor
            if len(a) < self.feature_dim:
                dual = a @ a.T + self.ridge * np.eye(len(a))
                coefficient = a.T @ solve(dual, b, assume_a="pos", check_finite=False)
            else:
                primal = a.T @ a + self.ridge * np.eye(self.feature_dim)
                coefficient = solve(primal, a.T @ b, assume_a="pos", check_finite=False)
            draws.append(self.prior_mean + coefficient)
        return np.stack(draws)

    def thompson_sample(self, *, rng):
        covariance = (self._inverse + self._inverse.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        return self.theta + eigenvectors @ (np.sqrt(np.maximum(eigenvalues, 0)) * rng.normal(size=self.feature_dim))
