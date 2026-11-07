"""E4 Gaussian working models with explicitly separated historical likelihoods.

Every historical objective is alpha*||W||_F^2 plus the sum, over authorized
questions and generation seeds, of raw twenty-reward residual quadratic forms
weighted by R_hist^-1. Online bundles use R_online and only delivered raw labels.
"""

from dataclasses import dataclass
import hashlib
import json
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from e4_common import LegacyTemplate, LegacyPosterior


def _validate(train, history, training_seeds):
    size = len(train["E"])
    history = np.asarray(history)
    if history.ndim != 1 or history.dtype.kind not in "iu" or not len(history):
        raise ValueError("history must contain integer question indices")
    history = history.astype(np.int64, copy=True)
    if len(np.unique(history)) != len(history) or np.any(history < 0) or np.any(history >= size):
        raise ValueError("Historical question indices are duplicated or invalid")
    seeds = tuple(training_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(
            isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 3
            for seed in seeds):
        raise ValueError("training_seeds must contain unique authorized indices from 0,1,2")
    source = np.asarray(train["Y"])
    if source.shape != (size, 3, 20):
        raise ValueError("Y must have the frozen question/three-seed/twenty-action shape")
    # No unrequested seed or nonhistorical reward is indexed or copied.
    rewards = source[np.ix_(history, np.asarray(seeds), np.arange(20))].astype(float)
    if not np.isfinite(rewards).all() or np.any((rewards < 0) | (rewards > 1)):
        raise ValueError("Authorized historical rewards must be finite and in [0,1]")
    return history, seeds, rewards


def _fingerprint(train, history, seeds, rewards, predictor, alpha):
    digest = hashlib.sha256()
    digest.update(json.dumps({"predictor": predictor, "alpha": alpha, "seeds": list(seeds)}, sort_keys=True).encode())
    for value in (history, np.asarray(train["E"])[history], np.asarray(train["task"])[history],
                  np.asarray(train["sub"])[history], rewards):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _geometry(train, history, predictor, refs, rewards):
    safe = {key: train[key] for key in ("E", "task", "sub", "ids")}
    safe["Y"] = np.zeros((len(train["E"]), 3, 20))
    safe["Y"][history] = rewards.mean(axis=1)[:, None, :]
    return LegacyTemplate(safe, history, predictor, refs, {"alpha": 100., "pca_components": 16})


def _e3_noise(rewards):
    repeats = rewards.shape[1]
    if repeats == 1:
        return np.eye(20)
    residual = rewards - rewards.mean(axis=1, keepdims=True)
    covariance = np.einsum("nla,nlb->ab", residual, residual) / ((repeats - 1) * len(rewards))
    variance = np.maximum(np.diag(covariance), .01)
    standard_deviation = np.sqrt(variance)
    corr = covariance / np.outer(standard_deviation, standard_deviation)
    np.fill_diagonal(corr, 1.)
    result = .5 * (np.eye(20) + corr)
    return (result + result.T) * .5


@dataclass(frozen=True)
class NoiseFit:
    source_fingerprint: str
    history: np.ndarray
    training_seeds: tuple
    predictor: str
    alpha: float
    diagonal: np.ndarray
    second_moment: np.ndarray
    centered_covariance: np.ndarray
    residual_bias: np.ndarray
    residual_correlation: np.ndarray
    heteroscedastic_correlated: np.ndarray
    oof_predictions: np.ndarray
    oof_fold: np.ndarray
    fold_provenance: tuple

    def summary(self):
        return {"source_fingerprint": self.source_fingerprint, "history_questions": len(self.history),
                "training_seeds": list(self.training_seeds), "predictor": self.predictor,
                "alpha": self.alpha, "oof_folds": 3, "shrinkage_to_identity": .5,
                "variance_floor": .01, "residual_target": "single-generation raw reward minus question-grouped OOF prediction",
                "estimator": "question-equal raw residual second moment; centered covariance and bias recorded separately",
                "diagonal": self.diagonal.tolist(), "residual_bias": self.residual_bias.tolist(),
                "centered_covariance_trace": float(np.trace(self.centered_covariance)),
                "second_moment_trace": float(np.trace(self.second_moment)),
                "covariance_interpretation": "working predictive residual scale, not an identified pure generation-noise covariance"}


def fit_noise(train, history, predictor, training_seeds=(0, 1, 2), alpha=100.0):
    """Fit shared N4/N5 scales using history-only three-fold, question-grouped OOF."""
    history, seeds, rewards = _validate(train, history, training_seeds)
    predictor, alpha = str(predictor).upper(), float(alpha)
    tasks = np.asarray(train["task"])[history]
    if min(np.sum(tasks == task) for task in set(tasks)) < 3:
        raise ValueError("OOF noise fitting needs at least three historical questions per broad task")
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=20261108)
    predictions = np.zeros((len(history), 20))
    assignments = np.full(len(history), -1, dtype=int)
    provenance = []
    for fold, (fit, held_out) in enumerate(splitter.split(history, tasks)):
        fit_ids, held_ids = history[fit], history[held_out]
        if set(fit_ids) & set(held_ids):
            raise AssertionError("OOF question identities overlap")
        model = GaussianTemplate(train, fit_ids, predictor, fit_ids[:1], variant="N1",
                                 alpha=alpha, training_seeds=seeds)
        coefficient = model.mean.reshape(model.question_dimension, model.action_dimension)
        predictions[held_out] = model.X[held_ids] @ coefficient @ model.B.T
        assignments[held_out] = fold
        provenance.append({"fold": fold, "fit_indices": fit_ids.tolist(), "predict_indices": held_ids.tolist(),
                           "pca_fit_indices": fit_ids.tolist(), "all_seeds_of_question_share_fold": True})
    if np.any(assignments < 0):
        raise AssertionError("OOF predictions do not cover historical questions")
    residual = rewards - predictions[:, None, :]
    second = np.einsum("nla,nlb->ab", residual, residual) / (len(history) * len(seeds))
    second = .5 * (second + second.T)
    bias = residual.mean(axis=(0, 1))
    centered = second - np.outer(bias, bias)
    centered = .5 * (centered + centered.T)
    diagonal = np.maximum(np.diag(second), .01)
    corr = second / np.sqrt(np.outer(diagonal, diagonal))
    np.fill_diagonal(corr, 1.)
    shrunk = .5 * (np.eye(20) + corr)
    correlated = shrunk * np.sqrt(np.outer(diagonal, diagonal))
    correlated = .5 * (correlated + correlated.T)
    np.linalg.cholesky(correlated)
    arrays = (history, diagonal, second, centered, bias, corr, correlated, predictions, assignments)
    for array in arrays:
        array.flags.writeable = False
    return NoiseFit(_fingerprint(train, history, seeds, rewards, predictor, alpha), history, seeds,
                    predictor, alpha, diagonal, second, centered, bias, corr, correlated,
                    predictions, assignments, tuple(provenance))


@dataclass(frozen=True)
class _InitialState:
    mean: np.ndarray
    covariance: np.ndarray
    ref_predictions: np.ndarray
    ref_cross: np.ndarray


class GaussianTemplate:
    """N1..N5 template compatible with the frozen E3 posterior/projected KG APIs."""

    def __init__(self, train, history, predictor, refs, variant="N1", alpha=100.0,
                 training_seeds=(0, 1, 2), noise_fit=None):
        history, seeds, rewards = _validate(train, history, training_seeds)
        self.variant, self.predictor = str(variant).upper(), str(predictor).upper()
        if self.variant not in ("N1", "N2", "N3", "N4", "N5"):
            raise ValueError("Unknown Gaussian likelihood variant")
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.training_seeds, self.training_replicates = seeds, len(seeds)
        geometry = _geometry(train, history, self.predictor, refs, rewards)
        for name in ("history", "reference_indices", "X", "B", "original_basis", "basis_projection",
                     "pca", "scaler", "subtask_names", "question_dimension", "action_dimension",
                     "dimension", "_identity_basis", "refF", "fixed_actions", "global_fixed_action",
                     "task_fixed_actions", "subtask_fixed_actions"):
            setattr(self, name, getattr(geometry, name))
        self._historical_means = rewards.mean(axis=1)
        self._gram = self.X[self.history].T @ self.X[self.history]
        self.e3_noise = _e3_noise(rewards)
        self.noise_fit = noise_fit
        if self.variant in ("N4", "N5") and noise_fit is None:
            self.noise_fit = fit_noise(train, history, self.predictor, seeds, self.alpha)
        if self.noise_fit is not None:
            expected = _fingerprint(train, history, seeds, rewards, self.predictor, self.alpha)
            if not isinstance(self.noise_fit, NoiseFit) or self.noise_fit.source_fingerprint != expected:
                raise ValueError("NoiseFit belongs to another history, seed permission, predictor or alpha")
        unit = np.eye(20)
        if self.variant == "N1":
            self.history_noise, self.online_noise = unit, unit
        elif self.variant == "N2":
            self.history_noise, self.online_noise = unit, self.e3_noise
        elif self.variant == "N3":
            self.history_noise = self.online_noise = self.e3_noise
        elif self.variant == "N4":
            self.history_noise = self.online_noise = np.diag(self.noise_fit.diagonal)
        else:
            self.history_noise = self.online_noise = self.noise_fit.heteroscedastic_correlated
        self.noise = self.online_noise
        self.R_hist, self.R_online = self.history_noise, self.online_noise
        self._initial = self._fit_initial()

    @property
    def mean(self):
        return self._initial.mean.copy()

    @property
    def cov(self):
        return self._initial.covariance.copy()

    def transform(self, embeddings, tasks=None, subtasks=None):
        if isinstance(embeddings, dict):
            embeddings, tasks, subtasks = embeddings["E"], embeddings["task"], embeddings["sub"]
        reduced = self.scaler.transform(self.pca.transform(np.asarray(embeddings, float)))
        columns = [np.ones((len(reduced), 1)), reduced]
        if self.predictor == "P3":
            tasks, subtasks = np.asarray(tasks), np.asarray(subtasks)
            bbh = tasks == "bbh"
            indicators = np.column_stack([bbh & (subtasks == name) for name in self.subtask_names])
            columns.extend((bbh[:, None], indicators))
        return np.column_stack(columns).astype(float)

    def _fit_initial(self):
        weighted_basis = np.linalg.solve(self.history_noise, self.B)
        action_gram = self.B.T @ weighted_basis
        qvalue, qvector = np.linalg.eigh(self._gram)
        avalue, avector = np.linalg.eigh(action_gram)
        inverse = 1. / (self.alpha + self.training_replicates * np.maximum(qvalue, 0)[:, None]
                        * np.maximum(avalue, 0)[None, :])
        rhs = self.training_replicates * self.X[self.history].T @ self._historical_means @ weighted_basis
        coefficient = qvector @ ((qvector.T @ rhs @ avector) * inverse) @ avector.T
        blocks = np.einsum("pa,qa,ab->pqb", qvector, qvector, inverse, optimize=True)
        covariance = np.einsum("pqb,ib,jb->piqj", blocks, avector, avector, optimize=True).reshape(self.dimension, self.dimension)
        covariance = .5 * (covariance + covariance.T)
        ref_predictions = self.X[self.reference_indices] @ coefficient @ self.B.T
        ref_cross = (self.refF.reshape(-1, self.dimension) @ covariance).reshape(self.refF.shape)
        return _InitialState(coefficient.reshape(-1), covariance, ref_predictions, ref_cross)

    def clone(self, independent_noise=False):
        if not isinstance(independent_noise, (bool, np.bool_)):
            raise TypeError("independent_noise must be boolean")
        return LegacyPosterior(self, self._initial, np.eye(20) if independent_noise else self.online_noise)


def _contrast_variances(posterior):
    covariance = np.einsum("mad,mbd->mab", posterior.refF, posterior.ref_cross)
    diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    return diagonal[:, :, None] + diagonal[:, None, :] - 2 * covariance


def qualification():
    rng = np.random.default_rng(20261108)
    n = 80
    train = {"E": rng.normal(size=(n, 24)), "Y": rng.integers(0, 2, (n, 3, 20)).astype(float),
             "task": np.array(["gsm8k" if i % 2 else "bbh" for i in range(n)]),
             "sub": np.array(["" if i % 2 else "logic" for i in range(n)]),
             "ids": np.array([str(i) for i in range(n)])}
    history, refs = np.arange(60), np.arange(4)
    results = []
    for predictor in ("P1", "P2"):
        noise_fit = fit_noise(train, history, predictor)
        for fold in noise_fit.fold_provenance:
            assert not set(fold["fit_indices"]) & set(fold["predict_indices"])
            assert fold["pca_fit_indices"] == fold["fit_indices"]
        previous_unit = None
        for variant in ("N1", "N2", "N3", "N4", "N5"):
            template = GaussianTemplate(train, history, predictor, refs, variant, noise_fit=noise_fit)
            posterior = template.clone()
            x, b = template.X[history], template.B
            weighted = np.linalg.solve(template.history_noise, b)
            precision = 100 * np.eye(template.dimension) + 3 * np.kron(x.T @ x, b.T @ weighted)
            rhs = (3 * x.T @ train["Y"][history].mean(1) @ weighted).reshape(-1)
            initial_mean = np.linalg.solve(precision, rhs)
            initial_cov = np.linalg.inv(precision)
            error = max(float(np.max(abs(posterior.mean - initial_mean))), float(np.max(abs(posterior.cov - initial_cov))))
            assert error < 1e-10
            if variant == "N1":
                previous_unit = posterior.mean.copy()
            if variant == "N2":
                legacy = LegacyTemplate(train, history, predictor, refs, {})
                assert np.array_equal(template.noise, legacy.noise)
                assert np.array_equal(posterior.mean, legacy.clone().mean)
                assert np.array_equal(posterior.cov, legacy.clone().cov)
                assert np.array_equal(posterior.mean, previous_unit)
            if predictor == "P1":
                raw_b = template.original_basis
                weighted_raw = np.linalg.solve(template.history_noise, raw_b)
                raw_precision = 100 * np.eye(17 * 22) + 3 * np.kron(x.T @ x, raw_b.T @ weighted_raw)
                raw_rhs = (3 * x.T @ train["Y"][history].mean(1) @ weighted_raw).reshape(-1)
                h_raw = np.einsum("p,ar->apr", template.X[70], raw_b).reshape(20, -1)
                h = posterior.observation(70, np.arange(20))
                assert np.max(abs(h_raw @ np.linalg.solve(raw_precision, raw_rhs) - posterior.predict(70))) < 1e-10
                assert np.max(abs(h_raw @ np.linalg.solve(raw_precision, h_raw.T) - h @ posterior.cov @ h.T)) < 1e-10
            before = posterior.cov.copy()
            contrast_before = _contrast_variances(posterior)
            actions, values = np.array([2, 5]), np.array([0., 1.])
            h = posterior.observation(70, actions)
            r = posterior.noise[np.ix_(actions, actions)]
            precision += h.T @ np.linalg.solve(r, h)
            rhs += h.T @ np.linalg.solve(r, values)
            posterior.update(70, actions, values)
            update_error = max(float(np.max(abs(posterior.mean - np.linalg.solve(precision, rhs)))),
                               float(np.max(abs(posterior.cov - np.linalg.inv(precision)))))
            assert update_error < 1e-10
            tolerance = 1e-10 * max(1., np.linalg.norm(before, 2))
            minimum_reduction = float(np.linalg.eigvalsh(before - posterior.cov).min())
            minimum_cov = float(np.linalg.eigvalsh(posterior.cov).min())
            trace_change = float(np.trace(posterior.cov) - np.trace(before))
            eigen_change = float(np.linalg.eigvalsh(posterior.cov).max() - np.linalg.eigvalsh(before).max())
            contrast_increase = float((_contrast_variances(posterior) - contrast_before).max())
            assert min(minimum_reduction, minimum_cov) >= -tolerance
            assert max(trace_change, eigen_change, contrast_increase) <= tolerance
            try:
                posterior.update(70, actions, values)
                raise AssertionError("Duplicate raw scores accepted")
            except ValueError:
                pass
            altered = {key: value.copy() for key, value in train.items()}
            altered["Y"][60:] = np.nan
            alternative = GaussianTemplate(altered, history, predictor, refs, variant, noise_fit=noise_fit)
            assert np.array_equal(template.mean, alternative.mean)
            assert np.array_equal(template.cov, alternative.cov)
            assert np.array_equal(template.noise, alternative.noise)
            results.append({"predictor": predictor, "variant": variant, "initial_normal_error": error,
                            "online_block_error": update_error, "covariance_min_eigenvalue": minimum_cov,
                            "reduction_min_eigenvalue": minimum_reduction, "trace_change": trace_change,
                            "max_eigenvalue_change": eigen_change, "max_fixed_contrast_variance_increase": contrast_increase})
        authorized_one = GaussianTemplate(train, history, predictor, refs, training_seeds=(0,), alpha=100 / 3)
        altered = {key: value.copy() for key, value in train.items()}
        altered["Y"][:, 1:] = np.nan
        altered["Y"][60:] = np.nan
        same_one = GaussianTemplate(altered, history, predictor, refs, training_seeds=(0,), alpha=100 / 3)
        assert np.array_equal(authorized_one.mean, same_one.mean)
        assert np.array_equal(authorized_one.cov, same_one.cov)
        for repeats in (1, 2, 3):
            model = GaussianTemplate(train, history, predictor, refs, training_seeds=tuple(range(repeats)), alpha=100 * repeats / 3)
            p = model.clone()
            x = model.X[history]
            a = model.alpha * np.eye(model.dimension) + repeats * np.kron(x.T @ x, model.B.T @ model.B)
            target = train["Y"][np.ix_(history, np.arange(repeats), np.arange(20))].mean(1)
            rhs = (repeats * x.T @ target @ model.B).reshape(-1)
            assert np.max(abs(p.mean - np.linalg.solve(a, rhs))) < 1e-10
    return {"status": "PASS", "new_llm_calls": 0, "results": results,
            "checks": ["all_variant_normal_equations", "N2_exact_E3_mean_covariance_online_noise",
                       "P1_original_B22_observable_equivalence", "raw_score_joint_updates_and_deduplication",
                       "posterior_and_reduction_PSD", "trace_and_maximum_eigenvalue_nonincrease",
                       "fixed_reference_all_combination_contrast_nonincrease", "OOF_question_and_PCA_isolation",
                       "unseen_Y_and_unrequested_seed_invariance", "L1_L2_L3_scaled_alpha_normal_equations"],
            "scientific_success": False}


if __name__ == "__main__":
    started = time.perf_counter()
    result = qualification()
    result["qualification_wall_s"] = time.perf_counter() - started
    print(json.dumps(result, ensure_ascii=False, indent=2))
