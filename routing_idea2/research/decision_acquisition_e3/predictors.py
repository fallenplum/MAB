"""Historical-only linear predictors with matching Gaussian bundle updates.

Historical initialization reproduces the original independent-score Ridge using
all three seeds. Its frozen working posterior becomes the stream prior. Online
likelihoods use history-estimated cross-combination noise for one question and
one seed, consistently in acquisition and actual updates. This is a two-stage
working model, not a single shared-noise likelihood for the entire dataset.
Updates consume only the raw rewards explicitly passed by the caller.
"""

from dataclasses import dataclass
from itertools import combinations
from collections.abc import Mapping

import numpy as np
from scipy.linalg import cho_solve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


TEAMS = tuple(combinations(range(6), 3))


def _setting(config, name, default):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _second_order_basis():
    membership = np.asarray([[float(i in team) for i in range(6)] for team in TEAMS])
    interactions = [membership[:, i] * membership[:, j]
                    for i, j in combinations(range(6), 2)]
    return np.column_stack((np.ones(20), membership, *interactions))


def _indices(values, size, name, *, allow_empty=False):
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise ValueError(name + " must be a one-dimensional integer array")
    result = result.astype(np.int64, copy=True)
    if (not allow_empty and not len(result)) or np.any(result < 0) or np.any(result >= size):
        raise ValueError(name + " is empty or outside its permitted range")
    if len(np.unique(result)) != len(result):
        raise ValueError(name + " contains duplicate indices")
    return result


@dataclass(frozen=True)
class _InitialState:
    mean: np.ndarray
    covariance: np.ndarray
    ref_predictions: np.ndarray
    ref_cross: np.ndarray


class ModelTemplate:
    """Fit preprocessing, online reward covariance and the original Ridge prior.

    P1 orthogonally removes the unobservable null space of the original 22-column
    basis. Its observable posterior equals isotropic Ridge in the original basis.
    P2 uses twenty action heads. P3 adds historical task/subtask indicators to P2.
    All historical scores use independent unit noise, preserving the original
    predictor. The estimated correlated covariance is used only after deployment.
    """

    def __init__(self, train, history_indices, predictor, reference_indices, config=None):
        embeddings = np.asarray(train["E"], dtype=float)
        outcomes = np.asarray(train["Y"], dtype=float)
        tasks = np.asarray(train["task"])
        subtasks = np.asarray(train["sub"])
        if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
            raise ValueError("E must contain finite question embeddings")
        size = len(embeddings)
        if outcomes.shape != (size, 3, 20):
            raise ValueError("Y must have shape (questions, 3, 20)")
        if tasks.shape != (size,) or subtasks.shape != (size,):
            raise ValueError("Task and subtask labels must align with questions")
        self.history = _indices(history_indices, size, "history_indices")
        self.reference_indices = _indices(reference_indices, size, "reference_indices")
        if not np.isin(self.reference_indices, self.history).all():
            raise ValueError("Reference questions must belong to historical inputs")
        historical_rewards = outcomes[self.history]
        if not np.isfinite(historical_rewards).all() or np.any(
                (historical_rewards < 0) | (historical_rewards > 1)):
            raise ValueError("Historical raw rewards must be finite and in [0,1]")
        self.predictor = str(predictor).upper()
        if self.predictor not in ("P1", "P2", "P3"):
            raise ValueError("predictor must be P1, P2 or P3")
        self.alpha = float(_setting(config, "alpha", 100.0))
        if self.alpha != 100.0:
            raise ValueError("The E3 protocol fixes alpha at 100")
        components = int(_setting(config, "pca_components", 16))
        if components != 16 or min(embeddings[self.history].shape) < components:
            raise ValueError("Historical data must support the frozen PCA16 representation")

        self.pca = PCA(n_components=components, svd_solver="full")
        reduced = self.pca.fit_transform(embeddings[self.history])
        self.scaler = StandardScaler().fit(reduced)
        historical_bbh = self.history[tasks[self.history] == "bbh"]
        self.subtask_names = tuple(sorted(set(subtasks[historical_bbh].tolist())))
        self.X = self.transform(embeddings, tasks, subtasks)
        self.original_basis = _second_order_basis()
        self.basis_projection = None
        if self.predictor == "P1":
            _, singular, right = np.linalg.svd(self.original_basis, full_matrices=False)
            tolerance = singular[0] * max(self.original_basis.shape) * np.finfo(float).eps
            rank = int(np.sum(singular > tolerance))
            if rank != 15:
                raise ValueError("The frozen six-choose-three quadratic basis must have rank 15")
            self.basis_projection = right[:rank].T
            self.B = self.original_basis @ self.basis_projection
        else:
            self.B = np.eye(20)
        self.question_dimension = self.X.shape[1]
        self.action_dimension = self.B.shape[1]
        self.dimension = self.question_dimension * self.action_dimension
        self._identity_basis = self.predictor != "P1"

        # Center within question, preserving paired cross-combination generation
        # fluctuations while removing between-question difficulty and model bias.
        residual = historical_rewards - historical_rewards.mean(axis=1, keepdims=True)
        empirical_covariance = np.einsum("nla,nlb->ab", residual, residual) / (2 * len(self.history))
        self.empirical_noise_variances = np.maximum(np.diag(empirical_covariance), 0.01)
        standard_deviation = np.sqrt(self.empirical_noise_variances)
        correlation = empirical_covariance / np.outer(standard_deviation, standard_deviation)
        np.fill_diagonal(correlation, 1.0)
        self.noise = 0.5 * (np.eye(20) + correlation)
        self.noise = (self.noise + self.noise.T) * 0.5
        np.linalg.cholesky(self.noise)
        self.fixed_actions = self._fixed_choices(historical_rewards, tasks, subtasks)

        self.refF = np.einsum(
            "mp,ar->mapr", self.X[self.reference_indices], self.B
        ).reshape(len(self.reference_indices), 20, self.dimension)
        self._historical_means = historical_rewards.mean(axis=1)
        self._gram = self.X[self.history].T @ self.X[self.history]
        self._initial_states = {}
        self._initial_states[False] = self._fit_initial(np.eye(20))

    def transform(self, embeddings, tasks=None, subtasks=None):
        """Transform inputs using historical preprocessing and indicator vocabulary."""
        if isinstance(embeddings, Mapping):
            data = embeddings
            embeddings, tasks, subtasks = data["E"], data["task"], data["sub"]
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
            raise ValueError("Prediction embeddings must be a finite two-dimensional array")
        reduced = self.scaler.transform(self.pca.transform(embeddings))
        columns = [np.ones((len(embeddings), 1)), reduced]
        if self.predictor == "P3":
            tasks, subtasks = np.asarray(tasks), np.asarray(subtasks)
            if tasks.shape != (len(embeddings),) or subtasks.shape != tasks.shape:
                raise ValueError("P3 requires aligned task and subtask input labels")
            bbh = tasks == "bbh"
            indicators = np.column_stack([
                bbh & (subtasks == name) for name in self.subtask_names
            ]) if self.subtask_names else np.empty((len(tasks), 0))
            columns.extend((bbh[:, None], indicators))
        return np.column_stack(columns).astype(float, copy=False)

    def _fixed_choices(self, historical_rewards, tasks, subtasks):
        values = historical_rewards.mean(axis=1)
        global_value = values.mean(axis=0)
        self.global_fixed_action = int(np.argmax(global_value))
        self.task_fixed_actions = {}
        self.subtask_fixed_actions = {}
        historical_tasks = tasks[self.history]
        historical_subtasks = subtasks[self.history]
        for task in sorted(set(tasks.tolist())):
            selected = historical_tasks == task
            mean = values[selected].mean(axis=0) if selected.any() else global_value
            self.task_fixed_actions[str(task)] = int(np.argmax(mean))
        for name in self.subtask_names:
            selected = (historical_tasks == "bbh") & (historical_subtasks == name)
            self.subtask_fixed_actions[str(name)] = int(np.argmax(values[selected].mean(axis=0)))
        choices = np.asarray([self.task_fixed_actions[str(task)] for task in tasks], dtype=np.int64)
        for name, action in self.subtask_fixed_actions.items():
            choices[(tasks == "bbh") & (subtasks == name)] = action
        return choices

    def _fit_initial(self, noise):
        noise_weighted_basis = np.linalg.solve(noise, self.B)
        action_gram = self.B.T @ noise_weighted_basis
        question_values, question_vectors = np.linalg.eigh(self._gram)
        action_values, action_vectors = np.linalg.eigh(action_gram)
        question_values = np.maximum(question_values, 0.0)
        action_values = np.maximum(action_values, 0.0)
        inverse_values = 1.0 / (
            self.alpha + 3.0 * question_values[:, None] * action_values[None, :]
        )
        rhs = 3.0 * self.X[self.history].T @ self._historical_means @ noise_weighted_basis
        rotated_rhs = question_vectors.T @ rhs @ action_vectors
        coefficient = question_vectors @ (rotated_rhs * inverse_values) @ action_vectors.T
        # Two small tensor contractions avoid a dense Kronecker eigensystem GEMM.
        question_blocks = np.einsum(
            "pa,qa,ab->pqb", question_vectors, question_vectors, inverse_values, optimize=True
        )
        covariance = np.einsum(
            "pqb,ib,jb->piqj", question_blocks, action_vectors, action_vectors, optimize=True
        ).reshape(self.dimension, self.dimension)
        covariance = (covariance + covariance.T) * 0.5
        mean = coefficient.reshape(-1)
        references = self.X[self.reference_indices] @ coefficient @ self.B.T
        ref_cross = (self.refF.reshape(-1, self.dimension) @ covariance).reshape(self.refF.shape)
        return _InitialState(mean, covariance, references, ref_cross)

    def clone(self, independent_noise=False):
        """Clone the common historical posterior, optionally ablate online noise.

        Independent noise is a conditional-on-common-prior ablation. It changes
        the subsequent likelihood and acquisition noise together, while keeping
        historical means and covariance identical to the primary experiment.
        """
        if not isinstance(independent_noise, (bool, np.bool_)):
            raise TypeError("independent_noise must be a boolean")
        noise = np.eye(20) if independent_noise else self.noise
        return Posterior(self, self._initial_states[False], noise)


class Posterior:
    """Mutable posterior for one strategy run, with raw-label replay protection."""

    def __init__(self, template, initial, noise):
        self.template = template
        self.mean = initial.mean.copy()
        self.cov = initial.covariance.copy()
        self.noise = noise.copy()
        self.refF = template.refF
        self.ref_predictions = initial.ref_predictions.copy()
        self.ref_cross = initial.ref_cross.copy()
        self.observed = set()
        self.update_count = 0

    def _question(self, q):
        if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)):
            raise ValueError("Question must be an integer row index")
        if not 0 <= q < len(self.template.X):
            raise ValueError("Question row is outside the available inputs")
        return int(q)

    def _actions(self, actions):
        return _indices(actions, 20, "actions")

    def predict(self, q):
        q = self._question(q)
        template = self.template
        coefficient = self.mean.reshape(template.question_dimension, template.action_dimension)
        return template.X[q] @ coefficient @ template.B.T

    def observation(self, q, actions):
        q, actions = self._question(q), self._actions(actions)
        return np.einsum("p,mr->mpr", self.template.X[q], self.template.B[actions]).reshape(
            len(actions), self.template.dimension
        )

    def _cross(self, matrix, q, actions):
        """Compute matrix @ H.T, touching only active question/action columns."""
        template = self.template
        x = template.X[q]
        active = np.flatnonzero(x)
        if template._identity_basis:
            columns = active[:, None] * template.action_dimension + actions[None, :]
            selected = matrix[:, columns]
            return np.einsum("dpm,p->dm", selected, x[active], optimize=True)
        blocks = matrix.reshape(len(matrix), template.question_dimension, template.action_dimension)
        projected = np.einsum("dpr,p->dr", blocks[:, active], x[active], optimize=True)
        return projected @ template.B[actions].T

    def cross(self, q, actions):
        q, actions = self._question(q), self._actions(actions)
        return self._cross(self.cov, q, actions)

    def reference_cross(self, q, actions):
        q, actions = self._question(q), self._actions(actions)
        cross = self._cross(self.ref_cross.reshape(-1, len(self.mean)), q, actions)
        return cross.reshape(*self.ref_predictions.shape, len(actions))

    def project(self, q, actions):
        q, actions = self._question(q), self._actions(actions)
        return self.predict(q)[actions], self._cross(self.cov, q, actions)

    def update(self, q, actions, values):
        q, actions = self._question(q), self._actions(actions)
        values = np.asarray(values, dtype=float)
        if values.shape != actions.shape or not np.isfinite(values).all() or np.any(
                (values < 0) | (values > 1)):
            raise ValueError("Updates require one finite raw reward in [0,1] per action")
        if any((q, int(action)) in self.observed for action in actions):
            raise ValueError("A raw question/action reward cannot be updated twice")
        if any(question == q for question, _ in self.observed):
            raise ValueError("Correlated same-question rewards must arrive in one joint bundle")
        observation = self.observation(q, actions)
        cross = self._cross(self.cov, q, actions)
        predictive = observation @ cross + self.noise[np.ix_(actions, actions)]
        predictive = (predictive + predictive.T) * 0.5
        cholesky = np.linalg.cholesky(predictive)
        innovation = values - self.predict(q)[actions]
        correction = cho_solve((cholesky, True), innovation, check_finite=False)
        whitened_cross = np.linalg.solve(cholesky, cross.T)
        reference_gain = self._cross(self.ref_cross.reshape(-1, len(self.mean)), q, actions)
        whitened_reference = np.linalg.solve(cholesky, reference_gain.T).T
        self.mean += cross @ correction
        self.cov -= whitened_cross.T @ whitened_cross
        self.cov = (self.cov + self.cov.T) * 0.5
        self.ref_predictions += (reference_gain @ correction).reshape(self.ref_predictions.shape)
        self.ref_cross -= (whitened_reference @ whitened_cross).reshape(self.ref_cross.shape)
        self.observed.update((q, int(action)) for action in actions)
        self.update_count += 1


def qualification():
    """Numerical/operator qualification on synthetic arrays, with no LLM calls."""
    rng = np.random.default_rng(20261107)
    n = 64
    train = {
        "E": rng.normal(size=(n, 32)),
        "Y": rng.integers(0, 2, size=(n, 3, 20)).astype(float),
        "task": np.asarray(["gsm8k" if i % 2 else "bbh" for i in range(n)]),
        "sub": np.asarray(["" if i % 2 else ("logic" if i % 4 else "arithmetic") for i in range(n)]),
        "ids": np.asarray(["qualification_" + str(i) for i in range(n)]),
    }
    history, references = np.arange(48), np.arange(5)
    errors = {}
    for predictor in ("P1", "P2", "P3"):
        template = ModelTemplate(train, history, predictor, references, {})
        posterior = template.clone()
        x = template.X[history]
        b = template.B
        # Historical initialization must reproduce the original independent-score
        # Ridge, even though subsequent raw bundles use correlated online noise.
        weighted_b = b
        precision = 100 * np.eye(template.dimension) + 3 * np.kron(x.T @ x, b.T @ weighted_b)
        rhs = (3 * x.T @ train["Y"][history].mean(1) @ weighted_b).reshape(-1)
        explicit_covariance = np.linalg.inv(precision)
        explicit_mean = np.linalg.solve(precision, rhs)
        initial_error = float(np.max(np.abs(posterior.mean - explicit_mean)))
        covariance_error = float(np.max(np.abs(posterior.cov - explicit_covariance)))
        assert max(initial_error, covariance_error) < 1e-10

        q, actions, rewards = 50, np.asarray([0, 1]), np.asarray([1.0, 0.0])
        h = posterior.observation(q, actions)
        assert np.allclose(posterior.cross(q, actions), posterior.cov @ h.T, atol=1e-12)
        assert np.allclose(posterior.reference_cross(q, actions).reshape(-1, 2),
                           posterior.ref_cross.reshape(-1, template.dimension) @ h.T, atol=1e-12)
        block = template.noise[np.ix_(actions, actions)]
        precision += h.T @ np.linalg.solve(block, h)
        rhs += h.T @ np.linalg.solve(block, rewards)
        posterior.update(q, actions, rewards)
        explicit_mean = np.linalg.solve(precision, rhs)
        explicit_covariance = np.linalg.inv(precision)
        update_error = float(np.max(np.abs(posterior.mean - explicit_mean)))
        update_covariance_error = float(np.max(np.abs(posterior.cov - explicit_covariance)))
        reference_error = float(np.max(np.abs(
            posterior.ref_cross.reshape(-1, template.dimension)
            - template.refF.reshape(-1, template.dimension) @ explicit_covariance
        )))
        assert max(update_error, update_covariance_error, reference_error) < 1e-10
        assert np.allclose(posterior.ref_predictions.reshape(-1),
                           template.refF.reshape(-1, template.dimension) @ explicit_mean, atol=1e-10)
        try:
            posterior.update(q, actions, rewards)
            raise AssertionError("Duplicate raw-score update was accepted")
        except ValueError:
            pass

        # The diagnostic policy may legally reveal all twenty scores at once.
        full_actions = np.arange(20)
        full_q, full_values = 51, train["Y"][51, 0]
        full_h = posterior.observation(full_q, full_actions)
        precision += full_h.T @ np.linalg.solve(template.noise, full_h)
        rhs += full_h.T @ np.linalg.solve(template.noise, full_values)
        posterior.update(full_q, full_actions, full_values)
        full_mean = np.linalg.solve(precision, rhs)
        full_covariance = np.linalg.inv(precision)
        full_error = float(np.max(np.abs(posterior.mean - full_mean)))
        full_covariance_error = float(np.max(np.abs(posterior.cov - full_covariance)))
        assert max(full_error, full_covariance_error) < 1e-10
        assert np.allclose(posterior.ref_cross.reshape(-1, template.dimension),
                           template.refF.reshape(-1, template.dimension) @ full_covariance, atol=1e-10)

        changed = {key: value.copy() for key, value in train.items()}
        changed["Y"][48:] = 1 - changed["Y"][48:]
        changed["E"][48:] += 100
        alternative = ModelTemplate(changed, history, predictor, references, {})
        assert np.array_equal(template.noise, alternative.noise)
        assert np.array_equal(template.clone().mean, alternative.clone().mean)
        assert np.array_equal(template.clone().cov, alternative.clone().cov)
        assert np.array_equal(template.fixed_actions, alternative.fixed_actions)
        independent = template.clone(independent_noise=True)
        assert np.array_equal(independent.noise, np.eye(20))
        assert np.array_equal(independent.mean, template.clone().mean)
        assert np.array_equal(independent.cov, template.clone().cov)
        errors[predictor] = {
            "dimension": template.dimension,
            "initial_mean_error": initial_error,
            "initial_covariance_error": covariance_error,
            "joint_update_mean_error": update_error,
            "joint_update_covariance_error": update_covariance_error,
            "incremental_reference_covariance_error": reference_error,
            "twenty_score_update_mean_error": full_error,
            "twenty_score_update_covariance_error": full_covariance_error,
        }
        if predictor == "P1":
            original = template.original_basis
            weighted_original = original
            original_precision = 100 * np.eye(x.shape[1] * 22) + 3 * np.kron(
                x.T @ x, original.T @ weighted_original
            )
            original_rhs = (3 * x.T @ train["Y"][history].mean(1) @ weighted_original).reshape(-1)
            original_mean = np.linalg.solve(original_precision, original_rhs)
            original_covariance = np.linalg.inv(original_precision)
            original_h = np.einsum("p,ar->apr", template.X[q], original).reshape(20, -1)
            reduced = template.clone()
            reduced_h = reduced.observation(q, np.arange(20))
            mean_error = float(np.max(np.abs(original_h @ original_mean - reduced.predict(q))))
            variance_error = float(np.max(np.abs(
                original_h @ original_covariance @ original_h.T
                - reduced_h @ reduced.cov @ reduced_h.T
            )))
            assert max(mean_error, variance_error) < 1e-10
            errors[predictor]["original_B22_mean_error"] = mean_error
            errors[predictor]["original_B22_predictive_covariance_error"] = variance_error
    return {"status": "PASS", "llm_calls": 0, "errors": errors,
            "checks": ["original_independent_score_historical_normal_equations", "joint_correlated_raw_score_update",
                       "incremental_reference_updates", "sparse_feature_cross_products",
                       "excluded_outcome_and_embedding_invariance", "duplicate_update_rejection",
                       "twenty_score_diagnostic_bundle", "common_initial_posterior_noise_ablation",
                       "P1_original_isotropic_B22_equivalence"]}


if __name__ == "__main__":
    import json
    import time
    started = time.perf_counter()
    result = qualification()
    result["qualification_wall_s"] = time.perf_counter() - started
    print(json.dumps(result, ensure_ascii=False, indent=2))
