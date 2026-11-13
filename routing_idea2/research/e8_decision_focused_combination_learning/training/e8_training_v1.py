"""History-only E8 linear objectives and nested model selection.

Every callable here accepts only its legal historical region.  Evaluation
rewards are intentionally absent from the training and prediction interfaces.
The preprocessing and penalized constant are imported from the A1/A2 recipe.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

from e8_assets import exact_argmax, fit_features, inner_splits, matched_mean_loss_lambda, ridge_weights


K = 20
PAIR_I, PAIR_J = np.array(list(itertools.combinations(range(K), 2)), dtype=int).T
PAIR_COUNT = len(PAIR_I)
INCIDENCE = np.zeros((PAIR_COUNT, K))
INCIDENCE[np.arange(PAIR_COUNT), PAIR_I] = 1.0
INCIDENCE[np.arange(PAIR_COUNT), PAIR_J] = -1.0

DEFAULT_SETTINGS = {
    "raw_alpha": 100.0,
    "lambda_multipliers": [0.1, 1.0, 10.0],
    "ranknet_pair_denominator": 190,
    "listnet_target_temperatures": [0.1, 0.3, 1.0],
    "listnet_prediction_temperature": 1.0,
    "competition_alphas": [0.0, 2.0, 5.0],
    "primary_top_k": 3,
    "lbfgs_max_iterations": 2000,
    "lbfgs_ftol": 1e-12,
    "lbfgs_gtol": 1e-7,
    "osqp_eps_abs": 1e-8,
    "osqp_eps_rel": 1e-8,
    "osqp_max_iterations": 200000,
    "osqp_polishing": True,
    "inner_selection_tie_tolerance": 1e-12,
    "permutation_seed": 2026111317,
    "oof_seed_offset": 10000,
}


def array_hash(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _rss_bytes():
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def _process_peak_rss_bytes():
    try:
        import psutil
        return getattr(psutil.Process().memory_info(), "peak_wset", None)
    except ImportError:
        return None


def _means(rewards):
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim == 3 and rewards.shape[1:] == (3, K):
        rewards = rewards.mean(axis=1)
    if rewards.ndim != 2 or rewards.shape[1] != K:
        raise ValueError("One row per question with 20 action rewards is required")
    if not np.isfinite(rewards).all() or np.any((rewards < 0) | (rewards > 1)):
        raise ValueError("Historical action rewards must be finite in [0,1]")
    return rewards


def true_optimal_weights(rewards, tie_rule="lexicographic"):
    rewards = _means(rewards)
    if tie_rule == "lexicographic":
        weights = np.zeros_like(rewards)
        weights[np.arange(len(rewards)), exact_argmax(rewards)] = 1.0
        return weights
    if tie_rule == "uniform":
        maxima = rewards == rewards.max(axis=1, keepdims=True)
        return maxima / maxima.sum(axis=1, keepdims=True)
    raise ValueError("Unknown true-optimum tie rule")


def ranknet_value_gradient(coefficient, phi, rewards, regularization):
    """Mean question loss, fixed 190 denominator and |true gap| weights."""
    scores = phi @ coefficient
    reward_gaps = rewards[:, PAIR_I] - rewards[:, PAIR_J]
    score_gaps = scores[:, PAIR_I] - scores[:, PAIR_J]
    signs, weights = np.sign(reward_gaps), np.abs(reward_gaps)
    signed_margin = signs * score_gaps
    loss = np.sum(weights * np.logaddexp(0.0, -signed_margin)) / (len(phi) * PAIR_COUNT)
    pair_gradient = -weights * signs * expit(-signed_margin) / (len(phi) * PAIR_COUNT)
    gradient = phi.T @ (pair_gradient @ INCIDENCE) + 2.0 * regularization * coefficient
    return float(loss + regularization * np.sum(coefficient**2)), gradient


def listnet_value_gradient(coefficient, phi, rewards, regularization,
                          target_temperature, prediction_temperature=1.0):
    target = softmax(rewards / target_temperature, axis=1)
    logits = (phi @ coefficient) / prediction_temperature
    log_probability = logits - logsumexp(logits, axis=1, keepdims=True)
    loss = -np.mean(np.sum(target * log_probability, axis=1))
    score_gradient = (np.exp(log_probability) - target) / (len(phi) * prediction_temperature)
    gradient = phi.T @ score_gradient + 2.0 * regularization * coefficient
    return float(loss + regularization * np.sum(coefficient**2)), gradient


def spo_plus_value_subgradient(coefficient, phi, rewards, regularization,
                             true_tie_rule="lexicographic"):
    """Exact reward-maximization SPO+, using a lexicographic surrogate max."""
    scores = phi @ coefficient
    true_weights = true_optimal_weights(rewards, true_tie_rule)
    adversarial_scores = 2.0 * scores - rewards
    adversarial_action = exact_argmax(adversarial_scores)
    adversarial_weights = np.zeros_like(rewards)
    adversarial_weights[np.arange(len(phi)), adversarial_action] = 1.0
    losses = (adversarial_scores[np.arange(len(phi)), adversarial_action]
              - 2.0 * np.sum(scores * true_weights, axis=1) + rewards.max(axis=1))
    subgradient = (2.0 / len(phi)) * phi.T @ (adversarial_weights - true_weights)
    subgradient += 2.0 * regularization * coefficient
    return float(np.mean(losses) + regularization * np.sum(coefficient**2)), subgradient


def competition_weights(oof_scores, alpha, top_k=3):
    """Reward-blind top sets from legal question-level OOF M0 scores."""
    oof_scores = np.asarray(oof_scores, dtype=float)
    if oof_scores.ndim != 2 or oof_scores.shape[1] != K or not 1 <= top_k < K:
        raise ValueError("A valid OOF score matrix and candidate count are required")
    top = np.argsort(-oof_scores, axis=1, kind="stable")[:, :top_k]
    indicator = np.zeros_like(oof_scores, dtype=bool)
    indicator[np.arange(len(top))[:, None], top] = True
    emphasized = indicator[:, PAIR_I] | indicator[:, PAIR_J]
    raw = 1.0 + float(alpha) * emphasized
    return raw / raw.mean(axis=1, keepdims=True), top


def fit_gap_ridge(phi, rewards, regularization, weights):
    """Solve the 340-dimensional convex quadratic exactly by Cholesky."""
    n, dimension = phi.shape
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n, PAIR_COUNT) or np.any(weights <= 0):
        raise ValueError("One positive normalized pair-weight vector per question is required")
    laplacian = np.einsum("pa,np,pb->nab", INCIDENCE, weights, INCIDENCE, optimize=True)
    question_outer = np.einsum("ni,nj->nij", phi, phi).reshape(n, dimension**2)
    normal = (question_outer.T @ laplacian.reshape(n, K*K)).reshape(dimension, dimension, K, K)
    normal = normal.transpose(0, 2, 1, 3).reshape(dimension*K, dimension*K)
    normal /= n * K*K
    normal.flat[::len(normal)+1] += regularization
    right = (phi.T @ np.einsum("na,nab->nb", rewards, laplacian)) / (n*K*K)
    coefficient = cho_solve(cho_factor(normal, lower=True, check_finite=False), right.reshape(-1), check_finite=False)
    coefficient = coefficient.reshape(dimension, K)
    errors = (phi @ coefficient - rewards) @ INCIDENCE.T
    objective = float(np.mean(np.sum(weights * errors**2, axis=1) / (K*K))
                      + regularization*np.sum(coefficient**2))
    gradient = 2.0*(normal @ coefficient.reshape(-1)-right.reshape(-1))
    return coefficient, dict(status="solved_exact_quadratic", success=True,
                             objective=objective, gradient_infinity_norm=float(np.max(np.abs(gradient))),
                             normal_condition_number=float(np.linalg.cond(normal)), iterations=1)


def fit_spo_qp(phi, rewards, regularization, tie_rule, settings):
    """Exact epigraph QP for SPO+; no smooth surrogate is substituted."""
    import osqp
    n, dimension = phi.shape
    true_weights = true_optimal_weights(rewards, tie_rule)
    coefficient_size = dimension*K
    quadratic = sparse.diags(np.r_[np.full(coefficient_size, 2*regularization), np.zeros(n)], format="csc")
    linear_coefficient = -(2.0/n)*(phi.T @ true_weights)
    linear = np.r_[linear_coefficient.reshape(-1, order="F"), np.full(n, 1.0/n)]
    action_blocks = sparse.kron(sparse.eye(K, format="csc"), 2.0*sparse.csc_matrix(phi), format="csc")
    epigraph_blocks = sparse.vstack([-sparse.eye(n, format="csc")]*K, format="csc")
    constraint = sparse.hstack([action_blocks, epigraph_blocks], format="csc")
    upper = rewards.reshape(-1, order="F")
    solver = osqp.OSQP()
    solver.setup(P=quadratic, q=linear, A=constraint, l=np.full(n*K, -np.inf), u=upper,
                 verbose=False, eps_abs=settings["osqp_eps_abs"], eps_rel=settings["osqp_eps_rel"],
                 max_iter=settings["osqp_max_iterations"], polishing=settings["osqp_polishing"],
                 adaptive_rho=True, warm_starting=False)
    solved = solver.solve(raise_error=False)
    if solved.x is None:
        raise RuntimeError("The exact SPO+ epigraph solver returned no primal point")
    coefficient = solved.x[:coefficient_size].reshape(dimension, K, order="F")
    objective, _ = spo_plus_value_subgradient(coefficient, phi, rewards, regularization, tie_rule)
    primal_residual = float(solved.info.prim_res)
    dual_residual = float(solved.info.dual_res)
    constraint_violation = float(np.maximum(constraint @ solved.x-upper, 0).max(initial=0.0))
    epigraph_objective = float(solved.info.obj_val + rewards.max(axis=1).mean())
    success = solved.info.status == "solved"
    scores = phi @ coefficient
    per_question_loss = (np.max(2*scores-rewards, axis=1)
                         -2*np.sum(scores*true_weights, axis=1)+rewards.max(axis=1))
    all_tied = np.all(rewards == rewards[:, :1], axis=1)
    return coefficient, dict(status=str(solved.info.status), success=success,
                             objective=objective, epigraph_objective=epigraph_objective,
                             objective_discrepancy=abs(objective-epigraph_objective),
                             primal_residual=primal_residual, dual_residual=dual_residual,
                             constraint_violation=constraint_violation,
                             iterations=int(solved.info.iter), solver="OSQP", solver_version=osqp.__version__,
                             true_tie_rule=tie_rule, surrogate_tie_rule="lexicographic",
                             all_twenty_tied_questions=int(all_tied.sum()),
                             all_twenty_tied_loss_contribution=float(np.sum(per_question_loss[all_tied])/n),
                             true_optimum_tie_fraction=float(np.mean((rewards == rewards.max(axis=1, keepdims=True)).sum(axis=1)>1)),
                             variables=coefficient_size+n, constraints=n*K)


def fit_coefficients(phi, rewards, method, parameters, settings=None, oof_scores=None,
                     top_k=3, true_tie_rule="lexicographic"):
    settings = dict(DEFAULT_SETTINGS if settings is None else settings)
    phi, rewards = np.asarray(phi, dtype=float), _means(rewards)
    regularization = matched_mean_loss_lambda(len(phi), settings["raw_alpha"]) * parameters["lambda_multiplier"]
    cpu_start, wall_start, rss_before = time.process_time(), time.perf_counter(), _rss_bytes()
    if method in ("M0_original", "M0_tuned"):
        coefficient = ridge_weights(phi, rewards, settings["raw_alpha"]*parameters["lambda_multiplier"])
        residual = phi @ coefficient-rewards
        objective = np.mean(residual**2)+regularization*np.sum(coefficient**2)
        gradient = 2.0*phi.T@residual/(len(phi)*K)+2.0*regularization*coefficient
        diagnostics = dict(status="solved_svd_ridge", success=True, objective=float(objective),
                           gradient_infinity_norm=float(np.max(np.abs(gradient))), iterations=1)
    elif method == "M3":
        if oof_scores is None:
            raise ValueError("M3 requires reward-blind historical OOF competitor scores")
        weights, top = competition_weights(oof_scores, parameters["alpha"], top_k)
        coefficient, diagnostics = fit_gap_ridge(phi, rewards, regularization, weights)
        diagnostics.update(pair_weights_sha256=array_hash(weights), top_actions_sha256=array_hash(top),
                           top_k=int(top_k), alpha=float(parameters["alpha"]))
    elif method in ("M1", "M2"):
        start = ridge_weights(phi, rewards, settings["raw_alpha"])
        def objective(flat):
            coefficient = flat.reshape(phi.shape[1], K)
            if method == "M1":
                value, gradient = ranknet_value_gradient(coefficient, phi, rewards, regularization)
            else:
                value, gradient = listnet_value_gradient(coefficient, phi, rewards, regularization,
                                                        parameters["target_temperature"],
                                                        settings["listnet_prediction_temperature"])
            return value, gradient.reshape(-1)
        solved = minimize(objective, start.reshape(-1), method="L-BFGS-B", jac=True,
                          options=dict(maxiter=settings["lbfgs_max_iterations"],
                                       ftol=settings["lbfgs_ftol"], gtol=settings["lbfgs_gtol"]))
        coefficient = solved.x.reshape(phi.shape[1], K)
        diagnostics = dict(status=str(solved.message), success=bool(solved.success),
                           objective=float(solved.fun), iterations=int(solved.nit), evaluations=int(solved.nfev),
                           gradient_infinity_norm=float(np.max(np.abs(solved.jac))))
    elif method == "M4":
        coefficient, diagnostics = fit_spo_qp(phi, rewards, regularization, true_tie_rule, settings)
    else:
        raise ValueError("Unknown frozen training objective")
    diagnostics.update(regularization=float(regularization), lambda_multiplier=float(parameters["lambda_multiplier"]),
                       n_questions=len(phi), feature_dimension=phi.shape[1], action_dimension=K,
                       coefficient_sha256=array_hash(coefficient),
                       cpu_seconds=time.process_time()-cpu_start, wall_seconds=time.perf_counter()-wall_start,
                       process_rss_before_bytes=rss_before, process_rss_after_bytes=_rss_bytes(),
                       process_peak_working_set_bytes=_process_peak_rss_bytes())
    if not diagnostics["success"]:
        raise RuntimeError("The declared optimizer failed its fixed stopping criteria: " + json.dumps(diagnostics))
    return coefficient, diagnostics


def permuted_rewards(data, seed):
    """Permute whole original question blocks inside this legal region only."""
    rng = np.random.default_rng(int(seed))
    mapping = np.arange(len(data["E"]))
    groups = np.array([str(task)+"/"+str(subtask) for task, subtask in zip(data["task"], data["sub"])])
    for group in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == group)
        mapping[rows] = rng.permutation(rows)
    return np.asarray(data["Y"])[mapping], mapping


def subset_data(data, rows):
    return {key: np.asarray(value)[rows] for key, value in data.items()
            if key in ("E", "Y", "task", "sub", "ids")}


def _region_data(data, permutation, seed):
    if not permutation:
        return data, np.arange(len(data["E"]))
    rewards, mapping = permuted_rewards(data, seed)
    region = dict(data)
    region["Y"] = rewards
    return region, mapping


def legal_oof_m0(data, seed, settings, permutation=False):
    """Fit every competitor recommendation without its held question reward."""
    rows = np.arange(len(data["E"]))
    scores = np.empty((len(rows), K))
    records = []
    for fold, (fit_rows, held_rows) in enumerate(inner_splits(data, rows, seed)):
        original_fit = subset_data(data, fit_rows)
        fitting, mapping = _region_data(original_fit, permutation, settings["permutation_seed"]+seed+fold)
        feature_map = fit_features(fitting["E"])
        phi = feature_map.phi(fitting["E"])
        coefficient = ridge_weights(phi, fitting["Y"], settings["raw_alpha"])
        scores[held_rows] = feature_map.phi(np.asarray(data["E"])[held_rows]) @ coefficient
        records.append(dict(fold=fold, fitting_rows=fit_rows, held_rows=held_rows,
                            fitting_reward_hash=array_hash(fitting["Y"]),
                            local_permutation=mapping, pca_mean_hash=array_hash(feature_map.pca.mean_),
                            coefficient_hash=array_hash(coefficient)))
    return scores, records


def candidate_grid(method, settings):
    multipliers = settings["lambda_multipliers"]
    if method == "M0_original":
        return [dict(lambda_multiplier=1.0)]
    if method == "M2":
        return [dict(lambda_multiplier=multiplier, target_temperature=temperature)
                for multiplier in multipliers for temperature in settings["listnet_target_temperatures"]]
    if method == "M3":
        return [dict(lambda_multiplier=multiplier, alpha=alpha)
                for multiplier in multipliers for alpha in settings["competition_alphas"]]
    return [dict(lambda_multiplier=multiplier) for multiplier in multipliers]


def _candidate_preference(parameters):
    return (parameters["lambda_multiplier"], -parameters.get("alpha", 0.),
            parameters.get("target_temperature", 0.))


def _balanced_accuracy(selected_reward, tasks):
    tasks = np.asarray(tasks)
    return float(np.mean([np.mean(selected_reward[tasks == task])
                          for task in ("gsm8k", "bbh") if np.any(tasks == task)]))


@dataclass
class LinearRouter:
    feature_map: object
    coefficient: np.ndarray
    method: str
    parameters: dict

    def predict(self, embeddings):
        return self.feature_map.phi(embeddings) @ self.coefficient

    def arrays(self):
        return dict(coefficient=self.coefficient, **self.feature_map.arrays())


def train_fold(history, method, seed, *, settings=None, parameters=None,
               top_k=3, true_tie_rule="lexicographic", permutation=False):
    """Select on historical inner folds, then fit all legal historical rows.

    Supplying parameters freezes configuration and skips parameter selection.
    This is used for Top-5, uniform true ties, and matched shuffled controls.
    Inner OOF scores are still produced for historical-only affine calibration.
    """
    settings = dict(DEFAULT_SETTINGS if settings is None else settings)
    n = len(history["E"])
    if np.asarray(history["E"]).shape != (n, 384):
        raise ValueError("The E8 feature contract requires full MiniLM 384-vectors")
    if np.asarray(history["Y"]).shape != (n, 3, K):
        raise ValueError("Three frozen generation repeats per historical question are required")
    original_means = _means(history["Y"])
    candidates = candidate_grid(method, settings) if parameters is None else [dict(parameters)]
    rows = np.arange(n)
    inner = inner_splits(history, rows, seed)
    predictions = np.empty((len(candidates), n, K))
    candidate_records = [[] for _ in candidates]
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    for fold, (fit_rows, held_rows) in enumerate(inner):
        original_fit = subset_data(history, fit_rows)
        fitting, mapping = _region_data(original_fit, permutation, settings["permutation_seed"]+seed+fold)
        feature_map = fit_features(fitting["E"])
        phi = feature_map.phi(fitting["E"])
        held_phi = feature_map.phi(np.asarray(history["E"])[held_rows])
        rewards = _means(fitting["Y"])
        oof, competitor_records = (legal_oof_m0(fitting, seed+settings["oof_seed_offset"]+fold,
                                               settings, permutation=False)
                                   if method == "M3" else (None, []))
        for index, candidate in enumerate(candidates):
            coefficient, diagnostics = fit_coefficients(phi, rewards, method, candidate, settings,
                                                        oof, top_k, true_tie_rule)
            predictions[index, held_rows] = held_phi @ coefficient
            candidate_records[index].append(dict(fold=fold, fitting_rows=fit_rows, held_rows=held_rows,
                                                  local_permutation=mapping,
                                                  fitting_reward_hash=array_hash(fitting["Y"]),
                                                  competitor_oof=competitor_records, diagnostics=diagnostics))
    accuracies = []
    for scores in predictions:
        actions = exact_argmax(scores)
        accuracies.append(_balanced_accuracy(original_means[np.arange(n), actions], history["task"]))
    best_accuracy = max(accuracies)
    tied = [i for i, accuracy in enumerate(accuracies)
            if best_accuracy-accuracy <= settings["inner_selection_tie_tolerance"]]
    chosen = max(tied, key=lambda i: _candidate_preference(candidates[i]))
    selected = candidates[chosen]
    fitting, mapping = _region_data(history, permutation, settings["permutation_seed"]+seed+100)
    feature_map = fit_features(fitting["E"])
    phi, rewards = feature_map.phi(fitting["E"]), _means(fitting["Y"])
    oof, competitor_records = (legal_oof_m0(fitting, seed+settings["oof_seed_offset"]+100,
                                           settings, permutation=False)
                               if method == "M3" else (None, []))
    coefficient, diagnostics = fit_coefficients(phi, rewards, method, selected, settings,
                                                oof, top_k, true_tie_rule)
    model = LinearRouter(feature_map, coefficient, method, selected)
    arrays = dict(inner_oof_scores=predictions[chosen], original_train_reward_means=original_means,
                  fitting_reward_means=rewards, fitting_permutation=mapping,
                  fitted_train_scores=phi@coefficient,
                  inner_fold=np.empty(n, dtype=int))
    for fold, (_, held_rows) in enumerate(inner):
        arrays["inner_fold"][held_rows] = fold
    if oof is not None:
        pair_weights, top = competition_weights(oof, selected["alpha"], top_k)
        arrays.update(competitor_oof_scores=oof, competition_pair_weights=pair_weights,
                      competition_top_actions=top)
    logs = dict(method=method, selected_parameters=selected,
                candidates=[dict(parameters=candidate, inner_balanced_accuracy=accuracies[i],
                                 fits=candidate_records[i]) for i, candidate in enumerate(candidates)],
                parameter_selection_performed=parameters is None and len(candidates)>1,
                frozen_configuration_sensitivity=parameters is not None,
                final_diagnostics=diagnostics, final_competitor_oof=competitor_records,
                permutation=bool(permutation), permutation_recalibrated=False,
                true_tie_rule=true_tie_rule, top_k=int(top_k),
                train_input_hash=array_hash(history["E"]), train_rewards_hash=array_hash(history["Y"]),
                final_fitting_rewards_hash=array_hash(fitting["Y"]),
                inner_oof_score_hash=array_hash(predictions[chosen]),
                final_state_hash=array_hash(coefficient), settings=settings,
                total_cpu_seconds=time.process_time()-cpu_start,
                total_wall_seconds=time.perf_counter()-wall_start)
    return model, arrays, logs
