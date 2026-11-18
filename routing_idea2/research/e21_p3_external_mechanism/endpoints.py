"""P3 matched all-action-tie ablations for TSPO+ and top-one ListNet."""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import logsumexp


REGULARIZATION = 100 / (3 * 480 * 20)
TAU = 1e-8
QUALIFICATION_TOLERANCE = 1e-7
LISTNET_GRADIENT_TOLERANCE = 1e-10


def uniform_policy(scores, tolerance=TAU):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim < 2 or not np.isfinite(scores).all() or not (0 <= tolerance < np.inf):
        raise ValueError("Uniform argmax requires finite score rows and a nonnegative tolerance")
    support = scores >= scores.max(axis=-1, keepdims=True) - tolerance
    return support / support.sum(axis=-1, keepdims=True)


def _validate_training(phi, rewards, regularization):
    phi = np.asarray(phi, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if (phi.ndim != 2 or rewards.ndim != 2 or rewards.shape[0] != phi.shape[0]
            or min(phi.shape) < 1 or rewards.shape[1] < 2):
        raise ValueError("Training needs matching nonempty question rows and at least two actions")
    if (not np.isfinite(phi).all() or not np.all(phi[:, 0] == 1)
            or not np.isfinite(rewards).all() or np.any((rewards < 0) | (rewards > 1))):
        raise ValueError("Use finite training features with first-column intercept and rewards in [0,1]")
    if not 0 < regularization < np.inf:
        raise ValueError("The constant feature is regularized along with all other coefficients")
    return phi, rewards


def _listnet_terms(coefficient, phi, target, target_log, tied, regularization, beta):
    scores = phi @ coefficient
    log_norm = logsumexp(scores, axis=1, keepdims=True)
    prediction_log = scores - log_norm
    prediction = np.exp(prediction_log)
    data_losses = np.sum(target * (target_log - prediction_log), axis=1)
    weights = np.where(tied, beta, 1.0)
    gradient = phi.T @ (weights[:, None] * (prediction - target)) / len(phi)
    gradient += 2 * regularization * coefficient
    penalty = regularization * float(np.sum(coefficient**2))
    without_ties = float(np.sum(data_losses[~tied]) / len(phi))
    tie_loss = log_norm[:, 0] - scores.mean(axis=1) - np.log(scores.shape[1])
    contraction = float(np.sum(tie_loss[tied]) / len(phi))
    data_loss = float(weights @ data_losses / len(phi))
    components = dict(
        objective=data_loss + penalty, data_loss=data_loss, regularization_penalty=penalty,
        F0=without_ties + penalty, J_T=contraction,
        F1=float(data_losses.mean()) + penalty,
        identity_F1_minus_F0_minus_JT=float(data_losses.mean() - without_ties - contraction),
        all_action_tied_questions=int(tied.sum()), non_all_tied_questions=int((~tied).sum()),
        original_question_denominator=len(phi), effective_question_weight_sum=float(weights.sum()),
    )
    return components, gradient, prediction, weights


def _fit_listnet(phi, rewards, regularization, beta, permutation):
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    action_count = rewards.shape[1]
    if permutation is None:
        permutation = np.arange(action_count)
    numeric_rewards = rewards[:, permutation]
    target_log = numeric_rewards - logsumexp(numeric_rewards, axis=1, keepdims=True)
    target = np.exp(target_log)
    tied = np.all(numeric_rewards == numeric_rewards[:, :1], axis=1)
    shape = (phi.shape[1], action_count)

    def terms(flat):
        return _listnet_terms(flat.reshape(shape), phi, target, target_log, tied, regularization, beta)

    def objective(flat):
        components, gradient, _, _ = terms(flat)
        return components["objective"], gradient.ravel()

    settings = {"gtol": 1e-12, "ftol": 1e-15, "maxiter": 2000, "maxls": 50, "maxcor": 20}
    fit = minimize(objective, np.zeros(np.prod(shape)), jac=True, method="L-BFGS-B", options=settings)
    if not fit.success:
        raise RuntimeError("ListNet L-BFGS-B failed: " + str(fit.message))
    flat = fit.x
    initial_gradient = float(np.linalg.norm(fit.jac, ord=np.inf))
    refinements = 0
    for _ in range(3):
        _, gradient, probability, weights = terms(flat)
        if np.abs(gradient).max() <= LISTNET_GRADIENT_TOLERANCE:
            break
        action_covariance = (-probability[:, :, None] * probability[:, None, :])
        index = np.arange(action_count)
        action_covariance[:, index, index] += probability
        hessian = np.einsum("ip,iq,iab,i->paqb", phi, phi, action_covariance, weights,
                            optimize=True).reshape(flat.size, flat.size) / len(phi)
        hessian.flat[::flat.size + 1] += 2 * regularization
        step = solve(hessian, gradient.ravel(), assume_a="pos")
        previous_loss = objective(flat)[0]
        if objective(flat - step)[0] > previous_loss + 1e-13:
            raise RuntimeError("ListNet Newton refinement increased the objective")
        flat = flat - step
        refinements += 1

    numeric_coefficient = flat.reshape(shape)
    coefficient = np.empty_like(numeric_coefficient)
    coefficient[:, permutation] = numeric_coefficient
    report, gradient, _, _ = terms(flat)
    gradient_inf = float(np.abs(gradient).max())
    if (gradient_inf > LISTNET_GRADIENT_TOLERANCE
            or abs(report["identity_F1_minus_F0_minus_JT"]) > QUALIFICATION_TOLERANCE):
        raise RuntimeError("ListNet numerical qualification failed")
    return coefficient, dict(
        status="Solved", solver="L-BFGS-B + qualified Newton refinement", **report,
        iterations=int(fit.nit), solver_settings=settings, message=str(fit.message),
        lbfgs_gradient_inf=initial_gradient, gradient_inf=gradient_inf,
        strong_convexity_objective_gap_bound=float(np.sum(gradient**2) / (4 * regularization)),
        newton_refinements=refinements, coefficient_dimension=flat.size,
        feature_dimension=shape[0], action_dimension=shape[1], regularization=regularization,
        strong_convexity_modulus=2 * regularization, beta=beta, gamma=0.0,
        numeric_to_physical_permutation=permutation.tolist(),
        coefficients_returned_in_physical_action_order=True,
        normalization="1/N original questions retained for both beta branches",
        target="unit-temperature softmax of empirical mean; target entropy subtracted from CE",
        penalized_constant_feature=True, cpu_seconds=time.process_time() - started_cpu,
        wall_seconds=time.perf_counter() - started_wall,
    )


def _fit_tspo(phi, rewards, regularization, beta, permutation):
    source = Path(__file__).resolve().parent.parent / "e19_symmetric_tie_constraint"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from weighted_objectives import fit_weighted_tspo

    return fit_weighted_tspo(phi, rewards, regularization, beta, permutation=permutation)


def fit_endpoint_pair(family, phi, rewards, *, regularization=REGULARIZATION,
                      reverse=False, audit_features=None):
    """Return B=beta1 and C=beta0 coefficients and training-only qualifications."""
    fitter = {"tspo": _fit_tspo, "listnet": _fit_listnet}.get(family)
    if fitter is None:
        raise ValueError("Expected family 'tspo' or 'listnet'")
    phi, rewards = _validate_training(phi, rewards, regularization)
    if audit_features is not None:
        audit_features = np.asarray(audit_features, dtype=np.float64)
        if (audit_features.ndim != 2 or audit_features.shape[1] != phi.shape[1]
                or len(audit_features) == 0 or not np.isfinite(audit_features).all()
                or not np.all(audit_features[:, 0] == 1)):
            raise ValueError("Mapping audit accepts only finite evaluation features with matching intercept")
    fits = [fitter(phi, rewards, regularization, beta, None) for beta in (1.0, 0.0)]
    coefficients, diagnostics = zip(*fits)
    f0_increase = diagnostics[0]["F0"] - diagnostics[1]["F0"]
    contraction_decrease = diagnostics[1]["J_T"] - diagnostics[0]["J_T"]
    allowed = QUALIFICATION_TOLERANCE
    checks = dict(
        lower_bound=f0_increase >= -allowed,
        upper_bound=f0_increase <= contraction_decrease + allowed,
        contraction_direction=contraction_decrease >= -allowed,
        decomposition=all(abs(row["identity_F1_minus_F0_minus_JT"]) <= allowed for row in diagnostics),
    )
    mappings = {}
    if reverse:
        reverse_order = np.arange(rewards.shape[1] - 1, -1, -1)
        for branch, beta, coefficient, diag in zip(("B", "C"), (1.0, 0.0), coefficients, diagnostics):
            mapped, mapped_diag = fitter(phi, rewards, regularization, beta, reverse_order)
            features = phi if audit_features is None else np.vstack((phi, audit_features))
            scores = features @ coefficient
            mapped_scores = features @ mapped
            delta = scores - mapped_scores
            centered_error = float(np.abs(delta - delta.mean(axis=1, keepdims=True)).max())
            probability_error = float(np.abs(uniform_policy(scores) - uniform_policy(mapped_scores)).max())
            objective_error = abs(diag["objective"] - mapped_diag["objective"])
            checks[branch + "_mapped_centered"] = centered_error <= allowed
            checks[branch + "_mapped_probabilities"] = probability_error <= allowed
            checks[branch + "_mapped_objective"] = objective_error <= allowed
            mappings[branch] = dict(centered_score_max_error=centered_error,
                                    probability_max_error=probability_error,
                                    objective_error=objective_error, fit=mapped_diag)
    if not all(checks.values()):
        failing = [key for key, passed in checks.items() if not passed]
        detail = {key: {metric: value for metric, value in row.items() if metric != "fit"}
                  for key, row in mappings.items()}
        raise RuntimeError(f"Endpoint qualification failed: {failing}; mapping diagnostics={detail}")
    return np.asarray(coefficients), dict(
        status="PASS", checks=checks, family=family, denominator=len(phi), regularization=regularization,
        F0_B_minus_F0_C=f0_increase, JT_C_minus_JT_B=contraction_decrease,
        fit_count=2 + 2 * bool(reverse), B=diagnostics[0], C=diagnostics[1], reverse=mappings,
        qualification_tolerance=allowed, deployment_tau=TAU,
    )
