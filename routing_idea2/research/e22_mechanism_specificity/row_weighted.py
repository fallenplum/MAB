"""Retained-question weighted TSPO+ and ListNet fits for E22 controls."""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import logsumexp


E19 = Path(__file__).resolve().parent.parent / "e19_symmetric_tie_constraint"
sys.path.insert(0, str(E19))
from weighted_objectives import STRICT_SOLVER_SETTINGS, QUALIFICATION_TOLERANCE, clarabel


def validate(phi, rewards, regularization, weights):
    phi = np.asarray(phi, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (phi.ndim != 2 or rewards.ndim != 2 or len(phi) != len(rewards)
            or min(phi.shape) < 1 or rewards.shape[1] < 2
            or weights.shape != (len(phi),)):
        raise ValueError("Expected matching question, feature, reward and row-weight axes")
    if (not np.isfinite(phi).all() or not np.all(phi[:, 0] == 1)
            or not np.isfinite(rewards).all() or np.any((rewards < 0) | (rewards > 1))
            or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1))):
        raise ValueError("Use finite features with intercept and rewards and weights within [0,1]")
    regularization = float(regularization)
    if not 0 < regularization < np.inf:
        raise ValueError("Strictly positive all-coefficient L2 regularization is required")
    return phi, rewards, regularization, weights


def target_probabilities(rewards):
    log_target = rewards - logsumexp(rewards, axis=1, keepdims=True)
    return np.exp(log_target), log_target


def data_terms(family, coefficient, phi, rewards):
    scores = phi @ coefficient
    if family == "tspo":
        target = rewards == rewards.max(axis=1, keepdims=True)
        target = target / target.sum(axis=1, keepdims=True)
        transformed = 2 * scores - rewards
        losses = transformed.max(axis=1) - np.sum(target * transformed, axis=1)
        tie_losses = 2 * (scores.max(axis=1) - scores.mean(axis=1))
    elif family == "listnet":
        target, log_target = target_probabilities(rewards)
        log_prediction = scores - logsumexp(scores, axis=1, keepdims=True)
        losses = np.sum(target * (log_target - log_prediction), axis=1)
        tie_losses = logsumexp(scores, axis=1) - scores.mean(axis=1) - np.log(scores.shape[1])
    else:
        raise ValueError("Expected tspo or listnet")
    return losses, tie_losses


def components(family, coefficient, phi, rewards, regularization, weights):
    losses, tie_losses = data_terms(family, coefficient, phi, rewards)
    tied = np.all(rewards == rewards[:, :1], axis=1)
    penalty = regularization * float(np.sum(coefficient**2))
    data_loss = float(weights @ losses / len(phi))
    non_tied = float(losses[~tied].sum() / len(phi))
    tie_term = float(tie_losses[tied].sum() / len(phi))
    return dict(objective=data_loss + penalty, data_loss=data_loss,
                regularization_penalty=penalty, F0=non_tied + penalty,
                F1=float(losses.mean()) + penalty, J_T=tie_term,
                all_action_tied_questions=int(tied.sum()),
                original_question_denominator=len(phi),
                effective_question_weight_sum=float(weights.sum()),
                weight_zero_questions=int(np.count_nonzero(weights == 0)),
                weight_one_questions=int(np.count_nonzero(weights == 1)))


def listnet_value_gradient(coefficient, phi, rewards, regularization, weights):
    target, _ = target_probabilities(rewards)
    scores = phi @ coefficient
    probability = np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
    row_gradient = weights[:, None] * (probability - target)
    gradient = phi.T @ row_gradient / len(phi) + 2 * regularization * coefficient
    report = components("listnet", coefficient, phi, rewards, regularization, weights)
    return report, gradient, row_gradient, probability


def fit_listnet(phi, rewards, regularization, weights):
    shape = (phi.shape[1], rewards.shape[1])

    def objective(flat):
        report, gradient, _, _ = listnet_value_gradient(
            flat.reshape(shape), phi, rewards, regularization, weights)
        return report["objective"], gradient.ravel()

    settings = dict(gtol=1e-12, ftol=1e-15, maxiter=2000, maxls=50, maxcor=20)
    optimized = minimize(objective, np.zeros(np.prod(shape)), jac=True,
                         method="L-BFGS-B", options=settings)
    if not optimized.success:
        raise RuntimeError("Weighted ListNet optimization failed " + str(optimized.message))
    flat = optimized.x
    refinements = 0
    for _ in range(3):
        _, gradient, _, probability = listnet_value_gradient(
            flat.reshape(shape), phi, rewards, regularization, weights)
        if np.abs(gradient).max() <= 1e-10:
            break
        covariance = -probability[:, :, None] * probability[:, None, :]
        action_indices = np.arange(shape[1])
        covariance[:, action_indices, action_indices] += probability
        hessian = np.einsum("ip,iq,iab,i->paqb", phi, phi, covariance, weights,
                            optimize=True).reshape(flat.size, flat.size) / len(phi)
        hessian.flat[::flat.size + 1] += 2 * regularization
        step = solve(hessian, gradient.ravel(), assume_a="pos")
        if objective(flat - step)[0] > objective(flat)[0] + 1e-13:
            raise RuntimeError("Weighted ListNet refinement increased the objective")
        flat -= step
        refinements += 1
    coefficient = flat.reshape(shape)
    report, gradient, row_gradient, _ = listnet_value_gradient(
        coefficient, phi, rewards, regularization, weights)
    if np.abs(gradient).max() > 1e-10:
        raise RuntimeError("Weighted ListNet stationarity qualification failed")
    report.update(status="Solved", solver="L-BFGS-B + qualified Newton refinement",
                  solver_settings=settings, gradient_inf=float(np.abs(gradient).max()),
                  iterations=int(optimized.nit), newton_refinements=refinements,
                  objective_gap_bound=float(np.sum(gradient**2) / (4 * regularization)))
    return coefficient, report, row_gradient


def fit_tspo(phi, rewards, regularization, weights):
    questions, features = phi.shape
    actions = rewards.shape[1]
    coefficient_count = features * actions
    maxima = rewards == rewards.max(axis=1, keepdims=True)
    target = maxima / maxima.sum(axis=1, keepdims=True)
    quadratic = sparse.diags(np.r_[np.full(coefficient_count, 2 * regularization),
                                  np.zeros(questions)], format="csc")
    linear = np.r_[(-(2 / questions) * phi.T @ (weights[:, None] * target)).reshape(-1, order="F"),
                   weights / questions]
    action_blocks = sparse.kron(sparse.eye(actions, format="csc"),
                               2 * sparse.csc_matrix(phi), format="csc")
    epigraph_blocks = sparse.vstack([-sparse.eye(questions, format="csc")] * actions, format="csc")
    constraint = sparse.hstack([action_blocks, epigraph_blocks], format="csc")
    upper = rewards.reshape(-1, order="F")
    settings = clarabel.DefaultSettings()
    for key, value in STRICT_SOLVER_SETTINGS.items():
        setattr(settings, key, value)
    solved = clarabel.DefaultSolver(quadratic, linear, constraint, upper,
                                   [clarabel.NonnegativeConeT(questions * actions)], settings).solve()
    if solved.status != clarabel.SolverStatus.Solved:
        raise RuntimeError("Weighted TSPO optimization status " + str(solved.status))
    primal, dual = np.asarray(solved.x), np.asarray(solved.z)
    coefficient = primal[:coefficient_count].reshape(features, actions, order="F")
    report = components("tspo", coefficient, phi, rewards, regularization, weights)
    constant = float(np.sum(weights[:, None] * target * rewards) / questions)
    slack = upper - constraint @ primal
    stationarity = quadratic @ primal + linear + constraint.T @ dual
    raw_mass = dual.reshape(questions, actions, order="F")
    positive = weights > 0
    normalized_mass = np.zeros_like(raw_mass)
    normalized_mass[positive] = raw_mass[positive] / raw_mass[positive].sum(axis=1, keepdims=True)
    row_gradient = 2 * weights[:, None] * (normalized_mass - target)
    reconstruction = phi.T @ row_gradient / questions + 2 * regularization * coefficient
    checks = dict(
        direct_epigraph_error=abs(report["objective"] - float(solved.obj_val + constant)),
        primal_violation=float(np.maximum(-slack, 0).max()),
        dual_violation=float(np.maximum(-dual, 0).max()),
        stationarity_inf=float(np.abs(stationarity).max()),
        complementarity_inf=float(np.abs(slack * dual).max()),
        primal_dual_objective_gap=abs(float(solved.obj_val - solved.obj_val_dual)),
        normalized_row_gradient_stationarity_inf=float(np.abs(reconstruction).max()),
        epigraph_dual_row_sum_error=float(np.abs(raw_mass.sum(axis=1) - weights/questions).max()),
    )
    if any(value > QUALIFICATION_TOLERANCE for value in checks.values()):
        raise RuntimeError("Weighted TSPO numerical qualification failed " + repr(checks))
    report.update(status="Solved", solver="Clarabel", solver_version=clarabel.__version__,
                  solver_settings=dict(STRICT_SOLVER_SETTINGS), iterations=int(solved.iterations),
                  epigraph_rows_retained=questions,
                  **checks)
    return coefficient, report, row_gradient


def fit(family, phi, rewards, regularization, weights):
    """Return W, aggregate diagnostics and weighted score gradients per row.

    Row gradients do not include the 1/N factor. They reconstruct stationarity
    through phi.T @ row_gradient / N + 2*regularization*W. The TSPO max
    subgradient is selected by the qualified epigraph dual, including at ties.
    """
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    phi, rewards, regularization, weights = validate(phi, rewards, regularization, weights)
    operation = {"tspo": fit_tspo, "listnet": fit_listnet}.get(family)
    if operation is None:
        raise ValueError("Expected family tspo or listnet")
    coefficient, report, row_gradient = operation(phi, rewards, regularization, weights)
    report.update(family=family, regularization=regularization,
                  parameter_norm=float(np.linalg.norm(coefficient)),
                  feature_dimension=phi.shape[1], action_dimension=rewards.shape[1],
                  penalized_intercept=True, training_rows_retained=len(phi),
                  normalization="1/N retained original questions; row weights never renormalized",
                  cpu_seconds=time.process_time() - start_cpu,
                  wall_seconds=time.perf_counter() - start_wall)
    return coefficient, report, row_gradient
