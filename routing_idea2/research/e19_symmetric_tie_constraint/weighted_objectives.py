"""Exact E19 beta0/beta1 uniform-TSPO objectives in the frozen E17 geometry.

All N epigraph rows remain in the quadratic program. The empirical all-action
tie indicator changes only each row's objective weight. The denominator stays
N, and all entries of W, including the constant-feature row, receive the same
positive L2 penalty. No fitting or reward access occurs at module import.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

_dependencies = (
    Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
    / "research/e8_decision_focused_combination_learning/private/dependencies"
)
sys.path.insert(0, str(_dependencies))

import clarabel
import numpy as np
from scipy import sparse


# These are the E17 main solver settings. E19 applies the same predeclared
# tighter tolerance to both beta branches and every action ordering.
SOLVER_SETTINGS = dict(
    tol_gap_abs=1e-9,
    tol_gap_rel=1e-9,
    tol_feas=1e-9,
    max_iter=1000,
    max_threads=1,
    verbose=False,
)
STRICT_SOLVER_SETTINGS = dict(
    SOLVER_SETTINGS, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10
)
QUALIFICATION_TOLERANCE = 1e-7


def _validated(phi, rewards, regularization, beta, permutation, gamma):
    phi = np.asarray(phi, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if phi.ndim != 2 or rewards.ndim != 2 or len(phi) != len(rewards):
        raise ValueError("Require matching question-by-feature and question-by-action arrays")
    if min(phi.shape) < 1 or rewards.shape[1] < 2:
        raise ValueError("A nonempty feature matrix and at least two actions are required")
    if not np.isfinite(phi).all() or not np.isfinite(rewards).all():
        raise ValueError("Features and empirical mean rewards must be finite")
    regularization = float(regularization)
    beta = float(beta)
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("The frozen E19 L2 coefficient must be strictly positive")
    if beta not in (0.0, 1.0) or float(gamma) != 0.0:
        raise ValueError("This qualified operator supports beta0/beta1 and gamma0 only")
    action_count = rewards.shape[1]
    if permutation is None:
        permutation = np.arange(action_count)
    permutation = np.asarray(permutation)
    if (permutation.shape != (action_count,)
            or not np.issubdtype(permutation.dtype, np.integer)
            or not np.array_equal(np.sort(permutation), np.arange(action_count))):
        raise ValueError("Numeric action columns must be a permutation of physical columns")
    return phi, rewards, regularization, beta, permutation.astype(int, copy=False)


def uniform_weights(rewards):
    maxima = rewards == rewards.max(axis=1, keepdims=True)
    return maxima / maxima.sum(axis=1, keepdims=True)


def objective_components(coefficient, phi, rewards, regularization, beta, *, gamma=0.0):
    phi, rewards, regularization, beta, _ = _validated(
        phi, rewards, regularization, beta, None, gamma
    )
    coefficient = np.asarray(coefficient, dtype=np.float64)
    if coefficient.shape != (phi.shape[1], rewards.shape[1]) or not np.isfinite(coefficient).all():
        raise ValueError("Coefficient dimensions must match the frozen feature and action axes")
    scores = phi @ coefficient
    alpha = uniform_weights(rewards)
    transformed = 2.0 * scores - rewards
    losses = transformed.max(axis=1) - np.sum(alpha * transformed, axis=1)
    all_tied = np.all(rewards == rewards[:, :1], axis=1)
    omega = np.where(all_tied, beta, 1.0)
    penalty = regularization * float(np.sum(coefficient**2))
    f0_data = float(np.sum(losses[~all_tied]) / len(phi))
    jt = float(2.0 * np.sum((scores.max(axis=1) - scores.mean(axis=1))[all_tied]) / len(phi))
    f1_data = float(np.mean(losses))
    return dict(
        objective=float(np.dot(omega, losses) / len(phi) + penalty),
        data_loss=float(np.dot(omega, losses) / len(phi)),
        regularization_penalty=penalty,
        F0=f0_data + penalty,
        F1=f1_data + penalty,
        J_T=jt,
        identity_F1_minus_F0_minus_JT=float(f1_data - f0_data - jt),
        all_action_tied_questions=int(all_tied.sum()),
        non_all_tied_questions=int((~all_tied).sum()),
        original_question_denominator=len(phi),
        effective_question_weight_sum=float(omega.sum()),
    )


def objective_value_and_gradient(coefficient, phi, rewards, regularization, beta, *, gamma=0.0):
    """Return F_beta and one valid max subgradient in physical action coordinates."""
    phi, rewards, regularization, beta, _ = _validated(
        phi, rewards, regularization, beta, None, gamma
    )
    coefficient = np.asarray(coefficient, dtype=np.float64)
    values = objective_components(coefficient, phi, rewards, regularization, beta, gamma=gamma)
    alpha = uniform_weights(rewards)
    transformed = 2.0 * (phi @ coefficient) - rewards
    winner = transformed.argmax(axis=1)
    direction = -alpha
    direction[np.arange(len(phi)), winner] += 1.0
    all_tied = np.all(rewards == rewards[:, :1], axis=1)
    omega = np.where(all_tied, beta, 1.0)
    gradient = (2.0 / len(phi)) * phi.T @ (omega[:, None] * direction)
    gradient += 2.0 * regularization * coefficient
    return values["objective"], gradient


def fit_weighted_tspo(phi, rewards, regularization, beta, *, permutation=None,
                      gamma=0.0, strict_tolerances=True):
    """Solve the exact retained-N epigraph QP and map coefficients to physical IDs.

    Numeric column j represents physical action permutation[j]. The numeric
    reward and uniform-optimal-weight columns are changed together. A fresh
    Clarabel DefaultSolver is used without a supplied warm start.
    """
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    phi, rewards, regularization, beta, permutation = _validated(
        phi, rewards, regularization, beta, permutation, gamma
    )
    numeric_rewards = rewards[:, permutation]
    n, dimension = phi.shape
    actions = rewards.shape[1]
    coefficient_count = dimension * actions
    alpha = uniform_weights(numeric_rewards)
    all_tied = np.all(numeric_rewards == numeric_rewards[:, :1], axis=1)
    omega = np.where(all_tied, beta, 1.0)
    quadratic = sparse.diags(
        np.r_[np.full(coefficient_count, 2.0 * regularization), np.zeros(n)], format="csc"
    )
    linear = np.r_[
        (-(2.0 / n) * phi.T @ (omega[:, None] * alpha)).reshape(-1, order="F"),
        omega / n,
    ]
    action_blocks = sparse.kron(
        sparse.eye(actions, format="csc"), 2.0 * sparse.csc_matrix(phi), format="csc"
    )
    epigraph_blocks = sparse.vstack([-sparse.eye(n, format="csc")] * actions, format="csc")
    constraint = sparse.hstack([action_blocks, epigraph_blocks], format="csc")
    upper = numeric_rewards.reshape(-1, order="F")
    chosen_settings = STRICT_SOLVER_SETTINGS if strict_tolerances else SOLVER_SETTINGS
    settings = clarabel.DefaultSettings()
    for name, value in chosen_settings.items():
        setattr(settings, name, value)
    solver = clarabel.DefaultSolver(
        quadratic, linear, constraint, upper,
        [clarabel.NonnegativeConeT(n * actions)], settings,
    )
    solved = solver.solve()
    if solved.status != clarabel.SolverStatus.Solved:
        raise RuntimeError("Weighted TSPO solver status " + str(solved.status))
    primal, dual = np.asarray(solved.x), np.asarray(solved.z)
    numeric_coefficient = primal[:coefficient_count].reshape(dimension, actions, order="F")
    coefficient = np.empty_like(numeric_coefficient)
    coefficient[:, permutation] = numeric_coefficient
    components = objective_components(coefficient, phi, rewards, regularization, beta, gamma=gamma)
    constant = float(np.sum(omega[:, None] * alpha * numeric_rewards) / n)
    epigraph_objective = float(solved.obj_val + constant)
    slack = upper - constraint @ primal
    stationarity = quadratic @ primal + linear + constraint.T @ dual
    primal_violation = float(np.maximum(-slack, 0).max())
    dual_violation = float(np.maximum(-dual, 0).max())
    stationarity_inf = float(np.abs(stationarity).max())
    complementarity_inf = float(np.abs(slack * dual).max())
    gap = float(abs(solved.obj_val - solved.obj_val_dual))
    direct_error = abs(components["objective"] - epigraph_objective)
    thresholds = dict(
        direct_epigraph_error=direct_error,
        primal_violation=primal_violation,
        dual_violation=dual_violation,
        stationarity_inf=stationarity_inf,
        complementarity_inf=complementarity_inf,
        primal_dual_objective_gap=gap,
    )
    if any(value > QUALIFICATION_TOLERANCE for value in thresholds.values()):
        raise AssertionError("Weighted TSPO numerical qualification failed " + repr(thresholds))
    diagnostics = dict(
        status=str(solved.status), solver="Clarabel", solver_version=clarabel.__version__,
        **components, **thresholds,
        epigraph_objective=epigraph_objective,
        primal_objective_without_constant=float(solved.obj_val),
        dual_objective_without_constant=float(solved.obj_val_dual),
        complementarity_sum=float(np.dot(slack, dual)),
        iterations=int(solved.iterations),
        coefficient_dimension=coefficient_count,
        feature_dimension=dimension,
        action_dimension=actions,
        epigraph_rows_retained=n,
        variables=coefficient_count + n,
        constraints=n * actions,
        regularization=regularization,
        penalized_constant_feature=True,
        penalty_operator="regularization * identity on all feature-by-action coefficients",
        strong_convexity_modulus=2.0 * regularization,
        beta=beta, gamma=0.0,
        normalization="1/N original questions retained for both beta branches",
        solver_settings=dict(chosen_settings),
        solver_initialization="fresh Clarabel DefaultSolver, no supplied warm start",
        numeric_to_physical_permutation=permutation.tolist(),
        coefficients_returned_in_physical_action_order=True,
        sparse_constraint_nonzeros=int(constraint.nnz),
        cpu_seconds=time.process_time() - started_cpu,
        wall_seconds=time.perf_counter() - started_wall,
    )
    return coefficient, diagnostics
