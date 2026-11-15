"""Exact finite-sample SPO+ objectives for the frozen E17 linear heads.

One original question contributes the mean of its joint reward draws. The data
term is divided by ``n * B`` and the entire coefficient matrix, including its
constant row, receives ``regularization * ||W||_F**2``. No near-optimal set or
practical-difference threshold enters these operators.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time


E8_DEPENDENCIES = (
    Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
    / "research/e8_decision_focused_combination_learning/private/dependencies"
)
sys.path.insert(0, str(E8_DEPENDENCIES))

import clarabel
import numpy as np
from scipy import sparse


RULES = ("lexicographic", "uniform", "semantic")
SOLVER_SETTINGS = dict(
    tol_gap_abs=1e-9,
    tol_gap_rel=1e-9,
    tol_feas=1e-9,
    max_iter=1000,
    max_threads=1,
    verbose=False,
)


def _identity_key(value):
    """Convert a semantic combination key to a comparable immutable value."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return tuple(_identity_key(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _semantic_order(physical_ids, actions):
    if physical_ids is None:
        return np.arange(actions)
    if len(physical_ids) != actions:
        raise ValueError("One physical identity is required for each action")
    identities = [_identity_key(value) for value in physical_ids]
    try:
        if len(set(identities)) != actions:
            raise ValueError("Physical action identities must be unique")
        return np.array(sorted(range(actions), key=identities.__getitem__), dtype=int)
    except TypeError as error:
        raise ValueError("Physical action keys must be hashable and mutually sortable") from error


def _validate(phi, samples, regularization, rule, physical_ids):
    if rule not in RULES:
        raise ValueError("Unknown true-optimum tie rule")
    phi = np.asarray(phi, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples[:, None, :]
    if phi.ndim != 2 or samples.ndim != 3:
        raise ValueError("Features must be n by d and rewards n by B by K")
    if len(phi) != len(samples) or min(phi.shape) < 1:
        raise ValueError("Features and reward blocks must share nonempty question rows")
    if samples.shape[1] < 1 or samples.shape[2] < 2:
        raise ValueError("At least one draw and two actions are required")
    if not np.isfinite(phi).all() or not np.isfinite(samples).all():
        raise ValueError("Features and rewards must be finite")
    regularization = float(regularization)
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("Regularization must be finite and nonnegative")
    order = _semantic_order(physical_ids, samples.shape[2])
    return phi, samples, regularization, order


def _optimal_weights(rewards, rule, semantic_order):
    maxima = rewards == rewards.max(axis=1, keepdims=True)
    if rule == "uniform":
        return maxima / maxima.sum(axis=1, keepdims=True)
    chosen = maxima.argmax(axis=1)
    if rule == "semantic":
        chosen = semantic_order[maxima[:, semantic_order].argmax(axis=1)]
    weights = np.zeros_like(rewards)
    weights[np.arange(len(rewards)), chosen] = 1.0
    return weights


def objective_value_and_gradient(
    coefficient, phi, samples, regularization, rule="lexicographic", physical_ids=None
):
    """Return the direct SPO+ value and one valid max subgradient.

    Two-dimensional rewards are treated as one draw per question. A semantic
    rule with omitted identities uses the canonical physical order 0 through
    K-1. At a transformed-score tie the first numeric maximizer defines the
    returned subgradient. This choice leaves the objective unchanged.
    """
    phi, samples, regularization, order = _validate(
        phi, samples, regularization, rule, physical_ids
    )
    coefficient = np.asarray(coefficient, dtype=np.float64)
    n, draws, actions = samples.shape
    if coefficient.shape != (phi.shape[1], actions) or not np.isfinite(coefficient).all():
        raise ValueError("Coefficient shape must match feature and action dimensions")
    weights = _optimal_weights(samples.reshape(n * draws, actions), rule, order)
    weights = weights.reshape(n, draws, actions)
    transformed = 2 * (phi @ coefficient)[:, None, :] - samples
    chosen = transformed.argmax(axis=2)
    maximum = np.take_along_axis(transformed, chosen[:, :, None], axis=2)[:, :, 0]
    data_loss = np.mean(maximum - np.sum(weights * transformed, axis=2))
    direction = -weights.copy()
    direction[np.arange(n)[:, None], np.arange(draws)[None, :], chosen] += 1.0
    gradient = (2 / n) * phi.T @ direction.mean(axis=1)
    gradient += 2 * regularization * coefficient
    value = data_loss + regularization * np.sum(coefficient ** 2)
    return float(value), gradient


def objective_value(
    coefficient, phi, samples, regularization, rule="lexicographic", physical_ids=None
):
    """Evaluate the frozen direct loss, including its single L2 penalty."""
    return objective_value_and_gradient(
        coefficient, phi, samples, regularization, rule, physical_ids
    )[0]


def fit_spo(phi, reward2d, regularization, rule="lexicographic", physical_ids=None):
    """Fit ordinary SPO+ to one action-reward vector per question."""
    reward2d = np.asarray(reward2d, dtype=np.float64)
    if reward2d.ndim != 2:
        raise ValueError("Ordinary SPO+ requires a two-dimensional reward matrix")
    return fit_expected_spo(phi, reward2d[:, None, :], regularization, rule, physical_ids)


def fit_expected_spo(phi, samples, regularization, rule="lexicographic", physical_ids=None):
    """Fit the exact n-question, B-draw SPO+ epigraph quadratic program.

    The finite draw collection is fixed by the caller. This function performs no
    sampling and accepts generic action counts, including the 2-action controls
    and the 20-action real task. Its convention is action-major coefficient and
    constraint flattening, with question-major draw rows.
    """
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    phi, samples, regularization, order = _validate(
        phi, samples, regularization, rule, physical_ids
    )
    n, draws, actions = samples.shape
    dimension = phi.shape[1]
    sample_count = n * draws
    coefficient_count = dimension * actions
    flat_phi = np.repeat(phi, draws, axis=0)
    flat_rewards = samples.reshape(sample_count, actions)
    target = _optimal_weights(flat_rewards, rule, order)
    quadratic = sparse.diags(
        np.r_[np.full(coefficient_count, 2 * regularization), np.zeros(sample_count)],
        format="csc",
    )
    linear = np.r_[
        (-(2 / sample_count) * flat_phi.T @ target).reshape(-1, order="F"),
        np.full(sample_count, 1 / sample_count),
    ]
    action_blocks = sparse.kron(
        sparse.eye(actions, format="csc"), 2 * sparse.csc_matrix(flat_phi), format="csc"
    )
    epigraph_blocks = sparse.vstack(
        [-sparse.eye(sample_count, format="csc")] * actions, format="csc"
    )
    constraint = sparse.hstack([action_blocks, epigraph_blocks], format="csc")
    upper = flat_rewards.reshape(-1, order="F")
    settings = clarabel.DefaultSettings()
    for name, value in SOLVER_SETTINGS.items():
        setattr(settings, name, value)
    solver = clarabel.DefaultSolver(
        quadratic, linear, constraint, upper,
        [clarabel.NonnegativeConeT(sample_count * actions)], settings,
    )
    solved = solver.solve()
    if solved.status != clarabel.SolverStatus.Solved:
        raise RuntimeError("SPO+ solver status " + str(solved.status))
    primal, dual = np.asarray(solved.x), np.asarray(solved.z)
    coefficient = primal[:coefficient_count].reshape(dimension, actions, order="F")
    direct = objective_value(coefficient, phi, samples, regularization, rule, physical_ids)
    constant = float(np.mean(np.sum(target * flat_rewards, axis=1)))
    epigraph = float(solved.obj_val + constant)
    objective_error = abs(direct - epigraph)
    violation = float(np.maximum(constraint @ primal - upper, 0).max())
    gap = float(abs(solved.obj_val - solved.obj_val_dual))
    if objective_error > 1e-7 or violation > 1e-7 or gap > 1e-7:
        raise AssertionError("Exact SPO+ epigraph qualification failed")
    stationarity = float(np.abs(quadratic @ primal + linear + constraint.T @ dual).max())
    diagnostics = dict(
        status=str(solved.status), solver="Clarabel", solver_version=clarabel.__version__,
        objective=direct, exact_loss=direct, epigraph_objective=epigraph,
        objective_error=objective_error, direct_epigraph_error=objective_error,
        constraint_violation=violation, primal_violation=violation,
        primal_dual_gap=gap, stationarity=stationarity, iterations=int(solved.iterations),
        cpu_seconds=time.process_time() - start_cpu, wall_seconds=time.perf_counter() - start_wall,
        questions=n, samples_per_question=draws, actions=actions,
        variables=coefficient_count + sample_count, constraints=sample_count * actions,
        coefficient_dimension=coefficient_count, regularization=regularization, tie_rule=rule,
        normalization="1/(n*B) + lambda*entire_coefficient_squared_norm",
        sparse_constraint_nonzeros=int(constraint.nnz),
        sparse_constraint_bytes=int(constraint.data.nbytes + constraint.indices.nbytes + constraint.indptr.nbytes),
    )
    return coefficient, diagnostics


def select_actions(scores_phys, numeric_order=None, tolerance=0, semantic=True):
    """Choose from a frozen exact/near-max set in physical action coordinates.

    Physical columns are in canonical semantic order. With ``semantic=False``,
    ``numeric_order`` lists physical columns in their current numeric priority.
    With ``semantic=True`` the smallest physical column is selected. The
    tolerance is an absolute score difference and does not alter training.
    """
    scores = np.asarray(scores_phys, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError("Finite question-by-physical-action scores are required")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Action-selection tolerance must be finite and nonnegative")
    actions = scores.shape[1]
    order = np.arange(actions)
    if numeric_order is not None:
        candidate = np.asarray(numeric_order)
        if candidate.shape != (actions,) or not np.issubdtype(candidate.dtype, np.integer):
            raise ValueError("Numeric order must be an integer physical-action permutation")
        if not np.array_equal(np.sort(candidate), np.arange(actions)):
            raise ValueError("Numeric order must visit each physical action once")
        if not semantic:
            order = candidate
    eligible = scores >= scores.max(axis=1, keepdims=True) - tolerance
    return order[eligible[:, order].argmax(axis=1)]
