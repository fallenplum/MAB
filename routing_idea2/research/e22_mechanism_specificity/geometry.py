"""Training-only equilibrium accounting for the fixed tie-row endpoints."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr


TAU = 1e-8
KKT_TOLERANCE = 1e-7


def _summary(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not len(values):
        return dict(count=0, mean=None, median=None, minimum=None, maximum=None)
    return dict(count=int(len(values)), mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()))


def _correlation(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if (len(first) < 3 or np.ptp(first) <= 1e-15 or np.ptp(second) <= 1e-15):
        return None
    value = float(spearmanr(first, second).statistic)
    return value if np.isfinite(value) else None


def _cosine(first, second):
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return None if denominator <= 1e-30 else float(np.sum(first * second) / denominator)


def _objective(family, coefficient, phi, rewards, regularization, tied):
    scores = phi @ coefficient
    if family == "tspo":
        maxima = rewards == rewards.max(axis=1, keepdims=True)
        target = maxima / maxima.sum(axis=1, keepdims=True)
        transformed = 2 * scores - rewards
        losses = transformed.max(axis=1) - (target * transformed).sum(axis=1)
        tie_values = 2 * (scores.max(axis=1) - scores.mean(axis=1))
    else:
        target_log = rewards - logsumexp(rewards, axis=1, keepdims=True)
        target = np.exp(target_log)
        prediction_log = scores - logsumexp(scores, axis=1, keepdims=True)
        losses = (target * (target_log - prediction_log)).sum(axis=1)
        tie_values = logsumexp(scores, axis=1) - scores.mean(axis=1) - np.log(scores.shape[1])
    penalty = regularization * float(np.sum(coefficient**2))
    f0 = float(losses[~tied].sum() / len(phi) + penalty)
    jt = float(tie_values[tied].sum() / len(phi))
    return dict(F0=f0, J_T=jt, F1=f0 + jt, penalty=penalty,
                total_data_loss=float(losses.mean()), non_tie_data_loss=f0 - penalty,
                raw_parameter_frobenius=float(np.linalg.norm(coefficient)),
                centered_parameter_frobenius=float(np.linalg.norm(
                    coefficient - coefficient.mean(axis=1, keepdims=True))))


def _check_rowgrad(family, phi, rewards, coefficient, rowgrad, weights):
    scores = phi @ coefficient
    if family == "listnet":
        target = np.exp(rewards - logsumexp(rewards, axis=1, keepdims=True))
        prediction = np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
        error = float(np.abs(rowgrad - weights[:, None] * (prediction - target)).max())
        if error > KKT_TOLERANCE:
            raise ValueError("ListNet row gradients differ from the fixed weighted objective")
        return dict(gradient_identity_error=error)
    maxima = rewards == rewards.max(axis=1, keepdims=True)
    target = maxima / maxima.sum(axis=1, keepdims=True)
    active = weights > 0
    rho = target[active] + rowgrad[active] / (2 * weights[active, None])
    transformed = 2 * scores[active] - rewards[active]
    differences = transformed.max(axis=1, keepdims=True) - transformed
    errors = dict(
        simplex_sum_error=float(np.abs(rho.sum(axis=1) - 1).max()) if len(rho) else 0.0,
        negative_dual_mass=float(np.maximum(-rho, 0).max()) if len(rho) else 0.0,
        max_subgradient_complementarity=float(np.abs(rho * differences).max()) if len(rho) else 0.0,
        zero_weight_gradient_error=float(np.abs(rowgrad[~active]).max()) if np.any(~active) else 0.0,
    )
    if max(errors.values()) > KKT_TOLERANCE:
        raise ValueError("TSPO row subgradients fail simplex or active-max qualification")
    return errors


def analyze(family, phi_h, phi_e, rewards, lam, WB, WC, rowgradB, rowgradC):
    """Return public summaries and private arrays from legal per-row gradients.

    Row gradients include row weights and exclude the common 1/N and L2 terms.
    The TSPO gradients must be recovered from the solver dual certificate.
    """
    if family not in {"tspo", "listnet"}:
        raise ValueError("Expected the frozen TSPO or ListNet family")
    phi_h, phi_e, rewards, WB, WC, rowgradB, rowgradC = (
        np.asarray(value, dtype=np.float64)
        for value in (phi_h, phi_e, rewards, WB, WC, rowgradB, rowgradC))
    n, dimension = phi_h.shape
    actions = rewards.shape[1]
    if (not n or not len(phi_e) or rewards.shape != (n, actions)
            or phi_e.shape[1] != dimension or WB.shape != (dimension, actions)
            or WC.shape != WB.shape or rowgradB.shape != rewards.shape
            or rowgradC.shape != rewards.shape or not 0 < lam < np.inf):
        raise ValueError("Fixed endpoint geometry dimensions differ")
    if not all(np.isfinite(value).all() for value in (phi_h, phi_e, rewards, WB, WC, rowgradB, rowgradC)):
        raise ValueError("Endpoint geometry requires finite arrays")
    tied = np.all(rewards == rewards[:, :1], axis=1)
    checks_b = _check_rowgrad(family, phi_h, rewards, WB, rowgradB, np.ones(n))
    checks_c = _check_rowgrad(family, phi_h, rewards, WC, rowgradC, (~tied).astype(float))
    g_tie = phi_h[tied].T @ rowgradB[tied] / n
    g_non_b = phi_h[~tied].T @ rowgradB[~tied] / n
    g_non_c = phi_h.T @ rowgradC / n
    residual_b = g_tie + g_non_b + 2 * lam * WB
    residual_c = g_non_c + 2 * lam * WC
    residual_inf = max(float(np.abs(residual_b).max()), float(np.abs(residual_c).max()))
    if residual_inf > KKT_TOLERANCE:
        raise ValueError("Endpoint geometry lacks the required numerical KKT certificate")
    delta = WC - WB
    predicted_delta = (g_tie + g_non_b - g_non_c) / (2 * lam)
    coeff_error = float(np.abs(delta - predicted_delta).max())
    residual_identity = float(np.abs(2 * lam * delta - g_tie - g_non_b + g_non_c).max())
    if residual_identity > 2 * residual_inf + 1e-12:
        raise ValueError("Equilibrium decomposition failed its residual identity")
    components_b = _objective(family, WB, phi_h, rewards, lam, tied)
    components_c = _objective(family, WC, phi_h, rewards, lam, tied)
    norm_delta = float(np.linalg.norm(delta))
    inner = float(np.sum(g_tie * delta))
    monotonic_slack = inner - 2 * lam * norm_delta**2 - float(np.sum((residual_b-residual_c) * delta))
    increase_f0 = components_b["F0"] - components_c["F0"]
    decrease_jt = components_c["J_T"] - components_b["J_T"]
    strong_f0_slack = increase_f0 - lam * norm_delta**2 + float(np.sum(residual_c * delta))
    strong_f1_slack = decrease_jt - increase_f0 - lam * norm_delta**2 - float(np.sum(residual_b * delta))
    numeric_tolerance = 1e-7 * max(1.0, abs(components_b["F1"]), abs(components_c["F1"]))
    if min(monotonic_slack, strong_f0_slack, strong_f1_slack) < -numeric_tolerance:
        raise ValueError("Endpoint strong-convexity geometry failed numerical qualification")

    scores_b, scores_c = phi_e @ WB, phi_e @ WC
    centered_b = scores_b - scores_b.mean(axis=1, keepdims=True)
    centered_c = scores_c - scores_c.mean(axis=1, keepdims=True)
    centered_delta = centered_c - centered_b
    sort_b, sort_c = np.sort(scores_b, axis=1), np.sort(scores_c, axis=1)
    margin_b, margin_c = sort_b[:, -1]-sort_b[:, -2], sort_c[:, -1]-sort_c[:, -2]
    action_b, action_c = scores_b.argmax(axis=1), scores_c.argmax(axis=1)
    strict = (margin_b > TAU) & (margin_c > TAU)
    switched = strict & (action_b != action_c)
    tie_score_component = phi_e @ g_tie / (2 * lam)
    non_score_component = phi_e @ (g_non_b-g_non_c) / (2 * lam)
    positions = np.arange(len(phi_e))
    def winner_contrast(values):
        return values[positions, action_c] - values[positions, action_b]
    actual_contrast = winner_contrast(scores_c-scores_b)
    tie_contrast = winner_contrast(tie_score_component)
    non_contrast = winner_contrast(non_score_component)
    gram_squared = np.sum((phi_e @ phi_h[tied].T)**2, axis=1) / n if tied.any() else np.zeros(len(phi_e))
    centered_tie_component = tie_score_component - tie_score_component.mean(axis=1, keepdims=True)
    actual_norm = np.linalg.norm(centered_delta, axis=1)
    tie_norm = np.linalg.norm(centered_tie_component, axis=1)
    norm_b_squared = float(np.sum(WB**2))
    positive_scale = max(0.0, float(np.sum(WB*WC)) / norm_b_squared) if norm_b_squared else 0.0
    nonradial = WC-positive_scale*WB
    pairs_i, pairs_j = np.triu_indices(actions, k=1)
    pair_difference = centered_delta[:, pairs_i]-centered_delta[:, pairs_j]
    score_identity_error = np.abs(scores_c-scores_b-tie_score_component-non_score_component)
    total_var = np.abs(
        (scores_c >= scores_c.max(axis=1, keepdims=True)-TAU) /
        (scores_c >= scores_c.max(axis=1, keepdims=True)-TAU).sum(axis=1, keepdims=True)
        -(scores_b >= scores_b.max(axis=1, keepdims=True)-TAU) /
        (scores_b >= scores_b.max(axis=1, keepdims=True)-TAU).sum(axis=1, keepdims=True)
    ).sum(axis=1)/2
    public = dict(
        schema="mab_tie_geometry_accounting_v1", family=family, training_questions=n,
        evaluation_questions=len(phi_e), fully_tied_training_questions=int(tied.sum()), regularization=lam,
        B=components_b, C=components_c,
        row_gradient_qualification={"B": checks_b, "C": checks_c},
        kkt_residual_inf=residual_inf, equilibrium_coefficient_identity_error=coeff_error,
        equilibrium_scaled_identity_residual=residual_identity,
        equilibrium_score_identity_max_error=float(score_identity_error.max()),
        tie_gradient_frobenius=float(np.linalg.norm(g_tie)),
        non_tie_feedback_gradient_change_frobenius=float(np.linalg.norm(g_non_b-g_non_c)),
        tie_non_tie_change_cosine=_cosine(g_tie, g_non_b-g_non_c),
        parameter_change_frobenius=norm_delta, tie_gradient_delta_inner_product=inner,
        tie_gradient_delta_cosine=_cosine(g_tie, delta),
        directional_strong_monotonicity_slack=monotonic_slack,
        F0_increase=increase_f0, JT_decrease=decrease_jt,
        F0_strong_convexity_slack=strong_f0_slack, F1_strong_convexity_slack=strong_f1_slack,
        parameter_norm_upper_bound=(float(np.linalg.norm(g_tie)) + float(np.linalg.norm(residual_b-residual_c))) / (2*lam),
        coefficient_cosine=_cosine(WB, WC), best_positive_rescaling=positive_scale,
        nonradial_parameter_change_frobenius=float(np.linalg.norm(nonradial)),
        nonradial_fraction_of_C_norm=float(np.linalg.norm(nonradial)/np.linalg.norm(WC)) if np.linalg.norm(WC) else None,
        margin_B=_summary(margin_b), margin_C=_summary(margin_c),
        centered_score_change_l2=_summary(actual_norm),
        all_pair_value_difference_change_absolute=_summary(np.abs(pair_difference)),
        both_strict_unique_questions=int(strict.sum()), strict_switch_questions=int(switched.sum()),
        probability_changed_questions=int(np.sum(total_var > 1e-14)),
        strict_switched_winner_contrast_change=_summary(actual_contrast[switched]),
        strict_switched_tie_component=_summary(tie_contrast[switched]),
        strict_switched_non_tie_response_component=_summary(non_contrast[switched]),
        strict_switched_positive_tie_component_questions=int(np.sum(tie_contrast[switched] > 0)),
        strict_switched_tie_component_direction_agreement_fraction=float(np.mean(tie_contrast[switched] > 0)) if switched.any() else None,
        descriptive_spearman_tie_influence_vs_actual_centered_shift=_correlation(tie_norm, actual_norm),
        descriptive_spearman_tie_feature_kernel_vs_actual_centered_shift=_correlation(gram_squared, actual_norm),
        descriptive_spearman_tie_winner_contrast_vs_actual=_correlation(tie_contrast[switched], actual_contrast[switched]),
        interpretation="Exact endpoint-equilibrium accounting with shared feature Gram projections; components are not separately identified causal effects.",
        evaluation_rewards_read=0,
    )
    private = dict(delta_coefficients=delta, tie_gradient_B=g_tie, non_tie_gradient_B=g_non_b,
                   non_tie_gradient_C=g_non_c, residual_B=residual_b, residual_C=residual_c,
                   strict_switch_mask=switched, both_strict_mask=strict,
                   margin_B=margin_b, margin_C=margin_c,
                   centered_score_change=centered_delta, tie_score_component=tie_score_component,
                   non_tie_response_score_component=non_score_component,
                   winner_contrast_change=actual_contrast, tie_winner_contrast=tie_contrast,
                   non_tie_winner_contrast=non_contrast, tie_feature_gram_squared=gram_squared)
    return public, private
