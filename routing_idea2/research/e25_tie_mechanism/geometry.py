"""Audit the score geometry of the frozen E24 heads using only cached inputs."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e24_training_deployment_replication"))
import replication as e24
import numpy as np


FROZEN_FILES = {
    "model": ("research/e20_independent_confirmation_preparation/private/final_models/FINAL600_MODELS_FROZEN.npz",
              e24.MODEL_SHA),
    "inputs": ("research/e24_training_deployment_replication/private/cohort_v1/INPUTS_ONLY.npz",
               "0b0b959297e51f29280c3ed410bdfef7072da34d238659d5c21a114c7a787fcc"),
    "prediction": ("research/e24_training_deployment_replication/private/cohort_v1/E24_POLICIES_FROZEN.npz",
                   "9595ee1ddf9e30e85a0c95e51af2155e6b3de4cbd36265114341f32f9c324cef"),
}
ABS_TOLERANCES = (0, 1e-12, 1e-10, e24.TAU, 1e-6, 1e-4, 1e-2)
REL_TOLERANCES = (1e-8, 1e-6, 1e-3)


def quantiles(values):
    return dict(zip(("min", "p10", "median", "p90", "max"),
                    np.quantile(values, [0, 0.1, 0.5, 0.9, 1]).tolist())) if len(values) else None


def global_groups(near):
    """Return connected groups and whether every pair in a group is close."""
    unseen = set(range(len(near)))
    groups = []
    while unseen:
        seed = min(unseen)
        group = {seed}
        queue = [seed]
        unseen.remove(seed)
        while queue:
            neighbors = unseen.intersection(np.flatnonzero(near[queue.pop()]).tolist())
            group.update(neighbors)
            unseen.difference_update(neighbors)
            queue.extend(neighbors)
        if len(group) > 1:
            actions = sorted(group)
            groups.append(dict(actions=actions, clique=bool(near[np.ix_(actions, actions)].all())))
    return groups


def repeated_support_block(scores, supported, prior, pattern_counts):
    """Compare a repeated multi-action support with its best outside score."""
    if len(pattern_counts) != 1:
        return None
    (block, count), = pattern_counts.items()
    inside = np.asarray(block)
    outside = np.setdiff1d(np.arange(scores.shape[1]), inside)
    if not len(outside):
        return None
    selected = supported[:, inside].all(axis=1)
    block_margin = scores[:, inside].mean(axis=1) - scores[:, outside].max(axis=1)
    # A positive or negative gap distinguishes the repeated support from an
    # outside winner if no row is in a mixed boundary region.
    return dict(
        actions=list(block), questions=count, other_actions=outside.tolist(),
        block_ranks_first_questions=int((block_margin > 0).sum()),
        block_vs_outside_gap_when_top=quantiles(block_margin[selected]),
        outside_vs_block_gap_when_top=quantiles(-block_margin[~selected]),
        prior_support_count_in_block_rows=(prior[selected] > 0).sum(axis=0).tolist(),
        outside_winner_counts=dict(zip(map(str, outside), supported[:, outside].sum(axis=0).tolist())),
    )


def describe_head(scores, weights, design, uniform, prior):
    n, actions = scores.shape
    score_max = scores.max(axis=1, keepdims=True)
    gaps = score_max - scores
    # Keep the raw subtraction and the exact frozen support comparator distinct.
    # They can differ within machine precision of a tolerance boundary.
    supported = uniform > 0
    sizes = supported.sum(axis=1)
    tied = sizes > 1
    order = np.argsort(-scores, axis=1, kind="stable")
    top_margin = gaps[np.arange(n), order[:, 1]]
    score_range = np.ptp(scores, axis=1)
    fraction_margin = np.divide(top_margin, score_range, out=np.zeros(n), where=score_range > 0)
    left, right = np.triu_indices(actions, k=1)
    weights_centered = weights - weights.mean(axis=1, keepdims=True)
    pair_contrasts = weights_centered[:, left] - weights_centered[:, right]
    pair_norms = np.linalg.norm(pair_contrasts, axis=0)
    pair_score_max = np.max(np.abs(scores[:, left] - scores[:, right]), axis=0)
    pair_bound = np.linalg.norm(design, axis=1).max() * pair_norms
    co_tied = supported[:, left] & supported[:, right]
    certified = pair_bound <= e24.TAU
    pair_occurrences = int(co_tied.sum())
    globally_close = np.eye(actions, dtype=bool)
    globally_close[left, right] = globally_close[right, left] = pair_score_max <= e24.TAU
    frequent_supports = Counter(tuple(np.flatnonzero(row).tolist()) for row in supported[tied])
    largest_supports = [dict(actions=list(group), questions=count) for group, count in frequent_supports.most_common(8)]
    winner_tied_norm = pair_contrasts[:, co_tied.any(axis=0)]
    # At the top-two boundary, distinguish small terms from opposing large
    # intercept and variable parts whose cancellation happens for certain rows.
    difference = weights[:, order[:, 0]].T - weights[:, order[:, 1]].T
    intercept = difference[:, 0]
    feature_term = np.sum(design[:, 1:] * difference[:, 1:], axis=1)
    absolute_sum = np.abs(intercept) + np.abs(feature_term)
    cancellation = np.divide(np.abs(intercept + feature_term), absolute_sum,
                             out=np.ones(n), where=absolute_sum > 0)
    pair_top = [
        dict(actions=[int(left[i]), int(right[i])], support_cooccurrences=int(co_tied[:, i].sum()),
             contrast_norm=float(pair_norms[i]), observed_max_score_gap=float(pair_score_max[i]),
             cauchy_schwarz_bound=float(pair_bound[i]))
        for i in np.argsort(-co_tied.sum(axis=0), kind="stable")[:12]
        if co_tied[:, i].any()
    ]
    return dict(
        questions=n, unique_questions=int((sizes == 1).sum()), near_tied_questions=int(tied.sum()),
        exact_float_max_tied_questions=int(((scores == score_max).sum(axis=1) > 1).sum()),
        support_size_counts={str(size): count for size, count in sorted(Counter(sizes.tolist()).items())},
        U_supported_questions_by_action=supported.sum(axis=0).tolist(),
        P_supported_questions_by_action=(prior > 0).sum(axis=0).tolist(),
        P_changes_U_questions=int(np.any(prior != uniform, axis=1).sum()),
        distinct_near_tied_supports=len(frequent_supports),
        most_frequent_near_tied_supports=largest_supports,
        repeated_support_block=repeated_support_block(scores, supported, prior, frequent_supports),
        fixed_action_intersection_across_near_tied_questions=(
            np.flatnonzero(supported[tied].all(axis=0)).tolist() if tied.any() else []),
        top_margin=quantiles(top_margin), near_tied_top_margin=quantiles(top_margin[tied]),
        unique_top_margin=quantiles(top_margin[~tied]), within_question_score_range=quantiles(score_range),
        top_margin_over_score_range=quantiles(fraction_margin),
        near_tied_score_range=quantiles(score_range[tied]),
        near_tied_top_margin_over_score_range=quantiles(fraction_margin[tied]),
        unique_top_margin_over_score_range=quantiles(fraction_margin[~tied]),
        absolute_tolerance_sensitivity=[
            dict(tau=tau, unique_questions=int(((scores >= score_max - tau).sum(axis=1) == 1).sum()))
            for tau in ABS_TOLERANCES],
        within_question_relative_tolerance_sensitivity=[
            dict(fraction=fraction, unique_questions=int((fraction_margin > fraction).sum()))
            for fraction in REL_TOLERANCES],
        centered_weight_frobenius_norm=float(np.linalg.norm(weights_centered)),
        centered_weight_singular_values=np.linalg.svd(weights_centered, compute_uv=False).tolist(),
        pair_contrast_norm=quantiles(pair_norms),
        ever_co_tied_pair_contrast_norm=quantiles(np.linalg.norm(winner_tied_norm, axis=0)),
        globally_close_pairs_on_1196_inputs=int((pair_score_max <= e24.TAU).sum()),
        globally_close_action_groups=global_groups(globally_close),
        pairs_certified_close_on_1196_input_norm_ball=int(certified.sum()),
        co_tied_pair_occurrences=pair_occurrences,
        certified_close_co_tied_pair_occurrences=int(co_tied[:, certified].sum()),
        top_two_contrast_norm_on_near_tied=quantiles(np.linalg.norm(difference[tied], axis=1)),
        max_design_row_norm=float(np.linalg.norm(design, axis=1).max()),
        top_two_abs_intercept_on_near_tied=quantiles(np.abs(intercept[tied])),
        top_two_abs_variable_term_on_near_tied=quantiles(np.abs(feature_term[tied])),
        top_two_cancellation_ratio_on_near_tied=quantiles(cancellation[tied]),
        top_two_gap_decomposition_max_abs_error=float(np.abs(top_margin - (intercept + feature_term)).max()),
        largest_co_tied_pair_counts=pair_top,
    )


def audit(assets):
    paths = {name: assets / relative for name, (relative, _) in FROZEN_FILES.items()}
    for name, path in paths.items():
        if e24.sha(path) != FROZEN_FILES[name][1]:
            raise ValueError(f"Frozen {name} bytes changed")
    if e24.sha(e24.E20 / "predict_final_inputs.py") != e24.OPERATOR_SHA:
        raise ValueError("Frozen E20 score operator changed")
    packet = e24.load_inputs(paths["inputs"])
    with np.load(paths["model"], allow_pickle=False) as model:
        state = {key: model[key] for key in ("weights", "pca_mean", "pca_components", "scaler_mean", "scaler_scale")}
        if not np.array_equal(model["conditions"], ["B", "C"]):
            raise ValueError("Head condition order changed")
    design = e24.features(packet["embeddings"], state)
    weights = state["weights"]
    score_replay = np.stack([design @ np.asfortranarray(head) for head in weights])
    with np.load(paths["prediction"], allow_pickle=False) as frozen:
        scores = frozen["scores"]
        probabilities = frozen["probabilities"]
        if not np.array_equal(packet["question_ids"], frozen["question_ids"]):
            raise ValueError("Cached design and scores use different question order")
        if not np.array_equal(e24.four_policies(score_replay, frozen["combo_success_counts"]), probabilities):
            raise ValueError("Score replay changed a frozen policy")
        if not np.allclose(score_replay, scores, atol=1e-12, rtol=0):
            raise ValueError("Score replay materially differs from archive")
    geometry = {label: describe_head(scores[i], weights[i], design, probabilities[2*i], probabilities[2*i+1])
                for i, label in enumerate(("B", "C"))}
    winner = scores.argmax(axis=-1)
    strict = ((probabilities[::2] > 0).sum(axis=-1) == 1).all(axis=0)
    return dict(
        schema="e25_frozen_head_geometry_v1", frozen_hashes={key: FROZEN_FILES[key][1] for key in paths},
        questions=len(design), actions=scores.shape[-1], tau=e24.TAU,
        exact_score_replay=bool(np.array_equal(score_replay, scores)),
        score_replay_max_abs_error=float(np.abs(score_replay - scores).max()),
        strict_switch_questions=int((strict & (winner[0] != winner[1])).sum()),
        heads=geometry, evaluation_rewards_read=0, llm_calls=0, encoder_calls=0,
        interpretation=dict(
            score_range="Within each input, largest minus smallest of all 20 scores.",
            dimensionless_margin="Divide top-2 score difference by within-input score range.",
            near_tied_pair_certificate="Norm of action coefficient contrast times maximum observed design norm.",
            cancellation_ratio="Absolute gap divided by sum of absolute intercept and variable terms.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    e24.write(args.output, audit(args.assets))


if __name__ == "__main__":
    main()
