"""Fit from original history rows; score frozen E27 policies in a separate phase."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
for sibling in ("e19_symmetric_tie_constraint", "e26_public_routing_audit"):
    sys.path.insert(0, str(HERE.parent / sibling))

from audit import load_fields, neighbor_scores, require_hash, tie_counts, uniform
from run_e19 import INPUT_SHA, REWARD_SHA, read_json, read_reward_rows, sha, write_json, write_npz


INPUT_FIELDS = ("embeddings", "question_ids", "canonical_hashes", "task_labels", "subtask_labels")
BASELINES = ("Sky_First", "Sky_U", "Sky_P", "Ridge", "Fixed")
FEATURE_SOURCE_SHA = "c5fc03c5dc381125844263a616a2bf8a80ee21f51490ee65109ef6876ba0b437"
PROTOCOL = HERE / "PROTOCOL.json"


def source_assets(asset_root, seeds):
    historical = asset_root / "artifacts/private/routing_redesign_20261104"
    inputs = historical / "full_feedback_inputs/v1/selector_train_inputs.npz"
    reward = historical / "ff_v1/tr/selector_train_outcomes.npz"
    require_hash(inputs, INPUT_SHA)
    require_hash(reward, REWARD_SHA)
    train = load_fields(inputs, INPUT_FIELDS)
    outcome_identity = load_fields(reward, ("question_ids", "canonical_hashes", "generation_seeds"))
    for key in ("question_ids", "canonical_hashes"):
        if not np.array_equal(train[key], outcome_identity[key]) or len(set(train[key])) != 600:
            raise ValueError(f"Bad original train600 identity: {key}")
    if (outcome_identity["generation_seeds"].tolist() != seeds
            or train["embeddings"].shape != (600, 384)
            or not np.isfinite(train["embeddings"]).all()
            or sorted(np.unique(train["task_labels"], return_counts=True)[1]) != [300, 300]
            or set(train["task_labels"]) != {"gsm8k", "bbh"}):
        raise ValueError("Wrong original train600 shape, tasks, or seed order")
    return train, reward


def original_feature_module(asset_root):
    e8_root = asset_root / "research/e8_decision_focused_combination_learning"
    path = e8_root / "public/qualification/e8_assets.py"
    require_hash(path, FEATURE_SOURCE_SHA)
    sys.path.insert(0, str(path.parent))
    import e8_assets
    return e8_assets, e8_root


def integer_rewards(path, rows):
    observed, audit = read_reward_rows(path, rows)
    if observed.shape != (3, len(rows), 20) or not np.isin(observed, (0, 1)).all():
        raise ValueError("Original reward records are not complete binary observations")
    return observed.astype(np.int64), audit["decoded_reward_values"]


def choose_ridge(operator, log, embeddings, rewards, tasks, subtasks, fold, grid):
    rows = np.arange(len(embeddings))
    inner = operator.inner_splits({"task": tasks, "sub": subtasks}, rows, 20261113 + fold)
    logged = log["candidates"][0]["fits"]
    if (len(inner) != 3 or len(logged) != 3 or sorted(np.concatenate([held for _, held in inner])) != rows.tolist()
            or any(sorted(np.r_[h, e]) != rows.tolist() for h, e in inner)):
        raise ValueError("Invalid inner cross-validation split")
    estimates = np.zeros((len(grid), len(rows)), dtype=np.float64)
    for (historical, held), original in zip(inner, logged, strict=True):
        if not (np.array_equal(historical, original["fitting_rows"]) and np.array_equal(held, original["held_rows"])):
            raise ValueError("Ridge inner folds do not match the frozen E8 rows")
        features = operator.fit_features(embeddings[historical])
        phi_train = features.phi(embeddings[historical])
        phi_held = features.phi(embeddings[held])
        for idx, alpha in enumerate(grid):
            model = operator.ridge_weights(phi_train, rewards[historical], raw_alpha=alpha)
            decisions = (phi_held @ model).argmax(axis=1)
            estimates[idx, held] = rewards[held, decisions]
    balanced = np.mean([estimates[:, tasks == task].mean(axis=1) for task in ("gsm8k", "bbh")], axis=0)
    selected = max(alpha for alpha, accuracy in zip(grid, balanced, strict=True)
                   if accuracy >= balanced.max() - 1e-12)
    return selected, [{"raw_alpha": alpha, "inner_accuracy_pct": float(100 * score)}
                      for alpha, score in zip(grid, balanced, strict=True)]


def first_policy(scores):
    return np.eye(scores.shape[1])[scores.argmax(axis=1)]


def sky_policies(scores, counts):
    support = scores == scores.max(axis=1, keepdims=True)
    historical = np.where(support, counts, -1)
    priority = support & (historical == historical.max(axis=1, keepdims=True))
    return first_policy(scores), uniform(support), uniform(priority)


def policy_geometry(scores, support):
    margins = np.diff(np.sort(scores, axis=1)[:, -2:], axis=1)[:, 0]
    return {"strict_unique": int(((support > 0).sum(axis=1) == 1).sum()),
            "margin_quantiles": np.quantile(margins, [0, .1, .5, .9, 1]).tolist()}


def fit(asset_root, private_root, protocol):
    from neural import check_gradient, fit_neural_listnet

    begun = time.perf_counter()
    if private_root.exists():
        raise FileExistsError("Choose a new private fit output directory")
    gradient_error = check_gradient()
    operator, e8_root = original_feature_module(asset_root)
    train, reward_file = source_assets(asset_root, protocol["seed_order"])
    fold_reference = read_json(HERE.parent / "e19_symmetric_tie_constraint/FROZEN_TSPO_REUSE_AUDIT.json")
    previous = read_json(HERE.parent / "e26_public_routing_audit/RESULTS.json")
    by_fold = []
    seen = []
    for fold, (h, e) in enumerate(operator.outer_splits(asset_root)):
        fold_start = time.perf_counter()
        packet_path = asset_root / f"research/e13_partial_combination_feedback/private/inputs/fold{fold}/INPUTS_ONLY.npz"
        fold_sha = require_hash(packet_path, fold_reference["folds"][fold]["input_packet_sha256"])
        frozen_fold = load_fields(packet_path, ("H", "E"))
        if not (np.array_equal(h, frozen_fold["H"]) and np.array_equal(e, frozen_fold["E"])):
            raise ValueError("Original E8 and E19 outer folds disagree")
        observed, decoded = integer_rewards(reward_file, h)
        successes = observed.sum(axis=0)
        targets = successes / 3
        history, evaluation = train["embeddings"][h], train["embeddings"][e]
        # The E26 distance-tie convention orders the history by original global index.
        order = np.argsort(h)
        counts, _, prior, _, _, neighbors = neighbor_scores(history[order], evaluation, successes[order])
        previous_fold = previous["existing_oof_fold_diagnostics"][fold]
        if (neighbors != protocol["sky"]["neighbors"] or tie_counts(counts) !=
                {**previous_fold["evaluation"], "best_set_size_counts":
                 {int(k): v for k, v in previous_fold["evaluation"]["best_set_size_counts"].items()}}
                or np.bincount(counts.argmax(axis=1), minlength=20).tolist() != previous_fold["first_exact_max_frequency"]):
            raise ValueError("Recomputed SkyLLM fold disagrees with E26 diagnostics")
        policies = dict(zip(BASELINES[:3], sky_policies(counts, prior), strict=True))
        fixed_action = int(prior.argmax())
        policies["Fixed"] = np.eye(20)[np.repeat(fixed_action, len(e))]
        log_path = e8_root / f"private/training/fold{fold}/M0_tuned/FIT_LOG.json"
        alpha, inner_scores = choose_ridge(operator, read_json(log_path), history, targets, train["task_labels"][h],
                                           train["subtask_labels"][h], fold, protocol["ridge"]["raw_alpha_grid"])
        transform = operator.fit_features(history)
        phi_train, phi_eval = transform.phi(history), transform.phi(evaluation)
        ridge = operator.ridge_weights(phi_train, targets, raw_alpha=alpha)
        ridge_scores = phi_eval @ ridge
        policies["Ridge"] = first_policy(ridge_scores)
        scores = {"Sky": counts / (3 * neighbors), "Ridge": ridge_scores}
        diagnostics = {}
        for seed in protocol["neural"]["optimizer_seeds"]:
            for beta in (1, 0):
                name = neural_name(seed, beta)
                value, diagnostics[name] = fit_neural_listnet(phi_train, targets, phi_eval, seed=seed, beta=beta)
                scores[name] = value
                mask = value >= value.max(axis=1, keepdims=True) - 1e-8
                policies[name] = uniform(mask)
                diagnostics[name]["deployment"] = policy_geometry(value, policies[name])
        names = list(BASELINES) + [neural_name(seed, beta) for seed in protocol["neural"]["optimizer_seeds"]
                                   for beta in (1, 0)]
        path = private_root / f"fold{fold}.npz"
        write_npz(path, history_indices=h, evaluation_indices=e, policies=np.stack([policies[n] for n in names]),
                  policy_names=np.asarray(names), historical_action_successes=prior, ridge_weights=ridge,
                  score_names=np.asarray(list(scores)), scores=np.stack(list(scores.values())), **transform.arrays())
        seen.extend(e.tolist())
        by_fold.append({
            "fold": fold,
            "fold_input_sha256": fold_sha,
            "ridge_inner_log_sha256": sha(log_path),
            "ridge_raw_alpha": alpha,
            "ridge_inner_scores": inner_scores,
            "fixed_action_zero_based": fixed_action,
            "sky_exact_ties": previous_fold["evaluation"]["multiple_best"],
            "all_tie_history_questions": int((successes == successes[:, :1]).all(axis=1).sum()),
            "decoded_history_rewards": decoded,
            "evaluation_rewards_decoded_for_this_fold_before_freeze": 0,
            "prediction_sha256": sha(path),
            "neural": diagnostics,
            "wall_seconds": time.perf_counter() - fold_start,
        })
        print(f"Fold {fold}: saved {len(names)} policies for {len(e)} held-out rows", flush=True)
    if sorted(seen) != list(range(600)):
        raise ValueError("Frozen holdout predictions must cover exactly 600 original rows")
    write_json(private_root / "FROZEN.json", {
        "protocol_sha256": sha(PROTOCOL), "trainer_sha256": sha(HERE / "neural.py"),
        "runner_sha256": sha(Path(__file__)), "scorer_sha256": sha(HERE / "metrics.py"),
        "historical_inputs_sha256": INPUT_SHA, "historical_rewards_sha256": REWARD_SHA,
        "e8_features_sha256": FEATURE_SOURCE_SHA, "gradient_check_max_error": gradient_error,
        "folds": by_fold, "fit_wall_seconds": time.perf_counter() - begun,
    })


def neural_name(seed, beta):
    return f"Neural_ListNet_{'all' if beta else 'mute'}_{seed}"


def requested_pairs(seeds):
    pairs = [("Sky_U", "Sky_First"), ("Sky_P", "Sky_First"), ("Sky_P", "Sky_U"), ("Ridge", "Fixed")]
    for name in ("Sky_First", "Sky_U", "Sky_P"):
        pairs.extend(((name, "Ridge"), (name, "Fixed")))
    for seed in seeds:
        native, mute = neural_name(seed, 1), neural_name(seed, 0)
        pairs.extend([(mute, native), (native, "Ridge"), (native, "Fixed"),
                      (mute, "Ridge"), (mute, "Fixed")])
    return pairs


def evaluate(asset_root, private_root, output, protocol):
    from metrics import summarize

    receipt_path = private_root / "FROZEN.json"
    receipt = read_json(receipt_path)
    require_hash(PROTOCOL, receipt["protocol_sha256"])
    require_hash(Path(__file__), receipt["runner_sha256"])
    require_hash(HERE / "neural.py", receipt["trainer_sha256"])
    require_hash(HERE / "metrics.py", receipt["scorer_sha256"])
    train, reward_file = source_assets(asset_root, protocol["seed_order"])
    operator, _ = original_feature_module(asset_root)
    fold_ids = np.full(600, -1, dtype=int)
    merged = {}
    for manifest, (h, e) in zip(receipt["folds"], operator.outer_splits(asset_root), strict=True):
        fold = manifest["fold"]
        path = private_root / f"fold{fold}.npz"
        require_hash(path, manifest["prediction_sha256"])
        frozen = load_fields(path, ("history_indices", "evaluation_indices", "policy_names", "policies"))
        if (not np.array_equal(h, frozen["history_indices"]) or not np.array_equal(e, frozen["evaluation_indices"])
                or (fold_ids[e] != -1).any()):
            raise ValueError("Wrong fold pair or duplicate holdout")
        fold_ids[e] = fold
        for name, policy in zip(frozen["policy_names"].tolist(), frozen["policies"], strict=True):
            if name not in merged:
                merged[name] = np.full((600, 20), np.nan)
            merged[name][e] = policy
    if (fold_ids < 0).any() or any(not np.isfinite(p).all() for p in merged.values()):
        raise ValueError("Incomplete frozen holdout predictions")
    # Outer correctness is decoded for scoring only after checking all policy hashes.
    observed, decoded = integer_rewards(reward_file, np.arange(600))
    result = summarize(merged, observed, train["task_labels"], train["subtask_labels"], protocol["seed_order"],
                       fold_ids, requested_pairs(protocol["neural"]["optimizer_seeds"]))
    result.update({
        "stage": "E27",
        "status": "COMPLETE_HISTORICAL_OOF_REPLAY",
        "freeze_sha256": sha(receipt_path),
        "fit_summary": receipt,
        "evaluation": {
            "independent_units_in_historical_cohort": 600,
            "generation_events_per_policy": 1800,
            "decoded_cached_rewards": decoded,
            "actual_new_member_calls": 0,
            "actual_new_aggregation_calls": 0,
            "actual_new_tokens": 0,
            "actual_new_encoder_forwards": 0,
            "new_neural_fits": 30,
            "new_ridge_fits": 50,
            "E24_fresh_or_sealed_reward_reads": 0,
        },
    })
    write_json(output, result)
    print("Scored all frozen policies on the 600 original holdout rows.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--phase", choices=("fit", "evaluate"), required=True)
    cli.add_argument("--asset-root", type=Path, required=True)
    cli.add_argument("--private-root", type=Path, required=True)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    spec = read_json(PROTOCOL)
    if args.phase == "fit":
        fit(args.asset_root, args.private_root, spec)
    else:
        if args.output is None:
            cli.error("--output is required for evaluate")
        evaluate(args.asset_root, args.private_root, args.output, spec)
