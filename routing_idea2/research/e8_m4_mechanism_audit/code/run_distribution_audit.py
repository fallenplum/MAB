"""Small E8 M4 distribution and true-optimum tie audit.

Training uses one legal history480 pack at a time. All intervention predictions
are sealed before the separate evaluation mode opens the original train600
outcomes. No held-out dataset is used here.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from scipy import sparse

E8 = Path(r"E:\语言模型组合路由\research\e8_decision_focused_combination_learning")
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(E8 / "private/dependencies"), str(E8 / "public/qualification"),
                str(E8 / "public/training")]
from e8_assets import FeatureMap, matched_mean_loss_lambda, load_historical
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import clarabel

TEAMS = np.array(list(itertools.combinations(range(6), 3)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item.item()) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def feature_map(state):
    pca = PCA(16, svd_solver="full")
    pca.components_ = state["pca_components"]
    pca.mean_ = state["pca_mean"]
    pca.n_features_in_ = 384
    pca.n_components_ = 16
    pca.explained_variance_ = np.ones(16)
    pca.whiten = False
    scaler = StandardScaler()
    scaler.mean_ = state["scaler_mean"]
    scaler.scale_ = state["scaler_scale"]
    scaler.n_features_in_ = 16
    return FeatureMap(pca, scaler)


def optimal_weights(reward, rule):
    if rule == "uniform":
        mask = reward == reward.max(axis=1, keepdims=True)
        return mask / mask.sum(axis=1, keepdims=True)
    chosen = reward.argmax(axis=1) if rule == "lexicographic" else 19 - reward[:, ::-1].argmax(axis=1)
    result = np.zeros_like(reward)
    result[np.arange(len(reward)), chosen] = 1
    return result


def fit_exact(phi, reward, regularization, rule):
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    n, dimension = phi.shape
    target = optimal_weights(reward, rule)
    p = sparse.diags(np.r_[np.full(dimension * 20, 2 * regularization), np.zeros(n)], format="csc")
    q = np.r_[(-(2 / n) * (phi.T @ target)).reshape(-1, order="F"), np.full(n, 1 / n)]
    action_blocks = sparse.kron(sparse.eye(20, format="csc"), 2 * sparse.csc_matrix(phi), format="csc")
    epigraph_blocks = sparse.vstack([-sparse.eye(n, format="csc")] * 20, format="csc")
    a = sparse.hstack([action_blocks, epigraph_blocks], format="csc")
    b = reward.reshape(-1, order="F")
    settings = clarabel.DefaultSettings()
    settings.tol_gap_abs = 1e-9
    settings.tol_gap_rel = 1e-9
    settings.tol_feas = 1e-9
    settings.max_iter = 1000
    settings.max_threads = 1
    settings.verbose = False
    solver = clarabel.DefaultSolver(p, q, a, b, [clarabel.NonnegativeConeT(n * 20)], settings)
    solution = solver.solve()
    if solution.status != clarabel.SolverStatus.Solved:
        raise RuntimeError(str(solution.status))
    primal, dual = np.asarray(solution.x), np.asarray(solution.z)
    coefficient = primal[:dimension * 20].reshape(dimension, 20, order="F")
    scores = phi @ coefficient
    direct = np.mean(np.max(2 * scores - reward, axis=1) - 2 * np.sum(target * scores, axis=1)
                     + reward.max(axis=1)) + regularization * np.sum(coefficient ** 2)
    objective_error = abs(direct - (solution.obj_val + reward.max(axis=1).mean()))
    violation = np.maximum(a @ primal - b, 0).max()
    gap = abs(solution.obj_val - solution.obj_val_dual)
    assert objective_error < 1e-7 and violation < 1e-7 and gap < 1e-7
    return coefficient, dict(status=str(solution.status), exact_loss=float(direct),
        direct_epigraph_error=float(objective_error), primal_violation=float(violation),
        stationarity=float(np.max(np.abs(p @ primal + q + a.T @ dual))),
        primal_dual_gap=float(gap), iterations=int(solution.iterations),
        cpu_seconds=time.process_time() - start_cpu, wall_seconds=time.perf_counter() - start_wall,
        regularization=float(regularization), tie_rule=rule, coefficient_dimension=dimension * 20)


def freeze_design():
    target = OUT / "private/DESIGN_LOCK.json"
    if target.exists():
        before = read(target)
        assert before["source_sha256"] == sha(__file__)
        return before
    design = dict(schema="M4_SMALL_MECHANISM_DISTRIBUTION_V1",
        timestamp_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        original_E8_protocol_sha256=sha(E8 / "public/E8_PROTOCOL.json"),
        E8_index_sha256=sha(E8 / "private/training/E8_FROZEN_TRAINING_INDEX.json"),
        E8_prediction_sha256=sha(E8 / "private/evaluation/PREDICTIONS_AND_CALIBRATION_FROZEN.npz"),
        folds=5, historical_questions_per_fold=480, evaluation_questions_per_fold=120,
        capacity="frozen PCA16 and constant plus linear20 for full interventions",
        interventions=["full_spo_lexicographic", "full_spo_reverselex", "full_spo_uniform",
                       "constant_spo_lexicographic", "constant_spo_uniform"],
        fits=25, regularization="original selected M4 lambda multiplier per fold, no retuning",
        solver="Clarabel0.11.1 exact epigraph, absolute/relative/feasibility1e-9",
        service_tie="exact numpy.argmax, lexicographic",
        fit_outcome_access="current history480 only",
        inference_outcome_access=False, fresh400_access=False, sealed200_access=False,
        physical_LLM_calls=0, evidence="exploratory frozen-development diagnostic",
        original_results_modified=False)
    put(target, design)
    return design


def train():
    freeze_design()
    destination = OUT / "private"
    if (destination / "INTERVENTIONS_FROZEN.npz").exists():
        raise FileExistsError("Frozen intervention predictions already exist")
    frozen = arrays(E8 / "private/evaluation/PREDICTIONS_AND_CALIBRATION_FROZEN.npz")
    records, output = [], {}
    names = ["train_global_fixed", "constant_ridge_prior", "full_spo_lexicographic", "full_spo_reverselex",
             "full_spo_uniform", "constant_spo_lexicographic", "constant_spo_uniform"]
    for name in names:
        output["raw__" + name] = np.empty((600, 20))
        output["actions__" + name] = np.empty(600, dtype=int)
    output["fold"] = frozen["fold"]
    training_ties = []
    for fold in range(5):
        folder = E8 / f"private/training/fold{fold}/M4"
        history_path = E8 / f"private/qualification/fold_packs/fold{fold}/history480.npz"
        inputs_path = E8 / f"private/qualification/fold_packs/fold{fold}/evaluation_inputs_only120.npz"
        state = arrays(folder / "model_state.npz")
        history, inputs = arrays(history_path), arrays(inputs_path)
        assert "Y" not in inputs
        assert not set(history["global_indices"]) & set(inputs["global_indices"])
        rows = inputs["global_indices"]
        fmap = feature_map(state)
        phi, eval_phi = fmap.phi(history["E"]), fmap.phi(inputs["E"])
        old_scores = eval_phi @ state["coefficient"]
        assert np.max(abs(old_scores - frozen["raw__M4"][rows])) < 1e-11
        assert np.array_equal(old_scores.argmax(axis=1), frozen["raw__M4"][rows].argmax(axis=1))
        reward = history["Y"].mean(axis=1)
        mean = history["Y"].mean(axis=(0, 1))
        global_action = int(mean.argmax())
        assert np.all(frozen["default_actions"][rows] == global_action)
        output["raw__train_global_fixed"][rows] = mean
        output["actions__train_global_fixed"][rows] = global_action
        prior = mean * 480 / (480 + 100 / 3)
        output["raw__constant_ridge_prior"][rows] = prior
        output["actions__constant_ridge_prior"][rows] = int(prior.argmax())
        log = read(folder / "FIT_LOG.json")
        multiplier = log["selected_parameters"]["lambda_multiplier"]
        regularization = matched_mean_loss_lambda(480, 100 * multiplier)
        all_tied = np.all(reward == reward[:, :1], axis=1)
        counts = (reward == reward.max(axis=1, keepdims=True)).sum(axis=1)
        tie_record = dict(fold=fold, all20_tied=int(all_tied.sum()), multiple_best=int((counts > 1).sum()),
            unique_best=int((counts == 1).sum()), lambda_multiplier=multiplier,
            true_label_mass={rule: optimal_weights(reward, rule).sum(axis=0) for rule in
                             ["lexicographic", "reverselex", "uniform"]},
            all20_tied_label_mass={rule: optimal_weights(reward[all_tied], rule).sum(axis=0) for rule in
                                  ["lexicographic", "reverselex", "uniform"]})
        training_ties.append(tie_record)
        for name in names[2:]:
            short, rule = name.rsplit("_", 1)
            fitting_phi = phi if short == "full_spo" else np.ones((480, 1))
            test_phi = eval_phi if short == "full_spo" else np.ones((120, 1))
            coefficient, diagnostic = fit_exact(fitting_phi, reward, regularization, rule)
            scores = test_phi @ coefficient
            output["raw__" + name][rows] = scores
            output["actions__" + name][rows] = scores.argmax(axis=1)
            model_path = destination / f"models/fold{fold}_{name}.npz"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(model_path, coefficient=coefficient, **fmap.arrays())
            diagnostic.update(fold=fold, variant=name, history_sha256=sha(history_path),
                inputs_sha256=sha(inputs_path), original_model_sha256=sha(folder / "model_state.npz"),
                model_sha256=sha(model_path), model_path=str(model_path),
                top_gap_min=float(np.min(np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2])),
                chosen_action_counts=np.bincount(scores.argmax(axis=1), minlength=20))
            if name == "full_spo_lexicographic":
                diagnostic["changes_vs_original_M4"] = int(np.sum(scores.argmax(axis=1) != frozen["raw__M4"][rows].argmax(axis=1)))
            records.append(diagnostic)
            print(f"fold={fold} {name} status={diagnostic['status']} cpu={diagnostic['cpu_seconds']:.3f}", flush=True)
    np.savez_compressed(destination / "INTERVENTIONS_FROZEN.npz", **output)
    receipt = dict(source_sha256=sha(__file__), design_sha256=sha(destination / "DESIGN_LOCK.json"),
        prediction_sha256=sha(destination / "INTERVENTIONS_FROZEN.npz"),
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), fits=records,
        training_ties=training_ties, evaluation_outcomes_opened=False,
        total_cpu_seconds=sum(r["cpu_seconds"] for r in records),
        total_wall_seconds=sum(r["wall_seconds"] for r in records))
    put(destination / "INTERVENTIONS_FROZEN_RECEIPT.json", receipt)


def distribution(scores, mask):
    q = scores[mask]
    action = q.argmax(axis=1)
    counts = np.bincount(action, minlength=20)
    p = counts[counts > 0] / len(action)
    entropy = -np.sum(p * np.log(p))
    sorted_scores = np.sort(q, axis=1)
    gaps = sorted_scores[:, -1] - sorted_scores[:, -2]
    return dict(questions=len(action), counts=counts, probabilities=counts / len(action),
        top1_action=int(counts.argmax()), top1_count=int(counts.max()),
        top1_fraction=float(counts.max() / len(action)), top3_fraction=float(np.sort(counts)[-3:].sum() / len(action)),
        entropy_nats=float(entropy), effective_combinations=float(np.exp(entropy)),
        herfindahl=float(np.sum((counts / len(action)) ** 2)),
        selected_mean_index=float(action.mean()), score_gap_min=float(gaps.min()),
        score_gap_median=float(np.median(gaps)), near_tie_counts={str(v): int(np.sum(gaps < v)) for v in [1e-8, 1e-10, 1e-12]},
        exact_top_tie_questions=int(np.sum((q == q.max(axis=1, keepdims=True)).sum(axis=1) > 1)))


def paired_ci(difference, task, fold, seed=2026111401, repeats=10000):
    rng = np.random.default_rng(seed)
    sample_means = np.zeros(repeats)
    for t in ("gsm8k", "bbh"):
        for f in range(5):
            values = difference[(task == t) & (fold == f)]
            if not len(values):
                continue
            sample = rng.integers(0, len(values), size=(repeats, len(values)))
            sample_means += values[sample].mean(axis=1) * len(values) / len(difference)
    return dict(difference=float(difference.mean()), ci95=np.quantile(sample_means, [.025, .975]),
                independent_questions=len(difference), repetitions=repeats,
                interpretation="exploratory conditional-development question-cluster paired interval")


def evaluate():
    private = OUT / "private"
    receipt = read(private / "INTERVENTIONS_FROZEN_RECEIPT.json")
    assert receipt["source_sha256"] == sha(__file__)
    assert receipt["prediction_sha256"] == sha(private / "INTERVENTIONS_FROZEN.npz")
    original = arrays(E8 / "private/evaluation/PREDICTIONS_AND_CALIBRATION_FROZEN.npz")
    new = arrays(private / "INTERVENTIONS_FROZEN.npz")
    raw = {"M0_original": original["raw__M0_original"], "M4_original": original["raw__M4"],
           "M4_old_uniformtie": original["raw__M4_uniformtie"]}
    raw.update({key.removeprefix("raw__"): value for key, value in new.items() if key.startswith("raw__")})
    frozen_hashes = {str(private / "INTERVENTIONS_FROZEN.npz"): sha(private / "INTERVENTIONS_FROZEN.npz")}
    # The first access to evaluation outcomes occurs after every intervention is sealed.
    data, refs, _ = load_historical(E8.parents[1])
    put(private / "EVALUATION_ACCESS_RECEIPT.json", dict(
        opened_at_utc=datetime.now(timezone.utc).isoformat(), predictions_frozen=frozen_hashes,
        data_scope="old train600 only", refs=refs, fresh400_access=False, sealed200_access=False))
    y, task, sub, fold = data["Y"], data["task"], data["sub"], original["fold"]
    assert y.shape == (600, 3, 20)
    groups = {"all": np.ones(600, bool), "gsm8k": task == "gsm8k", "bbh": task == "bbh"}
    groups.update({f"fold{j}": fold == j for j in range(5)})
    actions, rewards, distributions = {}, {}, {}
    for name, scores in raw.items():
        action = scores.argmax(axis=1)
        actions[name] = action
        rewards[name] = y[np.arange(600), :, action]
        distributions[name] = {key: distribution(scores, mask) for key, mask in groups.items()}
    assert np.sum(actions["M4_original"] != actions["M0_original"]) == 543
    comparisons = {}
    for name in raw:
        diff = rewards[name] - rewards["M0_original"]
        pair = paired_ci(diff.mean(axis=1), task, fold)
        pair.update(accuracy=float(rewards[name].mean()), seed_accuracy=rewards[name].mean(axis=0),
            gsm8k_accuracy=float(rewards[name][task == "gsm8k"].mean()),
            bbh_accuracy=float(rewards[name][task == "bbh"].mean()),
            changes_vs_original_M4=int(np.sum(actions[name] != actions["M4_original"])),
            changes_vs_M0=int(np.sum(actions[name] != actions["M0_original"])),
            corrected_events=int((diff > 0).sum()), destroyed_events=int((diff < 0).sum()),
            net_correct_events=int(diff.sum()),
            versus_M4=paired_ci((rewards[name] - rewards["M4_original"]).mean(axis=1), task, fold),
            per_subtask={str(s): dict(n=int(np.sum((task == "bbh") & (sub == s))),
                accuracy=float(rewards[name][(task == "bbh") & (sub == s)].mean()))
                for s in sorted(set(sub[task == "bbh"].tolist()))})
        comparisons[name] = pair
    tie_counts = (y.mean(axis=1) == y.mean(axis=1).max(axis=1, keepdims=True)).sum(axis=1)
    result = dict(schema="M4_DISTRIBUTION_AND_TIE_AUDIT_V1", status="COMPLETE",
        design=read(private / "DESIGN_LOCK.json"), training=receipt,
        original_frozen_prediction_sha256=sha(E8 / "private/evaluation/PREDICTIONS_AND_CALIBRATION_FROZEN.npz"),
        teams_zero_based=TEAMS, combinations_display_index="zero-based lexicographic 6choose3",
        distributions=distributions, comparisons=comparisons,
        empirical_ties=dict(unique_true_best=int(np.sum(tie_counts == 1)),
            multiple_true_best=int(np.sum(tie_counts > 1)), all20_true_tied=int(np.sum(tie_counts == 20))),
        original_M4_vs_M0=dict(action_changes=543, corrected_events=int(np.sum(rewards["M4_original"] > rewards["M0_original"])),
            destroyed_events=int(np.sum(rewards["M4_original"] < rewards["M0_original"])),
            net_correct_events=int(np.sum(rewards["M4_original"] - rewards["M0_original"]))),
        total_new_physical_LLM_calls=0, prior_E8_results_modified=False)
    np.savez_compressed(private / "EVALUATED_PAIRED_ACTIONS.npz", task=task, sub=sub, fold=fold,
        **{"actions__" + k: v for k, v in actions.items()}, **{"reward__" + k: v for k, v in rewards.items()})
    put(OUT / "public/M4_DISTRIBUTION_AND_TIE_RESULTS.json", result)
    print(json.dumps({k: {a: v[a] for a in ["accuracy", "changes_vs_original_M4", "net_correct_events"]}
                      for k, v in comparisons.items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "train", "evaluate"])
    args = parser.parse_args()
    {"freeze": freeze_design, "train": train, "evaluate": evaluate}[args.mode]()
