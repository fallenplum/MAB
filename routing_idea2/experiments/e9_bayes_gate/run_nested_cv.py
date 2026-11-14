"""Leakage-safe nested training, cached fixed controls, and prediction freezing.

This module never opens the train600 outcome archive. Each learner receives
only an original legal history packet and a reward-free evaluation packet.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from pathlib import Path
import time

from common import (
    ASSETS, BASE_SEED, CODE, E8, MULTIPLIERS, PERMUTATION_SEEDS, PRIVATE,
    RESULTS, SHRINKAGES, Timer, array_hash, balanced_accuracy, choice,
    jsonable, legal_fold, legal_metadata, load_npz, permute_blocks, prior_scores,
    read_json, selected_candidate, sha256, write_json,
)
import numpy as np
from e8_assets import fit_features, inner_splits
from train_hsmr import fit_hsmr, predict_historical_prior
from train_psmr import fit_psmr, predict_psmr


def candidates():
    values = []
    for loss, multiplier in itertools.product(("bce", "brier"), MULTIPLIERS):
        values.append(dict(family="PSMR", loss=loss, multiplier=multiplier))
    for shrinkage, multiplier in itertools.product(SHRINKAGES, MULTIPLIERS):
        values.append(dict(family="HSMR", shrinkage=shrinkage, multiplier=multiplier))
    return values


def candidate_key(config):
    detail = config.get("loss", str(config.get("shrinkage")))
    return f"{config['family']}_{detail}_m{config['multiplier']}"


def prepare_features(fold, history, evaluation):
    root = PRIVATE / "preprocessing" / f"fold{fold}"
    receipt_path = root / "FROZEN.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        for name, digest in receipt["files"].items():
            if sha256(root / name) != digest:
                raise ValueError("Preprocessing cache modified")
        return receipt
    root.mkdir(parents=True, exist_ok=True)
    rows = np.arange(len(history["Y"]))
    splits = inner_splits(history, rows, BASE_SEED + fold)
    old = read_json(E8 / f"private/training/fold{fold}/M0_tuned/FIT_LOG.json")
    original = old["candidates"][0]["fits"]
    records = []
    with Timer() as timer:
        for inner, (fit, held) in enumerate(splits):
            if not np.array_equal(fit, original[inner]["fitting_rows"]) or not np.array_equal(held, original[inner]["held_rows"]):
                raise ValueError("Inner split differs from original E8")
            transform = fit_features(history["E"][fit])
            path = root / f"inner{inner}.npz"
            np.savez_compressed(path, fitting_rows=fit, held_rows=held,
                                phi_fit=transform.phi(history["E"][fit]),
                                phi_query=transform.phi(history["E"][held]), **transform.arrays())
            records.append(dict(inner=inner, fitting_indices_hash=array_hash(history["global_indices"][fit]),
                                held_indices_hash=array_hash(history["global_indices"][held]),
                                fitting_questions=len(fit), held_questions=len(held)))
        transform = fit_features(history["E"])
        np.savez_compressed(root / "outer.npz", phi_fit=transform.phi(history["E"]),
                            phi_query=transform.phi(evaluation["E"]), **transform.arrays())
    saved = load_npz(E8 / f"private/training/fold{fold}/M0_original/model_state.npz")
    restored = np.column_stack((np.ones(len(evaluation["E"])),
        (((evaluation["E"] - saved["pca_mean"]) @ saved["pca_components"].T)
         - saved["scaler_mean"]) / saved["scaler_scale"]))
    maximum_error = float(np.max(np.abs(restored - transform.phi(evaluation["E"]))))
    if maximum_error > 1e-10:
        raise ValueError("Original feature recipe no longer reproduces saved PCA/scaler")
    receipt = dict(fold=fold, inner_splits=records, preprocessing_cost=timer.record,
                   outer_feature_restore_max_error=maximum_error,
                   files={p.name: sha256(p) for p in sorted(root.glob("*.npz"))})
    write_json(receipt_path, receipt)
    return receipt


def fit_cached(root, config, phi, rewards, task, sub):
    key = candidate_key(config)
    path, receipt_path = root / (key + ".npz"), root / (key + ".json")
    inputs = dict(phi=array_hash(phi), rewards=array_hash(rewards), task=array_hash(task), sub=array_hash(sub))
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["input_hashes"] != inputs or sha256(path) != receipt["model_sha256"]:
            raise ValueError("Fitting cache lineage mismatch")
        return load_npz(path)["weights"], receipt, True
    root.mkdir(parents=True, exist_ok=True)
    alpha = 100.0 * config["multiplier"] / 3.0
    if config["family"] == "PSMR":
        fit = fit_psmr(phi, rewards, alpha / (len(phi) * 20), config["loss"])
    else:
        fit = fit_hsmr(phi, rewards, task, sub, alpha, config["shrinkage"])
    if not fit["diagnostics"]["optimizer_success"]:
        write_json(root / (key + "_FAILED.json"), fit["diagnostics"])
        raise RuntimeError("Optimizer failed, preserve log and stop: " + key)
    np.savez_compressed(path, weights=fit["weights"])
    receipt = dict(config=config, input_hashes=inputs, diagnostics=fit["diagnostics"], model_sha256=sha256(path))
    write_json(receipt_path, receipt)
    return fit["weights"], receipt, False


def predict(phi, weights, config):
    return predict_psmr(phi, weights) if config["family"] == "PSMR" else phi @ weights


def run_fold(fold, permutation_seed=None, smoke=False):
    history, evaluation = legal_fold(fold)
    prefix = "main" if permutation_seed is None else f"controls/seed{permutation_seed}"
    root = PRIVATE / prefix / f"fold{fold}"
    done_path = root / "PREDICTIONS_FROZEN.json"
    if done_path.exists() and not smoke:
        done = read_json(done_path)
        if sha256(root / "PREDICTIONS.npz") != done["prediction_sha256"]:
            raise ValueError("Frozen predictions changed")
        return done
    with Timer() as timer:
        feature_receipt = prepare_features(fold, history, evaluation)
        configs = candidates()
        records = [dict(**config, inner_scores=np.full((480, 20), np.nan)) for config in configs]
        default_inner_scores = np.full((480, 20), np.nan)
        fits, reused, block_maps = [], 0, []
        for inner in range(1 if smoke else 3):
            feature = load_npz(PRIVATE / f"preprocessing/fold{fold}/inner{inner}.npz")
            fit, held = feature["fitting_rows"], feature["held_rows"]
            reward = history["Y"][fit]
            if permutation_seed is not None:
                reward, permutation = permute_blocks(reward, history["task"][fit], history["sub"][fit],
                                                      permutation_seed, fold, inner)
                block_maps.append(dict(region=f"inner{inner}", permutation=permutation))
            default_inner_scores[held] = prior_scores(history["Y"][fit], history["task"][fit], history["sub"][fit],
                                                     history["task"][held], history["sub"][held])
            for config, record in zip(configs, records):
                weights, receipt, cached = fit_cached(root / f"inner{inner}", config, feature["phi_fit"], reward,
                                                      history["task"][fit], history["sub"][fit])
                record["inner_scores"][held] = predict(feature["phi_query"], weights, config)
                fits.append(dict(region=f"inner{inner}", **receipt))
                reused += int(cached)
        if smoke:
            write_json(PRIVATE / "RESOURCE_PILOT.json", dict(status="PASS",
                       cpu_s=time.process_time()-timer.cpu, wall_s=time.perf_counter()-timer.wall,
                       fits=fits, candidate_heads=len(fits), note="First legal inner320 fits, no outer rewards accessed"))
            return dict(status="SMOKE_COMPLETE", fits=len(fits))
        for row in records:
            row["inner_accuracy"] = balanced_accuracy(history["Y"], choice(row.pop("inner_scores")), history["task"])
        selected = {name: selected_candidate(records, name) for name in ("PSMR", "HSMR")}
        default_inner_acc = balanced_accuracy(history["Y"], choice(default_inner_scores), history["task"])
        best_family = max(("HSMR", "PSMR"), key=lambda name: selected[name]["inner_accuracy"])
        selected_rule = best_family if selected[best_family]["inner_accuracy"] > default_inner_acc + 1e-12 else "SubtaskFixed"
        feature = load_npz(PRIVATE / f"preprocessing/fold{fold}/outer.npz")
        scores = {}
        reward = history["Y"]
        if permutation_seed is not None:
            reward, permutation = permute_blocks(reward, history["task"], history["sub"], permutation_seed, fold, 3)
            block_maps.append(dict(region="outer", permutation=permutation))
        for family in ("PSMR", "HSMR"):
            config = {k: v for k, v in selected[family].items() if k != "inner_accuracy"}
            weights, receipt, cached = fit_cached(root / "outer", config, feature["phi_fit"], reward,
                                                  history["task"], history["sub"])
            scores[family] = predict(feature["phi_query"], weights, config)
            fits.append(dict(region="outer", **receipt))
            reused += int(cached)
        metadata = legal_metadata(fold, evaluation["global_indices"])
        for method, kind in (("GlobalFixed", "global"), ("TaskFixed", "task"), ("SubtaskFixed", "subtask")):
            scores[method] = prior_scores(history["Y"], history["task"], history["sub"], metadata["task"], metadata["sub"], kind)
        scores["PriorShrink"] = predict_historical_prior(history["Y"], history["task"], history["sub"],
                                                         metadata["task"], metadata["sub"])
        # The fallback may use declared task/subtask metadata. Learned heads remain text-only.
        scores["E9Rule"] = scores[selected_rule].copy()
        keys = list(scores)
        np.savez_compressed(root / "PREDICTIONS.npz", methods=np.array(keys),
                            scores=np.stack([scores[name] for name in keys]),
                            actions=np.stack([choice(scores[name]) for name in keys]),
                            evaluation_indices=evaluation["global_indices"])
        write_json(root / "FITTING_LOG.json", dict(fits=fits, candidates=records, selected=selected,
                   default_inner_accuracy=default_inner_acc, selected_rule=selected_rule,
                   reused_fit_receipts=reused, block_permutations=block_maps))
    done = dict(status="FROZEN", fold=fold, permutation_seed=permutation_seed,
                selected=selected, selected_rule=selected_rule, fitting_heads=len(fits),
                reused_fit_receipts=reused, time=timer.record,
                prediction_sha256=sha256(root / "PREDICTIONS.npz"),
                fitting_log_sha256=sha256(root / "FITTING_LOG.json"),
                feature_receipt_sha256=sha256(PRIVATE / f"preprocessing/fold{fold}/FROZEN.json"))
    write_json(done_path, done)
    return done


def freeze_protocol():
    audit = read_json(RESULTS / "00_asset_manifest.json")
    if audit.get("status") != "PASS" or read_json(RESULTS / "01_math_qualification.json").get("status") != "PASS":
        raise ValueError("Passing asset and mathematics qualifications must precede training")
    source_files = sorted(CODE.rglob("*.py"))
    protocol = dict(
        experiment="E9_BAYES_GATE_DISTINCT_FROM_HISTORICAL_E9_CSCE",
        evidence="reused_train600_development_cross_validation", samples=600,
        seeds=[42, 137, 2024], models=6, members_per_action=3, actions=20,
        input="original MiniLM-L6-v2 384 -> fitting-only PCA16 -> original StandardScaler -> penalized constant",
        parameters_per_head=340, outer_folds=5, inner_folds=3,
        inner_split_seed="20261113+outer_fold, exact original E8 local rows",
        candidates=candidates(), regularization="lambda=100*multiplier/(3*n_fit*20); Ridge normal alpha=100*multiplier/3",
        hierarchy=dict(task_pseudocount=10, subtask_pseudocount=3, whole_question_LOO=True,
                       support_fallback="smooth LOO subgroup to task to global; n=1 prior .5"),
        calibration="none; optional calibration axis not adopted",
        main_rule="inner-best PSMR/HSMR if strictly greater than SubtaskFixed+1e-12, else SubtaskFixed; family tie HSMR",
        candidate_tie="higher regularization, higher shrinkage, BCE before Brier",
        action_tie="original exact np.argmax in frozen semantic combination order",
        metadata_permission="heads text-only; fixed/prior/fallback baselines use true task/subtask metadata explicitly",
        primary_comparisons=["E9Rule-SubtaskFixed", "E9Rule-M0_original"],
        bootstrap=dict(repeats=10000, strata="outerfold x GSM/BBH", unit="original question with all seed/action/method records",
                       primary_ci="97.5% Bonferroni simultaneous pairwise lower bounds for two-sided95% family coverage",
                       ordinary_ci="95% per-comparison percentile", p_adjustment="Holm for two predeclared paired comparisons"),
        negative_control=dict(repeats=100, seeds=list(PERMUTATION_SEEDS),
            operation="whole-question 3x20 blocks shuffled within task/BBH subtask in each legal fitting region separately",
            selection="same nested grid and inner real-outcome scoring as main, never outer outcomes",
            feature_preprocessing="reward-free and identical fitting indices, exact caches reused across controls"),
        go=dict(positive_simultaneous_primary_lower_bounds=True, positive_generation_directions_at_least=2,
                task_guard="both GSM and BBH point differences >=0 for both main comparators",
                content_guard="main E9Rule accuracy strictly above 95th percentile of 100 complete-rule controls",
                status="GO_FRESH_REQUEST requests separate confirmation design only"),
        permissions=dict(new_llm_calls=0, new_gold=0, validation200_reads=0, fresh400_reads=0, sealed200_reads=0,
                         push=False, merge=False),
        asset_manifest_sha256=sha256(RESULTS / "00_asset_manifest.json"),
        source_files={str(p.relative_to(CODE)): sha256(p) for p in source_files},
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write_json(CODE / "protocol_frozen.json", protocol)
    write_json(PRIVATE / "PROTOCOL_FREEZE_RECEIPT.json", dict(sha256=sha256(CODE / "protocol_frozen.json"),
               created_utc=protocol["created_utc"], actual_heads_before_pilot=0))
    print("PROTOCOL_FROZEN", sha256(CODE / "protocol_frozen.json"), flush=True)


def verify_sources():
    protocol = read_json(CODE / "protocol_frozen.json")
    expected = dict(protocol["source_files"])
    for path in sorted(CODE.glob("protocol_amendment_*.json")):
        amendment = read_json(path)
        for relative, record in amendment["source_updates"].items():
            if expected[relative] != record["previous_sha256"]:
                raise ValueError("Amendment source chain mismatch")
            expected[relative] = record["new_sha256"]
    for relative, digest in expected.items():
        if sha256(CODE / relative) != digest:
            raise ValueError("Frozen scientific source modified: " + relative)
    if read_json(RESULTS / "00_asset_manifest.json").get("status") != "PASS":
        raise ValueError("Asset qualification failed")
    if read_json(RESULTS / "01_math_qualification.json").get("status") != "PASS":
        raise ValueError("Mathematics qualification failed")
    return protocol


def train(workers):
    protocol = verify_sources()
    if not (PRIVATE / "RESOURCE_PILOT.json").exists():
        raise ValueError("Resource pilot is required")
    tasks = [(fold, seed) for seed in (None, *PERMUTATION_SEEDS) for fold in range(5)]
    start = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run_fold, fold, seed): (fold, seed) for fold, seed in tasks}
        for future in as_completed(pending):
            record = future.result()
            records.append(record)
            print(f"FROZEN {len(records)}/{len(tasks)} fold={record['fold']} seed={record['permutation_seed']} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    write_json(PRIVATE / "ALL_PREDICTIONS_FROZEN.json", dict(status="FROZEN", paths=records,
               protocol_sha256=sha256(CODE / "protocol_frozen.json"), wall_s=time.perf_counter()-start,
               new_llm_calls=0, retained_reward_payload_reads=0))
    print("ALL_PREDICTIONS_FROZEN", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "smoke", "train"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_protocol()
    elif args.command == "smoke":
        verify_sources()
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        result = run_fold(0, smoke=True)
        print(jsonable(dict(**result, cpu_s=time.process_time()-start_cpu, wall_s=time.perf_counter()-start_wall)), flush=True)
    else:
        train(args.workers)


if __name__ == "__main__":
    main()
