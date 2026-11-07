"""Frozen-alpha binary adaptation using only B's already-authorized feedback.

The fixed 100-question evaluator never participates in fitting, alpha selection,
or online updates. This runner makes no service/acquisition choices and performs
no LLM calls. Full refits are evaluator-side numerical comparisons only.
"""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from zipfile import BadZipFile

import numpy as np

from e4_common import PRIVATE, PUBLIC, SEEDS, TEAMS, e4_splits, load_train, put, sha
from e4_binary import BinaryModel, link
from e4_metrics import prediction_metrics
from run_binary import fit_model, pred_and_variance


MODES = ("text-only", "task-known")
FAMILIES = ("logistic", "probit")
CHECKPOINTS = (1, 5, 20)
METRICS = ("accuracy", "top1_regret", "raw_mse", "mean_mse", "brier", "log_loss",
           "pair_gap_mae", "non_tie_rank_accuracy", "empirical_mean_oracle_regret")


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _source_hashes():
    base = Path(__file__).parent
    names = ("run_binary_online.py", "run_binary.py", "e4_binary.py", "e4_gaussian.py",
             "e4_metadata.py", "e4_common.py", "e4_metrics.py")
    return {name: sha(base / name) for name in names}


def _brier_value(item):
    result = item["brier"]["balanced_v2"]
    return float(result["mean"] if isinstance(result, dict) else result)


def freeze_selection():
    """Freeze completed C choices, verifying their historical CV selection rule."""
    choices = []
    for fold in range(5):
        for predictor in ("P1", "P2"):
            for mode in MODES:
                path = PRIVATE / "binary" / f"f{fold}_{predictor}_{mode}.json"
                if not path.exists() or not path.with_suffix(".npz").exists():
                    raise ValueError("All primary C groups must complete before binary adaptation: " + str(path))
                case = json.loads(path.read_text(encoding="utf-8"))
                if (case["fold"], case["predictor"], case["mode"]) != (fold, predictor, mode):
                    raise ValueError("C case metadata differs from its filename")
                for family in FAMILIES:
                    matches = [item for item in case["models"] if item["family"] == family]
                    if len(matches) != 1:
                        raise ValueError("Missing or duplicate C family selection")
                    selected = matches[0]
                    winner = min(selected["search"], key=lambda item: (_brier_value(item), -float(item["alpha"])))
                    if float(selected["alpha"]) != float(winner["alpha"]):
                        raise ValueError("Frozen alpha is not the recorded history-CV winner")
                    choices.append({"fold": fold, "predictor": predictor, "mode": mode, "family": family,
                                    "alpha": float(selected["alpha"]), "selection_path": str(path),
                                    "selection_sha256": sha(path), "selection_metadata": selected["metadata"],
                                    "selection_criterion": "historical inner-OOF balanced_v2 Brier; largest alpha on ties"})
    selection = {"schema": "e4_binary_online_frozen_selection_v1", "choices": choices,
                 "source_hashes": _source_hashes(), "query_used_for_selection": False}
    selection["signature"] = _hash_json(selection)
    path = PRIVATE / "BINARY_ONLINE_SELECTION_FROZEN.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != selection:
            raise ValueError("Previously frozen binary-online selection or implementation changed")
    else:
        put(path, selection)
    return selection


def _validate_logger(receipt, fold, predictor, generation, orderseed, permission):
    expected = (fold["fold"], predictor, "N2", generation, int(SEEDS[generation]), orderseed, permission)
    actual = tuple(receipt[key] for key in ("fold", "predictor", "variant", "generation_index",
                                           "generation_seed", "orderseed", "permission"))
    if actual != expected or len(receipt["feedback"]) != 20:
        raise ValueError("B logger receipt does not match the required twenty-support case")
    order = list(map(int, fold["audit_orders"][orderseed]))
    if [int(event["q"]) for event in receipt["feedback"]] != order:
        raise ValueError("B logger feedback order differs from the frozen Support order")
    if set(order) & set(fold["history"]) or set(order) & set(fold["audit_query"]):
        raise ValueError("Support overlaps historical fitting or fixed Query")
    count = 1 if permission == "main" else 2
    for event in receipt["feedback"]:
        actions = list(map(int, event["actions"]))
        values = np.asarray(event["raw_scores"], dtype=float)
        if len(actions) != count or len(set(actions)) != count or values.shape != (count,):
            raise ValueError("Malformed original logger observation bundle")
        if any(action < 0 or action >= 20 for action in actions) or not np.isin(values, (0, 1)).all():
            raise ValueError("Logger must supply legal raw binary scores")
        if count == 2 and len(set(TEAMS[actions[0]]) ^ set(TEAMS[actions[1]])) != 2:
            raise ValueError("Pair permission must replace exactly one model")


def _refit_comparison(model, query, step):
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    before_mean, before_cov = model.mean.copy(), model.cov.copy()
    result = model.full_refit_online()
    if not result.success or not hasattr(result, "laplace_covariance"):
        raise RuntimeError("Exact composite MAP/Laplace comparison failed to converge")
    if not np.array_equal(before_mean, model.mean) or not np.array_equal(before_cov, model.cov):
        raise AssertionError("Evaluator full refit changed the sequential actor")
    coefficient = result.x.reshape(model.question_dimension, model.action_dimension)
    full_pred = link(model.X[query] @ coefficient @ model.B.T, model.family)
    sequential = model.predict_many(query)
    covariance = np.asarray(result.laplace_covariance)
    return {"step": step, "success": True, "iterations": int(result.nit),
            "gradient_infinity": float(np.max(np.abs(result.jac))),
            "sequential_batch_mean_l2": float(np.linalg.norm(model.mean - result.x)),
            "sequential_batch_mean_relative_l2": float(np.linalg.norm(model.mean - result.x) / max(np.linalg.norm(result.x), 1e-15)),
            "sequential_batch_covariance_relative_frobenius": float(np.linalg.norm(model.cov - covariance) / max(np.linalg.norm(covariance), 1e-15)),
            "query_probability_mean_absolute_difference": float(np.abs(sequential - full_pred).mean()),
            "query_probability_max_absolute_difference": float(np.abs(sequential - full_pred).max()),
            "query_service_disagreement_count": int(np.sum(np.argmax(sequential, axis=1) != np.argmax(full_pred, axis=1))),
            "sequential_trace": float(np.trace(model.cov)), "full_refit_trace": float(np.trace(covariance)),
            "cpu_s": time.process_time() - started_cpu, "wall_s": time.perf_counter() - started_wall,
            "scope": "Numerical sequential-Laplace approximation diagnostic; the full-refit result never updates the actor"}


def _run_path(base_model, data, query, receipt, generation, compare_refit=False):
    """Consume raw Support labels solely from receipt, never from data['Y'][Support]."""
    model = deepcopy(base_model)
    query = np.asarray(query, dtype=int)
    query_rewards = data["Y"][query]
    samegen_rewards = query_rewards[:, generation:generation + 1]
    snapshots, samegen, choices = [], [], []
    traces, mean_changes, update_times, comparisons = [], [], [], []
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    initial_pred, initial_latent_variance, initial_probability_variance = pred_and_variance(model, query)
    pred = initial_pred
    for step in range(21):
        all_metrics = prediction_metrics(pred, query_rewards)["per_question"]
        seed_metrics = prediction_metrics(pred, samegen_rewards)["per_question"]
        snapshots.append({key: all_metrics[key] for key in METRICS})
        samegen.append({key: seed_metrics[key] for key in METRICS})
        choices.append(np.argmax(pred, axis=1))
        traces.append(float(np.trace(model.cov)))
        if compare_refit and step in CHECKPOINTS:
            comparisons.append(_refit_comparison(model, query, step))
        if step == 20:
            break
        event = receipt["feedback"][step]
        before = model.mean.copy()
        clock = time.perf_counter()
        model.update(int(event["q"]), event["actions"], event["raw_scores"])
        update_times.append(time.perf_counter() - clock)
        mean_changes.append(float(np.linalg.norm(model.mean - before)))
        pred = model.predict_many(query)
    final_pred, final_latent_variance, final_probability_variance = pred_and_variance(model, query)
    if not np.allclose(pred, final_pred, atol=1e-14):
        raise AssertionError("Two frozen binary prediction entry points disagree")
    if np.max(np.diff(traces)) > 1e-9:
        raise AssertionError("Sequential Laplace covariance trace increased")
    arrays = {key: np.asarray([snapshot[key] for snapshot in snapshots]) for key in METRICS}
    arrays.update({"samegen_" + key: np.asarray([snapshot[key] for snapshot in samegen]) for key in METRICS})
    arrays.update({"query": query, "initial_pred": initial_pred, "final_pred": final_pred,
                   "initial_latent_variance": initial_latent_variance, "final_latent_variance": final_latent_variance,
                   "initial_probability_delta_variance": initial_probability_variance,
                   "final_probability_delta_variance": final_probability_variance,
                   "chosen": np.asarray(choices), "trace": np.asarray(traces),
                   "mean_change": np.asarray(mean_changes), "update_wall_s": np.asarray(update_times)})
    return arrays, {"full_refit_comparisons": comparisons,
                    "runtime": {"wall_s": time.perf_counter() - started_wall,
                                "process_cpu_s": time.process_time() - started_cpu,
                                "update_wall_s": float(sum(update_times)),
                                "full_refit_wall_s": float(sum(item["wall_s"] for item in comparisons))},
                    "legal_support_bundles_consumed": len(model.online_observations),
                    "query_feedback_used": False, "new_llm_calls": 0}


def worker(job):
    selection, frozen_signature, source_hashes = job
    fold_index, predictor, mode, family = (selection[key] for key in ("fold", "predictor", "mode", "family"))
    data, refs = load_train()
    fold = e4_splits(data)["folds"][fold_index]
    if sha(selection["selection_path"]) != selection["selection_sha256"]:
        raise ValueError("C model selection changed after freezing")
    base, metadata = fit_model(data, fold["history"], predictor, mode, family, selection["alpha"])
    completed = []
    for generation in range(3):
        for orderseed in sorted(fold["audit_orders"]):
            for permission in ("main", "pair"):
                source = PRIVATE / "posterior" / f"f{fold_index}_{predictor}_N2_g{generation}_o{orderseed}_{permission}.json"
                if not source.exists():
                    raise ValueError("Required completed B feedback receipt is absent: " + str(source))
                receipt = json.loads(source.read_text(encoding="utf-8"))
                _validate_logger(receipt, fold, predictor, generation, orderseed, permission)
                meta = {**selection, "generation_index": generation, "generation_seed": int(SEEDS[generation]),
                        "orderseed": orderseed, "permission": permission}
                identity = _hash_json({"selection": frozen_signature, "source_hashes": source_hashes,
                                       "logger_sha256": sha(source), "meta": meta,
                                       "inputs": {key: value.get("sha256") for key, value in refs.items() if isinstance(value, dict)}})
                path = PRIVATE / "binary_online" / f"f{fold_index}_{predictor}_{mode}_{family}_g{generation}_o{orderseed}_{permission}"
                json_path, npz_path = path.with_suffix(".json"), path.with_suffix(".npz")
                if json_path.exists() and npz_path.exists():
                    try:
                        existing = json.loads(json_path.read_text(encoding="utf-8"))
                        with np.load(npz_path, allow_pickle=False) as stored:
                            valid = existing.get("signature") == identity and str(stored["signature"]) == identity
                            valid &= stored["accuracy"].shape == (21, 100) and np.array_equal(stored["query"], fold["audit_query"])
                        if valid:
                            completed.append({"path": str(json_path), "reused": True})
                            continue
                    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                        pass
                arrays, details = _run_path(base, data, fold["audit_query"], receipt, generation,
                                            compare_refit=generation == 0 and orderseed == 17 and permission == "pair")
                path.parent.mkdir(parents=True, exist_ok=True)
                pending = npz_path.with_name(npz_path.name + ".pending")
                with pending.open("wb") as stream:
                    np.savez_compressed(stream, signature=np.asarray(identity), **arrays)
                with np.load(pending, allow_pickle=False) as check:
                    assert str(check["signature"]) == identity and check["accuracy"].shape == (21, 100)
                os.replace(pending, npz_path)
                payload = {"signature": identity, "meta": meta, "model_selection": selection,
                           "metadata_source": metadata, "fit_info": base.fit_info,
                           "logger_source": {"path": str(source), "sha256": sha(source), "variant": "N2 fixed B logging policy"},
                           "cost": receipt["cost"], "cost_scope": "Exact B receipt reference; adaptation/evaluator CPU reported separately",
                           "metric_scope": "Unsuffixed metrics average the three original Query generation repeats; samegen_ uses only the Support generation seed",
                           "roles": {"historical_questions": len(fold["history"]), "support_questions": 20, "query_questions": 100},
                           **details}
                put(json_path, payload)
                completed.append({"path": str(json_path), "reused": False})
    return {"fold": fold_index, "predictor": predictor, "mode": mode, "family": family, "cases": completed}


def qualification():
    rng = np.random.default_rng(20261107)
    size = 150
    x = np.column_stack((np.ones(size), rng.normal(size=(size, 3))))
    y = rng.integers(0, 2, (size, 3, 20)).astype(float)
    data = {"Y": y.copy()}
    history, support, query = np.arange(30), np.arange(30, 50), np.arange(50, 150)
    receipt = {"feedback": [{"q": int(q), "actions": [2, 5], "raw_scores": y[q, 0, [2, 5]].tolist()}
                            for q in support]}
    results = []
    for family in FAMILIES:
        base = BinaryModel(x, np.eye(20), history, y[history], 10., family)
        before_mean, before_cov = base.mean.copy(), base.cov.copy()
        data["Y"][support] = np.nan
        arrays, info = _run_path(base, data, query, receipt, 0, compare_refit=True)
        assert arrays["accuracy"].shape == arrays["samegen_accuracy"].shape == (21, 100)
        assert np.array_equal(base.mean, before_mean) and np.array_equal(base.cov, before_cov)
        assert info["legal_support_bundles_consumed"] == 20
        assert [item["step"] for item in info["full_refit_comparisons"]] == list(CHECKPOINTS)
        results.append({"family": family, "support_Y_nan_guard": True,
                        "base_model_unchanged": True, "checkpoints": info["full_refit_comparisons"]})
    return {"status": "PASS", "results": results, "new_llm_calls": 0, "E_files_written": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.qualify:
        print(json.dumps(qualification(), ensure_ascii=False, indent=2), flush=True)
        return
    if args.workers < 1:
        raise ValueError("workers must be positive")
    frozen = freeze_selection()
    if args.freeze_only:
        print(json.dumps({"status": "SELECTION_FROZEN", "models": len(frozen["choices"]), "signature": frozen["signature"]}), flush=True)
        return
    choices = frozen["choices"][:1] if args.smoke else frozen["choices"]
    jobs = [(choice, frozen["signature"], frozen["source_hashes"]) for choice in choices]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for result in pool.map(worker, jobs):
            results.append(result)
            put(PRIVATE / "BINARY_ONLINE_INDEX.json", results)
            print(json.dumps({key: value for key, value in result.items() if key != "cases"}), flush=True)
    put(PRIVATE / "BINARY_ONLINE_DONE.json", {"groups": len(results), "cases": sum(len(item["cases"]) for item in results),
                                             "smoke": args.smoke, "selection_signature": frozen["signature"], "new_llm_calls": 0})


if __name__ == "__main__":
    main()
