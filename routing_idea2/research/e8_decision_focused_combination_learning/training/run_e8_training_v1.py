"""Run E8 nested CPU training from history-only and input-only fold packs."""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


PRIMARY_METHODS = ["M0_original", "M0_tuned", "M1", "M2", "M3", "M4"]
VARIANTS = PRIMARY_METHODS + ["M3_top5", "M4_uniformtie"] + [
    "SHUFFLE_"+method for method in PRIMARY_METHODS]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+".pending")
    pending.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(pending, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+".pending")
    with pending.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(pending, path)


def verify_lock(lock_path):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    for path, expected in lock["files"].items():
        actual = sha(path)
        if actual != expected:
            raise RuntimeError("Frozen E8 source changed: "+path)
    return lock


def read_pack(item, evaluation_inputs=False):
    path = Path(item["path"])
    if sha(path) != item["sha256"]:
        raise RuntimeError("An E8 fold pack changed after qualification")
    with np.load(path, allow_pickle=False) as archive:
        if evaluation_inputs and set(archive.files) != {"E", "ids", "canonical", "global_indices"}:
            raise RuntimeError("Evaluation input pack contains unauthorized fields")
        return {key: archive[key] for key in archive.files}


def run_fold(job):
    root = Path(job["root"])
    sys.path.insert(0, str(root / "public/qualification"))
    sys.path.insert(0, str(root / "private/dependencies"))
    sys.path.insert(0, str(Path(job["training_module"]).parent))
    from e8_training_v1 import DEFAULT_SETTINGS, train_fold
    verify_lock(job["lock"])
    fold = job["fold"]
    spec = job["pack"]
    history = read_pack(spec["history"])
    inputs = read_pack(spec["evaluation_inputs"], evaluation_inputs=True)
    if set(history["ids"].tolist()) & set(inputs["ids"].tolist()):
        raise RuntimeError("Historical and evaluation question identifiers overlap")
    directory = root / "private/training" / ("fold"+str(fold))
    directory.mkdir(parents=True, exist_ok=True)
    completed_fold = directory / "FOLD_FROZEN_RECEIPT.json"
    if completed_fold.exists():
        receipt = json.loads(completed_fold.read_text(encoding="utf-8"))
        if receipt["protocol_sha256"] != job["protocol_sha256"] or receipt["source_lock_sha256"] != sha(job["lock"]):
            raise RuntimeError("The completed fold has a different protocol or source lock")
        if sha(receipt["predictions_path"]) != receipt["predictions_sha256"]:
            raise RuntimeError("Completed fold predictions changed")
        for variant, model_receipt in receipt["variants"].items():
            for filename, expected in model_receipt["artifact_hashes"].items():
                if sha(directory / variant / filename) != expected:
                    raise RuntimeError("A completed frozen model artifact changed")
        return receipt
    seed = job["base_seed"] + fold
    models, receipts = {}, {}
    predictions = dict(eval_indices=inputs["global_indices"], history_indices=history["global_indices"],
                       eval_ids=inputs["ids"], history_ids=history["ids"])
    started = datetime.now(timezone.utc).isoformat()
    for variant in VARIANTS:
        variant_directory = directory / variant
        receipt_path = variant_directory / "FROZEN_MODEL_RECEIPT.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["protocol_sha256"] != job["protocol_sha256"] or receipt["source_lock_sha256"] != sha(job["lock"]):
                raise RuntimeError("A cached model belongs to a different frozen protocol")
            for filename, expected in receipt["artifact_hashes"].items():
                if sha(variant_directory / filename) != expected:
                    raise RuntimeError("A frozen model artifact changed")
            selected = receipt["selected_parameters"]
            with np.load(variant_directory / "eval_predictions.npz", allow_pickle=False) as archive:
                scores, actions = archive["raw_q"], archive["actions"]
            with np.load(variant_directory / "train_outputs.npz", allow_pickle=False) as archive:
                inner_scores = archive["inner_oof_scores"]
            models[variant] = selected
        else:
            if variant == "M3_top5":
                method, parameters, top_k, tie_rule, permutation = "M3", models["M3"], 5, "lexicographic", False
            elif variant == "M4_uniformtie":
                method, parameters, top_k, tie_rule, permutation = "M4", models["M4"], 3, "uniform", False
            elif variant.startswith("SHUFFLE_"):
                method = variant.removeprefix("SHUFFLE_")
                parameters, top_k, tie_rule, permutation = models[method], 3, "lexicographic", True
            else:
                method, parameters, top_k, tie_rule, permutation = variant, None, 3, "lexicographic", False
            model, arrays, logs = train_fold(history, method, seed, settings=DEFAULT_SETTINGS,
                                             parameters=parameters, top_k=top_k,
                                             true_tie_rule=tie_rule, permutation=permutation)
            cpu_start, wall_start = time.process_time(), time.perf_counter()
            scores = model.predict(inputs["E"])
            actions = np.argmax(scores, axis=1)
            inference = dict(cpu_seconds=time.process_time()-cpu_start,
                             wall_seconds=time.perf_counter()-wall_start,
                             evaluation_inputs=len(inputs["E"]), evaluation_reward_access=False)
            selected, inner_scores = logs["selected_parameters"], arrays["inner_oof_scores"]
            save_npz(variant_directory / "model_state.npz", **model.arrays())
            save_npz(variant_directory / "train_outputs.npz", **arrays)
            save_npz(variant_directory / "eval_predictions.npz", raw_q=scores, actions=actions,
                     eval_indices=inputs["global_indices"], eval_ids=inputs["ids"])
            logs["inference"] = inference
            put(variant_directory / "FIT_LOG.json", logs)
            artifact_hashes = {name: sha(variant_directory / name) for name in
                               ("model_state.npz", "train_outputs.npz", "eval_predictions.npz", "FIT_LOG.json")}
            receipt = dict(schema="E8_frozen_model_v1", fold=fold, variant=variant, method=method,
                           selected_parameters=selected, raw_argmax_tie_rule="numpy_exact_lexicographic",
                           source_lock_sha256=sha(job["lock"]), protocol_sha256=job["protocol_sha256"],
                           history_pack_sha256=spec["history"]["sha256"],
                           evaluation_input_pack_sha256=spec["evaluation_inputs"]["sha256"],
                           evaluation_reward_access=False, artifact_hashes=artifact_hashes,
                           frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                           training_cpu_seconds=logs["total_cpu_seconds"], training_wall_seconds=logs["total_wall_seconds"],
                           inference=inference, final_diagnostics=logs["final_diagnostics"])
            verify_lock(job["lock"])
            put(receipt_path, receipt)
            models[variant] = selected
        predictions["raw_q__"+variant] = scores
        predictions["actions__"+variant] = actions
        predictions["inner_oof_q__"+variant] = inner_scores
        receipts[variant] = receipt
        print(json.dumps(dict(fold=fold, variant=variant, selected=selected,
                              status="FROZEN", frozen_at=receipt["frozen_at_utc"])), flush=True)
    save_npz(directory / "PREDICTIONS_FROZEN.npz", **predictions)
    fold_receipt = dict(fold=fold, variants=receipts,
                        predictions_path=str(directory / "PREDICTIONS_FROZEN.npz"),
                        predictions_sha256=sha(directory / "PREDICTIONS_FROZEN.npz"),
                        evaluation_reward_access=False, history_pack_path=spec["history"]["path"],
                        evaluation_input_pack_path=spec["evaluation_inputs"]["path"],
                        source_lock_sha256=sha(job["lock"]), protocol_sha256=job["protocol_sha256"],
                        started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat())
    put(directory / "FOLD_FROZEN_RECEIPT.json", fold_receipt)
    verify_lock(job["lock"])
    return fold_receipt


def main(args):
    root = Path(args.root)
    lock = verify_lock(args.lock)
    protocol_sha256 = sha(args.protocol)
    packs = json.loads(Path(args.packs_manifest).read_text(encoding="utf-8"))
    jobs = [dict(root=str(root), fold=fold["fold"], pack=fold,
                 training_module=str(Path(__file__).with_name("e8_training_v1.py")),
                 lock=str(Path(args.lock).resolve()), protocol_sha256=protocol_sha256,
                 base_seed=args.base_seed) for fold in packs["folds"]]
    records = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_fold, job): job["fold"] for job in jobs}
        for future in as_completed(futures):
            records.append(future.result())
            put(root / "private/training/E8_TRAINING_PROGRESS.json", dict(
                completed_folds=sorted(record["fold"] for record in records), total_folds=5,
                variants_per_fold=len(VARIANTS), evaluation_reward_access=False,
                elapsed_wall_seconds=time.perf_counter()-start))
    records.sort(key=lambda record: record["fold"])
    index = dict(schema="E8_all_outer_predictions_frozen_v1", complete=True,
                 variants=VARIANTS, folds=records, protocol_sha256=protocol_sha256,
                 source_lock_sha256=sha(args.lock), source_lock=lock,
                 evaluation_rewards_read=False, physical_llm_calls=0,
                 total_wall_seconds=time.perf_counter()-start,
                 all_frozen_at_utc=datetime.now(timezone.utc).isoformat())
    verify_lock(args.lock)
    put(root / "private/training/E8_FROZEN_TRAINING_INDEX.json", index)
    print(json.dumps(dict(status="ALL_PREDICTIONS_FROZEN", folds=len(records),
                          variants=len(VARIANTS), total_wall_seconds=index["total_wall_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--packs-manifest", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=20261113)
    main(parser.parse_args())
