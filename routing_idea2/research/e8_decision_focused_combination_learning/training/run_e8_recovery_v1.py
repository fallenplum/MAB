"""Finish only unsealed E8 models with an audited equivalent QP backend."""
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
import threading
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    for source, expected in value["files"].items():
        if sha(source) != expected:
            raise RuntimeError("A locked source changed: "+source)
    return value


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(pending, path)


def recover_fold(job):
    root = Path(job["root"])
    sys.path.insert(0, str(root / "public/qualification"))
    sys.path.insert(0, str(root / "private/dependencies"))
    sys.path.insert(0, str(root / "public/training"))
    import e8_training_v1 as frozen
    import e8_recovery_v1 as recovery
    import run_e8_training_v1 as original_runner
    verify(job["lock"])
    verify(job["recovery_lock"])
    original_functions = {name: getattr(frozen, name) for name in
                          ("train_fold", "fit_coefficients", "fit_gap_ridge", "ranknet_value_gradient",
                           "listnet_value_gradient", "competition_weights", "legal_oof_m0")}
    provenance = recovery.install_solver_proxy()
    if any(getattr(frozen, name) is not value for name, value in original_functions.items()):
        raise RuntimeError("The recovery changed an undeclared training function")
    fold_directory = root / "private/training" / ("fold"+str(job["fold"]))
    journal = fold_directory / "RECOVERY_FIT_DIAGNOSTICS.jsonl"
    fitting_code = frozen.fit_coefficients.__code__
    fit_counter = 0
    def observe_fit(frame, event, returned):
        nonlocal fit_counter
        if event != "return" or frame.f_code is not fitting_code:
            return
        diagnostics = frame.f_locals.get("diagnostics")
        if diagnostics is None:
            return
        fit_counter += 1
        event = dict(fold=job["fold"], sequence=fit_counter,
                     recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                     method=frame.f_locals.get("method"), parameters=frame.f_locals.get("parameters"),
                     phi_sha256=frozen.array_hash(frame.f_locals["phi"]),
                     historical_reward_means_sha256=frozen.array_hash(frame.f_locals["rewards"]),
                     diagnostics=diagnostics,
                     returned_successfully=isinstance(returned, tuple),
                     engineering_amendment_sha256=job["amendment_sha256"],
                     evaluation_reward_access=False)
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(original_runner.plain(event), ensure_ascii=False)+"\n")
            stream.flush()
    old_writer = original_runner.put
    def amended_writer(path, value):
        path = Path(path)
        if path.name in ("FROZEN_MODEL_RECEIPT.json", "FOLD_FROZEN_RECEIPT.json"):
            if path.exists():
                raise RuntimeError("The recovery cannot overwrite an existing frozen receipt")
            value.update(engineering_amendment_sha256=job["amendment_sha256"],
                         recovery_source_lock_sha256=sha(job["recovery_lock"]),
                         recovery_binding=provenance)
        verify(job["recovery_lock"])
        old_writer(path, value)
    original_runner.put = amended_writer
    sys.setprofile(observe_fit)
    try:
        result = original_runner.run_fold(job)
    except BaseException:
        failure = dict(fold=job["fold"], failed_at_utc=datetime.now(timezone.utc).isoformat(),
                       traceback=traceback.format_exc(), fit_diagnostics_path=str(journal),
                       engineering_amendment_sha256=job["amendment_sha256"],
                       evaluation_reward_access=False)
        put(fold_directory / "RECOVERY_FAILURE.json", failure)
        raise
    finally:
        sys.setprofile(None)
    verify(job["lock"])
    verify(job["recovery_lock"])
    return result


def observe_process_cost(root, stopped):
    import psutil
    root = Path(root)
    maxima, peak_current_memory = {}, 0
    main_process = psutil.Process()
    started = time.process_time()
    path = root / "private/training/RECOVERY_PROCESS_COST_TIMELINE.jsonl"
    while not stopped.is_set():
        observed = {}
        for process in [main_process]+main_process.children(recursive=True):
            try:
                cpu, memory = process.cpu_times(), process.memory_info()
                observed[process.pid] = dict(pid=process.pid, cpu_seconds=cpu.user+cpu.system,
                                             rss_bytes=memory.rss,
                                             peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        for pid, row in observed.items():
            previous = maxima.get(pid, row)
            maxima[pid] = dict(pid=pid, cpu_seconds=max(row["cpu_seconds"], previous["cpu_seconds"]),
                               peak_working_set_bytes=max(row["peak_working_set_bytes"], previous["peak_working_set_bytes"]))
        peak_current_memory = max(peak_current_memory, sum(row["rss_bytes"] for row in observed.values()))
        value = dict(at_utc=datetime.now(timezone.utc).isoformat(), unique_pid_maxima=list(maxima.values()),
                     run_cpu_seconds_lower_bound=sum(row["cpu_seconds"] for row in maxima.values()),
                     peak_observed_current_total_rss_bytes=peak_current_memory,
                     interval_seconds=1.0, exact_cost_claim=False)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value)+"\n")
        stopped.wait(1.)
    put(root / "private/training/RECOVERY_PROCESS_COST_LOWER_BOUND.json", value)


def main(args):
    root = Path(args.root)
    if (root / "private/training/E8_FROZEN_TRAINING_INDEX.json").exists():
        raise RuntimeError("A completed frozen training index already exists")
    original_lock, recovery_lock = verify(args.lock), verify(args.recovery_lock)
    protocol_sha256, amendment_sha256 = sha(args.protocol), sha(args.amendment)
    packs = json.loads(Path(args.packs_manifest).read_text(encoding="utf-8"))
    jobs = [dict(root=str(root), fold=fold["fold"], pack=fold,
                 training_module=str(root / "public/training/e8_training_v1.py"),
                 lock=str(Path(args.lock).resolve()), recovery_lock=str(Path(args.recovery_lock).resolve()),
                 protocol_sha256=protocol_sha256, amendment_sha256=amendment_sha256,
                 base_seed=args.base_seed) for fold in packs["folds"]]
    existing_receipts = {str(path): sha(path) for path in
                         (root / "private/training").glob("fold*/*/FROZEN_MODEL_RECEIPT.json")}
    stopped = threading.Event()
    observer = threading.Thread(target=observe_process_cost, args=(str(root), stopped), daemon=True)
    observer.start()
    records, failures = [], []
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(recover_fold, job): job["fold"] for job in jobs}
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except BaseException:
                    failure = dict(fold=futures[future], traceback=traceback.format_exc())
                    failures.append(failure)
                    put(root / "private/training/RECOVERY_FUTURE_FAILURES.json", failures)
                    print(json.dumps(dict(status="RECOVERY_FAILED", **failure), ensure_ascii=False), flush=True)
                put(root / "private/training/E8_RECOVERY_PROGRESS.json", dict(
                    completed_folds=sorted(record["fold"] for record in records),
                    failed_folds=[failure["fold"] for failure in failures],
                    elapsed_wall_seconds=time.perf_counter()-started, evaluation_rewards_read=False))
    finally:
        stopped.set()
        observer.join()
    if failures:
        raise RuntimeError("The E8 recovery has failed folds; no complete index was generated")
    records.sort(key=lambda record: record["fold"])
    from run_e8_training_v1 import VARIANTS
    index = dict(schema="E8_all_outer_predictions_frozen_with_engineering_amendment_v1", complete=True,
                 variants=VARIANTS, folds=records, protocol_sha256=protocol_sha256,
                 source_lock_sha256=sha(args.lock), source_lock=original_lock,
                 execution_amendments=[dict(path=str(Path(args.amendment).resolve()), sha256=amendment_sha256)],
                 recovery_source_lock=recovery_lock, recovery_source_lock_sha256=sha(args.recovery_lock),
                 evaluation_rewards_read=False, physical_llm_calls=0,
                 total_wall_seconds=time.perf_counter()-started,
                 total_wall_seconds_scope="Recovery run only; original failed-run wall and CPU bounds are separately retained",
                 original_v1_failure_audit_path=str(root / "private/training/V1_PARTIAL_FAILURE_AUDIT.json"),
                 all_frozen_at_utc=datetime.now(timezone.utc).isoformat())
    verify(args.lock)
    verify(args.recovery_lock)
    for path, expected in existing_receipts.items():
        if sha(path) != expected:
            raise RuntimeError("The recovery changed a previously frozen receipt")
    index["preserved_existing_receipt_hashes"] = existing_receipts
    put(root / "private/training/E8_FROZEN_TRAINING_INDEX.json", index)
    print(json.dumps(dict(status="ALL_PREDICTIONS_FROZEN", folds=len(records), variants=len(VARIANTS),
                          recovery_wall_seconds=index["total_wall_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--amendment", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--recovery-lock", required=True)
    parser.add_argument("--packs-manifest", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=20261113)
    main(parser.parse_args())
