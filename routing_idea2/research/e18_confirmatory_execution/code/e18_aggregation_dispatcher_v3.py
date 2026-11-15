"""Dynamically claim disjoint frozen E18 profile shards on one assigned GPU."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import e18_aggregation_worker_v3 as worker


PLAN_SCHEMA = "e18_aggregation_dispatch_plan_v3"


class ShardClaim:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def acquire(self):
        self.stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.stream.seek(0)
                if not self.stream.read(1):
                    self.stream.write(b"0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            self.stream.close()
            self.stream = None
            return False
        return True

    def release(self):
        if self.stream is None:
            return
        if os.name == "nt":
            import msvcrt

            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        self.stream = None


def admit_plan(plan):
    if plan.get("schema") != PLAN_SCHEMA or not plan.get("manifest_refs"):
        raise ValueError("An explicit nonempty E18 shard plan is required")
    if plan.get("worker_source_sha256") != worker.file_sha(worker.__file__):
        raise ValueError("Admitted worker source identity differs")
    manifests = []
    keys_seen = set()
    hashes_seen = set()
    shared = None
    for reference in plan["manifest_refs"]:
        manifest = worker.read_json(worker.checked_ref(reference))
        if reference["sha256"] in hashes_seen:
            raise ValueError("Duplicate shard manifest in a dispatch plan")
        hashes_seen.add(reference["sha256"])
        identity = (
            manifest["scope_ref"]["sha256"],
            manifest["approval_ref"]["sha256"],
            str(Path(manifest["owned_output_root"]).resolve()),
            manifest["question_records_ref"]["sha256"],
            manifest["adapter_ref"]["sha256"],
            str(Path(manifest["gpu_lock_root"]).resolve()),
            manifest["software_test_only"],
        )
        if shared is None:
            shared = identity
        elif shared != identity:
            raise ValueError(
                "All shards must share the exact approved input/runtime scope"
            )
        keys = worker.read_json(worker.checked_ref(manifest["task_keys_ref"]))
        for key in keys:
            semantic_key = (manifest["stage"], *key)
            if semantic_key in keys_seen:
                raise ValueError("Overlapping semantic execution keys across shards")
            keys_seen.add(semantic_key)
        manifests.append((reference, manifest))
    return manifests


def run_plan(
    plan,
    plan_sha,
    *,
    software_test_only=False,
    execute=worker.execute_manifest,
    idle_seconds=15,
    max_idle_rounds=None,
):
    manifests = admit_plan(plan)
    output_root = Path(manifests[0][1]["owned_output_root"]).resolve()
    state_root = output_root / "_dispatcher" / "plans" / plan_sha
    state_root.mkdir(parents=True, exist_ok=True)
    stop_path = state_root / "STOP_REQUIRES_RECONCILIATION.json"
    completed = 0
    idle_rounds = 0
    while True:
        if stop_path.exists():
            raise RuntimeError("A shard has stopped this plan for reconciliation")
        pending = []
        progressed = False
        for reference, manifest in manifests:
            digest = reference["sha256"]
            status_path = state_root / (digest + ".status.json")
            if status_path.exists():
                status = worker.read_json(status_path)
                if status["status"] == "DONE":
                    continue
                if status["status"] == "FAILED":
                    raise RuntimeError("A failed shard needs explicit reconciliation")
            pending.append(reference)
            claim = ShardClaim(state_root / (digest + ".claim"))
            if not claim.acquire():
                continue
            try:
                # Recheck after ownership in case the preceding owner committed.
                if status_path.exists():
                    locked_status = worker.read_json(status_path)["status"]
                    if locked_status == "DONE":
                        progressed = True
                        continue
                    if locked_status == "FAILED" or stop_path.exists():
                        raise RuntimeError(
                            "A failed shard needs explicit reconciliation"
                        )
                worker.save_json(
                    status_path,
                    dict(
                        status="RUNNING",
                        manifest_sha256=digest,
                        pid=os.getpid(),
                        started_at_unix=time.time(),
                        software_test_only=software_test_only,
                    ),
                )
                try:
                    summary = execute(
                        manifest, digest, software_test_only=software_test_only
                    )
                except BaseException as error:
                    failed = dict(
                        status="FAILED",
                        manifest_sha256=digest,
                        error_type=type(error).__name__,
                        at_unix=time.time(),
                    )
                    worker.save_json(status_path, failed)
                    worker.save_json(stop_path, failed)
                    raise
                if summary["failures"]:
                    failed = dict(
                        status="FAILED", manifest_sha256=digest, summary=summary
                    )
                    worker.save_json(status_path, failed)
                    worker.save_json(stop_path, failed)
                    raise RuntimeError("A shard returned a failed physical attempt")
                done = summary["completed"] == summary["planned"]
                worker.save_json(
                    status_path,
                    dict(
                        status="DONE" if done else "WAITING_FOR_DEPENDENCIES",
                        manifest_sha256=digest,
                        summary=summary,
                        finished_at_unix=time.time(),
                    ),
                )
                progressed = progressed or done
                completed += int(done)
            finally:
                claim.release()
                gc.collect()
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if not pending:
            return dict(
                status="DONE",
                plan_sha256=plan_sha,
                total_shards=len(manifests),
                locally_completed_shards=completed,
                software_test_only=software_test_only,
            )
        if progressed:
            idle_rounds = 0
            continue
        idle_rounds += 1
        if max_idle_rounds is not None and idle_rounds >= max_idle_rounds:
            return dict(
                status="WAITING",
                plan_sha256=plan_sha,
                total_shards=len(manifests),
                locally_completed_shards=completed,
                software_test_only=software_test_only,
            )
        time.sleep(idle_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    if worker.file_sha(args.plan) != args.plan_sha256:
        raise ValueError("Dispatch plan bytes differ")
    if not args.gpu.isdigit():
        raise ValueError("Explicit physical GPU index required")
    plan = worker.read_json(args.plan)
    manifests = admit_plan(plan)
    if any(manifest["software_test_only"] is not False for _, manifest in manifests):
        raise PermissionError("Native dispatch requires production-only shards")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    with worker.GPUOwnership(
        manifests[0][1]["gpu_lock_root"], args.gpu, args.plan_sha256
    ):
        result = run_plan(plan, args.plan_sha256)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
