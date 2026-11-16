"""Consume untouched original aggregation batches without owning the main queue.

This process keeps the original worker module and its backend context alive.
Only the existing dispatcher may claim shards, write native journals or execute
tasks. The follower's own metadata journal lives outside ``owned/_dispatcher``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import time

import e18_aggregation_dispatcher_v3 as dispatcher
import e18_aggregation_queue_daemon as queue
import e18_aggregation_worker_v3 as worker


EXPECTED_QUEUE_SHA = "4820354dc08fd5d9f83e114f469f1b9c9be5129a64fc10bb95c9b422b45c3f2f"


class AuxiliaryAggregationFollower:
    """Follow exactly one planner-admitted queue using original shard locks.

    Production has no injectable execution callback or backend. Software
    fixtures may inject a factory while retaining the original dispatcher and
    worker, so cross-batch context reuse is exercised by the real implementation.
    """

    def __init__(
        self,
        queue_root,
        gpu_lock_root,
        gpu,
        planner_config_ref,
        *,
        software_test_only=False,
        fixture_factory=None,
    ):
        if not software_test_only and fixture_factory is not None:
            raise PermissionError("Production rejects injected execution backends")
        if not str(gpu).isdigit():
            raise ValueError("An explicit physical GPU index is required")
        if (
            worker.file_sha(worker.__file__) != queue.EXPECTED_WORKER_SHA
            or worker.file_sha(dispatcher.__file__) != queue.EXPECTED_DISPATCHER_SHA
            or worker.file_sha(queue.__file__) != EXPECTED_QUEUE_SHA
            or dispatcher.worker is not worker
        ):
            raise ValueError("The original aggregation runtime sources differ")
        self.root = Path(queue_root).resolve(strict=True)
        self.lock_root = Path(gpu_lock_root).resolve()
        self.gpu = str(gpu)
        self.software_test_only = software_test_only
        self.fixture_factory = fixture_factory
        self.config_ref = dict(planner_config_ref)
        self.config = worker.read_json(worker.checked_ref(self.config_ref))
        if self.config.get("schema") != "e18_incremental_aggregation_planner_config_v1":
            raise ValueError("The original incremental planner configuration is required")
        self.server = Path(self.config["server_root"]).resolve()
        matches = [
            row
            for row in self.config["queues"]
            if Path(row["queue_root"]).resolve() == self.root
        ]
        if len(matches) != 1 or Path(matches[0]["gpu_lock_root"]).resolve() != self.lock_root:
            raise ValueError("Queue and original GPU-owner lock root do not match")
        if not self.root.is_relative_to(self.server):
            raise ValueError("The fixed queue escaped its original server root")
        self.scope = worker.read_json(worker.checked_ref(self.config["scope_ref"]))
        if (
            self.scope.get("schema") != "e18_scope_contract_v2"
            or self.scope.get("role") != "execution"
            or self.scope.get("cohort") != "e18_confirm"
        ):
            raise ValueError("The original execution scope is required")
        self.target_count = len(self.scope["required_aggregate_keys"])
        if not software_test_only and self.target_count != 30921:
            raise ValueError("The complete original30921-key target is required")
        self.planner_stop = Path(self.config["database_path"]).resolve().parent / "STOP_REQUIRES_RECONCILIATION.json"
        if not self.planner_stop.is_relative_to(self.server):
            raise ValueError("The planner stop location escaped its server root")
        self.identity = None
        self.accepted = {}
        self.key_owners = {}
        self.stop_requested = False
        self.completed_batches = 0
        self.attempted_batches = 0
        self.log_path = None

    def log(self, event, **fields):
        if self.log_path is None:
            directory = self.server / "auxiliary_aggregation_logs"
            directory.mkdir(parents=True, exist_ok=True)
            self.log_path = directory / (
                "aux_gpu" + self.gpu + "_pid" + str(os.getpid()) + "_" + str(time.time_ns()) + ".jsonl"
            )
        record = dict(event=event, at_unix=time.time(), pid=os.getpid(), gpu=self.gpu, **fields)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def admit(self, path):
        entry_ref, entry, plan = queue.read_stable_entry(path)
        previous = self.accepted.get(str(path))
        if previous is not None:
            if previous[0] != entry_ref:
                raise ValueError("A previously observed immutable ready entry changed")
            return previous
        manifests = dispatcher.admit_plan(plan)
        if len(manifests) != 1:
            raise ValueError("The original ready-batch publisher admits exactly one manifest per plan")
        for reference, manifest in manifests:
            if (
                manifest.get("schema") != worker.MANIFEST_SCHEMA
                or manifest.get("stage") != "aggregation"
                or manifest.get("software_test_only") is not self.software_test_only
                or manifest["scope_ref"] != self.config["scope_ref"]
                or Path(manifest["owned_output_root"]).resolve() != self.server / "owned"
                or Path(manifest["gpu_lock_root"]).resolve() != self.lock_root
            ):
                raise ValueError("The original aggregation manifest identity differs")
            identity = queue.shared_identity(manifest)
            if self.identity is None:
                self.identity = identity
            elif identity != self.identity:
                raise ValueError("A separate scope, backend or runtime requires a new process")
            keys = worker.read_json(worker.checked_ref(manifest["task_keys_ref"]))
            for key in keys:
                owner = self.key_owners.get(tuple(key))
                if owner is not None and owner != entry["plan_ref"]["sha256"]:
                    raise ValueError("Distinct ready plans contain a duplicate logical task")
                self.key_owners[tuple(key)] = entry["plan_ref"]["sha256"]
        result = (entry_ref, entry, plan, manifests)
        self.accepted[str(path)] = result
        return result

    def state(self, entry, manifests):
        """Inspect all original states, rejecting failures before selecting work."""
        state_root = self.server / "owned/_dispatcher/plans" / entry["plan_ref"]["sha256"]
        if (state_root / "STOP_REQUIRES_RECONCILIATION.json").exists():
            raise RuntimeError("An original plan requires reconciliation")
        states = []
        for reference, _ in manifests:
            path = state_root / (reference["sha256"] + ".status.json")
            if not path.exists():
                states.append("UNTOUCHED")
                continue
            value = worker.read_json(path)
            if value.get("manifest_sha256") != reference["sha256"]:
                raise ValueError("An original shard status has a different identity")
            summary = value.get("summary", {})
            if value.get("status") == "FAILED" or summary.get("failures"):
                raise RuntimeError("A failed original shard requires reconciliation")
            if summary.get("stop_requested"):
                raise RuntimeError("An original worker requested a safe stop")
            status = value.get("status")
            if status not in {"DONE", "RUNNING", "WAITING_FOR_DEPENDENCIES"}:
                raise ValueError("Unknown original shard state requires reconciliation")
            states.append(status)
        return state_root, states

    def candidate(self):
        if self.planner_stop.exists():
            raise RuntimeError("The original planner requires reconciliation")
        candidates = []
        for path in sorted(self.root.glob("*.ready.json"), reverse=True):
            admitted = self.admit(path)
            _, entry, _, manifests = admitted
            state_root, states = self.state(entry, manifests)
            if all(status == "UNTOUCHED" for status in states):
                candidates.append((admitted, state_root))
        # Check every published state before running any candidate. A failure in
        # an earlier batch must not be hidden by newer available work.
        for admitted, state_root in candidates:
            state_root.mkdir(parents=True, exist_ok=True)
            claims = []
            try:
                for reference, _ in admitted[3]:
                    claim = dispatcher.ShardClaim(state_root / (reference["sha256"] + ".claim"))
                    if not claim.acquire():
                        break
                    claims.append(claim)
                if len(claims) == len(admitted[3]):
                    return admitted
            finally:
                for claim in reversed(claims):
                    claim.release()
        return None

    def all_done(self):
        path = self.root / "ALL_DONE"
        if not path.exists():
            return False
        if not path.is_file() or path.is_symlink():
            raise ValueError("The original planner completion marker is not a regular file")
        value = worker.read_json(path)
        if (
            value.get("schema") != "e18_incremental_aggregation_all_done_v1"
            or value.get("status") != "ALL_FROZEN_NATIVE_COMMITS_SUCCESS"
            or value.get("config_sha256") != self.config_ref["sha256"]
            or value.get("execution_scope_sha256") != queue.digest(self.scope)
            or value.get("total_frozen_aggregate_keys") != self.target_count
            or value.get("startup_plan_sha256") != self.config["startup_plan_ref"]["sha256"]
            or value.get("gold_or_native_answer_payloads_read") != 0
            or len(value.get("native_success_metadata_digest", "")) != 64
        ):
            raise ValueError("Only the original complete-target planner marker is accepted")
        return True

    def execute(self, plan, plan_sha):
        if self.fixture_factory is None:
            return dispatcher.run_plan(
                plan, plan_sha, software_test_only=self.software_test_only, max_idle_rounds=1
            )

        def execute_fixture(manifest, digest, *, software_test_only):
            return worker.execute_manifest(
                manifest, digest, software_test_only=software_test_only,
                injected_factory=self.fixture_factory,
            )

        return dispatcher.run_plan(
            plan, plan_sha, software_test_only=True, execute=execute_fixture,
            idle_seconds=0, max_idle_rounds=1,
        )

    def run(self, *, sleep_seconds=15, max_idle_rounds=None, max_batches=None):
        if sleep_seconds < 0 or (max_idle_rounds is not None and max_idle_rounds < 1):
            raise ValueError("Invalid idle wait configuration")
        if max_batches is not None and max_batches < 1:
            raise ValueError("A batch limit must be positive")
        consumer = dispatcher.ShardClaim(self.root / ("AUX_CONSUMER_GPU" + self.gpu + ".lock"))
        if not consumer.acquire():
            return self.result("AUX_CONSUMER_BUSY")
        idle = 0
        try:
            with ExitStack() as resources:
                gpu_acquired = False
                self.log(
                    "AUX_START", follower_source_sha256=worker.file_sha(__file__),
                    planner_config_ref=self.config_ref, queue_root=str(self.root),
                    gpu_lock_root=str(self.lock_root), software_test_only=self.software_test_only,
                    original_worker_sha256=queue.EXPECTED_WORKER_SHA,
                    original_dispatcher_sha256=queue.EXPECTED_DISPATCHER_SHA,
                )
                while not self.stop_requested:
                    selected = self.candidate()
                    if self.stop_requested:
                        return self.result("STOP_REQUESTED")
                    if self.all_done():
                        return self.result("ALL_DONE")
                    if selected is None:
                        idle += 1
                        if max_idle_rounds is not None and idle >= max_idle_rounds:
                            return self.result("WAITING")
                        time.sleep(sleep_seconds)
                        continue
                    _, entry, plan, manifests = selected
                    plan_sha = entry["plan_ref"]["sha256"]
                    if not gpu_acquired:
                        resources.enter_context(worker.GPUOwnership(self.lock_root, self.gpu, plan_sha))
                        gpu_acquired = True
                    self.log("ORIGINAL_PLAN_SELECTED", plan_ref=entry["plan_ref"])
                    result = self.execute(plan, plan_sha)
                    self.attempted_batches += 1
                    self.state(entry, manifests)
                    if result.get("status") not in {"DONE", "WAITING"}:
                        raise RuntimeError("Unknown original dispatcher result")
                    self.completed_batches += int(result["status"] == "DONE")
                    self.log("ORIGINAL_PLAN_RETURNED", result=result)
                    idle = 0
                    if max_batches is not None and self.attempted_batches >= max_batches:
                        return self.result("BATCH_LIMIT_REACHED")
                return self.result("STOP_REQUESTED")
        except BaseException as error:
            self.log("STOP_REQUIRES_RECONCILIATION", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            consumer.release()

    def result(self, status):
        result = dict(
            schema="e18_auxiliary_aggregation_follower_result_v1", status=status,
            completed_batches=self.completed_batches, attempted_batches=self.attempted_batches,
            queue_root=str(self.root), gpu=self.gpu,
            original_planner_target=self.target_count, software_test_only=self.software_test_only,
        )
        if self.log_path is not None:
            result["metadata_log_path"] = str(self.log_path)
            self.log("AUX_RESULT", result=result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--gpu-lock-root", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--planner-config", type=Path, required=True)
    parser.add_argument("--planner-config-sha256", required=True)
    parser.add_argument("--sleep-seconds", type=float, default=15)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    if not args.gpu.isdigit():
        raise ValueError("An explicit physical GPU index is required")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    actor = AuxiliaryAggregationFollower(
        args.queue_root, args.gpu_lock_root, args.gpu,
        dict(path=str(args.planner_config), sha256=args.planner_config_sha256),
    )
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, lambda *_: setattr(actor, "stop_requested", True))
    try:
        result = actor.run(sleep_seconds=args.sleep_seconds, max_batches=args.max_batches)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
