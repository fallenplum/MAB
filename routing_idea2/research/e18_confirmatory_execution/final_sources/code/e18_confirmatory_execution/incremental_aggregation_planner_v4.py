"""Register explicitly bound member2 checkpoint journals with the original ledger.

The original configuration, SQLite schema, reservations, ready plans and
ALL_DONE identity are inherited unchanged. Additional operational sources are
recorded in a separate immutable sidecar and per-round metadata heartbeat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import incremental_aggregation_planner as original


ORIGINAL_PLANNER_SHA = "56c8956326427185dd5132b368fb787dba67e6ad4ef0cfb3d602943df0b187e6"
BASE_MEMBER_PLAN_SHA = "8473a296a638c677da1791867c87c09af441cfb5e90ae5cd9a84feb7e9564502"
MEMBER2_PLAN_SCHEMA = "e18_member2_dispatch_plan_v1"
MEMBER2_MANIFEST_SCHEMA = "e18_member2_worker_manifest_v1"
MEMBER2_WORKER_NAME = "e18_member2_worker_v1.py"


class IncrementalPlanner(original.IncrementalPlanner):
    def __init__(
        self, config_ref, *, extra_source_plan_refs=(), member2_worker_source_ref=None,
        qualification_only=False,
    ):
        if original.sha(original.__file__) != ORIGINAL_PLANNER_SHA:
            raise ValueError("The inherited original planner source differs")
        self.extra_source_plan_refs = [dict(reference) for reference in extra_source_plan_refs]
        self.member2_worker_source_ref = None if member2_worker_source_ref is None else dict(member2_worker_source_ref)
        self.qualification_only = qualification_only
        self.extension_ref = None
        if bool(self.extra_source_plan_refs) != (member2_worker_source_ref is not None):
            raise ValueError("Extra plans and an explicit bound member2 worker source are required together")
        if len({row["sha256"] for row in self.extra_source_plan_refs}) != len(self.extra_source_plan_refs):
            raise ValueError("Duplicate operational extra-source plans")
        super().__init__(config_ref, qualification_only=qualification_only)
        try:
            extra, audit = self.extra_registry()
            self._extra_manifest_count = len(extra)
            sidecar = dict(
                schema="e18_planner_operational_source_extension_v1",
                original_config_ref=dict(config_ref), original_config_sha256=self.config_sha,
                original_planner_source_sha256=ORIGINAL_PLANNER_SHA,
                extension_source_sha256=original.sha(__file__),
                extra_source_plan_refs=self.extra_source_plan_refs,
                member2_worker_source_ref=self.member2_worker_source_ref,
                source_audits=audit, additional_scientific_keys=0,
                original_config_modified=False, sqlite_schema_modified=False,
                all_done_config_sha256=self.config_sha,
            )
            path = self.database.parent / ("OPERATIONAL_SOURCE_EXTENSION_" + original.digest(sidecar) + ".json")
            original.publish_exact(path, original.encoded(sidecar))
            self.extension_ref = dict(path=str(path), sha256=original.sha(path))
        except BaseException:
            self.connection.close()
            raise

    def original_member2_parents(self, lease):
        base_sha = lease.get("base_member_plan_sha256")
        if not self.qualification_only and base_sha != BASE_MEMBER_PLAN_SHA:
            raise ValueError("The original member dispatch plan identity differs")
        matches = [row for row in self.config["source_plan_refs"] if row["sha256"] == base_sha]
        if len(matches) != 1:
            raise ValueError("The checkpoint base plan is absent or ambiguous in the original config")
        plan = original.read_json(original.checked_ref(matches[0]))
        source = Path(self.template["adapter_ref"]["path"]).parent / "e18_gpu_worker.py"
        if plan.get("schema") != "e18_gpu_dispatch_plan_v1" or plan.get("worker_source_sha256") != original.sha(source):
            raise ValueError("The checkpoint base plan changes the original worker")
        parents = []
        keys = set()
        for reference in plan.get("manifest_refs", []):
            manifest = original.read_json(original.checked_ref(reference))
            if manifest.get("member_index") != 2:
                continue
            item = super()._validate_manifest(reference, manifest, "e18_gpu_worker_manifest_v1")
            if item["stage"] != "member_generation" or keys & item["keys"]:
                raise ValueError("The original member2 parents overlap or change stage")
            parents.append(item)
            keys.update(item["keys"])
        if len(parents) != 2:
            raise ValueError("Exactly two original member2 parent manifests are required")
        callable_rows = self.scope.get("member_execution_keys", [])
        callable_keys = set(map(tuple, callable_rows))
        if len(callable_rows) != len(callable_keys) or keys != {key for key in callable_keys if key[2] == 2}:
            raise ValueError("The original member2 parents do not equal their frozen callable union")
        if not self.qualification_only and (
            len(keys) != 8005 or len(callable_keys) != 42569 or len(self.aggregate_keys) != 30921
        ):
            raise ValueError("The frozen8005/42569/30921 execution cardinalities differ")
        return parents, keys

    def extra_registry(self):
        if not self.extra_source_plan_refs:
            return {}, []
        worker_path = original.checked_ref(self.member2_worker_source_ref)
        expected_path = Path(self.template["adapter_ref"]["path"]).parent / MEMBER2_WORKER_NAME
        if worker_path != expected_path.resolve() or worker_path.name != MEMBER2_WORKER_NAME:
            raise ValueError("The member2 source must be the explicit runtime worker file")
        result, audits = {}, []
        all_extra_keys = set()
        for plan_ref in self.extra_source_plan_refs:
            plan = original.read_json(original.checked_ref(plan_ref))
            if (
                plan.get("schema") != MEMBER2_PLAN_SCHEMA
                or plan.get("worker_source_sha256") != self.member2_worker_source_ref["sha256"]
                or not plan.get("manifest_refs")
            ):
                raise ValueError("The operational member2 plan schema or source differs")
            lease_ref = plan.get("checkpoint_lease_ref")
            lease_path = original.checked_ref(lease_ref)
            lease = original.read_json(lease_path)
            former = lease.get("former_pids", [])
            if (
                lease.get("schema") != "e18_member2_checkpoint_lease_v1"
                or lease.get("status") != "LEASED"
                or lease.get("target_member_index") != 2
                or lease.get("execution_scope_sha256") != self.scope_sha
                or lease.get("approval_sha256") != self.approval_sha
                or lease.get("checkpoint_stop_verified") is not True
                or lease.get("pending_or_unknown_attempts") != 0
                or lease.get("software_test_only") is not False
                or len(former) != 2 or len(set(former)) != 2
                or any(type(pid) is not int or pid <= 0 for pid in former)
                or type(lease.get("owner_pid")) is not int or lease["owner_pid"] <= 0
                or lease["owner_pid"] in former
            ):
                raise ValueError("The operational member2 lease is not a settled two-parent checkpoint")
            parents, parent_keys = self.original_member2_parents(lease)
            parent_by_sha = {item["reference"]["sha256"]: item for item in parents}
            declared = lease.get("parent_shards", [])
            if len(declared) != 2 or {row.get("manifest_sha256") for row in declared} != set(parent_by_sha):
                raise ValueError("The lease does not declare the exact original parent pair")
            for row in declared:
                parent = parent_by_sha[row["manifest_sha256"]]
                expected_claim = self.output / "_dispatcher/plans" / lease["base_member_plan_sha256"] / (row["manifest_sha256"] + ".claim")
                if row.get("task_keys_sha256") != parent["manifest"]["task_keys_ref"]["sha256"] or Path(row.get("claim_path", "")).resolve() != expected_claim:
                    raise ValueError("A declared parent key source or claim namespace differs")
            remaining_rows = original.read_json(original.checked_ref(lease["remaining_task_keys_ref"]))
            remaining = set(map(tuple, remaining_rows))
            if (
                len(remaining) != len(remaining_rows) or not remaining <= parent_keys
                or lease.get("remaining_count") != len(remaining)
                or lease.get("original_member2_tasks") != len(parent_keys)
                or type(lease.get("checkpoint_successes")) is not int
                or lease["checkpoint_successes"] < 0
                or lease["checkpoint_successes"] + len(remaining) != len(parent_keys)
                or all_extra_keys & remaining
            ):
                raise ValueError("The operational remaining union is not exact or is duplicated")
            if not original.valid_sha(lease.get("checkpoint_source_audit_sha256")):
                raise ValueError("The checkpoint source audit SHA is required")
            audit_path = lease_path.parent / "CHECKPOINT_SOURCE_AUDIT.json"
            original.checked_ref(dict(path=str(audit_path), sha256=lease["checkpoint_source_audit_sha256"]))
            audit = original.read_json(audit_path)
            entries = audit.get("tasks", [])
            parent_hashes = {original.digest(list(key)) for key in parent_keys}
            remaining_hashes = {original.digest(list(key)) for key in remaining}
            if (
                audit.get("schema") != "e18_member2_native_checkpoint_audit_v1"
                or audit.get("original_tasks") != len(parent_keys)
                or audit.get("checkpoint_successes") != lease["checkpoint_successes"]
                or audit.get("pending_or_unknown_attempts") != 0
                or len(entries) != len(parent_keys)
                or {row.get("task_key_sha256") for row in entries} != parent_hashes
                or {row.get("task_key_sha256") for row in entries if row.get("status") == "UNEXECUTED"} != remaining_hashes
                or any(row.get("status") not in {"SUCCESS", "UNEXECUTED"} for row in entries)
                or any(
                    not original.valid_sha(row.get("native_record_sha256"))
                    or not original.valid_sha(row.get("native_receipt_file_sha256"))
                    for row in entries if row.get("status") == "SUCCESS"
                )
            ):
                raise ValueError("The bound checkpoint audit does not prove the exact settled/remaining partition")
            seen = set()
            for reference in plan["manifest_refs"]:
                manifest = original.read_json(original.checked_ref(reference))
                if (
                    manifest.get("schema") != MEMBER2_MANIFEST_SCHEMA
                    or manifest.get("stage") != "member_generation"
                    or manifest.get("member_index") != 2
                    or manifest.get("checkpoint_lease_ref") != lease_ref
                ):
                    raise ValueError("A child manifest changes its checkpoint/profile/schema")
                for name in ("scope_ref", "approval_ref", "question_records_ref", "adapter_ref", "asset_root", "owned_output_root", "software_test_only", "backend"):
                    if any(manifest.get(name) != parent["manifest"].get(name) for parent in parents):
                        raise ValueError("A child manifest changes an original source, profile or owner")
                rows = original.read_json(original.checked_ref(manifest["task_keys_ref"]))
                keys = set(map(tuple, rows))
                if not 1 <= len(rows) <= 512 or len(keys) != len(rows) or not keys <= remaining or keys & seen:
                    raise ValueError("A child chunk exceeds512, duplicates or escapes the remaining union")
                if reference["sha256"] in result:
                    raise ValueError("Duplicate or ambiguous operational manifest identity")
                seen.update(keys)
                result[reference["sha256"]] = dict(reference=reference, manifest=manifest, stage="member_generation", keys=keys)
            if seen != remaining:
                raise ValueError("The child plan is not the exact checkpoint remaining union")
            all_extra_keys.update(remaining)
            audits.append(dict(
                plan_ref=plan_ref, checkpoint_lease_ref=lease_ref,
                checkpoint_source_audit_ref=dict(path=str(audit_path), sha256=lease["checkpoint_source_audit_sha256"]),
                parent_manifest_sha256s=sorted(parent_by_sha), original_tasks=len(parent_keys),
                checkpoint_successes=lease["checkpoint_successes"], remaining_tasks=len(remaining),
                child_manifests=len(plan["manifest_refs"]), additional_scientific_keys=0,
            ))
        return result, audits

    def registry(self):
        result = super().registry()
        extra, _ = self.extra_registry()
        if set(result) & set(extra):
            raise ValueError("An extra manifest collides with an original registered source")
        result.update(extra)
        return result

    def once(self, *, after_reserve_hook=None):
        result = super().once(after_reserve_hook=after_reserve_hook)
        result["operational_source_extension_ref"] = self.extension_ref
        result["operational_extra_manifest_count"] = self._extra_manifest_count
        heartbeat = dict(
            schema="e18_planner_operational_heartbeat_v1", original_config_sha256=self.config_sha,
            operational_source_extension_ref=self.extension_ref, status=result["status"],
            extra_manifest_count=self._extra_manifest_count, at_unix=time.time(),
            physical_model_calls=0, gold_values_read=0, native_answer_payloads_read=0,
        )
        original.publish_exact(self.database.parent / ("OPERATIONAL_HEARTBEAT_" + str(time.time_ns()) + ".json"), original.encoded(heartbeat))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--extra-source-plan", type=Path, action="append", default=[])
    parser.add_argument("--extra-source-plan-sha256", action="append", default=[])
    parser.add_argument("--member2-worker-source", type=Path)
    parser.add_argument("--member2-worker-source-sha256")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--follow", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=60)
    args = parser.parse_args()
    if len(args.extra_source_plan) != len(args.extra_source_plan_sha256):
        raise ValueError("Every operational extra plan requires its explicit SHA")
    if (args.member2_worker_source is None) != (args.member2_worker_source_sha256 is None):
        raise ValueError("An explicit worker source path and SHA must be supplied together")
    worker_ref = None if args.member2_worker_source is None else dict(path=str(args.member2_worker_source), sha256=args.member2_worker_source_sha256)
    instance = IncrementalPlanner(
        dict(path=str(args.config), sha256=args.config_sha256),
        extra_source_plan_refs=[dict(path=str(path), sha256=value) for path, value in zip(args.extra_source_plan, args.extra_source_plan_sha256)],
        member2_worker_source_ref=worker_ref,
    )
    try:
        if args.once:
            print(json.dumps(instance.once(), ensure_ascii=False, indent=2))
        else:
            instance.follow(args.interval_seconds)
    finally:
        instance.connection.close()


if __name__ == "__main__":
    main()
