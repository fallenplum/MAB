"""Register claim-fenced R1 tail subpartitions with the unchanged E18 ledger.

This layer reads execution metadata. It retains the original configuration,
SQLite schema, task union, reservations and ALL_DONE identity. Historical source
admission remains valid after a partition broker exits. Paid execution checks
both live leases in the separate original-worker control wrapper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import incremental_aggregation_planner_v4 as member2


original = member2.original
V4_SHA = "1d7679b6ca56d29ac2004d208eb148ca977ee809063ffea2e3ae7c58872b1023"
R1_PLAN_SHA = "a627f3094c742ec0d352582a9abe8b6338af6eaf0967333ebf30f45727c97166"
PARENT_SHA = "826e72af7b84e48d0421417eab3c1ed9c8907633fa88fdd756229c3bdc7fe323"
R1_WORKER_SHA = "5ac9a17dd07d282bfa2cceaff3dd062ce837e68885b4086816ef509974af5786"


class IncrementalPlanner(member2.IncrementalPlanner):
    def __init__(
        self, config_ref, *, tail_source_plan_refs=(), tail_control_source_ref=None,
        **kwargs,
    ):
        if original.sha(member2.__file__) != V4_SHA:
            raise ValueError("Inherited operational planner bytes differ")
        self.tail_source_plan_refs = [dict(row) for row in tail_source_plan_refs]
        self.tail_control_source_ref = tail_control_source_ref
        self.tail_extension_ref = None
        if not self.tail_source_plan_refs or tail_control_source_ref is None:
            raise ValueError("Explicit tail plan and controller source references are required")
        if len({row["sha256"] for row in self.tail_source_plan_refs}) != len(self.tail_source_plan_refs):
            raise ValueError("Duplicate tail source plans")
        super().__init__(config_ref, **kwargs)
        try:
            registry, audits = self.tail_registry()
            self.tail_manifest_count = len(registry)
            sidecar = dict(
                schema="e18_planner_r1_tail_source_extension_v1",
                original_config_ref=dict(config_ref),
                inherited_v4_source_sha256=V4_SHA,
                extension_source_sha256=original.sha(__file__),
                tail_control_source_ref=self.tail_control_source_ref,
                tail_source_plan_refs=self.tail_source_plan_refs,
                source_audits=audits,
                additional_scientific_keys=0,
                original_config_modified=False,
                sqlite_schema_modified=False,
                all_done_config_sha256=self.config_sha,
                historical_source_admission_requires_live_broker=False,
            )
            path = self.database.parent / ("R1_TAIL_SOURCE_EXTENSION_" + original.digest(sidecar) + ".json")
            original.publish_exact(path, original.encoded(sidecar))
            self.tail_extension_ref = dict(path=str(path), sha256=original.sha(path))
        except BaseException:
            self.connection.close()
            raise

    def parent_source(self, lease):
        reference = lease["parent_plan_ref"]
        if reference not in self.config["source_plan_refs"]:
            raise ValueError("Tail parent plan is outside the unchanged source registry")
        if not self.qualification_only and reference["sha256"] != R1_PLAN_SHA:
            raise ValueError("The frozen original R1 plan identity differs")
        plan = original.read_json(original.checked_ref(reference))
        worker_path = Path(self.template["adapter_ref"]["path"]).parent / "e18_member4_worker_v3.py"
        worker_sha = original.sha(worker_path)
        if (
            plan.get("schema") != "e18_member4_dispatch_plan_v3"
            or plan.get("worker_source_sha256") != worker_sha
            or (not self.qualification_only and worker_sha != R1_WORKER_SHA)
        ):
            raise ValueError("Tail source changes the frozen R1 worker")
        parent_ref = lease["parent_manifest_ref"]
        if parent_ref not in plan.get("manifest_refs", []):
            raise ValueError("Tail parent manifest is absent from its original plan")
        if not self.qualification_only and parent_ref["sha256"] != PARENT_SHA:
            raise ValueError("The approved512-key tail parent differs")
        parent = original.read_json(original.checked_ref(parent_ref))
        item = self._validate_manifest(parent_ref, parent, "e18_member4_worker_manifest_v3")
        expected_claim = self.output / "_dispatcher/plans" / reference["sha256"] / (parent_ref["sha256"] + ".claim")
        if (
            item["stage"] != "member_generation" or parent.get("member_index") != 4
            or Path(lease.get("parent_claim_path", "")).resolve() != expected_claim
            or (not self.qualification_only and len(item["keys"]) != 512)
        ):
            raise ValueError("Tail parent stage, profile, size or claim namespace differs")
        return item, worker_sha

    def tail_registry(self):
        control = original.checked_ref(self.tail_control_source_ref)
        expected = Path(self.template["adapter_ref"]["path"]).parent / "member4_tail_partition.py"
        if control != expected.resolve():
            raise ValueError("Tail controller must be the explicitly bound runtime source")
        result, audits, all_fine_keys = {}, [], set()
        for plan_ref in self.tail_source_plan_refs:
            plan = original.read_json(original.checked_ref(plan_ref))
            lease_ref = plan.get("tail_parent_lease_ref")
            lease = original.read_json(original.checked_ref(lease_ref))
            if (
                lease.get("schema") != "e18_member4_tail_partition_lease_v1"
                or lease.get("status") != "LEASED"
                or lease.get("software_test_only") is not False
                or lease.get("execution_scope_sha256") != self.scope_sha
                or lease.get("approval_sha256") != self.approval_sha
                or lease.get("pending_or_unknown_attempts") != 0
                or lease.get("controller_source_ref") != self.tail_control_source_ref
                or type(lease.get("owner_pid")) is not int or lease["owner_pid"] <= 0
                or type(lease.get("former_pid")) is not int or lease["former_pid"] <= 0
                or lease["owner_pid"] == lease["former_pid"]
                or not isinstance(lease.get("owner_start_ticks"), str)
                or not lease["owner_start_ticks"].isdigit()
                or not isinstance(lease.get("former_start_ticks"), str)
                or not lease["former_start_ticks"].isdigit()
            ):
                raise ValueError("Tail lease is not a source-bound settled checkpoint")
            parent, worker_sha = self.parent_source(lease)
            parent_manifest, parent_keys = parent["manifest"], parent["keys"]
            if (
                plan.get("schema") != "e18_member4_dispatch_plan_v3"
                or plan.get("worker_source_sha256") != worker_sha
                or plan.get("tail_parent_plan_ref") != lease["parent_plan_ref"]
                or plan.get("tail_parent_manifest_ref") != lease["parent_manifest_ref"]
                or plan.get("checkpoint_lease_ref") != parent_manifest.get("checkpoint_lease_ref")
                or lease.get("upstream_checkpoint_lease_ref") != parent_manifest.get("checkpoint_lease_ref")
                or not plan.get("manifest_refs")
            ):
                raise ValueError("Fine plan changes its exact original worker or lease chain")
            original.checked_ref(lease["upstream_checkpoint_lease_ref"])
            rows = original.read_json(original.checked_ref(lease["remaining_task_keys_ref"]))
            remaining = set(map(tuple, rows))
            maximum = lease.get("maximum_child_keys")
            if (
                type(maximum) is not int or not 1 <= maximum <= 64
                or not remaining or len(rows) != len(remaining)
                or not remaining <= parent_keys or all_fine_keys & remaining
                or lease.get("original_parent_tasks") != len(parent_keys)
                or lease.get("remaining_count") != len(remaining)
                or type(lease.get("checkpoint_successes")) is not int
                or lease["checkpoint_successes"] < 0
                or lease["checkpoint_successes"] + len(remaining) != len(parent_keys)
            ):
                raise ValueError("Fine remaining keys do not partition the original parent")
            audit = original.read_json(original.checked_ref(lease["checkpoint_source_audit_ref"]))
            entries = audit.get("tasks", [])
            expected_hashes = {original.digest(list(key)) for key in parent_keys}
            remaining_hashes = {original.digest(list(key)) for key in remaining}
            if (
                audit.get("original_tasks") != len(parent_keys)
                or audit.get("checkpoint_successes") != lease["checkpoint_successes"]
                or audit.get("pending_or_unknown_attempts") != 0
                or len(entries) != len(parent_keys)
                or {row.get("task_key_sha256") for row in entries} != expected_hashes
                or {row.get("task_key_sha256") for row in entries if row.get("status") == "UNEXECUTED"} != remaining_hashes
                or sum(row.get("status") == "SUCCESS" for row in entries) != lease["checkpoint_successes"]
                or any(row.get("status") not in {"SUCCESS", "UNEXECUTED"} for row in entries)
                or any(
                    not original.valid_sha(row.get("native_record_sha256"))
                    or not original.valid_sha(row.get("native_receipt_file_sha256"))
                    for row in entries if row.get("status") == "SUCCESS"
                )
            ):
                raise ValueError("The audit does not prove the exact native-success/unexecuted partition")
            seen = set()
            for index, reference in enumerate(plan["manifest_refs"]):
                manifest = original.read_json(original.checked_ref(reference))
                item = self._validate_manifest(reference, manifest, "e18_member4_worker_manifest_v3")
                additions = {"tail_parent_lease_ref", "tail_partition_index"}
                if (
                    set(manifest) != set(parent_manifest) | additions
                    or manifest.get("tail_parent_lease_ref") != lease_ref
                    or manifest.get("tail_partition_index") != index
                    or any(manifest[name] != value for name, value in parent_manifest.items() if name != "task_keys_ref")
                    or not 1 <= len(item["keys"]) <= maximum
                    or not item["keys"] <= remaining or seen & item["keys"]
                    or reference["sha256"] in result
                ):
                    raise ValueError("A fine manifest changes requests, duplicates keys or exceeds its chunk")
                seen.update(item["keys"])
                result[reference["sha256"]] = item
            if seen != remaining:
                raise ValueError("Fine manifests do not cover the exact unexecuted union")
            all_fine_keys.update(remaining)
            audits.append(dict(
                plan_ref=plan_ref, tail_parent_lease_ref=lease_ref,
                parent_plan_ref=lease["parent_plan_ref"], parent_manifest_ref=lease["parent_manifest_ref"],
                checkpoint_source_audit_ref=lease["checkpoint_source_audit_ref"],
                original_tasks=len(parent_keys), checkpoint_successes=lease["checkpoint_successes"],
                remaining_tasks=len(remaining), fine_manifests=len(plan["manifest_refs"]),
                additional_scientific_keys=0,
            ))
        return result, audits

    def registry(self):
        registry = super().registry()
        fine, _ = self.tail_registry()
        if set(registry) & set(fine):
            raise ValueError("A fine manifest collides with an existing source identity")
        registry.update(fine)
        return registry

    def once(self, *, after_reserve_hook=None):
        result = super().once(after_reserve_hook=after_reserve_hook)
        result["operational_r1_tail_source_extension_ref"] = self.tail_extension_ref
        result["operational_r1_tail_manifest_count"] = self.tail_manifest_count
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--extra-source-plan", nargs=2, action="append", default=[])
    parser.add_argument("--member2-worker-source", nargs=2)
    parser.add_argument("--tail-source-plan", nargs=2, action="append", required=True)
    parser.add_argument("--tail-control-source", nargs=2, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--follow", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=15)
    args = parser.parse_args()
    def reference(pair):
        return dict(path=pair[0], sha256=pair[1])
    instance = IncrementalPlanner(
        reference([str(args.config), args.config_sha256]),
        extra_source_plan_refs=[reference(row) for row in args.extra_source_plan],
        member2_worker_source_ref=None if args.member2_worker_source is None else reference(args.member2_worker_source),
        tail_source_plan_refs=[reference(row) for row in args.tail_source_plan],
        tail_control_source_ref=reference(args.tail_control_source),
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
