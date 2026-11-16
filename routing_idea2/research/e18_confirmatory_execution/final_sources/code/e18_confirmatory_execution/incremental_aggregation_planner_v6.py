"""Explicit512/371-parent registry extension over frozen E18 planner v5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import incremental_aggregation_planner_v5 as previous


original = previous.original
V5_SHA = "5f5176787a8a5be8db1582256b5ff5e1e7dbd8d838b0d5beddda13ae125df6df"
PARENT_COUNTS = {
    "826e72af7b84e48d0421417eab3c1ed9c8907633fa88fdd756229c3bdc7fe323": 512,
    "5fc23696b12239804d4e22eb7fa4a7c15c0c3882e8b7ebc2ec2b0c56ee762995": 371,
}


class IncrementalPlanner(previous.IncrementalPlanner):
    def __init__(self, config_ref, *, tail371_entry_source_ref, **kwargs):
        if original.sha(previous.__file__) != V5_SHA:
            raise ValueError("Frozen inherited v5 source differs")
        self.tail371_entry_source_ref = dict(tail371_entry_source_ref)
        entry = original.checked_ref(self.tail371_entry_source_ref)
        if entry.name != "member4_tail371_broker.py":
            raise ValueError("Explicit371 broker entry source required")
        super().__init__(config_ref, **kwargs)
        try:
            value = dict(
                schema="e18_planner_explicit_two_parent_extension_v6",
                inherited_v5_source_sha256=V5_SHA,
                extension_source_sha256=original.sha(__file__),
                tail371_entry_source_ref=self.tail371_entry_source_ref,
                inherited_tail_extension_ref=self.tail_extension_ref,
                original_config_ref=dict(config_ref),
                allowed_original_parent_counts=dict(PARENT_COUNTS),
                additional_scientific_keys=0,
                original_config_modified=False, sqlite_schema_modified=False,
                all_done_config_sha256=self.config_sha,
            )
            path = self.database.parent / ("R1_EXPLICIT_TWO_PARENT_EXTENSION_" + original.digest(value) + ".json")
            original.publish_exact(path, original.encoded(value))
            self.tail_v6_extension_ref = dict(path=str(path), sha256=original.sha(path))
        except BaseException:
            self.connection.close()
            raise

    def parent_source(self, lease):
        reference = lease["parent_plan_ref"]
        if reference not in self.config["source_plan_refs"] or reference["sha256"] != previous.R1_PLAN_SHA:
            raise ValueError("Parent must belong to the exact original R1 source plan")
        plan = original.read_json(original.checked_ref(reference))
        worker_path = Path(self.template["adapter_ref"]["path"]).parent / "e18_member4_worker_v3.py"
        worker_sha = original.sha(worker_path)
        if (
            plan.get("schema") != "e18_member4_dispatch_plan_v3"
            or plan.get("worker_source_sha256") != worker_sha
            or worker_sha != previous.R1_WORKER_SHA
        ):
            raise ValueError("Frozen original R1 worker differs")
        parent_ref = lease["parent_manifest_ref"]
        expected_count = PARENT_COUNTS.get(parent_ref["sha256"])
        if expected_count is None or parent_ref not in plan.get("manifest_refs", []):
            raise ValueError("Only the explicit512/371 original parent identities are accepted")
        parent = original.read_json(original.checked_ref(parent_ref))
        item = self._validate_manifest(parent_ref, parent, "e18_member4_worker_manifest_v3")
        expected_claim = self.output / "_dispatcher/plans" / reference["sha256"] / (parent_ref["sha256"] + ".claim")
        if (
            item["stage"] != "member_generation" or parent.get("member_index") != 4
            or Path(lease["parent_claim_path"]).resolve() != expected_claim
            or len(item["keys"]) != expected_count
        ):
            raise ValueError("Explicit parent stage/profile/claim/key count differs")
        return item, worker_sha

    def once(self, *, after_reserve_hook=None):
        value = super().once(after_reserve_hook=after_reserve_hook)
        value["operational_explicit_two_parent_extension_ref"] = self.tail_v6_extension_ref
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--extra-source-plan", nargs=2, action="append", default=[])
    parser.add_argument("--member2-worker-source", nargs=2)
    parser.add_argument("--tail-source-plan", nargs=2, action="append", required=True)
    parser.add_argument("--tail-control-source", nargs=2, required=True)
    parser.add_argument("--tail371-entry-source", nargs=2, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--follow", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=15)
    args = parser.parse_args()
    def ref(pair):
        return dict(path=pair[0], sha256=pair[1])
    instance = IncrementalPlanner(
        ref([str(args.config), args.config_sha256]),
        extra_source_plan_refs=[ref(row) for row in args.extra_source_plan],
        member2_worker_source_ref=None if args.member2_worker_source is None else ref(args.member2_worker_source),
        tail_source_plan_refs=[ref(row) for row in args.tail_source_plan],
        tail_control_source_ref=ref(args.tail_control_source),
        tail371_entry_source_ref=ref(args.tail371_entry_source),
    )
    try:
        if args.once:
            print(json.dumps(instance.once(), ensure_ascii=False))
        else:
            instance.follow(args.interval_seconds)
    finally:
        instance.connection.close()


if __name__ == "__main__":
    main()
