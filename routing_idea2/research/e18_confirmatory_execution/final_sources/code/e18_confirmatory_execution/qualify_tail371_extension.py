"""Pure metadata/control qualifications for the optional371-key entry."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import member4_tail371_broker as entry
import incremental_aggregation_planner_v6 as planner


class Qualification(unittest.TestCase):
    def inspect(self, parent_sha=entry.PARENT_SHA, count=371, change=None):
        root = Path.cwd().resolve()
        plan_ref = dict(path=str(root / "synthetic_plan.json"), sha256=entry.PARENT_PLAN_SHA)
        parent_ref = dict(path=str(root / "synthetic_parent.json"), sha256=parent_sha)
        plan = dict(schema="e18_member4_dispatch_plan_v3", worker_source_sha256=planner.previous.R1_WORKER_SHA, manifest_refs=[parent_ref])
        parent = dict(member_index=4)
        lease = dict(parent_plan_ref=plan_ref, parent_manifest_ref=parent_ref, parent_claim_path=str(root / "owned/_dispatcher/plans" / plan_ref["sha256"] / (parent_sha + ".claim")))
        item = dict(stage="member_generation", keys=set(range(count)))
        instance = SimpleNamespace(config=dict(source_plan_refs=[plan_ref]), template=dict(adapter_ref=dict(path=str(root / "code/adapter.py"))), output=root / "owned", qualification_only=True)
        instance._validate_manifest = lambda *_: item
        state = dict(instance=instance, plan=plan, parent=parent, lease=lease, item=item)
        if change:
            change(state)
        def read(path):
            return plan if Path(path) == Path(plan_ref["path"]) else parent
        with patch.object(planner.original, "checked_ref", side_effect=lambda ref: Path(ref["path"])), patch.object(planner.original, "read_json", side_effect=read), patch.object(planner.original, "sha", return_value=planner.previous.R1_WORKER_SHA):
            return planner.IncrementalPlanner.parent_source(instance, lease)

    def test_entry371_allowed(self):
        entry.validate_parent(entry.PARENT_PLAN_SHA, entry.PARENT_SHA, 371)

    def test_entry_other_parent_rejected(self):
        with self.assertRaises(PermissionError):
            entry.validate_parent(entry.PARENT_PLAN_SHA, planner.previous.PARENT_SHA, 512)

    def test_entry_wrong_count_rejected(self):
        with self.assertRaises(PermissionError):
            entry.validate_parent(entry.PARENT_PLAN_SHA, entry.PARENT_SHA, 370)

    def test_entry_wrong_plan_rejected(self):
        with self.assertRaises(PermissionError):
            entry.validate_parent("0" * 64, entry.PARENT_SHA, 371)

    def test_entry_uses_original_generic_class_with371(self):
        args = SimpleNamespace(parent_plan=Path("synthetic.json"), lease_root=Path("owned/x"), fine_root=Path("fine"), fine_plan=Path("fine.json"), former_pid=1172939, former_gpu="7", chunk_size=8)
        dispatcher = SimpleNamespace(admit_plan=lambda _: [(dict(sha256=entry.PARENT_SHA), dict(task_keys_ref={}))])
        with patch.object(entry.tail.base, "file_sha", side_effect=lambda p: entry.CONTROL_SHA if str(p) == str(entry.tail.__file__) else entry.PARENT_PLAN_SHA), patch.object(entry.tail.base, "read_json", return_value={}), patch.object(entry.tail, "_keys", return_value=[("synthetic", i, 4) for i in range(371)]), patch.object(entry.tail, "TailPartitionBroker") as constructor:
            entry.make_broker(args, dispatcher, None, None, "a" * 64)
            self.assertEqual(constructor.call_args.kwargs["expected_tasks"], 371)

    def test_planner371_allowed(self):
        item, _ = self.inspect()
        self.assertEqual(len(item["keys"]), 371)

    def test_planner512_preserved(self):
        item, _ = self.inspect(planner.previous.PARENT_SHA, 512)
        self.assertEqual(len(item["keys"]), 512)

    def test_planner_wrong_count_rejected(self):
        with self.assertRaises(ValueError):
            self.inspect(count=370)

    def test_planner_third_parent_rejected_even_with_qualification_true(self):
        with self.assertRaises(ValueError):
            self.inspect(parent_sha="f" * 64)

    def test_planner_parent_not_in_plan_rejected(self):
        with self.assertRaises(ValueError):
            self.inspect(change=lambda s: s["plan"].update(manifest_refs=[]))

    def test_planner_wrong_worker_rejected(self):
        with self.assertRaises(ValueError):
            self.inspect(change=lambda s: s["plan"].update(worker_source_sha256="f" * 64))

    def test_planner_wrong_plan_rejected(self):
        def change(s):
            ref = dict(s["lease"]["parent_plan_ref"], sha256="f" * 64)
            s["lease"]["parent_plan_ref"] = ref
            s["instance"].config["source_plan_refs"] = [ref]
        with self.assertRaises(ValueError):
            self.inspect(change=change)

    def test_planner_wrong_profile_rejected(self):
        with self.assertRaises(ValueError):
            self.inspect(change=lambda s: s["parent"].update(member_index=2))

    def test_planner_wrong_stage_rejected(self):
        with self.assertRaises(ValueError):
            self.inspect(change=lambda s: s["item"].update(stage="aggregation"))

    def test_planner_wrong_claim_rejected(self):
        with self.assertRaises(ValueError):
            self.inspect(change=lambda s: s["lease"].update(parent_claim_path="/wrong/claim"))

    def test_full_scope_and_keyproof_logic_inherited(self):
        self.assertIs(planner.IncrementalPlanner.tail_registry, planner.previous.IncrementalPlanner.tail_registry)
        self.assertIs(planner.IncrementalPlanner.registry, planner.previous.IncrementalPlanner.registry)

    def test_all_constant_hashes_valid(self):
        for value in [entry.PARENT_PLAN_SHA, entry.PARENT_SHA, entry.CONTROL_SHA, planner.V5_SHA, *planner.PARENT_COUNTS]:
            self.assertEqual(len(value), 64)
            int(value, 16)

    def test_frozen_dependencies_unchanged(self):
        self.assertEqual(entry.tail.base.file_sha(entry.tail.__file__), entry.CONTROL_SHA)
        self.assertEqual(planner.original.sha(planner.previous.__file__), planner.V5_SHA)

    def test_production_cli_has_no_qualification_flag(self):
        for path in (entry.__file__, planner.__file__):
            self.assertNotIn('add_argument("--qualification', Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification is append-only")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Qualification))
    report = dict(schema="e18_tail371_optional_extension_qualification_v1", status="PASS" if result.wasSuccessful() else "FAIL", tests_run=result.testsRun, errors=len(result.errors), failures=len(result.failures), broker_entry_sha256=entry.tail.base.file_sha(entry.__file__), planner_v6_sha256=planner.original.sha(planner.__file__), qualification_sha256=planner.original.sha(__file__), original_control_sha256=entry.CONTROL_SHA, inherited_v5_sha256=planner.V5_SHA, software_test_only=True, signals_sent=0, model_calls=0, gold_reads=0, limitations=["New production parent acceptance exercised with metadata mocks", "Inherited complete fine-key partition, native scope and ledger behavior remain original v5"])
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
