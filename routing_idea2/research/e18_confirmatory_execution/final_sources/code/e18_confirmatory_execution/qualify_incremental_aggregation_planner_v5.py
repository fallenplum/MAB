"""Synthetic metadata qualification of the R1 tail registry extension."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import incremental_aggregation_planner_v5 as extension
from qualify_incremental_aggregation_planner_v4 import ExtendedFixture


original = extension.original


class TailFixture(ExtendedFixture):
    def __init__(self, root):
        super().__init__(root)
        self.parent_keys = [[qid, 42, 4] for qid in self.qids]
        self.tail_remaining = self.parent_keys[1:]
        upstream_ref = self.artifact(self.root / "original_r1_lease.json", dict(schema="synthetic_original_lease"))
        parent = original.read_json(self.member4_ref["path"])
        parent["task_keys_ref"] = self.artifact(self.inputs / "r1_parent_keys.json", self.parent_keys)
        parent["checkpoint_lease_ref"] = upstream_ref
        self.member4_ref = self.artifact(self.root / "r1_parent_manifest.json", parent)
        self.member4_plan_ref = self.artifact(self.root / "r1_parent_plan.json", dict(
            schema="e18_member4_dispatch_plan_v3",
            worker_source_sha256=original.sha(self.code / "e18_member4_worker_v3.py"),
            manifest_refs=[self.member4_ref],
        ))
        self.config["source_plan_refs"] = [self.member_plan_ref, self.member4_plan_ref]
        self.config_ref = self.artifact(self.root / "config.json", self.config)
        control = self.code / "member4_tail_partition.py"
        control.write_text("# synthetic control-source identity only\n", encoding="utf-8")
        self.control_ref = dict(path=str(control), sha256=original.sha(control))
        self.tail_audit = dict(
            schema="synthetic_bound_native_checkpoint",
            original_tasks=3, checkpoint_successes=1, pending_or_unknown_attempts=0,
            tasks=[dict(
                status="SUCCESS", task_key_sha256=original.digest(self.parent_keys[0]),
                native_record_sha256="a" * 64, native_receipt_file_sha256="b" * 64,
            )] + [dict(status="UNEXECUTED", task_key_sha256=original.digest(key)) for key in self.tail_remaining],
        )
        self.tail_lease = dict(
            schema="e18_member4_tail_partition_lease_v1", status="LEASED",
            software_test_only=False, owner_pid=999999, owner_start_ticks="1234",
            former_pid=999998, former_start_ticks="1233", controller_source_ref=self.control_ref,
            execution_scope_sha256=self.scope_sha, approval_sha256=self.approval_ref["sha256"],
            pending_or_unknown_attempts=0, parent_plan_ref=self.member4_plan_ref,
            parent_manifest_ref=self.member4_ref,
            parent_claim_path=str(self.output / "_dispatcher/plans" / self.member4_plan_ref["sha256"] / (self.member4_ref["sha256"] + ".claim")),
            upstream_checkpoint_lease_ref=upstream_ref, original_parent_tasks=3,
            checkpoint_successes=1, remaining_count=2, maximum_child_keys=1,
        )
        self.rebuild_tail()

    def rebuild_tail(self, *, duplicate=False, drop=False, changed_backend=False):
        self.tail_lease["remaining_task_keys_ref"] = self.artifact(self.inputs / "tail_remaining.json", self.tail_remaining)
        self.tail_lease["checkpoint_source_audit_ref"] = self.artifact(self.root / "tail_audit.json", self.tail_audit)
        self.tail_lease_ref = self.artifact(self.root / "tail_lease.json", self.tail_lease)
        parent = original.read_json(self.member4_ref["path"])
        references = []
        for index, key in enumerate(self.tail_remaining):
            value = self.tail_remaining[0] if duplicate else key
            manifest = copy.deepcopy(parent)
            manifest.update(
                task_keys_ref=self.artifact(self.inputs / ("fine_keys_" + str(index) + ".json"), [value]),
                tail_parent_lease_ref=self.tail_lease_ref, tail_partition_index=index,
            )
            if changed_backend:
                manifest["backend"] = dict(mode="forbidden_changed_model")
            references.append(self.artifact(self.root / ("fine_" + str(index) + ".json"), manifest))
        if drop:
            references = references[:-1]
        self.fine_refs = references
        self.fine_plan_ref = self.artifact(self.root / "fine_plan.json", dict(
            schema="e18_member4_dispatch_plan_v3",
            worker_source_sha256=original.sha(self.code / "e18_member4_worker_v3.py"),
            checkpoint_lease_ref=parent["checkpoint_lease_ref"],
            tail_parent_plan_ref=self.member4_plan_ref,
            tail_parent_manifest_ref=self.member4_ref,
            tail_parent_lease_ref=self.tail_lease_ref, manifest_refs=references,
        ))

    def open_tail(self):
        instance = extension.IncrementalPlanner(
            self.config_ref, qualification_only=True,
            extra_source_plan_refs=[self.extra_ref], member2_worker_source_ref=self.worker_ref,
            tail_source_plan_refs=[self.fine_plan_ref], tail_control_source_ref=self.control_ref,
        )
        self.connections.append(instance.connection)
        return instance


class TailRegistryQualification(unittest.TestCase):
    temporary_root = Path(__file__).resolve().parents[2] / ".agg_v5_toy"

    def setUp(self):
        parent = self.temporary_root
        parent.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="v5_", dir=parent)
        self.addCleanup(temporary.cleanup)
        self.f = TailFixture(temporary.name)
        self.addCleanup(self.f.close)

    def test_reopen_native_progress_and_original_all_done_identity(self):
        f = self.f
        initial = f.extended()
        initial.once()
        config_bytes = Path(f.config_ref["path"]).read_bytes()
        initial.connection.close()
        p = f.open_tail()
        f.append_event(f.fine_refs[1], (f.qids[2], 42, 4), "member_generation")
        self.assertEqual(p.once()["newly_reserved_keys"], 1)
        self.assertEqual(Path(f.config_ref["path"]).read_bytes(), config_bytes)
        f.append_event(f.child_refs[1], (f.qids[2], 42, 2), "member_generation")
        p.once()
        f.complete_all(p)
        self.assertEqual(p.once()["status"], "ALL_DONE")
        for name in original.QUEUES:
            marker = original.read_json(f.root / "aggregation_queues" / name / "ALL_DONE")
            self.assertEqual(marker["config_sha256"], f.config_ref["sha256"])
        self.assertTrue(original.checked_ref(p.tail_extension_ref).is_file())

    def test_production_source_bindings_are_valid_sha256(self):
        for value in (extension.V4_SHA, extension.R1_PLAN_SHA, extension.PARENT_SHA, extension.R1_WORKER_SHA):
            self.assertTrue(original.valid_sha(value))

    def test_repeat_deduplicates_native_events(self):
        f = self.f
        p = f.open_tail()
        f.append_event(f.fine_refs[1], (f.qids[2], 42, 4), "member_generation")
        p.once()
        count = p.connection.execute("SELECT COUNT(*) FROM members").fetchone()[0]
        p.once()
        self.assertEqual(p.connection.execute("SELECT COUNT(*) FROM members").fetchone()[0], count)

    def test_historical_source_accepts_exited_broker_without_paid_execution(self):
        p = self.f.open_tail()
        self.assertEqual(len(p.tail_registry()[0]), 2)

    def test_duplicate_fine_key_rejected(self):
        self.f.rebuild_tail(duplicate=True)
        with self.assertRaises(ValueError):
            self.f.open_tail()

    def test_dropped_fine_key_rejected(self):
        self.f.rebuild_tail(drop=True)
        with self.assertRaises(ValueError):
            self.f.open_tail()

    def test_changed_scientific_backend_rejected(self):
        self.f.rebuild_tail(changed_backend=True)
        with self.assertRaises(ValueError):
            self.f.open_tail()

    def test_wrong_parent_claim_rejected(self):
        self.f.tail_lease["parent_claim_path"] = str(self.f.root / "forbidden.claim")
        self.f.rebuild_tail()
        with self.assertRaises(ValueError):
            self.f.open_tail()

    def test_unsettled_checkpoint_rejected(self):
        self.f.tail_lease["pending_or_unknown_attempts"] = 1
        self.f.rebuild_tail()
        with self.assertRaises(ValueError):
            self.f.open_tail()

    def test_corrupt_partition_proof_rejected(self):
        self.f.tail_audit["tasks"][0]["status"] = "UNEXECUTED"
        self.f.rebuild_tail()
        with self.assertRaises(ValueError):
            self.f.open_tail()

    def test_controller_changed_bytes_rejected(self):
        Path(self.f.control_ref["path"]).write_text("changed\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.f.open_tail()

    def test_chunk_limit_and_checkpoint_count_rejected(self):
        self.f.tail_lease["maximum_child_keys"] = 65
        self.f.rebuild_tail()
        with self.assertRaises(ValueError):
            self.f.open_tail()

    def test_wrong_controller_reference_rejected(self):
        self.f.tail_lease["controller_source_ref"] = self.f.worker_ref
        self.f.rebuild_tail()
        with self.assertRaises(ValueError):
            self.f.open_tail()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporary-root", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification receipts are append-only")
    if args.temporary_root is not None:
        TailRegistryQualification.temporary_root = args.temporary_root.resolve()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TailRegistryQualification)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    value = dict(
        schema="e18_r1_tail_registry_qualification_v1", software_test_only=True,
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        status="PASS" if result.wasSuccessful() else "FAIL", source_sha256=original.sha(extension.__file__),
        qualification_source_sha256=original.sha(__file__), wall_seconds=time.perf_counter() - started,
        physical_model_calls=0, gold_values_read=0, native_answer_payloads_read=0,
        error_details=[dict(test=str(test), traceback=trace) for test, trace in result.errors],
        failure_details=[dict(test=str(test), traceback=trace) for test, trace in result.failures],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
