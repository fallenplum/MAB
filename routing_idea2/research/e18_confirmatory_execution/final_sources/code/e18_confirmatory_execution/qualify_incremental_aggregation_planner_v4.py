"""Toy metadata qualification for explicit operational member2 source plans."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import incremental_aggregation_planner as original
import incremental_aggregation_planner_v4 as extension
from qualify_incremental_aggregation_planner import Fixture


class ExtendedFixture(Fixture):
    def __init__(self, root):
        super().__init__(root)
        self.scope["member_execution_keys"] = self.members
        self.scope_ref = self.artifact(self.inputs / "scope.json", self.scope)
        self.scope_sha = original.digest(self.scope)
        self.template["scope_ref"] = self.scope_ref
        self.template_ref = self.artifact(self.root / "startup_manifest.json", self.template)
        startup = original.read_json(self.startup_plan_ref["path"])
        startup["manifest_refs"] = [self.template_ref]
        self.startup_plan_ref = self.artifact(self.root / "startup_plan.json", startup)
        for member, reference in list(self.member_refs.items()):
            manifest = original.read_json(reference["path"])
            manifest["scope_ref"] = self.scope_ref
            self.member_refs[member] = self.artifact(reference["path"], manifest)
        first = original.read_json(self.member_refs[2]["path"])
        first["task_keys_ref"] = self.artifact(self.inputs / "parent2a_keys.json", [[self.qids[0], 42, 2]])
        first_ref = self.artifact(self.root / "parent2a.json", first)
        second = dict(first)
        second["task_keys_ref"] = self.artifact(self.inputs / "parent2b_keys.json", [[self.qids[1], 42, 2], [self.qids[2], 42, 2]])
        second_ref = self.artifact(self.root / "parent2b.json", second)
        self.parents = [first_ref, second_ref]
        self.member_plan_ref = self.artifact(self.root / "member_plan.json", dict(
            schema="e18_gpu_dispatch_plan_v1",
            worker_source_sha256=original.sha(self.code / "e18_gpu_worker.py"),
            manifest_refs=self.parents + [self.member_refs[4]],
        ))
        member4 = original.read_json(self.member4_ref["path"])
        member4["scope_ref"] = self.scope_ref
        self.member4_ref = self.artifact(self.member4_ref["path"], member4)
        self.member4_plan_ref = self.artifact(self.root / "member4_plan.json", dict(
            schema="e18_member4_dispatch_plan_v3",
            worker_source_sha256=original.sha(self.code / "e18_member4_worker_v3.py"),
            manifest_refs=[self.member4_ref],
        ))
        self.config.update(
            scope_ref=self.scope_ref, template_manifest_ref=self.template_ref,
            startup_plan_ref=self.startup_plan_ref,
            source_plan_refs=[self.member_plan_ref, self.member4_plan_ref],
        )
        self.config_ref = self.artifact(self.root / "config.json", self.config)
        worker_path = self.code / extension.MEMBER2_WORKER_NAME
        worker_path.write_text("# synthetic member2 worker identity\n", encoding="utf-8")
        self.worker_ref = dict(path=str(worker_path), sha256=original.sha(worker_path))
        self.lease_root = self.output / "_dispatcher/member2_checkpoint"
        self.remaining = [[self.qids[1], 42, 2], [self.qids[2], 42, 2]]
        remaining_ref = self.artifact(self.lease_root / "remaining_task_keys.json", self.remaining)
        tasks = [dict(status="SUCCESS", task_key_sha256=original.digest([self.qids[0], 42, 2]), native_record_sha256="a" * 64, native_receipt_file_sha256="b" * 64)]
        tasks += [dict(status="UNEXECUTED", task_key_sha256=original.digest(row)) for row in self.remaining]
        audit_ref = self.artifact(self.lease_root / "CHECKPOINT_SOURCE_AUDIT.json", dict(
            schema="e18_member2_native_checkpoint_audit_v1", tasks=tasks,
            original_tasks=3, checkpoint_successes=1, pending_or_unknown_attempts=0,
        ))
        self.lease = dict(
            schema="e18_member2_checkpoint_lease_v1", status="LEASED",
            base_member_plan_sha256=self.member_plan_ref["sha256"],
            execution_scope_sha256=self.scope_sha, approval_sha256=self.approval_ref["sha256"],
            target_member_index=2, owner_pid=999, former_pids=[101, 202],
            parent_shards=[dict(
                manifest_sha256=ref["sha256"],
                task_keys_sha256=original.read_json(ref["path"])["task_keys_ref"]["sha256"],
                claim_path=str(self.output / "_dispatcher/plans" / self.member_plan_ref["sha256"] / (ref["sha256"] + ".claim")),
            ) for ref in self.parents],
            remaining_task_keys_ref=remaining_ref, original_member2_tasks=3,
            checkpoint_successes=1, remaining_count=2, checkpoint_stop_verified=True,
            pending_or_unknown_attempts=0, checkpoint_source_audit_sha256=audit_ref["sha256"],
            software_test_only=False,
        )
        self.rebuild_extra()

    def rebuild_extra(self, *, duplicate=False, drop=False, changed_backend=False):
        self.lease_ref = self.artifact(self.lease_root / "LEASE.json", self.lease)
        self.child_refs = []
        parent = original.read_json(self.parents[0]["path"])
        for index, key in enumerate(self.remaining):
            child = dict(parent, schema=extension.MEMBER2_MANIFEST_SCHEMA, checkpoint_lease_ref=self.lease_ref)
            child["task_keys_ref"] = self.artifact(self.inputs / ("child_keys_" + str(index) + ".json"), [self.remaining[0] if duplicate else key])
            if changed_backend:
                child["backend"] = dict(mode="other")
            self.child_refs.append(self.artifact(self.root / ("child_" + str(index) + ".json"), child))
        plan = dict(
            schema=extension.MEMBER2_PLAN_SCHEMA, worker_source_sha256=self.worker_ref["sha256"],
            checkpoint_lease_ref=self.lease_ref, manifest_refs=self.child_refs[:-1] if drop else self.child_refs,
        )
        self.extra_ref = self.artifact(self.root / "child_plan.json", plan)

    def extended(self, **kwargs):
        instance = extension.IncrementalPlanner(
            self.config_ref, qualification_only=True,
            extra_source_plan_refs=[self.extra_ref], member2_worker_source_ref=self.worker_ref,
            **kwargs,
        )
        self.connections.append(instance.connection)
        return instance


class ExtensionQualification(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[2] / ".agg_toy"
        parent.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="v4_", dir=parent)
        self.addCleanup(temporary.cleanup)
        self.f = ExtendedFixture(temporary.name)
        self.addCleanup(self.f.close)

    def test_reopen_original_ledger_preserves_config_ready_and_all_done_identity(self):
        f = self.f
        initial = f.open()
        initial.once()
        config_bytes = Path(f.config_ref["path"]).read_bytes()
        ready_hashes = {str(path): original.sha(path) for path in (f.root / "aggregation_queues").glob("*/*.ready.json")}
        initial.connection.close()
        p = f.extended()
        f.append_event(f.child_refs[1], (f.qids[2], 42, 2), "member_generation")
        result = p.once()
        self.assertEqual(result["newly_reserved_keys"], 1)
        self.assertEqual(Path(f.config_ref["path"]).read_bytes(), config_bytes)
        self.assertEqual(p.config_sha, f.config_ref["sha256"])
        self.assertEqual(p.connection.execute("SELECT value FROM metadata WHERE key='config_sha'").fetchone()[0], f.config_ref["sha256"])
        self.assertTrue(all(original.sha(path) == value for path, value in ready_hashes.items()))
        f.append_event(f.member4_ref, (f.qids[2], 42, 4), "member_generation")
        p.once()
        f.complete_all(p)
        self.assertEqual(p.once()["status"], "ALL_DONE")
        for name in original.QUEUES:
            self.assertEqual(original.read_json(f.root / "aggregation_queues" / name / "ALL_DONE")["config_sha256"], f.config_ref["sha256"])
        self.assertTrue(original.checked_ref(p.extension_ref).is_file())

    def test_native_answer_sentinels_are_never_decoded_and_repeat_does_not_duplicate(self):
        f = self.f
        p = f.extended()
        f.append_event(f.child_refs[1], (f.qids[2], 42, 2), "member_generation")
        p.once()
        count = p.connection.execute("SELECT COUNT(*) FROM members").fetchone()[0]
        self.assertEqual(p.once()["newly_reserved_keys"], 0)
        self.assertEqual(p.connection.execute("SELECT COUNT(*) FROM members").fetchone()[0], count)

    def test_duplicate_child_key_rejected(self):
        self.f.rebuild_extra(duplicate=True)
        with self.assertRaises(ValueError):
            self.f.extended()

    def test_incomplete_remaining_union_rejected(self):
        self.f.rebuild_extra(drop=True)
        with self.assertRaises(ValueError):
            self.f.extended()

    def test_changed_original_backend_rejected(self):
        self.f.rebuild_extra(changed_backend=True)
        with self.assertRaises(ValueError):
            self.f.extended()

    def test_source_sha_change_rejected(self):
        Path(self.f.worker_ref["path"]).write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.f.extended()

    def test_wrong_parent_claim_rejected(self):
        self.f.lease["parent_shards"][0]["claim_path"] = str(self.f.root / "foreign.claim")
        self.f.rebuild_extra()
        with self.assertRaises(ValueError):
            self.f.extended()

    def test_unsettled_lease_rejected(self):
        self.f.lease["pending_or_unknown_attempts"] = 1
        self.f.rebuild_extra()
        with self.assertRaises(ValueError):
            self.f.extended()

    def test_corrupt_bound_checkpoint_partition_rejected(self):
        f = self.f
        path = f.lease_root / "CHECKPOINT_SOURCE_AUDIT.json"
        audit = original.read_json(path)
        audit["tasks"][0]["status"] = "UNEXECUTED"
        f.artifact(path, audit)
        f.lease["checkpoint_source_audit_sha256"] = original.sha(path)
        f.rebuild_extra()
        with self.assertRaises(ValueError):
            f.extended()

    def test_extra_native_failure_requires_reconciliation(self):
        f = self.f
        p = f.extended()
        f.append_event(f.child_refs[1], (f.qids[2], 42, 2), "member_generation", status="FAILED_REQUIRES_RECONCILIATION")
        with self.assertRaises(RuntimeError):
            p.once()
        self.assertTrue(p.stop_path.exists())

    def test_without_extra_refs_reuses_original_planner(self):
        f = self.f
        p = extension.IncrementalPlanner(f.config_ref, qualification_only=True)
        f.connections.append(p.connection)
        self.assertEqual(p.once()["newly_reserved_keys"], 3)
        self.assertEqual(p.config_sha, f.config_ref["sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification receipts are append-only")
    started = time.perf_counter()
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExtensionQualification))
    report = dict(
        schema="e18_planner_operational_extension_qualification_v1",
        status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), wall_seconds=time.perf_counter() - started,
        planner_source_sha256=original.sha(extension.__file__), qualification_source_sha256=original.sha(__file__),
        inherited_planner_source_sha256=original.sha(original.__file__), test_output=capture.getvalue(),
        actual_gpu_queries=0, actual_model_calls=0, real_gold_reads=0,
        real_native_answer_payloads_read=0, production_queue_entries_published=0,
        server_logins=0, original_config_or_source_files_modified=False,
    )
    args.output.write_bytes(original.encoded(report))
    print(json.dumps({key: report[key] for key in ("status", "tests", "failures", "errors", "wall_seconds")}))
    if not result.wasSuccessful():
        print(capture.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
