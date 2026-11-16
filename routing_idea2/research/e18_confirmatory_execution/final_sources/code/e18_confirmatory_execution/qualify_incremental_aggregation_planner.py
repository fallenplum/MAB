"""Toy-only qualification of E18 incremental aggregation reservations."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import incremental_aggregation_planner as planner


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.connections = []
        self.code = self.root / "code"
        self.code.mkdir()
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.output = self.root / "owned"
        self.output.mkdir()
        for name in [
            "e18_gpu_worker.py",
            "e18_member4_worker_v3.py",
            "e18_aggregation_worker_v3.py",
            "e18_execution_adapter.py",
        ]:
            (self.code / name).write_text("# synthetic metadata " + name + "\n", encoding="utf-8")
        self.qids = ["synthetic-question-" + str(index) for index in range(3)]
        self.aggregates = [[qid, 42, action] for qid in self.qids for action in (0, 2)]
        self.members = sorted(
            {
                (qid, 42, member)
                for qid, _, action in self.aggregates
                for member in planner.TEAMS[action]
            }
        )
        self.scope = dict(
            schema="e18_scope_contract_v2",
            role="execution",
            cohort="e18_confirm",
            question_keys=[
                dict(question_id=qid, canonical_question_sha256=planner.digest(qid))
                for qid in self.qids
            ],
            required_aggregate_keys=self.aggregates,
            aggregate_execution_keys=self.aggregates,
            required_member_keys=self.members,
        )
        self.scope_ref = self.artifact(self.inputs / "scope.json", self.scope)
        self.scope_sha = planner.digest(self.scope)
        self.approval_ref = self.artifact(
            self.inputs / "approval.json", dict(schema="synthetic", status="APPROVED")
        )
        self.input_ref = self.artifact(
            self.inputs / "inputs.json", dict(schema="synthetic_input_identity_only")
        )
        historical = [key for key in self.members if key[0] != self.qids[2] or key[2] in (0, 1)]
        rows = [
            dict(
                task_key=list(key),
                record_ref=dict(
                    path=str(self.root / "historical" / (planner.digest(list(key)) + ".json")),
                    sha256=planner.digest(["proof", *key]),
                ),
            )
            for key in historical
        ]
        self.history_ref = self.artifact(self.inputs / "historical.json", rows)
        self.startup = [self.aggregates[0]]
        self.startup_ref = self.artifact(self.inputs / "startup_keys.json", self.startup)
        self.template = dict(
            schema="e18_aggregation_worker_manifest_v3",
            software_test_only=False,
            stage="aggregation",
            asset_root=str(self.root / "runtime_assets"),
            adapter_ref=dict(
                path=str(self.code / "e18_execution_adapter.py"),
                sha256=planner.sha(self.code / "e18_execution_adapter.py"),
            ),
            scope_ref=self.scope_ref,
            approval_ref=self.approval_ref,
            question_records_ref=self.input_ref,
            owned_output_root=str(self.output),
            task_keys_ref=self.startup_ref,
            gpu_lock_root=str(self.root / "gpu-locks"),
            install_signal_handlers=True,
            member_dependencies_ref=self.history_ref,
            backend=dict(
                mode="mapped", mapping_ref=dict(path=str(self.root / "map.json"), sha256="e" * 64)
            ),
        )
        self.template_ref = self.artifact(self.root / "startup_manifest.json", self.template)
        self.startup_plan_ref = self.artifact(
            self.root / "startup_plan.json",
            dict(
                schema="e18_aggregation_dispatch_plan_v3",
                worker_source_sha256=planner.sha(self.code / "e18_aggregation_worker_v3.py"),
                manifest_refs=[self.template_ref],
            ),
        )
        self.member_refs = {}
        for member in (2, 4):
            key_ref = self.artifact(
                self.inputs / f"member{member}_keys.json", [[self.qids[2], 42, member]]
            )
            manifest = dict(
                self.template,
                schema="e18_gpu_worker_manifest_v1",
                stage="member_generation",
                member_index=member,
                task_keys_ref=key_ref,
            )
            self.member_refs[member] = self.artifact(
                self.root / f"member{member}_manifest.json", manifest
            )
        self.member_plan_ref = self.artifact(
            self.root / "member_plan.json",
            dict(
                schema="e18_gpu_dispatch_plan_v1",
                worker_source_sha256=planner.sha(self.code / "e18_gpu_worker.py"),
                manifest_refs=list(self.member_refs.values()),
            ),
        )
        member4 = dict(
            planner.read_json(self.member_refs[4]["path"]),
            schema="e18_member4_worker_manifest_v3",
            checkpoint_lease_ref=dict(
                path=str(self.root / "synthetic_lease.json"), sha256="f" * 64
            ),
        )
        self.member4_ref = self.artifact(self.root / "member4_v3_manifest.json", member4)
        self.member4_plan_ref = self.artifact(
            self.root / "member4_plan.json",
            dict(
                schema="e18_member4_dispatch_plan_v3",
                worker_source_sha256=planner.sha(self.code / "e18_member4_worker_v3.py"),
                manifest_refs=[self.member4_ref],
            ),
        )
        self.config = dict(
            schema="e18_incremental_aggregation_planner_config_v1",
            server_root=str(self.root),
            scope_ref=self.scope_ref,
            template_manifest_ref=self.template_ref,
            historical_dependencies_ref=self.history_ref,
            startup_keys_ref=self.startup_ref,
            startup_plan_ref=self.startup_plan_ref,
            source_plan_refs=[self.member_plan_ref, self.member4_plan_ref],
            database_path=str(self.root / "planner/ledger.sqlite"),
            batch_max=1,
            queues=[
                dict(
                    name=name,
                    queue_root=str(self.root / "aggregation_queues" / name),
                    gpu_lock_root=str(self.root / "gpu-locks/aggregation" / name),
                )
                for name in planner.QUEUES
            ],
        )
        self.config_ref = self.artifact(self.root / "config.json", self.config)

    def artifact(self, path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(planner.encoded(value))
        return dict(path=str(path), sha256=planner.sha(path))

    def open(self, config_ref=None):
        instance = planner.IncrementalPlanner(
            config_ref or self.config_ref, qualification_only=True
        )
        self.connections.append(instance.connection)
        return instance

    def close(self):
        for connection in self.connections:
            connection.close()

    def append_event(self, reference, key, stage, *, status="SUCCESS", newline=True):
        proof = planner.digest(["native", stage, *key])
        attempt = "synthetic-attempt-" + proof
        owner = self.output / self.scope_sha / stage / planner.digest(list(key))
        owner.mkdir(parents=True, exist_ok=True)
        self.artifact(
            owner / "ATTEMPT_STATE.json",
            dict(
                status="SUCCESS_COMMITTED",
                scope_sha256=self.scope_sha,
                approval_sha256=self.approval_ref["sha256"],
                task_key=list(key),
                stage=stage,
                native_record_sha256=proof,
                software_test_only=False,
            ),
        )
        # These non-JSON sentinels make accidental native-payload decoding fail.
        (owner / "LAST_TASK_RECEIPT.json").write_text(
            "FORBIDDEN_NATIVE_ANSWER_SENTINEL", encoding="utf-8"
        )
        (owner / "records.jsonl").write_text("FORBIDDEN_NATIVE_ANSWER_SENTINEL", encoding="utf-8")
        task = dict(stage=stage, question_id=key[0], generation_seed=key[1])
        task["member_index" if stage == "member_generation" else "combination_index"] = key[2]
        event = dict(
            event="TASK_RESULT",
            task=task,
            status=status,
            physical_attempt_id=attempt,
            native_record_sha256=proof,
            from_cache=False,
            actual_usage=dict(prompt_tokens=12, completion_tokens=9),
            at_unix=1,
        )
        path = self.output / "_dispatcher" / reference["sha256"] / "EVENTS.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as stream:
            stream.write(json.dumps(event).encode() + (b"\n" if newline else b""))
        return path, owner

    def complete_all(self, instance, *, startup_done=True):
        for row in instance.connection.execute(
            "SELECT qid,generation,action,batch_id FROM aggregates"
        ):
            key = tuple(row[:3])
            if row[3] is None:
                reference = self.template_ref
            else:
                batch = instance.connection.execute(
                    "SELECT manifest_path,manifest_sha FROM batches WHERE id=?", (row[3],)
                ).fetchone()
                reference = dict(path=batch[0], sha256=batch[1])
            self.append_event(reference, key, "aggregation")
        if startup_done:
            self.artifact(
                self.output
                / "_dispatcher/plans"
                / self.startup_plan_ref["sha256"]
                / (self.template_ref["sha256"] + ".status.json"),
                dict(status="DONE", manifest_sha256=self.template_ref["sha256"]),
            )


class PlannerQualification(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[2] / ".agg_toy"
        parent.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="p_", dir=parent)
        self.addCleanup(temporary.cleanup)
        self.fixture = Fixture(temporary.name)
        self.addCleanup(self.fixture.close)

    def test_ready_keys_partition_three_queues_without_overlap(self):
        f = self.fixture
        p = f.open()
        result = p.once()
        self.assertEqual(result["newly_reserved_keys"], 3)
        rows = p.connection.execute("SELECT key_rows,queue FROM batches").fetchall()
        all_keys = [tuple(key) for row in rows for key in json.loads(row[0])]
        self.assertEqual(len(all_keys), len(set(all_keys)))
        self.assertEqual({row[1] for row in rows}, set(planner.QUEUES))

    def test_startup_reserved_keys_never_enter_queue(self):
        f = self.fixture
        p = f.open()
        p.once()
        keys = [
            tuple(key)
            for row in p.connection.execute("SELECT key_rows FROM batches")
            for key in json.loads(row[0])
        ]
        self.assertNotIn(tuple(f.startup[0]), keys)

    def test_partial_member_dependencies_do_not_publish(self):
        f = self.fixture
        p = f.open()
        p.once()
        self.assertEqual(
            p.connection.execute(
                "SELECT COUNT(*) FROM aggregates WHERE qid=? AND status='WAITING'", (f.qids[2],)
            ).fetchone()[0],
            2,
        )

    def test_new_member_events_unlock_only_legally_ready_combinations(self):
        f = self.fixture
        p = f.open()
        p.once()
        f.append_event(f.member_refs[2], (f.qids[2], 42, 2), "member_generation")
        result = p.once()
        self.assertEqual(result["newly_reserved_keys"], 1)
        f.append_event(f.member4_ref, (f.qids[2], 42, 4), "member_generation")
        self.assertEqual(p.once()["newly_reserved_keys"], 1)

    def test_commit_before_publish_recovers_without_duplicate_reservation(self):
        f = self.fixture
        p = f.open()
        p.refresh()
        self.assertEqual(p.reserve(), 3)
        self.assertFalse(any((f.root / "aggregation_queues").glob("*/*.ready.json")))
        p.connection.close()
        q = f.open()
        result = q.once()
        self.assertEqual(result["recovered_reserved_batches"], 3)
        self.assertEqual(result["newly_reserved_keys"], 0)
        self.assertEqual(q.connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0], 3)

    def test_repeated_once_is_idempotent(self):
        p = self.fixture.open()
        p.once()
        self.assertEqual(p.once()["newly_reserved_keys"], 0)
        self.assertEqual(p.connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0], 3)

    def test_queue_artifact_tampering_fails_closed(self):
        f = self.fixture
        p = f.open()
        p.once()
        path = next((f.root / "aggregation_queues").glob("*/*.ready.json"))
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            p.once()
        self.assertTrue(p.stop_path.is_file())

    def test_unknown_ready_entry_without_reservation_fails(self):
        f = self.fixture
        p = f.open()
        p.once()
        f.artifact(
            f.root / "aggregation_queues/gpu5_slot0/foreign.ready.json", dict(schema="foreign")
        )
        with self.assertRaises(RuntimeError):
            p.once()

    def test_unknown_configuration_schema_rejected(self):
        f = self.fixture
        config = dict(f.config, schema="unknown")
        with self.assertRaises(ValueError):
            f.open(f.artifact(f.root / "bad_config.json", config))

    def test_source_manifest_wrong_scope_rejected(self):
        f = self.fixture
        manifest = planner.read_json(f.member_refs[2]["path"])
        manifest["scope_ref"] = dict(manifest["scope_ref"], sha256="1" * 64)
        ref = f.artifact(f.root / "wrong_scope_manifest.json", manifest)
        source_plan = f.artifact(
            f.root / "wrong_scope_plan.json",
            dict(
                schema="e18_gpu_dispatch_plan_v1",
                worker_source_sha256=planner.sha(f.code / "e18_gpu_worker.py"),
                manifest_refs=[ref],
            ),
        )
        config = dict(f.config, source_plan_refs=[source_plan])
        p = f.open(f.artifact(f.root / "bad_config.json", config))
        with self.assertRaises(ValueError):
            p.once()

    def test_unknown_source_plan_schema_rejected(self):
        f = self.fixture
        source = f.artifact(f.root / "unknown_plan.json", dict(schema="unknown", manifest_refs=[]))
        p = f.open(
            f.artifact(f.root / "bad_config.json", dict(f.config, source_plan_refs=[source]))
        )
        with self.assertRaises(ValueError):
            p.once()

    def test_out_of_scope_manifest_key_rejected(self):
        f = self.fixture
        manifest = planner.read_json(f.member_refs[2]["path"])
        manifest["task_keys_ref"] = f.artifact(f.inputs / "foreign_key.json", [["foreign", 42, 2]])
        ref = f.artifact(f.root / "foreign_manifest.json", manifest)
        source = f.artifact(
            f.root / "foreign_plan.json",
            dict(
                schema="e18_gpu_dispatch_plan_v1",
                worker_source_sha256=planner.sha(f.code / "e18_gpu_worker.py"),
                manifest_refs=[ref],
            ),
        )
        p = f.open(
            f.artifact(f.root / "bad_config.json", dict(f.config, source_plan_refs=[source]))
        )
        with self.assertRaises(ValueError):
            p.once()

    def test_unknown_worker_journal_rejected(self):
        f = self.fixture
        p = f.open()
        f.append_event(dict(sha256="2" * 64), (f.qids[2], 42, 2), "member_generation")
        with self.assertRaises(ValueError):
            p.once()

    def test_broker_control_journal_is_metadata_only_and_ignored(self):
        f = self.fixture
        p = f.open()
        path = f.output / "_dispatcher/member4_checkpoint/EVENTS.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('{"event":"BROKER_STARTED","owner_pid":123}\n', encoding="utf-8")
        self.assertEqual(p.once()["newly_reserved_keys"], 3)

    def test_failed_event_never_gets_skipped(self):
        f = self.fixture
        p = f.open()
        f.append_event(
            f.member_refs[2],
            (f.qids[2], 42, 2),
            "member_generation",
            status="FAILED_REQUIRES_RECONCILIATION",
        )
        with self.assertRaises(RuntimeError):
            p.once()
        self.assertTrue(p.stop_path.exists())

    def test_success_requires_committed_fence_metadata(self):
        f = self.fixture
        p = f.open()
        _, owner = f.append_event(f.member_refs[2], (f.qids[2], 42, 2), "member_generation")
        fence = planner.read_json(owner / "ATTEMPT_STATE.json")
        fence["status"] = "RESERVED_BEFORE_BACKEND"
        f.artifact(owner / "ATTEMPT_STATE.json", fence)
        with self.assertRaises(RuntimeError):
            p.once()

    def test_partial_journal_tail_is_deferred_until_newline(self):
        f = self.fixture
        p = f.open()
        path, _ = f.append_event(
            f.member_refs[2], (f.qids[2], 42, 2), "member_generation", newline=False
        )
        self.assertEqual(p.once()["newly_reserved_keys"], 3)
        with path.open("ab") as stream:
            stream.write(b"\n")
        self.assertEqual(p.once()["newly_reserved_keys"], 1)

    def test_ingested_journal_rewrite_rejected(self):
        f = self.fixture
        p = f.open()
        path, _ = f.append_event(f.member_refs[2], (f.qids[2], 42, 2), "member_generation")
        p.once()
        path.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            p.once()

    def test_all_done_requires_startup_done_and_every_frozen_success(self):
        f = self.fixture
        p = f.open()
        p.once()
        f.append_event(f.member_refs[2], (f.qids[2], 42, 2), "member_generation")
        f.append_event(f.member4_ref, (f.qids[2], 42, 4), "member_generation")
        p.once()
        f.complete_all(p, startup_done=False)
        self.assertEqual(p.once()["status"], "RUNNING")
        f.artifact(
            f.output
            / "_dispatcher/plans"
            / f.startup_plan_ref["sha256"]
            / (f.template_ref["sha256"] + ".status.json"),
            dict(status="DONE", manifest_sha256=f.template_ref["sha256"]),
        )
        self.assertEqual(p.once()["status"], "ALL_DONE")
        for name in planner.QUEUES:
            self.assertTrue((f.root / "aggregation_queues" / name / "ALL_DONE").exists())

    def test_ready_schema_exact_and_queue_lock_paths_independent(self):
        f = self.fixture
        p = f.open()
        p.once()
        locks = set()
        for path in (f.root / "aggregation_queues").glob("*/*.ready.json"):
            entry = planner.read_json(path)
            self.assertEqual(set(entry), {"schema", "plan_ref"})
            self.assertEqual(entry["schema"], "e18_aggregation_queue_entry_v1")
            plan = planner.read_json(planner.checked_ref(entry["plan_ref"]))
            manifest = planner.read_json(planner.checked_ref(plan["manifest_refs"][0]))
            locks.add(manifest["gpu_lock_root"])
        self.assertEqual(len(locks), 3)

    def test_follow_fake_clock_exits_after_all_done(self):
        instance = self.fixture.open()
        results = [dict(status="RUNNING"), dict(status="RUNNING"), dict(status="ALL_DONE")]
        instance.once = lambda: results.pop(0)
        sleeps = []
        with redirect_stdout(io.StringIO()):
            final = instance.follow(60, sleeper=sleeps.append)
        self.assertEqual(final["status"], "ALL_DONE")
        self.assertEqual(sleeps, [60, 60])

    def test_follow_propagates_failure_without_retry(self):
        instance = self.fixture.open()
        calls = []

        def fail_once():
            calls.append("entered")
            raise RuntimeError("synthetic planner stop")

        instance.once = fail_once
        with self.assertRaises(RuntimeError):
            instance.follow(60, sleeper=lambda _: self.fail("Failure retried"))
        self.assertEqual(calls, ["entered"])
        self.assertTrue(instance.stop_path.exists())

    def test_follow_idle_exception_keeps_stop_without_retry(self):
        instance = self.fixture.open()
        calls = []
        instance.once = lambda: calls.append("entered") or dict(status="RUNNING")

        def interrupted_sleep(_):
            raise RuntimeError("synthetic idle interruption")

        with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            instance.follow(60, sleeper=interrupted_sleep)
        self.assertEqual(calls, ["entered"])
        self.assertTrue(instance.stop_path.exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=text, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PlannerQualification)
    )
    receipt = dict(
        schema="e18_incremental_aggregation_planner_toy_qualification_v1",
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,
        planner_source_sha256=planner.sha(Path(planner.__file__)),
        test_source_sha256=planner.sha(Path(__file__)),
        actual_model_calls=0,
        real_gold_reads=0,
        real_native_answer_payloads_read=0,
        production_queue_entries_published=0,
        wall_seconds=time.perf_counter() - started,
        test_output=text.getvalue(),
    )
    args.output.write_bytes(planner.encoded(receipt))
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "test_output"},
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result.wasSuccessful():
        print(text.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
