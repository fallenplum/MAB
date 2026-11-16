"""Software-only qualifications for safe retirement of an idle frozen follower."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import safe_idle_retirement as retirement


class MockClock:
    def __init__(self):
        self.value = 0.0
        self.on_sleep = None

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds
        if self.on_sleep is not None:
            self.on_sleep()


class MockProcesses:
    def __init__(self, request):
        self.request = request
        self.current = dict(pid=request["target_pid"], start_ticks=request["start_ticks"],
                            command=copy.deepcopy(request["command"]), state="S", sigterm_caught=True)
        self.alive = True
        self.frozen = False
        self.term_pending = False
        self.exit_after_continue = True
        self.actions = []
        self.claim_calls = 0
        self.claim_sequence = [[]]
        self.last_claims = []
        self.on_stop = None
        self.on_identity = None
        self.on_claim = None
        self.identity_error = False
        self.birth_error = False
        self.complete_selected_batches = 0

    def identity(self, pid):
        if self.on_identity is not None:
            self.on_identity()
        if self.identity_error:
            raise OSError("synthetic full identity unavailable")
        if not self.alive:
            return None
        if pid != self.request["target_pid"]:
            raise AssertionError("A different process was inspected")
        return copy.deepcopy(self.current)

    def start_ticks(self, pid):
        if self.birth_error:
            raise OSError("synthetic birth read unavailable")
        return self.current["start_ticks"] if self.alive else None

    def claim_fds(self, pid, claim_root):
        self.claim_calls += 1
        if self.on_claim is not None:
            self.on_claim()
        index = min(self.claim_calls - 1, len(self.claim_sequence) - 1)
        root = Path(claim_root).resolve()
        self.last_claims = [str(path) for path in self.claim_sequence[index] if Path(path).resolve().is_relative_to(root)]
        return self.last_claims.copy()

    def stop(self, pid):
        self.actions.append(("STOP", pid))
        if not self.alive:
            raise ProcessLookupError(pid)
        self.frozen = True
        self.current["state"] = "T"
        if self.on_stop is not None:
            self.on_stop()

    def terminate(self, pid):
        if not self.frozen or self.current["state"] != "T":
            raise AssertionError("TERM requires the previously verified stopped process")
        if self.current["start_ticks"] != self.request["start_ticks"] or self.current["command"] != self.request["command"]:
            raise AssertionError("TERM must never target a reused or changed process")
        self.actions.append(("TERM", pid))
        self.term_pending = True

    def continue_process(self, pid):
        if self.alive and self.current["start_ticks"] != self.request["start_ticks"]:
            raise AssertionError("CONT must never target a reused PID")
        self.actions.append(("CONT", pid))
        self.frozen = False
        self.current["state"] = "S"
        if self.term_pending and self.exit_after_continue:
            self.alive = False
        elif self.last_claims:
            self.complete_selected_batches += 1

    def kill(self, pid):
        self.actions.append(("KILL", pid))
        raise AssertionError("KILL is prohibited")


class RetirementQualification(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="e18_idle_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        source = self.root / "code/e18_auxiliary_aggregation_follower_v2.py"
        source.parent.mkdir()
        source.write_text("# isolated software fixture, never executable production source\n", encoding="utf-8")
        (self.root / "owned/_dispatcher/plans").mkdir(parents=True)
        self.claim = self.root / "owned/_dispatcher/plans/foreign-original-plan/shard.claim"
        queue_name = "gpu5_slot0"
        self.request = dict(
            schema="e18_safe_idle_retirement_request_v1", server_root=str(self.root), target_pid=7001,
            start_ticks="12345", command=["python", "-B", "-u", str(source), "--queue-root",
                str(self.root / "aggregation_queues" / queue_name), "--gpu-lock-root",
                str(self.root / "gpu-locks/aggregation" / queue_name), "--gpu", "7",
                "--planner-config", str(self.root / "AGGREGATION_INCREMENTAL_PLANNER_CONFIG_v1.json"),
                "--planner-config-sha256", "a" * 64],
            follower_source_ref=dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
            queue_name=queue_name, gpu="7", software_test_only=True, timeout_seconds=3, poll_seconds=1,
        )
        self.clock = MockClock()
        self.processes = MockProcesses(self.request)

    def actor(self, request=None, processes=None):
        return retirement.SafeIdleRetirement(
            self.request if request is None else request,
            self.processes if processes is None else processes,
            clock=self.clock.now, sleeper=self.clock.sleep,
        )

    def run_actor(self, actor=None, *, execute=True):
        actor = actor or self.actor()
        result = actor.run(execute=execute)
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        self.assertNotIn("KILL", [kind for kind, _ in self.processes.actions])
        return result

    def test_dryrun_eligible_has_no_signals(self):
        result = self.run_actor(execute=False)
        self.assertEqual(result["status"], "DRY_RUN_ELIGIBLE")
        self.assertEqual(self.processes.actions, [])
        self.assertTrue(self.processes.alive)

    def test_dryrun_claim_busy_has_no_signals(self):
        self.processes.claim_sequence = [[self.claim]]
        result = self.run_actor(execute=False)
        self.assertEqual(result["status"], "DRY_RUN_CLAIM_BUSY")
        self.assertEqual(self.processes.actions, [])

    def test_normal_retirement_stops_then_terms_then_continues_to_exit(self):
        result = self.run_actor()
        actions = [kind for kind, _ in self.processes.actions]
        self.assertEqual(result["status"], "EXITED")
        self.assertEqual(actions[:3], ["STOP", "TERM", "CONT"])
        self.assertFalse(self.processes.alive)
        self.assertFalse(self.processes.frozen)

    def test_claim_on_another_queue_blocks_term_until_timeout(self):
        self.processes.claim_sequence = [[self.claim]]
        result = self.run_actor()
        self.assertEqual(result["status"], "TIMEOUT_WAITING_IDLE")
        self.assertEqual(self.processes.actions, [])
        self.assertTrue(self.processes.alive)

    def test_claim_appearing_after_stop_resumes_without_term(self):
        self.processes.claim_sequence = [[], [self.claim], [self.claim]]
        result = self.run_actor()
        self.assertEqual(result["status"], "TIMEOUT_WAITING_IDLE")
        self.assertEqual([kind for kind, _ in self.processes.actions], ["STOP", "CONT"])
        self.assertFalse(self.processes.frozen)

    def test_already_selected_batch_can_complete_before_later_retirement(self):
        self.processes.claim_sequence = [[], [self.claim], [self.claim], [], []]
        result = self.run_actor()
        actions = [kind for kind, _ in self.processes.actions]
        self.assertEqual(result["status"], "EXITED")
        self.assertEqual(actions[:2], ["STOP", "CONT"])
        self.assertEqual(actions.count("TERM"), 1)
        self.assertEqual(self.processes.complete_selected_batches, 1)

    def test_pid_reuse_before_stop_has_no_signals(self):
        self.processes.current["start_ticks"] = "new-birth"
        with self.assertRaises((PermissionError, RuntimeError, ValueError)):
            self.run_actor()
        self.assertEqual(self.processes.actions, [])

    def test_full_command_change_before_stop_has_no_signals(self):
        self.processes.current["command"].append("--changed")
        with self.assertRaises((PermissionError, RuntimeError, ValueError)):
            self.run_actor()
        self.assertEqual(self.processes.actions, [])

    def test_pid_reuse_after_stop_never_signals_the_new_identity(self):
        def reuse():
            self.processes.current.update(start_ticks="new-birth", state="S")
            self.processes.frozen = False
        self.processes.on_stop = reuse
        with self.assertRaises((PermissionError, RuntimeError, ValueError)):
            self.run_actor()
        self.assertEqual([kind for kind, _ in self.processes.actions], ["STOP"])

    def test_changed_command_after_stop_is_never_terminated(self):
        def change():
            self.processes.current["command"].append("--changed-after-stop")
        self.processes.on_stop = change
        with self.assertRaises((PermissionError, RuntimeError, ValueError)):
            self.run_actor()
        self.assertNotIn("TERM", [kind for kind, _ in self.processes.actions])

    def test_claim_fd_failure_before_stop_fails_closed(self):
        def fail_claim_read():
            raise OSError("synthetic fd error")
        self.processes.on_claim = fail_claim_read
        with self.assertRaises(OSError):
            self.run_actor()
        self.assertEqual(self.processes.actions, [])

    def test_claim_fd_failure_after_stop_always_continues_same_process(self):
        def fail_second():
            if self.processes.claim_calls == 2:
                raise OSError("synthetic frozen fd error")
        self.processes.on_claim = fail_second
        with self.assertRaises(OSError):
            self.run_actor()
        self.assertEqual([kind for kind, _ in self.processes.actions], ["STOP", "CONT"])
        self.assertFalse(self.processes.frozen)

    def test_controller_interruption_after_stop_restores_same_process(self):
        def interrupt_second_read():
            if self.processes.claim_calls == 2:
                raise InterruptedError("synthetic controller interruption")
        self.processes.on_claim = interrupt_second_read
        with self.assertRaises(InterruptedError):
            self.run_actor()
        self.assertEqual([kind for kind, _ in self.processes.actions], ["STOP", "CONT"])
        self.assertFalse(self.processes.frozen)

    def test_external_continue_after_frozen_claim_check_refuses_term(self):
        def externally_continue():
            if self.processes.claim_calls == 2:
                self.processes.current["state"] = "S"
                self.processes.frozen = False
        self.processes.on_claim = externally_continue
        with self.assertRaises(RuntimeError):
            self.run_actor()
        self.assertEqual([kind for kind, _ in self.processes.actions], ["STOP", "CONT"])
        self.assertFalse(self.processes.frozen)

    def test_full_identity_failure_after_stop_uses_independent_birth_for_cont(self):
        self.processes.on_stop = lambda: setattr(self.processes, "identity_error", True)
        with self.assertRaises(OSError):
            self.run_actor()
        self.assertNotIn("TERM", [kind for kind, _ in self.processes.actions])
        self.assertFalse(self.processes.frozen)

    def test_both_identity_reads_failed_never_blindly_continue_or_term(self):
        def fail_reads():
            self.processes.identity_error = True
            self.processes.birth_error = True
        self.processes.on_stop = fail_reads
        actor = self.actor()
        with self.assertRaises((OSError, RuntimeError)):
            self.run_actor(actor)
        self.assertEqual([kind for kind, _ in self.processes.actions], ["STOP"])
        self.assertTrue(actor.events)

    def test_process_exiting_during_idle_wait_is_accepted_without_signals(self):
        self.processes.claim_sequence = [[self.claim]]
        self.clock.on_sleep = lambda: setattr(self.processes, "alive", False)
        result = self.run_actor()
        self.assertEqual(result["status"], "EXITED")
        self.assertEqual(self.processes.actions, [])

    def test_process_exit_at_stop_race_never_terms_another_process(self):
        original_stop = self.processes.stop
        def exit_before_stop(pid):
            self.processes.alive = False
            original_stop(pid)
        self.processes.stop = exit_before_stop
        actor = self.actor()
        try:
            result = self.run_actor(actor)
            self.assertEqual(result["status"], "EXITED")
        except ProcessLookupError:
            self.assertTrue(actor.events)
        self.assertNotIn("TERM", [kind for kind, _ in self.processes.actions])

    def test_timeout_after_term_never_escalates_to_kill(self):
        self.processes.exit_after_continue = False
        result = self.run_actor()
        self.assertEqual(result["status"], "TIMEOUT_AFTER_TERM")
        actions = [kind for kind, _ in self.processes.actions]
        self.assertEqual(actions.count("TERM"), 1)
        self.assertNotIn("KILL", actions)
        self.assertFalse(self.processes.frozen)

    def test_failed_stop_confirmation_resumes_and_does_not_term(self):
        self.processes.on_stop = lambda: self.processes.current.update(state="S")
        actor = self.actor()
        try:
            result = self.run_actor(actor)
            self.assertEqual(result["status"], "TIMEOUT_WAITING_IDLE")
        except (RuntimeError, TimeoutError):
            self.assertTrue(actor.events)
        self.assertNotIn("TERM", [kind for kind, _ in self.processes.actions])
        self.assertFalse(self.processes.frozen)

    def test_production_rejects_external_mock_processes(self):
        request = dict(self.request, software_test_only=False)
        with self.assertRaises(PermissionError):
            self.actor(request=request)
        self.assertEqual(self.processes.actions, [])

    def test_follower_source_hash_change_is_rejected(self):
        request = copy.deepcopy(self.request)
        request["follower_source_ref"]["sha256"] = "f" * 64
        with self.assertRaises((ValueError, PermissionError)):
            self.actor(request=request)

    def test_queue_alias_outside_original_three_is_rejected(self):
        request = dict(self.request, queue_name="foreign_queue")
        with self.assertRaises((ValueError, PermissionError)):
            self.actor(request=request)

    def test_nan_wait_bounds_are_rejected(self):
        for field in ("timeout_seconds", "poll_seconds"):
            with self.subTest(field=field):
                request = dict(self.request)
                request[field] = float("nan")
                with self.assertRaises(ValueError):
                    self.actor(request=request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(RetirementQualification)
    )
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    receipt = dict(
        schema="e18_safe_idle_retirement_software_qualification_v1",
        status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
        helper_source_sha256=sha(retirement.__file__), test_source_sha256=sha(__file__),
        software_test_only=True, actual_process_signals=0, actual_model_calls=0, real_gold_reads=0,
        production_state_files_written=0, real_retirement_work_during_wait_is_not_inferred_from_mock_counts=True,
        selected_batch_may_finish_before_idle=True,
        control_backend="MockProcesses, independent mock birth reader and MockClock in synthetic temporary directories",
        wall_seconds=time.perf_counter() - started, test_output=output.getvalue(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "test_output"}, ensure_ascii=False, indent=2))
    if not result.wasSuccessful():
        print(output.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
