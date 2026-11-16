"""Pure mocked telemetry/process qualification plus original paid-guard glue."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import cooperative_slot_admission as admission


class Identities:
    def __init__(self):
        self.rows = {}
        self.locks = {}

    def identity(self, pid):
        return copy.deepcopy(self.rows.get(pid))

    def lock_owner(self, path):
        return self.locks.get(str(Path(path).resolve()))


class MockOwnership:
    def __init__(self, identities, root, gpu, sha, slot):
        self.identities = identities
        self.root = Path(root).resolve()
        self.file_key = "gpu-" + str(gpu) + (".slot-1" if slot else "")
        self.path = str(self.root / (self.file_key + ".lock"))

    def __enter__(self):
        if self.path in self.identities.locks:
            raise RuntimeError("Mock original slot already owned")
        self.identities.locks[self.path] = os.getpid()
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *args):
        self.identities.locks.pop(self.path, None)
        return False


class CooperativeAdmissionQualification(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="e18_cooperative_qualification_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "study"
        (self.root / "code").mkdir(parents=True)
        (self.root / "owned").mkdir()
        self.requester = self.root / "code/member4_tail_partition.py"
        self.requester.write_text("synthetic requester", encoding="utf-8")
        self.own_source = self.root / "code/own_actor.py"
        self.own_source.write_text("synthetic known own actor", encoding="utf-8")
        self.scope = dict(schema="e18_scope_contract_v2", role="execution", cohort="e18_confirm")
        self.count = 0
        self.manifest = dict(
            schema="e18_member4_worker_manifest_v3", stage="member_generation", member_index=4,
            scope_ref=self.artifact(self.scope), approval_ref=self.artifact({"status": "synthetic"}),
            owned_output_root=str(self.root / "owned"), gpu_lock_root=str(self.root / "locks"),
            adapter_ref=dict(path="synthetic_adapter", sha256="1" * 64),
            question_records_ref=dict(path="synthetic_inputs", sha256="2" * 64), backend=dict(mode="synthetic"),
        )
        self.processes = Identities()
        self.processes.rows[os.getpid()] = self.identity(os.getpid(), self.requester)
        self.processes.rows[111] = self.identity(111, self.own_source)
        self.processes.rows[222] = self.identity(222, Path("/foreign/project.py"))
        self.state = dict(index=0, uuid="GPU-SYNTHETIC", free_memory_bytes=20 * 1024**3, compute_pids=[111, 222])
        self.policy = dict(
            schema="e18_cooperative_gpu_slot_policy_v1", cohort="e18_confirm", software_test_only=True,
            scope_sha256=admission.digest(self.scope), approval_sha256=self.manifest["approval_ref"]["sha256"],
            gpu_lock_root=self.manifest["gpu_lock_root"], max_total_compute=3, max_own_e18=2,
            minimum_free_memory_bytes=admission.MINIMUM_FREE_BYTES,
            requester_source_ref=self.reference(self.requester),
            gpus=[dict(index=0, uuid="GPU-SYNTHETIC", processes=[self.declaration(111, "own_e18"), self.declaration(222, "foreign")])],
        )
        self.worker = SimpleNamespace(
            __file__=str(self.own_source), _ACTIVE_GPU_SLOT_CONTEXT=None,
            GPUOwnership=lambda root, gpu, sha, slot: MockOwnership(self.processes, root, gpu, sha, slot),
            save_json=lambda path, value: Path(path).write_text(json.dumps(value), encoding="utf-8"),
        )

    def reference(self, path):
        return dict(path=str(path), sha256=admission.file_sha(path))

    def artifact(self, value):
        self.count += 1
        path = self.root / ("meta_" + str(self.count) + ".json")
        path.write_text(json.dumps(value), encoding="utf-8")
        return self.reference(path)

    def identity(self, pid, source, gpu="0"):
        command = [sys.executable, str(source), "--gpu", gpu]
        raw = b"\0".join(item.encode() for item in command) + b"\0"
        return dict(pid=pid, start_ticks=str(pid + 100), command=command, cmdline_sha256=hashlib.sha256(raw).hexdigest())

    def declaration(self, pid, role):
        row = self.processes.rows[pid]
        value = dict(pid=pid, start_ticks=row["start_ticks"], cmdline_sha256=row["cmdline_sha256"], role=role)
        if role == "own_e18":
            value["source_ref"] = self.reference(self.own_source)
        return value

    def context(self, slot=0, worker=None):
        return admission.CooperativeSlotAdmission(
            worker or self.worker, self.manifest, "3" * 64, "0", slot, self.artifact(self.policy),
            software_test_only=True, probe=lambda gpu: copy.deepcopy(self.state), processes=self.processes,
        )

    def test_one_own_plus_foreign_admits_requester_as_third_compute(self):
        with self.context() as receipt:
            self.assertIs(self.worker._ACTIVE_GPU_SLOT_CONTEXT, receipt)
            self.assertEqual((receipt["admitted_total_with_requester"], receipt["admitted_own_with_requester"]), (3, 2))
            self.assertEqual(receipt["known_e18_compute_pids"], [111])
            self.assertEqual(receipt["foreign_compute_pids"], [222])
        self.assertIsNone(self.worker._ACTIVE_GPU_SLOT_CONTEXT)
        self.assertEqual(self.processes.locks, {})

    def test_slot0_also_requires12gib(self):
        self.state["free_memory_bytes"] = admission.MINIMUM_FREE_BYTES - 1
        with self.assertRaises(PermissionError):
            with self.context(0):
                pass

    def test_slot1_requires12gib(self):
        self.state["free_memory_bytes"] = admission.MINIMUM_FREE_BYTES - 1
        with self.assertRaises(PermissionError):
            with self.context(1):
                pass

    def test_two_own_already_active_reject_requester(self):
        self.processes.rows[333] = self.identity(333, self.own_source)
        self.policy["gpus"][0]["processes"].append(self.declaration(333, "own_e18"))
        self.state["compute_pids"] = [111, 333]
        with self.assertRaises(PermissionError):
            with self.context():
                pass

    def test_three_existing_compute_reject_fourth(self):
        self.processes.rows[333] = self.identity(333, Path("/foreign/second.py"))
        self.policy["gpus"][0]["processes"].append(self.declaration(333, "foreign"))
        self.state["compute_pids"].append(333)
        with self.assertRaises(PermissionError):
            with self.context():
                pass

    def test_unknown_gpu_pid_rejected(self):
        self.state["compute_pids"].append(333)
        with self.assertRaises(PermissionError):
            with self.context():
                pass

    def test_pid_reuse_rejected(self):
        self.processes.rows[111]["start_ticks"] = "9999"
        with self.assertRaises(PermissionError):
            with self.context():
                pass

    def test_command_change_rejected(self):
        self.processes.rows[111]["cmdline_sha256"] = "a" * 64
        with self.assertRaises(PermissionError):
            with self.context():
                pass

    def test_own_source_bytes_change_rejected(self):
        self.own_source.write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            with self.context():
                pass

    def test_foreign_role_does_not_become_known_e18(self):
        self.state["compute_pids"] = [222]
        self.processes.rows.pop(111)
        with self.context() as receipt:
            self.assertEqual(receipt["foreign_compute_pids"], [222])
            self.assertEqual(receipt["known_e18_compute_pids"], [])
            self.assertEqual(receipt["admitted_own_with_requester"], 1)

    def test_own_source_outside_private_root_rejected(self):
        outside = self.root.parent / "outside.py"
        outside.write_text("outside", encoding="utf-8")
        self.processes.rows[111] = self.identity(111, outside)
        row = self.declaration(111, "own_e18")
        row["source_ref"] = self.reference(outside)
        self.policy["gpus"][0]["processes"][0] = row
        with self.assertRaises(PermissionError):
            with self.context():
                pass

    def test_own_actor_assigned_to_other_gpu_rejected(self):
        self.processes.rows[111] = self.identity(111, self.own_source, gpu="1")
        self.policy["gpus"][0]["processes"][0] = self.declaration(111, "own_e18")
        with self.assertRaises(PermissionError):
            with self.context():
                pass

    def test_uuid_mismatch_rejected(self):
        self.state["uuid"] = "OTHER"
        with self.assertRaises(PermissionError):
            with self.context():
                pass

    def test_max_total_cannot_expand(self):
        self.policy["max_total_compute"] = 4
        with self.assertRaises(PermissionError):
            with self.context():
                pass

    def test_max_own_cannot_expand(self):
        self.policy["max_own_e18"] = 3
        with self.assertRaises(PermissionError):
            with self.context():
                pass

    def test_unknown_role_rejected(self):
        self.policy["gpus"][0]["processes"][1]["role"] = "trusted"
        with self.assertRaises(ValueError):
            with self.context():
                pass

    def test_requester_source_mismatch_rejected(self):
        self.policy["requester_source_ref"] = self.reference(self.own_source)
        with self.assertRaises(PermissionError):
            with self.context():
                pass

    def test_live_own_without_cuda_still_counts_capacity(self):
        self.processes.rows[333] = self.identity(333, self.own_source)
        self.policy["gpus"][0]["processes"].append(self.declaration(333, "own_e18"))
        with self.assertRaises(PermissionError):
            with self.context():
                pass

    def test_other_original_slot_pending_unknown_owner_rejected(self):
        path = Path(self.manifest["gpu_lock_root"]) / "gpu-0.slot-1.lock"
        self.processes.locks[str(path.resolve())] = 333
        with self.assertRaises(PermissionError):
            with self.context():
                pass
        self.assertEqual(self.processes.locks, {str(path.resolve()): 333})

    def test_current_pid_compute_is_counted_once(self):
        self.state["compute_pids"].append(os.getpid())
        with self.context() as receipt:
            self.assertEqual(receipt["admitted_total_with_requester"], 3)

    def test_context_body_exception_clears_original_global_and_lock(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with self.context():
                raise RuntimeError("synthetic")
        self.assertIsNone(self.worker._ACTIVE_GPU_SLOT_CONTEXT)
        self.assertEqual(self.processes.locks, {})

    def test_admission_failure_clears_original_global_and_lock(self):
        self.state["compute_pids"].append(444)
        with self.assertRaises(PermissionError):
            with self.context():
                pass
        self.assertIsNone(self.worker._ACTIVE_GPU_SLOT_CONTEXT)
        self.assertEqual(self.processes.locks, {})

    def test_existing_active_context_rejected_without_overwrite(self):
        original = {"existing": True}
        self.worker._ACTIVE_GPU_SLOT_CONTEXT = original
        with self.assertRaises(RuntimeError):
            with self.context():
                pass
        self.assertIs(self.worker._ACTIVE_GPU_SLOT_CONTEXT, original)

    def test_production_rejects_mock_probe_or_identity(self):
        with self.assertRaises(PermissionError):
            admission.CooperativeSlotAdmission(
                self.worker, self.manifest, "a" * 64, "0", 0, self.artifact(self.policy), processes=self.processes,
            )

    def test_duplicate_whitelisted_pid_rejected(self):
        self.policy["gpus"][0]["processes"].append(copy.deepcopy(self.policy["gpus"][0]["processes"][0]))
        with self.assertRaises(ValueError):
            with self.context():
                pass

    def test_original_paid_guard_and_persistent_context_use_same_module(self):
        path = Path(__file__).with_name("e18_member4_worker_v3.py")
        self.assertEqual(admission.file_sha(path), admission.WORKER_SHA)
        spec = importlib.util.spec_from_file_location("_cooperative_original_worker_qualification", path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        original.GPUOwnership = self.worker.GPUOwnership
        profile = SimpleNamespace(fingerprint="same-original-profile")
        loaded = SimpleNamespace(spec=profile)
        created = []
        inference = SimpleNamespace(configuration_fingerprint=lambda value: value.fingerprint)
        adapter = SimpleNamespace(apis=SimpleNamespace(root=self.root, inference=inference), scope=self.scope, member_specs={4: profile})
        module = SimpleNamespace(canonical_digest=admission.digest)

        def factory(value):
            created.append(value)
            return loaded

        with self.context(worker=original) as receipt:
            context = original.owned_context(module, adapter, self.manifest, factory, False)
            self.assertIs(context.get(profile), loaded)
            self.assertIs(original.owned_context(module, adapter, self.manifest, factory, False), context)
            self.assertIs(context.get(profile), loaded)
            self.assertEqual(len(created), 1)
            self.assertIs(original._ACTIVE_GPU_SLOT_CONTEXT, receipt)
            self.assertEqual(admission.file_sha(Path(receipt["resource_policy_ref"]["path"])), receipt["policy_sha256"])
        with self.assertRaises(PermissionError):
            context.get(profile)
        self.assertEqual(admission.file_sha(path), admission.WORKER_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification outputs are append-only")
    started = time.perf_counter()
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CooperativeAdmissionQualification))
    record = dict(
        schema="e18_cooperative_slot_admission_qualification_v1", status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun, errors=len(result.errors), failures=len(result.failures), wall_seconds=time.perf_counter() - started,
        admission_source_sha256=admission.file_sha(admission.__file__), qualification_source_sha256=admission.file_sha(__file__),
        original_worker_sha256=admission.file_sha(Path(__file__).with_name("e18_member4_worker_v3.py")),
        actual_gpu_queries=0, actual_process_signals=0, actual_model_calls=0, real_gold_reads=0,
        production_admissions=0, test_output=capture.getvalue(),
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: record[key] for key in ("status", "tests", "errors", "failures", "wall_seconds")}))
    if not result.wasSuccessful():
        print(capture.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
