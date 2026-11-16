"""Operational GPU admission around the caller's original member4 module.

Admission binds live process births, raw argv hashes and declared roles. The
original GPUOwnership and paid-path global are reused through the supplied
module object. No worker, model, prompt, task or sampling source is modified.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


WORKER_SHA = "5ac9a17dd07d282bfa2cceaff3dd062ce837e68885b4086816ef509974af5786"
MINIMUM_FREE_BYTES = 12 * 1024**3


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_ref(reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("An exact source path/SHA reference is required")
    path = Path(reference["path"]).resolve(strict=True)
    if not path.is_file() or file_sha(path) != reference["sha256"]:
        raise ValueError("Referenced operational source bytes differ")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class LinuxProcessIdentities:
    """Read process identities and actual flock owners without sending signals."""

    def __init__(self):
        if sys.platform != "linux":
            raise RuntimeError("Production process admission requires Linux")

    def identity(self, pid):
        root = Path("/proc") / str(pid)
        try:
            before = (root / "stat").read_text()
            raw = (root / "cmdline").read_bytes()
            after = (root / "stat").read_text()
        except FileNotFoundError:
            return None
        first = before[before.rfind(")") + 2:].split()
        second = after[after.rfind(")") + 2:].split()
        if first[0] == "Z" or second[0] == "Z":
            return None
        if first[19] != second[19]:
            raise PermissionError("Process birth changed during admission")
        return dict(
            pid=pid, start_ticks=first[19],
            cmdline_sha256=hashlib.sha256(raw).hexdigest(),
            command=[item.decode() for item in raw.split(b"\0") if item],
        )

    def lock_owner(self, path):
        path = Path(path)
        if not path.exists():
            return None
        stat = path.stat()
        target = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
        owners = set()
        for line in Path("/proc/locks").read_text().splitlines():
            fields = line.split()
            if "->" in fields or "FLOCK" not in fields:
                continue
            index = fields.index("FLOCK")
            device = fields[index + 4].split(":")
            observed = (int(device[0], 16), int(device[1], 16), int(device[2]))
            if observed == target:
                owners.add(int(fields[index + 3]))
        if len(owners) > 1:
            raise RuntimeError("Physical slot ownership is ambiguous")
        return next(iter(owners)) if owners else None


def source_matches(identity, reference, server_root, gpu):
    source = checked_ref(reference)
    if not source.is_relative_to(server_root):
        raise PermissionError("An own E18 source escaped its private research root")
    command = identity["command"]
    scripts = [(index, Path(item).resolve()) for index, item in enumerate(command[1:], 1) if item.endswith(".py")]
    if not scripts or scripts[0][1] != source or command.count("--gpu") != 1:
        raise PermissionError("Own E18 process script/GPU identity differs")
    if any(flag in command[1:scripts[0][0]] for flag in ("-c", "-m")):
        raise PermissionError("The declared own script is not the Python entry point")
    index = command.index("--gpu") + 1
    if index == len(command) or command[index] != str(gpu):
        raise PermissionError("Own process is assigned to another physical GPU")


class CooperativeSlotAdmission(AbstractContextManager):
    def __init__(
        self, worker, manifest, manifest_sha, gpu, gpu_slot, resource_policy_ref,
        *, software_test_only=False, probe=None, processes=None,
    ):
        if not software_test_only and (probe is not None or processes is not None):
            raise PermissionError("Production requires real original telemetry and process identities")
        if not software_test_only and file_sha(worker.__file__) != WORKER_SHA:
            raise PermissionError("The original member4 worker bytes differ")
        if not str(gpu).isdigit() or type(gpu_slot) is not int or gpu_slot not in (0, 1):
            raise ValueError("An explicit physical GPU and original slot0 or1 are required")
        self.worker, self.manifest = worker, manifest
        self.manifest_sha, self.gpu, self.slot = manifest_sha, str(gpu), gpu_slot
        self.reference = dict(resource_policy_ref)
        self.mode, self.probe = software_test_only, probe
        self.processes = processes if processes is not None else LinuxProcessIdentities()
        self.owner = worker.GPUOwnership(manifest["gpu_lock_root"], self.gpu, manifest_sha, gpu_slot)
        self.receipt = None

    def admit(self):
        manifest = self.manifest
        policy = read_json(checked_ref(self.reference))
        if (
            policy.get("schema") != "e18_cooperative_gpu_slot_policy_v1"
            or policy.get("cohort") != "e18_confirm"
            or policy.get("software_test_only") is not self.mode
            or policy.get("max_total_compute") != 3
            or policy.get("max_own_e18") != 2
        ):
            raise PermissionError("The explicit cooperative three-total/two-own policy is required")
        if manifest.get("schema") != "e18_member4_worker_manifest_v3" or manifest.get("stage") != "member_generation" or manifest.get("member_index") != 4:
            raise PermissionError("This admission wraps the original member4 worker only")
        scope = read_json(checked_ref(manifest["scope_ref"]))
        if (
            policy.get("scope_sha256") != digest(scope)
            or policy.get("approval_sha256") != manifest["approval_ref"]["sha256"]
            or Path(policy["gpu_lock_root"]).resolve() != Path(manifest["gpu_lock_root"]).resolve()
        ):
            raise PermissionError("Resource scope, approval or original lock root differs")
        minimum = policy.get("minimum_free_memory_bytes")
        if type(minimum) is not int or minimum < MINIMUM_FREE_BYTES:
            raise ValueError("Every slot must require at least12GiB free memory")
        server = Path(manifest["owned_output_root"]).resolve().parent
        requester = self.processes.identity(os.getpid())
        if requester is None:
            raise PermissionError("The requesting process identity is unavailable")
        source_matches(requester, policy["requester_source_ref"], server, self.gpu)
        entries = [entry for entry in policy.get("gpus", []) if entry.get("index") == int(self.gpu)]
        if len(entries) != 1:
            raise PermissionError("The target physical GPU is not uniquely admitted")
        entry = entries[0]
        declarations = entry.get("processes")
        if not isinstance(declarations, list):
            raise ValueError("An explicit process identity and role whitelist is required")
        by_pid = {}
        for row in declarations:
            pid = row.get("pid")
            if (
                type(pid) is not int or pid <= 0 or pid in by_pid
                or not isinstance(row.get("start_ticks"), str) or not row["start_ticks"].isdigit()
                or not isinstance(row.get("cmdline_sha256"), str)
                or re.fullmatch("[0-9a-f]{64}", row["cmdline_sha256"]) is None
                or row.get("role") not in {"own_e18", "foreign"}
            ):
                raise ValueError("Process whitelist identity or role is malformed")
            by_pid[pid] = row
        state = (self.probe or self.worker.read_gpu_state)(self.gpu)
        if state.get("index") != int(self.gpu) or state.get("uuid") != entry.get("uuid"):
            raise PermissionError("Actual physical GPU index/UUID differs")
        if type(state.get("free_memory_bytes")) is not int or state["free_memory_bytes"] < minimum:
            raise PermissionError("The selected original slot has insufficient free memory")
        raw_active = state.get("compute_pids")
        if not isinstance(raw_active, list) or any(type(pid) is not int or pid <= 0 for pid in raw_active):
            raise ValueError("Actual GPU process telemetry is malformed")
        active = set(raw_active) - {os.getpid()}
        lock_root = Path(manifest["gpu_lock_root"]).resolve()
        other_name = "gpu-" + self.gpu + (".slot-1" if self.slot == 0 else "") + ".lock"
        other_owner = self.processes.lock_owner(lock_root / other_name)
        reserved = set() if other_owner is None or other_owner == os.getpid() else {other_owner}
        required = active | reserved
        if not required <= set(by_pid):
            raise PermissionError("An active or reserved process is absent from the identity whitelist")
        observations = []
        own = set()
        for pid, row in by_pid.items():
            if pid == os.getpid():
                raise ValueError("The requesting process is admitted through its bound runner source")
            observed = self.processes.identity(pid)
            if observed is None:
                if pid in required:
                    raise PermissionError("An active GPU/slot process exited during observation")
                continue
            if observed.get("start_ticks") != row["start_ticks"] or observed.get("cmdline_sha256") != row["cmdline_sha256"]:
                raise PermissionError("A whitelisted PID was reused or its command changed")
            if row["role"] == "own_e18":
                source_matches(observed, row["source_ref"], server, self.gpu)
                own.add(pid)
            elif pid in reserved:
                raise PermissionError("The original member4 slot is held by a foreign process")
            observations.append(dict(
                pid=pid, start_ticks=observed["start_ticks"], cmdline_sha256=observed["cmdline_sha256"],
                role=row["role"], observed_compute=pid in active, original_slot_reserved=pid in reserved,
            ))
        # Count known own actors before CUDA construction too. Otherwise two
        # admissions could each miss the other's not-yet-loaded model.
        admitted = active | own | reserved | {os.getpid()}
        own_count = len(own) + 1
        if len(admitted) > 3 or own_count > 2:
            raise PermissionError("Cooperative total/own process capacity is exhausted")
        return dict(
            schema="e18_cooperative_gpu_slot_admission_v1", status="PASS",
            gpu_slot=self.slot, policy_sha256=self.reference["sha256"],
            resource_policy_ref=self.reference, physical_gpu_index=int(self.gpu),
            telemetry=state, requesting_process=dict(
                pid=os.getpid(), start_ticks=requester["start_ticks"], cmdline_sha256=requester["cmdline_sha256"], role="own_e18",
            ),
            process_observations=observations,
            admitted_total_with_requester=len(admitted), admitted_own_with_requester=own_count,
            known_e18_compute_pids=sorted(active & own),
            foreign_compute_pids=sorted(active - own),
            original_slot_reserved_pids=sorted(reserved),
            operational_actor_pids_with_requester=sorted(admitted),
            minimum_free_memory_bytes=minimum, checked_at_unix=time.time(),
        )

    def __enter__(self):
        if self.worker._ACTIVE_GPU_SLOT_CONTEXT is not None:
            raise RuntimeError("This exact original worker already has a slot admission")
        self.owner.__enter__()
        try:
            self.receipt = self.admit()
            self.worker._ACTIVE_GPU_SLOT_CONTEXT = self.receipt
            self.worker.save_json(self.owner.root / (self.owner.file_key + ".cooperative-admission.json"), self.receipt)
        except BaseException:
            self.worker._ACTIVE_GPU_SLOT_CONTEXT = None
            self.owner.__exit__(None, None, None)
            raise
        return self.receipt

    def __exit__(self, exc_type, exc_value, traceback):
        self.worker._ACTIVE_GPU_SLOT_CONTEXT = None
        return self.owner.__exit__(exc_type, exc_value, traceback)
