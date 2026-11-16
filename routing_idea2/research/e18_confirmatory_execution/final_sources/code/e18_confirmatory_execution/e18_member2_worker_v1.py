"""E18 member2-only persistent profile worker with explicit GPU slot policy.

Only an explicit hash-bound production manifest and approval can reach a real
backend. Loading this module alone performs no model, input or network access.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


ADAPTER_SHA = "21e76f6a38fa5f2ed590a2b2068bda8255bc4ebc290dea9eb7c32a3a44a65bc4"
MANIFEST_SCHEMA = "e18_member2_worker_manifest_v1"
MINIMUM_EXTRA_SLOT_BYTES = 12 * 1024**3
_PROCESS_BACKEND_CONTEXT = None
_ACTIVE_GPU_SLOT_CONTEXT = None


def read_gpu_state(gpu):
    memory = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + str(gpu),
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    rows = list(csv.reader(line for line in memory.stdout.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("One physical GPU telemetry row required")
    index, uuid, free_mib = [value.strip() for value in rows[0]]
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    active = []
    for row in csv.reader(
        line for line in processes.stdout.splitlines() if line.strip()
    ):
        if len(row) != 2:
            raise ValueError("Compute PID telemetry could not be parsed")
        if row[0].strip() == uuid:
            active.append(int(row[1].strip()))
    return dict(
        index=int(index),
        uuid=uuid,
        free_memory_bytes=int(free_mib) * 1024**2,
        compute_pids=sorted(set(active)),
        observed_at_unix=time.time(),
    )


def validate_slot_resources(
    manifest,
    gpu,
    gpu_slot,
    *,
    resource_policy_ref=None,
    software_test_only=False,
    probe=None,
):
    if type(gpu_slot) is not int or gpu_slot not in (0, 1):
        raise ValueError("Only explicit GPU slots 0 and 1 are allowed")
    if not software_test_only and probe is not None:
        raise PermissionError("Production requires actual nvidia-smi telemetry")
    reference = resource_policy_ref or manifest.get("resource_policy_ref")
    policy = read_json(checked_ref(reference))
    if (
        policy.get("schema") != "e18_member2_gpu_slot_policy_v1"
        or policy.get("cohort") != "e18_confirm"
        or policy.get("software_test_only") is not software_test_only
        or policy.get("max_slots") != 2
    ):
        raise PermissionError("Explicit two-slot E18 resource policy required")
    module = load_adapter(manifest["adapter_ref"])
    scope = read_json(checked_ref(manifest["scope_ref"]))
    if (
        policy.get("scope_sha256") != module.canonical_digest(scope)
        or policy.get("approval_sha256") != manifest["approval_ref"]["sha256"]
        or Path(policy["gpu_lock_root"]).resolve()
        != Path(manifest["gpu_lock_root"]).resolve()
    ):
        raise PermissionError("Resource policy scope, approval or lock root differs")
    minimum = policy.get("minimum_free_memory_bytes")
    if type(minimum) is not int or minimum < MINIMUM_EXTRA_SLOT_BYTES:
        raise ValueError("Extra slot policy must require at least 12 GiB free VRAM")
    entries = [row for row in policy.get("gpus", []) if row.get("index") == int(gpu)]
    if len(entries) != 1:
        raise PermissionError("The physical GPU is not uniquely authorized")
    entry = entries[0]
    known = entry.get("known_e18_pids")
    if not isinstance(known, list) or any(
        type(pid) is not int or pid <= 0 for pid in known
    ):
        raise ValueError("Explicit positive known E18 process identifiers required")
    state = (probe or read_gpu_state)(gpu)
    if state["index"] != int(gpu) or state["uuid"] != entry.get("uuid"):
        raise PermissionError("Actual physical GPU identity differs")
    active = set(state["compute_pids"]) - {os.getpid()}
    if not active <= set(known):
        raise PermissionError("An unknown GPU compute process is active")
    if len(active) >= policy["max_slots"]:
        raise PermissionError("The two-process GPU slot limit is already reached")
    if gpu_slot == 1 and state["free_memory_bytes"] < minimum:
        raise PermissionError("The additional slot has less than approved free VRAM")
    return dict(
        schema="e18_member2_gpu_slot_admission_v1",
        status="PASS",
        gpu_slot=gpu_slot,
        policy_sha256=reference["sha256"],
        telemetry=state,
        known_e18_compute_pids=sorted(active),
        checked_at_unix=time.time(),
    )


def process_alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return (
                bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
                and code.value == 259
            )
        finally:
            kernel.CloseHandle(handle)
    status = Path("/proc") / str(pid) / "stat"
    if os.name != "nt" and status.is_file():
        text = status.read_text(encoding="utf-8")
        if text[text.rfind(")") + 2 :].split()[0] == "Z":
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    return True


def locks_owned_by(pid, paths):
    table = Path("/proc/locks")
    if not table.is_file():
        return os.name == "nt"
    owned = set()
    for line in table.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[1] == "FLOCK" and fields[4] == str(pid):
            device = fields[5].split(":")
            if len(device) == 3:
                owned.add((int(device[0], 16), int(device[1], 16), int(device[2])))
    for path in paths:
        try:
            status = Path(path).stat()
        except FileNotFoundError:
            return False
        if (
            os.major(status.st_dev),
            os.minor(status.st_dev),
            status.st_ino,
        ) not in owned:
            return False
    return True


def assert_lease_active(lease, *, software_test_only=False):
    if not process_alive(lease["owner_pid"]):
        raise PermissionError("Checkpoint broker is no longer active")
    if any(process_alive(pid) for pid in lease["former_pids"]):
        raise PermissionError("A former member2 worker is still active")
    if not software_test_only and not locks_owned_by(
        lease["owner_pid"], [row["claim_path"] for row in lease["parent_shards"]]
    ):
        raise PermissionError("Checkpoint broker no longer owns both parent claims")


def validate_checkpoint_lease(
    module, adapter, manifest, keys, *, software_test_only=False
):
    lease = read_json(checked_ref(manifest["checkpoint_lease_ref"]))
    if (
        lease.get("schema") != "e18_member2_checkpoint_lease_v1"
        or lease.get("status") != "LEASED"
        or lease.get("target_member_index") != 2
        or lease.get("execution_scope_sha256") != module.canonical_digest(adapter.scope)
        or lease.get("approval_sha256") != manifest["approval_ref"]["sha256"]
        or lease.get("checkpoint_stop_verified") is not True
        or lease.get("pending_or_unknown_attempts") != 0
    ):
        raise PermissionError(
            "A stopped and reconciled member2 checkpoint lease is required"
        )
    if not module.valid_sha(lease.get("base_member_plan_sha256")):
        raise ValueError("Original member plan identity required")
    parents = lease.get("parent_shards", [])
    former_pids = lease.get("former_pids")
    if len(parents) != 2 or any(
        not module.valid_sha(row.get("manifest_sha256"))
        or not module.valid_sha(row.get("task_keys_sha256"))
        or not Path(row.get("claim_path", "")).is_absolute()
        for row in parents
    ):
        raise ValueError("Both original member2 parent shard identities required")
    if (
        len({row["claim_path"] for row in parents}) != 2
        or len({row["manifest_sha256"] for row in parents}) != 2
        or type(lease.get("owner_pid")) is not int
        or lease["owner_pid"] <= 0
        or not isinstance(former_pids, list)
        or len(former_pids) != 2
        or any(type(pid) is not int or pid <= 0 for pid in former_pids)
        or len(set(former_pids)) != 2
        or lease["owner_pid"] in former_pids
    ):
        raise ValueError(
            "Distinct parent claims and explicit broker/former process identities required"
        )
    remaining = read_json(checked_ref(lease["remaining_task_keys_ref"]))
    remaining_keys = set(map(tuple, remaining))
    original_count = sum(key[2] == 2 for key in adapter.callable_members)
    if not software_test_only and (
        original_count != 8005
        or lease["base_member_plan_sha256"]
        != "8473a296a638c677da1791867c87c09af441cfb5e90ae5cd9a84feb7e9564502"
    ):
        raise ValueError("The original8005 Granite task union and member plan are required")
    if (
        len(remaining_keys) != len(remaining)
        or len(remaining) != lease.get("remaining_count")
        or lease.get("original_member2_tasks") != original_count
        or type(lease.get("checkpoint_successes")) is not int
        or lease["checkpoint_successes"] < 0
        or lease["checkpoint_successes"] + len(remaining) != original_count
        or any(len(key) != 3 or key[2] != 2 for key in remaining_keys)
        or not remaining_keys <= adapter.callable_members
        or not set(keys) <= remaining_keys
        or not 1 <= len(keys) <= 512
    ):
        raise ValueError("Worker keys exceed the exact remaining member2 checkpoint")
    assert_lease_active(lease, software_test_only=software_test_only)
    return lease


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked_ref(reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("An exact path/SHA reference is required")
    path = Path(reference["path"]).resolve(strict=True)
    if not path.is_file() or file_sha(path) != reference["sha256"]:
        raise ValueError("Referenced bytes differ " + str(path))
    return path


def load_adapter(reference):
    path = checked_ref(reference)
    if reference["sha256"] != ADAPTER_SHA:
        raise ValueError("The E18 frozen runtime adapter identity differs")
    name = "e18_execution_adapter"
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise ValueError("Another E18 adapter is already imported")
        return existing
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def append_event(path, event):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class GPUOwnership(AbstractContextManager):
    """Process-lifetime advisory lock shared by all E18 workers on the host."""

    def __init__(self, lock_root, gpu, manifest_sha, gpu_slot=0):
        if not str(gpu).isdigit():
            raise ValueError("An explicit numeric physical GPU index is required")
        self.root = Path(lock_root).resolve()
        self.gpu = str(gpu)
        self.manifest_sha = manifest_sha
        if type(gpu_slot) is not int or gpu_slot not in (0, 1):
            raise ValueError("Only GPU slot 0 or 1 may be owned")
        self.gpu_slot = gpu_slot
        self.file_key = "gpu-" + self.gpu + ("" if gpu_slot == 0 else ".slot-1")
        self.stream = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.stream = (self.root / (self.file_key + ".lock")).open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.stream.seek(0)
                if not self.stream.read(1):
                    self.stream.write(b"0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            self.stream.close()
            raise RuntimeError(
                "GPU is owned by another cooperating E18 worker"
            ) from None
        save_json(
            self.root / (self.file_key + ".owner.json"),
            dict(
                pid=os.getpid(),
                host=socket.gethostname(),
                gpu=self.gpu,
                manifest_sha256=self.manifest_sha,
                gpu_slot=self.gpu_slot,
                acquired_at_unix=time.time(),
            ),
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream is not None:
            if os.name == "nt":
                import msvcrt

                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
        return False


class SlotAdmission(AbstractContextManager):
    def __init__(
        self,
        manifest,
        manifest_sha,
        gpu,
        gpu_slot,
        resource_policy_ref,
        *,
        software_test_only=False,
        probe=None,
    ):
        self.manifest = manifest
        self.gpu = str(gpu)
        self.gpu_slot = gpu_slot
        self.resource_policy_ref = resource_policy_ref
        self.software_test_only = software_test_only
        self.probe = probe
        self.owner = GPUOwnership(
            manifest["gpu_lock_root"], gpu, manifest_sha, gpu_slot
        )

    def __enter__(self):
        global _ACTIVE_GPU_SLOT_CONTEXT
        if _ACTIVE_GPU_SLOT_CONTEXT is not None:
            raise RuntimeError("This process already owns a GPU slot context")
        self.owner.__enter__()
        try:
            receipt = validate_slot_resources(
                self.manifest,
                self.gpu,
                self.gpu_slot,
                resource_policy_ref=self.resource_policy_ref,
                software_test_only=self.software_test_only,
                probe=self.probe,
            )
            receipt["resource_policy_ref"] = self.resource_policy_ref
            receipt["physical_gpu_index"] = int(self.gpu)
            _ACTIVE_GPU_SLOT_CONTEXT = receipt
            save_json(
                self.owner.root / (self.owner.file_key + ".admission.json"), receipt
            )
        except BaseException:
            self.owner.__exit__(None, None, None)
            raise
        return receipt

    def __exit__(self, exc_type, exc_value, traceback):
        global _ACTIVE_GPU_SLOT_CONTEXT
        _ACTIVE_GPU_SLOT_CONTEXT = None
        return self.owner.__exit__(exc_type, exc_value, traceback)


class OwnedBackendContext:
    def __init__(self, key, module, create, expected_fingerprint, software_test_only):
        self.key = key
        self.module = module
        self.create = create
        self.expected_fingerprint = expected_fingerprint
        self.software_test_only = software_test_only
        self.backend = None
        self.fingerprint = None
        self.constructions = 0

    def get(self, spec):
        if not self.software_test_only and _ACTIVE_GPU_SLOT_CONTEXT is None:
            raise PermissionError(
                "Production backend requires a live GPU slot admission"
            )
        fingerprint = self.module.configuration_fingerprint(spec)
        if fingerprint != self.expected_fingerprint:
            raise ValueError("Requested profile differs from owned member2 identity")
        if self.backend is None:
            loaded = self.create(spec)
            if loaded.spec != spec:
                raise ValueError("Original backend changed its logical profile")
            self.backend = loaded
            self.fingerprint = fingerprint
            self.constructions += 1
        elif fingerprint != self.fingerprint:
            raise ValueError("A process backend context cannot switch logical profile")
        return self.backend


class PersistentBackend:
    def __init__(self, context):
        self.context = context
        self.loads = 0
        self.reused_previous_shard = context.backend is not None

    def __call__(self, spec):
        before = self.context.constructions
        backend = self.context.get(spec)
        self.loads += self.context.constructions - before
        return backend


def owned_context(module, adapter, manifest, factory, software_test_only):
    global _PROCESS_BACKEND_CONTEXT
    if _ACTIVE_GPU_SLOT_CONTEXT is not None:
        checked_ref(_ACTIVE_GPU_SLOT_CONTEXT["resource_policy_ref"])
    identity = dict(
        asset_root=str(adapter.apis.root),
        adapter_ref=manifest["adapter_ref"],
        scope_sha256=module.canonical_digest(adapter.scope),
        scope_ref=manifest["scope_ref"],
        approval_ref=manifest["approval_ref"],
        inputs_ref=manifest["question_records_ref"],
        owned_output_root=str(Path(manifest["owned_output_root"]).resolve()),
        logical_profile=adapter.apis.inference.configuration_fingerprint(
            adapter.member_specs[2]
        ),
        backend_loading=manifest["backend"],
        software_test_only=software_test_only,
        runtime_slot_policy=_ACTIVE_GPU_SLOT_CONTEXT,
    )
    key = module.canonical_digest(identity)
    if _PROCESS_BACKEND_CONTEXT is None:
        _PROCESS_BACKEND_CONTEXT = OwnedBackendContext(
            key,
            adapter.apis.inference,
            factory,
            identity["logical_profile"],
            software_test_only,
        )
    elif _PROCESS_BACKEND_CONTEXT.key != key:
        raise ValueError(
            "Different member2 scope, profile, mapping or slot admission requires a separate process"
        )
    return _PROCESS_BACKEND_CONTEXT


def load_worker(manifest, *, software_test_only=False, injected_factory=None):
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Unexpected worker manifest schema")
    if (
        manifest.get("stage") != "member_generation"
        or manifest.get("member_index") != 2
    ):
        raise ValueError("This worker admits frozen member profile 2 only")
    if manifest.get("software_test_only") is not software_test_only:
        raise PermissionError("Worker and manifest execution modes differ")
    if not software_test_only and injected_factory is not None:
        raise PermissionError("Production accepts original backend factories only")
    adapter_module = load_adapter(manifest["adapter_ref"])
    scope = read_json(checked_ref(manifest["scope_ref"]))
    approval = adapter_module.validate_scope_approval(
        manifest["approval_ref"],
        scope,
        manifest["stage"],
        software_test_only=software_test_only,
    )
    output_root = Path(manifest["owned_output_root"]).resolve()
    if output_root != Path(approval["approved_output_root"]).resolve():
        raise PermissionError("Worker output root differs from explicit approval")
    records_path = checked_ref(manifest["question_records_ref"])
    with records_path.open(encoding="utf-8-sig", newline="") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    apis = adapter_module.load_original_apis(manifest["asset_root"])
    pool = adapter_module.read_json(apis.root / "configs/pool_v1.json")["models"]
    members = {
        row["index"]: apis.inference.ModelSpec(
            **adapter_module.read_json(apis.root / row["config"])
        )
        for row in pool
    }
    aggregator = apis.inference.ModelSpec(
        **adapter_module.read_json(
            apis.root
            / "configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json"
        )
    )
    adapter = adapter_module.E18ExecutionAdapter(
        apis, scope, records, members, aggregator
    )
    stage = manifest["stage"]
    if stage not in {"member_generation", "aggregation"}:
        raise ValueError("Worker must own one native stage")
    raw_keys = read_json(checked_ref(manifest["task_keys_ref"]))
    keys = [tuple(key) for key in raw_keys]
    if len(keys) != len(set(keys)):
        raise ValueError("Worker task keys must be unique")
    required = (
        adapter.required_members
        if stage == "member_generation"
        else adapter.required_aggregates
    )
    if not set(keys) <= required:
        raise ValueError("Worker task exceeds the frozen sparse action union")
    if stage == "member_generation":
        profile = manifest["member_index"]
        if (
            type(profile) is not int
            or profile not in range(6)
            or any(key[2] != profile for key in keys)
        ):
            raise ValueError(
                "Member worker must contain exactly its frozen model profile"
            )
    else:
        profile = "aggregator"
    if injected_factory is not None:
        factory = injected_factory
    elif manifest["backend"]["mode"] == "original":
        factory = adapter_module.original_backend_factory(apis.root)
    elif manifest["backend"]["mode"] == "mapped":
        config = manifest["backend"]
        factory = adapter_module.mapped_backend_factory(
            apis.root,
            config["physical_root"],
            config["mapping_ref"],
            config["helper_ref"],
        )
    else:
        raise ValueError("Unknown original backend loading mode")
    lease = validate_checkpoint_lease(
        adapter_module, adapter, manifest, keys, software_test_only=software_test_only
    )
    adapter._member2_checkpoint_lease = lease
    dependencies = {}
    if manifest.get("member_dependencies_ref") is not None:
        for row in read_json(checked_ref(manifest["member_dependencies_ref"])):
            key = tuple(row["task_key"])
            if key in dependencies or key not in adapter.required_members:
                raise ValueError("Dependency index has duplicate or unapproved keys")
            dependencies[key] = row["record_ref"]
    context = owned_context(
        adapter_module, adapter, manifest, factory, software_test_only
    )
    persistent = PersistentBackend(context)
    return adapter_module, adapter, keys, profile, persistent, dependencies


def member_dependency(module, adapter, output_root, external, key):
    if key in external:
        record = read_json(checked_ref(external[key]))
    else:
        owner = (
            output_root
            / module.canonical_digest(adapter.scope)
            / "member_generation"
            / module.canonical_digest(list(key))
        )
        receipt_path = owner / "LAST_TASK_RECEIPT.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(
                "Required member dependency has not committed " + str(key)
            )
        receipt = read_json(receipt_path)
        if (
            receipt.get("status") != "SUCCESS"
            or receipt.get("scope_sha256") != module.canonical_digest(adapter.scope)
            or receipt.get("software_test_only") is not adapter._worker_software_mode
        ):
            raise ValueError("Member dependency receipt scope, status or mode differs")
        record = receipt["native_record"]
    return adapter.checked_member_dependency(*key, record)


def native_usage(row):
    result = row.get("result") or row.get("actual_usage") or {}
    usage = {key: result.get(key) for key in ("prompt_tokens", "completion_tokens")}
    if any(type(value) is not int or value < 0 for value in usage.values()):
        return None
    return usage


def execute_manifest(
    manifest,
    manifest_sha,
    *,
    software_test_only=False,
    injected_factory=None,
    injected_tokenizer=None,
):
    module, adapter, keys, profile, backend, external = load_worker(
        manifest,
        software_test_only=software_test_only,
        injected_factory=injected_factory,
    )
    adapter._worker_software_mode = software_test_only
    output_root = Path(manifest["owned_output_root"]).resolve()
    journal_root = output_root / "_dispatcher" / manifest_sha
    journal_root.mkdir(parents=True, exist_ok=True)
    journal = journal_root / "EVENTS.jsonl"
    stage = manifest["stage"]
    stop = [False]
    handlers = {}
    if manifest.get("install_signal_handlers", True):
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(
                signum, lambda *_: stop.__setitem__(0, True)
            )
    completed = reused = 0
    failures = []
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    tokenizer = injected_tokenizer
    try:
        for qid, seed, index in keys:
            if stop[0]:
                break
            assert_lease_active(
                adapter._member2_checkpoint_lease, software_test_only=software_test_only
            )
            task = dict(stage=stage, question_id=qid, generation_seed=seed)
            task[
                "member_index" if stage == "member_generation" else "combination_index"
            ] = index
            records = None
            if stage == "aggregation":
                try:
                    records = {
                        member: member_dependency(
                            module, adapter, output_root, external, (qid, seed, member)
                        )
                        for member in module.TEAMS[index]
                    }
                except FileNotFoundError:
                    append_event(
                        journal,
                        dict(
                            event="DEPENDENCY_NOT_READY", task=task, at_unix=time.time()
                        ),
                    )
                    continue
                if tokenizer is None:
                    tokenizer = backend(adapter.aggregator_spec).tokenizer
            append_event(
                journal, dict(event="TASK_ENTER", task=task, at_unix=time.time())
            )
            try:
                receipt = module.run_approved_task(
                    adapter,
                    task,
                    manifest["approval_ref"],
                    output_root,
                    backend,
                    tokenizer=tokenizer,
                    member_records=records,
                    software_test_only=software_test_only,
                )
            except BaseException as error:
                append_event(
                    journal,
                    dict(
                        event="STOP_REQUIRES_RECONCILIATION",
                        task=task,
                        error_type=type(error).__name__,
                        error=str(error),
                        at_unix=time.time(),
                    ),
                )
                failures.append(dict(task=task, error_type=type(error).__name__))
                raise
            append_event(
                journal,
                dict(
                    event="TASK_RESULT",
                    task=task,
                    status=receipt["status"],
                    physical_attempt_id=receipt["physical_attempt_id"],
                    native_record_sha256=receipt["native_record_sha256"],
                    from_cache=receipt["from_cache"],
                    actual_usage=native_usage(receipt["native_record"]),
                    at_unix=time.time(),
                ),
            )
            if receipt["status"] != "SUCCESS":
                failures.append(
                    dict(task=task, physical_attempt_id=receipt["physical_attempt_id"])
                )
                break
            completed += 1
            reused += int(receipt["from_cache"])
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        summary = dict(
            schema="e18_gpu_worker_summary_v1",
            manifest_sha256=manifest_sha,
            stage=stage,
            profile=profile,
            planned=len(keys),
            completed=completed,
            cached=reused,
            new_successes=completed - reused,
            backend_loads=backend.loads,
            cross_shard_backend_reused=backend.reused_previous_shard,
            process_backend_constructions=backend.context.constructions,
            runtime_gpu_slot_admission=_ACTIVE_GPU_SLOT_CONTEXT,
            failures=failures,
            stop_requested=stop[0],
            cpu_seconds=time.process_time() - cpu_start,
            wall_seconds=time.perf_counter() - wall_start,
            software_test_only=software_test_only,
            actual_gpu_runtime_status="NOT_RUNTIME_TESTED"
            if software_test_only
            else "ORIGINAL_BACKEND_CALLED",
        )
        save_json(journal_root / "SUMMARY.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--gpu-slot", type=int, choices=(0, 1), default=0)
    parser.add_argument("--resource-policy", type=Path, required=True)
    parser.add_argument("--resource-policy-sha256", required=True)
    arguments = parser.parse_args()
    if file_sha(arguments.manifest) != arguments.manifest_sha256:
        raise ValueError("Worker manifest bytes differ from the admitted plan")
    manifest = read_json(arguments.manifest)
    if manifest.get("software_test_only") is not False:
        raise PermissionError(
            "The native CLI accepts explicit production manifests only"
        )
    if not arguments.gpu.isdigit():
        raise ValueError("Explicit physical GPU index required")
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
    policy_ref = dict(
        path=str(arguments.resource_policy), sha256=arguments.resource_policy_sha256
    )
    with SlotAdmission(
        manifest,
        arguments.manifest_sha256,
        arguments.gpu,
        arguments.gpu_slot,
        policy_ref,
    ):
        summary = execute_manifest(manifest, arguments.manifest_sha256)
    print(json.dumps(summary, ensure_ascii=False))
    if summary["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
