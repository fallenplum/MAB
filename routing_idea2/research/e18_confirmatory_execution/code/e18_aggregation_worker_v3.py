"""E18 aggregation-only worker with verified physical tokenizer path loading.

Only an explicit hash-bound production manifest and approval can reach a real
backend. Loading this module alone performs no model, input or network access.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import AbstractContextManager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time


ADAPTER_SHA = "21e76f6a38fa5f2ed590a2b2068bda8255bc4ebc290dea9eb7c32a3a44a65bc4"
MANIFEST_SCHEMA = "e18_aggregation_worker_manifest_v3"
_PROCESS_BACKEND_CONTEXT = None


def restore_logical_tokenizer_path(
    tokenizer,
    logical_spec,
    source_receipt,
    frozen_tokenizer_files,
    execution_api,
    *,
    mapping_provenance=None,
):
    """Keep the same verified tokenizer instance and restore its logical name.

    The four frozen tokenizer byte identities are checked before aliasing. No
    tokenizer method, vocabulary, revision, template or compression code changes.
    This can also qualify a real tokenizer without loading any language model.
    """
    physical = Path(source_receipt["path"]).resolve(strict=True)
    before_name = getattr(tokenizer, "name_or_path", None)
    if not isinstance(before_name, str) or Path(before_name).resolve() != physical:
        raise ValueError("Tokenizer loading directory differs from the source receipt")
    if (source_receipt.get("repo_id"), source_receipt.get("revision")) != (
        logical_spec.model_id,
        logical_spec.revision,
    ):
        raise ValueError("Tokenizer source repository or revision differs")
    if not source_receipt.get("completed_at_unix"):
        raise ValueError("Completed tokenizer source receipt required")
    expected_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    }
    if set(frozen_tokenizer_files) != expected_names:
        raise ValueError("Exact frozen four-file Qwen tokenizer identity required")
    verified = {}
    for name, expected in frozen_tokenizer_files.items():
        if file_sha(physical / name) != expected:
            raise ValueError(
                "Physical tokenizer bytes differ from the frozen operator " + name
            )
        verified[name] = expected
    revision = (getattr(tokenizer, "init_kwargs", {}) or {}).get("revision")
    commit = (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
    if (revision or commit) not in {None, logical_spec.revision}:
        raise ValueError("Tokenizer instance reports another revision")
    before_type = type(tokenizer)
    before_init = copy.deepcopy(getattr(tokenizer, "init_kwargs", {}))
    before_encode = getattr(tokenizer.encode, "__func__", tokenizer.encode)
    before_decode = getattr(tokenizer.decode, "__func__", tokenizer.decode)
    probes = (
        "Synthetic arithmetic. Final answer: 7",
        "Synthetic logical expressions: true and false. (A) (B)",
        "测试题目与推理。 α β ∑ 1/3 — 🙂\n" * 90,
        "Synthetic member reasoning with preserved head and tail. " * 300,
    )

    def signatures():
        values = []
        for text in probes:
            ids = list(tokenizer.encode(text, add_special_tokens=False))
            values.append(
                dict(
                    ids=ids,
                    decoded=tokenizer.decode(ids, skip_special_tokens=True),
                    head_tail=execution_api.head_tail_candidate(text, tokenizer),
                )
            )
        return values

    before = signatures()
    tokenizer.name_or_path = logical_spec.model_path
    try:
        after = signatures()
        if (
            type(tokenizer) is not before_type
            or getattr(tokenizer, "init_kwargs", {}) != before_init
            or getattr(tokenizer.encode, "__func__", tokenizer.encode)
            is not before_encode
            or getattr(tokenizer.decode, "__func__", tokenizer.decode)
            is not before_decode
            or before != after
        ):
            raise ValueError("Tokenizer behavior changed during loading-path alias")
    except BaseException:
        tokenizer.name_or_path = before_name
        raise
    return dict(
        schema="e18_tokenizer_loading_path_alias_receipt_v1",
        status="PASS",
        physical_loading_path=str(physical),
        logical_identity_path=logical_spec.model_path,
        tokenizer_class=before_type.__module__ + "." + before_type.__qualname__,
        tokenizer_revision=revision or commit,
        tokenizer_files_sha256=verified,
        object_reused=True,
        methods_unchanged=True,
        init_kwargs_unchanged=True,
        encode_decode_head_tail_probes=len(probes),
        probe_signatures_unchanged=True,
        probe_source_token_counts=[len(row["ids"]) for row in before],
        mapping_provenance=mapping_provenance,
        language_model_generations=0,
    )


def frozen_tokenizer_files(apis):
    lock = (
        apis.root
        / "results/decision_feedback_v2/s1_typed_aggcal_generation42_9ef6c02bc9fb/OPERATOR_FREEZE_LOCK_v1.json"
    )
    if (
        file_sha(lock)
        != "66d77c2ab5eef0c6a360cfe2d8df01429369927c96fe3ae7b6a63697f4fc09b0"
    ):
        raise ValueError("Frozen operator lock bytes differ")
    return read_json(lock)["operator_identity"]["tokenizer"]["files_sha256"]


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

    def __init__(self, lock_root, gpu, manifest_sha):
        if not str(gpu).isdigit():
            raise ValueError("An explicit numeric physical GPU index is required")
        self.root = Path(lock_root).resolve()
        self.gpu = str(gpu)
        self.manifest_sha = manifest_sha
        self.stream = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.stream = (self.root / ("gpu-" + self.gpu + ".lock")).open("a+b")
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
            self.root / ("gpu-" + self.gpu + ".owner.json"),
            dict(
                pid=os.getpid(),
                host=socket.gethostname(),
                gpu=self.gpu,
                manifest_sha256=self.manifest_sha,
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


class OwnedBackendContext:
    """One exact production identity and one original backend per process."""

    def __init__(self, key, module, create, expected_fingerprint):
        self.key = key
        self.module = module
        self.create = create
        self.expected_fingerprint = expected_fingerprint
        self.backend = None
        self.fingerprint = None
        self.constructions = 0
        self.tokenizer = None
        self.tokenizer_receipt = None
        self.tokenizer_init = None
        self.tokenizer_methods = None
        self.source_receipt_sha256 = None

    def get(self, spec):
        fingerprint = self.module.configuration_fingerprint(spec)
        if fingerprint != self.expected_fingerprint:
            raise ValueError(
                "Requested profile differs from its owned process identity"
            )
        if self.backend is None:
            loaded = self.create(spec)
            if loaded.spec != spec:
                raise ValueError(
                    "Original backend changed its requested logical profile"
                )
            self.backend = loaded
            self.fingerprint = fingerprint
            self.constructions += 1
        elif fingerprint != self.fingerprint:
            raise ValueError("A process backend context cannot change logical profile")
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


def owned_context(adapter_module, adapter, manifest, factory, software_test_only):
    global _PROCESS_BACKEND_CONTEXT
    identity = dict(
        asset_root=str(adapter.apis.root),
        adapter_ref=manifest["adapter_ref"],
        scope_sha256=adapter_module.canonical_digest(adapter.scope),
        scope_ref=manifest["scope_ref"],
        approval_ref=manifest["approval_ref"],
        inputs_ref=manifest["question_records_ref"],
        owned_output_root=str(Path(manifest["owned_output_root"]).resolve()),
        logical_profile=adapter.apis.inference.configuration_fingerprint(
            adapter.aggregator_spec
        ),
        backend_loading=manifest["backend"],
        software_test_only=software_test_only,
    )
    key = adapter_module.canonical_digest(identity)
    if _PROCESS_BACKEND_CONTEXT is None:
        _PROCESS_BACKEND_CONTEXT = OwnedBackendContext(
            key, adapter.apis.inference, factory, identity["logical_profile"]
        )
    elif _PROCESS_BACKEND_CONTEXT.key != key:
        raise ValueError(
            "A different scope, approval, profile or loading path requires a separate worker process"
        )
    return _PROCESS_BACKEND_CONTEXT


def context_tokenizer(adapter_module, adapter, backend):
    loaded = backend(adapter.aggregator_spec)
    context = backend.context
    files = frozen_tokenizer_files(adapter.apis)
    if context.tokenizer_receipt is None:
        tokenizer = loaded.tokenizer
        receipt = restore_logical_tokenizer_path(
            tokenizer,
            adapter.aggregator_spec,
            loaded.source_receipt,
            files,
            adapter.apis.execution,
            mapping_provenance=getattr(loaded, "server_runtime_provenance", None),
        )
        context.tokenizer = tokenizer
        context.tokenizer_receipt = copy.deepcopy(receipt)
        context.tokenizer_init = copy.deepcopy(getattr(tokenizer, "init_kwargs", {}))
        context.tokenizer_methods = (
            type(tokenizer),
            getattr(tokenizer.encode, "__func__", tokenizer.encode),
            getattr(tokenizer.decode, "__func__", tokenizer.decode),
        )
        context.source_receipt_sha256 = adapter_module.canonical_digest(
            loaded.source_receipt
        )
        reused = False
    else:
        tokenizer = loaded.tokenizer
        receipt = copy.deepcopy(context.tokenizer_receipt)
        expected_methods = (
            type(tokenizer),
            getattr(tokenizer.encode, "__func__", tokenizer.encode),
            getattr(tokenizer.decode, "__func__", tokenizer.decode),
        )
        if (
            tokenizer is not context.tokenizer
            or tokenizer.name_or_path != adapter.aggregator_spec.model_path
            or getattr(tokenizer, "init_kwargs", {}) != context.tokenizer_init
            or expected_methods != context.tokenizer_methods
            or adapter_module.canonical_digest(loaded.source_receipt)
            != context.source_receipt_sha256
            or files != receipt["tokenizer_files_sha256"]
        ):
            raise ValueError("Cross-shard tokenizer or source identity changed")
        physical = Path(receipt["physical_loading_path"])
        for name, expected in files.items():
            if file_sha(physical / name) != expected:
                raise ValueError("Cross-shard physical tokenizer bytes changed " + name)
        reused = True
    receipt.update(
        process_backend_context_sha256=context.key,
        cross_shard_backend_reused=reused,
        process_physical_backend_constructions=context.constructions,
        current_shard_tokenizer_files_reverified=True,
    )
    return tokenizer, receipt


def load_worker(manifest, *, software_test_only=False, injected_factory=None):
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Unexpected worker manifest schema")
    if manifest.get("stage") != "aggregation":
        raise ValueError("This version admits the aggregation stage only")
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
                    tokenizer, alias_receipt = context_tokenizer(
                        module, adapter, backend
                    )
                    save_json(
                        journal_root / "TOKENIZER_LOADING_PATH_RECEIPT.json",
                        alias_receipt,
                    )
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
    with GPUOwnership(
        manifest["gpu_lock_root"], arguments.gpu, arguments.manifest_sha256
    ):
        summary = execute_manifest(manifest, arguments.manifest_sha256)
    print(json.dumps(summary, ensure_ascii=False))
    if summary["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
