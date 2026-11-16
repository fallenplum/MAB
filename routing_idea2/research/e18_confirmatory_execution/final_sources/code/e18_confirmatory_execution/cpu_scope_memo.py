"""Scope-only CPU memoization around the unchanged E18 finalizer.

Only explicitly registered, recursively read-only scope identities are cached.
Native records, jobs, permissions, resource checks and reference scans retain
their original code paths. Source and scope integrity are rechecked before the
original finalizer publishes results and when this context closes.
"""

from __future__ import annotations

import argparse
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import time


SOURCE_PINS = {
    "e18_final_scoring.py": "9b164c40ad6b3a1b21f84c7dc446438a11ce6421d56991d82d224ae2c4b0daa8",
    "e18_aggregation_worker_v3.py": "de5c3cc8a509be10e66b3290899862bab73a2784a62aa743e7c48d41a7842495",
    "e18_execution_adapter.py": "21e76f6a38fa5f2ed590a2b2068bda8255bc4ebc290dea9eb7c32a3a44a65bc4",
    "e18_scoring_adapter.py": "7eb390f0424f2f18fda806c188e0f3e8e7ed2fc633605c5a3d170d271ca06d8a",
    "e18_reference_adapter.py": "8848aa235d98e11005a33ff969274dc0309d02228297f4242f207663e57178ea",
}
_ACTIVE_SCOPE_MEMO = None


def _deny(*args, **kwargs):
    raise TypeError("An approved scope snapshot is read-only")


class FrozenDict(dict):
    """JSON-compatible dictionary with ordinary mutation operations disabled."""

    def __init__(self, *args, **kwargs):
        if getattr(self, "_sealed", False):
            _deny()
        dict.__init__(self, *args, **kwargs)
        object.__setattr__(self, "_sealed", True)

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _deny
    __ior__ = __setattr__ = __delattr__ = _deny

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def copy(self):
        return self


class FrozenList(list):
    """JSON-compatible list with ordinary mutation operations disabled."""

    def __init__(self, *args):
        if getattr(self, "_sealed", False):
            _deny()
        list.__init__(self, *args)
        object.__setattr__(self, "_sealed", True)

    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = _deny
    reverse = sort = __iadd__ = __imul__ = __setattr__ = __delattr__ = _deny

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def copy(self):
        return self


def freeze_scope(value):
    """Copy a JSON tree into read-only JSON-compatible dict/list subclasses."""
    if isinstance(value, (FrozenDict, FrozenList)):
        return value
    if isinstance(value, dict):
        return FrozenDict({key: freeze_scope(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze_scope(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("Scope snapshots accept JSON values only")


def original_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            checksum.update(block)
    return checksum.hexdigest()


class ScopeMemo:
    """Install a reversible memo for the three file-bound E18 scope roles.

    The built-in base-class mutators can deliberately bypass a Python subclass.
    Such unsupported bypasses are detected by the original full integrity pass
    before final result publication. This is an operational optimization inside
    one trusted process, with no cache of mutable native records or permissions.
    """

    def __init__(self, finalizer_module, manifest, *, software_test_only=False,
                 enabled=True):
        self.final = finalizer_module
        self.worker = finalizer_module.worker
        self.manifest = manifest
        self.software_test_only = software_test_only
        self.enabled = enabled
        self._patches = []
        self._patched = set()
        self._objects = {}
        self._files = {}
        self._role_sources = {}
        self._role_digests = {}
        self._source_roles = {}
        self._original_digest_functions = {}
        self._entered = False
        self._counts = dict(digest_calls=0, scope_cache_hits=0, original_digest_calls=0,
                            scope_objects_registered=0, integrity_passes=0)
        self._times = dict(digest_cpu_s=0.0, integrity_cpu_s=0.0)
        self._original_read_json = self.worker.read_json
        self._pin_module(self.final)
        self._pin_module(self.worker)
        execution_ref = manifest["execution_manifest_ref"]
        self._pin_ref(execution_ref)
        execution = self._original_read_json(execution_ref["path"])
        self._pin_ref(execution["approval_ref"])
        for role, reference in (
            ("execution", execution["scope_ref"]),
            ("reference", manifest["reference_scope_ref"]),
            ("scoring", manifest["scoring_scope_ref"]),
        ):
            path = self._pin_ref(reference)
            scope = self._original_read_json(path)
            if scope.get("schema") != "e18_scope_contract_v2" or scope.get("role") != role:
                raise ValueError("The file-bound scope role differs")
            self._role_sources[role] = dict(path=str(path), sha256=reference["sha256"])
            self._role_digests[role] = original_digest(scope)
            self._source_roles[path] = role
        for reference in (execution["adapter_ref"], manifest["source_validator_ref"],
                          manifest["reference_adapter_ref"]):
            path = self._pin_ref(reference)
            if SOURCE_PINS.get(path.name) != reference["sha256"]:
                raise ValueError("A performance-wrapper scientific dependency differs")

    def _pin_ref(self, reference):
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError("An exact source path and SHA reference is required")
        path = Path(reference["path"]).resolve(strict=True)
        if not path.is_file() or file_sha(path) != reference["sha256"]:
            raise ValueError("A bound performance source file differs")
        old = self._files.get(path)
        if old is not None and old != reference["sha256"]:
            raise ValueError("One source path has incompatible byte identities")
        self._files[path] = reference["sha256"]
        return path

    def _pin_module(self, module):
        path = Path(module.__file__).resolve(strict=True)
        expected = SOURCE_PINS.get(path.name)
        if expected is None or file_sha(path) != expected:
            raise ValueError("The original frozen module source differs")
        self._files[path] = expected

    def register_scope(self, value, role):
        """Return a file-bound immutable snapshot and retain its strong identity."""
        existing = self._objects.get(id(value))
        if existing is not None and existing[0] is value:
            if existing[2] != role:
                raise ValueError("One immutable scope was registered under another role")
            return value
        if (value.get("schema"), value.get("role"), value.get("cohort")) != (
            "e18_scope_contract_v2", role, "e18_confirm"
        ) or original_digest(value) != self._role_digests[role]:
            raise ValueError("The scope copy differs from its SHA-bound source")
        frozen = freeze_scope(value)
        for function in self._original_digest_functions.values():
            if function(frozen) != self._role_digests[role]:
                raise ValueError("The original module scope digest differs from the bound canonical digest")
        self._objects[id(frozen)] = (frozen, self._role_digests[role], role)
        self._counts["scope_objects_registered"] += 1
        return frozen

    def digest(self, value, original=None):
        started = time.process_time()
        self._counts["digest_calls"] += 1
        row = self._objects.get(id(value))
        if self.enabled and row is not None and row[0] is value:
            self._counts["scope_cache_hits"] += 1
            answer = row[1]
        else:
            self._counts["original_digest_calls"] += 1
            answer = (original or original_digest)(value)
        self._times["digest_cpu_s"] += time.process_time() - started
        return answer

    def _digest_hook(self, original):
        """Capture the real module callable for every non-scope input."""
        if getattr(original, "_e18_scope_memo_owner", None) is self:
            return original
        if id(original) not in self._original_digest_functions:
            self._original_digest_functions[id(original)] = original
            checked_roles = set()
            for value, expected, role in self._objects.values():
                if role not in checked_roles and original(value) != expected:
                    raise ValueError("Original digest semantics differ for an approved scope")
                checked_roles.add(role)

        @wraps(original)
        def digest(value):
            return self.digest(value, original)

        digest._e18_scope_memo_owner = self
        return digest

    def verify_integrity(self):
        """Use the original complete digest and file SHA, bypassing all memo hits."""
        started = time.process_time()
        for path, expected in self._files.items():
            if file_sha(path) != expected:
                raise ValueError("A bound original source changed during CPU evaluation")
        for value, expected, role in self._objects.values():
            if original_digest(value) != expected:
                raise ValueError("A read-only scope was bypass-mutated: " + role)
        self._counts["integrity_passes"] += 1
        self._times["integrity_cpu_s"] += time.process_time() - started

    def stats(self):
        return dict(schema="e18_scope_only_cpu_memo_stats_v1", enabled=self.enabled,
                    **self._counts, **self._times,
                    scope_source_sha256_by_role={role: row["sha256"] for role, row in self._role_sources.items()},
                    original_canonical_scope_sha256_by_role=dict(self._role_digests),
                    source_files_verified=len(self._files),
                    native_or_job_objects_cached=0, approval_checks_skipped=0,
                    reference_source_scans_skipped=0)

    def _patch(self, target, name, value):
        identity = (id(target), name)
        if identity in self._patched:
            return
        self._patches.append((target, name, getattr(target, name)))
        self._patched.add(identity)
        setattr(target, name, value)

    def _adapter(self, module):
        self._pin_module(module)
        if (id(module), "canonical_digest") not in self._patched:
            self._patch(module, "canonical_digest", self._digest_hook(module.canonical_digest))
        return module

    def _validator(self, module):
        self._pin_module(module)
        if (id(module), "digest") not in self._patched:
            self._patch(module, "digest", self._digest_hook(module.digest))
        base = module.SparseScoringAdapter
        manager = self

        class MemoScorer(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.scope_contract = manager.register_scope(self.scope_contract, "scoring")

        self._patch(module, "SparseScoringAdapter", MemoScorer)
        return module

    def _reference(self, module):
        self._pin_module(module)
        if (id(module), "canonical_digest") not in self._patched:
            self._patch(module, "canonical_digest", self._digest_hook(module.canonical_digest))
        base = module.E18ReferenceProvider
        manager = self

        class MemoProvider(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.scope = manager.register_scope(self.scope, "reference")

        self._patch(module, "E18ReferenceProvider", MemoProvider)
        return module

    def __enter__(self):
        global _ACTIVE_SCOPE_MEMO
        if self._entered:
            raise RuntimeError("One scope memo context cannot be entered twice")
        if _ACTIVE_SCOPE_MEMO is not None:
            raise RuntimeError("Another scope memo context already owns the process hooks")
        self._entered = True
        self.verify_integrity()
        _ACTIVE_SCOPE_MEMO = self
        load_adapter = self.worker.load_adapter
        load_worker = self.worker.load_worker
        load_validator = self.final.audit_startup16.load_source_validator
        load_reference = self.final.load_reference_api

        def read_json(path):
            value = self._original_read_json(path)
            role = self._source_roles.get(Path(path).resolve())
            return self.register_scope(value, role) if role else value

        def adapter_loader(reference):
            return self._adapter(load_adapter(reference))

        def worker_loader(*args, **kwargs):
            result = load_worker(*args, **kwargs)
            module, adapter = result[:2]
            self._adapter(module)
            adapter.scope = self.register_scope(adapter.scope, "execution")
            return result

        def validator_loader(reference):
            return self._validator(load_validator(reference))

        def reference_loader(reference):
            return self._reference(load_reference(reference))

        def publication_guard(function):
            def guarded(*args, **kwargs):
                self.verify_integrity()
                return function(*args, **kwargs)
            return guarded

        self._patch(self.worker, "read_json", read_json)
        self._patch(self.worker, "load_adapter", adapter_loader)
        self._patch(self.worker, "load_worker", worker_loader)
        self._patch(self.final.audit_startup16, "load_source_validator", validator_loader)
        self._patch(self.final, "load_reference_api", reference_loader)
        for name in ("write_new_json", "write_npz_once"):
            self._patch(self.final, name, publication_guard(getattr(self.final, name)))
        try:
            self.verify_integrity()
        except BaseException:
            self._restore_patches()
            raise
        return self

    def _restore_patches(self):
        global _ACTIVE_SCOPE_MEMO
        for target, name, original in reversed(self._patches):
            setattr(target, name, original)
        self._patches.clear()
        if _ACTIVE_SCOPE_MEMO is self:
            _ACTIVE_SCOPE_MEMO = None

    def __exit__(self, exception_type, exception, traceback):
        try:
            self.verify_integrity()
        finally:
            self._restore_patches()
        return False


def main():
    import e18_final_scoring as final

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--native-read-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--performance-receipt", type=Path, required=True)
    parser.add_argument("--disable-scope-memo", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute-scoring", action="store_true")
    args = parser.parse_args()
    if args.performance_receipt.exists():
        raise FileExistsError("Performance receipts are append-only")
    if file_sha(args.manifest) != args.manifest_sha256:
        raise ValueError("Final manifest bytes differ")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    manifest = final.worker.read_json(args.manifest)
    memo = ScopeMemo(final, manifest, enabled=not args.disable_scope_memo)
    memo._pin_ref(dict(path=str(args.manifest), sha256=args.manifest_sha256))
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    status, error_type, context = "FAILED", None, None
    try:
        with memo:
            context = final.execution_gate(manifest, native_read_root=args.native_read_root)
            private = Path(args.runtime_root or manifest.get("cpu_output_root") or manifest["private_output_root"])
            final.write_new_json(private / ("PRE_SCORING_ADMISSION_" + str(time.time_ns()) + ".json"), context["gate"])
            if args.preflight_only:
                status = "EXECUTION_COMPLETE_REFERENCE_UNREAD"
            else:
                result = final.execute_scoring(manifest, context, runtime_root=args.runtime_root)
                status = result["status"]
    except BaseException as error:
        status = "FAILED_INTEGRITY_OR_EXECUTION"
        error_type = type(error).__name__
        raise
    finally:
        receipt = dict(schema="e18_scope_only_cpu_performance_receipt_v1", status=status,
                       error_type=error_type, wrapper_source_sha256=file_sha(__file__),
                       original_finalizer_source_sha256=SOURCE_PINS["e18_final_scoring.py"],
                       manifest_sha256=args.manifest_sha256,
                       mode="preflight_only" if args.preflight_only else "execute_scoring",
                       wall_seconds=time.perf_counter()-started_wall,
                       cpu_seconds=time.process_time()-started_cpu, scope_memo=memo.stats(),
                       scientific_sources_modified=False,
                       reference_access_possible=not args.preflight_only)
        args.performance_receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.performance_receipt.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(dict(status=status, error_type=error_type,
                              performance_receipt=str(args.performance_receipt),
                              scope_cache_hits=memo.stats()["scope_cache_hits"])))


if __name__ == "__main__":
    main()
