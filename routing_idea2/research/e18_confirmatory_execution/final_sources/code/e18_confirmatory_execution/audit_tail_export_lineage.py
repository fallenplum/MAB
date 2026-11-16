"""Read-only E18 fine-plan to closed-native provenance audit.

Only metadata JSON values are decoded. Native answer payloads are streamed past
inside gzip/TAR traversal and are never JSON-decoded. A previously completed,
separately pinned native audit supplies the body/ModelSpec integrity proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time
import zipfile


GATE_SHA = "9b164c40ad6b3a1b21f84c7dc446438a11ce6421d56991d82d224ae2c4b0daa8"
NATIVE_AUDITOR_SHA = "c506dff8698666fa1928ed922c6d0df746e7a7ca7a0bada0f9d37541d6433948"
ADAPTER_SHA = "21e76f6a38fa5f2ed590a2b2068bda8255bc4ebc290dea9eb7c32a3a44a65bc4"
WORKER_SHA = "5ac9a17dd07d282bfa2cceaff3dd062ce837e68885b4086816ef509974af5786"
CONTROLLER_SHA = "45c88b376059a4b390d065ef61e38db24fa1862cf8b98f7658c110d89c423f9f"
PLAN_SHA = "dd01ef0f069ed3a55a81bbf4600a3de4234002a32fa9d08187306a3773aaff5d"
PARENT_PLAN_SHA = "a627f3094c742ec0d352582a9abe8b6338af6eaf0967333ebf30f45727c97166"
PARENT_SHA = "826e72af7b84e48d0421417eab3c1ed9c8907633fa88fdd756229c3bdc7fe323"
MAX_FILE_BYTES = 64 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}")
FORBIDDEN = {
    "gold", "gold_answer", "reference", "references", "reference_answer",
    "reference_answers", "ground_truth", "target", "targets", "label",
    "labels", "reward", "rewards", "score", "scores", "is_correct", "y",
    "text", "question", "prompt", "answer", "completion", "content",
    "native_record", "password", "credential", "secret",
}


def file_sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def metadata_json(raw):
    """Reject protected property names before decoding any JSON values."""
    position = 0
    while position < len(raw):
        if raw[position] != 34:
            position += 1
            continue
        start = position
        position += 1
        while position < len(raw):
            if raw[position] == 92:
                position += 2
            elif raw[position] == 34:
                position += 1
                break
            else:
                position += 1
        end = position
        while position < len(raw) and raw[position] in b" \r\n\t":
            position += 1
        if position < len(raw) and raw[position] == 58:
            if json.loads(raw[start:end]).casefold() in FORBIDDEN:
                raise ValueError("Protected payload property refused before value decoding")
    return json.loads(raw)


def checked_file(reference, *, decode=True):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Exact path/SHA reference required")
    path = Path(reference["path"])
    before = path.stat()
    if not path.is_file() or before.st_size > MAX_FILE_BYTES:
        raise ValueError("Only bounded explicit metadata/code files are accepted")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Explicit source SHA differs")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed during read")
    return metadata_json(raw) if decode else raw


class MetadataBundle:
    def __init__(self, reference):
        path = Path(reference["path"])
        if file_sha(path) != reference["sha256"]:
            raise ValueError("Static source bundle SHA differs")
        self.reference = dict(reference)
        self.raw = {}
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                pure = PurePosixPath(item.filename)
                kind = stat.S_IFMT(item.external_attr >> 16)
                if (
                    pure.is_absolute() or ".." in pure.parts or "\\" in item.filename
                    or item.filename in self.raw or item.is_dir()
                    or kind not in (0, stat.S_IFREG)
                    or not 0 <= item.file_size <= MAX_FILE_BYTES
                ):
                    raise ValueError("Unsafe or duplicate metadata bundle member")
                self.raw[item.filename] = archive.read(item)
        inventory = metadata_json(self.raw["BUNDLE_MANIFEST.json"])
        rows = inventory["files"]
        expected = {row["path"]: row for row in rows}
        if len(rows) != len(expected) or set(self.raw) != set(expected) | {"BUNDLE_MANIFEST.json"}:
            raise ValueError("Bundle file inventory differs")
        for name, row in expected.items():
            raw = self.raw[name]
            if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Bundle payload bytes differ")
        self.plan_name = "MEMBER4_TAIL_PLAN_v1.json"
        self.plan = metadata_json(self.raw[self.plan_name])
        self.plan_ref = dict(
            path=str(PurePosixPath(self.plan["tail_parent_plan_ref"]["path"]).parent / self.plan_name),
            sha256=hashlib.sha256(self.raw[self.plan_name]).hexdigest(),
        )
        self.remote_root = PurePosixPath(self.plan_ref["path"]).parent
        self.used = {}

    def read(self, reference, *, decode=True):
        path = PurePosixPath(reference["path"])
        if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(self.remote_root):
            raise ValueError("A bundle source escaped the exact remote experiment root")
        name = str(path.relative_to(self.remote_root))
        if name not in self.raw or hashlib.sha256(self.raw[name]).hexdigest() != reference["sha256"]:
            raise ValueError("An exact logical source path/SHA is absent from the static bundle")
        self.used[reference["path"]] = dict(reference)
        return metadata_json(self.raw[name]) if decode else self.raw[name]


def unique_keys(rows):
    if not isinstance(rows, list):
        raise ValueError("Task-key metadata must be an array")
    keys = [tuple(row) for row in rows]
    if (
        len(keys) != len(set(keys))
        or any(len(key) != 3 or not isinstance(key[0], str) or type(key[1]) is not int or key[2] != 4 for key in keys)
    ):
        raise ValueError("Unique member4 task metadata required")
    return set(keys)


def inspect_fine_sources(bundle, scope, *, qualification_only=False):
    plan = bundle.plan
    lease_ref = plan["tail_parent_lease_ref"]
    lease = bundle.read(lease_ref)
    parent_ref = lease["parent_manifest_ref"]
    parent_plan_ref = lease["parent_plan_ref"]
    parent_plan = bundle.read(parent_plan_ref)
    parent = bundle.read(parent_ref)
    upstream_ref = lease["upstream_checkpoint_lease_ref"]
    upstream = bundle.read(upstream_ref)
    parent_keys = unique_keys(bundle.read(parent["task_keys_ref"]))
    remaining = unique_keys(bundle.read(lease["remaining_task_keys_ref"]))
    upstream_keys = unique_keys(bundle.read(upstream["remaining_task_keys_ref"]))
    audit_ref = lease["checkpoint_source_audit_ref"]
    audit = bundle.read(audit_ref)
    controller_ref = lease["controller_source_ref"]
    bundle.read(controller_ref, decode=False)
    if (
        plan.get("schema") != "e18_member4_dispatch_plan_v3"
        or plan.get("worker_source_sha256") != WORKER_SHA
        or lease.get("schema") != "e18_member4_tail_partition_lease_v1"
        or lease.get("status") != "LEASED" or lease.get("software_test_only") is not False
        or lease.get("pending_or_unknown_attempts") != 0
        or lease.get("execution_scope_sha256") != digest(scope)
        or lease.get("approval_sha256") != parent["approval_ref"]["sha256"]
        or controller_ref["sha256"] != CONTROLLER_SHA
        or parent_ref not in parent_plan["manifest_refs"]
        or plan["tail_parent_plan_ref"] != parent_plan_ref
        or plan["tail_parent_manifest_ref"] != parent_ref
        or plan["checkpoint_lease_ref"] != parent["checkpoint_lease_ref"]
        or upstream_ref != parent["checkpoint_lease_ref"]
        or parent.get("stage") != "member_generation" or parent.get("member_index") != 4
        or parent.get("software_test_only") is not False
        or parent["adapter_ref"]["sha256"] != ADAPTER_SHA
        or not parent_keys <= upstream_keys
        or not remaining <= parent_keys
        or len(parent_keys) != lease["original_parent_tasks"]
        or len(remaining) != lease["remaining_count"]
        or lease["checkpoint_successes"] + len(remaining) != len(parent_keys)
        or type(lease.get("maximum_child_keys")) is not int
        or not 1 <= lease["maximum_child_keys"] <= 64
    ):
        raise ValueError("Fine plan/lease/original-source identities differ")
    if not qualification_only and (
        bundle.plan_ref["sha256"] != PLAN_SHA
        or parent_plan_ref["sha256"] != PARENT_PLAN_SHA
        or parent_ref["sha256"] != PARENT_SHA
        or len(parent_keys) != 512
        or lease["checkpoint_successes"] != 195 or len(remaining) != 317
    ):
        raise ValueError("Frozen real512/195/317 fine checkpoint differs")
    expected_claim = bundle.remote_root / "owned/_dispatcher/plans" / parent_plan_ref["sha256"] / (parent_ref["sha256"] + ".claim")
    if PurePosixPath(lease["parent_claim_path"]) != expected_claim:
        raise ValueError("Original child-claim namespace differs")
    by_hash = {row["task_key_sha256"]: row for row in audit["tasks"]}
    if (
        len(by_hash) != len(audit["tasks"]) or len(by_hash) != len(parent_keys)
        or set(by_hash) != {digest(list(key)) for key in parent_keys}
        or {key for key, value in by_hash.items() if value.get("status") == "UNEXECUTED"} != {digest(list(key)) for key in remaining}
        or sum(value.get("status") == "SUCCESS" for value in by_hash.values()) != lease["checkpoint_successes"]
        or any(value.get("status") not in {"SUCCESS", "UNEXECUTED"} for value in by_hash.values())
        or audit.get("pending_or_unknown_attempts") != 0
    ):
        raise ValueError("Checkpoint audit is not an exact SUCCESS/unexecuted partition")
    by_key, children, registered_refs = {}, {}, [bundle.plan_ref]
    for index, reference in enumerate(plan["manifest_refs"]):
        manifest = bundle.read(reference)
        keys = unique_keys(bundle.read(manifest["task_keys_ref"]))
        if (
            set(manifest) != set(parent) | {"tail_parent_lease_ref", "tail_partition_index"}
            or manifest["tail_parent_lease_ref"] != lease_ref
            or manifest["tail_partition_index"] != index
            or any(manifest[name] != value for name, value in parent.items() if name != "task_keys_ref")
            or not 1 <= len(keys) <= lease["maximum_child_keys"]
            or not keys <= remaining or set(by_key).intersection(keys)
        ):
            raise ValueError("A fine manifest changes requests or its exact key partition")
        for key in keys:
            by_key[key] = reference["sha256"]
        children[reference["sha256"]] = manifest
        registered_refs.extend([reference, manifest["task_keys_ref"]])
    if set(by_key) != remaining:
        raise ValueError("Fine manifests do not cover all and only remaining keys")
    sidecars = [name for name in bundle.raw if "R1_TAIL_SOURCE_EXTENSION_" in name and name.endswith(".json")]
    if len(sidecars) != 1:
        raise ValueError("Exactly one planner tail-source sidecar is required")
    sidecar = metadata_json(bundle.raw[sidecars[0]])
    expected_sidecar_sources = dict(
        plan_ref=bundle.plan_ref, tail_parent_lease_ref=lease_ref,
        parent_plan_ref=parent_plan_ref, parent_manifest_ref=parent_ref,
        checkpoint_source_audit_ref=audit_ref,
    )
    if (
        sidecar.get("schema") != "e18_planner_r1_tail_source_extension_v1"
        or sidecar.get("tail_source_plan_refs") != [bundle.plan_ref]
        or sidecar.get("tail_control_source_ref") != controller_ref
        or sidecar.get("additional_scientific_keys") != 0
        or sidecar.get("original_config_modified") is not False
        or sidecar.get("sqlite_schema_modified") is not False
        or sidecar.get("historical_source_admission_requires_live_broker") is not False
        or len(sidecar.get("source_audits", [])) != 1
        or any(sidecar["source_audits"][0].get(name) != value for name, value in expected_sidecar_sources.items())
        or sidecar.get("all_done_config_sha256") != sidecar["original_config_ref"]["sha256"]
    ):
        raise ValueError("Planner sidecar is detached from the exact fine checkpoint")
    planner_raw = bundle.raw["planner_v5_runtime/incremental_aggregation_planner_v5.py"]
    if hashlib.sha256(planner_raw).hexdigest() != sidecar["extension_source_sha256"]:
        raise ValueError("Planner extension source bytes differ")
    pilot_name = "resource_scheduling/MEMBER4_TAIL_PILOT_NATIVE_ADMISSION_v1.json"
    pilot = metadata_json(bundle.raw[pilot_name])
    if (
        pilot.get("status") != "PASS_ORIGINAL_MODELSPEC_JOB_OWNER_AND_ACTUAL_COST"
        or pilot.get("fine_plan_ref") != bundle.plan_ref
        or pilot.get("fine_manifest_ref", {}).get("sha256") not in children
        or pilot.get("task_key_sha256") not in {digest(list(key)) for key in remaining}
        or pilot.get("native_validator_source_sha256") != "a2054daad25fc3ce63f9ec260b17cca86e098bec8077096e6f2edab237364922"
        or pilot.get("original_adapter_source_sha256") != ADAPTER_SHA
        or any(pilot.get(name) != 0 for name in ("new_model_calls_by_validation", "gold_reads", "scorer_calls", "backend_loads_by_validation"))
    ):
        raise ValueError("Existing real-pilot source admission differs")
    return dict(
        plan_ref=bundle.plan_ref, lease_ref=lease_ref, parent_ref=parent_ref,
        parent_plan_ref=parent_plan_ref, lease=lease, by_key=by_key, children=children,
        parent_keys=parent_keys, remaining=remaining, checkpoint_rows=by_hash,
        registered_refs=registered_refs, pilot=pilot, sidecar_name=sidecars[0],
        sidecar_sha256=hashlib.sha256(bundle.raw[sidecars[0]]).hexdigest(),
        external_refs=[lease_ref, audit_ref, controller_ref, upstream_ref],
    )


def export_metadata(path, expected, native_audit_row):
    if path.stat().st_size != expected["bytes"] or file_sha(path) != expected["sha256"]:
        raise ValueError("Closed native archive SHA/size differs")
    raw = None
    names = set()
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            pure = PurePosixPath(member.name)
            if (
                not member.isfile() or pure.is_absolute() or ".." in pure.parts
                or "\\" in member.name or member.name in names
                or not 0 <= member.size <= MAX_FILE_BYTES
            ):
                raise ValueError("Unsafe native archive member")
            names.add(member.name)
            if member.name == "EXPORT_MANIFEST.json":
                raw = archive.extractfile(member).read()
    if raw is None:
        raise ValueError("Closed archive metadata is absent")
    metadata = metadata_json(raw)
    if (
        hashlib.sha256(raw).hexdigest() != native_audit_row["manifest_raw_sha256"]
        or digest(metadata) != native_audit_row["manifest_semantic_sha256"]
        or len(metadata["tasks"]) != native_audit_row["native_tasks"]
        or len(metadata["files"]) != 3 * len(metadata["tasks"])
        or names != {row["path"] for row in metadata["files"]} | {"EXPORT_MANIFEST.json"}
    ):
        raise ValueError("Export metadata is detached from its completed native integrity audit")
    return metadata


def cache_flag_predecessor_sha(raw):
    """Hash the exact prior bytes with only a top-level cache True set False.

    String values, including generated answer strings, are skipped. Only JSON
    property names are decoded, and nested cache-looking strings are ignored.
    """
    position, depth, matches = 0, 0, []
    while position < len(raw):
        value = raw[position]
        if value in (123, 91):
            depth += 1
            position += 1
        elif value in (125, 93):
            depth -= 1
            position += 1
        elif value == 34:
            start = position
            position += 1
            while position < len(raw):
                if raw[position] == 92:
                    position += 2
                elif raw[position] == 34:
                    position += 1
                    break
                else:
                    position += 1
            end = position
            after = end
            while after < len(raw) and raw[after] in b" \r\n\t":
                after += 1
            if depth == 1 and after < len(raw) and raw[after] == 58:
                if json.loads(raw[start:end]) == "from_cache":
                    after += 1
                    while after < len(raw) and raw[after] in b" \r\n\t":
                        after += 1
                    if raw[after:after + 4] != b"true":
                        raise ValueError("Only the documented cache False-to-True transition is allowed")
                    matches.append((after, after + 4))
        else:
            position += 1
    if depth != 0 or len(matches) != 1:
        raise ValueError("An unambiguous top-level cache flag is required")
    start, end = matches[0]
    return hashlib.sha256(raw[:start] + b"false" + raw[end:]).hexdigest()


def verified_cache_only_transition(private_root, row, previous_sha):
    path = private_root / "native_exports" / row["_archive_filename"]
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if member.name == row["_receipt_member_name"]:
                raw = archive.extractfile(member).read()
                if hashlib.sha256(raw).hexdigest() != row["_receipt_file_sha256"]:
                    raise ValueError("Current cache receipt bytes differ from the native-audited archive")
                if cache_flag_predecessor_sha(raw) != previous_sha:
                    raise ValueError("Receipt changed beyond its documented cache flag")
                return True
    raise ValueError("The current cache receipt is absent")


def audit(
    private_root, bundle_ref, native_audit_ref, scope_ref, gate_source_ref,
    *, require_complete_fine=False, qualification_only=False,
):
    private_root = Path(private_root).resolve()
    scope = checked_file(scope_ref)
    native_audit = checked_file(native_audit_ref)
    gate_raw = checked_file(gate_source_ref, decode=False)
    if (
        hashlib.sha256(gate_raw).hexdigest() != GATE_SHA
        or native_audit.get("status") != "PASS_INDEPENDENT_STREAMING_NATIVE_AUDIT"
        or native_audit.get("independent_tar_auditor_sha256") != NATIVE_AUDITOR_SHA
        or native_audit.get("execution_scope_sha256") != digest(scope)
        or native_audit.get("execution_scope_file_sha256") != scope_ref["sha256"]
        or any(native_audit.get(name) != 0 for name in ("evaluation_results_read", "gold_or_reward_sources_read", "model_or_scorer_calls"))
    ):
        raise ValueError("Pinned CPU gate, scope or completed native audit differs")
    bundle = MetadataBundle(bundle_ref)
    fine = inspect_fine_sources(bundle, scope, qualification_only=qualification_only)
    allowed = set(map(tuple, scope["member_execution_keys"]))
    if not fine["parent_keys"] <= allowed:
        raise ValueError("Tail parent exceeds the frozen new-member execution union")
    audit_archives = {row["sha256"]: row for row in native_audit["archives"]}
    seen_archives, tasks, attempts, sources = set(), {}, {}, []
    archive_reports = []
    for source in native_audit["source_batches"]:
        source_path = Path(source["path"]).resolve()
        if source_path.parent != private_root or re.fullmatch(r"NATIVE_EXPORT_RESULT_v\d+\.json", source_path.name) is None:
            raise ValueError("Only explicit canonical closed-export catalogues are allowed")
        reference = dict(path=str(source_path), sha256=source["sha256"])
        catalogue = checked_file(reference)
        sources.append(reference)
        for entry in catalogue["new_archives"]:
            checksum = entry["sha256"]
            if checksum in seen_archives or checksum not in audit_archives:
                raise ValueError("Duplicate or unaudited closed archive")
            name = PurePosixPath(entry["path"]).name
            if re.fullmatch(r"native-[0-9a-f]{32}\.tar\.gz", name) is None:
                raise ValueError("Unexpected native archive name")
            metadata = export_metadata(private_root / "native_exports" / name, entry, audit_archives[checksum])
            if metadata.get("scope_sha256") != digest(scope) or metadata.get("software_test_only") is not False:
                raise ValueError("Archive execution scope differs")
            registered = {row["sha256"] for row in metadata["metadata_sources"]}
            workers = {row["manifest_sha256"]: row for row in metadata["worker_sources"]}
            files = {row["path"]: row for row in metadata["files"]}
            count = 0
            for row in metadata["tasks"]:
                task_id, attempt = row["task_id"], row["physical_attempt_id"]
                if task_id in tasks or attempt in attempts:
                    raise ValueError("Repeated task/physical attempt across closed archives")
                tasks[task_id], attempts[attempt] = row, task_id
                key = tuple(row["task_key"])
                if row["stage"] != "member_generation" or key not in fine["parent_keys"]:
                    continue
                key_hash = digest(list(key))
                row["_source_archive_sha256"] = checksum
                row["_archive_filename"] = name
                prefix = digest(scope) + "/member_generation/" + key_hash + "/"
                row["_receipt_member_name"] = prefix + "LAST_TASK_RECEIPT.json"
                row["_receipt_file_sha256"] = files[prefix + "LAST_TASK_RECEIPT.json"]["sha256"]
                row["_native_file_sha256"] = files[prefix + "records.jsonl"]["sha256"]
                row["_fence_file_sha256"] = files[prefix + "ATTEMPT_STATE.json"]["sha256"]
                if fine["parent_ref"]["sha256"] not in row["manifest_sha256s"]:
                    raise ValueError("An archived parent task lacks its original source registration")
                if key in fine["remaining"]:
                    count += 1
                    child_sha = fine["by_key"][key]
                    if child_sha not in row["manifest_sha256s"]:
                        raise ValueError("An archived fine task omits its real fine producer reference")
                    if any(reference["sha256"] not in registered for reference in fine["registered_refs"]):
                        raise ValueError("An archived fine task lacks the full fine metadata registration")
                    for child_sha, child in fine["children"].items():
                        if child_sha not in workers or any(workers[child_sha][field] != child[field] for field in ("adapter_ref", "backend", "task_keys_ref")):
                            raise ValueError("Registered fine producer fields differ")
            seen_archives.add(checksum)
            archive_reports.append(dict(archive_sha256=checksum, native_tasks=len(metadata["tasks"]), fine_native_tasks=count))
    if seen_archives != set(audit_archives) or len(tasks) != native_audit["native_tasks_checked"]:
        raise ValueError("Catalogue/closed-native audit coverage differs")
    by_key = {tuple(row["task_key"]): row for row in tasks.values() if row["stage"] == "member_generation"}
    fine_rows, prefix_rows, prefix_cache_transitions = [], [], 0
    for key in sorted(fine["parent_keys"]):
        row = by_key.get(key)
        if row is None:
            continue
        key_hash = digest(list(key))
        checkpoint = fine["checkpoint_rows"][key_hash]
        item = dict(task_key_sha256=key_hash, native_record_sha256=row["native_record_sha256"], physical_attempt_id=row["physical_attempt_id"], source_archive_sha256=row["_source_archive_sha256"])
        if key in fine["remaining"]:
            if row["model_fingerprint"] != fine["pilot"]["model_fingerprint"]:
                raise ValueError("Fine result ModelSpec differs from the original admitted pilot")
            item["fine_manifest_sha256"] = fine["by_key"][key]
            fine_rows.append(item)
        else:
            if row["native_record_sha256"] != checkpoint["native_record_sha256"]:
                raise ValueError("A checkpoint-success native result changed")
            if row["_receipt_file_sha256"] != checkpoint["native_receipt_file_sha256"]:
                verified_cache_only_transition(private_root, row, checkpoint["native_receipt_file_sha256"])
                prefix_cache_transitions += 1
                item["receipt_snapshot_comparison"] = "VERIFIED_CACHE_FLAG_ONLY_TRANSITION"
            else:
                item["receipt_snapshot_comparison"] = "EXACT_CHECKPOINT_FILE_SHA"
            prefix_rows.append(item)
    pilot_matches = [row for row in by_key.values() if row["stage"] == "member_generation" and digest(list(row["task_key"])) == fine["pilot"]["task_key_sha256"]]
    pilot_status = "NOT_YET_EXPORTED"
    pilot_cache_transition = False
    if pilot_matches:
        row, pilot = pilot_matches[0], fine["pilot"]
        if any(row[field] != pilot[pilot_field] for field, pilot_field in (("native_record_sha256", "native_record_sha256"), ("native_job_id", "job_id"), ("physical_attempt_id", "physical_attempt_id"), ("model_fingerprint", "model_fingerprint"))):
            raise ValueError("Exported pilot differs from its independent native admission")
        for name, field in (("ATTEMPT_STATE.json", "_fence_file_sha256"), ("LAST_TASK_RECEIPT.json", "_receipt_file_sha256"), ("records.jsonl", "_native_file_sha256")):
            if row[field] != pilot["native_file_refs"][name]["sha256"]:
                if name != "LAST_TASK_RECEIPT.json":
                    raise ValueError("Pilot original native-file bytes differ")
                pilot_cache_transition = verified_cache_only_transition(private_root, row, pilot["native_file_refs"][name]["sha256"])
        pilot_status = (
            "EXACT_NATIVE_AND_FENCE_BYTES_WITH_VERIFIED_RECEIPT_CACHE_FLAG_TRANSITION"
            if pilot_cache_transition else "EXACT_THREE_NATIVE_FILES_AND_TASK_METADATA_BOUND"
        )
    complete = len(fine_rows) == len(fine["remaining"]) and len(prefix_rows) == fine["lease"]["checkpoint_successes"]
    if require_complete_fine and not complete:
        raise ValueError("Complete512 lineage requires every checkpoint and fine native archive")
    for reference in [bundle_ref, native_audit_ref, scope_ref, gate_source_ref, *sources]:
        if file_sha(reference["path"]) != reference["sha256"]:
            raise ValueError("An input source changed during provenance review")
    return dict(
        schema="e18_tail_export_lineage_audit_v1",
        status="PASS_COMPLETE512_SOURCE_AND_NATIVE_EXPORT_BINDING" if complete else "PASS_SOURCE_CHAIN_WITH_PARTIAL_NATIVE_EXPORT_COVERAGE",
        static_source_bundle_ref=bundle_ref,
        completed_native_audit_ref=native_audit_ref,
        scope_ref=scope_ref,
        frozen_cpu_gate_ref=gate_source_ref,
        fine_plan_ref=fine["plan_ref"], tail_parent_lease_ref=fine["lease_ref"],
        original_parent_manifest_ref=fine["parent_ref"],
        expected_original_parent_tasks=len(fine["parent_keys"]),
        expected_checkpoint_successes=fine["lease"]["checkpoint_successes"],
        expected_fine_tasks=len(fine["remaining"]),
        archived_unchanged_checkpoint_successes=len(prefix_rows),
        checkpoint_receipt_cache_flag_only_transitions=prefix_cache_transitions,
        archived_fine_native_tasks=len(fine_rows),
        missing_fine_tasks=len(fine["remaining"]) - len(fine_rows),
        fine_manifest_count=len(fine["children"]),
        pilot_export_binding=pilot_status,
        pilot_receipt_cache_flag_only_transition=pilot_cache_transition,
        planner_sidecar=dict(path=fine["sidecar_name"], sha256=fine["sidecar_sha256"]),
        separately_bound_operation_sources=fine["external_refs"],
        operation_lease_sources_embedded_by_original_exporter=False,
        archives_checked=len(archive_reports), all_native_tasks_cross_checked=len(tasks),
        unique_physical_attempts_cross_checked=len(attempts),
        archive_reports=archive_reports,
        checkpoint_task_metadata=prefix_rows,
        fine_task_metadata=fine_rows,
        final_cpu_gate_execution="PENDING_COMPLETE_GLOBAL_NATIVE_ADMISSION_BY_ROOT",
        model_or_scoring_calls=0, gold_values_decoded=0,
        fresh_input_text_values_decoded=0, native_answer_values_decoded=0,
        native_payload_json_decoded=0, files_extracted=0, signals_sent=0,
        additional_scientific_keys=0, original_scientific_sources_modified=0,
        review_source_sha256=file_sha(__file__),
        interpretation="The immutable operation bundle closes sources absent from the original export metadata traversal. The unchanged final CPU gate separately admits all native ModelSpec/job/owner/cost identities before Gold. This receipt reports current export coverage and carries no quality conclusion.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    for name in ("bundle", "native-audit", "scope", "gate-source"):
        parser.add_argument("--" + name + "-ref", nargs=2, required=True)
    parser.add_argument("--require-complete-fine", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Source audit receipts are append-only")
    reference = lambda pair: dict(path=pair[0], sha256=pair[1])
    started = time.perf_counter()
    result = audit(args.private_root, reference(args.bundle_ref), reference(args.native_audit_ref), reference(args.scope_ref), reference(args.gate_source_ref), require_complete_fine=args.require_complete_fine)
    result["audit_wall_seconds"] = time.perf_counter() - started
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({name: result[name] for name in ("status", "archives_checked", "archived_unchanged_checkpoint_successes", "archived_fine_native_tasks", "missing_fine_tasks", "pilot_export_binding")}))


if __name__ == "__main__":
    main()
