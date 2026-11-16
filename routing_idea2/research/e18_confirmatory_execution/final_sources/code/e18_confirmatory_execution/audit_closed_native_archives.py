"""Independent streaming audit of downloaded E18 closed native archives.

No exporter module is imported. Tar entries are read in memory, never extracted.
Only completed native generation outputs and frozen execution/model metadata
are accepted. Gold sources, scoring outputs and evaluation arrays are excluded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
import time


OWNER_FILES = {"ATTEMPT_STATE.json", "LAST_TASK_RECEIPT.json", "records.jsonl"}
STAGES = {"member_generation", "aggregation"}
TEAMS = tuple(combinations(range(6), 3))
MAX_ENTRY_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
CORE_SOURCE_SHAS = {
    "src/budget_router/inference.py": "3be35e1be1d30487e10b45fabbeae7254a5931c297359d9ef37adf80dd16d0ef",
    "src/budget_router/member_execution.py": "b1eac1cb346f7c67f14de398df130c644193c4f527a33b00ab18de1844f6dfb2",
    "src/budget_router/execution.py": "2dd0fc4bf9e36eb63e4ade5143d07505e14780e59338230ea86817ddd8eae17f",
    "scripts/v2/typed_aggregation_execution.py": "acbc9cd911635f26da0e23cce4174bf6660a7604d29a5d7224f1df08e083f368",
}
MODEL_DEFAULTS = dict(
    backend="transformers",
    model_path=None,
    device="cuda",
    dtype="float16",
    max_input_tokens=6144,
    max_new_tokens=256,
    temperature=0.0,
    top_p=1.0,
    base_url=None,
    api_key_env=None,
    timeout_s=300.0,
    quantization=None,
    enable_thinking=None,
)


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in archived evidence")
        result[key] = value
    return result


def strict_json(raw):
    def reject_constant(value):
        raise ValueError("Non-finite JSON number " + value)

    return json.loads(raw, object_pairs_hook=no_duplicate_keys, parse_constant=reject_constant)


def read_json(path):
    return strict_json(Path(path).read_bytes())


def model_fingerprint(config):
    unknown = set(config) - set(MODEL_DEFAULTS) - {"model_id", "revision"}
    if unknown or not config.get("model_id") or not config.get("revision"):
        raise ValueError("Unexpected frozen ModelSpec fields")
    complete = {**MODEL_DEFAULTS, **config}
    fingerprint_data = dict(complete)
    if fingerprint_data["enable_thinking"] is None:
        del fingerprint_data["enable_thinking"]
    return complete, digest(fingerprint_data)


class FrozenContext:
    def __init__(self, asset_root, scope_path, approval_path):
        self.asset_root = Path(asset_root)
        self.scope = read_json(scope_path)
        self.scope_file_sha = file_sha(scope_path)
        self.scope_sha = digest(self.scope)
        if (self.scope.get("schema"), self.scope.get("role"), self.scope.get("cohort")) != (
            "e18_scope_contract_v2",
            "execution",
            "e18_confirm",
        ):
            raise ValueError("The formal E18 execution scope is required")
        self.approval = read_json(approval_path)
        self.approval_sha = file_sha(approval_path)
        if (
            self.approval.get("status") != "APPROVED"
            or self.approval.get("software_test_only") is not False
            or self.approval.get("cohort") != "e18_confirm"
            or self.approval.get("scope_sha256_by_role", {}).get("execution") != self.scope_sha
            or any(
                self.approval["bindings"].get(key) != value
                for key, value in self.scope["bindings"].items()
            )
        ):
            raise ValueError("The approved execution metadata differs")
        self.owned_root = PurePosixPath(self.approval["approved_output_root"])
        self.identities = {
            row["question_id"]: row["canonical_question_sha256"]
            for row in self.scope["question_keys"]
        }
        self.allowed = {
            "member_generation": set(map(tuple, self.scope["required_member_keys"])),
            "aggregation": set(map(tuple, self.scope["required_aggregate_keys"])),
        }
        self.source_hashes = {}
        for relative, expected in CORE_SOURCE_SHAS.items():
            actual = file_sha(self.asset_root / relative)
            if actual != expected:
                raise ValueError("Frozen scientific source bytes differ")
            self.source_hashes[relative] = actual
        pool_relative = "configs/pool_v1.json"
        pool = read_json(self.asset_root / pool_relative)
        self.source_hashes[pool_relative] = file_sha(self.asset_root / pool_relative)
        if sorted(row["index"] for row in pool["models"]) != list(range(6)):
            raise ValueError("Exactly six frozen member profiles are required")
        self.profiles = {}
        self.aliases = {}
        for row in pool["models"]:
            config = read_json(self.asset_root / row["config"])
            profile, fingerprint = model_fingerprint(config)
            if fingerprint != row["fingerprint"]:
                raise ValueError("Independently computed member ModelSpec fingerprint differs")
            self.profiles["member_generation", row["index"]] = profile, fingerprint
            self.aliases[row["index"]] = row["alias"]
            self.source_hashes[row["config"]] = file_sha(self.asset_root / row["config"])
        relative = "configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json"
        profile, fingerprint = model_fingerprint(read_json(self.asset_root / relative))
        if fingerprint != "32aed6079298ebcf8eee99add17dd5c4a6828a504f13dd5d9236b234776861d3":
            raise ValueError("Frozen aggregator ModelSpec differs")
        self.profiles["aggregation", 0] = profile, fingerprint
        self.source_hashes[relative] = file_sha(self.asset_root / relative)


def audit_owner(context, stage, key_hash, files):
    fence = strict_json(files["ATTEMPT_STATE.json"])
    receipt = strict_json(files["LAST_TASK_RECEIPT.json"])
    rows = [strict_json(line) for line in files["records.jsonl"].split(b"\n") if line.strip()]
    if len(rows) != 1:
        raise ValueError("A closed owner does not have exactly one native attempt")
    native = rows[0]
    task = receipt.get("task_key", {})
    index_field = "member_index" if stage == "member_generation" else "combination_index"
    key = (task.get("question_id"), task.get("generation_seed"), task.get(index_field))
    if key not in context.allowed[stage] or digest(list(key)) != key_hash:
        raise ValueError("Archived task is outside its scope or hashed owner path")
    if task.get("canonical_id") != context.identities[key[0]]:
        raise ValueError("Canonical input identity differs")
    if (
        any(
            receipt.get(name) != value
            for name, value in dict(
                schema="e18_execution_task_receipt_v2",
                status="SUCCESS",
                scope_sha256=context.scope_sha,
                approval_sha256=context.approval_sha,
                stage=stage,
                software_test_only=False,
                cohort="e18_confirm",
                bindings=context.scope["bindings"],
                transport_split=context.scope["transport_split"],
            ).items()
        )
        or PurePosixPath(receipt.get("approved_output_root", "")) != context.owned_root
    ):
        raise ValueError("Native receipt scope, approval, ownership or mode differs")
    native_sha = digest({name: value for name, value in native.items() if name != "record_sha256"})
    if native.get("record_sha256") != native_sha:
        raise ValueError("Native row checksum is invalid")
    if native.get("status") != "success" or native.get("cost_status") != "recorded_actual_usage":
        raise ValueError("Native row is not a settled successful execution")
    if (
        native.get("question_id") != key[0]
        or native.get("split") != context.scope["transport_split"]
    ):
        raise ValueError("Native row question or execution transport differs")
    if receipt.get("native_record") != native or receipt.get("native_record_sha256") != native_sha:
        raise ValueError("Receipt embeds another native row")
    job_id, attempt_id = native.get("job_id"), native.get("attempt_id")
    if (
        not isinstance(job_id, str)
        or len(job_id) != 64
        or not isinstance(attempt_id, str)
        or not attempt_id
    ):
        raise ValueError("Native job or physical attempt identity is absent")
    if receipt.get("native_job_id") != job_id or receipt.get("physical_attempt_id") != attempt_id:
        raise ValueError("Receipt/native physical attempt does not match")
    expected_fence = dict(
        status="SUCCESS_COMMITTED",
        scope_sha256=context.scope_sha,
        approval_sha256=context.approval_sha,
        task_key=list(key),
        stage=stage,
        native_job_id=job_id,
        native_record_sha256=native_sha,
        software_test_only=False,
    )
    if any(fence.get(name) != value for name, value in expected_fence.items()):
        raise ValueError("Native fence is not the same fully committed attempt")
    profile, fingerprint = context.profiles[stage, key[2] if stage == "member_generation" else 0]
    result = native.get("result", {})
    seed = key[1] if stage == "member_generation" else 42
    if (
        native.get("model_fingerprint") != fingerprint
        or result.get("model_id") != profile["model_id"]
        or result.get("model_revision") != profile["revision"]
        or result.get("backend") != profile["backend"]
        or result.get("model_identity_verified") is not True
        or native.get("seed") != seed
        or result.get("seed") != seed
        or not isinstance(result.get("text"), str)
        or not result["text"].strip()
    ):
        raise ValueError("Frozen ModelSpec, generation seed or output identity differs")
    if stage == "member_generation":
        if (
            native.get("model_alias") != context.aliases[key[2]]
            or native.get("sample_index") != 0
            or native.get("sampling_control") is not False
            or native.get("prompt_version") != "independent_user_only_brief_final_v2"
        ):
            raise ValueError("Original member-generation contract differs")
    else:
        if (
            receipt.get("aggregator_fingerprint") != context.scope["operator_fingerprint"]
            or receipt.get("aggregation_seed") != 42
            or native.get("prompt_version") != "anonymous_final_answer_typed_v3"
            or task.get("team") != list(TEAMS[key[2]])
            or native.get("members") != sorted(context.aliases[index] for index in TEAMS[key[2]])
        ):
            raise ValueError("Frozen typed aggregation or semantic team differs")
    for field, maximum in (
        ("prompt_tokens", profile["max_input_tokens"]),
        ("completion_tokens", profile["max_new_tokens"]),
    ):
        value = result.get(field)
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError("Actual Token usage is absent or outside the frozen ModelSpec")
    wall = result.get("wall_time_s")
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
        raise ValueError("Actual native wall-clock cost is invalid")
    return dict(
        task_id=digest([context.scope_sha, stage, list(key)]),
        stage=stage,
        task_key=list(key),
        native_job_id=job_id,
        native_record_sha256=native_sha,
        physical_attempt_id=attempt_id,
        model_fingerprint=fingerprint,
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        wall_time_s=wall,
    )


def audit_archive(context, path, expected):
    """Read each three-file owner once and retain metadata, never raw outputs."""
    path = Path(path)
    before = path.stat()
    if before.st_size != expected["bytes"] or file_sha(path) != expected["sha256"]:
        raise ValueError("Downloaded archive external SHA/size differs")
    seen_names, file_records, task_records = set(), {}, {}
    current_prefix, current_files = None, {}
    manifest, manifest_raw_sha = None, None
    total_uncompressed = 0
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            name = member.name
            pure = PurePosixPath(name)
            if (
                not member.isfile()
                or pure.is_absolute()
                or ".." in pure.parts
                or "\\" in name
                or name in seen_names
                or not 0 <= member.size <= MAX_ENTRY_BYTES
            ):
                raise ValueError("Unsafe, duplicate or excessive tar member")
            if manifest is not None:
                raise ValueError("An archive has content after its closing manifest")
            seen_names.add(name)
            total_uncompressed += member.size
            if total_uncompressed > MAX_ARCHIVE_BYTES:
                raise ValueError("Closed archive exceeds its audit resource bound")
            checksum, chunks, length = hashlib.sha256(), [], 0
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
                    chunks.append(chunk)
                    length += len(chunk)
            if length != member.size:
                raise ValueError("Tar payload size is incomplete")
            raw = b"".join(chunks)
            if name == "EXPORT_MANIFEST.json":
                if current_files:
                    raise ValueError("A native owner is incomplete before the manifest")
                manifest = strict_json(raw)
                manifest_raw_sha = checksum.hexdigest()
                continue
            if (
                len(pure.parts) != 4
                or pure.parts[0] != context.scope_sha
                or pure.parts[1] not in STAGES
                or pure.name not in OWNER_FILES
            ):
                raise ValueError("Tar contains a non-native or out-of-scope file")
            prefix = pure.parts[:3]
            if current_prefix is None:
                current_prefix = prefix
            if prefix != current_prefix:
                raise ValueError("Native owner files are incomplete or noncontiguous")
            current_files[pure.name] = raw
            file_records[name] = dict(bytes=length, sha256=checksum.hexdigest())
            if set(current_files) == OWNER_FILES:
                task = audit_owner(context, prefix[1], prefix[2], current_files)
                if task["task_id"] in task_records:
                    raise ValueError("A closed archive repeats a native task")
                task_records[task["task_id"]] = task
                current_prefix, current_files = None, {}
    if manifest is None or current_files:
        raise ValueError("Closed archive lacks a complete final manifest")
    if path.name != "native-" + str(manifest.get("archive_id")) + ".tar.gz":
        raise ValueError("Archive filename and closed manifest identity differ")
    if (
        manifest.get("schema") != "e18_closed_native_export_manifest_v1"
        or manifest.get("scope_sha256") != context.scope_sha
        or manifest.get("scope_ref", {}).get("sha256") != context.scope_file_sha
        or manifest.get("scope_bindings") != context.scope["bindings"]
        or manifest.get("software_test_only") is not False
        or PurePosixPath(manifest.get("owned_root", "")) != context.owned_root
        or manifest.get("original_files_retained") is not True
        or any(
            manifest.get(field) != 0 for field in ("new_model_calls", "scorer_calls", "gold_reads")
        )
    ):
        raise ValueError("Export manifest scope, ownership or side-effect declarations differ")
    declared_files = {row["path"]: row for row in manifest["files"]}
    if len(declared_files) != len(manifest["files"]) or set(declared_files) != set(file_records):
        raise ValueError("Export manifest file inventory is duplicated or incomplete")
    for name, actual in file_records.items():
        if any(declared_files[name].get(field) != value for field, value in actual.items()):
            raise ValueError("Export manifest native file SHA/size differs")
    declared_tasks = {row["task_id"]: row for row in manifest["tasks"]}
    if len(declared_tasks) != len(manifest["tasks"]) or set(declared_tasks) != set(task_records):
        raise ValueError("Manifest task inventory is duplicated or inconsistent")
    for task_id, task in task_records.items():
        if any(
            declared_tasks[task_id].get(field) != value
            for field, value in task.items()
            if field != "wall_time_s"
        ):
            raise ValueError("Manifest task/attempt/ModelSpec/Token metadata differs")
    if len(file_records) != 3 * len(task_records) or not 1 <= len(task_records) <= 1000:
        raise ValueError("A closed archive does not have exactly three files per native task")
    if len(task_records) != expected["tasks"] or digest(manifest) != expected["manifest_sha256"]:
        raise ValueError("Closed archive task count or external manifest digest differs")
    if not set(context.source_hashes.values()) <= set(
        manifest["scientific_model_source_hashes"].values()
    ):
        raise ValueError("Manifest scientific ModelSpec/source fingerprints are incomplete")
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Downloaded archive changed during its audit")
    return dict(
        filename=path.name,
        archive_id=manifest["archive_id"],
        sha256=expected["sha256"],
        compressed_bytes=after.st_size,
        uncompressed_native_and_manifest_bytes=total_uncompressed,
        manifest_raw_sha256=manifest_raw_sha,
        manifest_semantic_sha256=digest(manifest),
        native_tasks=len(task_records),
        native_files=len(file_records),
        tasks=list(task_records.values()),
    )


def audit_all(context, source_result_path, archive_directory, expected_archives, expected_tasks):
    source = read_json(source_result_path)
    entries = source["new_archives"]
    if len(entries) != expected_archives or sum(row["tasks"] for row in entries) != expected_tasks:
        raise ValueError("Requested closed-export cohort differs from its external inventory")
    seen_tasks, seen_attempts, seen_archives, questions = set(), set(), set(), set()
    reports, stages, profiles, seeds = [], Counter(), Counter(), Counter()
    prompt_tokens = completion_tokens = 0
    stage_costs = {
        stage: dict(tasks=0, prompt_tokens=0, completion_tokens=0, native_wall_seconds=0.0)
        for stage in STAGES
    }
    for entry in entries:
        name = PurePosixPath(entry["path"]).name
        if name in seen_archives or not name.startswith("native-") or not name.endswith(".tar.gz"):
            raise ValueError("External inventory contains duplicate or unexpected archive paths")
        seen_archives.add(name)
        result = audit_archive(context, Path(archive_directory) / name, entry)
        for task in result.pop("tasks"):
            if task["task_id"] in seen_tasks or task["physical_attempt_id"] in seen_attempts:
                raise ValueError("Cross-archive native task or physical attempt is duplicated")
            seen_tasks.add(task["task_id"])
            seen_attempts.add(task["physical_attempt_id"])
            qid, generation, action = task["task_key"]
            questions.add(qid)
            stages[task["stage"]] += 1
            profiles[
                task["stage"],
                action if task["stage"] == "member_generation" else "Qwen3_aggregator",
            ] += 1
            seeds[task["stage"], generation] += 1
            prompt_tokens += task["prompt_tokens"]
            completion_tokens += task["completion_tokens"]
            cost = stage_costs[task["stage"]]
            cost["tasks"] += 1
            cost["prompt_tokens"] += task["prompt_tokens"]
            cost["completion_tokens"] += task["completion_tokens"]
            cost["native_wall_seconds"] += task["wall_time_s"]
        reports.append(result)
    return dict(
        schema="e18_independent_closed_native_archive_audit_v1",
        status="PASS_INDEPENDENT_STREAMING_NATIVE_AUDIT",
        archives_checked=len(reports),
        native_tasks_checked=len(seen_tasks),
        unique_physical_attempts=len(seen_attempts),
        native_files_checked=sum(row["native_files"] for row in reports),
        compressed_bytes=sum(row["compressed_bytes"] for row in reports),
        by_stage=dict(stages),
        by_profile=[
            dict(stage=key[0], member_index_or_type=key[1], tasks=count)
            for key, count in sorted(profiles.items(), key=str)
        ],
        by_member_generation_seed=[
            dict(stage=key[0], member_generation_seed=key[1], tasks=count)
            for key, count in sorted(seeds.items())
        ],
        distinct_question_clusters=len(questions),
        actual_prompt_tokens=prompt_tokens,
        actual_completion_tokens=completion_tokens,
        actual_total_tokens=prompt_tokens + completion_tokens,
        stage_costs=stage_costs,
        archives=reports,
        execution_scope_file_sha256=context.scope_file_sha,
        execution_scope_sha256=context.scope_sha,
        source_result_sha256=file_sha(source_result_path),
        frozen_model_metadata_sources=context.source_hashes,
        native_task_identity_set_sha256=digest(sorted(seen_tasks)),
        physical_attempt_identity_set_sha256=digest(sorted(seen_attempts)),
        original_files_modified=0,
        archive_members_extracted=0,
        remote_files_deleted=0,
        model_or_scorer_calls=0,
        gold_or_reward_sources_read=0,
        evaluation_results_read=0,
        interpretation=(
            "Task and generation counts describe executed records. Multiple generations of one "
            "question remain one question cluster; equal seed labels do not establish independent "
            "or strictly paired random experiments."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--archive-directory", type=Path, required=True)
    parser.add_argument("--expected-archives", type=int, default=12)
    parser.add_argument("--expected-tasks", type=int, default=5814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit outputs are append-only")
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    context = FrozenContext(args.asset_root, args.scope, args.approval)
    result = audit_all(
        context,
        args.source_result,
        args.archive_directory,
        args.expected_archives,
        args.expected_tasks,
    )
    result["audit_source_sha256"] = file_sha(Path(__file__))
    result["audit_cpu_seconds"] = time.process_time() - started_cpu
    result["audit_wall_seconds"] = time.perf_counter() - started_wall
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "archives_checked",
                    "native_tasks_checked",
                    "native_files_checked",
                    "compressed_bytes",
                    "by_stage",
                    "distinct_question_clusters",
                    "actual_prompt_tokens",
                    "actual_completion_tokens",
                    "actual_total_tokens",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
