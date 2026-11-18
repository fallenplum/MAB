"""Prepare the approved E23 missing-call manifests on the inference server."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import supplement_adapter as adapter


PARENT_ROOT = Path("/data2/chendong/e20_p2_execution")
EXPECTED_COUNTS = dict(questions=840, member_generation=1770, aggregation=4263, dependencies=9570)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError("An immutable supplementary artifact already differs")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    return dict(path=str(path), sha256=sha_bytes(data))


def load_parent():
    sys.path.insert(0, str(PARENT_ROOT / "code"))
    from p2_manifest_registry import iter_successes, load_registry

    registry = load_registry(PARENT_ROOT / "control/manifest_registry_v3.jsonl")
    if registry["scope_canonical_sha256"] != adapter.PARENT_SCOPE_SHA:
        raise ValueError("The supplementary parent execution scope differs")
    scope = read(registry["common"]["scope_ref"]["path"])
    required = {
        "member_generation": set(map(tuple, scope["required_member_keys"])),
        "aggregation": set(map(tuple, scope["required_aggregate_keys"])),
    }
    completed = {
        stage: {key for _, _, key in iter_successes(registry, stage)}
        for stage in required
    }
    if completed != required:
        raise ValueError("The original P2 required outputs are not completely committed")
    return registry, scope, completed


def cached_dependencies(parent, scope, e23_adapter):
    owner_root = Path(parent["common"]["owned_output_root"])
    parent_hash = adapter.PARENT_SCOPE_SHA
    references = []
    for key_values in scope["cached_member_dependency_keys"]:
        key = tuple(key_values)
        folder = owner_root / parent_hash / "member_generation" / adapter.canonical_digest(list(key))
        if folder.is_symlink() or not folder.resolve(strict=True).is_relative_to(owner_root.resolve(strict=True)):
            raise ValueError("Original member dependency owner escaped its parent root")
        receipt = read(folder / "LAST_TASK_RECEIPT.json")
        fence = read(folder / "ATTEMPT_STATE.json")
        source = folder / "records.jsonl"
        if source.is_symlink():
            raise ValueError("A parent native record cannot be a symbolic link")
        raw = source.read_bytes()
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        native = receipt.get("native_record")
        if (len(records) != 1 or records[0] != native
                or receipt.get("schema") != "p2_execution_task_receipt_v1"
                or receipt.get("cohort") != "e20_p2_confirm"
                or receipt.get("scope_sha256") != parent_hash
                or receipt.get("status") != "SUCCESS"
                or receipt.get("software_test_only") is not False
                or fence.get("status") != "SUCCESS_COMMITTED"
                or fence.get("scope_sha256") != parent_hash
                or fence.get("stage") != "member_generation"
                or tuple(fence.get("task_key", ())) != key
                or fence.get("native_record_sha256") != receipt.get("native_record_sha256")):
            raise ValueError("The parent committed member identity or fence differs")
        e23_adapter.checked_member_dependency(*key, native)
        references.append(dict(task_key=list(key), record_ref=dict(path=str(source), sha256=sha_bytes(raw))))
    return references


def prepare(root):
    root = Path(root).resolve()
    if root != Path("/data2/chendong/e23_reference_completion"):
        raise ValueError("The approved E23 execution root differs")
    inputs = root / "inputs"
    input_path, policy_path = inputs / "INPUTS.jsonl", inputs / "POLICIES.json"
    missing_path, origin_path = inputs / "MISSING.json", inputs / "ORIGIN.json"
    origin = read(origin_path)
    records = [json.loads(line) for line in input_path.read_bytes().splitlines() if line.strip()]
    policies, missing = read(policy_path), read(missing_path)
    member_keys = set(map(tuple, missing["missing_member_keys"]))
    aggregate_keys = set(map(tuple, missing["missing_aggregate_keys"]))
    if (len(records) != EXPECTED_COUNTS["questions"]
            or len(policies) != 12
            or len(member_keys) != EXPECTED_COUNTS["member_generation"]
            or len(aggregate_keys) != EXPECTED_COUNTS["aggregation"]
            or len(member_keys) != len(missing["missing_member_keys"])
            or len(aggregate_keys) != len(missing["missing_aggregate_keys"])):
        raise ValueError("Seed bundle differs from the approved exact supplementary call counts")
    registry, parent_scope, completed = load_parent()
    parent_inputs = {row["question_id"]: row for row in parent_scope["question_keys"]}
    for row in records:
        parent = parent_inputs.get(row["question_id"])
        if (parent is None or parent["canonical_question_sha256"] != row["canonical_question_sha256"]
                or parent["question_sha256"] != sha_bytes(row["question"].encode())
                or parent["task"] != row["task"]
                or parent["subtask"] != (row.get("subtask") or "")):
            raise ValueError("Supplementary inputs differ from the original P2 question pool")
    if member_keys & completed["member_generation"] or aggregate_keys & completed["aggregation"]:
        raise ValueError("An approved supplementary paid key already has a committed P2 result")
    dependencies = {
        (question, seed, member)
        for question, seed, action in aggregate_keys for member in adapter.TEAMS[action]
    }
    if (len(dependencies) != EXPECTED_COUNTS["dependencies"]
            or dependencies - completed["member_generation"] != member_keys):
        raise ValueError("Missing member keys do not equal the uncached aggregate dependencies")
    bindings = {name: origin[name] for name in (
        "protocol_sha256", "frozen_actions_sha256", "budget_sha256", "cache_evidence_sha256")}
    bindings.update(
        target_inputs_sha256=file_sha(input_path),
        target_ids_sha256=adapter.canonical_digest([
            [row["question_id"], row["canonical_question_sha256"]] for row in records]),
        parent_scope_sha256=adapter.PARENT_SCOPE_SHA,
        deployed_policies_sha256=file_sha(policy_path),
        missing_keys_sha256=file_sha(missing_path),
        origin_receipt_sha256=file_sha(origin_path),
    )
    if origin["parent_scope_sha256"] != adapter.PARENT_SCOPE_SHA:
        raise ValueError("The origin receipt identifies a different parent scope")
    scope = adapter.build_scope_contract(
        records, policies, bindings, sorted(member_keys), sorted(aggregate_keys))
    scope_ref = freeze(inputs / "EXECUTION_SCOPE.json", scope)
    approval = dict(
        schema=adapter.APPROVAL_SCHEMA, status="APPROVED", cohort=adapter.COHORT,
        software_test_only=False, operations=["member_generation", "aggregation"],
        scope_sha256_by_role={"execution": adapter.canonical_digest(scope)},
        bindings=bindings, approved_output_root=str(root / "owned"),
        execution_limits=scope["execution_limits"],
        authorization="User explicitly approved completing only the frozen 1770 member and 4263 aggregate missing calls",
        sealed200_answer_access=False, new_question_acquisition=False,
    )
    approval_ref = freeze(inputs / "AUTHORIZATION.json", approval)
    adapter_ref = dict(path=str(root / "code/supplement_adapter.py"),
                       sha256=file_sha(root / "code/supplement_adapter.py"))
    common = dict(
        software_test_only=False,
        asset_root=registry["common"]["asset_root"],
        adapter_ref=adapter_ref, scope_ref=scope_ref,
        question_records_ref=dict(path=str(input_path), sha256=file_sha(input_path)),
        approval_ref=approval_ref, owned_output_root=str(root / "owned"),
        install_signal_handlers=True,
    )
    apis = adapter.load_original_apis(common["asset_root"])
    pool = read(apis.root / "configs/pool_v1.json")["models"]
    member_specs = {row["index"]: apis.inference.ModelSpec(
        **read(apis.root / row["config"])) for row in pool}
    aggregator_spec = apis.inference.ModelSpec(
        **read(apis.root / "configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json"))
    e23_adapter = adapter.P2ExecutionAdapter(apis, scope, records, member_specs, aggregator_spec)
    external = cached_dependencies(registry, scope, e23_adapter)
    if len(external) != EXPECTED_COUNTS["dependencies"] - EXPECTED_COUNTS["member_generation"]:
        raise ValueError("Parent cached member dependency coverage is incomplete")
    external_ref = freeze(inputs / "MEMBER_DEPENDENCIES.json", external)
    manifests = []
    identity = adapter.canonical_digest(scope)[:20]
    for member in range(6):
        keys = sorted(key for key in member_keys if key[2] == member)
        if not keys:
            continue
        template = next(row for row in registry["entries"]
                        if row["stage"] == "member_generation" and row["member_index"] == member)
        shard_count = 2 if member == 4 else 1
        subsets = [[] for _ in range(shard_count)]
        for key in keys:
            slot = int(adapter.canonical_digest([key[0], key[1]]), 16) % shard_count
            subsets[slot].append(key)
        for shard, subset in enumerate(subsets):
            if not subset:
                continue
            suffix = f"member{member}" if shard_count == 1 else f"member{member}_shard{shard}"
            worker_id = f"e23-{identity}-{suffix}"
            key_ref = freeze(inputs / f"{suffix}_tasks.json", [list(key) for key in subset])
            manifest = dict(common, schema="e18_gpu_worker_manifest_v1", worker_id=worker_id,
                            stage="member_generation", member_index=member, task_keys_ref=key_ref,
                            backend=template["manifest"]["backend"],
                            gpu_lock_root=str(root / "gpu-locks" / worker_id))
            reference = freeze(root / "manifests" / f"{suffix}.json", manifest)
            manifests.append(dict(worker_id=worker_id, stage="member_generation",
                                  profile=member, shard=shard, task_count=len(subset), manifest_ref=reference))
    template = next(row for row in registry["entries"] if row["worker_id"] == "aggregation-v2")
    shards = [[] for _ in range(8)]
    for key in sorted(aggregate_keys):
        shard = int(adapter.canonical_digest([key[0], key[1]]), 16) % 8
        shards[shard].append(key)
    for shard, keys in enumerate(shards):
        if not keys:
            continue
        worker_id = f"e23-{identity}-aggregation{shard}"
        key_ref = freeze(inputs / f"aggregation{shard}_tasks.json", [list(key) for key in keys])
        manifest = dict(common, schema="e18_aggregation_worker_manifest_v3", worker_id=worker_id,
                        stage="aggregation", task_keys_ref=key_ref, member_dependencies_ref=external_ref,
                        backend=template["manifest"]["backend"],
                        gpu_lock_root=str(root / "gpu-locks" / worker_id))
        reference = freeze(root / "manifests" / f"aggregation{shard}.json", manifest)
        manifests.append(dict(worker_id=worker_id, stage="aggregation", profile=shard,
                              task_count=len(keys), manifest_ref=reference))
    plan = dict(
        schema="e23_supplement_execution_plan_v1", status="READY_NOT_LAUNCHED",
        cohort=adapter.COHORT, root=str(root), scope_ref=scope_ref,
        scope_canonical_sha256=adapter.canonical_digest(scope), approval_ref=approval_ref,
        input_questions=len(records), required_new_member_calls=len(member_keys),
        required_new_aggregate_calls=len(aggregate_keys),
        cached_member_dependency_records=len(external),
        member_dependency_ref=external_ref, manifests=manifests,
        original_parent_scope_sha256=adapter.PARENT_SCOPE_SHA,
        original_parent_outputs_modified=0,
        source_sha256={path.name: file_sha(path) for path in sorted((root / "code").glob("*.py"))},
        member_token_cap=6144, aggregate_token_cap=8192,
        conservative_incremental_token_cap=len(member_keys)*6144+len(aggregate_keys)*8192,
        physical_calls_launched=0, reference_values_decoded=0,
    )
    plan_ref = freeze(root / "EXECUTION_PLAN.json", plan)
    return dict(status=plan["status"], plan_ref=plan_ref, scope_sha256=plan["scope_canonical_sha256"],
                new_member_calls=len(member_keys), new_aggregate_calls=len(aggregate_keys),
                cached_member_dependencies=len(external), member_workers=sum(row["stage"] == "member_generation" for row in manifests),
                aggregate_workers=sum(row["stage"] == "aggregation" for row in manifests),
                physical_calls_launched=0, reference_values_decoded=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root)))
