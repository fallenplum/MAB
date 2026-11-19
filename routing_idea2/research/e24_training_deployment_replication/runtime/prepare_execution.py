"""Freeze E24's exact native task scopes and persistent-worker manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import supplement_adapter as adapter


ROOT = Path("/data2/chendong/e24_training_deployment_replication")
P2_ROOT = Path("/data2/chendong/e20_p2_execution")
USER_AUTHORIZATION = "继续开始实验验证啊"
INPUT_FIELDS = {
    "question", "question_id", "task", "subtask", "source_split", "source_frame", "source_file",
    "source_index", "source_revision", "canonical_question_sha256",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def freeze(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError("A previously frozen E24 artifact differs")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def parse_member_shards(raw):
    values = [int(value) for value in raw.split(",")]
    if len(values) != 6 or not all(1 <= value <= 8 for value in values):
        raise ValueError("Use six comma-separated member shard counts, each from 1 to 8")
    return values


def input_scope(directory):
    input_path = directory / "INPUTS.jsonl"
    policy_path = directory / "POLICIES.json"
    keys_path = directory / "EXECUTION_KEYS.json"
    origin_path = directory / "ORIGIN.json"
    origin = load(origin_path)
    if (origin.get("authorization") != USER_AUTHORIZATION
            or origin.get("input_sha256") != sha(input_path)
            or origin.get("policies_sha256") != sha(policy_path)
            or origin.get("execution_keys_sha256") != sha(keys_path)):
        raise PermissionError("The approved input-only E24 bundle was not transported intact")
    records = [json.loads(line) for line in input_path.read_bytes().splitlines()]
    if any(set(row) != INPUT_FIELDS for row in records):
        raise ValueError("The E24 inference input contains unexpected fields")
    policies = load(policy_path)
    execution = load(keys_path)
    if execution.get("prediction_sha256") != origin["prediction_sha256"] or execution.get("from_zero") is not True:
        raise ValueError("The E24 execution keys must come from the frozen from-zero budget")
    bindings = dict(
        protocol_sha256=origin["protocol_sha256"],
        budget_sha256=origin["budget_sha256"],
        frozen_actions_sha256=origin["prediction_sha256"],
        target_inputs_sha256=origin["input_sha256"],
        target_ids_sha256=adapter.canonical_digest([
            [row["question_id"], row["canonical_question_sha256"]] for row in records]),
        deployed_policies_sha256=origin["policies_sha256"],
        execution_keys_sha256=origin["execution_keys_sha256"],
        origin_receipt_sha256=sha(origin_path),
        reference_plan_sha256=origin["reference_plan_sha256"],
        parent_scope_sha256=adapter.PARENT_SCOPE_SHA,
    )
    scope = adapter.build_scope_contract(records, policies, bindings,
                                           execution["missing_member_keys"], execution["missing_aggregate_keys"])
    return scope, records, ref(input_path)


def parent_profiles():
    """Read the P2 backend mapping only. E24 reuses no prior question output."""
    sys.path.insert(0, str(P2_ROOT / "code"))
    from p2_manifest_registry import load_registry

    registry = load_registry(P2_ROOT / "control/manifest_registry_v3.jsonl")
    if registry["scope_canonical_sha256"] != adapter.PARENT_SCOPE_SHA:
        raise ValueError("Original P2 backend template scope differs")
    backends = registry["profile_backends"]
    if set(backends) != {f"member{index}" for index in range(6)} | {"aggregation"}:
        raise ValueError("Exactly the original six member and one aggregation backend mappings are required")
    templates = {index: backends[f"member{index}"] for index in range(6)}
    templates["aggregation"] = backends["aggregation"]
    return Path(registry["common"]["asset_root"]), templates


def shards(keys, count):
    """Keep all actions for one question and seed together within a stage."""
    if not 1 <= count <= 16:
        raise ValueError("Worker shard count is outside the supported range")
    result = [[] for _ in range(count)]
    for key in keys:
        slot = int(adapter.canonical_digest(key[:2]), 16) % count
        result[slot].append(key)
    if any(not part for part in result):
        raise ValueError("One requested shard is empty")
    return result


def build_plan(root, scope, input_ref, asset_root, templates, member_shards, aggregate_shards,
               *, software_test_only=False):
    scope_hash = adapter.canonical_digest(scope)
    scope_ref = freeze(root / "inputs/EXECUTION_SCOPE.json", scope)
    owned = root / "owned"
    approval = dict(schema=adapter.APPROVAL_SCHEMA, cohort=adapter.COHORT, status="APPROVED",
                    bindings=scope["bindings"], scope_sha256_by_role={"execution": scope_hash},
                    software_test_only=software_test_only, operations=list(adapter.COUNTS), approved_output_root=str(owned),
                    execution_limits=adapter.COUNTS, sealed200_answer_access=False,
                    authorization=USER_AUTHORIZATION)
    approval_ref = freeze(root / "inputs/AUTHORIZATION.json", approval)
    empty_external = freeze(root / "inputs/MEMBER_DEPENDENCIES.json", [])
    common = dict(software_test_only=software_test_only, asset_root=str(asset_root),
                  adapter_ref=ref(root / "code/supplement_adapter.py"), scope_ref=scope_ref,
                  question_records_ref=input_ref, approval_ref=approval_ref,
                  owned_output_root=str(owned), install_signal_handlers=True)
    manifests = []

    def add(name, stage, profile, index, keys, backend, **specific):
        identity = f"e24-{scope_hash[:20]}-{name}"
        task_ref = freeze(root / "inputs" / f"{name}_tasks.json", keys)
        manifest = dict(common, worker_id=identity, stage=stage, task_keys_ref=task_ref, backend=backend,
                        gpu_lock_root=str(root / "gpu-locks" / identity), **specific)
        manifest_ref = freeze(root / "manifests" / f"{name}.json", manifest)
        manifests.append(dict(worker_id=identity, stage=stage, profile=profile, shard=index,
                              task_count=len(keys), manifest_ref=manifest_ref))

    for member in range(6):
        own = [key for key in scope["member_execution_keys"] if key[2] == member]
        count = member_shards[member]
        for index, keys in enumerate(shards(own, count)):
            name = f"member{member}" if count == 1 else f"member{member}_shard{index}"
            add(name, "member_generation", member, index, keys, templates[member],
                schema="e18_gpu_worker_manifest_v1", member_index=member)
    for index, keys in enumerate(shards(scope["aggregate_execution_keys"], aggregate_shards)):
        add(f"aggregation{index}", "aggregation", index, index, keys, templates["aggregation"],
            schema="e18_aggregation_worker_manifest_v3", member_dependencies_ref=empty_external)
    plan = dict(schema="e24_independent_execution_plan_v1", status="READY_NOT_LAUNCHED",
                cohort=adapter.COHORT, root=str(root), scope_ref=scope_ref, approval_ref=approval_ref,
                scope_canonical_sha256=scope_hash, input_questions=len(scope["question_keys"]),
                required_new_member_calls=adapter.COUNTS["member_generation"],
                required_new_aggregate_calls=adapter.COUNTS["aggregation"],
                cached_member_dependency_records=0, member_dependency_ref=empty_external, manifests=manifests,
                source_sha256={path.name: sha(path) for path in sorted((root / "code").glob("*.py"))},
                original_parent_scope_sha256=adapter.PARENT_SCOPE_SHA,
                member_token_cap=6144, aggregate_token_cap=8192,
                conservative_incremental_token_cap=566532096, physical_calls_launched=0, reference_values_decoded=0)
    plan_ref = freeze(root / "EXECUTION_PLAN.json", plan)
    return plan, plan_ref


def prepare(root, member_shards, aggregate_shards):
    root = root.resolve()
    if root != ROOT:
        raise ValueError("The E24 output root differs from the independent execution root")
    scope, records, input_ref = input_scope(root / "inputs")
    asset_root, templates = parent_profiles()
    apis = adapter.load_original_apis(asset_root)
    pool = load(asset_root / "configs/pool_v1.json")["models"]
    members = {row["index"]: apis.inference.ModelSpec(**load(asset_root / row["config"])) for row in pool}
    aggregator = apis.inference.ModelSpec(**load(asset_root / "configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json"))
    adapter.P2ExecutionAdapter(apis, scope, records, members, aggregator)
    plan, plan_ref = build_plan(root, scope, input_ref, asset_root, templates, member_shards, aggregate_shards)
    return dict(status=plan["status"], plan_ref=plan_ref, scope_ref=plan["scope_ref"],
                approval_ref=plan["approval_ref"], scope_canonical_sha256=plan["scope_canonical_sha256"],
                questions=plan["input_questions"], execution_limits=adapter.COUNTS,
                cached_member_dependency_records=0, physical_calls_launched=0, reference_values_decoded=0,
                workers=[dict(name=Path(row["manifest_ref"]["path"]).stem, stage=row["stage"],
                              profile=row["profile"], tasks=row["task_count"]) for row in plan["manifests"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--member-shards", type=parse_member_shards, default=parse_member_shards("2,2,2,2,4,2"))
    parser.add_argument("--aggregate-shards", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.member_shards, args.aggregate_shards), ensure_ascii=False))
