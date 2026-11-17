"""Prepare disjoint continuation subsets for one frozen member profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from p2_manifest_registry import digest, iter_successes, load_registry, read_json, reference
from prepare_aggregation_shards_v3 import write_exact


def profile_required(registry, member_index):
    if type(member_index) is not int or member_index not in range(6):
        raise ValueError("A frozen integer member index from zero through five is required")
    scope = read_json(registry["common"]["scope_ref"]["path"])
    return {tuple(key) for key in scope["required_member_keys"] if key[2] == member_index}


def partition_remaining(required, completed, shard_count=4):
    if type(shard_count) is not int or shard_count not in range(2, 9):
        raise ValueError("An integer shard count from two through eight is required")
    required, completed = set(map(tuple, required)), set(map(tuple, completed))
    if not completed <= required:
        raise ValueError("Completed member keys exceed the frozen profile")
    shards = [[] for _ in range(shard_count)]
    for key in sorted(required - completed):
        shards[int(digest(list(key[:2])), 16) % shard_count].append(key)
    seen = set(completed)
    for rows in shards:
        if seen.intersection(rows):
            raise AssertionError("Member subsets overlap")
        seen.update(rows)
    if seen != required:
        raise AssertionError("Completed and remaining member subsets fail exact coverage")
    return shards


def prepare(registry_path, output_dir, worker_source_ref, member_index=4, shard_count=4):
    registry = load_registry(registry_path)
    if reference(worker_source_ref["path"]) != worker_source_ref:
        raise ValueError("The original member worker source changed")
    required = profile_required(registry, member_index)
    original = [entry for entry in registry["entries"] if entry["stage"] == "member_generation"
                and entry["member_index"] == member_index and set(entry["task_keys"]) == required]
    if len(original) != 1 or original[0]["worker_source_ref"] != worker_source_ref:
        raise ValueError("The exact original full-profile manifest and worker source are required")
    base = original[0]
    completed = {key for _, _, key in iter_successes(registry, "member_generation") if key[2] == member_index}
    shards = partition_remaining(required, completed, shard_count)
    completed_sha = digest([list(key) for key in sorted(completed)])
    plan_id = digest(dict(scope=registry["scope_canonical_sha256"], registry_head=registry["last_event_sha256"],
                          completed=completed_sha, worker_source=worker_source_ref,
                          member_index=member_index, shard_count=shard_count))[:20]
    folder = Path(output_dir).resolve() / plan_id
    completed_ref = write_exact(folder / "COMPLETED_MEMBER_KEYS.json", [list(key) for key in sorted(completed)])
    refs = []
    for index, keys in enumerate(shards):
        worker_id = f"member{member_index}-parallel-{plan_id}-shard{index}"
        keys_ref = write_exact(folder / f"shard{index}_keys.json", [list(key) for key in keys])
        manifest = dict(base["manifest"], worker_id=worker_id, execution_plan_id=plan_id,
                        task_keys_ref=keys_ref,
                        gpu_lock_root=str(Path(registry["common"]["owned_output_root"]).parent / "gpu-locks" / worker_id))
        manifest_ref = write_exact(folder / f"shard{index}_manifest.json", manifest)
        entry = dict(worker_id=worker_id, manifest_ref=manifest_ref, worker_source_ref=worker_source_ref)
        entry_ref = write_exact(folder / f"shard{index}_registry_addition.json", entry)
        refs.append(dict(entry, keys_ref=keys_ref, registry_addition_ref=entry_ref, task_count=len(keys)))
    plan = dict(schema="p2_disjoint_member_dispatch_plan_v1", plan_id=plan_id,
                software_test_only=registry["software_test_only"], common=registry["common"], member_index=member_index,
                source_registry_head=registry["last_event_sha256"], base_manifest_ref=base["manifest_ref"],
                required_member_calls=len(required), completed_at_preparation=len(completed),
                completed_keys_ref=completed_ref, completed_key_set_sha256=completed_sha,
                shard_count=shard_count, shards=refs,
                partition_rule=f"SHA256([question_id,generation_seed]) modulo{shard_count}",
                changes=dict(scope=0, approval=0, owned_output_root=0, model_spec=0, policy=0, generation_parameters=0),
                reference_values_decoded=0, production_processes_started=0)
    return dict(plan_ref=write_exact(folder / "DISPATCH_PLAN.json", plan), plan=plan)


def validate_dispatch_plan(registry, plan):
    if (plan.get("schema") != "p2_disjoint_member_dispatch_plan_v1"
            or plan.get("common") != registry["common"]
            or plan.get("software_test_only") is not registry["software_test_only"]
            or plan.get("source_registry_head") not in registry["event_hashes"]):
        raise ValueError("The member plan differs from the registered execution identity")
    required = profile_required(registry, plan["member_index"])
    base_entries = [entry for entry in registry["entries"] if entry["manifest_ref"] == plan["base_manifest_ref"]]
    if (len(base_entries) != 1 or base_entries[0]["stage"] != "member_generation"
            or base_entries[0]["member_index"] != plan["member_index"]
            or set(base_entries[0]["task_keys"]) != required
            or reference(plan["base_manifest_ref"]["path"]) != plan["base_manifest_ref"]):
        raise ValueError("The original full member source changed")
    base = base_entries[0]
    completed_rows = list(map(tuple, read_json(plan["completed_keys_ref"]["path"])))
    completed = set(completed_rows)
    if (len(completed_rows) != len(completed)
            or reference(plan["completed_keys_ref"]["path"]) != plan["completed_keys_ref"]
            or digest([list(key) for key in sorted(completed)]) != plan["completed_key_set_sha256"]):
        raise ValueError("Completed member snapshot changed")
    count = plan.get("shard_count")
    if type(count) is not int or count not in range(2, 9) or len(plan["shards"]) != count:
        raise ValueError("Explicit member subset count differs")
    if len({item["worker_id"] for item in plan["shards"]}) != count:
        raise ValueError("Member worker identities must be unique")
    seen = set(completed)
    sizes = []
    permitted = {"worker_id", "execution_plan_id", "task_keys_ref", "gpu_lock_root"}
    for item in plan["shards"]:
        for name in ("manifest_ref", "keys_ref", "worker_source_ref"):
            if reference(item[name]["path"]) != item[name]:
                raise ValueError("A member subset reference changed")
        if item["worker_source_ref"] != base["worker_source_ref"]:
            raise ValueError("The original member executable must be reused")
        manifest = read_json(item["manifest_ref"]["path"])
        if (manifest.get("worker_id") != item["worker_id"]
                or manifest.get("execution_plan_id") != plan["plan_id"]
                or manifest.get("task_keys_ref") != item["keys_ref"]
                or {key: value for key, value in manifest.items() if key not in permitted}
                != {key: value for key, value in base["manifest"].items() if key not in permitted}):
            raise ValueError("A member subset changed its frozen scientific payload")
        rows = list(map(tuple, read_json(item["keys_ref"]["path"])))
        if len(rows) != len(set(rows)) or seen.intersection(rows):
            raise ValueError("Repeated or overlapping member subset keys")
        seen.update(rows)
        sizes.append(len(rows))
    if seen != required:
        raise ValueError("Member subsets do not partition the original required profile")
    successor_shas = {item["manifest_ref"]["sha256"] for item in plan["shards"]}
    historical = {key for entry, _, key in iter_successes(registry, "member_generation")
                  if key[2] == plan["member_index"] and entry["manifest_ref"]["sha256"] not in successor_shas}
    if historical != completed:
        raise RuntimeError("Historical member outputs advanced after the prepared snapshot")
    return dict(status="DISJOINT_MEMBER_DISPATCH_QUALIFIED", member_index=plan["member_index"],
                completed=len(completed), remaining=sizes, required=len(required))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-source", type=Path, required=True)
    parser.add_argument("--member-index", type=int, default=4)
    parser.add_argument("--shards", type=int, choices=range(2, 9), default=4)
    args = parser.parse_args()
    result = prepare(args.registry, args.output_dir, reference(args.worker_source), args.member_index, args.shards)
    print(json.dumps(dict(plan_ref=result["plan_ref"], completed=result["plan"]["completed_at_preparation"],
                          shard_counts=[item["task_count"] for item in result["plan"]["shards"]], processes_started=0)))
