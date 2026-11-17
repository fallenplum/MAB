"""Prepare two through eight disjoint subsets of original remaining aggregate keys."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from p2_manifest_registry import digest,iter_successes,load_registry,read_json,reference
from prepare_aggregation_shards_v3 import write_exact


def partition_remaining(required,completed,shard_count=5):
    if type(shard_count) is not int or shard_count not in range(2,9):
        raise ValueError("An explicit integer shard count from two through eight is required")
    required=set(map(tuple,required))
    completed=set(map(tuple,completed))
    if not completed<=required:
        raise ValueError("Completed keys exceed the original aggregation support")
    shards=[[] for _ in range(shard_count)]
    for key in sorted(required-completed):
        slot=int(digest([key[0],key[1]]),16)%shard_count
        shards[slot].append(key)
    seen=set(completed)
    for rows in shards:
        if seen.intersection(rows):
            raise AssertionError("Completed keys and shard keys must be pairwise disjoint")
        seen.update(rows)
    if seen!=required:
        raise AssertionError("Completed keys and all shards must cover the original required keys exactly")
    return shards


def prepare(registry_path,output_dir,worker_source_ref,shard_count=5):
    if reference(worker_source_ref["path"])!=worker_source_ref:
        raise ValueError("The registered successor worker source changed")
    registry=load_registry(registry_path)
    scope=read_json(registry["common"]["scope_ref"]["path"])
    required=set(map(tuple,scope["required_aggregate_keys"]))
    original=[entry for entry in registry["entries"] if entry["stage"]=="aggregation" and set(entry["task_keys"])==required]
    if not original:
        raise ValueError("The original full aggregation evidence source is required")
    base=original[0]
    completed={key for _,_,key in iter_successes(registry,"aggregation")}
    shards=partition_remaining(required,completed,shard_count)
    completed_sha=digest([list(key) for key in sorted(completed)])
    plan_id=digest(dict(scope=registry["scope_canonical_sha256"],registry_head=registry["last_event_sha256"],
                        completed=completed_sha,worker_source=worker_source_ref,shard_count=shard_count))[:20]
    folder=Path(output_dir).resolve()/plan_id
    completed_ref=write_exact(folder/"COMPLETED_AGGREGATE_KEYS.json",[list(key) for key in sorted(completed)])
    refs=[]
    for index,keys in enumerate(shards):
        worker_id=f"aggregation-v4-{plan_id}-shard{index}"
        keys_ref=write_exact(folder/f"shard{index}_keys.json",[list(key) for key in keys])
        manifest=dict(base["manifest"],worker_id=worker_id,execution_plan_id=plan_id,task_keys_ref=keys_ref,
                      gpu_lock_root=str(Path(registry["common"]["owned_output_root"]).parent/"gpu-locks"/worker_id))
        manifest_ref=write_exact(folder/f"shard{index}_manifest.json",manifest)
        entry=dict(worker_id=worker_id,manifest_ref=manifest_ref,worker_source_ref=worker_source_ref)
        entry_ref=write_exact(folder/f"shard{index}_registry_addition.json",entry)
        refs.append(dict(entry,keys_ref=keys_ref,registry_addition_ref=entry_ref,task_count=len(keys)))
    plan=dict(schema="p2_disjoint_aggregation_dispatch_plan_v2",plan_id=plan_id,
              software_test_only=registry["software_test_only"],common=registry["common"],
              source_registry_head=registry["last_event_sha256"],base_manifest_ref=base["manifest_ref"],
              required_aggregate_calls=len(required),completed_at_preparation=len(completed),
              completed_keys_ref=completed_ref,completed_key_set_sha256=completed_sha,shards=refs,shard_count=shard_count,
              partition_rule=f"SHA256([question_id,generation_seed]) modulo{shard_count}, preserving all actions of that group",
              required_checks=dict(pairwise_disjoint=True,exact_full_union=True),
              launch_gate="Original overlapping actors must have exited. Historical completed keys must still match this snapshot, excluding outputs of this plan's registered successors.",
              changes=dict(scope=0,approval=0,owned_output_root=0,model_spec=0,policy=0,generation_parameters=0),
              production_processes_started=0,reference_values_decoded=0)
    plan_ref=write_exact(folder/"DISPATCH_PLAN.json",plan)
    return dict(plan_ref=plan_ref,plan=plan)


def validate_dispatch_plan(registry,plan):
    if plan.get("schema")!="p2_disjoint_aggregation_dispatch_plan_v2" or plan.get("common")!=registry["common"]:
        raise ValueError("The dispatch plan changed the original common execution identity")
    if plan.get("software_test_only") is not registry["software_test_only"]:
        raise ValueError("Dispatch mode differs from the source registry")
    if plan.get("source_registry_head") not in registry["event_hashes"]:
        raise ValueError("The plan's original registry prefix is missing")
    completed_rows=list(map(tuple,read_json(plan["completed_keys_ref"]["path"])))
    completed=set(completed_rows)
    if len(completed_rows)!=len(completed):
        raise ValueError("Repeated key in completed-key snapshot")
    if reference(plan["completed_keys_ref"]["path"])!=plan["completed_keys_ref"] or digest([list(key) for key in sorted(completed)])!=plan["completed_key_set_sha256"]:
        raise ValueError("Completed-key snapshot changed")
    scope=read_json(registry["common"]["scope_ref"]["path"])
    required=set(map(tuple,scope["required_aggregate_keys"]))
    base=read_json(plan["base_manifest_ref"]["path"])
    if reference(plan["base_manifest_ref"]["path"])!=plan["base_manifest_ref"]:
        raise ValueError("Original full aggregation manifest changed")
    base_entries=[entry for entry in registry["entries"] if entry["manifest_ref"]==plan["base_manifest_ref"]]
    if len(base_entries)!=1 or base_entries[0]["stage"]!="aggregation" or set(base_entries[0]["task_keys"])!=required:
        raise ValueError("Original full aggregation source is not registered")
    if type(plan.get("shard_count")) is not int or plan["shard_count"] not in range(2,9) or len(plan["shards"])!=plan["shard_count"]:
        raise ValueError("The explicit shard count must be an integer from two through eight")
    if len({item["worker_id"] for item in plan["shards"]})!=len(plan["shards"]):
        raise ValueError("Shard worker IDs must be unique")
    pieces=[]
    for item in plan["shards"]:
        if reference(item["manifest_ref"]["path"])!=item["manifest_ref"] or reference(item["keys_ref"]["path"])!=item["keys_ref"]:
            raise ValueError("A shard source changed")
        if reference(item["worker_source_ref"]["path"])!=item["worker_source_ref"]:
            raise ValueError("A shard worker source changed")
        manifest=read_json(item["manifest_ref"]["path"])
        if manifest["task_keys_ref"]!=item["keys_ref"] or manifest.get("worker_id")!=item["worker_id"] or manifest.get("execution_plan_id")!=plan["plan_id"]:
            raise ValueError("Shard manifest and key snapshot differ")
        permitted={"worker_id","execution_plan_id","task_keys_ref","gpu_lock_root"}
        if {key:value for key,value in manifest.items() if key not in permitted}!={key:value for key,value in base.items() if key not in permitted}:
            raise ValueError("A shard changed original scope, owner, backend or scientific configuration")
        rows=list(map(tuple,read_json(item["keys_ref"]["path"])))
        if len(rows)!=len(set(rows)):
            raise ValueError("Repeated semantic key inside a dispatch shard")
        pieces.append(set(rows))
    seen=set(completed)
    for piece in pieces:
        if seen&piece:
            raise ValueError("Completed/shard keys overlap")
        seen.update(piece)
    if seen!=required:
        raise ValueError("Prepared completed/shard keys do not partition the full original required set")
    successor_shas={item["manifest_ref"]["sha256"] for item in plan["shards"]}
    previous_done={key for entry,_,key in iter_successes(registry,"aggregation") if entry["manifest_ref"]["sha256"] not in successor_shas}
    if previous_done!=completed:
        raise RuntimeError("Historical outputs advanced after preparation; prepare a new frozen remaining-key split")
    return dict(status="DISJOINT_DISPATCH_QUALIFIED",completed=len(completed),remaining=[len(piece) for piece in pieces],required=len(required))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--registry",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--worker-source",type=Path,required=True)
    parser.add_argument("--shards",type=int,choices=range(2,9),default=5)
    args=parser.parse_args()
    result=prepare(args.registry,args.output_dir,reference(args.worker_source),args.shards)
    print(json.dumps(dict(plan_ref=result["plan_ref"],completed=result["plan"]["completed_at_preparation"],shard_counts=[item["task_count"] for item in result["plan"]["shards"]],processes_started=0)))
