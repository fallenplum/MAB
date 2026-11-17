"""Record five-shard first successes from completed local control responses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from record_three_shard_activation import CODE, read_response, sha, utc


PLAN_SHA = "eff30ade1a613584f485a953869e7e88b6a012c7902f8dba6a5d8d52eb2bf58d"
COUNTS = [17536,16900,16882,17051,17325]
GPUS = [0,1,3,1,6]
PIDS = [1462112,1462800,1463437,1464025,1464877]
PREDECESSOR_PIDS = {2884571,2885196,2885924}


def responses(control_root):
    records=[]
    for path in sorted((control_root/"logs").glob("*.json")):
        if int(path.name.split("_",1)[0]) < 1794887400000000000:
            continue
        outer=json.loads(path.read_bytes())
        if not outer.get("request_id","").startswith("p2-five-") or not outer.get("stdout"):
            continue
        value,fingerprint=read_response(path)
        records.append((outer["request_id"],value,fingerprint))
    return records


def select(records,prefix,predicate):
    selected=[(value,fingerprint) for request,value,fingerprint in records
              if request.startswith(prefix) and predicate(value)]
    if len(selected)!=1:
        raise ValueError("A unique matching completed control response is required for " + prefix)
    return selected[0]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--control-root",type=Path,default=CODE.parents[1]/".e20_p2_root_control")
    parser.add_argument("--output",type=Path,default=CODE/"P2_FIVE_SHARD_ACTIVATION.json")
    args=parser.parse_args()
    records=responses(args.control_root)
    checkpoint,checkpoint_sha=select(records,"p2-five-stop-check-",lambda row:row.get("status")=="THREE_OWNED_ACTORS_SAFELY_STOPPED")
    if (checkpoint["completed_unique_aggregations"]!=20842
            or checkpoint["duplicate_paid_success_keys"]!=0
            or {row["pid"] for row in checkpoint["predecessors"]}!=PREDECESSOR_PIDS
            or any(row["status"]!="SAFELY_STOPPED" for row in checkpoint["predecessors"])):
        raise ValueError("The three predecessor checkpoints differ")
    plan,plan_sha=select(records,"p2-five-prepare-",lambda row:row.get("plan_ref",{}).get("sha256")==PLAN_SHA)
    if plan["completed"]!=20842 or plan["shard_counts"]!=COUNTS or sum(COUNTS)+20842!=106536:
        raise ValueError("The frozen remaining aggregate partition differs")
    launches=[]
    evidence=dict(old_actor_checkpoint=checkpoint_sha,remaining_partition=plan_sha)
    for index in range(5):
        registration,registered_sha=select(records,f"p2-five-register-{index}-",lambda row:row.get("entries")==11+index)
        launch,launch_sha=select(records,f"p2-five-launch-{index}-",lambda row:row.get("pid")==PIDS[index])
        if (launch["status"]!="LAUNCHED" or launch["gpu"]!=GPUS[index]
                or launch["task_keys"]!=COUNTS[index] or launch["dispatch_plan_ref"]["sha256"]!=PLAN_SHA
                or launch["dispatch_plan_check"]["status"]!="DISJOINT_DISPATCH_QUALIFIED"
                or launch["reference_access"] or launch["scope_model_or_generation_changes"]!=0):
            raise ValueError("A launched actor differs from its five-way plan")
        samples=[int(row["free_mib"]) for row in launch["admitted_gpu_snapshots"]]
        if len(samples)!=3 or min(samples)<launch["required_free_mib"] or not launch["shared_authorization_used"]:
            raise ValueError("A launched actor lacks its repeated resource admission")
        launches.append(launch)
        evidence[f"registry_add{index}"]=registered_sha
        evidence[f"shard{index}_launch"]=launch_sha
    positive=[]
    for request,monitor,fingerprint in records:
        if not request.startswith("p2-five-loading-monitor-") or monitor.get("at_unix",0)<max(row["at_unix"] for row in launches):
            continue
        workers={row["worker_id"]:row for row in monitor.get("workers",[])}
        if all(launch["worker_id"] in workers and workers[launch["worker_id"]]["unique_successful_keys"]>0 for launch in launches):
            positive.append((monitor,fingerprint))
    if not positive:
        raise RuntimeError("Five launched actors have not yet all produced a successful output")
    monitor,monitor_sha=min(positive,key=lambda pair:pair[0]["at_unix"])
    workers={row["worker_id"]:row for row in monitor["workers"]}
    if (sum(row["live_actor_count"] for row in monitor["workers"])!=10
            or monitor["registered_manifest_journals_read"]!=15
            or monitor["duplicate_success_rows_deduplicated"]!=0
            or monitor["additional_paid_attempts_for_already_successful_keys"]!=0
            or monitor["reference_access"] or monitor["quality_or_scores_read"]!=0):
        raise ValueError("The first-success monitor fails active-worker or paid-usage qualification")
    shards=[]
    for index,launch in enumerate(launches):
        worker=workers[launch["worker_id"]]
        matching=[row for row in worker["actors"] if row["pid"]==PIDS[index] and row["alive"]]
        if len(matching)!=1 or worker["live_actor_count"]!=1:
            raise ValueError("A five-way successor is not alive at the selected snapshot")
        shards.append(dict(shard_index=index,gpu=GPUS[index],pid=PIDS[index],
            assigned_remaining_aggregations=COUNTS[index],launch_utc=utc(launch["at_unix"]),
            first_snapshot_successes=worker["unique_successful_keys"],active_at_snapshot=True,
            observed_free_mib=[row["free_mib"] for row in launch["admitted_gpu_snapshots"]],
            shared_resource_authorization=True,manifest_sha256=launch["manifest_sha256"],
            worker_source_sha256=launch["worker_source_sha256"]))
    first_successes=sum(row["first_snapshot_successes"] for row in shards)
    if monitor["stages"]["aggregation"]["unique_successful_keys"]!=20842+first_successes:
        raise ValueError("Historical and five-way successful outputs do not restore exactly once")
    member_profiles=[]
    for worker in monitor["workers"]:
        if worker["stage"]!="member_generation":
            continue
        live=[actor for actor in worker["actors"] if actor["alive"]]
        complete=worker["unique_successful_keys"]==worker["planned_subset_keys"]
        if len(live)>1 or (not live and not complete):
            raise ValueError("A member profile lacks its active or completed state")
        row=dict(member_index=worker["member_index"],successful_outputs=worker["unique_successful_keys"],
            planned_outputs=worker["planned_subset_keys"],active_actor_count=len(live),complete=complete)
        if live:
            row.update(gpu=live[0]["gpu"],pid=live[0]["pid"])
        member_profiles.append(row)
    source=(CODE/"P2_FIVE_SHARD_TOOLING_QUALIFICATION.json").read_bytes()
    qualification=json.loads(source)
    if qualification["checks_passed"]!=16 or qualification["production_deployed"] is not False:
        raise ValueError("The original preparation qualification was modified")
    for field in ("old_source_sha256","source_sha256"):
        for filename,expected in qualification[field].items():
            if sha((CODE/filename).read_bytes())!=expected:
                raise ValueError("Five-way tooling source differs from its frozen qualification")
    evidence["first_success_monitor"]=monitor_sha
    result=dict(schema="p2_five_registered_shard_activation_v1",
        status="FIVE_SHARDS_FIRST_SUCCESSES_VERIFIED_AT_SNAPSHOT",snapshot_utc=utc(monitor["at_unix"]),
        user_requested_additional_gpu_indices=[1,6],old_actor_stop_utc=checkpoint["utc"],
        retired_three_actor_count=3,old_unique_completed_aggregations=20842,
        duplicate_paid_success_keys_at_checkpoint=0,registered_manifests_before_add=10,
        registered_manifests_after_add=15,dispatch_plan_sha256=PLAN_SHA,
        disjoint_remaining_partition_verified=True,remaining_assigned_aggregations=sum(COUNTS),
        required_aggregations=106536,aggregation_shards=shards,
        member_profiles=sorted(member_profiles,key=lambda row:row["member_index"]),
        active_actor_count_at_snapshot=10,first_five_shard_successes=first_successes,
        duplicate_success_rows_at_snapshot=0,extra_paid_attempts_for_successful_keys_at_snapshot=0,
        remote_successful_outputs_at_snapshot={stage:dict(successful_outputs=row["unique_successful_keys"],
            prompt_tokens=row["actual_tokens"]["prompt_tokens"],completion_tokens=row["actual_tokens"]["completion_tokens"],
            total_successful_tokens=row["actual_tokens"]["total_tokens"]) for stage,row in monitor["stages"].items()},
        software_checks_at_preparation=16,preparation_qualification_report_preserved=True,
        preparation_qualification_sha256=sha(source),control_evidence_sha256=evidence,
        science_scope_model_policy_or_generation_changes=0,reference_values_read=0,
        deterministic_scores_computed=0,local_publication_new_llm_calls=0,
        failure_closure_pending=True,all_attempt_total_tokens=None,
        claims_scope="Actual checkpoint, five-way partition, registration, repeated resource admission and first successful paid journal events; no throughput-improvement claim")
    with args.output.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"],snapshot_utc=result["snapshot_utc"],
        shard_first_successes=[row["first_snapshot_successes"] for row in shards],
        member_successes=monitor["stages"]["member_generation"]["unique_successful_keys"],
        aggregate_successes=monitor["stages"]["aggregation"]["unique_successful_keys"],reference_values_read=0),ensure_ascii=False))


if __name__=="__main__":
    main()
