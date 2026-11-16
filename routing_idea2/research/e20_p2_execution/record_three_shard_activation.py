"""Create an aggregate-only activation snapshot from existing local receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


CODE = Path(__file__).resolve().parent
LOGS = {
    "registry_initialization": "1794847969248673900_3eddeaee.json",
    "old_actor_checkpoint": "1794848014880493000_5b605224.json",
    "remaining_partition": "1794848034568241700_fe1bcf2c.json",
    "registry_add0": "1794848049715655500_8995f7e0.json",
    "registry_add1": "1794848056321707400_edcee661.json",
    "registry_add2": "1794848062403159500_bc22f758.json",
    "shard0_launch": "1794848083152947200_46c973cf.json",
    "shard1_launch": "1794848091252584600_9a355906.json",
    "shard2_launch": "1794848099153282100_719ef6ea.json",
    "first_success_monitor": "1794848142627093300_75df9004.json",
}
PLAN_SHA = "c2d41b9a4796b2fb79eb94d1aaa8e589c2bd62d579ac7986377aa8695b737d69"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc(timestamp):
    return datetime.fromtimestamp(timestamp,timezone.utc).isoformat()


def read_response(path):
    raw=Path(path).read_bytes()
    result=json.loads(raw)
    if result.get("status")!="OK" or result.get("exit_code",0)!=0:
        raise ValueError("An activation control response did not succeed")
    return json.loads(result["stdout"]),sha(raw)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--control-root",type=Path,default=CODE.parents[1]/".e20_p2_root_control")
    parser.add_argument("--collection-state",type=Path,default=CODE.parents[1]/".e20_p2_cpu_runtime/COLLECTION_PROGRESS.json")
    parser.add_argument("--output",type=Path,default=CODE/"P2_THREE_SHARD_ACTIVATION.json")
    args=parser.parse_args()
    responses,evidence={},{}
    for key,name in LOGS.items():
        responses[key],evidence[key]=read_response(args.control_root/"logs"/name)
    stopped=responses["old_actor_checkpoint"]
    plan=responses["remaining_partition"]
    monitor=responses["first_success_monitor"]
    if (stopped["status"]!="OLD_FULL_ACTOR_SAFELY_STOPPED" or not stopped["old_actor_exited"]
            or stopped["completed_unique_aggregations"]!=586 or stopped["repeated_success_keys"]!=0):
        raise ValueError("Old actor lacks the recorded complete unique checkpoint")
    if (plan["completed"]!=586 or plan["shard_counts"]!=[34883,35826,35241]
            or plan["plan_ref"]["sha256"]!=PLAN_SHA or sum(plan["shard_counts"])+586!=106536):
        raise ValueError("The actual plan does not partition the frozen remaining union")
    entries=[responses["registry_initialization"]["entries"]]+[
        responses[f"registry_add{index}"]["entries"] for index in range(3)]
    if entries!=[7,8,9,10] or monitor["registered_manifest_journals_read"]!=10:
        raise ValueError("Original source registration or ADD-only succession differs")
    workers={row["worker_id"]:row for row in monitor["workers"]}
    shards=[]
    for index in range(3):
        launch=responses[f"shard{index}_launch"]
        worker=workers[launch["worker_id"]]
        actor=[row for row in worker["actors"] if row["pid"]==launch["pid"]]
        if (launch["status"]!="LAUNCHED" or launch["task_keys"]!=plan["shard_counts"][index]
                or launch["dispatch_plan_ref"]["sha256"]!=PLAN_SHA
                or launch["dispatch_plan_check"]["status"]!="DISJOINT_DISPATCH_QUALIFIED"
                or worker["unique_successful_keys"]<1 or worker["live_actor_count"]!=1
                or len(actor)!=1 or not actor[0]["alive"]):
            raise ValueError("A registered shard has no verified live first-success path")
        samples=[int(row["free_mib"]) for row in launch["admitted_gpu_snapshots"]]
        if len(samples)!=3 or min(samples)<launch["required_free_mib"] or not launch["shared_authorization_used"]:
            raise ValueError("A shard lacks its actual repeated headroom admission")
        shards.append(dict(shard_index=index,gpu=launch["gpu"],pid=launch["pid"],
            launch_utc=utc(launch["at_unix"]),assigned_remaining_aggregations=launch["task_keys"],
            first_snapshot_successes=worker["unique_successful_keys"],active_at_snapshot=True,
            observed_free_mib=samples,shared_resource_authorization=True,
            manifest_sha256=launch["manifest_sha256"],worker_source_sha256=launch["worker_source_sha256"]))
    members=[]
    for row in monitor["workers"]:
        if row["stage"]!="member_generation":
            continue
        live=[actor for actor in row["actors"] if actor["alive"]]
        if len(live)!=1:
            raise ValueError("The snapshot does not preserve one live actor per member profile")
        members.append(dict(member_index=row["member_index"],gpu=live[0]["gpu"],
            pid=live[0]["pid"],active_at_snapshot=True,successful_outputs=row["unique_successful_keys"]))
    members.sort(key=lambda row:row["member_index"])
    if [row["gpu"] for row in members]!=[4,7,7,7,5,6]:
        raise ValueError("The actual member-profile placement differs")
    if (monitor["duplicate_success_rows_deduplicated"]!=0
            or monitor["additional_paid_attempts_for_already_successful_keys"]!=0
            or monitor["quality_or_scores_read"]!=0 or monitor["reference_access"]):
        raise ValueError("The activation monitor reports duplicate charging or reference access")
    collection_raw=args.collection_state.read_bytes()
    collection=json.loads(collection_raw)
    if collection["all_required_outputs_frozen"] or collection["reference_values_read"] or collection["deterministic_scores_computed"]:
        raise ValueError("This activation snapshot must remain in the partial blind execution phase")
    reports=["P2_REGISTRY_DISPATCH_QUALIFICATION.json","P2_REGISTRY_CONSUMERS_QUALIFICATION.json","P2_V3_TOOLING_QUALIFICATION.json"]
    qualification_hashes={}
    for name in reports:
        source=(CODE/name).read_bytes()
        value=json.loads(source)
        qualification_hashes[name]=sha(source)
        for key in ("source_sha256","new_source_sha256","original_pinned_sources_match"):
            for filename,expected in value.get(key,{}).items():
                if sha((CODE/filename).read_bytes())!=expected:
                    raise ValueError("Ready tooling source differs from its qualification")
    result=dict(schema="p2_three_registered_shard_activation_v1",
        status="THREE_SHARDS_FIRST_SUCCESSES_VERIFIED_AT_SNAPSHOT",snapshot_utc=utc(monitor["at_unix"]),
        old_actor_stop_utc=stopped["utc"],old_actor_pid=2637354,old_actor_exited=True,
        old_unique_completed_aggregations=586,old_repeated_success_keys=0,
        original_registered_manifests=7,registered_manifests_after_add=10,
        dispatch_plan_sha256=PLAN_SHA,disjoint_remaining_partition_verified=True,
        remaining_assigned_aggregations=sum(plan["shard_counts"]),required_aggregations=106536,
        aggregation_shards=shards,member_profiles=members,active_actor_count_at_snapshot=9,
        duplicate_success_rows_at_snapshot=0,extra_paid_attempts_for_successful_keys_at_snapshot=0,
        remote_successful_outputs_at_snapshot={stage:dict(successful_outputs=row["unique_successful_keys"],
            prompt_tokens=row["actual_tokens"]["prompt_tokens"],completion_tokens=row["actual_tokens"]["completion_tokens"],
            total_successful_tokens=row["actual_tokens"]["total_tokens"]) for stage,row in monitor["stages"].items()},
        local_collection_at_its_own_snapshot=dict(snapshot_utc=collection["utc"],
            imported_archives=collection["imported_archives"],successful_outputs=collection["successful_outputs"],
            members=collection["stages"]["member_generation"]["successful_outputs"],
            aggregations=collection["stages"]["aggregation"]["successful_outputs"],
            successful_output_total_tokens=collection["successful_output_total_tokens"],
            all_required_outputs_frozen=False,failure_closure_pending=collection["failure_closure_pending"],
            all_attempt_total_tokens=collection["all_attempt_total_tokens"]),
        software_checks_at_preparation=34,preparation_qualification_reports_preserved=True,
        preparation_qualification_sha256=qualification_hashes,
        control_evidence_sha256={**evidence,"local_collection_snapshot":sha(collection_raw)},
        science_scope_model_policy_or_generation_changes=0,reference_values_read=0,
        deterministic_scores_computed=0,local_publication_new_llm_calls=0,
        claims_scope="Actual graceful checkpoint, disjoint registration, launch and first successes; throughput effect awaits a subsequent matched time window")
    with args.output.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"],snapshot_utc=result["snapshot_utc"],
        shard_first_successes=[row["first_snapshot_successes"] for row in shards],
        local_successful_outputs=collection["successful_outputs"],reference_values_read=0),ensure_ascii=False))


if __name__=="__main__":
    main()
