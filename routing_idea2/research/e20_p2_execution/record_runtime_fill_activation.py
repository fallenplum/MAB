"""Record paid first successes from existing local P2 fill-control receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from record_three_shard_activation import CODE, read_response, sha, utc


SCOPE_SHA = "b355581aaaab1a581e76dab35107a897948da7b4086a1d9e2a33be4a3e1cd1b9"
MEMBER_PLAN_SHA = "5bd363aca0fc1d9c5fa95b86ee5065e43d52c70a1dbe98de2290a8e9f873af21"
AGGREGATE_PLAN_SHA = "b8abd107494166dcb2777663c32164a5fd7632e86f8b1c72dc2be540b5ded4ba"
LAYOUTS = {"member": [5, 4, 4, 5], "aggregation": [0, 1, 3, 1, 6, 3]}
COUNTS = {"member": [1145, 1205, 1231, 1214],
          "aggregation": [11529, 11821, 11784, 11131, 10490, 10842]}
COMPLETED = {"member": 1778, "aggregation": 38939}
REQUIRED = {"member": 6573, "aggregation": 106536}


def local_responses(control_root):
    records = []
    for path in sorted((control_root / "logs").glob("*.json")):
        if int(path.name.split("_", 1)[0]) < 1794908300000000000:
            continue
        outer = json.loads(path.read_bytes())
        if (not outer.get("request_id", "").startswith("p2-fill-")
                or outer.get("status") != "OK" or outer.get("exit_code") != 0):
            continue
        try:
            value, fingerprint = read_response(path)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append((outer["request_id"], value, fingerprint))
    return records


def one(records, prefix, predicate):
    matches = [(value, fingerprint) for request, value, fingerprint in records
               if request.startswith(prefix) and predicate(value)]
    if len(matches) != 1:
        raise ValueError("One successful matching control receipt is required for " + prefix)
    return matches[0]


def collect_launches(records, kind, evidence):
    plan_sha = MEMBER_PLAN_SHA if kind == "member" else AGGREGATE_PLAN_SHA
    checkpoint, fingerprint = one(records, "p2-fill-stopped-" + (
        "member4-" if kind == "member" else "aggregation-"),
        lambda value: value.get("status") == "OWNED_WORKERS_SAFELY_STOPPED")
    if (checkpoint["scope_canonical_sha256"] != SCOPE_SHA
            or checkpoint["completed_unique_keys"] != COMPLETED[kind]
            or len(checkpoint["predecessors"]) != (1 if kind == "member" else 5)
            or any(row["status"] != "SAFELY_STOPPED" for row in checkpoint["predecessors"])
            or checkpoint["reference_values_decoded"] != 0):
        raise ValueError("The predecessor checkpoint differs from the frozen continuation")
    evidence[kind + "_checkpoint"] = fingerprint
    plan, fingerprint = one(records, "p2-fill-prepare-" + kind + "-",
        lambda value: value.get("plan_ref", {}).get("sha256") == plan_sha)
    if (plan["completed"] != COMPLETED[kind] or plan["shard_counts"] != COUNTS[kind]
            or sum(COUNTS[kind]) + COMPLETED[kind] != REQUIRED[kind]):
        raise ValueError("The prepared subsets fail exact remaining coverage")
    evidence[kind + "_plan"] = fingerprint
    launches = []
    expected_check = "DISJOINT_MEMBER_DISPATCH_QUALIFIED" if kind == "member" else "DISJOINT_DISPATCH_QUALIFIED"
    for index, gpu in enumerate(LAYOUTS[kind]):
        launch, fingerprint = one(records, f"p2-fill-launch-{kind}-{index}-",
            lambda value: value.get("status") == "LAUNCHED")
        samples = [int(row["free_mib"]) for row in launch["admitted_gpu_snapshots"]]
        if (launch["gpu"] != gpu or launch["task_keys"] != COUNTS[kind][index]
                or launch["dispatch_plan_ref"]["sha256"] != plan_sha
                or launch["dispatch_plan_check"]["status"] != expected_check
                or launch["scope_sha256"] != SCOPE_SHA
                or launch["reference_access"] or launch["scope_model_or_generation_changes"] != 0
                or len(samples) != 3 or min(samples) < launch["required_free_mib"]
                or not launch["shared_authorization_used"]):
            raise ValueError("A launch differs from its qualified subset or resource admission")
        if kind == "member" and launch.get("member_index") != 4:
            raise ValueError("The member continuation changes frozen profile identity")
        evidence[f"{kind}_launch_{index}"] = fingerprint
        launches.append(launch)
    return checkpoint, launches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, default=CODE.parents[1] / ".e20_p2_root_control")
    parser.add_argument("--output", type=Path, default=CODE / "P2_RUNTIME_FILL_ACTIVATION.json")
    args = parser.parse_args()
    records, evidence = local_responses(args.control_root), {}
    member_checkpoint, member_launches = collect_launches(records, "member", evidence)
    aggregate_checkpoint, aggregate_launches = collect_launches(records, "aggregation", evidence)
    launches = member_launches + aggregate_launches
    ready = []
    for request, monitor, fingerprint in records:
        if monitor.get("at_unix", 0) < max(row["at_unix"] for row in launches):
            continue
        workers = {row["worker_id"]: row for row in monitor.get("workers", [])}
        if all(launch["worker_id"] in workers
               and workers[launch["worker_id"]]["unique_successful_keys"] > 0 for launch in launches):
            ready.append((monitor, fingerprint))
    if not ready:
        raise RuntimeError("The four member and six aggregation actors have not all produced paid successes")
    monitor, fingerprint = min(ready, key=lambda pair: pair[0]["at_unix"])
    if (monitor["registered_manifest_journals_read"] != 25
            or monitor["duplicate_success_rows_deduplicated"] != 0
            or monitor["additional_paid_attempts_for_already_successful_keys"] != 0
            or monitor["reference_access"] or monitor["quality_or_scores_read"] != 0):
        raise ValueError("The actual first-success monitor fails journal or paid-usage qualification")
    evidence["first_success_monitor"] = fingerprint
    workers = {row["worker_id"]: row for row in monitor["workers"]}
    public_launches = {"member": [], "aggregation": []}
    for kind, batch in (("member", member_launches), ("aggregation", aggregate_launches)):
        for index, launch in enumerate(batch):
            worker = workers[launch["worker_id"]]
            matching = [actor for actor in worker["actors"]
                        if actor["pid"] == launch["pid"] and actor["alive"]]
            if len(matching) != 1 or worker["live_actor_count"] != 1:
                raise ValueError("A new actor is no longer alive at the first-success snapshot")
            public_launches[kind].append(dict(shard_index=index, gpu=launch["gpu"], pid=launch["pid"],
                assigned_outputs=launch["task_keys"], first_snapshot_successes=worker["unique_successful_keys"],
                launch_utc=utc(launch["at_unix"]), active_at_snapshot=True,
                observed_free_mib=[row["free_mib"] for row in launch["admitted_gpu_snapshots"]],
                manifest_sha256=launch["manifest_sha256"], worker_source_sha256=launch["worker_source_sha256"]))
    new_aggregate = sum(row["first_snapshot_successes"] for row in public_launches["aggregation"])
    if monitor["stages"]["aggregation"]["unique_successful_keys"] != COMPLETED["aggregation"] + new_aggregate:
        raise ValueError("The six-way successes do not restore historical aggregation exactly once")
    member4_total = sum(worker["unique_successful_keys"] for worker in monitor["workers"]
                        if worker["stage"] == "member_generation" and worker["member_index"] == 4)
    new_member = sum(row["first_snapshot_successes"] for row in public_launches["member"])
    if member4_total != COMPLETED["member"] + new_member:
        raise ValueError("The four-way successes do not restore historical member output exactly once")
    qualification_raw = (CODE / "P2_MEMBER_PARALLEL_QUALIFICATION.json").read_bytes()
    qualification = json.loads(qualification_raw)
    if qualification["checks_passed"] != 17 or qualification["production_deployed"] is not False:
        raise ValueError("The original software qualification snapshot changed")
    for field in ("source_sha256", "frozen_source_sha256"):
        for name, expected in qualification[field].items():
            if sha((CODE / name).read_bytes()) != expected:
                raise ValueError("The member tools differ from their prepared qualification")
    result = dict(schema="p2_runtime_fill_activation_v1",
        status="FOUR_MEMBER_AND_SIX_AGGREGATION_FIRST_SUCCESSES_VERIFIED",
        snapshot_utc=utc(monitor["at_unix"]), member_index=4,
        member_stop_utc=member_checkpoint["utc"], aggregation_stop_utc=aggregate_checkpoint["utc"],
        historical_member4_successes=COMPLETED["member"], historical_aggregation_successes=COMPLETED["aggregation"],
        member4_required_outputs=REQUIRED["member"], required_aggregations=REQUIRED["aggregation"],
        member_plan_sha256=MEMBER_PLAN_SHA, aggregation_plan_sha256=AGGREGATE_PLAN_SHA,
        member4_shards=public_launches["member"], aggregation_shards=public_launches["aggregation"],
        disjoint_remaining_partitions_verified=True, registered_manifests_after_add=25,
        active_actor_count_at_snapshot=sum(row["live_actor_count"] for row in monitor["workers"]),
        remote_successful_outputs_at_snapshot={stage:dict(successful_outputs=row["unique_successful_keys"],
            prompt_tokens=row["actual_tokens"]["prompt_tokens"], completion_tokens=row["actual_tokens"]["completion_tokens"],
            total_successful_tokens=row["actual_tokens"]["total_tokens"]) for stage, row in monitor["stages"].items()},
        duplicate_success_rows_at_snapshot=0, extra_paid_attempts_for_successful_keys_at_snapshot=0,
        software_checks_at_preparation=17, preparation_qualification_report_preserved=True,
        preparation_qualification_sha256=sha(qualification_raw), control_evidence_sha256=evidence,
        science_scope_model_policy_or_generation_changes=0, reference_values_read=0,
        deterministic_scores_computed=0, local_publication_new_llm_calls=0,
        failure_closure_pending=True, all_attempt_total_tokens=None,
        claims_scope="实际安全停止、互斥接续、连续资源准入与首成功快照；吞吐变化另用后续窗口核算")
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], snapshot_utc=result["snapshot_utc"],
        member_first_successes=[row["first_snapshot_successes"] for row in public_launches["member"]],
        aggregation_first_successes=[row["first_snapshot_successes"] for row in public_launches["aggregation"]],
        active_actor_count=result["active_actor_count_at_snapshot"], reference_values_read=0), ensure_ascii=False))


if __name__ == "__main__":
    main()
