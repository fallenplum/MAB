"""Publish one de-identified activation snapshot from live native metadata."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


CODE = Path(__file__).resolve().parent
WORKSPACE = CODE.parents[1]
ASSETS = Path(r"E:\语言模型组合路由")
STUDY = ASSETS / "research/e20_p2_execution"


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    logs = WORKSPACE / ".e20_p2_runtime/remote_logs"
    candidates = []
    for path in logs.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("request_id") == "p2-parent-monitor" and row.get("exit_code") == 0:
            candidates.append((path, row))
    source_path, raw = max(candidates, key=lambda pair: pair[0].stat().st_mtime_ns)
    live = json.loads(raw["stdout"])
    profiles = live["profiles"]
    expected = {"member"+str(index) for index in range(6)} | {"aggregation"}
    if {row["profile"] for row in profiles} != expected or not all(row["alive"] for row in profiles):
        raise RuntimeError("All six member profiles and aggregation follower must be alive")
    members = [row for row in profiles if row["profile"] != "aggregation"]
    aggregation = next(row for row in profiles if row["profile"] == "aggregation")
    member_completed = sum(row["event_counts"].get("successful_new", 0) for row in members)
    aggregate_completed = aggregation["event_counts"].get("successful_new", 0)
    if member_completed <= 0 or aggregate_completed <= 0:
        raise RuntimeError("Activation requires real native member and aggregation commits")
    usage = {}
    for name, rows in (("member_generation",members),("aggregation",[aggregation])):
        usage[name] = {key: sum(row["actual_tokens"].get(key, 0) for row in rows)
                       for key in ("prompt_tokens", "completion_tokens")}
        usage[name]["total_tokens"] = sum(usage[name].values())
    launch = json.loads((STUDY / "private/P2_PRODUCTION_LAUNCH_RECEIPT_v2.json").read_text(encoding="utf-8"))
    status = dict(
        schema="p2_native_activation_status_v1", status="RUNNING",
        snapshot_utc=datetime.fromtimestamp(live["at_unix"],tz=timezone.utc).isoformat(),
        original_questions=2400, task_counts={"gsm8k":1200,"bbh":1200},
        required_member_generations=41289, required_aggregations=106536,
        successful_member_generations=member_completed, successful_aggregations=aggregate_completed,
        actual_successful_output_tokens=usage,
        total_actual_successful_output_tokens=sum(value["total_tokens"] for value in usage.values()),
        live_member_profiles=6, live_aggregation_followers=1,
        gpu_indices_in_use=[4,5,6,7], shared_gpu_indices=[4,5,6],
        per_profile=[dict(profile=row["profile"], gpu_index=row["gpu"], alive=row["alive"],
                          successful_new_calls=row["event_counts"].get("successful_new",0),
                          prompt_tokens=row["actual_tokens"].get("prompt_tokens",0),
                          completion_tokens=row["actual_tokens"].get("completion_tokens",0)) for row in profiles],
        scope_canonical_sha256=launch["scope_canonical_sha256"],
        scope_file_sha256=next(row["sha256"] for row in launch["generated_files"] if row["path"]=="inputs/EXECUTION_SCOPE_RUNTIME.json"),
        protocol_sha256=launch["bindings"]["protocol_sha256"],
        final_model_sha256=launch["bindings"]["final_model_sha256"],
        prediction_sha256=launch["bindings"]["final_prediction_sha256"],
        input_sha256=launch["bindings"]["target_inputs_sha256"],
        final_accuracy_evaluated=False, gold_reference_values_decoded=0,
        sealed200_answer_values_decoded=0, fresh400_reward_values_read=0,
        new_router_fits=0, confidence_intervals_computed=0, bootstrap_repetitions=0,
        startup_recoveries=[
            dict(issue="Initial execution approval omitted required per-stage execution_limits",affected_worker_starts=2,
                 llm_calls=0,native_owner_reservations=0,resolution="Versioned approval and manifests bind both exact stage limits and absolute output root"),
            dict(issue="Aggregation context_tokenizer returns tokenizer and receipt; initial follower omitted tuple unpack",affected_worker_starts=1,
                 llm_calls=0,native_owner_reservations=0,resolution="Correct original helper interface and preserve tokenizer-alias qualification receipt")],
        startup_model_loading_gpu_time="Not separately measured; retained in process logs, distinct from native output token costs",
        completed=False, pipeline="All member profiles execute full frozen key lists; aggregation follower incrementally consumes committed dependencies with one persistent original backend",
    )
    write_once(CODE / "P2_ACTIVATION_STATUS.json",status)
    write_once(STUDY / "public/P2_ACTIVATION_STATUS.json",status)
    write_once(STUDY / "private/control/ACTIVATION_METADATA_PRIVATE.json",live)
    handoff = dict(status="ACTIVE_RUNTIME_CONTROL", ssh_control_session_id=54037,
                   remote_control_program=str(WORKSPACE / "experiments/e18_confirmatory_execution/remote_control.py"),
                   local_request_file=str(WORKSPACE / ".e20_p2_runtime/requests/monitor.json"),
                   request_line={"operation":"request_file","path":str(WORKSPACE / ".e20_p2_runtime/requests/monitor.json")},
                   local_control_log_root=str(logs),active_workers=[dict(profile=row["profile"],pid=row["pid"],gpu=row["gpu"],
                   manifest_sha256=row["manifest_sha256"],log_path=row["log_path"]) for row in profiles],
                   remote_root="/data2/chendong/e20_p2_execution",
                   runtime_python="/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python",
                   remote_monitor_argv=["/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python","-B","/data2/chendong/e20_p2_execution/code/monitor_native_status.py"],
                   scopes_and_approvals="private/runtime_launch_v2/inputs", credentials_stored=False,
                   controls="The open interactive SSH process is alive at this handoff; if its tool handle is unavailable, reopen the existing pinned control program with credentials held only in memory")
    write_once(STUDY / "private/control/RUNTIME_HANDOFF_PRIVATE.json",handoff)
    print(json.dumps({"status":"RUNNING","member_successes":member_completed,"aggregation_successes":aggregate_completed,
                      "tokens":status["total_actual_successful_output_tokens"],"snapshot_utc":status["snapshot_utc"]}))


if __name__=="__main__":
    main()
