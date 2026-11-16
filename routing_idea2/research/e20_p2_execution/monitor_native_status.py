"""Report live P2 native progress and resource use without answers or scores."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
import time


ROOT = Path("/data2/chendong/e20_p2_execution")


def main():
    profiles = []
    latest = {}
    starts = [json.loads(path.read_text()) for path in (ROOT / "control").glob("*.start.json")]
    for row in starts:
        if row["profile"] not in latest or row["at_unix"] > latest[row["profile"]]["at_unix"]:
            latest[row["profile"]] = row
    for row in sorted(latest.values(), key=lambda value: value["profile"]):
        pid = row["pid"]
        proc = Path("/proc") / str(pid)
        live = proc.exists()
        record = dict(profile=row["profile"], pid=pid, gpu=row["gpu"], alive=live,
                      start_unix=row["at_unix"], manifest_sha256=row["manifest_sha256"], log_path=row["log_path"])
        if live:
            record["process_io"] = {key: int(value) for key, value in
                                    (line.split(": ", 1) for line in (proc / "io").read_text().splitlines())}
            record["process_state"] = next(line for line in (proc / "status").read_text().splitlines() if line.startswith("State:"))
        journal = ROOT / "owned/_dispatcher" / row["manifest_sha256"] / "EVENTS.jsonl"
        counts = Counter()
        usage = Counter()
        last_time = None
        if journal.is_file():
            with journal.open() as stream:
                for line in stream:
                    if not line.endswith("\n"):
                        break
                    event = json.loads(line)
                    counts[event["event"]] += 1
                    if event.get("event") == "TASK_RESULT" and event.get("status") == "SUCCESS":
                        counts["successful_cached" if event["from_cache"] else "successful_new"] += 1
                        usage.update(event.get("actual_usage") or {})
                        last_time = event.get("at_unix")
        record.update(event_counts=dict(counts), actual_tokens=dict(usage), last_success_unix=last_time)
        profiles.append(record)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                          capture_output=True,text=True,check=True).stdout.splitlines()
    result = dict(schema="p2_live_native_status_v1",at_unix=time.time(),profiles=profiles,gpu=gpu,
                  earlier_start_receipts_preserved=len(starts)-len(latest),
                  reference_access=False,quality_or_scores_read=0)
    print(json.dumps(result))


if __name__=="__main__":
    main()
