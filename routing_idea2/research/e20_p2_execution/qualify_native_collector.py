"""Synthetic interruption, rollback and concurrency checks for the P2 collector."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import uuid

import collect_native_exports as collector_api


class SyntheticContext:
    def __init__(self):
        self.scope_sha = "a"*64
        self.scope_file_sha = "b"*64
        self.bindings = dict(scope_sha256=self.scope_file_sha,scope_canonical_sha256=self.scope_sha,
            approval_sha256="c"*64,prediction_sha256="d"*64,final_model_sha256="e"*64,reference_plan_sha256="f"*64)
        self.allowed = {stage:{(f"synthetic-{index}",42,0) for index in (0,1)} for stage in collector_api.FILES}
        self.audit = SimpleNamespace(strict_json=json.loads,read_json=lambda path:json.loads(Path(path).read_text(encoding="utf-8")))

    def audit_owner(self,stage,key_hash,files):
        receipt = json.loads(files["LAST_TASK_RECEIPT.json"])
        native = receipt["native_record"]
        key = receipt["key"]
        rows = [json.loads(line) for line in files["records.jsonl"].splitlines()]
        if (tuple(key) not in self.allowed[stage] or collector_api.digest(key) != key_hash
                or rows != [native] or native["record_sha256"] != collector_api.digest({k:v for k,v in native.items() if k!="record_sha256"})
                or json.loads(files["ATTEMPT_STATE.json"])["status"] != "SUCCESS_COMMITTED"):
            raise ValueError("Synthetic closed-owner fixture differs")
        result = native["result"]
        return receipt,dict(stage=stage,task_key=key,native_record_sha256=native["record_sha256"],
            physical_attempt_id=native["attempt_id"],prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],wall_time_s=result["wall_time_s"])


def archive(path,tasks,batch):
    context = SyntheticContext()
    manifest = dict(schema="p2_closed_native_export_v1",batch=batch,scope_sha256=context.scope_sha,
                    gold_reads=0,tasks=[],prompt_tokens=0,completion_tokens=0)
    with tarfile.open(path,"x:gz") as target:
        for stage,index,variant in tasks:
            key = [f"synthetic-{index}",42,0]
            key_hash = collector_api.digest(key)
            native = dict(attempt_id=collector_api.digest([stage,key]),result=dict(
                text="artificial output "+variant,prompt_tokens=7,completion_tokens=3,wall_time_s=.2))
            native["record_sha256"] = collector_api.digest(native)
            receipt = dict(key=key,native_record=native)
            payloads = {"ATTEMPT_STATE.json":collector_api.encoded(dict(status="SUCCESS_COMMITTED")),
                        "LAST_TASK_RECEIPT.json":collector_api.encoded(receipt),
                        "records.jsonl":collector_api.encoded(native)+b"\n"}
            files = []
            for name,payload in payloads.items():
                relative=f"owned/{context.scope_sha}/{stage}/{key_hash}/{name}"
                item=tarfile.TarInfo(relative)
                item.size=len(payload)
                target.addfile(item,io.BytesIO(payload))
                files.append(dict(path=relative,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))
            manifest["tasks"].append(dict(stage=stage,task_key=key,digest=key_hash,
                native_record_sha256=native["record_sha256"],files=files))
            manifest["prompt_tokens"]+=7
            manifest["completion_tokens"]+=3
        payload=collector_api.encoded(manifest)
        item=tarfile.TarInfo("P2_NATIVE_EXPORT_MANIFEST.json")
        item.size=len(payload)
        target.addfile(item,io.BytesIO(payload))


def worker(work,source):
    current = collector_api.Collector(SyntheticContext(),work)
    try:
        result = current.import_archive(source)
        progress = current.snapshot()
    finally:
        current.close()
    print(json.dumps(dict(result=result,count=progress["successful_outputs"])))


def main():
    root = Path(r"D:\develop\CV\多臂老虎机\.e20_p2_collector_tests")/uuid.uuid4().hex
    root.mkdir(parents=True)
    first = root/"first.tar.gz"
    overlap = root/"overlap.tar.gz"
    conflict = root/"conflict.tar.gz"
    archive(first,[("member_generation",0,"same"),("aggregation",0,"same")],"toy-first")
    archive(overlap,[("member_generation",0,"same"),("aggregation",1,"same")],"toy-overlap")
    archive(conflict,[("member_generation",1,"same"),("member_generation",0,"different")],"toy-conflict")
    checks = {}
    work = root/"recovery"
    current = collector_api.Collector(SyntheticContext(),work)
    try:
        checks["first_closed_archive_commits_two_rows"] = current.import_archive(first)["new_outputs"] == 2
        sizes = {name:(work/name).stat().st_size for name in collector_api.FILES.values()}
        original = current._consume_archive
        current._consume_archive = lambda *args: (_ for _ in ()).throw(AssertionError("Known archive rescanned"))
        checks["unchanged_archive_skips_tar_scan"] = current.import_archive(first)["status"] == "ALREADY_IMPORTED_UNCHANGED"
        current._consume_archive = original
        checks["repeat_does_not_append"] = sizes == {name:(work/name).stat().st_size for name in collector_api.FILES.values()}
        result = current.import_archive(overlap)
        checks["overlapping_batches_deduplicate_semantic_native_key"] = result["new_outputs"] == 1 and result["duplicate_outputs"] == 1
        sizes = {name:(work/name).stat().st_size for name in collector_api.FILES.values()}
        try:
            current.import_archive(conflict)
        except ValueError:
            checks["conflicting_native_output_rejected"] = True
        checks["late_failure_rolls_back_rows_and_file_tail"] = current.snapshot()["successful_outputs"] == 3 and sizes == {name:(work/name).stat().st_size for name in collector_api.FILES.values()}
        checks["partial_collection_has_no_final_freeze"] = not (work/"OUTPUTS_FROZEN.json").exists()
    finally:
        current.close()
    crash_code = """from pathlib import Path
import os,sqlite3,sys
root=Path(sys.argv[1])
connection=sqlite3.connect(root/'NATIVE_COLLECTION.sqlite',isolation_level=None)
connection.execute('BEGIN IMMEDIATE')
tail=b'UNCOMMITTED-CRASH-TAIL\\n'
with (root/'MEMBER_NATIVE_RECEIPTS.jsonl').open('ab',buffering=0) as stream:
    stream.write(tail)
    os.fsync(stream.fileno())
connection.execute('UPDATE offsets SET committed_offset=committed_offset+? WHERE stage=?',(len(tail),'member_generation'))
os._exit(73)
"""
    crash = subprocess.run([sys.executable,"-B","-c",crash_code,str(work)],capture_output=True,text=True)
    checks["hard_exit_after_fsync_before_database_commit_exercised"] = crash.returncode == 73
    current = collector_api.Collector(SyntheticContext(),work)
    try:
        progress = current.snapshot()
        checks["startup_truncates_to_committed_offset"] = progress["successful_outputs"] == 3 and progress["recovered_uncommitted_bytes"] > 0 and sizes == {name:(work/name).stat().st_size for name in collector_api.FILES.values()}
        final = root/"final.tar.gz"
        archive(final,[("member_generation",1,"same")],"toy-final")
        current.import_archive(final)
        complete = current.snapshot()
        frozen = json.loads((work/"OUTPUTS_FROZEN.json").read_text(encoding="utf-8"))
        checks["final_freeze_requires_full_union_and_hashes_exact_files"] = (
            complete["all_required_outputs_frozen"] and frozen["completed_required_member_calls"] == 2
            and frozen["completed_required_aggregate_calls"] == 2
            and frozen["member_records_sha256"] == collector_api.file_sha(work/collector_api.FILES["member_generation"])
            and frozen["aggregate_records_sha256"] == collector_api.file_sha(work/collector_api.FILES["aggregation"]))
    finally:
        current.close()
    concurrent = root/"concurrent"
    command = [sys.executable,"-B","-X","utf8",str(Path(__file__).resolve()),"--worker",str(concurrent),"--archive",str(first)]
    processes = [subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="utf-8") for _ in range(2)]
    outputs = []
    for process in processes:
        stdout,stderr = process.communicate(timeout=60)
        if process.returncode != 0:
            raise RuntimeError("Concurrent toy worker failed " + stderr)
        outputs.append(json.loads(stdout))
    checks["concurrent_import_commits_each_output_once"] = sum(row["result"]["new_outputs"] for row in outputs) == 2 and outputs[-1]["count"] == 2
    checks["concurrent_import_keeps_one_archive_identity"] = sqlite3.connect(concurrent/"NATIVE_COLLECTION.sqlite").execute("SELECT COUNT(*) FROM archives").fetchone()[0] == 1
    if not all(checks.values()) or len(checks) != 12:
        raise AssertionError(checks)
    report=dict(schema="p2_native_collector_software_qualification_v1",status="PASS",checks=checks,
        collector_sha256=collector_api.file_sha(Path(collector_api.__file__)),
        synthetic_fixture_only=True,real_native_archives_modified=False,new_llm_calls=0,
        reference_values_read=0,deterministic_scores_computed=0,
        private_test_artifacts=str(root))
    destination=Path(__file__).with_name("P2_NATIVE_COLLECTOR_SOFTWARE_QUALIFICATION.json")
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="PASS",checks=len(checks),new_llm_calls=0)))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",type=Path)
    parser.add_argument("--archive",type=Path)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker,args.archive)
    else:
        main()
