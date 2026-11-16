"""Incrementally close committed P2 native owners into local-transfer TAR batches."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import time

import p2_execution_adapter as adapter


ROOT = Path("/data2/chendong/e20_p2_execution")
OWNER_FILES = ("ATTEMPT_STATE.json", "LAST_TASK_RECEIPT.json", "records.jsonl")


def encoded(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()


def stable_bytes(path):
    before=path.stat()
    data=path.read_bytes()
    after=path.stat()
    signature=lambda row:(row.st_dev,row.st_ino,row.st_size,row.st_mtime_ns)
    if signature(before)!=signature(after) or len(data)!=before.st_size:
        raise RuntimeError("Native output changed during closure")
    return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-tasks",type=int,default=500)
    args=parser.parse_args()
    scope=adapter.read_json(ROOT/"inputs/EXECUTION_SCOPE_RUNTIME.json")
    scope_sha=adapter.canonical_digest(scope)
    required={stage:set(map(tuple,scope[field])) for stage,field in
              (("member_generation","required_member_keys"),("aggregation","required_aggregate_keys"))}
    exports=ROOT/"exports"
    exports.mkdir(exist_ok=True)
    connection=sqlite3.connect(exports/"exported.sqlite",timeout=30)
    connection.execute("CREATE TABLE IF NOT EXISTS exported (stage TEXT, digest TEXT, batch TEXT, PRIMARY KEY(stage,digest))")
    exported={(row[0],row[1]) for row in connection.execute("SELECT stage,digest FROM exported")}
    candidates=[]
    for manifest_path in sorted((ROOT/"manifests").glob("*_v2.json")):
        manifest=adapter.read_json(manifest_path)
        manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        journal=ROOT/"owned/_dispatcher"/manifest_sha/"EVENTS.jsonl"
        if not journal.is_file():
            continue
        for line in journal.read_text().splitlines():
            event=json.loads(line)
            if event.get("event")!="TASK_RESULT" or event.get("status")!="SUCCESS":
                continue
            task=event["task"]
            stage=task["stage"]
            key=(task["question_id"],task["generation_seed"],task["member_index" if stage=="member_generation" else "combination_index"])
            if key not in required[stage]:
                raise ValueError("Journal key escaped frozen scope")
            digest=adapter.canonical_digest(list(key))
            if (stage,digest) not in exported:
                candidates.append((stage,key,digest))
                exported.add((stage,digest))
    selected=candidates[:args.max_tasks]
    if not selected:
        print(json.dumps(dict(status="NO_NEW_COMMITTED_TASKS",new_tasks=0,gold_reads=0)))
        return
    batch="p2_native_"+str(time.time_ns())
    target=exports/(batch+".tar.gz")
    rows=[]
    prompt=completion=0
    with tarfile.open(target,"x:gz") as archive:
        for stage,key,digest in selected:
            owner=ROOT/"owned"/scope_sha/stage/digest
            payload={name:stable_bytes(owner/name) for name in OWNER_FILES}
            fence=json.loads(payload["ATTEMPT_STATE.json"])
            receipt=json.loads(payload["LAST_TASK_RECEIPT.json"])
            native=receipt["native_record"]
            native_sha=adapter.canonical_digest({name:value for name,value in native.items() if name!="record_sha256"})
            if (receipt.get("schema")!="p2_execution_task_receipt_v1" or receipt.get("cohort")!=adapter.COHORT
                    or receipt.get("scope_sha256")!=scope_sha or receipt.get("status")!="SUCCESS"
                    or receipt.get("software_test_only") is not False or fence.get("status")!="SUCCESS_COMMITTED"
                    or fence.get("scope_sha256")!=scope_sha or native.get("record_sha256")!=native_sha
                    or receipt.get("native_record_sha256")!=native_sha or fence.get("native_record_sha256")!=native_sha
                    or native.get("status")!="success" or native.get("cost_status")!="recorded_actual_usage"):
                raise ValueError("Native owner integrity or actual-cost qualification differs")
            if [json.loads(line) for line in payload["records.jsonl"].splitlines()]!=[native]:
                raise ValueError("Native Store does not contain exactly the committed record")
            usage=native.get("result") or native.get("actual_usage") or {}
            if any(type(usage.get(key)) is not int or usage[key]<0 for key in ("prompt_tokens","completion_tokens")):
                raise ValueError("Native actual token usage missing")
            prompt+=usage["prompt_tokens"]
            completion+=usage["completion_tokens"]
            file_rows=[]
            for name,data in payload.items():
                relative=(owner/name).relative_to(ROOT).as_posix()
                info=tarfile.TarInfo(relative)
                info.size=len(data)
                archive.addfile(info,io.BytesIO(data))
                file_rows.append(dict(path=relative,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
            rows.append(dict(stage=stage,task_key=list(key),digest=digest,native_record_sha256=native_sha,files=file_rows))
        manifest=dict(schema="p2_closed_native_export_v1",batch=batch,scope_sha256=scope_sha,tasks=rows,
                      prompt_tokens=prompt,completion_tokens=completion,gold_reads=0)
        data=encoded(manifest)
        info=tarfile.TarInfo("P2_NATIVE_EXPORT_MANIFEST.json")
        info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
    connection.executemany("INSERT INTO exported(stage,digest,batch) VALUES(?,?,?)",[(stage,digest,batch) for stage,_,digest in selected])
    connection.commit()
    connection.close()
    receipt=dict(status="CLOSED_NATIVE_BATCH",path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                 bytes=target.stat().st_size,tasks=len(rows),prompt_tokens=prompt,completion_tokens=completion,
                 scope_sha256=scope_sha,gold_reads=0,new_llm_calls=0)
    (exports/(batch+".json")).write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__=="__main__":
    main()
