"""Persistent P2 subset aggregation using the explicit append-only source registry."""
from __future__ import annotations

import argparse
from collections import defaultdict,deque
import json
import os
from pathlib import Path
import signal
import time

from p2_aggregation_follower import MemberJournalTail,publish
import p2_aggregation_worker as worker
from p2_manifest_registry import iter_successes,load_registry


def member_journal_paths(registry):
    return sorted({entry["journal_path"] for entry in registry["entries"] if entry["stage"]=="member_generation"})


def restored_successes(registry,keys):
    allowed=set(keys)
    return {key for _,_,key in iter_successes(registry,"aggregation") if key in allowed}


class RegisteredMemberTail:
    def __init__(self,path,worker_id,manifest_ref,worker_ref,*,software_test_only=False):
        self.path=Path(path)
        self.worker_id=worker_id
        self.manifest_ref=manifest_ref
        self.worker_ref=worker_ref
        self.software_test_only=software_test_only
        self.tail=MemberJournalTail([])
        self.registry=None
        self.signature=None
        self.last_check=0.0
        self.refresh(force=True)

    def refresh(self,force=False):
        now=time.monotonic()
        if not force and now-self.last_check<2:
            return
        self.last_check=now
        stat=self.path.stat()
        signature=(stat.st_size,stat.st_mtime_ns)
        if signature==self.signature:
            return
        current=load_registry(self.path)
        if current["software_test_only"] is not self.software_test_only:
            raise PermissionError("The native follower and registry execution modes differ")
        if self.registry is not None:
            previous=self.registry
            if (current["common"]!=previous["common"] or current["event_count"]<previous["event_count"]
                    or current["event_hashes"][previous["event_count"]-1]!=previous["last_event_sha256"]):
                raise ValueError("An active evidence registry changed its existing prefix")
        own=[entry for entry in current["entries"] if entry["worker_id"]==self.worker_id]
        if len(own)!=1 or own[0]["manifest_ref"]!=self.manifest_ref or own[0]["worker_source_ref"]!=self.worker_ref:
            raise ValueError("Follower identity differs from its registered manifest or source")
        for path in member_journal_paths(current):
            self.tail.positions.setdefault(Path(path),0)
        self.registry=current
        self.signature=signature

    def completed(self):
        self.refresh()
        yield from self.tail.completed()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--manifest-sha256",required=True)
    parser.add_argument("--gpu",required=True)
    parser.add_argument("--registry",type=Path,required=True)
    parser.add_argument("--worker-id",required=True)
    args=parser.parse_args()
    if worker.file_sha(args.manifest)!=args.manifest_sha256:
        raise ValueError("The subset manifest changed")
    manifest=worker.read_json(args.manifest)
    if manifest.get("software_test_only") is not False:
        raise PermissionError("A production subset manifest is required")
    manifest_ref=dict(path=str(args.manifest.resolve()),sha256=args.manifest_sha256)
    worker_ref=dict(path=str(Path(__file__).resolve()),sha256=worker.file_sha(__file__))
    source_tail=RegisteredMemberTail(args.registry,args.worker_id,manifest_ref,worker_ref)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    stop=[False]
    for signum in (signal.SIGTERM,signal.SIGINT):
        signal.signal(signum,lambda *_:stop.__setitem__(0,True))
    with worker.GPUOwnership(manifest["gpu_lock_root"],args.gpu,args.manifest_sha256):
        module,adapter,keys,_,backend,external=worker.load_worker(manifest)
        adapter._worker_software_mode=False
        scope_sha=module.canonical_digest(adapter.scope)
        if scope_sha!=source_tail.registry["scope_canonical_sha256"]:
            raise ValueError("The registered scope differs from the original execution adapter")
        journal_root=Path(manifest["owned_output_root"])/"_dispatcher"/args.manifest_sha256
        journal_root.mkdir(parents=True,exist_ok=True)
        journal=journal_root/"EVENTS.jsonl"
        completed=restored_successes(source_tail.registry,keys)
        restored_count=len(completed)
        pending={}
        dependants=defaultdict(list)
        for key in keys:
            if key in completed:
                continue
            pending[key]=3
            for member in module.TEAMS[key[2]]:
                dependants[(key[0],key[1],member)].append(key)
        seen_members=set()
        ready=deque()
        tokenizer=None
        started_cpu,started_wall=time.process_time(),time.perf_counter()
        last_status=0.0

        def status(state):
            return dict(status=state,worker_id=args.worker_id,stage="aggregation",scope_sha256=scope_sha,
                        planned=len(keys),completed=len(completed),restored_successes=restored_count,
                        observed_member_successes=len(seen_members),ready=len(ready),
                        registry_head=source_tail.registry["last_event_sha256"],
                        registry_entries=len(source_tail.registry["entries"]),backend_loads=backend.context.constructions,
                        pid=os.getpid(),at_unix=time.time(),gold_reads=0)

        while len(completed)<len(keys) and not stop[0]:
            for member_key in source_tail.completed():
                if member_key in seen_members:
                    continue
                if member_key not in adapter.required_members:
                    raise ValueError("A member success escaped the original execution scope")
                seen_members.add(member_key)
                for key in dependants.get(member_key,()):
                    pending[key]-=1
                    if pending[key]==0:
                        ready.append(key)
            if not ready:
                now=time.monotonic()
                if now-last_status>30:
                    publish(journal_root/"SUMMARY.json",status("WAITING_FOR_COMMITTED_MEMBERS"))
                    last_status=now
                time.sleep(2)
                continue
            key=ready.popleft()
            if key in completed:
                continue
            records={member:worker.member_dependency(module,adapter,Path(manifest["owned_output_root"]),external,
                     (key[0],key[1],member)) for member in module.TEAMS[key[2]]}
            if tokenizer is None:
                tokenizer,alias_receipt=worker.context_tokenizer(module,adapter,backend)
                publish(journal_root/"TOKENIZER_ALIAS_RECEIPT.json",alias_receipt)
            task=dict(stage="aggregation",question_id=key[0],generation_seed=key[1],combination_index=key[2])
            worker.append_event(journal,dict(event="TASK_ENTER",task=task,worker_id=args.worker_id,at_unix=time.time()))
            try:
                receipt=module.run_approved_task(adapter,task,manifest["approval_ref"],manifest["owned_output_root"],
                                                 backend,tokenizer=tokenizer,member_records=records)
            except BaseException as error:
                worker.append_event(journal,dict(event="STOP_REQUIRES_RECONCILIATION",task=task,
                        error_type=type(error).__name__,error=str(error),at_unix=time.time()))
                raise
            worker.append_event(journal,dict(event="TASK_RESULT",task=task,status=receipt["status"],
                    physical_attempt_id=receipt["physical_attempt_id"],native_record_sha256=receipt["native_record_sha256"],
                    from_cache=receipt["from_cache"],actual_usage=worker.native_usage(receipt["native_record"]),at_unix=time.time()))
            if receipt["status"]!="SUCCESS":
                raise RuntimeError("Native aggregation requires owner reconciliation")
            completed.add(key)
            publish(journal_root/"SUMMARY.json",status("RUNNING"))
        final=status("COMPLETE" if len(completed)==len(keys) else "SAFELY_STOPPED")
        final.update(stopped=stop[0],cpu_seconds=time.process_time()-started_cpu,wall_seconds=time.perf_counter()-started_wall)
        publish(journal_root/"SUMMARY.json",final)


if __name__=="__main__":
    main()
