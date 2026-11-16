"""Claim-backed, graceful checkpoint handover for the frozen E18 member4 plan.

Default CLI mode only inspects metadata. Real process signalling is reachable
only through --execute on Linux. The broker never calls a model or a scorer.
Its static lease is separate from the mutable heartbeat and completion record.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


BASE_PLAN_SHA="8473a296a638c677da1791867c87c09af441cfb5e90ae5cd9a84feb7e9564502"
DISPATCHER_SHA="5d273799e2c74f0b5cbd40aa6a6c6ae9555fa9892672bc47f3cba174b9c481a2"
WORKER_SHA="17f74192c3ccf10930633f3f1dac48487ce9f62b25c848e2703f03fc6bc9c4f0"
TARGET_PID=3274326
TARGET_GPU="7"
ORIGINAL_TASKS=8127


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def write_once(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())


def write_atomic(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+".tmp-"+str(os.getpid()))
    with temporary.open("w",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def checked_ref(reference):
    if not isinstance(reference,dict) or set(reference)!={"path","sha256"}:
        raise ValueError("An exact artifact path/SHA reference is required")
    path=Path(reference["path"]).resolve(strict=True)
    if file_sha(path)!=reference["sha256"]:
        raise ValueError("Artifact SHA differs "+str(path))
    return path


def summary_signature(path):
    path=Path(path)
    if not path.exists():
        return None
    stat=path.stat()
    return stat.st_ino,stat.st_mtime_ns,file_sha(path)


class LinuxProcesses:
    """Read process/lock evidence and send only the two authorized SIGTERMs."""

    def __init__(self):
        if sys.platform!="linux":
            raise RuntimeError("Real checkpoint control requires Linux")

    def identity(self,pid):
        root=Path("/proc")/str(pid)
        try:
            stat=(root/"stat").read_text()
            rest=stat[stat.rfind(")")+2:].split()
            if rest[0]=="Z":
                return None
            command=[item.decode() for item in (root/"cmdline").read_bytes().split(b"\0") if item]
            fields=dict(line.split(":",1) for line in (root/"status").read_text().splitlines() if ":" in line)
            caught=int(fields["SigCgt"].strip(),16)
            return dict(pid=pid,start_ticks=rest[19],command=command,
                sigterm_caught=bool(caught&(1<<(signal.SIGTERM-1))))
        except FileNotFoundError:
            return None

    def claim_owner(self,path):
        stat=Path(path).stat()
        expected=(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino)
        owners=[]
        for line in Path("/proc/locks").read_text().splitlines():
            tokens=line.split()
            if "->" in tokens or "FLOCK" not in tokens:
                continue
            start=tokens.index("FLOCK")
            values=tokens[start+4].split(":")
            observed=(int(values[0],16),int(values[1],16),int(values[2]))
            if observed==expected:
                owners.append(int(tokens[start+3]))
        if len(set(owners))>1:
            raise RuntimeError("Ambiguous exclusive parent-claim ownership")
        return owners[0] if owners else None

    def terminate(self,pid):
        os.kill(pid,signal.SIGTERM)

    def gpu_context_present(self,pid):
        completed=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],
            check=True,capture_output=True,text=True,timeout=20)
        return str(pid) in {line.strip() for line in completed.stdout.splitlines()}


def command_matches(identity,dispatcher_path,plan_path,plan_sha,gpu):
    if identity is None:
        return False
    command=identity["command"]
    def argument(name):
        return command[command.index(name)+1] if name in command and command.index(name)+1<len(command) else None
    script_paths=[Path(token).resolve() for token in command if token.endswith("e18_gpu_dispatcher.py")]
    observed_plan=argument("--plan")
    return (Path(dispatcher_path).resolve() in script_paths
        and observed_plan is not None and Path(observed_plan).resolve()==Path(plan_path).resolve()
        and argument("--plan-sha256")==plan_sha and argument("--gpu")==str(gpu))


def validate_checkpoint_owner(owner,key,scope_sha,approval_sha,validator):
    """Return SUCCESS or UNEXECUTED, rejecting every unsettled paid owner."""
    owner=Path(owner)
    if not owner.exists():
        return dict(status="UNEXECUTED",task_key_sha256=digest(list(key)))
    fence_path,receipt_path=owner/"ATTEMPT_STATE.json",owner/"LAST_TASK_RECEIPT.json"
    records_path=owner/"records.jsonl"
    if not fence_path.exists() and not receipt_path.exists():
        if all(path.name=="records.jsonl" and path.stat().st_size==0 for path in owner.iterdir()):
            return dict(status="UNEXECUTED",task_key_sha256=digest(list(key)))
        raise RuntimeError("An owner has artifacts without a settled paid fence")
    if not fence_path.is_file() or not receipt_path.is_file() or not records_path.is_file():
        raise RuntimeError("An existing paid owner lacks a complete commit/receipt")
    fence,receipt=read_json(fence_path),read_json(receipt_path)
    if fence.get("status")!="SUCCESS_COMMITTED" or receipt.get("status")!="SUCCESS":
        raise RuntimeError("Reserved, failed or unknown paid owner requires reconciliation")
    if any(fence.get(name)!=value for name,value in dict(scope_sha256=scope_sha,
        approval_sha256=approval_sha,task_key=list(key),stage="member_generation",software_test_only=False).items()):
        raise ValueError("Checkpoint fence identity differs")
    if any(receipt.get(name)!=value for name,value in dict(schema="e18_execution_task_receipt_v2",scope_sha256=scope_sha,
        approval_sha256=approval_sha,stage="member_generation",software_test_only=False).items()):
        raise ValueError("Checkpoint receipt scope/mode differs")
    task=receipt.get("task_key",{})
    if (task.get("question_id"),task.get("generation_seed"),task.get("member_index"))!=tuple(key):
        raise ValueError("Checkpoint receipt semantic key differs")
    native=receipt["native_record"]
    native_sha=digest({name:value for name,value in native.items() if name!="record_sha256"})
    if native_sha!=native.get("record_sha256") or any(value!=native_sha for value in (fence.get("native_record_sha256"),receipt.get("native_record_sha256"))):
        raise ValueError("Native member commit hash differs")
    if native.get("status")!="success" or native.get("cost_status")!="recorded_actual_usage":
        raise RuntimeError("Checkpoint native execution/cost is unsettled")
    with records_path.open(encoding="utf-8") as stream:
        records=[json.loads(line) for line in stream if line.strip()]
    if records!=[native]:
        raise RuntimeError("Checkpoint Store contains another or unsettled physical attempt")
    if (fence.get("native_job_id")!=native.get("job_id") or receipt.get("native_job_id")!=native.get("job_id")
        or receipt.get("physical_attempt_id")!=native.get("attempt_id")):
        raise ValueError("Checkpoint physical native identity differs")
    validator(*key,native)
    return dict(status="SUCCESS",task_key_sha256=digest(list(key)),native_record_sha256=native_sha,
        native_receipt_file_sha256=file_sha(receipt_path))


class CheckpointBroker:
    def __init__(self,plan_path,plan_sha,dispatcher,processes,lease_root,child_plan_path,*,
                 target_pid=TARGET_PID,target_gpu=TARGET_GPU,owner_pid=None,expected_tasks=ORIGINAL_TASKS,
                 clock=time.monotonic,sleeper=time.sleep,poll_seconds=2,timeout_seconds=1800,
                 child_timeout_seconds=172800,validator=None,scope_sha=None):
        self.plan_path=Path(plan_path).resolve()
        self.plan_sha=plan_sha
        if file_sha(self.plan_path)!=plan_sha:
            raise ValueError("Original dispatch-plan SHA differs")
        self.dispatcher,self.worker=dispatcher,dispatcher.worker
        self.manifests=dispatcher.admit_plan(read_json(self.plan_path))
        if len(self.manifests)!=12:
            raise ValueError("Exactly twelve original parent manifests are required")
        self.parents=[item for item in self.manifests if item[1].get("member_index")==4]
        if len(self.parents)!=2 or any(item[1].get("stage")!="member_generation" for item in self.manifests):
            raise ValueError("Exactly two member4 parent shards are required")
        self.parent_keys=[]
        for _,manifest in self.parents:
            self.parent_keys.extend(tuple(key) for key in read_json(checked_ref(manifest["task_keys_ref"])))
        if len(self.parent_keys)!=expected_tasks or len(set(self.parent_keys))!=expected_tasks or any(key[2]!=4 for key in self.parent_keys):
            raise ValueError("Member4 original task-key union differs")
        self.output_root=Path(self.parents[0][1]["owned_output_root"]).resolve()
        self.state_root=self.output_root/"_dispatcher/plans"/plan_sha
        self.processes,self.target_pid,self.target_gpu=processes,target_pid,str(target_gpu)
        self.owner_pid=os.getpid() if owner_pid is None else owner_pid
        self.root=Path(lease_root).resolve()
        if not self.root.is_relative_to(self.output_root):
            raise ValueError("Broker ownership must stay inside the approved output root")
        self.child_plan_path=Path(child_plan_path).resolve()
        self.clock,self.sleeper,self.poll_seconds,self.timeout=clock,sleeper,poll_seconds,timeout_seconds
        self.child_timeout=child_timeout_seconds
        self.held={}
        self.stop_requested=False
        self.former_identity=None
        self.active_parent=None
        self.validator=validator
        self.scope_sha=scope_sha
        self.approval_sha=self.parents[0][1]["approval_ref"]["sha256"]
        self.stage="INITIALIZED"

    def status_path(self,reference):
        return self.state_root/(reference["sha256"]+".status.json")

    def claim_path(self,reference):
        return self.state_root/(reference["sha256"]+".claim")

    def heartbeat(self):
        if self.root.exists():
            write_atomic(self.root/"HEARTBEAT.json",dict(schema="e18_member4_checkpoint_heartbeat_v1",owner_pid=self.owner_pid,
                former_pid=self.target_pid,state=self.stage,at_unix=time.time(),held_parent_claims=sorted(self.held)))

    def event(self,event,**values):
        with (self.root/"EVENTS.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event=event,at_unix=time.time(),**values),ensure_ascii=False)+"\n")
            stream.flush();os.fsync(stream.fileno())

    def deadline_wait(self,predicate,message,*,timeout_seconds=None):
        deadline=self.clock()+(self.timeout if timeout_seconds is None else timeout_seconds)
        while True:
            if self.stop_requested:
                raise RuntimeError("Broker stop requested; no completion is certified")
            value=predicate()
            if value:
                return value
            if self.clock()>=deadline:
                raise TimeoutError(message)
            self.heartbeat();self.sleeper(self.poll_seconds)

    def inspect_target(self):
        identity=self.processes.identity(self.target_pid)
        if not command_matches(identity,self.dispatcher.__file__,self.plan_path,self.plan_sha,self.target_gpu):
            raise PermissionError("Target PID command/plan/GPU identity differs")
        active=[]
        for reference,manifest in self.manifests:
            status=read_json(self.status_path(reference)) if self.status_path(reference).exists() else {}
            if status.get("status")=="RUNNING" and status.get("pid")==self.target_pid:
                active.append((reference,manifest))
        if len(active)!=1 or active[0][1]["member_index"]!=4:
            raise PermissionError("Target PID is not the unique RUNNING member4 parent owner")
        if self.processes.claim_owner(self.claim_path(active[0][0]))!=self.target_pid:
            raise PermissionError("Target PID lacks the active original parent FLOCK")
        self.former_identity,self.active_parent=identity,active[0]
        return dict(status="DRY_RUN_METADATA_VALID",base_member_plan_sha256=self.plan_sha,
            target_pid=self.target_pid,target_gpu=self.target_gpu,target_member_index=4,
            target_sigterm_handler_caught=identity["sigterm_caught"],original_member4_tasks=len(self.parent_keys),
            parent_shards=[dict(manifest_sha256=ref["sha256"],claim_path=str(self.claim_path(ref))) for ref,_ in self.parents],
            process_signals_sent=0,model_calls=0,gold_reads=0)

    def hold_available(self):
        if (self.state_root/"STOP_REQUIRES_RECONCILIATION.json").exists():
            raise RuntimeError("The original plan is stopped for reconciliation")
        for reference,_ in self.manifests:
            key=reference["sha256"]
            if key in self.held:
                continue
            status=read_json(self.status_path(reference)) if self.status_path(reference).exists() else {}
            if status.get("status")=="DONE":
                continue
            if status.get("status")=="FAILED":
                raise RuntimeError("An original parent is FAILED")
            claim=self.dispatcher.ShardClaim(self.claim_path(reference))
            if claim.acquire():
                self.held[key]=claim

    def same_former_identity(self):
        observed=self.processes.identity(self.target_pid)
        if observed is not None and observed["start_ticks"]!=self.former_identity["start_ticks"]:
            raise PermissionError("Target PID was reused by another process")
        return observed

    def checkpoint_stop(self):
        self.hold_available()
        active_sha=self.active_parent[0]["sha256"]
        other_sha=next(ref["sha256"] for ref,_ in self.parents if ref["sha256"]!=active_sha)
        if other_sha not in self.held:
            raise RuntimeError("The unowned second member4 parent must be held before signalling")
        if active_sha in self.held:
            raise RuntimeError("Target left its active parent before the checkpoint request")
        identity=self.same_former_identity()
        if identity is None or self.processes.claim_owner(self.claim_path(self.active_parent[0]))!=self.target_pid:
            raise PermissionError("Target active owner changed before checkpoint signalling")
        summary_path=self.output_root/"_dispatcher"/active_sha/"SUMMARY.json"
        before=summary_signature(summary_path)
        self.stage="CHECKPOINT_REQUESTED"
        self.event("FIRST_SIGTERM_REQUESTED",target_pid=self.target_pid,sigterm_caught=identity["sigterm_caught"])
        try:
            self.processes.terminate(self.target_pid)
        except ProcessLookupError:
            self.event("TARGET_EXITED_AT_FIRST_SIGNAL_WINDOW")
        evidence={}
        def ready():
            self.hold_available()
            observed=self.same_former_identity()
            fresh=summary_signature(summary_path)
            summary=read_json(summary_path) if fresh is not None and fresh!=before else None
            valid_summary=(summary is not None and summary.get("stop_requested") is True
                and summary.get("manifest_sha256")==active_sha and summary.get("stage")=="member_generation"
                and summary.get("profile")==4 and summary.get("software_test_only") is False
                and summary.get("failures")==[])
            if active_sha not in self.held:
                return False
            if observed is None:
                evidence.update(kind="SUMMARY_CONFIRMED" if valid_summary else "PROCESS_EXIT_AND_NATIVE_COMMIT_CONFIRMED",
                    summary_file_sha256=file_sha(summary_path) if valid_summary else None)
                return True
            if not valid_summary or observed["sigterm_caught"]:
                return False
            for reference,_ in self.manifests:
                if self.processes.claim_owner(self.claim_path(reference))==self.target_pid:
                    raise RuntimeError("Former PID acquired another parent; no blind second signal")
            evidence.update(kind="SUMMARY_CONFIRMED",summary_file_sha256=file_sha(summary_path))
            self.event("SECOND_SIGTERM_AT_QUIESCENT_DISPATCHER",target_pid=self.target_pid)
            try:
                self.processes.terminate(self.target_pid)
            except ProcessLookupError:
                pass
            return True
        self.deadline_wait(ready,"No safe checkpoint/parent ownership evidence arrived")
        self.deadline_wait(lambda:self.same_former_identity() is None,"Former dispatcher did not exit")
        self.deadline_wait(lambda:not self.processes.gpu_context_present(self.target_pid),"Former PID retains a GPU context")
        for reference,_ in self.parents:
            if reference["sha256"] not in self.held or self.processes.claim_owner(self.claim_path(reference))!=self.owner_pid:
                raise RuntimeError("Broker does not own both original parent claims")
        for key in list(self.held):
            if key not in {ref["sha256"] for ref,_ in self.parents}:
                self.held.pop(key).release()
        self.stage="FORMER_EXITED_PARENT_CLAIMS_HELD"
        self.event("FORMER_EXIT_AND_GPU_CONTEXT_RELEASE_VERIFIED",evidence=evidence)
        return evidence

    def audit_tasks(self,require_all=False):
        audit=[];remaining=[]
        for key in self.parent_keys:
            owner=self.output_root/self.scope_sha/"member_generation"/digest(list(key))
            item=validate_checkpoint_owner(owner,key,self.scope_sha,self.approval_sha,self.validator)
            audit.append(item)
            if item["status"]=="UNEXECUTED":
                remaining.append(list(key))
        if require_all and remaining:
            raise RuntimeError("Child DONE flags do not imply all original native executions committed")
        return audit,remaining

    def write_lease(self,evidence):
        audit,remaining=self.audit_tasks()
        remaining_path=self.root/"remaining_task_keys.json"
        write_once(remaining_path,remaining)
        audit_path=self.root/"CHECKPOINT_SOURCE_AUDIT.json"
        write_once(audit_path,dict(schema="e18_member4_native_checkpoint_audit_v1",tasks=audit,
            original_tasks=len(audit),checkpoint_successes=len(audit)-len(remaining),pending_or_unknown_attempts=0))
        lease=dict(schema="e18_member4_checkpoint_lease_v1",status="LEASED",base_member_plan_sha256=self.plan_sha,
            execution_scope_sha256=self.scope_sha,approval_sha256=self.approval_sha,target_member_index=4,owner_pid=self.owner_pid,
            parent_shards=[dict(manifest_sha256=ref["sha256"],claim_path=str(self.claim_path(ref)),
                task_keys_sha256=manifest["task_keys_ref"]["sha256"]) for ref,manifest in self.parents],
            remaining_task_keys_ref=dict(path=str(remaining_path),sha256=file_sha(remaining_path)),
            original_member4_tasks=len(self.parent_keys),checkpoint_successes=len(audit)-len(remaining),remaining_count=len(remaining),
            former_pid=self.target_pid,checkpoint_stop_verified=True,pending_or_unknown_attempts=0,
            checkpoint_evidence=evidence,checkpoint_source_audit_sha256=file_sha(audit_path),software_test_only=False)
        lease_path=self.root/"LEASE.json"
        write_once(lease_path,lease)
        self.stage="LEASED_WAITING_FOR_CHILD_PLAN"
        self.heartbeat();self.event("STATIC_LEASE_COMMITTED",lease_sha256=file_sha(lease_path),remaining=len(remaining))
        return lease,remaining

    def admit_child_plan(self,remaining):
        plan_sha=file_sha(self.child_plan_path)
        plan=read_json(self.child_plan_path)
        lease_ref=dict(path=str(self.root/"LEASE.json"),sha256=file_sha(self.root/"LEASE.json"))
        if plan.get("schema")!="e18_member4_dispatch_plan_v3" or plan.get("checkpoint_lease_ref")!=lease_ref or not plan.get("manifest_refs"):
            raise ValueError("Child plan must bind the exact static checkpoint lease")
        base=self.parents[0][1]
        seen=set()
        refs=[]
        for reference in plan["manifest_refs"]:
            manifest=read_json(checked_ref(reference))
            if manifest.get("checkpoint_lease_ref")!=lease_ref or manifest.get("member_index")!=4 or manifest.get("stage")!="member_generation" or manifest.get("software_test_only") is not False:
                raise ValueError("Child manifest checkpoint/profile/mode differs")
            if any(manifest.get(key)!=base.get(key) for key in ("scope_ref","approval_ref","question_records_ref","owned_output_root","adapter_ref")):
                raise ValueError("Child manifest changes an original scientific source or owner")
            key_rows=read_json(checked_ref(manifest["task_keys_ref"]))
            keys={tuple(key) for key in key_rows}
            if len(keys)!=len(key_rows):
                raise ValueError("Duplicate keys inside a checkpoint child shard")
            if seen&keys:
                raise ValueError("Overlapping child task keys")
            seen.update(keys);refs.append(reference)
        if seen!={tuple(key) for key in remaining}:
            raise ValueError("Child plan is not the exact remaining task union")
        return plan_sha,refs

    def wait_for_completion(self,lease,remaining,completion_delay):
        child_sha=None
        if remaining:
            self.deadline_wait(lambda:self.child_plan_path.is_file(),"Root has not supplied the checkpoint child plan",timeout_seconds=self.child_timeout)
            child_sha,refs=self.admit_child_plan(remaining)
            state=self.output_root/"_dispatcher/plans"/child_sha
            self.stage="LEASED_CHILD_PLAN_RUNNING"
            def children_done():
                if file_sha(self.child_plan_path)!=child_sha:
                    raise ValueError("Child plan changed after lease admission")
                if (state/"STOP_REQUIRES_RECONCILIATION.json").exists():
                    raise RuntimeError("A checkpoint child plan stopped for reconciliation")
                statuses=[]
                for reference in refs:
                    path=state/(reference["sha256"]+".status.json")
                    status=read_json(path) if path.exists() else {}
                    if status.get("status")=="FAILED":
                        raise RuntimeError("A checkpoint child shard failed")
                    statuses.append(status.get("status")=="DONE" and status.get("manifest_sha256")==reference["sha256"])
                return all(statuses)
            self.deadline_wait(children_done,"Checkpoint child plan has not completed",timeout_seconds=self.child_timeout)
        audit,_=self.audit_tasks(require_all=True)
        complete=dict(schema="e18_member4_checkpoint_completion_v1",status="COMPLETED",owner_pid=self.owner_pid,
            lease_sha256=file_sha(self.root/"LEASE.json"),child_plan_sha256=child_sha,original_tasks_verified=len(audit),
            checkpoint_successes=lease["checkpoint_successes"],new_remaining_tasks=lease["remaining_count"],
            native_source_audit_sha256=digest(audit),at_unix=time.time())
        for reference,manifest in self.parents:
            write_atomic(self.status_path(reference),dict(status="DONE",manifest_sha256=reference["sha256"],
                completion_kind="CHECKPOINT_CHILD_PLAN_AND_ALL_NATIVE_COMMITS_VERIFIED",checkpoint_lease_sha256=complete["lease_sha256"],
                child_plan_sha256=child_sha,verified_native_tasks=len(read_json(checked_ref(manifest["task_keys_ref"]))),finished_at_unix=time.time()))
        write_once(self.root/"LEASE_COMPLETED.json",complete)
        self.stage="COMPLETED_HOLDING_PARENT_CLAIMS"
        self.event("ALL_ORIGINAL_MEMBER4_NATIVE_COMMITS_VERIFIED",tasks=len(audit))
        deadline=self.clock()+completion_delay
        while self.clock()<deadline:
            self.heartbeat();self.sleeper(min(self.poll_seconds,deadline-self.clock()))
        return complete

    def run(self,*,completion_delay=60):
        if self.validator is None or not isinstance(self.scope_sha,str) or len(self.scope_sha)!=64:
            raise ValueError("Native dependency validator and canonical execution scope are required")
        self.inspect_target()
        self.root.mkdir(parents=True,exist_ok=False)
        self.event("BROKER_STARTED",owner_pid=self.owner_pid,former_pid=self.target_pid)
        try:
            evidence=self.checkpoint_stop()
            lease,remaining=self.write_lease(evidence)
            return self.wait_for_completion(lease,remaining,completion_delay)
        except BaseException as error:
            self.stage="STOP_REQUIRES_RECONCILIATION"
            write_atomic(self.root/"STOP_REQUIRES_RECONCILIATION.json",dict(status=self.stage,error_type=type(error).__name__,
                error=str(error),owner_pid=self.owner_pid,former_pid=self.target_pid,lease_exists=(self.root/"LEASE.json").exists(),at_unix=time.time()))
            self.heartbeat()
            raise
        finally:
            for claim in self.held.values():
                claim.release()
            self.held.clear()


def load_dispatcher(path):
    path=Path(path).resolve(strict=True)
    if file_sha(path)!=DISPATCHER_SHA or file_sha(path.with_name("e18_gpu_worker.py"))!=WORKER_SHA:
        raise ValueError("Original dispatcher/worker source identity differs")
    sys.path.insert(0,str(path.parent))
    specification=importlib.util.spec_from_file_location("e18_gpu_dispatcher",path)
    module=importlib.util.module_from_spec(specification)
    sys.modules[specification.name]=module
    specification.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--plan-sha256",default=BASE_PLAN_SHA)
    parser.add_argument("--dispatcher",type=Path,default=Path(__file__).with_name("e18_gpu_dispatcher.py"))
    parser.add_argument("--lease-root",type=Path,required=True)
    parser.add_argument("--child-plan",type=Path,required=True)
    parser.add_argument("--target-pid",type=int,default=TARGET_PID)
    parser.add_argument("--target-gpu",default=TARGET_GPU)
    parser.add_argument("--timeout-seconds",type=float,default=1800)
    parser.add_argument("--child-timeout-seconds",type=float,default=172800)
    parser.add_argument("--poll-seconds",type=float,default=2)
    parser.add_argument("--execute",action="store_true")
    arguments=parser.parse_args()
    if arguments.plan_sha256!=BASE_PLAN_SHA or arguments.target_pid!=TARGET_PID or arguments.target_gpu!=TARGET_GPU:
        raise PermissionError("The authorized original plan, former PID and GPU must match")
    dispatcher=load_dispatcher(arguments.dispatcher)
    broker=CheckpointBroker(arguments.plan,arguments.plan_sha256,dispatcher,LinuxProcesses(),arguments.lease_root,arguments.child_plan,
        target_pid=arguments.target_pid,target_gpu=arguments.target_gpu,poll_seconds=arguments.poll_seconds,
        timeout_seconds=arguments.timeout_seconds,child_timeout_seconds=arguments.child_timeout_seconds)
    if not arguments.execute:
        print(json.dumps(broker.inspect_target(),ensure_ascii=False,indent=2))
        return
    module,adapter,_,_,_,_=dispatcher.worker.load_worker(broker.parents[0][1],software_test_only=False)
    broker.scope_sha=module.canonical_digest(adapter.scope)
    broker.validator=adapter.checked_member_dependency
    for signum in (signal.SIGINT,signal.SIGTERM):
        signal.signal(signum,lambda *_:setattr(broker,"stop_requested",True))
    result=broker.run()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
