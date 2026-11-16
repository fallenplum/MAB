"""Run one unused original E18 shard through its unchanged manifest and journal.

The caller admits the physical GPU/extra-slot lock root after a live resource
check. This module creates no scope, task list or modified worker manifest.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import e18_gpu_dispatcher as dispatcher
import e18_gpu_worker as worker


def run_one(plan,plan_sha,*,manifest_sha=None,member_index=2,gpu="0",gpu_lock_root=None,
            software_test_only=False,execute=None):
    """Claim only an unused shard, return BUSY without taking over any owner."""
    if not str(gpu).isdigit() or type(member_index) is not int or not 0<=member_index<6:
        raise ValueError("Explicit physical GPU and original member index are required")
    if execute is not None and not software_test_only:
        raise PermissionError("Production must call the original worker entry point")
    execute=execute or worker.execute_manifest
    admitted=dispatcher.admit_plan(plan)
    if any(manifest.get("software_test_only") is not software_test_only for _,manifest in admitted):
        raise PermissionError("Plan execution mode differs")
    candidates=[(reference,manifest) for reference,manifest in admitted
        if manifest.get("stage")=="member_generation" and manifest.get("member_index")==member_index
        and (manifest_sha is None or reference["sha256"]==manifest_sha)]
    if not candidates:
        raise ValueError("Selected manifest is not the requested original member shard")
    output_root=Path(admitted[0][1]["owned_output_root"]).resolve()
    state_root=output_root/"_dispatcher/plans"/plan_sha
    state_root.mkdir(parents=True,exist_ok=True)
    stop_path=state_root/"STOP_REQUIRES_RECONCILIATION.json"
    base=dict(plan_sha256=plan_sha,member_index=member_index,physical_gpu=str(gpu),
        original_plan_and_manifest_unchanged=True,software_test_only=software_test_only)
    if stop_path.exists():
        return dict(base,status="STOP_REQUIRES_RECONCILIATION",executed_shards=0)
    for reference,_ in admitted:
        existing=state_root/(reference["sha256"]+".status.json")
        if existing.exists() and worker.read_json(existing).get("status")=="FAILED":
            return dict(base,status="STOP_REQUIRES_RECONCILIATION",executed_shards=0)
    unavailable=[]
    for reference,manifest in candidates:
        checksum=reference["sha256"]
        status_path=state_root/(checksum+".status.json")
        if status_path.exists():
            state=worker.read_json(status_path).get("status")
            unavailable.append(dict(manifest_sha256=checksum,status=state))
            if manifest_sha is not None:
                return dict(base,status="ALREADY_DONE" if state=="DONE" else "BUSY_OR_ALREADY_USED",
                    manifest_sha256=checksum,original_shard_status=state,executed_shards=0)
            continue
        claim=dispatcher.ShardClaim(state_root/(checksum+".claim"))
        if not claim.acquire():
            unavailable.append(dict(manifest_sha256=checksum,status="CLAIM_BUSY"))
            continue
        try:
            if stop_path.exists():
                return dict(base,status="STOP_REQUIRES_RECONCILIATION",executed_shards=0)
            if status_path.exists():
                unavailable.append(dict(manifest_sha256=checksum,status=worker.read_json(status_path).get("status")))
                continue
            lock_root=Path(gpu_lock_root) if gpu_lock_root is not None else Path(manifest["gpu_lock_root"])
            ownership=worker.GPUOwnership(lock_root,str(gpu),plan_sha)
            try:
                ownership.__enter__()
            except RuntimeError:
                return dict(base,status="GPU_BUSY",manifest_sha256=checksum,executed_shards=0,
                    admitted_gpu_lock_root=str(lock_root.resolve()))
            try:
                # Recheck both original controls before marking this unused shard.
                if stop_path.exists() or status_path.exists():
                    return dict(base,status="BUSY_OR_STOPPED_BEFORE_EXECUTION",manifest_sha256=checksum,executed_shards=0)
                worker.save_json(status_path,dict(status="RUNNING",manifest_sha256=checksum,pid=os.getpid(),
                    started_at_unix=time.time(),software_test_only=software_test_only))
                try:
                    summary=execute(manifest,checksum,software_test_only=software_test_only)
                    if not isinstance(summary,dict) or any(type(summary.get(name)) is not int or summary[name]<0
                        for name in ("completed","planned")) or summary["completed"]>summary["planned"] or "failures" not in summary:
                        raise ValueError("Original worker returned an invalid completion summary")
                    if summary.get("failures"):
                        raise RuntimeError("Original worker returned failed physical attempts")
                except BaseException as error:
                    failed=dict(status="FAILED",manifest_sha256=checksum,error_type=type(error).__name__,at_unix=time.time())
                    worker.save_json(status_path,failed)
                    worker.save_json(stop_path,failed)
                    raise
                done=summary["completed"]==summary["planned"]
                worker.save_json(status_path,dict(status="DONE" if done else "WAITING_FOR_DEPENDENCIES",
                    manifest_sha256=checksum,summary=summary,finished_at_unix=time.time()))
                return dict(base,status="DONE" if done else "WAITING_FOR_DEPENDENCIES",manifest_sha256=checksum,
                    manifest_ref=reference,executed_shards=1,state_root=str(state_root),
                    admitted_gpu_lock_root=str(lock_root.resolve()),summary=summary)
            finally:
                ownership.__exit__(None,None,None)
        finally:
            claim.release()
    return dict(base,status="BUSY_OR_NO_UNUSED_SHARD",executed_shards=0,observed_original_states=unavailable)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--plan-sha256",required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--member-index",type=int,default=2)
    parser.add_argument("--gpu",required=True)
    parser.add_argument("--gpu-lock-root",type=Path)
    args=parser.parse_args()
    if worker.file_sha(args.plan)!=args.plan_sha256:
        raise ValueError("Original dispatch plan bytes differ")
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    print(json.dumps(run_one(worker.read_json(args.plan),args.plan_sha256,
        manifest_sha=args.manifest_sha256,member_index=args.member_index,gpu=args.gpu,
        gpu_lock_root=args.gpu_lock_root),ensure_ascii=False,allow_nan=False))
