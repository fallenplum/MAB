"""Toy scheduling/claim qualification. No backend, GPU or remote login occurs."""
from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import e18_gpu_dispatcher as dispatcher
import e18_gpu_worker as worker
import run_original_plan_shard as runner


class OriginalPlanShardQualification(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="e18-targeted-shard-toy-")
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.owned=self.root/"owned"
        self.owned.mkdir()
        self.calls=[]
        self.manifests=[]
        refs=[]
        for index in range(2):
            keys=self.write("keys"+str(index)+".json",[["synthetic-granite-"+str(index),42,2]])
            manifest=dict(schema="e18_gpu_worker_manifest_v1",stage="member_generation",member_index=2,
                scope_ref=dict(path="synthetic-scope",sha256="1"*64),approval_ref=dict(path="synthetic-approval",sha256="2"*64),
                question_records_ref=dict(path="synthetic-inputs",sha256="3"*64),
                adapter_ref=dict(path="synthetic-adapter",sha256="4"*64),
                owned_output_root=str(self.owned),gpu_lock_root=str(self.root/"original_gpu_locks"),
                software_test_only=True,task_keys_ref=keys)
            refs.append(self.write("manifest"+str(index)+".json",manifest))
            self.manifests.append(manifest)
        self.plan=dict(schema="e18_gpu_dispatch_plan_v1",worker_source_sha256=worker.file_sha(worker.__file__),manifest_refs=refs)
        self.plan_ref=self.write("ORIGINAL_PLAN.json",self.plan)
        self.plan_sha=self.plan_ref["sha256"]
        self.state=self.owned/"_dispatcher/plans"/self.plan_sha
        self.state.mkdir(parents=True)

    def write(self,name,value):
        path=self.root/name
        path.write_text(json.dumps(value,sort_keys=True),encoding="utf-8")
        return dict(path=str(path),sha256=worker.file_sha(path))

    def status(self,index,status):
        reference=self.plan["manifest_refs"][index]
        worker.save_json(self.state/(reference["sha256"]+".status.json"),dict(status=status,manifest_sha256=reference["sha256"]))

    def execute(self,manifest,checksum,software_test_only=False):
        self.assertTrue(software_test_only)
        self.calls.append((copy.deepcopy(manifest),checksum))
        return dict(completed=1,planned=1,failures=[])

    def run_one(self,**kwargs):
        return runner.run_one(self.plan,self.plan_sha,software_test_only=True,execute=self.execute,**kwargs)

    def test_only_unused_second_shard_runs_original_manifest(self):
        self.status(0,"RUNNING")
        before=[worker.file_sha(Path(reference["path"])) for reference in self.plan["manifest_refs"]]
        result=self.run_one()
        self.assertEqual(result["status"],"DONE")
        self.assertEqual(result["manifest_sha256"],self.plan["manifest_refs"][1]["sha256"])
        self.assertEqual(self.calls[0][0],self.manifests[1])
        self.assertEqual(before,[worker.file_sha(Path(reference["path"])) for reference in self.plan["manifest_refs"]])
        self.assertEqual(worker.file_sha(Path(self.plan_ref["path"])),self.plan_sha)

    def test_active_explicit_manifest_is_busy_without_takeover(self):
        self.status(0,"RUNNING")
        result=self.run_one(manifest_sha=self.plan["manifest_refs"][0]["sha256"])
        self.assertEqual(result["status"],"BUSY_OR_ALREADY_USED")
        self.assertEqual(self.calls,[])

    def test_used_incomplete_shard_is_not_resumed(self):
        self.status(0,"WAITING_FOR_DEPENDENCIES")
        result=self.run_one(manifest_sha=self.plan["manifest_refs"][0]["sha256"])
        self.assertEqual(result["executed_shards"],0)
        self.assertEqual(self.calls,[])

    def test_nonblocking_original_claim_prevents_duplicate_execution(self):
        locks=[dispatcher.ShardClaim(self.state/(reference["sha256"]+".claim")) for reference in self.plan["manifest_refs"]]
        for lock in locks:
            self.assertTrue(lock.acquire())
            self.addCleanup(lock.release)
        result=self.run_one()
        self.assertEqual(result["status"],"BUSY_OR_NO_UNUSED_SHARD")
        self.assertEqual(self.calls,[])

    def test_original_dispatcher_recognizes_targeted_done(self):
        self.status(0,"DONE")
        self.run_one()
        result=dispatcher.run_plan(self.plan,self.plan_sha,software_test_only=True,
            execute=lambda *args,**kwargs:self.fail("Original plan reran a DONE shard"))
        self.assertEqual(result["status"],"DONE")
        self.assertEqual(result["locally_completed_shards"],0)

    def test_independent_admitted_gpu_lock_preserves_manifest(self):
        lock_root=self.root/"admitted-extra-slot"
        result=self.run_one(gpu="0",gpu_lock_root=lock_root)
        self.assertEqual(result["admitted_gpu_lock_root"],str(lock_root.resolve()))
        self.assertEqual(self.calls[0][0]["gpu_lock_root"],self.manifests[0]["gpu_lock_root"])

    def test_busy_gpu_does_not_mark_shard_running(self):
        lock_root=self.root/"admitted-extra-slot"
        with worker.GPUOwnership(lock_root,"0",self.plan_sha):
            result=self.run_one(gpu="0",gpu_lock_root=lock_root)
        self.assertEqual(result["status"],"GPU_BUSY")
        self.assertEqual(list(self.state.glob("*.status.json")),[])
        self.assertEqual(self.calls,[])

    def test_failed_original_plan_is_not_extended(self):
        self.status(0,"FAILED")
        result=self.run_one()
        self.assertEqual(result["status"],"STOP_REQUIRES_RECONCILIATION")
        self.assertEqual(self.calls,[])

    def test_worker_exception_writes_original_failed_and_stop(self):
        def fail(*args,**kwargs): raise RuntimeError("synthetic worker failure")
        with self.assertRaises(RuntimeError):
            runner.run_one(self.plan,self.plan_sha,software_test_only=True,execute=fail)
        self.assertTrue((self.state/"STOP_REQUIRES_RECONCILIATION.json").exists())
        self.assertEqual(worker.read_json(next(self.state.glob("*.status.json")))["status"],"FAILED")

    def test_production_rejects_test_callback_and_mode(self):
        with self.assertRaises(PermissionError):
            runner.run_one(self.plan,self.plan_sha,execute=self.execute)
        with self.assertRaises(PermissionError):
            runner.run_one(self.plan,self.plan_sha)

    def test_foreign_manifest_is_not_a_new_plan(self):
        with self.assertRaises(ValueError): self.run_one(manifest_sha="0"*64)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    capture=io.StringIO()
    started=time.perf_counter()
    result=unittest.TextTestRunner(stream=capture,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OriginalPlanShardQualification))
    report=dict(schema="e18_original_plan_targeted_shard_qualification_v1",status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),wall_s=time.perf_counter()-started,
        source_sha256=worker.file_sha(runner.__file__),qualification_source_sha256=worker.file_sha(__file__),
        original_dispatcher_sha256=worker.file_sha(dispatcher.__file__),original_worker_sha256=worker.file_sha(worker.__file__),
        test_log=capture.getvalue(),software_fixture_only=True,actual_gpu_queries=0,model_calls=0,server_logins=0,
        production_launch_performed=False)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:report[key] for key in ("status","tests","failures","errors","wall_s")}))
    if not result.wasSuccessful():
        print(capture.getvalue())
        raise SystemExit(1)
