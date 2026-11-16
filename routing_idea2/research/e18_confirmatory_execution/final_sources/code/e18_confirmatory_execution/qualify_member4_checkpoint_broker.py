"""Pure mock-control qualifications for the E18 member4 checkpoint broker."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import member4_checkpoint_broker as broker


class MockClock:
    def __init__(self):
        self.value=0
        self.callback=None

    def now(self):
        return self.value

    def sleep(self,seconds):
        self.value+=seconds
        if self.callback is not None:
            self.callback()


class Fixture:
    def __init__(self,root):
        self.root=Path(root)
        self.output=self.root/"owned"
        self.owner_pid,self.former_pid=9001,7001
        self.scope_sha="c"*64
        self.approval_sha="a"*64
        self.locks={}
        self.clock=MockClock()
        self.validator_calls=[]
        self.code=self.root/"e18_gpu_dispatcher.py"
        self.code.write_text("# synthetic dispatcher\n",encoding="utf-8")
        self.manifests=[]
        self.keys=[]
        for index,profile in enumerate([4,3,2,0,1,5,4,3,2,0,1,5]):
            keys=[[f"synthetic-{index}-{j}",42,profile] for j in range(3 if index==0 else 2 if index==6 else 1)]
            task_path=self.root/f"tasks-{index}.json"
            broker.write_once(task_path,keys)
            manifest=dict(member_index=profile,stage="member_generation",software_test_only=False,
                scope_ref=dict(path=str(self.root/"scope.json"),sha256=self.scope_sha),
                approval_ref=dict(path=str(self.root/"approval.json"),sha256=self.approval_sha),
                question_records_ref=dict(path=str(self.root/"inputs.jsonl"),sha256="b"*64),
                adapter_ref=dict(path=str(self.root/"adapter.py"),sha256="d"*64),
                owned_output_root=str(self.output),task_keys_ref=dict(path=str(task_path),sha256=broker.file_sha(task_path)))
            manifest_path=self.root/f"manifest-{index}.json"
            broker.write_once(manifest_path,manifest)
            self.manifests.append((dict(path=str(manifest_path),sha256=broker.file_sha(manifest_path)),manifest))
            if profile==4:
                self.keys.extend(tuple(key) for key in keys)
        self.plan_path=self.root/"base_plan.json"
        broker.write_once(self.plan_path,dict(manifest_refs=[ref for ref,_ in self.manifests]))
        self.plan_sha=broker.file_sha(self.plan_path)
        self.state=self.output/"_dispatcher/plans"/self.plan_sha
        self.state.mkdir(parents=True)
        self.target_ref=self.manifests[0][0]
        (self.output/"_dispatcher"/self.target_ref["sha256"]).mkdir(parents=True)
        self.target_claim=self.state/(self.target_ref["sha256"]+".claim")
        self.other_claim=self.state/(self.manifests[1][0]["sha256"]+".claim")
        self.locks[str(self.target_claim)]=self.former_pid
        self.locks[str(self.other_claim)]=8001
        broker.write_once(self.state/(self.target_ref["sha256"]+".status.json"),dict(status="RUNNING",pid=self.former_pid))
        broker.write_once(self.state/(self.manifests[1][0]["sha256"]+".status.json"),dict(status="RUNNING",pid=8001))
        fixture=self
        class Claim:
            def __init__(self,path):
                self.path=Path(path);self.held=False
            def acquire(self):
                if str(self.path) in fixture.locks:
                    return False
                fixture.locks[str(self.path)]=fixture.owner_pid;self.held=True
                return True
            def release(self):
                if self.held:
                    fixture.locks.pop(str(self.path),None);self.held=False
        self.dispatcher=SimpleNamespace(__file__=str(self.code),worker=SimpleNamespace(),ShardClaim=Claim,
            admit_plan=lambda plan:self.manifests)
        self.processes=MockProcesses(self)
        self.instance=broker.CheckpointBroker(self.plan_path,self.plan_sha,self.dispatcher,self.processes,
            self.output/"_dispatcher/checkpoint",self.root/"child_plan.json",target_pid=self.former_pid,
            owner_pid=self.owner_pid,expected_tasks=5,clock=self.clock.now,sleeper=self.clock.sleep,
            poll_seconds=1,timeout_seconds=3,child_timeout_seconds=4,validator=self.validate,scope_sha=self.scope_sha)
        self.make_success(self.keys[0])

    def validate(self,*values):
        self.validator_calls.append(values[:3])
        return values[3]

    def owner(self,key):
        return self.output/self.scope_sha/"member_generation"/broker.digest(list(key))

    def make_success(self,key):
        owner=self.owner(key);owner.mkdir(parents=True,exist_ok=True)
        native=dict(status="success",cost_status="recorded_actual_usage",job_id=broker.digest(["job",*key]),
            attempt_id="synthetic-attempt-"+broker.digest(list(key)),result=dict(text="synthetic",prompt_tokens=1,completion_tokens=2))
        native["record_sha256"]=broker.digest(native)
        receipt=dict(schema="e18_execution_task_receipt_v2",status="SUCCESS",scope_sha256=self.scope_sha,
            approval_sha256=self.approval_sha,stage="member_generation",software_test_only=False,
            task_key=dict(question_id=key[0],generation_seed=key[1],member_index=key[2]),native_record=native,
            native_record_sha256=native["record_sha256"],native_job_id=native["job_id"],physical_attempt_id=native["attempt_id"])
        fence=dict(status="SUCCESS_COMMITTED",scope_sha256=self.scope_sha,approval_sha256=self.approval_sha,
            task_key=list(key),stage="member_generation",software_test_only=False,native_job_id=native["job_id"],native_record_sha256=native["record_sha256"])
        broker.write_atomic(owner/"ATTEMPT_STATE.json",fence)
        broker.write_atomic(owner/"LAST_TASK_RECEIPT.json",receipt)
        (owner/"records.jsonl").write_text(json.dumps(native)+"\n",encoding="utf-8")

    def checkpoint(self):
        item=self.instance
        item.inspect_target();item.root.mkdir(parents=True)
        evidence=item.checkpoint_stop()
        lease,remaining=item.write_lease(evidence)
        return lease,remaining

    def release(self):
        for claim in self.instance.held.values():
            claim.release()

    def create_children(self,*,omit_native=False,duplicate=False,extra=False):
        item=self.instance
        lease_path=item.root/"LEASE.json"
        remaining=broker.read_json(item.root/"remaining_task_keys.json")
        lease_ref=dict(path=str(lease_path),sha256=broker.file_sha(lease_path))
        keys=copy.deepcopy(remaining)
        if duplicate:keys.append(keys[0])
        if extra:keys.append(["synthetic-foreign",42,4])
        key_path=self.root/"child_keys.json";broker.write_once(key_path,keys)
        manifest=dict(self.manifests[0][1],checkpoint_lease_ref=lease_ref,
            task_keys_ref=dict(path=str(key_path),sha256=broker.file_sha(key_path)))
        child_path=self.root/"child_manifest.json";broker.write_once(child_path,manifest)
        reference=dict(path=str(child_path),sha256=broker.file_sha(child_path))
        broker.write_once(item.child_plan_path,dict(schema="e18_member4_dispatch_plan_v3",checkpoint_lease_ref=lease_ref,
            worker_source_sha256="f"*64,manifest_refs=[reference]))
        child_sha=broker.file_sha(item.child_plan_path)
        state=self.output/"_dispatcher/plans"/child_sha
        state.mkdir(parents=True)
        broker.write_once(state/(reference["sha256"]+".status.json"),dict(status="DONE",manifest_sha256=reference["sha256"]))
        for index,key in enumerate(remaining):
            if omit_native and index==0:continue
            self.make_success(tuple(key))
        return reference


class MockProcesses:
    def __init__(self,fixture):
        self.fixture=fixture
        self.mode="graceful"
        self.alive=True
        self.caught=True
        self.gpu=True
        self.signals=[]
        self.reused=False
        self.keep_gpu=False

    def identity(self,pid):
        if not self.alive:return None
        f=self.fixture
        return dict(pid=pid,start_ticks="reused" if self.reused else "original",
            command=["python","-B","-u",str(f.code.resolve()),"--plan",str(f.plan_path.resolve()),
                "--plan-sha256",f.plan_sha,"--gpu","7"],sigterm_caught=self.caught)

    def claim_owner(self,path):
        return self.fixture.locks.get(str(path))

    def terminate(self,pid):
        f=self.fixture;self.signals.append(pid)
        if len(self.signals)==1:
            if self.mode=="stuck":return
            f.locks.pop(str(f.target_claim),None)
            self.caught=False
            if self.mode=="exit_no_summary":
                self.alive=False;self.gpu=self.keep_gpu;return
            summary=f.output/"_dispatcher"/f.target_ref["sha256"]/"SUMMARY.json"
            broker.write_atomic(summary,dict(stop_requested=True,manifest_sha256=f.target_ref["sha256"],stage="member_generation",
                profile=4,software_test_only=False,failures=[],completed=1,planned=3))
            if self.mode=="pid_reused":self.reused=True
            if self.mode=="foreign_claim":
                f.locks[str(f.other_claim)]=pid
        else:
            if any(owner==pid for owner in f.locks.values()):
                raise AssertionError("Second signal attempted while a parent task was owned")
            if self.caught:raise AssertionError("Second signal attempted in worker signal-handler phase")
            self.alive=False;self.gpu=self.keep_gpu

    def gpu_context_present(self,pid):
        return self.gpu


class BrokerQualification(unittest.TestCase):
    def setUp(self):
        # Two SHA-addressed ownership levels exceed Windows MAX_PATH under the
        # sandbox's deeply nested default Temp. Keep synthetic files shallow.
        temporary_parent=Path(__file__).resolve().parents[2]/".broker_toy"
        temporary_parent.mkdir(exist_ok=True)
        temporary=tempfile.TemporaryDirectory(prefix="b_",dir=temporary_parent)
        self.addCleanup(temporary.cleanup)
        self.fixture=Fixture(temporary.name)
        self.addCleanup(self.fixture.release)

    def test_dryrun_sends_no_signal_or_lease(self):
        f=self.fixture
        self.assertEqual(f.instance.inspect_target()["status"],"DRY_RUN_METADATA_VALID")
        self.assertEqual(f.processes.signals,[]);self.assertFalse(f.instance.root.exists())

    def test_command_wrong_gpu_rejected(self):
        f=self.fixture;f.instance.target_gpu="6"
        with self.assertRaises(PermissionError):f.instance.inspect_target()
        self.assertEqual(f.processes.signals,[])

    def test_target_must_own_running_parent(self):
        f=self.fixture;f.locks[str(f.target_claim)]=123
        with self.assertRaises(PermissionError):f.instance.inspect_target()

    def test_holds_pending_without_stealing_other_active_claim(self):
        f=self.fixture;f.instance.inspect_target();f.instance.hold_available()
        self.assertEqual(f.locks[str(f.other_claim)],8001)
        self.assertEqual(len(f.instance.held),10)

    def test_second_parent_must_be_held_before_signal(self):
        f=self.fixture;ref=f.instance.parents[1][0];f.locks[str(f.instance.claim_path(ref))]=8888
        with self.assertRaises(RuntimeError):f.instance.run(completion_delay=0)
        self.assertEqual(f.processes.signals,[]);self.assertFalse((f.instance.root/"LEASE.json").exists())

    def test_graceful_stop_uses_two_signals_and_keeps_two_parent_locks(self):
        f=self.fixture;lease,remaining=f.checkpoint()
        self.assertEqual(lease["checkpoint_evidence"]["kind"],"SUMMARY_CONFIRMED")
        self.assertEqual(len(f.processes.signals),2);self.assertEqual(len(f.instance.held),2)
        self.assertEqual(lease["checkpoint_successes"],1);self.assertEqual(len(remaining),4)
        self.assertEqual(f.locks[str(f.other_claim)],8001)

    def test_exit_signal_race_can_use_native_commit_evidence(self):
        f=self.fixture;f.processes.mode="exit_no_summary"
        lease,_=f.checkpoint()
        self.assertEqual(lease["checkpoint_evidence"]["kind"],"PROCESS_EXIT_AND_NATIVE_COMMIT_CONFIRMED")
        self.assertEqual(len(f.processes.signals),1)

    def test_unknown_reserved_owner_blocks_lease(self):
        f=self.fixture;f.processes.mode="exit_no_summary"
        owner=f.owner(f.keys[1]);owner.mkdir(parents=True);broker.write_once(owner/"ATTEMPT_STATE.json",dict(status="RESERVED_BEFORE_BACKEND"))
        with self.assertRaises(RuntimeError):f.instance.run(completion_delay=0)
        self.assertFalse((f.instance.root/"LEASE.json").exists())
        self.assertTrue((f.instance.root/"STOP_REQUIRES_RECONCILIATION.json").exists())

    def test_failed_owner_blocks_lease(self):
        f=self.fixture;fence=f.owner(f.keys[0])/"ATTEMPT_STATE.json"
        value=broker.read_json(fence);value["status"]="STOP_REQUIRES_RECONCILIATION";broker.write_atomic(fence,value)
        with self.assertRaises(RuntimeError):f.instance.run(completion_delay=0)

    def test_native_hash_tampering_rejected(self):
        f=self.fixture;path=f.owner(f.keys[0])/"LAST_TASK_RECEIPT.json";value=broker.read_json(path)
        value["native_record"]["result"]["text"]="changed";broker.write_atomic(path,value)
        with self.assertRaises(ValueError):f.instance.run(completion_delay=0)

    def test_store_extra_attempt_rejected(self):
        f=self.fixture;path=f.owner(f.keys[0])/"records.jsonl"
        with path.open("a",encoding="utf-8") as stream:stream.write('{"status":"failure"}\n')
        with self.assertRaises(RuntimeError):f.instance.run(completion_delay=0)

    def test_empty_pre_reservation_owner_is_unexecuted(self):
        f=self.fixture;owner=f.owner(f.keys[1]);owner.mkdir(parents=True);(owner/"records.jsonl").touch()
        lease,_=f.checkpoint();self.assertEqual(lease["remaining_count"],4)

    def test_stale_summary_does_not_authorize_second_signal(self):
        f=self.fixture;f.processes.mode="stuck"
        path=f.output/"_dispatcher"/f.target_ref["sha256"]/"SUMMARY.json"
        broker.write_atomic(path,dict(stop_requested=True,manifest_sha256=f.target_ref["sha256"],stage="member_generation",profile=4,software_test_only=False,failures=[]))
        with self.assertRaises(TimeoutError):f.instance.run(completion_delay=0)
        self.assertEqual(len(f.processes.signals),1)

    def test_pid_reuse_rejected_without_second_signal(self):
        f=self.fixture;f.processes.mode="pid_reused"
        with self.assertRaises(PermissionError):f.instance.run(completion_delay=0)
        self.assertEqual(len(f.processes.signals),1)

    def test_no_second_signal_when_former_holds_another_claim(self):
        f=self.fixture;f.processes.mode="foreign_claim"
        with self.assertRaises(RuntimeError):f.instance.run(completion_delay=0)
        self.assertEqual(len(f.processes.signals),1)

    def test_gpu_context_must_disappear(self):
        f=self.fixture;f.processes.mode="exit_no_summary";f.processes.keep_gpu=True
        with self.assertRaises(TimeoutError):f.instance.run(completion_delay=0)
        self.assertFalse((f.instance.root/"LEASE.json").exists())

    def test_heartbeat_does_not_change_static_lease(self):
        f=self.fixture;f.checkpoint();path=f.instance.root/"LEASE.json";before=broker.file_sha(path)
        f.instance.stage="TEST_HEARTBEAT";f.instance.heartbeat()
        self.assertEqual(broker.file_sha(path),before)

    def test_every_success_calls_dependency_validator(self):
        f=self.fixture;f.checkpoint();self.assertEqual(f.validator_calls,[f.keys[0]])

    def test_child_union_must_match_remaining(self):
        f=self.fixture;_,remaining=f.checkpoint();f.create_children(extra=True)
        with self.assertRaises(ValueError):f.instance.admit_child_plan(remaining)

    def test_duplicate_child_keys_rejected(self):
        f=self.fixture;_,remaining=f.checkpoint();f.create_children(duplicate=True)
        with self.assertRaises(ValueError):f.instance.admit_child_plan(remaining)

    def test_child_done_is_insufficient_without_native_commits(self):
        f=self.fixture;lease,remaining=f.checkpoint();f.create_children(omit_native=True)
        with self.assertRaises(RuntimeError):f.instance.wait_for_completion(lease,remaining,0)
        self.assertFalse((f.instance.root/"LEASE_COMPLETED.json").exists())
        self.assertEqual(broker.read_json(f.instance.status_path(f.target_ref))["status"],"RUNNING")

    def test_complete_children_mark_original_parents_done_after_all_native(self):
        f=self.fixture;lease,remaining=f.checkpoint();f.create_children()
        before=broker.file_sha(f.instance.root/"LEASE.json")
        complete=f.instance.wait_for_completion(lease,remaining,60)
        self.assertEqual(complete["original_tasks_verified"],5)
        self.assertEqual(f.clock.value,60)
        for reference,_ in f.instance.parents:self.assertEqual(broker.read_json(f.instance.status_path(reference))["status"],"DONE")
        self.assertEqual(broker.file_sha(f.instance.root/"LEASE.json"),before)

    def test_original_stop_fence_blocks_first_signal(self):
        f=self.fixture;broker.write_once(f.state/"STOP_REQUIRES_RECONCILIATION.json",dict(status="STOP"))
        with self.assertRaises(RuntimeError):f.instance.run(completion_delay=0)
        self.assertEqual(f.processes.signals,[])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=io.StringIO();started=time.perf_counter()
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BrokerQualification))
    receipt=dict(schema="e18_member4_checkpoint_broker_mock_qualification_v1",status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,broker_source_sha256=broker.file_sha(Path(broker.__file__)),test_source_sha256=broker.file_sha(Path(__file__)),
        base_member_plan_sha256=broker.BASE_PLAN_SHA,actual_process_signals=0,actual_model_calls=0,real_gold_reads=0,
        production_state_or_lease_files_written=0,control_backend="MockProcesses and MockClaims in temporary synthetic directories",
        wall_seconds=time.perf_counter()-started,test_output=output.getvalue())
    broker.write_once(args.output,receipt)
    print(json.dumps({key:value for key,value in receipt.items() if key!="test_output"},ensure_ascii=False,indent=2))
    if not result.wasSuccessful():
        print(output.getvalue());raise SystemExit(1)


if __name__=="__main__":main()
