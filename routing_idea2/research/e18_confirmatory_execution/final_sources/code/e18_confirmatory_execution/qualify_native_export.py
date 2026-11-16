"""Mock-only closure, boundary and crash-recovery qualification for native export."""
from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import shutil
import sqlite3
import stat
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import export_completed_native as exporter


ASSETS=Path("E:/语言模型组合路由")


class NativeExportQualification(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="e18-export-mock-")
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()
        self.owned=self.root/"owned"
        self.owned.mkdir()
        self.assets=self.root/"runtime_assets"
        relatives=["src/budget_router/inference.py","src/budget_router/member_execution.py","src/budget_router/execution.py",
            "scripts/v2/typed_aggregation_execution.py","configs/pool_v1.json",
            "configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json"]
        pool=json.loads((ASSETS/"configs/pool_v1.json").read_text(encoding="utf-8-sig"))["models"]
        relatives.extend(row["config"] for row in pool)
        for relative in relatives:
            destination=self.assets/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ASSETS/relative,destination)
        self.profile=json.loads((self.assets/pool[0]["config"]).read_text(encoding="utf-8-sig"))
        self.model_fp=pool[0]["fingerprint"]
        self.alias=pool[0]["alias"]
        self.keys=[("synthetic-export-"+str(index),42,0) for index in range(3)]
        bindings={name:exporter.digest(name) for name in ("protocol_sha256","target_inputs_sha256","frozen_actions_sha256","budget_sha256","target_ids_sha256")}
        self.scope=dict(schema="e18_scope_contract_v2",role="execution",cohort="e18_confirm",bindings=bindings,
            transport_split="online_stream",operator_fingerprint="9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e",
            question_keys=[dict(question_id=key[0],canonical_question_sha256=exporter.digest(key[0])) for key in self.keys],
            required_member_keys=self.keys,required_aggregate_keys=[])
        self.scope_sha=exporter.digest(self.scope)
        self.scope_ref=self.write_ref("inputs/SCOPE.json",self.scope)
        self.approval_ref=self.write_ref("inputs/APPROVAL.json",dict(status="APPROVED",cohort="e18_confirm",software_test_only=True,
            approved_output_root=str(self.owned),scope_sha256_by_role={"execution":self.scope_sha},operations=["member_generation"],bindings=bindings))
        self.task_ref=self.write_ref("inputs/TASKS.json",self.keys)
        adapter=self.root/"code/adapter.py"
        adapter.parent.mkdir()
        adapter.write_text("# Synthetic metadata-only code fixture\n",encoding="utf-8")
        self.manifest=dict(schema="e18_gpu_worker_manifest_v1",stage="member_generation",member_index=0,software_test_only=True,
            scope_ref=self.scope_ref,approval_ref=self.approval_ref,adapter_ref=dict(path=str(adapter),sha256=exporter.file_sha(adapter)),
            task_keys_ref=self.task_ref,asset_root=str(self.assets),owned_output_root=str(self.owned),backend=dict(mode="synthetic_fixture"))
        self.manifest_ref=self.write_ref("manifests/member.json",self.manifest)
        self.plan_ref=self.write_ref("plans/dispatch.json",dict(schema="e18_gpu_dispatch_plan_v1",manifest_refs=[self.manifest_ref]))

    def write_ref(self,name,value):
        path=self.root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(exporter.encoded(value))
        return dict(path=str(path),sha256=exporter.file_sha(path))

    def make_owner(self,index=0):
        key=self.keys[index]
        owner=self.owned/self.scope_sha/"member_generation"/exporter.digest(list(key))
        owner.mkdir(parents=True)
        row=dict(job_id=exporter.digest(["synthetic-job",list(key)]),question_id=key[0],seed=42,split="online_stream",
            model_fingerprint=self.model_fp,model_alias=self.alias,sample_index=0,sampling_control=False,
            prompt_version="independent_user_only_brief_final_v2",attempt_id="synthetic-physical-"+str(index),
            status="success",cost_status="recorded_actual_usage",result=dict(text="Synthetic generated output",prompt_tokens=8,
                completion_tokens=4,wall_time_s=.001,model_id=self.profile["model_id"],model_revision=self.profile["revision"],
                backend="transformers",seed=42,model_identity_verified=True))
        row["record_sha256"]=exporter.digest(row)
        task=dict(question_id=key[0],canonical_id=exporter.digest(key[0]),generation_seed=42,member_index=0)
        receipt=dict(schema="e18_execution_task_receipt_v2",status="SUCCESS",scope_sha256=self.scope_sha,stage="member_generation",
            approval_sha256=self.approval_ref["sha256"],bindings=self.scope["bindings"],approved_output_root=str(self.owned),
            software_test_only=True,cohort="e18_confirm",transport_split="online_stream",task_key=task,
            native_record=row,native_record_sha256=row["record_sha256"],native_job_id=row["job_id"],physical_attempt_id=row["attempt_id"])
        fence=dict(status="SUCCESS_COMMITTED",scope_sha256=self.scope_sha,stage="member_generation",task_key=list(key),
            native_job_id=row["job_id"],native_record_sha256=row["record_sha256"],software_test_only=True,approval_sha256=self.approval_ref["sha256"])
        for name,value in (("ATTEMPT_STATE.json",fence),("LAST_TASK_RECEIPT.json",receipt)):
            (owner/name).write_bytes(exporter.encoded(value))
        (owner/"records.jsonl").write_bytes(exporter.encoded(row)+b"\n")
        return owner

    def run_export(self,**kwargs):
        return exporter.export_completed(self.root,self.scope_ref,[self.plan_ref],batch_size=2,software_test_only=True,**kwargs)

    def test_closure_keeps_original_bytes_and_modelspec(self):
        owner=self.make_owner()
        originals={name:(owner/name).read_bytes() for name in exporter.OWNER_FILES}
        report=self.run_export()
        self.assertEqual(report["registered_tasks"],1)
        with tarfile.open(report["new_archives"][0]["path"],"r:gz") as archive:
            manifest=json.load(archive.extractfile("EXPORT_MANIFEST.json"))
            for item in manifest["files"]:
                self.assertEqual(archive.extractfile(item["path"]).read(),originals[Path(item["path"]).name])
        self.assertTrue(all((owner/name).is_file() for name in exporter.OWNER_FILES))

    def test_repeat_export_has_no_duplicate_batch(self):
        self.make_owner()
        self.run_export()
        report=self.run_export()
        self.assertEqual(report["new_archives"],[])
        self.assertEqual(report["registered_tasks"],1)

    def test_small_final_chunk_and_no_empty_archive(self):
        for index in range(3): self.make_owner(index)
        report=self.run_export()
        self.assertEqual([row["tasks"] for row in report["new_archives"]],[2,1])

    def test_no_completed_tasks_returns_only_metadata(self):
        report=self.run_export()
        self.assertEqual(report["status"],"NO_NEW_STABLE_COMPLETED_TASKS")
        self.assertEqual(list((self.root/"exports").glob("*.tar.gz")),[])

    def test_reserved_task_is_not_closed(self):
        owner=self.make_owner()
        fence=json.loads((owner/"ATTEMPT_STATE.json").read_bytes())
        fence["status"]="RESERVED_BEFORE_BACKEND"
        (owner/"ATTEMPT_STATE.json").write_bytes(exporter.encoded(fence))
        self.assertEqual(self.run_export()["registered_tasks"],0)

    def test_wrong_modelspec_and_tokens_are_refused(self):
        owner=self.make_owner()
        row=json.loads((owner/"records.jsonl").read_bytes())
        row["result"]["model_revision"]="0"*40
        row["record_sha256"]=exporter.digest({key:value for key,value in row.items() if key!="record_sha256"})
        (owner/"records.jsonl").write_bytes(exporter.encoded(row))
        self.assertEqual(self.run_export()["registered_tasks"],0)

    def test_protected_payload_is_refused_before_archive(self):
        owner=self.make_owner()
        value=json.loads((owner/"LAST_TASK_RECEIPT.json").read_bytes())
        value["gold"]="synthetic forbidden reference"
        (owner/"LAST_TASK_RECEIPT.json").write_bytes(exporter.encoded(value))
        self.assertEqual(self.run_export()["registered_tasks"],0)

    def test_extra_weight_file_never_enters_archive(self):
        owner=self.make_owner()
        (owner/"model.safetensors").write_bytes(b"synthetic weight placeholder")
        result=self.run_export()
        with tarfile.open(result["new_archives"][0]["path"],"r:gz") as archive:
            self.assertEqual(len(archive.getnames()),4)
            self.assertFalse(any("safetensors" in name for name in archive.getnames()))

    def test_symlink_mode_is_refused(self):
        owner=self.make_owner()
        target=owner/"records.jsonl"
        original=Path.lstat
        def synthetic_lstat(path,*args,**kwargs):
            if path==target: return SimpleNamespace(st_mode=stat.S_IFLNK|0o777,st_file_attributes=0)
            return original(path,*args,**kwargs)
        with patch.object(Path,"lstat",synthetic_lstat):
            result=self.run_export()
        self.assertEqual(result["registered_tasks"],0)

    def test_source_change_during_compression_never_publishes_batch(self):
        owner=self.make_owner()
        def change(event,path):
            if event=="after_archive_build_before_source_recheck":
                with (owner/"LAST_TASK_RECEIPT.json").open("ab") as stream: stream.write(b" ")
        result=self.run_export(crash_hook=change)
        self.assertEqual(result["new_archives"],[])
        self.assertEqual(result["registered_tasks"],0)
        self.assertEqual(self.run_export()["registered_tasks"],1)

    def test_crash_after_closed_archive_recovers_registry_without_new_export(self):
        self.make_owner()
        def interrupt(event,path):
            if event=="after_archive_publish_before_registry": raise RuntimeError("synthetic process interruption")
        with self.assertRaises(RuntimeError): self.run_export(crash_hook=interrupt)
        result=self.run_export()
        self.assertEqual(len(result["recovered_archives"]),1)
        self.assertEqual(result["new_archives"],[])
        self.assertEqual(result["registered_tasks"],1)

    def test_crash_before_publish_retains_partial_without_claiming_data(self):
        self.make_owner()
        def interrupt(event,path):
            if event=="before_archive_publish": raise RuntimeError("synthetic interruption before publish")
        with self.assertRaises(RuntimeError): self.run_export(crash_hook=interrupt)
        result=self.run_export()
        self.assertEqual(result["registered_tasks"],1)
        self.assertEqual(result["unclosed_partial_archives"],1)

    def test_corrupt_registered_archive_cannot_remain_done_silently(self):
        self.make_owner()
        result=self.run_export()
        with Path(result["new_archives"][0]["path"]).open("ab") as stream: stream.write(b"synthetic corruption")
        with self.assertRaises(ValueError): self.run_export()

    def test_production_mode_rejects_synthetic_owner_sources(self):
        self.make_owner()
        with self.assertRaises(ValueError):
            exporter.export_completed(self.root,self.scope_ref,[self.plan_ref],batch_size=500)


def main():
    global ASSETS
    parser=argparse.ArgumentParser()
    parser.add_argument("--asset-root",type=Path,default=ASSETS)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    ASSETS=args.asset_root
    started=time.perf_counter()
    capture=io.StringIO()
    result=unittest.TextTestRunner(stream=capture,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeExportQualification))
    report=dict(schema="e18_native_closed_export_mock_qualification_v1",status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),wall_s=time.perf_counter()-started,
        exporter_source_sha256=exporter.file_sha(Path(exporter.__file__)),qualification_source_sha256=exporter.file_sha(__file__),
        test_log=capture.getvalue(),mock_native_outputs_only=True,real_server_logins=0,
        real_model_calls=0,real_gold_reads=0,real_scoring_calls=0,remote_files_deleted=0)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:report[key] for key in ("status","tests","failures","errors","wall_s")}))
    if not result.wasSuccessful():
        print(capture.getvalue())
        raise SystemExit(1)


if __name__=="__main__": main()
