"""Qualify native-export v2 with the real three-role approval metadata shape."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
import unittest

import export_completed_native_v2 as exporter
import qualify_native_export as previous


previous.exporter=exporter


class NativeExportV2Qualification(previous.NativeExportQualification):
    def setUp(self):
        super().setUp()
        self.replace_approval(json.loads(Path(self.approval_ref["path"]).read_bytes()))

    def replace_approval(self,value):
        value["schema"]="e18_scope_approval_v2"
        value["scope_sha256_by_role"].update(reference=exporter.digest("synthetic reference component"),
            scoring=exporter.digest("synthetic scoring component"))
        self.approval_ref=self.write_ref("inputs/APPROVAL.json",value)
        self.manifest["approval_ref"]=self.approval_ref
        self.manifest_ref=self.write_ref("manifests/member.json",self.manifest)
        self.plan_ref=self.write_ref("plans/dispatch.json",dict(schema="e18_gpu_dispatch_plan_v1",manifest_refs=[self.manifest_ref]))

    def test_real_shape_three_role_approval_is_metadata(self):
        self.make_owner()
        report=self.run_export()
        self.assertEqual(report["registered_tasks"],1)

    def test_native_reference_sha_field_is_still_forbidden(self):
        owner=self.make_owner()
        value=json.loads((owner/"LAST_TASK_RECEIPT.json").read_bytes())
        value["reference"]="a"*64
        (owner/"LAST_TASK_RECEIPT.json").write_bytes(exporter.encoded(value))
        self.assertEqual(self.run_export()["registered_tasks"],0)

    def test_approval_reference_answer_outside_role_is_refused(self):
        value=json.loads(Path(self.approval_ref["path"]).read_bytes())
        value["reference"]="synthetic forbidden answer"
        self.replace_approval(value)
        with self.assertRaises(ValueError): self.run_export()

    def test_approval_reference_role_must_be_sha_not_answer(self):
        value=json.loads(Path(self.approval_ref["path"]).read_bytes())
        value["scope_sha256_by_role"]["reference"]="synthetic answer"
        raw=exporter.encoded(value)
        with self.assertRaises(ValueError): exporter.approval_metadata_json(raw)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--asset-root",type=Path,default=previous.ASSETS)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    previous.ASSETS=args.asset_root
    capture=io.StringIO()
    started=time.perf_counter()
    result=unittest.TextTestRunner(stream=capture,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeExportV2Qualification))
    report=dict(schema="e18_native_closed_export_mock_qualification_v2",status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),wall_s=time.perf_counter()-started,
        exporter_source_sha256=exporter.file_sha(exporter.__file__),qualification_source_sha256=exporter.file_sha(__file__),
        previous_v1_exporter_sha256=exporter.file_sha(Path(__file__).with_name("export_completed_native.py")),
        inherited_mock_qualification_sha256=exporter.file_sha(previous.__file__),
        repair_reason="v1 falsely treated the legitimate approval.scope_sha256_by_role.reference SHA as a reference-answer payload; output payload protection is unchanged",
        test_log=capture.getvalue(),mock_native_outputs_only=True,real_server_logins=0,
        real_model_calls=0,real_gold_reads=0,real_scoring_calls=0,remote_files_deleted=0,
        adds_approval_requirement=False)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:report[key] for key in ("status","tests","failures","errors","wall_s")}))
    if not result.wasSuccessful():
        print(capture.getvalue())
        raise SystemExit(1)
