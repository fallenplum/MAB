"""Small synthetic archives for byte-restoration, refusal and restart checks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import materialize_closed_native as materializer


class NativeMaterializerQualification(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="e18-materializer-toy-")
        self.root=Path(self.temporary.name).resolve()
        self.addCleanup(self.cleanup_fixture)
        self.archives=self.root/"archives"
        self.archives.mkdir()
        self.destination=self.root/"private-native-view"
        self.scope=materializer.digest("synthetic frozen execution scope")
        self.key=["synthetic-question",42,0]
        self.prefix=self.scope+"/member_generation/"+materializer.digest(self.key)+"/"
        self.payloads={name:(" { \"synthetic_native_file\": "+json.dumps(name)+" } \r\n").encode()
            for name in sorted(materializer.OWNER_FILES)}
        self.expected=self.make_archive("1"*32)
        self.catalog=self.make_catalog([self.expected],"catalog1.json")

    def cleanup_fixture(self):
        temporary_root=Path(tempfile.gettempdir()).resolve()
        if self.root.parent!=temporary_root or not self.root.name.startswith("e18-materializer-toy-"):
            raise ValueError("Fixture cleanup escaped its exact generated temporary root")
        if materializer.io_path(self.root).exists():
            shutil.rmtree(materializer.io_path(self.root))
        self.temporary.cleanup()

    def make_archive(self,identifier,payloads=None,mutate_manifest=None,unsafe_member=None):
        payloads=payloads or self.payloads
        manifest=dict(schema="e18_closed_native_export_manifest_v1",archive_id=identifier,scope_sha256=self.scope,
            software_test_only=True,owned_root="/synthetic-server/e18/owned",original_files_retained=True,
            new_model_calls=0,scorer_calls=0,gold_reads=0,
            files=[dict(path=self.prefix+name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in payloads.items()],
            tasks=[dict(task_id=materializer.digest([self.scope,"member_generation",self.key]),stage="member_generation",
                task_key=self.key,native_job_id=materializer.digest("synthetic job"),native_record_sha256=materializer.digest("synthetic native"),
                physical_attempt_id="synthetic-physical-attempt",model_fingerprint=materializer.digest("synthetic model"),
                prompt_tokens=2,completion_tokens=1,manifest_sha256s=[materializer.digest("synthetic manifest")])])
        if mutate_manifest: mutate_manifest(manifest)
        path=self.archives/("native-"+identifier+".tar.gz")
        with tarfile.open(path,"x:gz") as archive:
            for name,raw in payloads.items():
                entry=tarfile.TarInfo(self.prefix+name)
                entry.size=len(raw)
                if unsafe_member and name=="records.jsonl":
                    entry.type=tarfile.SYMTYPE
                    entry.linkname="../../outside"
                    entry.size=0
                archive.addfile(entry,io.BytesIO(raw) if entry.isfile() else None)
            raw=json.dumps(manifest,sort_keys=True,ensure_ascii=False).encode()
            entry=tarfile.TarInfo("EXPORT_MANIFEST.json")
            entry.size=len(raw)
            archive.addfile(entry,io.BytesIO(raw))
        return dict(path="/synthetic-server/exports/"+path.name,sha256=materializer.file_sha(path),bytes=path.stat().st_size,
            tasks=1,manifest_sha256=materializer.digest(manifest))

    def make_catalog(self,entries,name):
        path=self.root/name
        value=dict(schema="e18_completed_native_export_result_v1",scope_sha256=self.scope,software_test_only=True,
            new_archives=entries,recovered_archives=[],new_model_calls=0,scorer_calls=0,gold_reads=0)
        path.write_text(json.dumps(value),encoding="utf-8")
        return dict(path=str(path),sha256=materializer.file_sha(path))

    def run_materializer(self,refs=None,**kwargs):
        return materializer.materialize(refs or [self.catalog],self.archives,self.destination,software_test_only=True,**kwargs)

    def test_exact_original_bytes_and_idempotent_resume(self):
        report=self.run_materializer()
        self.assertEqual(report["created_files"],3)
        for name,raw in self.payloads.items():
            self.assertEqual(self.destination.joinpath(*self.prefix.strip("/").split("/"),name).read_bytes(),raw)
        repeated=self.run_materializer()
        self.assertEqual(repeated["created_files"],0)
        self.assertEqual(repeated["existing_identical_files_reused"],3)
        self.assertEqual(materializer.file_sha(self.archives/Path(self.expected["path"]).name),self.expected["sha256"])

    def test_repeated_catalog_entry_is_explicit_reuse(self):
        report=self.run_materializer([self.catalog,self.catalog])
        self.assertEqual(report["unique_native_files"],3)
        self.assertEqual(report["repeated_catalog_archive_entries_reused"],1)

    def test_consistent_cross_archive_task_and_files_are_reused(self):
        second=self.make_archive("2"*32)
        report=self.run_materializer([self.catalog,self.make_catalog([second],"catalog2.json")])
        self.assertEqual(report["unique_native_tasks"],1)
        self.assertEqual(report["identical_cross_archive_tasks_reused"],1)
        self.assertEqual(report["identical_cross_archive_paths_reused"],3)

    def test_conflicting_cross_archive_bytes_fail_before_destination_creation(self):
        changed=copy.deepcopy(self.payloads)
        changed["records.jsonl"]=b'{"synthetic":"different native bytes"}\n'
        second=self.make_archive("2"*32,changed)
        with self.assertRaises(ValueError):
            self.run_materializer([self.catalog,self.make_catalog([second],"catalog2.json")])
        self.assertFalse(self.destination.exists())

    def test_wrong_outer_archive_sha_fails(self):
        row=dict(self.expected,sha256="0"*64)
        with self.assertRaises(ValueError): self.run_materializer([self.make_catalog([row],"wrong.json")])
        self.assertFalse(self.destination.exists())

    def test_wrong_manifest_file_sha_fails(self):
        second=self.make_archive("2"*32,mutate_manifest=lambda value:value["files"][0].update(sha256="0"*64))
        with self.assertRaises(ValueError): self.run_materializer([self.make_catalog([second],"wrong-inner.json")])
        self.assertFalse(self.destination.exists())

    def test_tar_symlink_is_never_materialized(self):
        second=self.make_archive("2"*32,unsafe_member=True)
        with self.assertRaises(ValueError): self.run_materializer([self.make_catalog([second],"symlink.json")])
        self.assertFalse(self.destination.exists())

    def test_target_conflict_is_not_overwritten(self):
        path=self.destination.joinpath(*self.prefix.strip("/").split("/"),"records.jsonl")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"existing unrelated private content")
        with self.assertRaises(ValueError): self.run_materializer()
        self.assertEqual(path.read_bytes(),b"existing unrelated private content")
        self.assertFalse(path.with_name("ATTEMPT_STATE.json").exists())

    def test_interruption_before_publish_is_resumable(self):
        def stop(event,temporary,target): raise RuntimeError("synthetic interruption before atomic publish")
        with self.assertRaises(RuntimeError): self.run_materializer(crash_hook=stop)
        self.assertFalse(self.destination.joinpath(*self.prefix.strip("/").split("/"),"ATTEMPT_STATE.json").exists())
        report=self.run_materializer()
        self.assertEqual(report["created_files"],3)

    def test_reparse_destination_ancestor_is_refused(self):
        original=Path.lstat
        def fake_info(path,*args,**kwargs):
            if path==materializer.io_path(self.destination): return SimpleNamespace(st_mode=stat.S_IFLNK|0o777,st_file_attributes=0)
            return original(path,*args,**kwargs)
        with patch.object(Path,"lstat",fake_info):
            with self.assertRaises(ValueError): self.run_materializer()

    def test_only_native_exact_paths_are_allowed(self):
        for name in ("../escape","/absolute/path","./"+self.prefix+"records.jsonl",self.prefix+"model.safetensors",
                     self.scope+"/member_generation/"+"A"*64+"/records.jsonl"):
            with self.assertRaises(ValueError): materializer.native_path(name,self.scope)

    def test_gold_payload_is_rejected_without_copy(self):
        changed=dict(self.payloads)
        changed["records.jsonl"]=b'{"reference":"synthetic forbidden value"}'
        second=self.make_archive("2"*32,changed)
        with self.assertRaises(ValueError): self.run_materializer([self.make_catalog([second],"protected.json")])
        self.assertFalse(self.destination.exists())

    def test_canonical_archive_directory_cannot_be_destination(self):
        with self.assertRaises(ValueError):
            materializer.materialize([self.catalog],self.archives,self.archives,software_test_only=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    started=time.perf_counter()
    capture=io.StringIO()
    result=unittest.TextTestRunner(stream=capture,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeMaterializerQualification))
    report=dict(schema="e18_native_materializer_toy_qualification_v1",status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),wall_s=time.perf_counter()-started,
        source_sha256=materializer.file_sha(materializer.__file__),qualification_source_sha256=materializer.file_sha(__file__),
        qualification_scope="synthetic byte-transport fixtures only; scientific/native execution validation remains the preceding independent archive audit",
        test_log=capture.getvalue(),real_archives_materialized=0,real_native_payloads_decoded=0,gold_reads=0,scoring_calls=0,model_calls=0,
        remote_operations=0,canonical_archives_modified=0)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:report[key] for key in ("status","tests","failures","errors","wall_s")}))
    if not result.wasSuccessful():
        print(capture.getvalue())
        raise SystemExit(1)
