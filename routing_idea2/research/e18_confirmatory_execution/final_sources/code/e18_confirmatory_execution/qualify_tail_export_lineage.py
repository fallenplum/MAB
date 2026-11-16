"""Synthetic source/export bindings with deliberately unparsed native bodies."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

import audit_tail_export_lineage as audit


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def reference(path):
    return dict(path=str(path), sha256=audit.file_sha(path))


class Fixture:
    def __init__(self, directory, *, complete=True, transform_export=None, transform_bundle=None):
        self.root = Path(directory)
        self.remote = "/data/e18_synthetic"
        self.raw = {}
        self.keys = [[f"synthetic-{i}", 42, 4] for i in range(5)]
        self.scope = dict(member_execution_keys=self.keys)
        self.scope_path = self.root / "EXECUTION_SCOPE.json"
        self.scope_path.write_bytes(encoded(self.scope))
        self.scope_ref = reference(self.scope_path)
        self.scope_sha = audit.digest(self.scope)

        def put(name, value, *, raw=False):
            blob = value if raw else encoded(value)
            self.raw[name] = blob
            return dict(path=self.remote + "/" + name, sha256=hashlib.sha256(blob).hexdigest())

        parent_key_ref = put("inputs/parent.keys.json", self.keys)
        upstream_ref = put("owned/upstream/LEASE.json", dict(remaining_task_keys_ref=parent_key_ref))
        controller_ref = put("code/member4_tail_partition.py", Path(audit.__file__).with_name("member4_tail_partition.py").read_bytes(), raw=True)
        parent = dict(schema="e18_member4_worker_manifest_v3", stage="member_generation", member_index=4, software_test_only=False, checkpoint_lease_ref=upstream_ref, task_keys_ref=parent_key_ref, approval_ref=dict(path=self.remote + "/approval.json", sha256="a" * 64), scope_ref=dict(path=self.remote + "/EXECUTION_SCOPE.json", sha256=self.scope_ref["sha256"]), adapter_ref=dict(path=self.remote + "/adapter.py", sha256=audit.ADAPTER_SHA), backend=dict(mode="synthetic"))
        parent_ref = put("manifests/parent.json", parent)
        parent_plan_ref = put("MEMBER4_DISPATCH_PLAN_v3.json", dict(manifest_refs=[parent_ref]))
        native_rows, files, payloads = [], [], {}
        for key in self.keys:
            key_hash = audit.digest(key)
            prefix = self.scope_sha + "/member_generation/" + key_hash + "/"
            for name in ("ATTEMPT_STATE.json", "LAST_TASK_RECEIPT.json", "records.jsonl"):
                # Invalid JSON and a protected marker prove that native values are skipped.
                raw = b"not-json-native-answer-value-" + key_hash.encode() + name.encode()
                payloads[prefix + name] = raw
                files.append(dict(path=prefix + name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            native_rows.append(dict(task_id=audit.digest([self.scope_sha, "member_generation", key]), stage="member_generation", task_key=key, native_record_sha256=audit.digest(["native", key]), native_job_id=audit.digest(["job", key]), physical_attempt_id="synthetic-attempt-" + key_hash, model_fingerprint="f" * 64, prompt_tokens=2, completion_tokens=3, manifest_sha256s=[parent_ref["sha256"]]))
        by_file = {row["path"]: row for row in files}
        checkpoint = []
        for index, row in enumerate(native_rows):
            item = dict(task_key_sha256=audit.digest(row["task_key"]), status="SUCCESS" if index < 2 else "UNEXECUTED")
            if index < 2:
                item.update(native_record_sha256=row["native_record_sha256"], native_receipt_file_sha256=by_file[self.scope_sha + "/member_generation/" + item["task_key_sha256"] + "/LAST_TASK_RECEIPT.json"]["sha256"])
            checkpoint.append(item)
        remaining_ref = put("owned/tail/remaining_task_keys.json", self.keys[2:])
        checkpoint_ref = put("owned/tail/CHECKPOINT_SOURCE_AUDIT.json", dict(tasks=checkpoint, pending_or_unknown_attempts=0))
        lease = dict(schema="e18_member4_tail_partition_lease_v1", status="LEASED", software_test_only=False, pending_or_unknown_attempts=0, execution_scope_sha256=self.scope_sha, approval_sha256="a" * 64, controller_source_ref=controller_ref, parent_plan_ref=parent_plan_ref, parent_manifest_ref=parent_ref, parent_claim_path=self.remote + "/owned/_dispatcher/plans/" + parent_plan_ref["sha256"] + "/" + parent_ref["sha256"] + ".claim", upstream_checkpoint_lease_ref=upstream_ref, remaining_task_keys_ref=remaining_ref, checkpoint_source_audit_ref=checkpoint_ref, original_parent_tasks=5, checkpoint_successes=2, remaining_count=3, maximum_child_keys=2)
        lease_ref = put("owned/tail/LEASE.json", lease)
        children, child_objects = [], []
        for index, keys in enumerate((self.keys[2:4], self.keys[4:])):
            key_ref = put(f"fine/part{index}.keys.json", keys)
            child = dict(parent, task_keys_ref=key_ref, tail_parent_lease_ref=lease_ref, tail_partition_index=index)
            child_ref = put(f"fine/part{index}.manifest.json", child)
            children.append(child_ref)
            child_objects.append(child)
            for row in native_rows:
                if row["task_key"] in keys:
                    row["manifest_sha256s"].append(child_ref["sha256"])
        plan = dict(schema="e18_member4_dispatch_plan_v3", worker_source_sha256=audit.WORKER_SHA, checkpoint_lease_ref=upstream_ref, tail_parent_lease_ref=lease_ref, tail_parent_plan_ref=parent_plan_ref, tail_parent_manifest_ref=parent_ref, manifest_refs=children)
        plan_ref = put("MEMBER4_TAIL_PLAN_v1.json", plan)
        planner_ref = put("planner_v5_runtime/incremental_aggregation_planner_v5.py", b"# synthetic planner source\n", raw=True)
        source_audit = dict(plan_ref=plan_ref, tail_parent_lease_ref=lease_ref, parent_plan_ref=parent_plan_ref, parent_manifest_ref=parent_ref, checkpoint_source_audit_ref=checkpoint_ref)
        sidecar = dict(schema="e18_planner_r1_tail_source_extension_v1", tail_source_plan_refs=[plan_ref], tail_control_source_ref=controller_ref, additional_scientific_keys=0, original_config_modified=False, sqlite_schema_modified=False, historical_source_admission_requires_live_broker=False, source_audits=[source_audit], original_config_ref=dict(path=self.remote + "/config.json", sha256="c" * 64), all_done_config_sha256="c" * 64, extension_source_sha256=planner_ref["sha256"])
        put("owned/planner/R1_TAIL_SOURCE_EXTENSION_synthetic.json", sidecar)
        pilot_row = native_rows[2]
        pilot_prefix = self.scope_sha + "/member_generation/" + audit.digest(pilot_row["task_key"]) + "/"
        pilot = dict(status="PASS_ORIGINAL_MODELSPEC_JOB_OWNER_AND_ACTUAL_COST", fine_plan_ref=plan_ref, fine_manifest_ref=children[0], task_key_sha256=audit.digest(pilot_row["task_key"]), native_record_sha256=pilot_row["native_record_sha256"], physical_attempt_id=pilot_row["physical_attempt_id"], job_id=pilot_row["native_job_id"], model_fingerprint=pilot_row["model_fingerprint"], native_validator_source_sha256="a2054daad25fc3ce63f9ec260b17cca86e098bec8077096e6f2edab237364922", original_adapter_source_sha256=audit.ADAPTER_SHA, new_model_calls_by_validation=0, gold_reads=0, scorer_calls=0, backend_loads_by_validation=0, native_file_refs={name:dict(path=self.remote + "/owned/" + pilot_prefix + name, sha256=by_file[pilot_prefix + name]["sha256"]) for name in ("ATTEMPT_STATE.json", "LAST_TASK_RECEIPT.json", "records.jsonl")})
        put("resource_scheduling/MEMBER4_TAIL_PILOT_NATIVE_ADMISSION_v1.json", pilot)
        if transform_bundle:
            transform_bundle(self.raw)
        bundle_path = self.root / "SOURCE_BUNDLE.zip"
        inventory = dict(files=[dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for name, raw in self.raw.items()])
        with zipfile.ZipFile(bundle_path, "x") as archive:
            for name, raw in self.raw.items():
                archive.writestr(name, raw)
            archive.writestr("BUNDLE_MANIFEST.json", encoded(inventory))
        self.bundle_ref = reference(bundle_path)
        included = native_rows if complete else native_rows[:4]
        included_prefixes = {self.scope_sha + "/member_generation/" + audit.digest(row["task_key"]) + "/" for row in included}
        export = dict(scope_sha256=self.scope_sha, software_test_only=False, tasks=copy.deepcopy(included), files=[row for row in files if any(row["path"].startswith(prefix) for prefix in included_prefixes)], metadata_sources=[dict(path=row["path"], sha256=row["sha256"]) for row in [plan_ref, *children, *(child["task_keys_ref"] for child in child_objects)]], worker_sources=[dict(manifest_sha256=reference["sha256"], adapter_ref=child["adapter_ref"], backend=child["backend"], task_keys_ref=child["task_keys_ref"]) for reference, child in zip(children, child_objects)])
        if transform_export:
            transform_export(export)
        export_raw = encoded(export)
        archive_root = self.root / "native_exports"
        archive_root.mkdir()
        archive_path = archive_root / ("native-" + "1" * 32 + ".tar.gz")
        with tarfile.open(archive_path, "w:gz") as archive:
            for row in export["files"]:
                raw = payloads[row["path"]]
                item = tarfile.TarInfo(row["path"])
                item.size = len(raw)
                archive.addfile(item, io.BytesIO(raw))
            item = tarfile.TarInfo("EXPORT_MANIFEST.json")
            item.size = len(export_raw)
            archive.addfile(item, io.BytesIO(export_raw))
        catalogue_path = self.root / "NATIVE_EXPORT_RESULT_v1.json"
        catalogue_path.write_bytes(encoded(dict(new_archives=[dict(path=self.remote + "/exports/" + archive_path.name, sha256=audit.file_sha(archive_path), bytes=archive_path.stat().st_size)])))
        native_audit = dict(status="PASS_INDEPENDENT_STREAMING_NATIVE_AUDIT", independent_tar_auditor_sha256=audit.NATIVE_AUDITOR_SHA, execution_scope_sha256=self.scope_sha, execution_scope_file_sha256=self.scope_ref["sha256"], evaluation_results_read=0, gold_or_reward_sources_read=0, model_or_scorer_calls=0, native_tasks_checked=len(export["tasks"]), archives=[dict(sha256=audit.file_sha(archive_path), manifest_raw_sha256=hashlib.sha256(export_raw).hexdigest(), manifest_semantic_sha256=audit.digest(export), native_tasks=len(export["tasks"]))], source_batches=[reference(catalogue_path)])
        audit_path = self.root / "NATIVE_AUDIT.json"
        audit_path.write_bytes(encoded(native_audit))
        self.native_audit_ref = reference(audit_path)
        gate_path = self.root / "frozen_gate.py"
        gate_path.write_bytes(Path(audit.__file__).with_name("e18_final_scoring.py").read_bytes())
        self.gate_ref = reference(gate_path)

    def run(self, **kwargs):
        return audit.audit(self.root, self.bundle_ref, self.native_audit_ref, self.scope_ref, self.gate_ref, qualification_only=True, **kwargs)


class Qualification(unittest.TestCase):
    def setUp(self):
        root = Path.cwd() / ".e18_tail_lineage_toy"
        root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_complete_source_and_native_binding(self):
        result = Fixture(self.root).run(require_complete_fine=True)
        self.assertEqual(result["archived_fine_native_tasks"], 3)
        self.assertEqual(result["archived_unchanged_checkpoint_successes"], 2)
        self.assertEqual(result["native_payload_json_decoded"], 0)
        self.assertEqual(result["pilot_export_binding"], "EXACT_THREE_NATIVE_FILES_AND_TASK_METADATA_BOUND")

    def test_partial_coverage_is_explicit(self):
        result = Fixture(self.root, complete=False).run()
        self.assertEqual(result["missing_fine_tasks"], 1)
        self.assertIn("PARTIAL", result["status"])

    def test_require_complete_rejects_partial(self):
        with self.assertRaises(ValueError):
            Fixture(self.root, complete=False).run(require_complete_fine=True)

    def test_protected_metadata_values_refused(self):
        for raw in (b'{"text":"private answer"}', b'{"Y":[1]}', b'{"reward":0}'):
            with self.assertRaisesRegex(ValueError, "before value"):
                audit.metadata_json(raw)

    def test_protected_looking_string_is_not_a_property(self):
        self.assertEqual(audit.metadata_json(b'{"note":"\\\"text\\\": value"}')["note"], '"text": value')

    def test_missing_fine_task_source_rejected(self):
        def change(value):
            value["tasks"][2]["manifest_sha256s"] = value["tasks"][2]["manifest_sha256s"][:1]
        with self.assertRaises(ValueError):
            Fixture(self.root, transform_export=change).run()

    def test_missing_plan_registration_rejected(self):
        with self.assertRaises(ValueError):
            Fixture(self.root, transform_export=lambda value: value["metadata_sources"].pop(0)).run()

    def test_changed_registered_backend_rejected(self):
        def change(value):
            value["worker_sources"][0]["backend"] = dict(mode="changed")
        with self.assertRaises(ValueError):
            Fixture(self.root, transform_export=change).run()

    def test_duplicate_physical_attempt_rejected(self):
        def change(value):
            value["tasks"][1]["physical_attempt_id"] = value["tasks"][0]["physical_attempt_id"]
        with self.assertRaises(ValueError):
            Fixture(self.root, transform_export=change).run()

    def test_changed_prefix_native_rejected(self):
        def change(value):
            value["tasks"][0]["native_record_sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            Fixture(self.root, transform_export=change).run()

    def test_changed_model_fingerprint_rejected(self):
        def change(value):
            value["tasks"][2]["model_fingerprint"] = "a" * 64
        with self.assertRaises(ValueError):
            Fixture(self.root, transform_export=change).run()

    def test_static_lease_bytes_are_required(self):
        def change(raw):
            raw.pop("owned/tail/LEASE.json")
        with self.assertRaises(ValueError):
            Fixture(self.root, transform_bundle=change).run()

    def test_frozen_gate_bytes_required(self):
        item = Fixture(self.root)
        Path(item.gate_ref["path"]).write_bytes(b"# altered gate\n")
        item.gate_ref = reference(Path(item.gate_ref["path"]))
        with self.assertRaises(ValueError):
            item.run()

    def test_native_audit_authority_required(self):
        item = Fixture(self.root)
        path = Path(item.native_audit_ref["path"])
        value = json.loads(path.read_bytes())
        value["independent_tar_auditor_sha256"] = "0" * 64
        path.write_bytes(encoded(value))
        item.native_audit_ref = reference(path)
        with self.assertRaises(ValueError):
            item.run()

    def test_production_has_no_qualification_switch(self):
        source = Path(audit.__file__).read_text(encoding="utf-8")
        self.assertNotIn('add_argument("--qualification-only"', source)

    def test_cache_flag_transition_uses_exact_prior_bytes(self):
        before = b'{"from_cache": false, "native_record": {"text": "answer stays opaque"}}\n'
        after = before.replace(b'"from_cache": false', b'"from_cache": true', 1)
        self.assertEqual(audit.cache_flag_predecessor_sha(after), hashlib.sha256(before).hexdigest())

    def test_nested_or_answer_cache_tokens_are_not_top_level(self):
        before = encoded(dict(from_cache=False, native_record=dict(from_cache=True, text='"from_cache": true')))
        after = before.replace(b'"from_cache": false', b'"from_cache": true', 1)
        self.assertEqual(audit.cache_flag_predecessor_sha(after), hashlib.sha256(before).hexdigest())

    def test_other_receipt_changes_do_not_match_prior_hash(self):
        before = b'{"from_cache": false, "status": "SUCCESS"}'
        after = b'{"from_cache": true, "status": "changed"}'
        self.assertNotEqual(audit.cache_flag_predecessor_sha(after), hashlib.sha256(before).hexdigest())

    def test_false_or_missing_current_cache_flag_rejected(self):
        for raw in (b'{"from_cache": false}', b'{"nested": {"from_cache": true}}'):
            with self.assertRaises(ValueError):
                audit.cache_flag_predecessor_sha(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualifications are append-only")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Qualification))
    report = dict(schema="e18_tail_export_lineage_qualification_v1", status="PASS" if result.wasSuccessful() else "FAIL", tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), source_sha256=audit.file_sha(audit.__file__), qualification_sha256=audit.file_sha(__file__), software_test_only=True, model_or_scoring_calls=0, gold_reads=0, native_answer_values_decoded=0, source_gate_modified=False)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
