"""Synthetic-only approval, physical-format and reference-reuse qualification."""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e18_execution_adapter import canonical_digest, sha256
from e18_reference_adapter import E18ReferenceProvider, _canonical_input, build_reference_scope


def toy_bindings(records):
    values={name:hashlib.sha256(name.encode()).hexdigest() for name in (
        "protocol_sha256","target_inputs_sha256","frozen_actions_sha256","budget_sha256")}
    values["target_ids_sha256"]=canonical_digest([[row["question_id"],row["canonical_question_sha256"]] for row in records])
    return values


class ReferenceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.root=Path(self.directory.name)
        self.question="A synthetic child has two toy apples and receives two more."
        self.qid="toy_allowed"
        self.version="toy_reference_v1"
        self.deny=[dict(question_id="toy_sealed",canonical_question_sha256=_canonical_input("distinct synthetic protected input"))]
        self.allowed=dict(question_id=self.qid,task="gsm8k",subtask=None,source_split="train",question=self.question,reference="4",choices={"A":"synthetic option"})
        # Both unapproved rows contain deliberately non-JSON reference literals.
        # A full json.loads on either would fail. The approved row alone is decoded.
        self.raw=(b'{"question_id":"toy_unknown","question":"unapproved toy input","reference":MUST_NOT_BE_DECODED}\n'+
            (json.dumps(self.allowed,ensure_ascii=False)+"\n").encode()+
            b'{"question_id":"toy_sealed","question":MUST_NOT_BE_DECODED,"reference":MUST_NOT_BE_DECODED}\n')
        self.path=self.root/"normalized.jsonl"
        self.path.write_bytes(self.raw)

    def tearDown(self):
        self.directory.cleanup()

    def make(self,*,format="normalized_qid_jsonl_v1",path=None,row_index=1,question=None,qid=None,
             task="gsm8k",subtask=None,source_hash=None,allow_scan=True):
        path=path or self.path
        records=[dict(question_id=qid or self.qid,canonical_question_sha256=_canonical_input(question or self.question),
            task=task,subtask=subtask,source_split="train" if task=="gsm8k" else "test",source_id="toy",row_index=row_index)]
        source=dict(source_id="toy",format=format,path=str(path),sha256=source_hash or sha256(path),
            task=task,subtask=subtask,source_split=records[0]["source_split"])
        scope=build_reference_scope(records,[source],toy_bindings(records),self.version,self.deny,allow_structural_byte_scan=allow_scan)
        return scope,self.approve(scope)

    def approve(self,scope):
        path=self.root/("approval_"+canonical_digest(scope)+".json")
        payload=dict(schema="e18_scope_approval_v2",status="APPROVED",cohort="e18_confirm",software_test_only=True,
            operations=["read_reference"],scope_sha256_by_role={"reference":canonical_digest(scope)},bindings=scope["bindings"])
        path.write_text(json.dumps(payload,sort_keys=True),encoding="utf-8")
        return dict(path=str(path),sha256=sha256(path))

    def get(self,provider,qid=None,canonical=None,version=None):
        return provider.get_reference(qid or self.qid,canonical or _canonical_input(self.question),version or self.version)

    def test_only_approved_reference_is_decoded_and_unapproved_values_are_skipped(self):
        scope,approval=self.make();provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        result=self.get(provider)
        self.assertEqual(result["reference"],"4")
        self.assertTrue(result["reference_revealed_new"])
        self.assertFalse(result["cache_hit"])
        stats=provider.stats()
        self.assertEqual(stats["reference_values_decoded"],1)
        self.assertEqual(stats["gold_reveal_accounts_consumed"],1)
        self.assertEqual(stats["unapproved_encoded_record_values_skipped"],2)
        self.assertEqual(stats["unapproved_reference_values_decoded"],0)
        self.assertEqual(stats["protected_input_values_decoded"],0)

    def test_shared_question_reuses_one_canonical_version_and_defensive_cache_copy(self):
        scope,approval=self.make();provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        first=self.get(provider);first["choices"]["A"]="caller mutation"
        second=self.get(provider)
        self.assertEqual(second["choices"]["A"],"synthetic option")
        self.assertTrue(second["cache_hit"]);self.assertFalse(second["reference_revealed_new"])
        self.assertEqual(provider.stats()["reference_values_decoded"],1)
        self.assertEqual(provider.stats()["reference_cache_hits"],1)

    def test_unknown_qid_is_denied_before_source_scan(self):
        scope,approval=self.make();provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        with self.assertRaises(PermissionError):self.get(provider,qid="toy_unknown")
        self.assertEqual(provider.stats()["source_scan_attempts"],0)

    def test_wrong_canonical_and_reference_version_are_denied(self):
        scope,approval=self.make();provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        with self.assertRaises(PermissionError):self.get(provider,canonical="0"*64)
        with self.assertRaises(PermissionError):self.get(provider,version="different_toy_version")
        self.assertEqual(provider.stats()["source_scan_attempts"],0)

    def test_sealed_qid_or_canonical_cannot_enter_scope(self):
        scope,_=self.make()
        for field,value in self.deny[0].items():
            records=copy.deepcopy(scope["reference_records"]);records[0][field]=value
            with self.assertRaises(PermissionError):
                build_reference_scope(records,scope["reference_sources"],toy_bindings(records),self.version,self.deny,allow_structural_byte_scan=True)

    def test_duplicate_qid_canonical_or_source_row_is_rejected(self):
        scope,_=self.make();base=scope["reference_records"][0]
        variants=[dict(base),{**base,"question_id":"another_toy_id"},
                  {**base,"question_id":"third_toy_id","canonical_question_sha256":_canonical_input("another input")},
                  {**base,"canonical_question_sha256":_canonical_input("fourth input"),"row_index":2}]
        for second in variants:
            with self.assertRaises(ValueError):
                build_reference_scope([base,second],scope["reference_sources"],toy_bindings([base,second]),self.version,self.deny,allow_structural_byte_scan=True)

    def test_wrong_source_sha_fails_before_reference_decode(self):
        scope,approval=self.make(source_hash="0"*64);provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        with self.assertRaises(ValueError):self.get(provider)
        self.assertEqual(provider.stats()["reference_values_decoded"],0)
        self.assertEqual(provider.stats()["gold_reveal_accounts_consumed"],0)

    def test_source_mutation_after_approval_fails_before_reference_decode(self):
        scope,approval=self.make();provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        self.path.write_bytes(self.raw+b" \n")
        with self.assertRaises(ValueError):self.get(provider)
        self.assertEqual(provider.stats()["reference_values_decoded"],0)

    def test_input_mismatch_fails_before_reference_value_is_decoded(self):
        scope,approval=self.make(question="a different approved synthetic input")
        provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        with self.assertRaises(ValueError):
            provider.get_reference(self.qid,_canonical_input("a different approved synthetic input"),self.version)
        self.assertEqual(provider.stats()["reference_values_decoded"],0)
        self.assertEqual(provider.stats()["gold_reveal_accounts_consumed"],0)

    def test_row_locator_mismatch_is_rejected(self):
        scope,approval=self.make(row_index=0);provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        with self.assertRaises(ValueError):self.get(provider)
        self.assertEqual(provider.stats()["reference_values_decoded"],0)

    def test_duplicate_source_qid_is_rejected_before_gold_decode(self):
        self.path.write_bytes(self.raw+(json.dumps(self.allowed)+"\n").encode())
        scope,approval=self.make();provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        with self.assertRaises(ValueError):self.get(provider)
        self.assertEqual(provider.stats()["gold_reveal_accounts_consumed"],0)

    def test_missing_qid_header_has_no_full_json_fallback(self):
        reversed_row={"task":"gsm8k",**self.allowed}
        self.path.write_text(json.dumps(reversed_row)+"\n",encoding="utf-8")
        scope,approval=self.make(row_index=0);provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        with self.assertRaises(ValueError):self.get(provider)
        self.assertEqual(provider.stats()["reference_values_decoded"],0)

    def test_explicit_raw_byte_scan_permission_is_required(self):
        scope,approval=self.make(allow_scan=False);provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        with self.assertRaises(PermissionError):self.get(provider)
        self.assertEqual(provider.stats()["source_scan_attempts"],0)

    def test_synthetic_approval_cannot_authorize_production_provider(self):
        scope,approval=self.make()
        with self.assertRaises(PermissionError):E18ReferenceProvider(scope,approval)

    def test_cached_reference_still_requires_unchanged_approval(self):
        scope,approval=self.make();provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        self.get(provider)
        path=Path(approval["path"]);value=json.loads(path.read_text());value["status"]="REVOKED"
        path.write_text(json.dumps(value),encoding="utf-8")
        with self.assertRaises((ValueError,PermissionError)):self.get(provider)
        self.assertEqual(provider.stats()["reference_cache_hits"],0)

    def test_scope_source_binding_mutation_is_rejected(self):
        scope,approval=self.make();scope["reference_sources"][0]["sha256"]="f"*64
        with self.assertRaises(ValueError):E18ReferenceProvider(scope,approval,software_test_only=True)

    def test_official_gsm_approved_row_uses_original_final_delimiter_rule(self):
        path=self.root/"gsm.jsonl"
        path.write_bytes(b'{"question":DO_NOT_DECODE,"answer":DO_NOT_DECODE}\n'+
            (json.dumps({"question":self.question,"answer":"synthetic worked text\n#### 4"})+"\n").encode())
        scope,approval=self.make(format="gsm8k_official_jsonl_v1",path=path)
        provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        self.assertEqual(self.get(provider)["reference"],"4")
        self.assertEqual(provider.stats()["reference_values_decoded"],1)

    def test_official_bbh_structure_skips_unapproved_input_and_target_values(self):
        question="Choose a synthetic option."
        path=self.root/"bbh.json"
        selected=json.dumps({"input":question,"target":"(A)"}).encode()
        path.write_bytes(b'{"examples":[{"input":DO_NOT_DECODE,"target":DO_NOT_DECODE},'+selected+
            b',{"input":DO_NOT_DECODE,"target":DO_NOT_DECODE}],"metadata":{"reference":DO_NOT_DECODE}}')
        scope,approval=self.make(format="bbh_official_examples_v1",path=path,task="bbh",subtask="logical_deduction_three_objects",question=question)
        provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        result=provider.get_reference(self.qid,_canonical_input(question),self.version)
        self.assertEqual(result["reference"],"(A)")
        self.assertEqual(provider.stats()["physical_source_records_streamed"],3)
        self.assertEqual(provider.stats()["unapproved_reference_values_decoded"],0)

    def test_official_bbh_missing_or_duplicate_examples_array_fails_closed(self):
        for data in (b'{"different_array":[]}',b'{"examples":[],"examples":[]}'):
            path=self.root/"bbh_invalid.json";path.write_bytes(data)
            scope,approval=self.make(format="bbh_official_examples_v1",path=path,task="bbh",subtask="boolean_expressions")
            provider=E18ReferenceProvider(scope,approval,software_test_only=True)
            with self.assertRaises(ValueError):self.get(provider)
            self.assertEqual(provider.stats()["reference_values_decoded"],0)

    def test_failure_after_reference_access_is_charged_once_and_fenced(self):
        bad=dict(self.allowed,reference=False)
        self.path.write_text(json.dumps(bad)+"\n",encoding="utf-8")
        scope,approval=self.make(row_index=0);provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        with self.assertRaises(ValueError):self.get(provider)
        with self.assertRaises(RuntimeError):self.get(provider)
        self.assertEqual(provider.stats()["reference_values_decoded"],1)
        self.assertEqual(provider.stats()["gold_reveal_accounts_consumed"],1)
        self.assertEqual(provider.stats()["source_scan_attempts"],1)
        self.assertEqual(provider.stats()["failed_after_reference_access"],1)

    def test_parallel_calls_decode_reference_once(self):
        scope,approval=self.make();provider=E18ReferenceProvider(scope,approval,software_test_only=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda _:self.get(provider),range(8)))
        self.assertEqual(sum(result["reference_revealed_new"] for result in results),1)
        self.assertEqual(provider.stats()["reference_values_decoded"],1)
        self.assertEqual(provider.stats()["reference_cache_hits"],7)

    def test_unsupported_codec_and_missing_locator_fail_before_source_access(self):
        scope,_=self.make();sources=copy.deepcopy(scope["reference_sources"]);sources[0]["format"]="decode_entire_json"
        with self.assertRaises(ValueError):
            build_reference_scope(scope["reference_records"],sources,toy_bindings(scope["reference_records"]),self.version,self.deny)
        records=copy.deepcopy(scope["reference_records"]);records[0]["row_index"]=-1
        with self.assertRaises(ValueError):
            build_reference_scope(records,scope["reference_sources"],toy_bindings(records),self.version,self.deny)


if __name__=="__main__":
    unittest.main()
