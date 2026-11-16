"""Qualification of the P2 output-freeze and reference boundaries using toys."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import blind_evaluation as blind
import score_paid_outputs as paid


def rejects(function):
    try:
        function()
    except (ValueError, PermissionError):
        return True
    return False


def main():
    checks = {}
    # Reference loading is unavailable to the incomplete-output gate by design.
    with patch.object(blind,"read_json",return_value={"status":"RUNNING"}), \
            patch.object(blind,"load_predictions",return_value={}), \
            patch.object(blind,"qualify_scope",return_value={"required_aggregate_calls":3,"required_member_calls":9}), \
            patch.object(blind,"file_sha",return_value="0"*64):
        checks["incomplete_outputs_rejected_before_reference_loader"] = rejects(lambda: paid.complete_output_gate(
            Path("scope"),Path("predictions"),Path("running"),Path("nonexistent")))

    # Use the exact frozen byte-level parser on entirely artificial source text.
    reference_api, projection, native, scorer = paid.load_frozen_apis()
    with tempfile.TemporaryDirectory(prefix="p2-reference-toy-") as folder:
        root = Path(folder)
        path = root/"toy_source.jsonl"
        question = "Synthetic qualification question only."
        canonical = paid.canonical_input(question)
        path.write_text(json.dumps({"question":question,"answer":"demo steps #### 7"})+"\n",encoding="utf-8")
        source = dict(source_id="synthetic",path=str(path),sha256=blind.file_sha(path),
            format="gsm8k_official_jsonl_v1",task="gsm8k",source_split="train",subtask=None)
        row = dict(question_id="synthetic-approved",canonical_question_sha256=canonical,
            task="gsm8k",subtask=None,source_split="train",source_id="synthetic",row_index=0)
        scope = dict(question_keys=[dict(row)])
        plan = dict(sources=[source],records=[row],reference_version=paid.REFERENCE_VERSION,
            sealed_exclusions=[dict(question_id="synthetic-protected",canonical_question_sha256="f"*64)])
        events = io.StringIO()
        provider = paid.ApprovedReferences(plan,scope,reference_api,projection,events)
        checks["provider_construction_decodes_zero_reference_values"] = provider.references_decoded == 0
        checks["unapproved_question_rejected_before_source_access"] = rejects(lambda: provider.get("synthetic-protected"))
        reference = provider.get("synthetic-approved")
        checks["official_gsm_terminal_reference_projection"] = reference["reference"] == "7"
        checks["canonical_and_reference_version_deduplicates_gold"] = provider.get("synthetic-approved") == reference and len(provider.accounts) == 1 and provider.references_decoded == 1
        checks["frozen_scorer_correct_synthetic_answer"] = scorer.score_answer("gsm8k","Final answer: 7",reference["reference"]).score == 1
        checks["frozen_scorer_incorrect_synthetic_answer"] = scorer.score_answer("gsm8k","Final answer: 8",reference["reference"]).score == 0
        bad_plan = json.loads(json.dumps(plan))
        bad_plan["sealed_exclusions"] = [dict(question_id="synthetic-approved",canonical_question_sha256=canonical)]
        checks["sealed_identity_cannot_be_a_reference"] = rejects(lambda: paid.ApprovedReferences(bad_plan,scope,reference_api,projection,io.StringIO()))
        bad_plan = json.loads(json.dumps(plan))
        bad_plan["sources"][0]["sha256"] = "0"*64
        provider_bad = paid.ApprovedReferences(bad_plan,scope,reference_api,projection,io.StringIO())
        checks["source_change_aborts_before_reference_decode"] = rejects(lambda: provider_bad.get("synthetic-approved")) and provider_bad.references_decoded == 0
        checks["integrity_errors_are_not_unscorable_exclusions"] = not issubclass(ValueError,paid.UnscorableReference)
        bbh_path = root/"toy_bbh.json"
        bbh_question = "Synthetic boolean statement is True."
        bbh_canonical = paid.canonical_input(bbh_question)
        bbh_path.write_text(json.dumps({"examples":[{"input":bbh_question,"target":"True"}]}),encoding="utf-8")
        bbh_source = dict(source_id="synthetic-bbh",path=str(bbh_path),sha256=blind.file_sha(bbh_path),
            format="bbh_official_examples_v1",task="bbh",source_split="test",subtask="boolean_expressions")
        bbh_row = dict(question_id="synthetic-bbh-approved",canonical_question_sha256=bbh_canonical,
            task="bbh",subtask="boolean_expressions",source_split="test",source_id="synthetic-bbh",row_index=0)
        bbh_plan = dict(sources=[bbh_source],records=[bbh_row],reference_version=paid.REFERENCE_VERSION,
            sealed_exclusions=plan["sealed_exclusions"])
        bbh_provider = paid.ApprovedReferences(bbh_plan,{"question_keys":[bbh_row]},reference_api,projection,io.StringIO())
        bbh_reference = bbh_provider.get("synthetic-bbh-approved")
        checks["official_bbh_string_example_converted_to_frozen_byte_projection"] = bbh_reference["reference"] == "True"
    if not all(checks.values()):
        raise AssertionError(checks)
    report = dict(schema="p2_paid_scorer_synthetic_qualification_v1",status="PASS",checks=checks,
        synthetic_reference_values_decoded=2,real_reference_values_decoded=0,new_llm_calls=0,
        frozen_scorer_version=scorer.SCORER_VERSION,
        score_paid_outputs_sha256=blind.file_sha(Path(paid.__file__)),
        blind_evaluation_sha256=blind.file_sha(Path(blind.__file__)),
        confidence_intervals_computed=False,bootstrap_repetitions=0,
        fixture_correction="The first toy used unsupported FINAL: syntax; corrected to the existing parser's Final answer: format. Frozen scorer unchanged.")
    Path(__file__).with_name("P2_PAID_SCORER_QUALIFICATION.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","checks":len(checks),"real_gold_values_read":0},ensure_ascii=False))


if __name__ == "__main__":
    main()
