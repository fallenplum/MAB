"""Score P2 paid outputs only after the complete native-output freeze.

The reference interface decodes only the approved question at its frozen source
locator. The old deterministic scorer and native-record checks are reused.
No source answer is opened at import or during the default gate-only CLI mode.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata

import numpy as np

import blind_evaluation as evaluation
import p2_execution_adapter as execution


CODE = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
PREPARATION = CODE.parent / "e18_mechanism_and_confirmation_preparation"
REFERENCE_API_SHA = "8848aa235d98e11005a33ff969274dc0309d02228297f4242f207663e57178ea"
REFERENCE_PLAN_SHA = "b24b870647fcd70c0b9479375e2eed2317e76f3dc963a21c830f231e9ae18060"
REFERENCE_VERSION = "safe_exact_v2_0_terminal_final_heading_fix"


class UnscorableReference(ValueError):
    """A verified approved source has an incompatible reference value."""


def canonical_input(question):
    return hashlib.sha256(" ".join(unicodedata.normalize("NFKC", question).split()).encode()).hexdigest()


def load_frozen_apis():
    source = PREPARATION / "e18_reference_adapter.py"
    if evaluation.file_sha(source) != REFERENCE_API_SHA:
        raise ValueError("Frozen reference projection source changed")
    sys.path.insert(0, str(PREPARATION))
    reference = importlib.import_module("e18_reference_adapter")
    projection = importlib.import_module("audit_sources")
    native = importlib.import_module("e18_scoring_adapter")
    scorer = native.load_frozen_scorer(ASSETS / "src")
    return reference, projection, native, scorer


def complete_output_gate(scope_path, prediction_path, outputs_frozen, aggregate_records):
    scope = evaluation.read_json(scope_path)
    packet = evaluation.load_predictions(prediction_path)
    qualified = evaluation.qualify_scope(scope, packet)
    frozen = evaluation.read_json(outputs_frozen)
    expected = qualified["required_aggregate_calls"]
    if (frozen.get("status") != "ALL_REQUIRED_OUTPUTS_FROZEN"
            or frozen.get("scope_sha256") != evaluation.file_sha(scope_path)
            or frozen.get("prediction_sha256") != evaluation.file_sha(prediction_path)
            or frozen.get("final_model_sha256") != evaluation.MODEL_SHA256
            or frozen.get("completed_required_aggregate_calls") != expected
            or frozen.get("completed_required_member_calls") != qualified["required_member_calls"]
            or not isinstance(frozen.get("aggregate_records_sha256"), str)):
        raise PermissionError("Complete matching native outputs must be frozen before any reference access")
    if not Path(aggregate_records).is_file():
        raise ValueError("Frozen aggregate receipt file is unavailable")
    return scope, packet, frozen, qualified


def verify_native_outputs(scope, packet, freeze, native_api, source_path, destination):
    """One streamed native check before gold decoding, retaining compact outputs."""
    expected = set(map(tuple, scope["required_aggregate_keys"]))
    canonical = {row["question_id"]: row["canonical_question_sha256"] for row in scope["question_keys"]}
    scope_digest = execution.canonical_digest(scope)
    seen, source_hash = set(), hashlib.sha256()
    with Path(source_path).open("rb") as source, Path(destination).open("x",encoding="utf-8",newline="\n") as target:
        for raw in source:
            source_hash.update(raw)
            if not raw.strip():
                continue
            receipt = json.loads(raw)
            if (receipt.get("schema") != "p2_execution_task_receipt_v1"
                    or receipt.get("cohort") != execution.COHORT
                    or receipt.get("scope_sha256") != scope_digest
                    or receipt.get("approval_sha256") != freeze["approval_sha256"]
                    or receipt.get("bindings") != scope["bindings"]
                    or receipt.get("software_test_only") is not False):
                raise ValueError("Native P2 receipt scope or approval differs")
            # Only the schema/cohort namespace is adapted. Scientific native
            # payload and its checksum remain the original paid execution.
            legacy_view = dict(receipt, schema="e18_execution_task_receipt_v2", cohort="e18_confirm")
            verified = native_api.verified_aggregation_source(legacy_view)
            verified["source_execution_receipt_sha256"] = execution.canonical_digest(receipt)
            key = (verified["question_id"],verified["generation_seed"],verified["combination_index"])
            if key not in expected or key in seen or canonical[key[0]] != verified["canonical_id"]:
                raise ValueError("Native output identity is duplicated or outside the frozen support")
            seen.add(key)
            target.write(json.dumps(verified,ensure_ascii=False)+"\n")
        target.flush()
        os.fsync(target.fileno())
    if seen != expected or source_hash.hexdigest() != freeze["aggregate_records_sha256"]:
        raise ValueError("Frozen aggregate outputs are incomplete or their source bytes changed")
    return dict(status="ALL_REQUIRED_NATIVE_OUTPUTS_VERIFIED",aggregate_calls=len(seen),
                aggregate_records_sha256=source_hash.hexdigest(),
                compact_verified_outputs_sha256=evaluation.file_sha(destination),
                new_reference_values_decoded=0)


class ApprovedReferences:
    """Source-indexed raw record cache with question-level gold accounting."""

    def __init__(self, plan, scope, reference_api, projection_api, event_stream):
        self.reference_api, self.projection_api = reference_api, projection_api
        self.events = event_stream
        self.sources = {row["source_id"]: row for row in plan["sources"]}
        self.records = {row["question_id"]: row for row in plan["records"]}
        self.version = plan["reference_version"]
        if self.version != REFERENCE_VERSION:
            raise ValueError("Frozen reference/scorer version differs")
        expected = {row["question_id"]: row["canonical_question_sha256"] for row in scope["question_keys"]}
        if (len(self.sources) != len(plan["sources"]) or len(self.records) != len(plan["records"])
                or {key: row["canonical_question_sha256"] for key,row in self.records.items()} != expected):
            raise PermissionError("Reference locators differ from the frozen approved cohort")
        forbidden_ids = {row["question_id"] for row in plan["sealed_exclusions"]}
        forbidden_hashes = {row["canonical_question_sha256"] for row in plan["sealed_exclusions"]}
        if (not forbidden_ids or set(self.records) & forbidden_ids
                or set(expected.values()) & forbidden_hashes):
            raise PermissionError("Sealed references cannot enter P2")
        self.allowed_rows = defaultdict(set)
        metadata = {row["question_id"]: row for row in scope["question_keys"]}
        for record in self.records.values():
            source = self.sources[record["source_id"]]
            if (record["task"] != metadata[record["question_id"]]["task"]
                    or (record.get("subtask") or "") != (metadata[record["question_id"]].get("subtask") or "")
                    or record["task"] != source["task"]
                    or record["source_split"] != source["source_split"]):
                raise ValueError("Reference locator task/source identity differs")
            self.allowed_rows[record["source_id"]].add(record["row_index"])
        self.raw_rows, self.cache, self.accounts = {}, {}, set()
        self.references_decoded = 0

    def _source_records(self, source_id):
        if source_id in self.raw_rows:
            return self.raw_rows[source_id]
        source = self.sources[source_id]
        path = Path(source["path"])
        if any(part.casefold() in {"sealed", "sealed_test", "sealed200"} for part in path.parts):
            raise PermissionError("Sealed source path is forbidden")
        if evaluation.file_sha(path) != source["sha256"]:
            raise ValueError("Frozen reference source bytes changed before decoding")
        needed = self.allowed_rows[source_id]
        if source["format"] == "bbh_official_examples_v1":
            records = self.projection_api.official_example_objects(path)
            selected = {index:raw.encode("utf-8") for index,raw in enumerate(records) if index in needed}
        elif source["format"] in {"gsm8k_official_jsonl_v1","normalized_qid_jsonl_v1"}:
            with path.open("rb") as stream:
                selected = {index:raw for index,raw in enumerate(stream) if index in needed}
        else:
            raise ValueError("Unsupported approved reference source format")
        if set(selected) != needed:
            raise ValueError("A frozen approved reference row is missing")
        self.raw_rows[source_id] = selected
        return selected

    def get(self, question_id):
        if question_id not in self.records:
            raise PermissionError("Question is outside the approved reference cohort")
        if question_id in self.cache:
            return self.cache[question_id]
        record = self.records[question_id]
        source = self.sources[record["source_id"]]
        raw = self._source_records(record["source_id"])[record["row_index"]]
        projection = self.reference_api._project_object
        field = {"bbh_official_examples_v1":"input", "gsm8k_official_jsonl_v1":"question", "normalized_qid_jsonl_v1":"question"}[source["format"]]
        question = projection(raw,{field}).get(field)
        if not isinstance(question,str) or canonical_input(question) != record["canonical_question_sha256"]:
            raise ValueError("Source input identity differs before gold decoding")
        account = (record["canonical_question_sha256"],self.version)
        self.events.write(json.dumps(dict(event="REFERENCE_ACCESS_STARTED",question_id=question_id,
            canonical_question_sha256=account[0],reference_version=self.version,
            gold_new=account not in self.accounts),ensure_ascii=False)+"\n")
        self.events.flush()
        self.accounts.add(account)
        target_field = {"bbh_official_examples_v1":"target", "gsm8k_official_jsonl_v1":"answer", "normalized_qid_jsonl_v1":"reference"}[source["format"]]
        payload = projection(raw,{target_field,"choices"})
        self.references_decoded += 1
        reference = payload.get(target_field)
        if not isinstance(reference,str) or not reference.strip():
            raise UnscorableReference("Approved reference is not nonempty text")
        if source["format"] == "gsm8k_official_jsonl_v1":
            if "####" not in reference:
                raise UnscorableReference("Frozen GSM final-reference delimiter is absent")
            reference = reference.rsplit("####",1)[1].strip()
        result = dict(question_id=question_id,canonical_question_sha256=account[0],
            reference_version=self.version,task=record["task"],subtask=record.get("subtask"),
            reference=reference,choices=payload.get("choices"),question=question)
        self.cache[question_id] = result
        self.events.write(json.dumps(dict(event="REFERENCE_PROJECTED",question_id=question_id,
            canonical_question_sha256=account[0],reference_version=self.version),ensure_ascii=False)+"\n")
        return result


def canonical_reference(reference, scorer):
    """Reuse the E18 unique visible-choice-body normalization rule."""
    result = dict(reference)
    candidate, _ = scorer.original._candidate(result["reference"])
    evaluator = scorer._evaluator(result["task"],result.get("subtask"))
    if scorer.original._canonical(evaluator,candidate,result.get("choices")) is not None:
        return result
    if result["task"] != "bbh" or result.get("subtask") not in scorer.original.CHOICE_TASKS:
        raise UnscorableReference("Frozen reference is unscorable for its evaluator")
    matches = list(re.finditer(r"^[ \t]*\(([A-Z])\)[ \t]*(.*?)(?=^[ \t]*\([A-Z]\)[ \t]*|\Z)",
                               result["question"], re.MULTILINE|re.DOTALL))
    labels = [match[1] for match in matches]
    if len(labels)<2 or len(set(labels))!=len(labels):
        raise UnscorableReference("Reference has no unique visible option labels")
    target = scorer.original._normal_text(candidate)
    chosen = [match[1] for match in matches if scorer.original._normal_text(match[2]) == target]
    if len(chosen) != 1:
        raise UnscorableReference("Reference body does not uniquely match a visible choice")
    result["reference"] = "("+chosen[0]+")"
    return result


def run(args):
    scope, packet, freeze, qualified = complete_output_gate(
        args.scope,args.predictions,args.outputs_frozen,args.aggregate_records)
    if not args.execute_scoring:
        return {**qualified,"status":"MATCHING_OUTPUT_FREEZE_GATE_PASSED"}
    approval = dict(path=str(args.approval),sha256=evaluation.file_sha(args.approval))
    execution.validate_scope_approval(approval,scope,"read_reference")
    execution.validate_scope_approval(approval,scope,"score_outputs")
    if approval["sha256"] != freeze["approval_sha256"]:
        raise PermissionError("Scoring approval differs from native execution approval")
    plan = evaluation.read_json(args.reference_plan)
    if evaluation.file_sha(args.reference_plan) != REFERENCE_PLAN_SHA or freeze["reference_plan_sha256"] != REFERENCE_PLAN_SHA:
        raise PermissionError("Reference locators changed after the complete-output freeze")
    reference_api, projection, native, scorer = load_frozen_apis()
    args.output_root.mkdir(parents=True,exist_ok=True)
    verified_path = args.output_root/"VERIFIED_NATIVE_AGGREGATIONS.jsonl"
    audit = verify_native_outputs(scope,packet,freeze,native,args.aggregate_records,verified_path)
    evaluation.write_json(args.output_root/"NATIVE_OUTPUTS_VERIFIED_BEFORE_GOLD.json",audit)
    excluded, normalized = {}, {}
    scores, scorer_calls = [], 0
    started = time.process_time()
    with (args.output_root/"REFERENCE_ACCESS_EVENTS.jsonl").open("x",encoding="utf-8") as events, \
            (args.output_root/"PAID_SUPPORT_SCORES.jsonl").open("x",encoding="utf-8") as target:
        provider = ApprovedReferences(plan,scope,reference_api,projection,events)
        with verified_path.open(encoding="utf-8") as stream:
            for raw in stream:
                output = json.loads(raw)
                question_id = output["question_id"]
                if question_id in excluded:
                    continue
                if question_id not in normalized:
                    try:
                        reference = provider.get(question_id)
                        normalized[question_id] = canonical_reference(reference,scorer)
                    except UnscorableReference as error:
                        excluded[question_id] = dict(question_id=question_id,error_type=type(error).__name__,reason=str(error))
                        continue
                reference = normalized[question_id]
                scored = scorer.score_answer(reference["task"],output["prediction"],reference["reference"],
                    subtask=reference.get("subtask"),choices=reference.get("choices"))
                scorer_calls += 1
                if scored.scorer_version != REFERENCE_VERSION or scored.score not in (0,1):
                    raise ValueError("Frozen deterministic scorer returned an invalid result")
                row = dict(question_id=question_id,generation_seed=output["generation_seed"],
                    combination_index=output["combination_index"],score=float(scored.score),
                    parsed=bool(scored.parsed),reason=scored.reason,
                    native_record_sha256=output["aggregation_record_sha256"],
                    reference_version=REFERENCE_VERSION)
                target.write(json.dumps(row,ensure_ascii=False)+"\n")
                scores.append(row)
        target.flush()
        os.fsync(target.fileno())
        provider_summary = dict(unique_gold_accounts=len(provider.accounts),
            reference_values_decoded=provider.references_decoded,deterministic_scorer_calls=scorer_calls,
            scoring_cpu_seconds=time.process_time()-started,new_llm_calls=0,sealed_answer_values_decoded=0)
    evaluation.write_json(args.output_root/"UNSCORABLE_CASES_PRIVATE.json",dict(cases=list(excluded.values())))
    evaluation.write_json(args.output_root/"REFERENCE_AND_SCORING_COSTS.json",provider_summary)
    report, arrays = evaluation.project_scores(packet,scores,excluded)
    report.update(scope_sha256=evaluation.file_sha(args.scope),prediction_sha256=evaluation.file_sha(args.predictions),
                  final_model_sha256=evaluation.MODEL_SHA256,reference_and_scoring_costs=provider_summary)
    evaluation.write_json(args.output_root/"P2_POINT_RESULTS.json",report)
    with (args.output_root/"P2_POLICY_UTILITIES_PRIVATE.npz").open("xb") as stream:
        np.savez_compressed(stream,**arrays)
    return dict(status="P2_FROZEN_POLICY_SCORING_COMPLETE",valid_questions=report["valid_questions"],
                requested_questions=report["requested_questions"],primary_difference_pp=report["primary"]["difference_pp"],
                excluded_questions=len(excluded),new_llm_calls=0,bootstrap_repetitions=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for argument in ("scope","predictions","outputs-frozen","aggregate-records","output-root"):
        parser.add_argument("--"+argument,type=Path,required=True)
    parser.add_argument("--reference-plan",type=Path)
    parser.add_argument("--approval",type=Path)
    parser.add_argument("--execute-scoring",action="store_true")
    args = parser.parse_args()
    if args.execute_scoring and (args.approval is None or args.reference_plan is None):
        parser.error("Authorized scoring needs the explicit approval and frozen reference locator plan")
    print(json.dumps(run(args),ensure_ascii=False))
