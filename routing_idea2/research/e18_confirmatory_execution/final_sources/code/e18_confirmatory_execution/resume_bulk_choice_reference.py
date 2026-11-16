"""Resume bulk scores with an explicit reference-only choice-body adaptation.

The original812e bulk source, scorer, prediction parser and raw targets remain
unchanged. A noncanonical official choice-task reference body can be mapped to
its unique visible option label. Completed prior JSONL scores are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import bulk_scoring_from_audit as bulk


ORIGINAL_BULK_SHA = "812e72bed58ae6ef2aa50fd4464318965d3a6223f67da46afb45b75d89e3afd0"


def adapt_choice_reference(reference, question, original):
    """Map only a uniquely matched noncanonical choice reference to its label."""
    if reference["task"] != "bbh" or reference.get("subtask") not in original.CHOICE_TASKS:
        return reference, None
    candidate, _ = original._candidate(reference["reference"])
    if original._canonical(reference["subtask"], candidate, reference.get("choices")) is not None:
        return reference, None
    matches = list(re.finditer(r"^[ \t]*\(([A-Z])\)[ \t]*(.*?)(?=^[ \t]*\([A-Z]\)[ \t]*|\Z)",
                               question, re.MULTILINE | re.DOTALL))
    labels = [match[1] for match in matches]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Noncanonical official choice reference lacks unique visible option labels")
    target = original._normal_text(candidate)
    selected = [match[1] for match in matches if original._normal_text(match[2]) == target]
    if len(selected) != 1:
        raise ValueError("Official choice reference body does not uniquely match one visible option")
    adapted = dict(reference, reference="("+selected[0]+")")
    # Keep choices unchanged, preserving the original prediction acceptance.
    event = dict(rule="REFERENCE_ONLY_UNIQUE_VISIBLE_CHOICE_BODY_TO_LABEL",
                 subtask=reference["subtask"], chosen_label=selected[0], visible_options=len(labels),
                 raw_reference_sha256=hashlib.sha256(reference["reference"].encode()).hexdigest())
    return adapted, event


def read_previous_scores(path):
    previous = {}
    with Path(path).open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            encoded = raw.rstrip(b"\r\n")
            receipt = json.loads(encoded)
            key = (receipt["question_id"], receipt["generation_seed"], receipt["combination_index"])
            if key in previous:
                raise ValueError("The previous completed score stream repeats a key")
            previous[key] = (receipt, encoded)
    return previous


def install_reference_adaptation(final, context):
    memo_loader = final.load_reference_api
    original = context["validator"].load_frozen_scorer(context["adapter"].apis.root / "src").original

    def load(reference_ref):
        module = memo_loader(reference_ref)
        base = module.E18ReferenceProvider

        class ReferenceBodyAdapter(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._choice_mapping_events = []

            def get_reference(self, qid, canonical, version):
                reference = super().get_reference(qid, canonical, version)
                result, event = adapt_choice_reference(reference, context["adapter"].questions[qid]["question"], original)
                if event is not None:
                    self._choice_mapping_events.append(dict(event, question_id=qid, canonical_id=canonical,
                                                           reference_version=version,
                                                           source_identity_hash=reference["source_identity_hash"]))
                return result

            def stats(self):
                result = super().stats()
                result.update(reference_only_choice_body_mappings=len(self._choice_mapping_events),
                              reference_choice_mapping_by_subtask={name: sum(e["subtask"] == name for e in self._choice_mapping_events)
                                                                  for name in sorted({e["subtask"] for e in self._choice_mapping_events})},
                              prediction_parser_unchanged=True)
                return result

            def audit_events(self):
                return super().audit_events()+[dict(event="REFERENCE_ONLY_CHOICE_BODY_MAPPED", **event)
                                              for event in self._choice_mapping_events]

        module.E18ReferenceProvider = ReferenceBodyAdapter
        return module

    final.load_reference_api = load


def install_resuming_scorer(context, output_root, previous):
    base = bulk.install_bulk_scorer(context, output_root)
    module = context["validator"]

    class ResumingBulkScorer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._reused_previous = 0

        def score_one(self, execution_receipt, reference_version):
            task = execution_receipt["task_key"]
            key = (task["question_id"], task["generation_seed"], task["combination_index"])
            if key not in previous:
                return super().score_one(execution_receipt, reference_version)
            receipt, encoded = previous[key]
            expected = dict(question_id=key[0], canonical_id=task["canonical_id"], generation_seed=key[1],
                            combination_index=key[2], reference_version=reference_version,
                            scope_sha256=self.scope_contract["execution_scope_sha256"],
                            approval_sha256=self.approval_ref["sha256"],
                            aggregation_record_sha256=execution_receipt["native_record_sha256"],
                            physical_attempt_id=execution_receipt["physical_attempt_id"])
            # The receipt's scope is the scoring component, while upstream is
            # separately checked with the unchanged method.
            expected["scope_sha256"] = self._scope_sha
            if key in self._scored or any(receipt.get(name) != value for name, value in expected.items()):
                raise ValueError("A previous committed score differs from this frozen source")
            self._validate_upstream(execution_receipt)
            source = execution_receipt["native_record"]["result"]
            if receipt["actual_source_input_tokens"] != source["prompt_tokens"] or receipt["actual_source_output_tokens"] != source["completion_tokens"]:
                raise ValueError("Previous score actual Token fields differ")
            sparse = {field: receipt[field] for field in module.FIELDS if field != "scoring_receipt_sha256"}
            sparse["scoring_receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
            self._receipt_stream.write(encoded.decode("utf-8")+"\n")
            resource = dict(physical_attempt_id=receipt["physical_attempt_id"],
                            aggregation_record_sha256=receipt["aggregation_record_sha256"],
                            prompt_tokens=source["prompt_tokens"], completion_tokens=source["completion_tokens"],
                            tokens=source["prompt_tokens"]+source["completion_tokens"], source_wall_time_s=source["wall_time_s"],
                            resource_kind="aggregation_source_reference", charged_by_execution_adapter=True,
                            incremental_member_calls_here=0, incremental_aggregation_calls_here=0)
            self._resources[resource["physical_attempt_id"]] = resource
            self._scored[key] = sparse
            self._gold[(receipt["canonical_id"], reference_version)] = receipt["reference_source_identity_hash"]
            self._reused_previous += 1
            event = dict(status="REUSED_PREVIOUS_COMMITTED_SCORE", logical_key=list(key),
                         scoring_receipt_sha256=sparse["scoring_receipt_sha256"], deterministic_scoring_ops=0)
            self._attempts.append(event)
            self._event_stream.write(json.dumps(event, ensure_ascii=False)+"\n")
            return dict(sparse_row=sparse, scoring_receipt=receipt, from_scoring_cache=True)

        def ledger(self):
            result = super().ledger()
            result.update(previous_committed_scores_reused=self._reused_previous,
                          new_deterministic_scores=len(self._scored)-self._reused_previous,
                          reference_only_choice_interface_adaptation=True,
                          original_prediction_parser_unchanged=True,
                          gold_accounting="canonical/reference-version union of cached prior scores and new provider reads")
            return result

    module.SparseScoringAdapter = ResumingBulkScorer


def main():
    import e18_final_scoring as final

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--completed-audit", type=Path, required=True)
    parser.add_argument("--archive-directory", type=Path, required=True)
    parser.add_argument("--historical-zip", type=Path, required=True)
    parser.add_argument("--previous-scores", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--performance-receipt", type=Path, required=True)
    parser.add_argument("--execute-scoring", action="store_true", required=True)
    args = parser.parse_args()
    fast, memo_api = bulk.fast, bulk.memo_api
    fast.require(memo_api.file_sha(bulk.__file__) == ORIGINAL_BULK_SHA, "Original bulk source differs")
    fast.require(memo_api.file_sha(args.manifest) == args.manifest_sha256, "Original manifest bytes differ")
    fast.require(not args.performance_receipt.exists(), "Resume performance receipt must be new")
    previous = read_previous_scores(args.previous_scores)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    manifest = final.worker.read_json(args.manifest)
    manifest["public_output_root"] = str(args.public_root)
    memo = memo_api.ScopeMemo(final, manifest)
    memo._pin_ref(dict(path=str(args.manifest), sha256=args.manifest_sha256))
    started = time.perf_counter()
    status, error_type = "FAILED", None
    try:
        with memo:
            memo._patch(final, "final_statistics", bulk.point_only_statistics(final.final_statistics))
            context = fast.context_from_completed_audit(final, manifest,
                dict(path=str(args.completed_audit), sha256=fast.FINAL_AUDIT_SHA),
                args.archive_directory, historical_zip=args.historical_zip,
                progress=lambda row: print(json.dumps(dict(event="AUDITED_RECEIPTS_LOADED", **row)), flush=True))
            context["gate"].update(reference_interface_adaptation="REFERENCE_ONLY_UNIQUE_VISIBLE_CHOICE_BODY_TO_LABEL",
                                   prediction_parser_unchanged=True, previous_successful_scores_available=len(previous),
                                   original_manifest_file_sha256=args.manifest_sha256,
                                   resume_entry_source_sha256=memo_api.file_sha(__file__))
            install_reference_adaptation(final, context)
            install_resuming_scorer(context, args.runtime_root, previous)
            final.write_new_json(args.runtime_root/("PRE_SCORING_ADMISSION_RESUME_"+str(time.time_ns())+".json"), context["gate"])
            print(json.dumps(dict(event="RESUMING_PREVIOUS_SCORES_THEN_FROZEN_SCORING", cached_scores=len(previous))), flush=True)
            result = final.execute_scoring(manifest, context, runtime_root=args.runtime_root)
            status = result["status"]
    except BaseException as error:
        status, error_type = "FAILED_INTEGRITY_OR_EXECUTION", type(error).__name__
        raise
    finally:
        args.performance_receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.performance_receipt.open("x", encoding="utf-8") as stream:
            json.dump(dict(schema="e18_reference_choice_resume_performance_v1", status=status,
                           error_type=error_type, wall_seconds=time.perf_counter()-started,
                           source_sha256=memo_api.file_sha(__file__), original_bulk_source_sha256=ORIGINAL_BULK_SHA,
                           previous_successful_scores_available=len(previous), original_manifest_sha256=args.manifest_sha256,
                           reference_only_adaptation=True, prediction_parser_unchanged=True,
                           scope_memo=memo.stats()), stream, ensure_ascii=False, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
