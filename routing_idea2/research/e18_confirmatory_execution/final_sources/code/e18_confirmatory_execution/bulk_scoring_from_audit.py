"""Bulk local scoring I/O with the unchanged approved reference/scorer/statistics.

Reuses the completed native audit and frozen score_answer. Receipts and failure
events are appended to two buffered JSONL files and closed once. This entry is
an explicitly identified operational I/O adapter, with no router/model changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
import uuid
from types import SimpleNamespace

import cpu_scope_memo as memo_api
import fast_scoring_from_audit as fast


def paired_question_point_test(candidate, baseline, task_labels, alpha=0.05):
    """Original equal-task paired t point estimate/p-value without any interval."""
    import numpy as np
    from scipy import stats

    candidate, baseline = np.asarray(candidate, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    labels = np.asarray(task_labels)
    if candidate.shape != baseline.shape or candidate.ndim != 2 or candidate.shape[0] != 3:
        raise ValueError("Require matched three-generation by question reward arrays")
    if labels.shape != (candidate.shape[1],) or set(labels) != {"gsm8k", "bbh"}:
        raise ValueError("Require both GSM8K and BBH question labels")
    if not (np.isin(candidate, (0, 1)).all() and np.isin(baseline, (0, 1)).all()) or not 0 < alpha < 1:
        raise ValueError("Binary rewards and a valid significance level are required")
    differences = (candidate-baseline).mean(axis=0)
    groups, components = {}, []
    for task in ("gsm8k", "bbh"):
        values = differences[labels == task]
        if len(values) < 2:
            raise ValueError("Each task needs at least two independent questions")
        variance = float(values.var(ddof=1))
        components.append((0.25*variance/len(values), len(values)-1))
        groups[task] = dict(questions=len(values), mean=float(values.mean()), variance=variance)
    estimate = sum(0.5*group["mean"] for group in groups.values())
    variance = sum(component for component, _ in components)
    common = dict(estimate=estimate, estimate_pp=100*estimate, confidence_interval=None,
                  confidence_level=None, task_groups=groups, original_questions=len(differences),
                  user_reporting_override="Point estimates and paired t p-values only; confidence intervals and CI bootstrap are not computed")
    if variance == 0:
        return dict(common, p_value=None, status="DEGENERATE_VARIANCE_REQUIRES_REVIEW")
    standard_error = math.sqrt(variance)
    degrees = variance**2/sum(component**2/df for component, df in components)
    statistic = estimate/standard_error
    return dict(common, standard_error=standard_error, degrees_of_freedom=degrees,
                p_value=float(2*stats.t.sf(abs(statistic), degrees)), statistic=statistic,
                generation_records=candidate.size,
                method="equal-task-weighted paired t with Satterthwaite df",
                guarantee="model-based independent-question inference, not unconditional finite-sample coverage")


def point_only_statistics(original_statistics):
    """Retain original quality/correction/Holm logic and replace only CI helper."""
    def calculate(rewards, metadata, power_api):
        result = original_statistics(rewards, metadata,
                                     SimpleNamespace(paired_question_test=paired_question_point_test))
        result["user_reporting_override"] = "No confidence intervals or CI-only bootstrap; original point estimates, paired t p-values and Holm family retained"
        return result
    return calculate


def install_bulk_scorer(context, output_root):
    """Install one bulk-I/O subclass on this context's original validator module."""
    module = context["validator"]
    base = module.SparseScoringAdapter
    destination = Path(output_root)

    class BulkSparseScoringAdapter(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Retain the original constructor's scope/scorer qualifications.
            # Its empty SQLite journal is closed before any scored observation.
            self.journal.close()
            authorization = self._authorize()
            if (authorization.get("scope_sha256_by_role", {}).get("execution") != self.scope_contract["execution_scope_sha256"]
                    or authorization.get("scope_sha256_by_role", {}).get("reference") != self.scope_contract["reference_scope_sha256"]):
                raise PermissionError("Bulk scoring requires the original approved component scopes")
            destination.mkdir(parents=True, exist_ok=True)
            self._receipt_stream = (destination / "BULK_SCORING_RECEIPTS.jsonl").open("x", encoding="utf-8")
            self._event_stream = (destination / "BULK_SCORING_EVENTS.jsonl").open("x", encoding="utf-8")
            self._reference_cache = {}
            self._resources = {}
            self._scored = {}
            self._gold = {}
            self._attempts = []
            self._closed = False
            self._scope_sha = module.digest(self.scope_contract)

        def score_one(self, execution_receipt, reference_version):
            started_wall, started_cpu = time.perf_counter(), time.process_time()
            attempt_id = uuid.uuid4().hex
            task = execution_receipt["task_key"]
            key = (task["question_id"], task["generation_seed"], task["combination_index"])
            event = dict(attempt_id=attempt_id, logical_key=list(key), status="STARTED",
                         software_test_only=self.software_test_only, reference_access_may_have_occurred=False)
            self._event_stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            try:
                self._validate_upstream(execution_receipt)
                if execution_receipt.get("approval_sha256") != self.approval_ref["sha256"]:
                    raise PermissionError("Bulk execution and scoring approval differ")
                if (key not in self.required_keys
                        or self.question_keys.get(key[0]) != task["canonical_id"]):
                    raise PermissionError("Bulk score lies outside the approved chosen-combination union")
                if key in self._scored:
                    raise ValueError("A chosen aggregation key was scored twice")
                native = execution_receipt["native_record"]
                if native["status"] != "success" or native["cost_status"] != "recorded_actual_usage":
                    raise ValueError("Bulk scoring requires audited successful paid outputs")
                result_source = native["result"]
                gold_key = (task["canonical_id"], reference_version)
                local_hit = gold_key in self._reference_cache
                if not local_hit:
                    event["reference_access_may_have_occurred"] = True
                    reference = self.reference_provider.get_reference(key[0], task["canonical_id"], reference_version)
                    self._reference_cache[gold_key] = reference
                reference = self._reference_cache[gold_key]
                if (reference["question_id"] != key[0]
                        or reference["canonical_question_sha256"] != task["canonical_id"]
                        or reference["reference_version"] != reference_version):
                    raise ValueError("Bulk projected reference identity differs")
                gold_new = gold_key not in self._gold
                self._gold[gold_key] = reference["source_identity_hash"]
                scorer_cpu, scorer_wall = time.process_time(), time.perf_counter()
                result = self.scorer.score_answer(reference["task"], result_source["text"], reference["reference"],
                                                  subtask=reference.get("subtask"), choices=reference.get("choices"))
                if result.scorer_version != module.SCORER_VERSION or result.score not in (0, 1):
                    raise ValueError("Frozen deterministic scorer result differs")
                receipt = dict(
                    schema="e18_sparse_scoring_receipt_v2", scoring_attempt_id=attempt_id,
                    scoring_io_mode="BULK_BUFFERED_JSONL_FROZEN_SCORE_ANSWER",
                    scope_sha256=self._scope_sha, approval_sha256=self.approval_ref["sha256"],
                    bindings=self.scope_contract["bindings"], software_test_only=self.software_test_only,
                    question_id=key[0], canonical_id=task["canonical_id"], generation_seed=key[1],
                    aggregation_seed=42, combination_index=key[2], team=task["team"],
                    reference_version=reference_version, reference_source_identity_hash=reference["source_identity_hash"],
                    gold_new_in_scoring_journal=gold_new, scorer_version=module.SCORER_VERSION,
                    reference_provider_cache_hit=False, reference_provider_revealed_new=not local_hit,
                    bulk_reference_cache_hit=local_hit,
                    scorer_source_sha256=module.SCORER_SOURCE_SHA256, aggregator_fingerprint=module.OPERATOR_FP,
                    source_execution_receipt_sha256=None,
                    aggregation_record_sha256=execution_receipt["native_record_sha256"],
                    physical_attempt_id=execution_receipt["physical_attempt_id"],
                    actual_source_input_tokens=result_source["prompt_tokens"],
                    actual_source_output_tokens=result_source["completion_tokens"],
                    execution_resources_already_charged_upstream=True, incremental_member_calls=0,
                    incremental_aggregation_calls=0, scoring_model_calls=0,
                    score=float(result.score), parsed=bool(result.parsed), reason=result.reason,
                    scorer_cpu_s=time.process_time()-scorer_cpu, scorer_wall_s=time.perf_counter()-scorer_wall,
                    scoring_cpu_s=time.process_time()-started_cpu, scoring_wall_s=time.perf_counter()-started_wall)
                receipt_bytes = module.canonical_bytes(receipt)
                receipt_sha = module.hashlib.sha256(receipt_bytes).hexdigest()
                sparse = {field: receipt[field] for field in module.FIELDS if field != "scoring_receipt_sha256"}
                sparse["scoring_receipt_sha256"] = receipt_sha
                self._receipt_stream.write(receipt_bytes.decode("utf-8") + "\n")
                resource = dict(physical_attempt_id=execution_receipt["physical_attempt_id"],
                                aggregation_record_sha256=execution_receipt["native_record_sha256"],
                                prompt_tokens=result_source["prompt_tokens"], completion_tokens=result_source["completion_tokens"],
                                tokens=result_source["prompt_tokens"]+result_source["completion_tokens"],
                                source_wall_time_s=result_source["wall_time_s"],
                                resource_kind="aggregation_source_reference", charged_by_execution_adapter=True,
                                incremental_member_calls_here=0, incremental_aggregation_calls_here=0)
                old = self._resources.get(resource["physical_attempt_id"])
                if old is not None and old != resource:
                    raise ValueError("A physical aggregation has conflicting bulk resource fields")
                self._resources[resource["physical_attempt_id"]] = resource
                self._scored[key] = sparse
                event.update(status="SUCCESS", deterministic_scoring_ops=1,
                             scoring_receipt_sha256=receipt_sha,
                             scoring_cpu_s=time.process_time()-started_cpu,
                             scoring_wall_s=time.perf_counter()-started_wall)
                self._event_stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                self._attempts.append(event)
                if len(self._scored) % 1000 == 0:
                    print(json.dumps(dict(event="BULK_SCORING_PROGRESS", scored_keys=len(self._scored),
                                          distinct_gold=len(self._gold))), flush=True)
                return dict(sparse_row=sparse, scoring_receipt=receipt, from_scoring_cache=False)
            except BaseException as error:
                event.update(status="FAILED_REQUIRES_REVIEW", error_type=type(error).__name__,
                             provider_stats=self.reference_provider.stats())
                self._event_stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                self._event_stream.flush()
                self._attempts.append(event)
                raise

        def ledger(self):
            resources = list(self._resources.values())
            return dict(schema="e18_sparse_scoring_resource_ledger_v2",
                        scoring_io_mode="BULK_BUFFERED_JSONL_FROZEN_SCORE_ANSWER",
                        source_execution_resources=resources, physical_aggregation_sources=len(resources),
                        actual_source_input_tokens=sum(row["prompt_tokens"] for row in resources),
                        actual_source_output_tokens=sum(row["completion_tokens"] for row in resources),
                        confirmed_reference_keys_in_scoring_journal=len(self._gold),
                        successful_sparse_keys=len(self._scored), attempts=list(self._attempts),
                        deterministic_scoring_model_calls=0, incremental_member_calls=0,
                        incremental_aggregation_calls=0, execution_resources_already_charged_upstream=True,
                        reference_provider_stats=self.reference_provider.stats(),
                        reference_cost_owner="canonical-question/reference-version; shared across recorded scoring runs",
                        source_native_validation_basis="completed independent audit reused under later explicit user authorization",
                        per_record_native_rehash=False, sql_transactions_per_scored_key=0,
                        receipt_files="BULK_SCORING_RECEIPTS.jsonl", event_file="BULK_SCORING_EVENTS.jsonl",
                        batch_scope_approval_checked=True)

        def close(self):
            if not self._closed:
                for stream in (self._receipt_stream, self._event_stream):
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
                self._closed = True

    module.SparseScoringAdapter = BulkSparseScoringAdapter
    return BulkSparseScoringAdapter


def main():
    import e18_final_scoring as final

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--completed-audit", type=Path, required=True)
    parser.add_argument("--archive-directory", type=Path, required=True)
    parser.add_argument("--historical-zip", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--performance-receipt", type=Path, required=True)
    parser.add_argument("--execute-scoring", action="store_true", required=True)
    args = parser.parse_args()
    fast.require(memo_api.file_sha(args.manifest) == args.manifest_sha256, "Original manifest SHA differs")
    fast.require(not args.performance_receipt.exists(), "Bulk performance receipt must be new")
    fast.require(memo_api.file_sha(memo_api.__file__) == fast.FROZEN_MEMO_SHA, "Original scope memo differs")
    manifest = final.worker.read_json(args.manifest)
    # Only runtime publication destinations change. Scientific fields and source
    # references retain the original8341 manifest; both identities are recorded.
    original_manifest_digest = memo_api.original_digest(manifest)
    manifest["public_output_root"] = str(args.public_root)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    memo = memo_api.ScopeMemo(final, manifest)
    memo._pin_ref(dict(path=str(args.manifest), sha256=args.manifest_sha256))
    started = time.perf_counter()
    status, error_type = "FAILED", None
    try:
        with memo:
            memo._patch(final, "final_statistics", point_only_statistics(final.final_statistics))
            context = fast.context_from_completed_audit(
                final, manifest, dict(path=str(args.completed_audit), sha256=fast.FINAL_AUDIT_SHA),
                args.archive_directory, historical_zip=args.historical_zip,
                progress=lambda row: print(json.dumps(dict(event="AUDITED_RECEIPTS_LOADED", **row)), flush=True))
            context["gate"].update(scoring_io_mode="BULK_BUFFERED_JSONL_FROZEN_SCORE_ANSWER",
                                   original_manifest_file_sha256=args.manifest_sha256,
                                   original_manifest_canonical_sha256=original_manifest_digest,
                                   bulk_entry_source_sha256=memo_api.file_sha(__file__))
            install_bulk_scorer(context, args.runtime_root)
            final.write_new_json(args.runtime_root / ("PRE_SCORING_ADMISSION_BULK_"+str(time.time_ns())+".json"), context["gate"])
            print(json.dumps(dict(event="ENTERING_BULK_FROZEN_SCORING", aggregate_keys=len(context["receipts"]))), flush=True)
            result = final.execute_scoring(manifest, context, runtime_root=args.runtime_root)
            status = result["status"]
    except BaseException as error:
        status, error_type = "FAILED_INTEGRITY_OR_EXECUTION", type(error).__name__
        raise
    finally:
        args.performance_receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.performance_receipt.open("x", encoding="utf-8") as stream:
            json.dump(dict(schema="e18_bulk_scoring_performance_v1", status=status, error_type=error_type,
                           wall_seconds=time.perf_counter()-started, source_sha256=memo_api.file_sha(__file__),
                           original_manifest_file_sha256=args.manifest_sha256,
                           original_manifest_canonical_sha256=original_manifest_digest,
                           runtime_public_root=str(args.public_root), runtime_output_root=str(args.runtime_root),
                           scope_memo=memo.stats(), original9b_execute_scoring_unchanged=True,
                           original_score_answer_and_reference_provider_unchanged=True),
                      stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
