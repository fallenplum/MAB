"""Finish every evaluable frozen output and explicitly retain unscorable inputs.

Reuse prior committed scores. Keep original inputs, models, raw targets and the
frozen prediction scorer. Only valid questions enter accuracy and paired tests.
All executed resources remain charged. Missing outcomes are never filled zero.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import resume_bulk_choice_reference as resume


bulk = resume.bulk
fast = bulk.fast
memo_api = bulk.memo_api


def full_cohort_bounds(rewards, valid_mask, metadata, actions, policies):
    """Deterministic missing-outcome bounds over the original requested pool."""
    labels = np.asarray([row["task"] for row in metadata])
    valid_labels = labels[valid_mask]
    quality = {}
    for index, policy in enumerate(policies):
        lower = upper = 0.0
        for task in ("gsm8k", "bbh"):
            total = int((labels == task).sum())
            known = float(rewards[index][:, valid_labels == task].mean(axis=0).sum())
            unknown = int(((labels == task) & ~valid_mask).sum())
            lower += 0.5*known/total
            upper += 0.5*(known+unknown)/total
        quality[policy] = dict(lower=lower, upper=upper, lower_pp=100*lower, upper_pp=100*upper)
    comparisons = {}
    for a, b in ((2, 1), (2, 0), (1, 0), (2, 3), (1, 3), (0, 3)):
        known, radius = 0.0, 0.0
        unknown_different_actions = 0
        for task in ("gsm8k", "bbh"):
            total = int((labels == task).sum())
            delta = (rewards[a]-rewards[b]).mean(axis=0)
            known += 0.5*float(delta[valid_labels == task].sum())/total
            ambiguous = (~valid_mask) & (labels == task) & (actions[a] != actions[b])
            count = int(ambiguous.sum())
            unknown_different_actions += count
            radius += 0.5*count/total
        comparisons[policies[a]+" minus "+policies[b]] = dict(
            known_difference_pp=100*known, lower_pp=100*(known-radius), upper_pp=100*(known+radius),
            excluded_questions_with_different_actions=unknown_different_actions,
            basis="Deterministic worst/best missing outcomes; identical selected actions have exactly zero paired difference; not a confidence interval")
    return quality, comparisons


def finish(final, manifest, context, previous, runtime_root, *, quarantine_subtasks=()):
    runtime_root = Path(runtime_root)
    metadata = context["metadata"]
    provider_module = final.load_reference_api(manifest["reference_adapter_ref"])
    provider = provider_module.E18ReferenceProvider(context["reference_scope"], context["approval_ref"])
    resume.install_resuming_scorer(context, runtime_root, previous)
    module = context["validator"]
    scorer = module.SparseScoringAdapter(context["approval_ref"], context["scoring_scope"], provider,
        runtime_root/"SCORING_JOURNAL.sqlite3", runtime_root/"scoring_receipts",
        scorer_source_root=context["adapter"].apis.root/"src")
    by_qid = {}
    for key in sorted(context["by_key"]):
        by_qid.setdefault(key[0], []).append(key)
    valid = np.ones(len(metadata), dtype=bool)
    excluded, mappings, sparse, gold_keys = [], [], [], set()
    version = context["reference_scope"]["reference_version"]
    try:
        for question_index, meta in enumerate(metadata):
            qid, canonical = meta["question_id"], meta["canonical_question_sha256"]
            keys = by_qid[qid]
            fully_cached = all(key in previous for key in keys)
            reason, reference = None, None
            if meta["subtask"] in quarantine_subtasks:
                reason = "EXPLICIT_SOURCE_CONVERSION_QUARANTINE"
            elif not fully_cached:
                before_gold = provider.stats()["gold_reveal_accounts_consumed"]
                try:
                    reference = provider.get_reference(qid, canonical, version)
                except ValueError:
                    if provider.stats()["gold_reveal_accounts_consumed"] <= before_gold:
                        raise
                    gold_keys.add((canonical, version))
                    reason = "INVALID_OFFICIAL_REFERENCE_FORMAT_AFTER_DECODE"
                if reference is not None:
                    gold_keys.add((canonical, version))
                    candidate, _ = scorer.scorer.original._candidate(reference["reference"])
                    evaluator = scorer.scorer._evaluator(reference["task"], reference.get("subtask"))
                    normalized = scorer.scorer.original._canonical(evaluator, candidate, reference.get("choices"))
                    if normalized is None:
                        try:
                            reference, mapping = resume.adapt_choice_reference(
                                reference, context["adapter"].questions[qid]["question"], scorer.scorer.original)
                            if mapping is None:
                                reason = "INVALID_REFERENCE_FOR_FROZEN_EVALUATOR"
                            else:
                                mappings.append(dict(mapping, question_id=qid, canonical_id=canonical))
                        except ValueError:
                            reason = "CHOICE_REFERENCE_HAS_NO_UNIQUE_VISIBLE_OPTION"
            if reason is not None:
                valid[question_index] = False
                excluded.append(dict(question_id=qid, canonical_id=canonical, task=meta["task"],
                                     subtask=meta["subtask"], source_frame=meta["source_frame"],
                                     reason=reason, status="UNSCORABLE_INPUT_REFERENCE"))
                print(json.dumps(dict(event="UNSCORABLE_QUESTION_RETAINED", excluded_questions=len(excluded),
                                      subtask=meta["subtask"], reason=reason)), flush=True)
                continue
            if reference is not None:
                scorer._reference_cache[(canonical, version)] = reference
            for key in keys:
                scored = scorer.score_one(context["by_key"][key], version)
                sparse.append(scored["sparse_row"])
                gold_keys.add((canonical, version))
            if (question_index+1) % 100 == 0:
                print(json.dumps(dict(event="EVALUABLE_SCORING_PROGRESS", questions_processed=question_index+1,
                                      scored_keys=len(sparse), excluded_questions=len(excluded),
                                      cached_scores_reused=scorer._reused_previous)), flush=True)
        ledger = scorer.ledger()
    finally:
        scorer.close()
    valid_metadata = [row for row, keep in zip(metadata, valid) if keep]
    actions = np.asarray(context["adapter"].scope["frozen_policy_actions"], dtype=np.int64)
    from sparse_policy_projection import project_frozen_policy_scores
    projected = project_frozen_policy_scores(list(final.POLICIES), actions[:, valid],
        [row["question_id"] for row in valid_metadata],
        [row["canonical_question_sha256"] for row in valid_metadata], sparse)
    rewards = projected["rewards"]
    statistics = bulk.point_only_statistics(final.final_statistics)(rewards, valid_metadata, None)
    quality_bounds, gain_bounds = full_cohort_bounds(rewards, valid, metadata, actions, list(final.POLICIES))
    statistics.update(quality_full_requested_bounds=quality_bounds,
                      primary_full_requested_gain_bounds=gain_bounds["TSPO_uniform minus SPO_coordinate"],
                      full_requested_gain_bounds=gain_bounds,
                      accuracy_scope="Complete evaluable original questions only; equal GSM8K/BBH task weights")
    costs = list(context["costs"].values())
    totals = {stage: dict(physical_attempts=sum(row["stage"] == stage for row in costs),
                         prompt_tokens=sum(row["prompt_tokens"] for row in costs if row["stage"] == stage),
                         completion_tokens=sum(row["completion_tokens"] for row in costs if row["stage"] == stage))
              for stage in ("member_generation", "aggregation")}
    service = {}
    for p, policy in enumerate(final.POLICIES):
        prompt = completion = 0
        for index, row in enumerate(metadata):
            for seed in (42, 137, 2024):
                action = int(actions[p, index])
                records = [context["by_key"][(row["question_id"], seed, action)]["native_record"]]
                records.extend(context["members"][(row["question_id"], seed, member)] for member in context["module"].TEAMS[action])
                prompt += sum(record["result"]["prompt_tokens"] for record in records)
                completion += sum(record["result"]["completion_tokens"] for record in records)
        service[policy] = dict(member_calls=9*len(metadata), aggregation_calls=3*len(metadata),
                               measured_prompt_tokens=prompt, measured_completion_tokens=completion,
                               accounting="Full requested cohort, including unscorable inputs; single-policy cost projection")
    result = dict(schema="e18_partial_confirmation_results_v1",
        status="COMPLETE_WITH_UNSCORABLE_INPUTS" if excluded else "COMPLETE",
        requested_questions=len(metadata), requested_original_questions=len(metadata),
        valid_questions=len(valid_metadata), excluded_questions=len(excluded), original_questions=len(valid_metadata),
        unscorable_by_task=dict(Counter(row["task"] for row in excluded)),
        unscorable_by_subtask=dict(Counter(row["subtask"] for row in excluded)),
        unscorable_by_reason=dict(Counter(row["reason"] for row in excluded)),
        valid_questions_by_task=dict(Counter(row["task"] for row in valid_metadata)),
        generation_seeds=[42, 137, 2024], unique_aggregate_keys=len(context["receipts"]),
        scored_aggregate_keys=len(sparse), unique_member_keys=len(context["members"]),
        execution_gate=context["gate"], results=statistics, physical_cost_totals=totals,
        selected_policy_service_costs=service, distinct_gold_accounts=len(gold_keys),
        original_source_cached_member_records=sum(row["reused_historical_source"] for row in costs),
        reference_provider_stats=provider.stats(), previous_committed_scores_reused=scorer._reused_previous,
        new_deterministic_scores=ledger["new_deterministic_scores"],
        reference_only_choice_body_mappings=len(mappings),
        reference_choice_mapping_by_subtask=dict(Counter(row["subtask"] for row in mappings)),
        prediction_parser_unchanged=True, scoring_model_calls=0, sealed_reference_values_decoded=0,
        missing_rewards_filled_with_zero=False,
        reward_array_scope="Only valid questions; requested3000 mask and identities retained separately",
        statistical_reporting="Point estimates, paired t p-values and Holm only; no CI or bootstrap",
        source_sha256=memo_api.file_sha(__file__), original_bulk_source_sha256=resume.ORIGINAL_BULK_SHA)
    arrays = dict(rewards=rewards.astype(np.int8), policy_ids=np.asarray(final.POLICIES, dtype="U"),
        question_ids=np.asarray([r["question_id"] for r in valid_metadata], dtype="U"),
        canonical_hashes=np.asarray([r["canonical_question_sha256"] for r in valid_metadata], dtype="U"),
        task_labels=np.asarray([r["task"] for r in valid_metadata], dtype="U"),
        subtask_labels=np.asarray([r["subtask"] for r in valid_metadata], dtype="U"),
        source_frames=np.asarray([r["source_frame"] for r in valid_metadata], dtype="U"),
        frozen_policy_actions=actions[:, valid], valid_question_mask=valid,
        requested_question_ids=np.asarray([r["question_id"] for r in metadata], dtype="U"),
        requested_canonical_hashes=np.asarray([r["canonical_question_sha256"] for r in metadata], dtype="U"),
        requested_task_labels=np.asarray([r["task"] for r in metadata], dtype="U"),
        requested_subtask_labels=np.asarray([r["subtask"] for r in metadata], dtype="U"))
    final.write_npz_once(runtime_root/"FINAL_POLICY_REWARDS.npz", arrays)
    for name, value in (("UNSCORABLE_CASES_PRIVATE.json", excluded),
                        ("REFERENCE_INTERFACE_MAPPINGS_PRIVATE.json", mappings),
                        ("SPARSE_SCORED_ROWS.json", sparse), ("SCORING_LEDGER.json", ledger),
                        ("PHYSICAL_COST_LEDGER.json", dict(rows=costs, totals=totals)),
                        ("REFERENCE_ACCESS_EVENTS.json", provider.audit_events()),
                        ("FINAL_RESULTS_PRIVATE.json", result)):
        final.write_new_json(runtime_root/name, value)
    final.write_new_json(Path(manifest["public_output_root"])/"E18_FINAL_RESULTS.json", result)
    return result


def main():
    import e18_final_scoring as final
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "completed-audit", "archive-directory", "historical-zip", "previous-scores",
                 "runtime-root", "public-root", "performance-receipt"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--quarantine-subtask", action="append", default=[])
    parser.add_argument("--execute-scoring", action="store_true", required=True)
    args = parser.parse_args()
    fast.require(memo_api.file_sha(args.manifest) == args.manifest_sha256, "Original manifest differs")
    fast.require(not args.performance_receipt.exists(), "Performance receipt must be new")
    previous = resume.read_previous_scores(args.previous_scores)
    manifest = final.worker.read_json(args.manifest)
    manifest["public_output_root"] = str(args.public_root)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    memo = memo_api.ScopeMemo(final, manifest)
    memo._pin_ref(dict(path=str(args.manifest), sha256=args.manifest_sha256))
    started, status, error_type = time.perf_counter(), "FAILED", None
    try:
        with memo:
            context = fast.context_from_completed_audit(final, manifest,
                dict(path=str(args.completed_audit), sha256=fast.FINAL_AUDIT_SHA),
                args.archive_directory, historical_zip=args.historical_zip,
                progress=lambda row: print(json.dumps(dict(event="AUDITED_RECEIPTS_LOADED", **row)), flush=True))
            context["gate"].update(reporting_scope="EVALUABLE_QUESTIONS_WITH_EXPLICIT_MISSING_MASK",
                                   original_manifest_file_sha256=args.manifest_sha256,
                                   input_reference_exclusions_do_not_remove_executed_cost=True,
                                   quarantine_subtasks=args.quarantine_subtask)
            result = finish(final, manifest, context, previous, args.runtime_root,
                            quarantine_subtasks=args.quarantine_subtask)
            status = result["status"]
            print(json.dumps(dict(status=status, valid_questions=result["valid_questions"],
                                  excluded_questions=result["excluded_questions"],
                                  quality=result["results"]["quality"])), flush=True)
    except BaseException as error:
        status, error_type = "FAILED", type(error).__name__
        raise
    finally:
        args.performance_receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.performance_receipt.open("x", encoding="utf-8") as stream:
            json.dump(dict(schema="e18_evaluable_bulk_performance_v1", status=status, error_type=error_type,
                           wall_seconds=time.perf_counter()-started, source_sha256=memo_api.file_sha(__file__),
                           previous_successful_scores_available=len(previous), scope_memo=memo.stats()),
                      stream, ensure_ascii=False, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
