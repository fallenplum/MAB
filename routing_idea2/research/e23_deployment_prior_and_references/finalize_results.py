"""Score settled E23 additions and complete the frozen full-cohort references."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

import analyze_references as prior
import supplement_adapter as supplement

sys.path.insert(0, str(prior.P2))
import score_paid_outputs as scoring
from collect_native_exports import load_auditor


EXPECTED_COUNTS = {"member_generation": 1770, "aggregation": 4263}
EXPECTED_NAMES = {
    "B_uniform", "C_uniform", "B_prior", "C_prior", "B_raw", "C_raw",
    "GlobalFixed", "Ridge", "Uniform",
    "Matched2026111801", "Matched2026111802", "Matched2026111803",
}


def json_rows(path):
    with Path(path).open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n") or not raw.strip():
                raise ValueError("Native and score JSONL must contain complete nonempty rows")
            yield raw, json.loads(raw)


def frozen_policies(root):
    folder = root / "private"
    freeze = prior.read(folder / "FREEZE.json")
    if (freeze["source_binding"] != prior.binding()
            or prior.digest(folder / "POLICIES_FROZEN.npz") != freeze["policy_sha256"]
            or prior.digest(folder / "REGIONS_FROZEN.npz") != freeze["regions_sha256"]):
        raise ValueError("E23 policies or training-only source changed")
    packet = prior.archive(folder / "POLICIES_FROZEN.npz")
    regions = prior.archive(folder / "REGIONS_FROZEN.npz")
    names = packet["names"].astype(str).tolist()
    probabilities = packet["probabilities"]
    ids = packet["question_ids"].astype(str)
    tasks = packet["task_labels"].astype(str)
    if (len(ids) != 2400 or len(set(ids)) != 2400 or len(names) != 12
            or set(names) != EXPECTED_NAMES or probabilities.shape != (12, 2400, 20)
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=-1), 1, rtol=0, atol=1e-12)
            or Counter(tasks) != {"gsm8k": 1200, "bbh": 1200}):
        raise ValueError("The frozen full2400 twelve-policy cohort differs")
    if any(regions[key].shape != (2400,) or regions[key].dtype != np.bool_
           for key in ("b_unique", "strict_switched")):
        raise ValueError("Frozen input-only evaluation regions differ")
    policies = dict(zip(names, probabilities))
    if not np.array_equal(policies["C_prior"], policies["C_uniform"]):
        raise ValueError("The unique frozen C policy changed")
    return freeze, packet, regions, policies


def required_keys(ids, policies):
    support = np.any(np.array(list(policies.values())) > 0, axis=0)
    return {(question, int(seed), int(action)) for question, row in zip(ids, support)
            for seed in prior.SEEDS for action in np.flatnonzero(row)}


def check_scope(args, policy_freeze, packet, policies):
    scope = prior.read(args.scope)
    canonical = supplement.canonical_digest(scope)
    approval_ref = dict(path=str(args.approval), sha256=prior.digest(args.approval))
    approval = supplement.validate_scope_approval(approval_ref, scope, "aggregation")
    supplement.validate_scope_approval(approval_ref, scope, "member_generation")
    parent = prior.read(prior.P2_PRIVATE / "runtime_launch_v2/inputs/EXECUTION_SCOPE_RUNTIME.json")
    if (supplement.canonical_digest(parent) != supplement.PARENT_SCOPE_SHA
            or scope["bindings"]["frozen_actions_sha256"] != policy_freeze["policy_sha256"]
            or scope["bindings"]["protocol_sha256"] != prior.digest(prior.CODE / "PROTOCOL.json")):
        raise ValueError("Supplementary scope has another parent, protocol or policy identity")
    expected_missing = prior.read(args.root / "private/MISSING_KEYS.json")["union"]
    expected = {"member_generation": set(map(tuple, expected_missing["missing_member_keys"])),
                "aggregation": set(map(tuple, expected_missing["missing_aggregate_keys"]))}
    for stage, field in (("member_generation", "member_execution_keys"),
                         ("aggregation", "aggregate_execution_keys")):
        rows = list(map(tuple, scope[field]))
        if (len(rows) != EXPECTED_COUNTS[stage] or len(set(rows)) != len(rows)
                or set(rows) != expected[stage] or approval["execution_limits"][stage] != len(rows)):
            raise ValueError("The exact approved missing-call union differs")
    identities = {row["question_id"]: row for row in parent["question_keys"]}
    questions = scope["question_keys"]
    if (len(questions) != 840 or len({row["question_id"] for row in questions}) != 840
            or any(identities.get(row["question_id"]) != row for row in questions)):
        raise ValueError("Supplementary inputs differ from the original P2 input-only metadata")
    positions = {question: index for index, question in enumerate(packet["question_ids"].astype(str))}
    if set(scope["frozen_policy_probabilities"]) != set(policies):
        raise ValueError("The executed supplementary strategy set differs")
    selected = [positions[row["question_id"]] for row in questions]
    for name, values in scope["frozen_policy_probabilities"].items():
        if not np.array_equal(np.asarray(values), policies[name][selected]):
            raise ValueError("Supplementary policy probabilities changed before scoring")
    parent_keys = set(map(tuple, parent["required_aggregate_keys"]))
    full = required_keys(packet["question_ids"].astype(str), policies)
    if expected["aggregation"] != full - parent_keys or expected["aggregation"] & parent_keys:
        raise ValueError("Executed aggregate additions do not complete the frozen strategy union")
    return scope, parent, approval_ref, identities, expected


def check_export(args, scope, approval_ref):
    frozen = prior.read(args.outputs_frozen)
    if (frozen.get("schema") != "e23_supplement_outputs_frozen_v1"
            or frozen.get("status") != "ALL_REQUIRED_SUPPLEMENT_OUTPUTS_FROZEN"
            or frozen.get("cohort") != supplement.COHORT
            or frozen.get("scope_sha256") != prior.digest(args.scope)
            or frozen.get("scope_canonical_sha256") != supplement.canonical_digest(scope)
            or frozen.get("approval_sha256") != approval_ref["sha256"]
            or frozen.get("scope_bindings") != scope["bindings"]
            or frozen.get("completed_member_calls") != 1770
            or frozen.get("completed_aggregate_calls") != 4263
            or frozen.get("unknown_or_failed_attempts") != 0
            or frozen.get("new_gold_values_read") != 0):
        raise PermissionError("All6033 settled supplementary outputs must freeze before scoring")
    source_files = {"member_generation": args.member_records, "aggregation": args.aggregate_records}
    for stage, path in source_files.items():
        binding = frozen["records"][stage]
        if (binding.get("records") != EXPECTED_COUNTS[stage]
                or binding.get("bytes") != path.stat().st_size
                or binding.get("sha256") != prior.digest(path)):
            raise ValueError("Local supplementary receipt bytes differ from the closed export")
    return frozen


def member_profiles():
    audit, _ = load_auditor()
    profiles = {}
    for row in prior.read(prior.ASSETS / "configs/pool_v1.json")["models"]:
        profile, fingerprint = audit.model_fingerprint(prior.read(prior.ASSETS / row["config"]))
        if fingerprint != row["fingerprint"]:
            raise ValueError("Original member ModelSpec changed")
        profiles[row["index"]] = profile, fingerprint, row["alias"]
    if set(profiles) != set(range(6)):
        raise ValueError("Exactly six member profile identities are required")
    return profiles


def verify_receipts(args, scope, approval_ref, identities, expected, native_api):
    checked, attempts, costs = {}, set(), {}
    profiles = member_profiles()
    canonical_scope = supplement.canonical_digest(scope)
    approved_root = prior.read(args.approval)["approved_output_root"]
    destination = args.output_root / "VERIFIED_SUPPLEMENT_AGGREGATIONS.jsonl"
    with destination.open("x", encoding="utf-8", newline="\n") as target:
        for stage, path in (("member_generation", args.member_records), ("aggregation", args.aggregate_records)):
            seen, usage = set(), dict(prompt_tokens=0, completion_tokens=0, wall_seconds=0.0)
            index_field = "member_index" if stage == "member_generation" else "combination_index"
            for _, receipt in json_rows(path):
                task = receipt.get("task_key", {})
                key = (task.get("question_id"), task.get("generation_seed"), task.get(index_field))
                if (receipt.get("schema") != "p2_execution_task_receipt_v1"
                        or receipt.get("cohort") != supplement.COHORT or receipt.get("stage") != stage
                        or receipt.get("status") != "SUCCESS" or receipt.get("software_test_only") is not False
                        or receipt.get("scope_sha256") != canonical_scope
                        or receipt.get("approval_sha256") != approval_ref["sha256"]
                        or receipt.get("approved_output_root") != approved_root
                        or receipt.get("bindings") != scope["bindings"]
                        or receipt.get("from_cache") is not False
                        or type(key[1]) is not int or type(key[2]) is not int
                        or key not in expected[stage] or key in seen):
                    raise ValueError("Supplementary native receipt escaped its approved identity")
                metadata = identities[key[0]]
                native = receipt.get("native_record", {})
                result = native.get("result", {})
                native_sha = supplement.canonical_digest({key: value for key, value in native.items()
                                                          if key != "record_sha256"})
                attempt = native.get("attempt_id")
                if (not isinstance(attempt, str) or not attempt or attempt in attempts
                        or native.get("record_sha256") != native_sha
                        or receipt.get("native_record_sha256") != native_sha
                        or native.get("status") != "success" or native.get("cost_status") != "recorded_actual_usage"
                        or native.get("job_id") != receipt.get("native_job_id")
                        or receipt.get("physical_attempt_id") != attempt
                        or native.get("question_id") != key[0]
                        or task.get("canonical_id") != metadata["canonical_question_sha256"]
                        or native.get("task") != metadata["task"]
                        or (native.get("subtask") or "") != (metadata.get("subtask") or "")):
                    raise ValueError("Native record is unsettled, repeated or has another input identity")
                for field in ("prompt_tokens", "completion_tokens"):
                    if type(result.get(field)) is not int or result[field] < 0:
                        raise ValueError("Native supplementary actual token usage is absent")
                    usage[field] += result[field]
                wall = result.get("wall_time_s")
                if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
                    raise ValueError("Native supplementary runtime is absent")
                usage["wall_seconds"] += wall
                if stage == "aggregation":
                    legacy = dict(receipt, schema="e18_execution_task_receipt_v2", cohort="e18_confirm")
                    verified = native_api.verified_aggregation_source(legacy)
                    verified["supplement_execution_receipt_sha256"] = supplement.canonical_digest(receipt)
                    target.write(json.dumps(verified, ensure_ascii=False) + "\n")
                else:
                    profile, fingerprint, alias = profiles[key[2]]
                    if (native.get("model_fingerprint") != fingerprint or native.get("model_alias") != alias
                            or native.get("source_question_sha256") != metadata["question_sha256"]
                            or native.get("seed") != key[1] or result.get("seed") != key[1]
                            or result.get("model_id") != profile["model_id"]
                            or result.get("model_revision") != profile["revision"]
                            or result.get("backend") != profile["backend"]
                            or result.get("model_identity_verified") is not True
                            or native.get("sample_index") != 0 or native.get("sampling_control") is not False
                            or native.get("prompt_version") != "independent_user_only_brief_final_v2"
                            or not isinstance(result.get("text"), str) or not result["text"].strip()
                            or result["prompt_tokens"] > profile["max_input_tokens"]
                            or result["completion_tokens"] > profile["max_new_tokens"]):
                        raise ValueError("Supplementary member changed the original inference contract")
                seen.add(key)
                attempts.add(attempt)
            if seen != expected[stage]:
                raise ValueError("All exact supplementary outputs must be complete before scoring")
            checked[stage] = len(seen)
            costs[stage] = dict(calls=len(seen), **usage,
                                total_tokens=usage["prompt_tokens"] + usage["completion_tokens"])
    observed = {key: sum(row[key] for row in costs.values()) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    frozen = prior.read(args.outputs_frozen)
    if observed != frozen["actual_tokens"]:
        raise ValueError("Native actual token sums differ from the export closure")
    return destination, dict(status="ALL6033_SUPPLEMENT_NATIVE_OUTPUTS_VERIFIED", completed=checked,
                             by_stage=costs, actual_tokens=observed, native_attempts=len(attempts),
                             unknown_or_failed_attempts=0, reference_values_decoded=0)


def score_new(args, parent_scope, verified_path, reference_api, projection_api, scorer):
    plan_path = prior.P2_PRIVATE / "P2_REFERENCE_LOCATORS_FROZEN.json"
    if prior.digest(plan_path) != scoring.REFERENCE_PLAN_SHA:
        raise ValueError("Original P2 reference locators changed")
    parent_approval_path = prior.P2_PRIVATE / "runtime_launch_v2/inputs/AUTHORIZATION_v2.json"
    parent_approval = dict(path=str(parent_approval_path), sha256=prior.digest(parent_approval_path))
    for operation in ("read_reference", "score_outputs"):
        scoring.execution.validate_scope_approval(parent_approval, parent_scope, operation)
    plan = prior.read(plan_path)
    start = time.process_time()
    scores = []
    original_scoring = prior.read(prior.P2_SCORED / "REFERENCE_AND_SCORING_COSTS.json")
    if (original_scoring.get("unique_gold_accounts") != 2400
            or original_scoring.get("reference_values_decoded") != 2400
            or original_scoring.get("sealed_answer_values_decoded") != 0):
        raise ValueError("Complete original P2 reference accounting is required for legal gold reuse")
    existing_accounts = {(row["canonical_question_sha256"], scoring.REFERENCE_VERSION)
                         for row in parent_scope["question_keys"]}
    score_path = args.output_root / "SUPPLEMENT_PAID_SUPPORT_SCORES.jsonl"
    with (args.output_root / "REFERENCE_REUSE_EVENTS.jsonl").open("x", encoding="utf-8") as events, \
            score_path.open("x", encoding="utf-8", newline="\n") as output:
        provider = scoring.ApprovedReferences(plan, parent_scope, reference_api, projection_api, events)
        provider.accounts.update(existing_accounts)
        normalized = {}
        for _, native in json_rows(verified_path):
            question = native["question_id"]
            if question not in normalized:
                normalized[question] = scoring.canonical_reference(provider.get(question), scorer)
            reference = normalized[question]
            scored = scorer.score_answer(reference["task"], native["prediction"], reference["reference"],
                                         subtask=reference.get("subtask"), choices=reference.get("choices"))
            if scored.scorer_version != scoring.REFERENCE_VERSION or scored.score not in (0, 1):
                raise ValueError("Original frozen scorer returned an unexpected version or value")
            row = dict(question_id=question, generation_seed=native["generation_seed"],
                       combination_index=native["combination_index"], score=float(scored.score),
                       native_record_sha256=native["aggregation_record_sha256"], parsed=bool(scored.parsed),
                       reason=scored.reason, reference_version=scoring.REFERENCE_VERSION)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            scores.append(row)
    if len(scores) != 4263 or len(normalized) != 840:
        raise ValueError("Supplementary scoring must preserve every approved output and question")
    return scores, dict(existing_question_references_reused=len(normalized),
                        reference_values_decoded=provider.references_decoded,
                        additional_gold_accounts_purchased=len(provider.accounts-existing_accounts),
                        sealed_answer_values_decoded=0,
                        deterministic_scorer_calls=len(scores), scoring_cpu_seconds=time.process_time()-start,
                        new_llm_calls=0)


def merge_scores(args, packet, policies, parent_scope, expected, scores):
    ids = packet["question_ids"].astype(str)
    positions = {question: index for index, question in enumerate(ids)}
    old_path = prior.P2_SCORED / "PAID_SUPPORT_SCORES.jsonl"
    original_report = prior.read(prior.CODE / "RESULTS.json")
    if (original_report["policy_sha256"] != prior.digest(args.root / "private/POLICIES_FROZEN.npz")
            or original_report["existing_scores_sha256"] != prior.digest(old_path)):
        raise ValueError("Original P2 scores changed after the frozen E23 cache audit")
    old_required = set(map(tuple, parent_scope["required_aggregate_keys"]))
    values = np.full((3, len(ids), 20), np.nan)
    seen = set()
    for rows, required in (((row for _, row in json_rows(old_path)), old_required),
                           (iter(scores), expected["aggregation"])):
        group_seen = set()
        for row in rows:
            key = (row["question_id"], row["generation_seed"], row["combination_index"])
            if (key not in required or key in seen or row.get("score") not in (0, 1)
                    or row.get("reference_version") != scoring.REFERENCE_VERSION):
                raise ValueError("A score is duplicated, out of support, or uses another scorer")
            group_seen.add(key)
            seen.add(key)
            values[prior.SEEDS.index(key[1]), positions[key[0]], key[2]] = row["score"]
        if group_seen != required:
            raise ValueError("Existing or supplemental scoring did not preserve full executed coverage")
    for probability in policies.values():
        prior.utility(probability, values)
    if seen != required_keys(ids, policies):
        raise ValueError("Scored outputs do not exactly cover the frozen twelve-policy support union")
    return values


def summarize(packet, regions, policies, values):
    tasks, subtasks = packet["task_labels"].astype(str), packet["subtask_labels"].astype(str)
    rows, utilities = {}, {}
    for name, probabilities in policies.items():
        utility = prior.utility(probabilities, values)
        utilities[name] = utility
        rows[name] = dict(status="FULL_COHORT_SCORED", questions=len(tasks),
                          generation_events=3*len(tasks), accuracy_pct=float(50*sum(
                              utility[:, tasks == task].mean() for task in ("gsm8k", "bbh"))),
                          by_task={task: float(100*utility[:, tasks == task].mean()) for task in ("gsm8k", "bbh")},
                          by_seed={str(seed): float(50*sum(utility[index, tasks == task].mean()
                                   for task in ("gsm8k", "bbh"))) for index, seed in enumerate(prior.SEEDS)},
                          by_bbh_subtask={key: float(100*utility[:, (tasks == "bbh") & (subtasks == key)].mean())
                                          for key in sorted(set(subtasks[tasks == "bbh"]))})
    masks = dict(original_B_unique=regions["b_unique"], original_B_near_tied=~regions["b_unique"],
                 original_strict_switched=regions["strict_switched"])
    comparisons = {}
    for baseline, candidate in (("B_uniform", "C_uniform"), ("B_uniform", "B_prior"),
                                ("B_prior", "C_prior"), ("B_raw", "C_raw"),
                                ("GlobalFixed", "C_uniform"), ("Ridge", "C_uniform"),
                                ("Uniform", "C_uniform"), ("Matched2026111801", "C_uniform"),
                                ("Matched2026111802", "C_uniform"), ("Matched2026111803", "C_uniform")):
        comparisons[f"{candidate}_minus_{baseline}"] = prior.compare(
            policies[baseline], policies[candidate], values, tasks, subtasks, masks)
    matched = np.mean([policies[f"Matched{seed}"] for seed in (2026111801, 2026111802, 2026111803)], axis=0)
    comparisons["C_uniform_minus_MatchedMean"] = prior.compare(
        matched, policies["C_uniform"], values, tasks, subtasks, masks)
    original = comparisons["C_uniform_minus_B_uniform"]["difference_pp"]
    deployment = comparisons["B_prior_minus_B_uniform"]["difference_pp"]
    remainder = comparisons["C_prior_minus_B_prior"]["difference_pp"]
    if abs(original-deployment-remainder) > 1e-10:
        raise ValueError("The frozen deployment decomposition changed")
    return rows, comparisons, utilities, dict(original_gap_pp=original, B_deployment_gain_pp=deployment,
                                             remaining_gap_pp=remainder)


def run(args):
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    policy_freeze, packet, regions, policies = frozen_policies(args.root)
    scope, parent, approval_ref, identities, expected = check_scope(args, policy_freeze, packet, policies)
    export = check_export(args, scope, approval_ref)
    args.output_root.mkdir(parents=True, exist_ok=False)
    reference_api, projection_api, native_api, scorer = scoring.load_frozen_apis()
    verified, native_audit = verify_receipts(args, scope, approval_ref, identities, expected, native_api)
    prior.write(args.output_root / "NATIVE_SUPPLEMENT_VERIFIED_BEFORE_SCORING.json", native_audit)
    scores, scoring_cost = score_new(args, parent, verified, reference_api, projection_api, scorer)
    values = merge_scores(args, packet, policies, parent, expected, scores)
    strategies, comparisons, utilities, gaps = summarize(packet, regions, policies, values)
    original_result = prior.read(prior.CODE / "RESULTS.json")
    if any(abs(original_result[key]-value) > 1e-10 for key, value in gaps.items()):
        raise ValueError("Existing paid-support prior results changed during supplement finalization")
    with np.load(prior.P2_SCORED / "P2_POLICY_UTILITIES_PRIVATE.npz", allow_pickle=False) as original:
        if (not original["valid_mask"].all() or not np.allclose(original["utilities"],
                np.array([utilities["B_uniform"], utilities["C_uniform"]]), rtol=0, atol=1e-12)):
            raise ValueError("Original P2 frozen policy projection changed")
    parent_ledger_path = prior.P2 / "P2_FINAL_EXECUTION_LEDGER.json"
    parent_ledger = prior.read(parent_ledger_path)
    if (parent_ledger.get("scope_canonical_sha256") != supplement.PARENT_SCOPE_SHA
            or parent_ledger.get("status") != "COMPLETE_SUCCESS_AND_NATIVE_COST_CLOSED"
            or parent_ledger.get("failure_closure_pending") is not False
            or type(parent_ledger.get("all_attempt_total_tokens")) is not int):
        raise ValueError("Closed original P2 cost evidence is required for combined accounting")
    combined_cost = dict(
        original_p2_ledger_sha256=prior.digest(parent_ledger_path),
        original_p2_member_calls=len(parent["required_member_keys"]),
        original_p2_aggregate_calls=len(parent["required_aggregate_keys"]),
        original_p2_tokens=parent_ledger["all_attempt_total_tokens"],
        supplementary_member_calls=1770, supplementary_aggregate_calls=4263,
        supplementary_tokens=native_audit["actual_tokens"]["total_tokens"],
        combined_member_calls=len(parent["required_member_keys"])+1770,
        combined_aggregate_calls=len(parent["required_aggregate_keys"])+4263,
        combined_tokens=parent_ledger["all_attempt_total_tokens"]+native_audit["actual_tokens"]["total_tokens"],
        new_gold_accounts_purchased=scoring_cost["additional_gold_accounts_purchased"],
        reused_aggregate_calls=len(parent["required_aggregate_keys"]),
        distinct_original_question_gold_accounts=2400,
        native_token_costs_closed=True,
    )
    with (args.output_root / "FINAL_POLICY_UTILITIES_PRIVATE.npz").open("xb") as target:
        np.savez_compressed(target, names=np.array(list(utilities)), utilities=np.array(list(utilities.values())),
                            task_labels=packet["task_labels"], subtask_labels=packet["subtask_labels"])
    result = dict(schema="mab_e23_final_reference_results_v1", status="COMPLETE_FULL2400_REFERENCE_EVALUATION",
                  source_binding=prior.binding(), finalizer_sha256=prior.digest(Path(__file__)),
                  policy_sha256=policy_freeze["policy_sha256"], scope_sha256=prior.digest(args.scope),
                  scope_canonical_sha256=supplement.canonical_digest(scope),
                  approval_sha256=approval_ref["sha256"], outputs_frozen_sha256=prior.digest(args.outputs_frozen),
                  parent_scope_canonical_sha256=supplement.PARENT_SCOPE_SHA,
                  original_scores_sha256=original_result["existing_scores_sha256"],
                  new_scores_sha256=prior.digest(args.output_root / "SUPPLEMENT_PAID_SUPPORT_SCORES.jsonl"),
                  questions=2400, original_generation_events_per_policy=7200,
                  task_weights={"gsm8k": 0.5, "bbh": 0.5}, baseline_strategies=strategies,
                  paired_comparisons=comparisons, **gaps,
                  train_action_success_counts=policy_freeze["train_action_success_counts"],
                  training_global_best_actions=policy_freeze["training_global_best_actions"],
                  matching=policy_freeze["matching"], native_supplement_cost=native_audit,
                  combined_execution_cost=combined_cost,
                  reference_reuse_and_scoring_cost=scoring_cost,
                  unique_scored_aggregate_outputs=len(parent["required_aggregate_keys"])+len(scores),
                  original_paid_aggregate_outputs=len(parent["required_aggregate_keys"]),
                  supplementary_paid_aggregate_outputs=len(scores),
                  mixed_control_interpretation="MatchedMean uses the equal mixture of all three frozen action-probability policies; correction counts are coupling expectations.",
                  evaluation_scope="Supplemental analysis on already-exposed P2 new2400 outcomes, all reference policies frozen before missing-call execution",
                  new_llm_calls=6033, new_prediction_head_fits=4,
                  new_llm_calls_during_finalization=0, new_prediction_head_fits_during_finalization=0,
                  confidence_intervals_computed=False,
                  bootstrap_repetitions=0, finalization_cpu_seconds=time.process_time()-started_cpu,
                  finalization_wall_seconds=time.perf_counter()-started_wall,
                  exported_actual_tokens=export["actual_tokens"])
    prior.write(args.output, result)
    print(json.dumps(dict(status=result["status"], questions=2400, **gaps,
                           accuracies_pct={name: row["accuracy_pct"] for name, row in strategies.items()})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("root", "scope", "approval", "outputs-frozen", "member-records", "aggregate-records", "output-root", "output"):
        parser.add_argument("--"+flag, type=Path, required=True)
    run(parser.parse_args())
