"""Small synthetic P2 scope, permissions and probability-evaluation checks."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

import numpy as np

import blind_evaluation as blind
import p2_execution_adapter as execution
from p2_scope_freeze import freeze_scope


def must_reject(function):
    try:
        function()
    except (ValueError, PermissionError, TypeError):
        return True
    return False


def main():
    questions = [dict(question_id=f"synthetic-question-{index}",
        question=f"Synthetic scope qualification question {index}.",
        task="gsm8k" if index < 2 else "bbh", subtask="" if index < 2 else "synthetic")
        for index in range(4)]
    for question in questions:
        question["canonical_question_sha256"] = execution.canonical_question(question["question"])
    scores = np.full((2, 4, 20), -2.0)
    scores[0, :, :2] = 1
    scores[1, :, 1] = 2
    packet = dict(question_ids=np.array([row["question_id"] for row in questions]),
        canonical_hashes=np.array([row["canonical_question_sha256"] for row in questions]),
        task_labels=np.array([row["task"] for row in questions]),
        subtask_labels=np.array([row["subtask"] for row in questions]),
        scores=scores, probabilities=blind.decoder(scores, blind.TAU),
        raw_exact_probabilities=blind.decoder(scores, 0))
    support, aggregate_keys, member_keys = blind.required_union(packet["question_ids"], packet["probabilities"], packet["raw_exact_probabilities"])
    binding = {name: execution.canonical_digest(name) for name in execution.REQUIRED_BINDINGS}
    binding["target_ids_sha256"] = execution.canonical_digest(
        [[row["question_id"], row["canonical_question_sha256"]] for row in questions])
    binding["final_model_sha256"] = blind.MODEL_SHA256
    scope = execution.build_scope_contract(questions, packet["probabilities"].tolist(), binding,
                                           sorted(member_keys), sorted(aggregate_keys))
    checks = {}
    qualification = blind.qualify_scope(scope, packet, expected_questions=None)
    checks["two_probability_heads_exact_support_union"] = qualification["required_aggregate_calls"] == 24
    checks["single_member_replacement_reuses_two_members"] = qualification["required_member_calls"] == 48
    reuse_scope = execution.build_scope_contract(questions, packet["probabilities"].tolist(), binding, [], [])
    checks["cache_execution_subset_does_not_shrink_required_support"] = (
        reuse_scope["required_member_keys"] == scope["required_member_keys"]
        and reuse_scope["required_aggregate_keys"] == scope["required_aggregate_keys"]
        and reuse_scope["execution_limits"] == {"member_generation": 0, "aggregation": 0})
    checks["out_of_support_aggregation_rejected"] = must_reject(lambda: execution.build_scope_contract(
        questions, packet["probabilities"].tolist(), binding, sorted(member_keys),
        sorted(aggregate_keys) + [(questions[0]["question_id"],42,19)]))
    checks["duplicate_execution_ownership_rejected"] = must_reject(lambda: execution.build_scope_contract(
        questions, packet["probabilities"].tolist(), binding, sorted(member_keys)+[next(iter(member_keys))],sorted(aggregate_keys)))
    changed = deepcopy(scope)
    changed["required_aggregate_keys"] = changed["required_aggregate_keys"][:-1]
    checks["missing_support_action_rejected"] = must_reject(lambda: blind.qualify_scope(changed,packet,expected_questions=None))
    changed = deepcopy(scope)
    changed["bindings"]["final_model_sha256"] = "0"*64
    checks["different_model_identity_rejected"] = must_reject(lambda: blind.qualify_scope(changed,packet,expected_questions=None))
    changed = deepcopy(scope)
    changed["frozen_policy_probabilities"][0][0][0] = .4
    changed["frozen_policy_probabilities"][0][0][1] = .6
    checks["changed_probabilities_with_same_support_rejected"] = must_reject(lambda: blind.qualify_scope(changed,packet,expected_questions=None))
    frozen = freeze_scope(scope)
    before = execution.canonical_digest(frozen)
    checks["immutable_top_level"] = must_reject(lambda: frozen.update({"role":"reference"}))
    checks["immutable_nested_probabilities"] = must_reject(lambda: frozen["frozen_policy_probabilities"][0][0].append(0))
    scope["frozen_policy_probabilities"][0][0][0] = .1
    checks["frozen_snapshot_is_detached_from_mutable_input"] = execution.canonical_digest(frozen) == before
    scope = deepcopy(dict(frozen))
    # Exercise the actual hash-bound approval validator with synthetic metadata.
    with tempfile.TemporaryDirectory(prefix="p2-permissions-") as folder:
        path = Path(folder)/"APPROVAL.json"
        approval = dict(schema="p2_scope_approval_v1", status="APPROVED", cohort=execution.COHORT,
            software_test_only=True, operations=["member_generation"], bindings=binding,
            scope_sha256_by_role={"execution":execution.canonical_digest(scope)})
        path.write_text(json.dumps(approval),encoding="utf-8")
        reference = dict(path=str(path),sha256=execution.sha256(path))
        checks["matching_explicit_approval_accepted"] = execution.validate_scope_approval(
            reference,scope,"member_generation",software_test_only=True)["status"] == "APPROVED"
        checks["unapproved_reference_access_rejected"] = must_reject(lambda: execution.validate_scope_approval(
            reference,scope,"read_reference",software_test_only=True))
        checks["synthetic_approval_cannot_authorize_production"] = must_reject(lambda: execution.validate_scope_approval(
            reference,scope,"member_generation",software_test_only=False))
        changed = json.loads(json.dumps(scope))
        changed["execution_limits"]["aggregation"] += 1
        checks["scope_mutation_breaks_approval_digest"] = must_reject(lambda: execution.validate_scope_approval(
            reference,changed,"member_generation",software_test_only=True))
        contaminated = Path(folder)/"CONTAMINATED.npz"
        np.savez_compressed(contaminated, **packet, reward=np.zeros(1))
        checks["reward_archive_rejected_before_evaluation"] = must_reject(lambda: blind.load_predictions(contaminated))
    rewards = [(0,1),(1,0),(0,1),(1,1)]
    rows = [dict(question_id=row["question_id"],generation_seed=seed,combination_index=action,score=rewards[index][action])
            for index,row in enumerate(questions) for seed in blind.SEEDS for action in (0,1)]
    report, arrays = blind.project_scores(packet,rows)
    checks["randomized_policy_expected_accuracy"] = abs(report["primary"]["difference_pp"]-12.5)<1e-12
    checks["seed_repeats_are_not_independent_questions"] = report["primary"]["original_questions"] == 4
    checks["fractional_corrections_match_net_utility"] = abs(report["expected_net_corrections"]-1.5)<1e-12
    checks["missing_output_rejected"] = must_reject(lambda: blind.project_scores(packet,rows[:-1]))
    checks["duplicate_score_rejected"] = must_reject(lambda: blind.project_scores(packet,rows+[rows[0]]))
    checks["no_ci_or_bootstrap"] = report["bootstrap_repetitions"] == 0 and not report["confidence_intervals_computed"]
    if not all(checks.values()):
        raise AssertionError(checks)
    output = dict(schema="p2_blind_and_scope_synthetic_qualification_v1",status="PASS",checks=checks,
        synthetic_questions=4,new_real_question_inputs_read=0,new_gold_values_read=0,new_llm_calls=0,
        execution_adapter_sha256=execution.sha256(Path(execution.__file__)),
        blind_evaluation_sha256=blind.file_sha(Path(blind.__file__)),
        bootstrap_repetitions=0,confidence_intervals_computed=False)
    destination = Path(__file__).with_name("P2_BLIND_AND_SCOPE_QUALIFICATION.json")
    destination.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","checks":len(checks),"real_llm_calls":0},ensure_ascii=False))


if __name__ == "__main__":
    main()
