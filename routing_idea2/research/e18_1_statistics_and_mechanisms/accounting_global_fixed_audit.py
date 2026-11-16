"""Read-only E18.1 accounting and historical GlobalFixed provenance audit.

Only legal train600 reward members are decoded for the fixed-action check.
No sealed source, evaluation gold, language model or archive payload is opened.
Existing actual-cost ledgers supply physical execution accounting.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(r"E:\语言模型组合路由")
RUNTIME = ROOT / ".e18_cpu_runtime/evaluable_bulk_v1"
STUDY = Path(__file__).resolve().parent
TEAMS = tuple(combinations(range(6), 3))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    final = read(RUNTIME / "FINAL_RESULTS_PRIVATE.json")
    ledger = read(RUNTIME / "PHYSICAL_COST_LEDGER.json")["rows"]
    excluded = read(RUNTIME / "UNSCORABLE_CASES_PRIVATE.json")
    audit = read(ROOT / "experiments/e18_confirmatory_execution/CLOSED_NATIVE_INDEPENDENT_AUDIT_FINAL.json")
    execution = read(ROOT / ".e18_cpu_runtime/metadata/CPU_READONLY_EXECUTION_MANIFEST.json")
    scope = read(execution["scope_ref"]["path"])
    question_ids = [row["question_id"] for row in scope["question_keys"]]
    actions = np.asarray(scope["frozen_policy_actions"], dtype=int)
    required_agg = set(map(tuple, scope["required_aggregate_keys"]))
    required_members = set(map(tuple, scope["required_member_keys"]))
    assert actions.shape == (4, 3000) and len(required_agg) == 30921 and len(required_members) == 47286
    assert len(ledger) == len({row["physical_attempt_id"] for row in ledger}) == 78207
    totals = {}
    for historical in (False, True):
        selected = [row for row in ledger if row["reused_historical_source"] is historical]
        totals["historical" if historical else "new"] = dict(
            physical_attempts=len(selected), by_stage=dict(Counter(row["stage"] for row in selected)),
            prompt_tokens=sum(row["prompt_tokens"] for row in selected),
            completion_tokens=sum(row["completion_tokens"] for row in selected),
            total_tokens=sum(row["prompt_tokens"]+row["completion_tokens"] for row in selected))
    assert totals["new"]["physical_attempts"] == 73490 and totals["new"]["total_tokens"] == 46587516
    assert totals["historical"]["physical_attempts"] == 4717 and totals["historical"]["total_tokens"] == 1944226
    assert totals["new"]["prompt_tokens"] == audit["actual_prompt_tokens"]
    assert totals["new"]["completion_tokens"] == audit["actual_completion_tokens"]
    actual_agg = {tuple(row["task_key"]) for row in ledger if row["stage"] == "aggregation"}
    actual_members = {tuple(row["task_key"]) for row in ledger if row["stage"] == "member_generation"}
    assert actual_agg == required_agg and actual_members == required_members
    excluded_keys, excluded_members, private_rows, public_cases = set(), set(), [], []
    for row in excluded:
        qid = row["question_id"]
        index = question_ids.index(qid)
        selected_actions = set(map(int, actions[:, index]))
        agg_keys = {(qid, seed, action) for seed in (42, 137, 2024) for action in selected_actions}
        member_set = set().union(*(set(TEAMS[action]) for action in selected_actions))
        member_keys = {(qid, seed, member) for seed in (42, 137, 2024) for member in member_set}
        assert agg_keys <= actual_agg and member_keys <= actual_members
        excluded_keys |= agg_keys
        excluded_members |= member_keys
        aggregate_costs = [item for item in ledger if item["stage"] == "aggregation" and tuple(item["task_key"]) in agg_keys]
        member_costs = [item for item in ledger if item["stage"] == "member_generation" and tuple(item["task_key"]) in member_keys]
        public_cases.append(dict(task=row["task"], subtask=row["subtask"], questions=1,
            policies=4, logical_policy_aggregates=12, distinct_combinations=len(selected_actions),
            generation_seeds=[42, 137, 2024], executed_aggregate_calls=len(agg_keys),
            aggregate_calls_per_seed={str(seed): sum(key[1] == seed for key in agg_keys) for seed in (42,137,2024)},
            unique_member_profiles=len(member_set), executed_member_calls=len(member_keys),
            aggregate_tokens=sum(item["prompt_tokens"]+item["completion_tokens"] for item in aggregate_costs),
            member_tokens=sum(item["prompt_tokens"]+item["completion_tokens"] for item in member_costs)))
        private_rows.append(dict(question_id=qid, canonical_id=row["canonical_id"], subtask=row["subtask"],
                                 policy_actions=actions[:, index].tolist(), distinct_actions=sorted(selected_actions),
                                 aggregate_keys=sorted(agg_keys), member_keys=sorted(member_keys)))
    assert len(excluded_keys) == 21 and len(required_agg)-len(excluded_keys) == final["scored_aggregate_keys"] == 30900
    with np.load(RUNTIME / "FINAL_POLICY_REWARDS.npz", allow_pickle=False) as packet:
        valid = packet["valid_question_mask"]
        requested_ids = packet["requested_question_ids"]
        assert valid.shape == (3000,) and valid.sum() == 2998
        assert set(requested_ids[~valid].tolist()) == {row["question_id"] for row in excluded}

    e17 = ASSETS / "research/e17_supervision_distortion_audit"
    e17private = e17 / "private/confirmation_input_only"
    source_manifest = read(e17private / "INPUT_AND_CACHE_MANIFEST.json")
    frozen = read(e17private / "CONFIRMATION_PREPARATION_FROZEN.json")
    fit_log = read(e17private / "FULL600_FIT_LOG.json")
    config_path = e17 / "public/CONFIRMATION_INPUT_ONLY_CONFIG_FROZEN.json"
    config = read(config_path)
    assert sha(config_path) == frozen["config_sha256"]
    pinned_fit = next(row for row in frozen["private_artifacts"] if Path(row["path"]).name == "FULL600_FIT_LOG.json")
    assert sha(e17private / "FULL600_FIT_LOG.json") == pinned_fit["sha256"]
    train_path = ASSETS / "artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_outcomes.npz"
    input_path = ASSETS / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/selector_train_inputs.npz"
    for path in (train_path, input_path):
        declaration = next(row for row in source_manifest["files"] if Path(row["path"]) == path)
        assert sha(path) == declaration["sha256"]
    with np.load(train_path, allow_pickle=False) as packet:
        scores = packet["aggregate_scores"]
        train_ids = packet["question_ids"]
        train_canonical = packet["canonical_hashes"]
        seeds = packet["generation_seeds"]
    with np.load(input_path, allow_pickle=False) as packet:
        assert np.array_equal(train_ids, packet["question_ids"])
        assert np.array_equal(train_canonical, packet["canonical_hashes"])
        train_tasks = packet["task_labels"]
    assert scores.shape == (3, 600, 20) and np.isin(scores, (0,1)).all()
    assert list(seeds) == [42,137,2024] and Counter(train_tasks.tolist()) == {"gsm8k":300,"bbh":300}
    correct = scores.sum(axis=(0,1))
    means = scores.mean(axis=0).mean(axis=0)
    winners = np.flatnonzero(means == means.max())
    fixed = int(fit_log["fixed_policies"]["global_action"])
    assert fixed == 5 and int(winners[0]) == fixed
    config_sources = {}
    for name, expected in config["source_hashes"].items():
        path = ROOT / "experiments/e17_supervision_distortion_audit" / name
        assert sha(path) == expected
        config_sources[name] = expected
    fresh_action_path = e17private / "FRESH_POLICY_ACTIONS_FROZEN.npz"
    declared = next(row for row in frozen["private_artifacts"] if Path(row["path"]).name == fresh_action_path.name)
    assert sha(fresh_action_path) == declared["sha256"]
    with np.load(fresh_action_path, allow_pickle=False) as packet:
        e17_policies = packet["policies"].tolist()
        e17_fixed = packet["actions"][e17_policies.index("GlobalFixed")]
        old_ids = packet["question_ids"]
    e18_predictions = ASSETS / "research/e18_mechanism_and_confirmation_preparation/private/FROZEN3000_INPUT_PREDICTIONS.npz"
    with np.load(e18_predictions, allow_pickle=False) as packet:
        e18_actions = packet["actions"]
        e18_ids = packet["question_ids"]
        e18_canonical = packet["canonical_hashes"]
        e18_policies = packet["policies"].tolist()
    assert set(train_ids.tolist()).isdisjoint(e18_ids.tolist())
    assert set(train_canonical.tolist()).isdisjoint(e18_canonical.tolist())
    assert np.array_equal(e18_ids[:400], old_ids)
    assert np.all(e17_fixed == fixed) and np.all(e18_actions[e18_policies.index("GlobalFixed")] == fixed)
    assert np.array_equal(e18_actions, actions) and np.array_equal(e18_ids, np.asarray(question_ids))
    source_hashes = dict(
        completed_native_audit=sha(ROOT / "experiments/e18_confirmatory_execution/CLOSED_NATIVE_INDEPENDENT_AUDIT_FINAL.json"),
        final_result=sha(RUNTIME / "FINAL_RESULTS_PRIVATE.json"),
        train600_reward_npz=sha(train_path), train600_input_npz=sha(input_path),
        e17_full600_fit_log=sha(e17private / "FULL600_FIT_LOG.json"),
        e17_input_cache_manifest=sha(e17private / "INPUT_AND_CACHE_MANIFEST.json"),
        e17_confirmation_frozen_receipt=sha(e17private / "CONFIRMATION_PREPARATION_FROZEN.json"),
        e17_prefit_config=sha(config_path), e17_fresh_frozen_actions=sha(fresh_action_path),
        e18_frozen_input_actions=sha(e18_predictions),
        e18_execution_scope_expected_from_manifest=execution["scope_ref"]["sha256"],
        e18_frozen_input_budget_source=sha(ROOT / "experiments/e18_mechanism_and_confirmation_preparation/frozen_input_budget.py"))
    result = dict(schema="e18_1_accounting_and_global_fixed_audit_v1",status="PASS_READONLY_ACCOUNTING_AND_TRAIN_SOURCE",
        publication_base_commit="95c6c8a38a71866d6efdb224aec04b9a158adf17",
        requested_questions=3000, valid_questions=2998, excluded_questions=2,
        required_aggregates=30921, valid_scored_aggregates=30900, executed_unscorable_aggregates=21,
        accounting=totals, total_tokens=totals["new"]["total_tokens"]+totals["historical"]["total_tokens"],
        all_physical_attempts=len(ledger), stage_totals=final["physical_cost_totals"],
        unscorable_execution_by_subtask=public_cases,
        unscorable_aggregate_tokens=sum(row["aggregate_tokens"] for row in public_cases),
        unscorable_member_tokens=sum(row["member_tokens"] for row in public_cases),
        global_fixed=dict(action=fixed, team_indices=list(TEAMS[fixed]), train_questions=600,
            train_generation_events=1800, exact_maximizers=winners.tolist(),
            selected_correct_generation_events=int(correct[fixed]), selected_mean_accuracy=float(means[fixed]),
            runner_up_action=int(np.argsort(-means, kind="stable")[1]),
            all20_training_means=[dict(action=index,correct_events=int(correct[index]),
                                      generation_events=1800,mean_accuracy=float(means[index])) for index in range(20)],
            e17_fresh_input_action_count=len(e17_fixed), e18_frozen_action_count=len(e18_actions[3]),
            all_e17_global_actions_equal5=True, all_e18_global_actions_equal5=True,
            train_id_and_canonical_alignment=True, e17_first400_to_e18_identity_alignment=True,
            train600_e18_question_id_overlap=0, train600_e18_canonical_overlap=0,
            e18_scope_and_frozen_predictions_actions_equal=True,
            prefit_config_fresh_rewards_read=config.get("fresh_rewards_read"),
            frozen_receipt_fresh_rewards_read=frozen.get("fresh_rewards_read"),
            frozen_receipt_independent_evaluation_executed=frozen.get("independent_evaluation_executed"),
            frozen_receipt_sealed_payloads_read=frozen.get("sealed_payloads_read"),
            source_code_hashes=config_sources), source_hashes=source_hashes,
        new_llm_calls=0, training_runs=0, sealed_sources_opened=0,
        evaluation_reward_matrix_members_decoded=0, standard_answer_sources_opened=0,
        archive_payloads_opened_or_hashed=0,
        limitation="This audit verifies existing immutable lineage and recomputes the historical mean. It is not a time-travel proof beyond the frozen receipts and source bindings.")
    private = ROOT / ".e18_cpu_runtime/e18_1_statistics_and_mechanisms"
    private.mkdir(parents=True, exist_ok=True)
    with (private / "ACCOUNTING_EXCLUDED_PRIVATE.json").open("x",encoding="utf-8") as stream:
        json.dump(private_rows,stream,ensure_ascii=False,indent=2)
        stream.write("\n")
    with (STUDY / "ACCOUNTING_GLOBAL_FIXED_AGGREGATES.json").open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"],tokens=result["total_tokens"],
                         global_action=fixed,global_train_mean=float(means[fixed]),
                         excluded_aggregate_calls=len(excluded_keys),cases=public_cases),ensure_ascii=False))


if __name__ == "__main__":
    main()
