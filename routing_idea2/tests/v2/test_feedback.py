"""Regression tests for persistent v2 reference permissions and actual receipts."""
from dataclasses import replace
from hashlib import sha256
from itertools import combinations

import pytest

from budget_router_v2 import (BudgetFeedbackLedger, ExecutionReceipt,
                             ScoreObservation, LedgerError, UnknownCostError)


def receipt(name="member-42", *, role="member", seed=42, question="training-q", **changes):
    item = ExecutionReceipt(name, name + "-attempt", question, role, "model-fingerprint",
                            seed, sha256(name.encode()).hexdigest(), "success", 11, 3, 0.25)
    return replace(item, **changes)


def observation(execution_id="member-42", **changes):
    item = ScoreObservation(execution_id, "scorer-v1", "reference-v1", 1.0, 0)
    return replace(item, **changes)


def reveal(ledger, question="training-q", reference="reference-v1"):
    assert ledger.request_gold(question, reference, now=0, available_at=0)
    assert ledger.arrive_gold(question, reference, now=0)


def test_pending_gold_survives_restart_and_arrives_only_once(tmp_path):
    path = tmp_path / "ledger.sqlite"
    with BudgetFeedbackLedger(path) as ledger:
        assert ledger.record_execution(receipt())
        assert ledger.request_gold("training-q", "reference-v1", now=0, available_at=5)
        assert not ledger.request_gold("training-q", "reference-v1", now=1, available_at=6)
        with pytest.raises(LedgerError):
            ledger.arrive_gold("training-q", "reference-v1", now=4)
        with pytest.raises(LedgerError):
            ledger.record_score(observation(available_at=5), now=5)
    with BudgetFeedbackLedger(path) as ledger:
        assert ledger.summary()["gold_pending"] == 1
        assert ledger.arrive_gold("training-q", "reference-v1", now=5)
        assert not ledger.arrive_gold("training-q", "reference-v1", now=6)
        assert ledger.record_score(observation(available_at=5), now=5)
        assert ledger.summary()["gold_reveals"] == 1
        assert ledger.summary()["gold_pending"] == 0


def test_repeated_gold_request_after_original_due_preserves_pending_schedule(tmp_path):
    path = tmp_path / "ledger.sqlite"
    with BudgetFeedbackLedger(path) as ledger:
        assert ledger.request_gold("training-q", "reference-v1", now=1, available_at=3)
    with BudgetFeedbackLedger(path) as ledger:
        assert not ledger.request_gold("training-q", "reference-v1", now=10, available_at=3)
        assert ledger.arrive_gold("training-q", "reference-v1", now=10)
        assert not ledger.request_gold("training-q", "reference-v1", now=12, available_at=3)
        assert ledger.summary()["gold_reveals"] == 1


def test_one_question_reference_scores_six_members_and_twenty_teams(tmp_path):
    names = [("member-" + str(i), "member") for i in range(6)]
    names += [("team-" + "".join(map(str, team)), "aggregator")
              for team in combinations(range(6), 3)]
    with BudgetFeedbackLedger(tmp_path / "ledger.sqlite") as ledger:
        reveal(ledger)
        for name, role in names:
            assert ledger.record_execution(receipt(name, role=role))
            assert ledger.record_score(observation(name), now=0)
            assert not ledger.record_score(observation(name), now=0)
        summary = ledger.summary()
        assert summary["gold_reveals"] == summary["gold_arrived"] == 1
        assert summary["scoring_ops"] == 26
        assert summary["roles"]["member"]["calls"] == 6
        assert summary["roles"]["aggregator"]["calls"] == 20
        assert summary["roles"]["judge"]["calls"] == 0


def test_new_seed_uses_new_execution_but_reuses_arrived_question_gold(tmp_path):
    with BudgetFeedbackLedger(tmp_path / "ledger.sqlite") as ledger:
        reveal(ledger)
        first = receipt()
        assert ledger.record_execution(first)
        assert ledger.record_score(observation(), now=0)
        with pytest.raises(LedgerError):
            ledger.record_execution(replace(first, generation_seed=43))
        with pytest.raises(LedgerError):
            ledger.record_execution(replace(first, attempt_id="changed-seed-attempt", generation_seed=43))
        second = receipt("member-43", seed=43)
        assert ledger.record_execution(second)
        assert ledger.record_score(observation("member-43"), now=0)
        assert ledger.summary()["gold_reveals"] == 1
        assert ledger.summary()["scoring_ops"] == 2
        assert ledger.summary()["roles"]["member"]["calls"] == 2


def test_execution_receipts_are_idempotent_across_restart(tmp_path):
    path = tmp_path / "ledger.sqlite"
    item = receipt()
    with BudgetFeedbackLedger(path) as ledger:
        assert not ledger.has_success(item.execution_id)
        assert ledger.record_execution(item)
    with BudgetFeedbackLedger(path) as ledger:
        assert ledger.has_success(item.execution_id)
        assert not ledger.record_execution(item)
        with pytest.raises(LedgerError):
            ledger.record_execution(replace(item, attempt_id="duplicate-physical-attempt"))
        assert ledger.summary()["roles"]["member"]["calls"] == 1


def test_known_failed_attempt_and_successful_retry_both_keep_actual_cost(tmp_path):
    failure = receipt(status="failure", output_hash=None, input_tokens=7,
                      output_tokens=0, wall_time_s=0.4)
    success = replace(receipt(), attempt_id="retry-success")
    with BudgetFeedbackLedger(tmp_path / "ledger.sqlite") as ledger:
        assert ledger.record_execution(failure)
        assert not ledger.record_execution(failure)
        assert not ledger.has_success(failure.execution_id)
        assert ledger.record_execution(success)
        total = ledger.summary()["roles"]["member"]
        assert total["calls"] == 2
        assert total["failure_attempts"] == total["successful_executions"] == 1
        assert total["input_tokens"] == 18
        assert total["output_tokens"] == 3
        assert total["wall_time_s"] == pytest.approx(0.65)


def test_unknown_failure_is_durable_and_stops_new_work_until_reconciled(tmp_path):
    path = tmp_path / "ledger.sqlite"
    unknown = receipt(status="failure", output_hash=None, input_tokens=None,
                      output_tokens=None, wall_time_s=None)
    with BudgetFeedbackLedger(path) as ledger:
        with pytest.raises(UnknownCostError):
            ledger.record_execution(unknown)
        assert ledger.summary()["blocked"]
    with BudgetFeedbackLedger(path) as ledger:
        assert ledger.summary()["unknown_cost_attempts"] == 1
        assert ledger.summary()["roles"]["member"]["input_tokens"] is None
        with pytest.raises(UnknownCostError):
            ledger.request_gold("other-q", "reference-v1", now=0, available_at=0)
        with pytest.raises(UnknownCostError):
            ledger.record_execution(receipt("other-execution"))
        with pytest.raises(UnknownCostError):
            ledger.record_score(observation(), now=0)
        assert ledger.reconcile_execution(unknown.attempt_id, input_tokens=7,
                                          output_tokens=0, wall_time_s=0.4)
        assert not ledger.reconcile_execution(unknown.attempt_id, input_tokens=7,
                                              output_tokens=0, wall_time_s=0.4)
        assert not ledger.summary()["blocked"]
        assert ledger.record_execution(replace(receipt(), attempt_id="known-retry"))
        assert ledger.summary()["roles"]["member"]["calls"] == 2


def test_reconciliation_cannot_replace_already_known_cost(tmp_path):
    partial = receipt(status="failure", output_hash=None, input_tokens=7,
                      output_tokens=None, wall_time_s=None)
    with BudgetFeedbackLedger(tmp_path / "ledger.sqlite") as ledger:
        with pytest.raises(UnknownCostError):
            ledger.record_execution(partial)
        with pytest.raises(LedgerError):
            ledger.reconcile_execution(partial.attempt_id, input_tokens=8,
                                        output_tokens=0, wall_time_s=0.4)
        assert ledger.summary()["blocked"]


def test_scoring_requires_success_matching_reference_and_arrival(tmp_path):
    with BudgetFeedbackLedger(tmp_path / "ledger.sqlite") as ledger:
        with pytest.raises(LedgerError):
            ledger.record_score(observation(), now=0)
        assert ledger.record_execution(receipt())
        with pytest.raises(LedgerError):
            ledger.record_score(observation(), now=0)
        reveal(ledger, question="different-q")
        with pytest.raises(LedgerError):
            ledger.record_score(observation(), now=0)
        reveal(ledger)
        with pytest.raises(LedgerError):
            ledger.record_score(observation(reference_version="wrong-reference"), now=0)
        with pytest.raises(LedgerError):
            ledger.record_score(observation(available_at=1), now=0)
        assert ledger.summary()["scoring_ops"] == 0


def test_score_target_dedup_ignores_replay_timing_and_rejects_changed_value(tmp_path):
    with BudgetFeedbackLedger(tmp_path / "ledger.sqlite") as ledger:
        reveal(ledger)
        ledger.record_execution(receipt())
        assert ledger.record_score(observation(score_cpu_s=0.1, score_wall_s=0.2), now=0)
        assert not ledger.record_score(observation(available_at=3, score_cpu_s=4,
                                                    score_wall_s=5), now=3)
        with pytest.raises(LedgerError):
            ledger.record_score(observation(value=0.0), now=0)
        summary = ledger.summary()
        assert summary["scoring_ops"] == 1
        assert summary["score_cpu_s"] == pytest.approx(0.1)
        assert summary["score_wall_s"] == pytest.approx(0.2)


def test_external_judge_needs_own_successful_same_question_receipt(tmp_path):
    with BudgetFeedbackLedger(tmp_path / "ledger.sqlite") as ledger:
        reveal(ledger)
        ledger.record_execution(receipt())
        ledger.record_execution(receipt("wrong-role"))
        ledger.record_execution(receipt("wrong-question", role="judge", question="other-q"))
        for judge_id in ("missing", "wrong-role", "wrong-question"):
            with pytest.raises(LedgerError):
                ledger.record_score(observation(judge_execution_id=judge_id), now=0)
        ledger.record_execution(receipt("paid-judge", role="judge"))
        assert ledger.record_score(observation(judge_execution_id="paid-judge"), now=0)
        assert ledger.summary()["roles"]["judge"]["calls"] == 2
        assert ledger.summary()["scoring_ops"] == 1


def test_actual_planning_and_update_overheads_are_separate_idempotent_events(tmp_path):
    path = tmp_path / "ledger.sqlite"
    with BudgetFeedbackLedger(path) as ledger:
        assert ledger.record_overhead("plan-1", "planning", cpu_s=0.2, wall_s=0.3)
        assert not ledger.record_overhead("plan-1", "planning", cpu_s=0.2, wall_s=0.3)
        assert ledger.record_overhead("update-1", "update", cpu_s=0.4, wall_s=0.5)
    with BudgetFeedbackLedger(path) as ledger:
        summary = ledger.summary()
        assert summary["overheads"]["planning"] == {"events": 1, "cpu_s": 0.2, "wall_s": 0.3}
        assert summary["overheads"]["update"] == {"events": 1, "cpu_s": 0.4, "wall_s": 0.5}
        with pytest.raises(LedgerError):
            ledger.record_overhead("plan-1", "planning", cpu_s=9, wall_s=9)
