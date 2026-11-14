"""Small synthetic qualification of whole-package realized-cost matching."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_matching import RESOURCE_FIELDS, execution_packages, match_pair_prefix


def resource(member, aggregate, gold, score, tokens):
    result = dict.fromkeys(RESOURCE_FIELDS, 0)
    result.update(member_calls=member, aggregator_calls=aggregate, gold=gold,
                  scoring_ops=score, input_tokens=tokens, tokens=tokens)
    return result


def fixture(scenario, main_costs=(10, 10, 10), extra_costs=(5, 5, 5)):
    total = dict.fromkeys(RESOURCE_FIELDS, 0)
    phase_totals = {name: dict(total) for name in ("logger_service", "feedback")}
    entries = []
    for question, (main_cost, extra_cost) in enumerate(zip(main_costs, extra_costs)):
        service_cost = resource(3, 1, 0, 0, main_cost)
        pair = scenario == "Pair"
        feedback_cost = resource(int(pair), int(pair), 1, 2 if pair else 1,
                                 extra_cost if pair else 0)
        for phase, cost in (("logger_service", service_cost), ("feedback", feedback_cost)):
            total = {key: total[key] + cost[key] for key in total}
            phase_totals[phase] = {key: phase_totals[phase][key] + cost[key] for key in total}
            entry = {"question_index": question, "generation_index": 0, "phase": phase,
                     "resources": cost, "cumulative": dict(total),
                     "member_keys": [[question, 0, i + (3 if phase == "feedback" else 0)] for i in range(cost["member_calls"])],
                     "aggregate_keys": [[question, 0, i + (1 if phase == "feedback" else 0)] for i in range(cost["aggregator_calls"])]}
            if phase == "logger_service":
                entry["main_action"] = 0
            else:
                entry["actions"] = [0, 1] if pair else [0]
                entry["observations"] = [
                    {"question_index": question, "generation_index": 0, "action": action,
                     "reward": 1, "standalone_tokens": 10000}
                    for action in entry["actions"]]
            entries.append(entry)
    return {"execution_log": entries,
            "final_summary": {"generation_index": 0, "served_questions": len(main_costs),
                              "score_observations": len(main_costs) * (2 if scenario == "Pair" else 1),
                              "ledger": {**phase_totals, "total": dict(total)}}}


class TokenMatchingTests(unittest.TestCase):
    def setUp(self):
        self.main_log, self.pair_log = fixture("Main"), fixture("Pair")
        self.main = execution_packages(self.main_log)
        self.pair = execution_packages(self.pair_log)

    def test_actual_shared_resources_ignore_standalone_counterfactual(self):
        package = self.pair[0]
        self.assertEqual(package["resources"]["tokens"], 15)
        self.assertEqual(package["resources"]["member_calls"], 4)
        self.assertEqual(package["resources"]["aggregator_calls"], 2)
        self.assertEqual(package["resources"]["gold"], 1)
        self.assertEqual(package["resources"]["scoring_ops"], 2)

    def test_zero_anchor_returns_a_legal_empty_feedback_prefix(self):
        match = match_pair_prefix(self.main, self.pair, 0)
        self.assertEqual(match["main"]["observations"], [])
        self.assertEqual(match["pair"]["observations"], [])
        self.assertEqual(match["pair"]["resources"]["tokens"], 0)
        self.assertTrue(match["actual_token_spend_equal"])

    def test_insufficient_allowance_returns_empty_pair_prefix(self):
        match = match_pair_prefix(self.main, self.pair, 1)
        self.assertEqual(match["pair"]["n_packages"], 0)
        self.assertEqual(match["remaining_tokens"], 10)
        self.assertEqual(match["next_pair_package_excess_over_remaining"], 5)

    def test_complete_pair_package_is_never_split(self):
        match = match_pair_prefix(self.main, self.pair, 2)
        self.assertEqual(match["common_token_cap"], 20)
        self.assertEqual(match["pair"]["n_packages"], 1)
        self.assertEqual(len(match["pair"]["observations"]), 2)
        self.assertEqual(match["remaining_tokens"], 5)
        self.assertAlmostEqual(match["remaining_token_fraction"], .25)
        self.assertFalse(match["actual_token_spend_equal"])

    def test_exact_actual_token_match_is_recorded_when_it_occurs(self):
        match = match_pair_prefix(self.main, self.pair, 3)
        self.assertEqual(match["pair"]["n_packages"], 2)
        self.assertEqual(match["remaining_tokens"], 0)
        self.assertTrue(match["actual_token_spend_equal"])

    def test_cumulative_tampering_is_rejected(self):
        bad = deepcopy(self.main_log)
        bad["execution_log"][1]["cumulative"]["tokens"] += 1
        bad["execution_log"][1]["cumulative"]["input_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "cumulative"):
            execution_packages(bad)

    def test_final_ledger_tampering_is_rejected(self):
        bad = deepcopy(self.pair_log)
        bad["final_summary"]["ledger"]["total"]["gold"] += 1
        with self.assertRaisesRegex(ValueError, "Final total"):
            execution_packages(bad)

    def test_incomplete_or_reordered_package_is_rejected(self):
        bad = deepcopy(self.pair_log)
        bad["execution_log"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            execution_packages(bad)
        bad = deepcopy(self.pair_log)
        bad["execution_log"][0], bad["execution_log"][1] = bad["execution_log"][1], bad["execution_log"][0]
        with self.assertRaisesRegex(ValueError, "service then feedback"):
            execution_packages(bad)

    def test_common_question_order_and_main_action_are_required(self):
        bad = deepcopy(self.pair)
        bad[0]["main_action"] = 3
        with self.assertRaisesRegex(ValueError, "original question order"):
            match_pair_prefix(self.main, bad, 2)
        bad = deepcopy(self.pair)
        bad[0], bad[1] = bad[1], bad[0]
        with self.assertRaisesRegex(ValueError, "original question order"):
            match_pair_prefix(self.main, bad, 2)

    def test_reused_members_must_not_be_charged_twice(self):
        bad = deepcopy(self.pair_log)
        bad["execution_log"][1]["member_keys"] = [[0, 0, 0]]
        with self.assertRaisesRegex(ValueError, "charged more than once"):
            execution_packages(bad)

    def test_remaining_tokens_cannot_skip_an_expensive_earlier_package(self):
        pair = execution_packages(fixture("Pair", extra_costs=(100, 0, 0)))
        match = match_pair_prefix(self.main, pair, 3)
        self.assertEqual(match["pair"]["n_packages"], 0)
        self.assertEqual(match["remaining_tokens"], 30)
        self.assertEqual(match["next_pair_package_tokens"], 110)


if __name__ == "__main__":
    unittest.main()
