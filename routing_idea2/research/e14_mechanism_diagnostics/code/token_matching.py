"""Whole-package retrospective matching of already executed E13 feedback.

Token matching uses the actual logger-service plus feedback resource ledger.
It does not purchase labels, fit a learner, inspect held-out rewards, or split
a main-plus-neighbor package to fill a remaining token allowance.
"""
from __future__ import annotations

from copy import deepcopy
import math


RESOURCE_FIELDS = (
    "member_calls", "aggregator_calls", "gold", "scoring_ops",
    "input_tokens", "output_tokens", "tokens",
)
INTEGER_FIELDS = frozenset(RESOURCE_FIELDS)


def _checked_resources(values):
    if not isinstance(values, dict) or not set(RESOURCE_FIELDS) <= set(values):
        raise ValueError("Execution resources lack required ledger fields")
    output = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Resource {name} must be a finite nonnegative number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Resource {name} must be a finite nonnegative number")
        if name in INTEGER_FIELDS and int(value) != value:
            raise ValueError(f"Resource {name} must be an integer")
        output[name] = int(value) if name in INTEGER_FIELDS else float(value)
    if output["tokens"] != output["input_tokens"] + output["output_tokens"]:
        raise ValueError("Actual total tokens must equal input plus output tokens")
    return output


def _zero_like(resources):
    return {name: 0 for name in resources}


def _add_resources(left, right):
    if set(left) != set(right):
        raise ValueError("Resource ledgers have different fields")
    return {name: left[name] + right[name] for name in left}


def _same_resources(left, right, context):
    if set(left) != set(right):
        raise ValueError(f"{context} has different resource fields")
    for name in left:
        if name in INTEGER_FIELDS:
            same = left[name] == right[name]
        else:
            same = math.isclose(left[name], right[name], rel_tol=1e-12, abs_tol=1e-8)
        if not same:
            raise ValueError(f"{context} mismatch in {name}")


def _execution_keys(entry, question, generation, resources):
    result = {}
    for field, expected in (("member_keys", resources["member_calls"]),
                            ("aggregate_keys", resources["aggregator_calls"])):
        keys = entry.get(field)
        if not isinstance(keys, list) or len(keys) != expected:
            raise ValueError(f"{field} disagrees with actual newly executed calls")
        normalized = []
        for key in keys:
            if len(key) != 3 or key[0] != question or key[1] != generation:
                raise ValueError(f"{field} contains an inconsistent execution identity")
            normalized.append(tuple(key))
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{field} contains a duplicated paid execution")
        result[field] = set(normalized)
    return result


def execution_packages(log):
    """Validate and join each E13 service/feedback pair into one whole package.

    The original execution order is retained. All observed tuples are copied
    from completed feedback events. Standalone tuple costs are never summed.
    Both each intermediate cumulative ledger and the final phase totals are
    checked against sums of the already committed ``resources`` fields.
    """
    entries = log.get("execution_log")
    summary = log.get("final_summary")
    if not isinstance(entries, list) or not isinstance(summary, dict):
        raise ValueError("An E13 execution log and final_summary are required")
    ledger = summary.get("ledger")
    if not isinstance(ledger, dict) or set(ledger) != {"logger_service", "feedback", "total"}:
        raise ValueError("Final execution ledger must contain all three phases")
    final = {name: _checked_resources(values) for name, values in ledger.items()}
    total = _zero_like(final["total"])
    phase_totals = {name: _zero_like(final["total"]) for name in ("logger_service", "feedback")}
    if len(entries) % 2:
        raise ValueError("An incomplete service/feedback package cannot be matched")
    packages, questions = [], set()
    paid_keys = {"member_keys": set(), "aggregate_keys": set()}
    scenario = None
    generation = summary.get("generation_index")
    for offset in range(0, len(entries), 2):
        service, feedback = entries[offset:offset + 2]
        if service.get("phase") != "logger_service" or feedback.get("phase") != "feedback":
            raise ValueError("E13 packages must retain service then feedback order")
        question = service.get("question_index")
        if (not isinstance(question, int) or question in questions
                or feedback.get("question_index") != question):
            raise ValueError("A complete package must contain one unique historical question")
        questions.add(question)
        if generation is None:
            generation = service.get("generation_index")
        if (service.get("generation_index") != generation
                or feedback.get("generation_index") != generation):
            raise ValueError("One execution account must use a single generation repeat")
        main_action = service.get("main_action")
        actions = feedback.get("actions")
        if (not isinstance(main_action, int) or not 0 <= main_action < 20
                or not isinstance(actions, list) or len(actions) not in (1, 2)
                or len(set(actions)) != len(actions) or main_action not in actions):
            raise ValueError("A feedback package must be the logged main or main plus neighbor")
        package_scenario = "Main" if len(actions) == 1 else "Pair"
        if scenario is not None and package_scenario != scenario:
            raise ValueError("Main and Pair actions cannot be mixed in one E13 account")
        scenario = package_scenario
        observations = feedback.get("observations")
        if not isinstance(observations, list) or len(observations) != len(actions):
            raise ValueError("A whole package must reveal every paid action exactly once")
        if {row.get("action") for row in observations} != set(actions):
            raise ValueError("Observed actions disagree with the executed feedback package")
        if any(row.get("question_index") != question or row.get("generation_index") != generation
               for row in observations):
            raise ValueError("An observed tuple escaped its executed question/generation package")
        service_resources = _checked_resources(service.get("resources"))
        feedback_resources = _checked_resources(feedback.get("resources"))
        if tuple(service_resources[name] for name in ("member_calls", "aggregator_calls", "gold", "scoring_ops")) != (3, 1, 0, 0):
            raise ValueError("Logger service must execute three members and one aggregator")
        expected_feedback = (0, 0, 1, 1) if scenario == "Main" else (1, 1, 1, 2)
        if tuple(feedback_resources[name] for name in ("member_calls", "aggregator_calls", "gold", "scoring_ops")) != expected_feedback:
            raise ValueError("Paid feedback calls must use the logged Main/Pair execution reuse")
        for entry, resources in ((service, service_resources), (feedback, feedback_resources)):
            keys = _execution_keys(entry, question, generation, resources)
            for key_type in paid_keys:
                if paid_keys[key_type] & keys[key_type]:
                    raise ValueError("A shared execution has been charged more than once")
                paid_keys[key_type].update(keys[key_type])
            total = _add_resources(total, resources)
            phase_totals[entry["phase"]] = _add_resources(phase_totals[entry["phase"]], resources)
            _same_resources(total, _checked_resources(entry.get("cumulative")), "Committed cumulative ledger")
        resources = _add_resources(service_resources, feedback_resources)
        expected_package = (3, 1, 1, 1) if scenario == "Main" else (4, 2, 1, 2)
        if tuple(resources[name] for name in ("member_calls", "aggregator_calls", "gold", "scoring_ops")) != expected_package:
            raise ValueError("Complete question package has incorrect paid resources")
        packages.append({
            "question_index": question, "generation_index": generation,
            "main_action": main_action, "actions": list(actions), "scenario": scenario,
            "observations": deepcopy(observations), "resources": resources,
            "cumulative": dict(total), "logger_service_resources": service_resources,
            "feedback_resources": feedback_resources,
        })
    _same_resources(total, final["total"], "Final total ledger")
    for phase in phase_totals:
        _same_resources(phase_totals[phase], final[phase], f"Final {phase} ledger")
    if summary.get("served_questions") != len(packages):
        raise ValueError("Final served question count disagrees with executed packages")
    if summary.get("score_observations") != sum(len(row["observations"]) for row in packages):
        raise ValueError("Final observed score count disagrees with executed packages")
    return packages


def _prefix_summary(packages, count, resource_template):
    resources = _zero_like(resource_template)
    observations = []
    for package in packages[:count]:
        resources = _add_resources(resources, _checked_resources(package["resources"]))
        observations.extend(deepcopy(package["observations"]))
    return {"n_packages": count,
            "question_indices": [row["question_index"] for row in packages[:count]],
            "observations": observations, "resources": resources}


def match_pair_prefix(main_packages, pair_packages, main_gold):
    """Anchor a common token cap at an executed Main prefix, then fit Pair under it.

    Only the longest complete original Pair prefix within the cap is selected.
    All costs come from already executed packages. This is retrospective
    controlled accounting and does not assume future unexecuted costs are known.
    A common upper cap generally leaves different actual token expenditures.
    """
    if isinstance(main_gold, bool) or not isinstance(main_gold, int) or not 0 <= main_gold <= len(main_packages):
        raise ValueError("Main gold anchor must be an available nonnegative complete prefix")
    if len(main_packages) != len(pair_packages):
        raise ValueError("Matched worlds must cover the same executed historical pool")
    for main, pair in zip(main_packages, pair_packages):
        if (main["scenario"] != "Main" or pair["scenario"] != "Pair"
                or main["question_index"] != pair["question_index"]
                or main["main_action"] != pair["main_action"]
                or main["generation_index"] != pair["generation_index"]):
            raise ValueError("Main and Pair must share the original question order and main actions")
        _same_resources(main["logger_service_resources"], pair["logger_service_resources"],
                        "Shared logger main execution")
        if pair["resources"]["tokens"] < main["resources"]["tokens"]:
            raise ValueError("Pair cannot cost less than its identical main execution")
    resource_template = main_packages[0]["resources"] if main_packages else dict.fromkeys(RESOURCE_FIELDS, 0)
    main = _prefix_summary(main_packages, main_gold, resource_template)
    target = main["resources"]["tokens"]
    spent, selected = 0, 0
    for package in pair_packages:
        next_spend = spent + package["resources"]["tokens"]
        if next_spend > target:
            break
        spent, selected = next_spend, selected + 1
    pair = _prefix_summary(pair_packages, selected, resource_template)
    remaining = target - pair["resources"]["tokens"]
    next_package = pair_packages[selected] if selected < len(pair_packages) else None
    next_tokens = next_package["resources"]["tokens"] if next_package else None
    if remaining < 0 or (next_tokens is not None and next_tokens <= remaining):
        raise ValueError("Pair prefix is not the maximal whole-package prefix under the cap")
    if pair["resources"]["gold"] > main_gold:
        raise ValueError("Pair prefix exceeds the common original-question gold allowance")
    return {
        "rule": "Main actual total-token anchor, longest executed complete Pair prefix",
        "scope": "retrospective controlled diagnostic using already executed E13 costs",
        "main_gold": main_gold, "main_questions": main_gold, "pair_questions": selected,
        "common_token_cap": target, "common_gold_cap": main_gold,
        "main": main, "pair": pair,
        "main_actual_tokens": main["resources"]["tokens"],
        "pair_actual_tokens": pair["resources"]["tokens"],
        "remaining_tokens": remaining,
        "remaining_token_fraction": remaining / target if target else 0.0,
        "pair_gold_remaining": main_gold - pair["resources"]["gold"],
        "next_pair_package_tokens": next_tokens,
        "next_pair_package_excess_over_remaining": next_tokens - remaining if next_tokens is not None else None,
        "pair_execution_pool_exhausted": next_package is None,
        "actual_token_spend_equal": main["resources"]["tokens"] == pair["resources"]["tokens"],
        "resource_difference_main_minus_pair": {name: main["resources"][name] - pair["resources"][name]
                                                for name in resource_template},
        "same_upper_limit_comparison": True,
        "cost_based_prefix_selection_only": True,
        "accuracy_interpolation": False,
    }
