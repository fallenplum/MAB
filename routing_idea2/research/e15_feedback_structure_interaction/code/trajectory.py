"""Frozen feedback structures and retrospective whole-package token prefixes.

The random logger sees only original question identity and frozen main actions.
The cache manager alone reads the selected executed rewards and cost cells.
All comparisons retain both the complete replay-pool cost and prefix cost.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json
import math
from numbers import Integral
import random


TEAMS = tuple(itertools.combinations(range(6), 3))
GENERATION_SEEDS = (42, 137, 2024)
TOKEN_CAPS = (125000, 250000, 500000, 750000)
RANDOM_SECOND_HASH_PREFIX = "E15-RandomPair-v1"
REFERENCE_VERSION = "V2_selector_train_frozen_bundle_15095d4cf8b4f2d564b62f814102a63fe05a9a9e5c9702e14c9b8232b80649db"
RESOURCE_FIELDS = (
    "member_calls", "aggregator_calls", "gold", "scoring_ops",
    "input_tokens", "output_tokens", "tokens", "source_model_wall_s",
    "member_input_tokens", "member_output_tokens", "member_source_wall_s",
    "aggregator_input_tokens", "aggregator_output_tokens", "aggregator_source_wall_s",
)
REQUIRED_RESOURCES = frozenset(RESOURCE_FIELDS[:7])
INTEGER_FIELDS = frozenset(name for name in RESOURCE_FIELDS if not name.endswith("wall_s"))


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def choose_random_second(qid, fold, logger, main):
    """Uniformly draw from all other19 actions, with no cost/reward input.

    A separate hash domain from the E13 main logger gives a frozen random plan.
    Generation seed is deliberately absent, so all three repeats share actions.
    Neighbors remain eligible, together with disjoint or partly shared teams.
    """
    fold = _integer(fold, "fold")
    logger = _integer(logger, "logger")
    main = _integer(main, "main action")
    if not 0 <= main < 20 or fold < 0:
        raise ValueError("Invalid frozen main action or fold")
    identity = json.dumps([RANDOM_SECOND_HASH_PREFIX, str(qid), fold, logger],
                          ensure_ascii=False, separators=(",", ":"))
    entropy = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest(), "big")
    position = random.Random(entropy).randrange(19)
    return position if position < main else position + 1


def _zero_resources():
    return {name: 0 for name in RESOURCE_FIELDS}


def _checked_resources(values):
    if not isinstance(values, dict) or not REQUIRED_RESOURCES <= values.keys():
        raise ValueError("Execution ledger is missing required resource fields")
    result = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Resource {name} must be a finite nonnegative scalar")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Resource {name} must be a finite nonnegative scalar")
        if name in INTEGER_FIELDS and int(value) != value:
            raise ValueError(f"Resource {name} must be an integer")
        result[name] = int(value) if name in INTEGER_FIELDS else float(value)
    if result["tokens"] != result["input_tokens"] + result["output_tokens"]:
        raise ValueError("Total tokens must equal input plus output tokens")
    for total, first, second in (
        ("input_tokens", "member_input_tokens", "aggregator_input_tokens"),
        ("output_tokens", "member_output_tokens", "aggregator_output_tokens"),
        ("source_model_wall_s", "member_source_wall_s", "aggregator_source_wall_s"),
    ):
        if {total, first, second} <= result.keys():
            if not math.isclose(result[total], result[first] + result[second], rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError(f"Detailed source costs disagree for {total}")
    return result


def _add(left, right):
    if set(left) != set(right):
        raise ValueError("Resource ledgers use different fields")
    return {name: left[name] + right[name] for name in left}


def _same(left, right, context):
    if set(left) != set(right):
        raise ValueError(f"{context} has different resource fields")
    for name, value in left.items():
        equal = value == right[name] if name in INTEGER_FIELDS else math.isclose(
            value, right[name], rel_tol=1e-12, abs_tol=1e-8)
        if not equal:
            raise ValueError(f"{context} disagrees for {name}")


def _check_execution_keys(entry, question, generation, members, actions):
    for name, columns in (("member_keys", members), ("aggregate_keys", actions)):
        keys = [tuple(key) for key in entry.get(name, [])]
        expected = {(question, generation, column) for column in columns}
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError(f"{name} must identify the actual newly executed union")


def execution_packages(log, scenario=None):
    """Parse unchanged E13 logs or E15 arbitrary-pair logs into complete packages.

    Main uses3 members and1 aggregate. Any distinct second action uses the exact
    4/5/6-member union and2 aggregates. NeighborPair additionally requires a
    single-member replacement. Cumulative and phase ledgers are checked.
    """
    requested = scenario or log.get("scenario")
    if requested == "Pair":
        requested = "NeighborPair"
    if requested not in (None, "Main", "NeighborPair", "RandomPair"):
        raise ValueError("Unknown feedback scenario")
    entries, summary = log.get("execution_log"), log.get("final_summary")
    if not isinstance(entries, list) or not isinstance(summary, dict):
        raise ValueError("A complete execution log and final summary are required")
    final = summary.get("ledger")
    if not isinstance(final, dict) or set(final) != {"logger_service", "feedback", "total"}:
        raise ValueError("Final ledger must contain all three execution phases")
    final = {name: _checked_resources(value) for name, value in final.items()}
    total = {name: 0 for name in final["total"]}
    phases = {name: dict(total) for name in ("logger_service", "feedback")}
    if len(entries) % 2:
        raise ValueError("Incomplete service/feedback packages cannot enter a prefix")
    packages, visited = [], set()
    generation = summary.get("generation_index")
    action_count = None
    for position in range(0, len(entries), 2):
        service, feedback = entries[position:position + 2]
        if service.get("phase") != "logger_service" or feedback.get("phase") != "feedback":
            raise ValueError("Original service then feedback order must be retained")
        question = _integer(service.get("question_index"), "question index")
        if question in visited or feedback.get("question_index") != question:
            raise ValueError("A package must contain one unique original question")
        visited.add(question)
        if generation is None:
            generation = service.get("generation_index")
        if service.get("generation_index") != generation or feedback.get("generation_index") != generation:
            raise ValueError("Each account must contain a single generation repeat")
        main = _integer(service.get("main_action"), "main action")
        actions = feedback.get("actions")
        if not 0 <= main < 20 or not isinstance(actions, list) or len(actions) not in (1, 2):
            raise ValueError("Feedback must contain the main or main plus one other action")
        actions = [_integer(action, "feedback action") for action in actions]
        if main not in actions or len(set(actions)) != len(actions) or any(not 0 <= action < 20 for action in actions):
            raise ValueError("Feedback actions must be distinct legal combinations")
        if action_count is not None and action_count != len(actions):
            raise ValueError("A world cannot mix one-action and two-action packages")
        action_count = len(actions)
        if requested == "Main" and len(actions) != 1:
            raise ValueError("Main scenario must purchase one action")
        if requested in ("NeighborPair", "RandomPair") and len(actions) != 2:
            raise ValueError("Pair scenarios must purchase two actions")
        union = set().union(*(set(TEAMS[action]) for action in actions))
        if requested == "NeighborPair" and len(union) != 4:
            raise ValueError("NeighborPair must replace exactly one member")
        observed = feedback.get("observations")
        if not isinstance(observed, list) or len(observed) != len(actions):
            raise ValueError("Every paid action must provide exactly one observed tuple")
        if {row.get("action") for row in observed} != set(actions):
            raise ValueError("Observed tuples disagree with the executed action set")
        if any(row.get("question_index") != question or row.get("generation_index") != generation for row in observed):
            raise ValueError("An observed tuple escaped its package scope")
        service_resource = _checked_resources(service.get("resources"))
        feedback_resource = _checked_resources(feedback.get("resources"))
        if tuple(service_resource[name] for name in ("member_calls", "aggregator_calls", "gold", "scoring_ops")) != (3, 1, 0, 0):
            raise ValueError("Frozen main service has incorrect execution counts")
        expected = (len(union) - 3, len(actions) - 1, 1, len(actions))
        if tuple(feedback_resource[name] for name in ("member_calls", "aggregator_calls", "gold", "scoring_ops")) != expected:
            raise ValueError("Actual union reuse or paid feedback counts are incorrect")
        _check_execution_keys(service, question, generation, set(TEAMS[main]), {main})
        _check_execution_keys(feedback, question, generation, union - set(TEAMS[main]), set(actions) - {main})
        for entry, resource in ((service, service_resource), (feedback, feedback_resource)):
            total = _add(total, resource)
            phases[entry["phase"]] = _add(phases[entry["phase"]], resource)
            _same(total, _checked_resources(entry.get("cumulative")), "Committed cumulative ledger")
        inferred = "Main" if len(actions) == 1 else "RandomPair"
        packages.append({"question_index": question, "generation_index": generation,
                         "main_action": main, "actions": actions, "scenario": requested or inferred,
                         "observations": deepcopy(observed), "resources": _add(service_resource, feedback_resource),
                         "cumulative": dict(total), "logger_service_resources": service_resource,
                         "feedback_resources": feedback_resource, "union_members": sorted(union)})
    _same(total, final["total"], "Final total ledger")
    for phase in phases:
        _same(phases[phase], final[phase], f"Final {phase} ledger")
    if summary.get("served_questions") != len(packages):
        raise ValueError("Final served question count disagrees with complete packages")
    if summary.get("score_observations") != sum(len(row["observations"]) for row in packages):
        raise ValueError("Final score observations disagree with complete packages")
    return packages


def _prefix_ledger(packages, count, template):
    phases = {name: {key: 0 for key in template} for name in ("logger_service", "feedback", "total")}
    for package in packages[:count]:
        phases["logger_service"] = _add(phases["logger_service"], package["logger_service_resources"])
        phases["feedback"] = _add(phases["feedback"], package["feedback_resources"])
        phases["total"] = _add(phases["total"], package["resources"])
    return phases


def summarize_prefix(packages, n_questions):
    """Summarize a fixed original-order question prefix without any cost selection."""
    n_questions = _integer(n_questions, "prefix question count")
    if not 0 <= n_questions <= len(packages):
        raise ValueError("Question prefix exceeds the already executed plan pool")
    template = packages[0]["resources"] if packages else _zero_resources()
    ledger = _prefix_ledger(packages, n_questions, template)
    full = _prefix_ledger(packages, len(packages), template)
    selected = packages[:n_questions]
    next_package = packages[n_questions] if n_questions < len(packages) else None
    return {"selection": "fixed_original_question_prefix", "n_questions": n_questions,
            "packages": deepcopy(selected),
            "observations": [deepcopy(row) for package in selected for row in package["observations"]],
            "ledger": ledger, "actual_tokens": ledger["total"]["tokens"],
            "token_cap": ledger["total"]["tokens"], "remaining_tokens": 0,
            "remaining_token_fraction": 0.0,
            "next_package_tokens": next_package["resources"]["tokens"] if next_package else None,
            "next_package_resources": deepcopy(next_package["resources"]) if next_package else None,
            "full_plan_questions": len(packages), "full_plan_ledger": full,
            "pool_exhausted": next_package is None,
            "new_physical_model_calls": 0, "new_gold_procurement": 0}


def select_prefix(packages, cap):
    """Select only a longest complete executed prefix under an actual-token cap.

    Costs are known because the complete cache plan has already been replayed.
    This function does not implement a prospective hard-budget acquisition rule.
    """
    cap = _integer(cap, "token cap")
    if cap < 0:
        raise ValueError("Token cap must be nonnegative")
    spent, count = 0, 0
    for package in packages:
        increment = _checked_resources(package["resources"])["tokens"]
        if spent + increment > cap:
            break
        spent, count = spent + increment, count + 1
    result = summarize_prefix(packages, count)
    remaining = cap - result["actual_tokens"]
    if result["next_package_tokens"] is not None and result["next_package_tokens"] <= remaining:
        raise ValueError("Selected prefix is not maximal under the complete-package rule")
    result.update(selection="retrospective_actual_token_cap", token_cap=cap,
                  remaining_tokens=remaining,
                  remaining_token_fraction=remaining / cap if cap else 0.0,
                  actual_spend_equals_cap=remaining == 0,
                  next_package_excess_over_remaining=(result["next_package_tokens"] - remaining
                                                      if result["next_package_tokens"] is not None else None),
                  online_hard_budget_guarantee=False, accuracy_interpolation=False)
    return result


class _RandomPairCacheManager:
    """E15-local cache manager for the fixed arbitrary second-action plan.

    E13's immutable environment restricts second actions to neighbors. This
    manager changes that admissible set and retains its execution/gold units.
    No dense reward array is exposed in the returned learner observations.
    """

    def __init__(self, train, allowed, forbidden, generation):
        self.allowed, self.forbidden = set(allowed), set(forbidden)
        if not self.allowed or self.allowed & self.forbidden:
            raise ValueError("Historical purchase pool and evaluation pool must be disjoint")
        self.generation = generation
        self._rewards = train["Y"]
        self._costs = train["cost"]
        self._canonical = train["canonical"]
        if len({str(self._canonical[q]) for q in self.allowed}) != len(self.allowed):
            raise ValueError("Original purchase pool must have distinct canonical questions")
        self.ledger = {name: _zero_resources() for name in ("logger_service", "feedback", "total")}
        self.execution_log, self.access_log, self.observations = [], [], []
        self.members, self.aggregates, self.known_costs = {}, {}, {}
        self.gold_keys = set()

    def _execution(self, question, actions, phase):
        import numpy as np

        union = set().union(*(set(TEAMS[action]) for action in actions))
        members = union - self.members.get(question, set())
        aggregates = set(actions) - self.aggregates.get(question, set())
        resources = _zero_resources()
        pending = {}
        reads = []
        for stem, columns, count_key, detail in (
            ("member", sorted(members), "member_calls", "member"),
            ("aggregate", sorted(aggregates), "aggregator_calls", "aggregator"),
        ):
            if not columns:
                continue
            for suffix, total_key, detail_key in (
                ("execution_counts", count_key, None),
                ("prompt_tokens", "input_tokens", f"{detail}_input_tokens"),
                ("completion_tokens", "output_tokens", f"{detail}_output_tokens"),
                ("wall_time_s", "source_model_wall_s", f"{detail}_source_wall_s"),
            ):
                values = np.asarray(self._costs[f"{stem}_{suffix}"][self.generation, question, columns])
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("An accepted execution has invalid source costs")
                if suffix == "execution_counts" and not np.all(values == 1):
                    raise ValueError("Unexpected original retry counts require explicit accounting")
                amount = float(values.sum()) if suffix == "wall_time_s" else int(values.sum())
                resources[total_key] += amount
                if detail_key:
                    resources[detail_key] += amount
                for column, value in zip(columns, values):
                    pending[(question, self.generation, stem, suffix, column)] = float(value)
                reads.append({"question_index": question, "generation_index": self.generation,
                              "phase": phase, "kind": "cost", "stem": stem,
                              "field": suffix, "columns": columns})
        resources["tokens"] = resources["input_tokens"] + resources["output_tokens"]
        self.known_costs.update(pending)
        return resources, members, aggregates, reads

    def _commit(self, question, phase, resources, members, aggregates, reads):
        self.ledger[phase] = _add(self.ledger[phase], resources)
        self.ledger["total"] = _add(self.ledger["total"], resources)
        self.members.setdefault(question, set()).update(members)
        self.aggregates.setdefault(question, set()).update(aggregates)
        self.access_log.extend(reads)

    def _keys(self, question, members, aggregates):
        return {"member_keys": [[question, self.generation, member] for member in sorted(members)],
                "aggregate_keys": [[question, self.generation, action] for action in sorted(aggregates)]}

    def _standalone_tokens(self, question, action):
        tokens = 0.0
        for stem, columns in (("member", TEAMS[action]), ("aggregate", (action,))):
            for suffix in ("prompt_tokens", "completion_tokens"):
                for column in columns:
                    key = (question, self.generation, stem, suffix, column)
                    if key not in self.known_costs:
                        raise ValueError("Standalone tuple cost references an unexecuted source cell")
                    tokens += self.known_costs[key]
        return int(tokens)

    def execute(self, question, main, second):
        import numpy as np

        if question not in self.allowed or question in self.forbidden or question in self.members:
            raise ValueError("Only a new question in the fixed historical plan may be executed")
        resources, members, aggregates, reads = self._execution(question, [main], "logger_service")
        self._commit(question, "logger_service", resources, members, aggregates, reads)
        self.execution_log.append({"question_index": question, "generation_index": self.generation,
            "main_action": main, "phase": "logger_service", "resources": resources,
            "cumulative": dict(self.ledger["total"]), **self._keys(question, members, aggregates)})
        actions = sorted([main, second])
        resources, members, aggregates, reads = self._execution(question, actions, "feedback")
        gold_key = (str(self._canonical[question]), REFERENCE_VERSION)
        resources["gold"] = int(gold_key not in self.gold_keys)
        resources["scoring_ops"] = len(actions)
        values = np.asarray(self._rewards[question, self.generation, actions], dtype=float)
        if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
            raise ValueError("Selected cached final rewards must be real binary values")
        observed = [{"question_index": question, "generation_index": self.generation,
                     "generation_seed": GENERATION_SEEDS[self.generation], "action": action,
                     "reward": float(value), "standalone_tokens": self._standalone_tokens(question, action)}
                    for action, value in zip(actions, values)]
        reads.append({"question_index": question, "generation_index": self.generation,
                      "phase": "feedback", "kind": "reward", "actions": actions})
        self._commit(question, "feedback", resources, members, aggregates, reads)
        self.gold_keys.add(gold_key)
        self.observations.extend(observed)
        self.execution_log.append({"question_index": question, "generation_index": self.generation,
            "actions": actions, "phase": "feedback", "resources": resources,
            "observations": observed, "gold_reference_keys": [list(gold_key)] if resources["gold"] else [],
            "cumulative": dict(self.ledger["total"]), **self._keys(question, members, aggregates)})

    def summary(self):
        return {"ledger": deepcopy(self.ledger), "generation_index": self.generation,
                "generation_seed": GENERATION_SEEDS[self.generation], "reference_version": REFERENCE_VERSION,
                "gold_revealed": len(self.gold_keys), "gold_keys": [list(key) for key in sorted(self.gold_keys)],
                "served_questions": len(self.members), "score_observations": len(self.observations),
                "member_execution_cache_entries": sum(len(values) for values in self.members.values()),
                "aggregate_execution_cache_entries": sum(len(values) for values in self.aggregates.values()),
                "new_physical_model_calls": 0, "new_physical_gold_acquisition_calls": 0,
                "gpu_time_s": None,
                "gpu_time_note": "Original source-call wall sums are retained; no new GPU inference",
                "resource_limits": {}, "account_scope": "fixed full historical RandomPair cache plan"}


def build_random_world(train, inputs, main_packages, fold, logger, generation, question_ids):
    """Replay one full frozen RandomPair plan, then return its log and packages.

    The complete blind action plan is constructed before the manager obtains
    reward or cost arrays. Learning code must receive only selected observations
    returned by select_prefix/summarize_prefix, not this manager's dense cache.
    """
    generation = _integer(generation, "generation")
    generation_index = GENERATION_SEEDS.index(generation) if generation in GENERATION_SEEDS else generation
    if not 0 <= generation_index < 3:
        raise ValueError("Generation must be an original seed or its index")
    history = [int(q) for q in inputs["H"]]
    evaluation = {int(q) for q in inputs["E"]}
    questions = [int(package["question_index"]) for package in main_packages]
    if len(set(questions)) != len(questions) or set(questions) != set(history) or set(history) & evaluation:
        raise ValueError("RandomPair must preserve the complete original historical question traversal")
    plan_rows = []
    for package in main_packages:
        question = int(package["question_index"])
        main = _integer(package["main_action"], "main action")
        second = choose_random_second(question_ids[question], fold, logger, main)
        plan_rows.append({"question_index": question, "main_action": main, "second_action": second})
    plan = {"scenario": "RandomPair", "fold": int(fold), "logger_seed": int(logger),
            "hash_prefix": RANDOM_SECOND_HASH_PREFIX, "action_rows": plan_rows,
            "generation_independent_actions": True, "uses_costs_or_rewards": False,
            "eligible_second_actions": "all other19 actions, including the9 single-member neighbors"}
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manager = _RandomPairCacheManager(train, history, evaluation, generation_index)
    for row in plan_rows:
        manager.execute(row["question_index"], row["main_action"], row["second_action"])
    log = {"scenario": "RandomPair", "plan": plan,
           "plan_frozen_before_reward_cost_reads": True,
           "access_log": manager.access_log, "execution_log": manager.execution_log,
           "final_summary": manager.summary()}
    packages = execution_packages(log, scenario="RandomPair")
    log["complete_plan_pool_cost"] = deepcopy(log["final_summary"]["ledger"])
    return {"plan": plan, "log": log, "packages": packages}
