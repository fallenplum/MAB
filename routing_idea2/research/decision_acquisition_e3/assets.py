"""Verified train600 assets and permission-gated, fully charged CPU replay."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ASSET_ROOT = Path(r"E:\语言模型组合路由")
SEEDS = (42, 137, 2024)
ORDER_SEEDS = (17, 29, 43)
TEAMS = tuple(itertools.combinations(range(6), 3))
OPERATOR_FP = "9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e"
BUNDLE_SHA = "15095d4cf8b4f2d564b62f814102a63fe05a9a9e5c9702e14c9b8232b80649db"
FOLDS_SHA = "6b433dcd2d8e6cd9938d2dd2d97261e18c3ae3d18325d7a04c6590e4d0600d50"
SCORER = "safe_exact_v2_0_terminal_final_heading_fix"
MEMBER_INPUT_CAP = 4096
AGGREGATOR_INPUT_CAP = 6144
OUTPUT_CAP = 2048
RESOURCE_KEYS = (
    "member_calls", "aggregator_calls", "gold", "scoring_ops",
    "input_tokens", "output_tokens", "tokens", "source_model_wall_s",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def zero_resources():
    return {name: 0 for name in RESOURCE_KEYS}


def add_resources(left, right):
    return {name: left[name] + right[name] for name in RESOURCE_KEYS}


def priced_units(resources):
    return (resources["gold"] + resources["member_calls"]
            + resources["aggregator_calls"] + resources["tokens"] / 1000
            + resources["scoring_ops"] * 0.001)


def _verified_reference(record, root):
    path = Path(record["path"])
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    require(path.is_relative_to(root.resolve()), "Asset reference escaped asset root")
    require(sha256(path) == record["sha256"], "Asset SHA mismatch: " + path.name)
    return path


def load_assets(root=ASSET_ROOT):
    """Read only historical train600, never validation/fresh/sealed payloads."""
    root = Path(root)
    bundle_path = root / "artifacts/private/routing_redesign_20261104/ff_v1/BUNDLE_TRAINVAL.json"
    require(sha256(bundle_path) == BUNDLE_SHA, "Frozen bundle changed")
    bundle = read_json(bundle_path)
    require(bundle["schema"] == "routing_full_feedback_bundle_v2"
            and bundle["N"] == 6 and bundle["K"] == 3
            and tuple(bundle["generation_seeds"]) == SEEDS, "Wrong bundle identity")
    require(bundle["scorer"] == {"version": SCORER, "no_model_calls": True}, "Wrong scorer")
    operator = bundle["aggregator_operator"]
    require(operator["model_id"] == "Qwen/Qwen3-8B"
            and operator["fingerprint"] == OPERATOR_FP, "Wrong aggregator operator")
    refs = {"bundle": {"path": str(bundle_path), "sha256": BUNDLE_SHA}}
    for name, record in (("member_pool", bundle["member_pool"]),
                         ("operator_freeze", operator["freeze_evidence"])):
        path = _verified_reference(record, root)
        refs[name] = {"path": str(path), "sha256": record["sha256"]}
    archives = {}
    for group in ("inputs", "outcomes", "matrix_receipts"):
        record = bundle[group]["selector_train"]
        path = _verified_reference(record, root)
        refs[group] = {"path": str(path), "sha256": record["sha256"]}
        if group == "matrix_receipts":
            receipt = read_json(path)
        else:
            with np.load(path, allow_pickle=False) as archive:
                archives[group] = {name: archive[name].copy() for name in archive.files}
    require(receipt["status"] == "REAL_FULL_INFORMATION_READY"
            and receipt["software_test_only"] is False
            and receipt["questions"] == 600
            and receipt["aggregator_fingerprint"] == OPERATOR_FP
            and receipt["all_combinations_executed"] is True
            and receipt["real_frozen_llm_aggregation"] is True
            and receipt["aggregate_scores_from_executed_aggregator"] is True
            and receipt["failure_costs_included"] is True
            and receipt["scorer_version"] == SCORER, "Incomplete real matrix receipt")
    inputs, outcomes = archives["inputs"], archives["outcomes"]
    for name in ("question_ids", "canonical_hashes"):
        require(np.array_equal(inputs[name], outcomes[name]), "Matrix question order differs")
        require(len(set(inputs[name])) == 600, "Duplicate question identity")
    require(np.array_equal(outcomes["generation_seeds"], SEEDS), "Generation seed mismatch")
    rewards = outcomes["aggregate_scores"].transpose(1, 0, 2).astype(float)
    require(rewards.shape == (600, 3, 20) and np.isin(rewards, [0, 1]).all(), "Bad true rewards")
    require(inputs["embeddings"].shape == (600, 384)
            and np.isfinite(inputs["embeddings"]).all(), "Bad frozen embeddings")
    tasks, subtasks = inputs["task_labels"], inputs["subtask_labels"]
    require(set(tasks) == {"gsm8k", "bbh"}
            and np.sum(tasks == "gsm8k") == np.sum(tasks == "bbh") == 300
            and len(set(subtasks[tasks == "bbh"])) == 27, "Wrong semantic task coverage")
    costs = {name: value for name, value in outcomes.items()
             if name.startswith(("member_", "aggregate_"))}
    for stem, width, prompt_cap in (("member", 6, MEMBER_INPUT_CAP),
                                     ("aggregate", 20, AGGREGATOR_INPUT_CAP)):
        for suffix in ("execution_counts", "prompt_tokens", "completion_tokens", "wall_time_s"):
            values = costs[stem + "_" + suffix]
            require(values.shape == (3, 600, width)
                    and np.isfinite(values).all() and (values >= 0).all(), "Invalid resource cells")
        require(np.all(costs[stem + "_execution_counts"] == 1), "Unexpected retry count needs new caps")
        require(np.all(costs[stem + "_prompt_tokens"] <= prompt_cap)
                and np.all(costs[stem + "_completion_tokens"] <= OUTPUT_CAP), "Public token bound exceeded")
    train = {"E": inputs["embeddings"].astype(float), "Y": rewards,
             "task": tasks, "sub": subtasks, "ids": inputs["question_ids"],
             "canonical": inputs["canonical_hashes"], "cost": costs,
             "asset_root": root}
    for value in train.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    for value in costs.values():
        value.flags.writeable = False
    refs["identity"] = {"operator_fingerprint": OPERATOR_FP, "scorer": SCORER,
                        "generation_seeds": list(SEEDS), "new_llm_calls": 0,
                        "decoded_payload_splits": ["selector_train"]}
    return train, refs


def algorithm_seed(fold, order_seed, generation_index=0):
    """Independent policy RNG, distinct from the historical fold-generation seed."""
    return int(np.random.SeedSequence([20261107, int(fold), int(order_seed),
                                     int(generation_index), 919]).generate_state(1)[0])


def make_splits(train):
    """Reuse verified five folds, with nested task-balanced history and 54 references."""
    fold_path = Path(train.get("asset_root", ASSET_ROOT)) / "research/conditional_gain_20261106/private/FOLDS_PRIVATE.npz"
    require(sha256(fold_path) == FOLDS_SHA, "Historical fold artifact changed")
    folds, held_out = [], []
    with np.load(fold_path, allow_pickle=False) as archive:
        for fold in range(5):
            history = archive[f"fold{fold}_train"].astype(int)
            stream = archive[f"fold{fold}_holdout"].astype(int)
            require(len(history) == 480 and len(stream) == 120
                    and len(set(history) | set(stream)) == 600
                    and not set(history) & set(stream), "Invalid historical fold")
            held_out.extend(stream.tolist())
            require(np.sum(train["task"][stream] == "gsm8k") == 60
                    and np.sum(train["task"][stream] == "bbh") == 60, "Unbalanced stream fold")
            rng = np.random.default_rng(np.random.SeedSequence([20261107, fold, 120240]))
            gsm = rng.permutation(history[train["task"][history] == "gsm8k"])
            names = sorted(set(train["sub"][history][train["task"][history] == "bbh"]))
            require(len(names) == 27, "Historical fold lacks a BBH subtask")
            names = [names[j] for j in rng.permutation(len(names))]
            queues = [rng.permutation(history[(train["task"][history] == "bbh")
                                              & (train["sub"][history] == name)]) for name in names]
            bbh = np.array([queue[position] for position in range(max(map(len, queues)))
                            for queue in queues if position < len(queue)], dtype=int)
            nested = {size: np.sort(np.concatenate((gsm[:size // 2], bbh[:size // 2])))
                      for size in (120, 240)}
            nested[480] = history.copy()
            require(set(nested[120]) <= set(nested[240]) <= set(history), "History is not nested")
            smallest = nested[120]
            reference_rng = np.random.default_rng(np.random.SeedSequence([20261107, fold, 54]))
            reference_gsm = reference_rng.choice(smallest[train["task"][smallest] == "gsm8k"], 27, replace=False)
            reference_bbh = [int(reference_rng.choice(smallest[(train["task"][smallest] == "bbh")
                                                              & (train["sub"][smallest] == name)]))
                             for name in sorted(names)]
            references = np.concatenate((reference_gsm, reference_bbh)).astype(int)
            orders = {seed: np.random.default_rng(np.random.SeedSequence([seed, fold, 731]))
                      .permutation(stream) for seed in ORDER_SEEDS}
            folds.append({"fold": fold, "history": history, "stream": stream,
                          "nested": nested, "orders": orders,
                          "references": {size: references.copy() for size in (120, 240, 480)},
                          "reference_weights": np.full(54, 1 / 54),
                          "algorithm_seed": 20261107 + 1009 * fold})
    require(len(held_out) == len(set(held_out)) == 600, "OOF questions are not a partition")
    return {"folds": folds, "metadata": {
        "fold_source": str(fold_path), "fold_source_sha256": FOLDS_SHA,
        "fold_original_seed": 20261106, "fold_original_seed_is_new_algorithm_seed": False,
        "nested_seed": 20261107, "order_seeds": list(ORDER_SEEDS),
        "algorithm_seed_rule": "SeedSequence([20261107,fold,order_seed,generation_index,919])",
        "reference_count": 54, "reference_gsm": 27, "reference_bbh_subtasks": 27,
        "reference_weights": "GSM half, BBH27-subtask macro half",
        "query_each_question_exactly_once": True}}


class Env:
    """One fold/history/order/generation/policy account. Learners receive no evaluator state."""

    def __init__(self, train, history, stream_order, gen_index, gold_budget, *, diagnostic=False):
        self._train = train
        self.history = np.asarray(history, dtype=int)
        self.stream = np.asarray(stream_order, dtype=int)
        self.g = int(gen_index)
        self.gold_budget = int(gold_budget)
        self.diagnostic = bool(diagnostic)
        require(0 <= self.g < 3 and self.gold_budget >= 0
                and len(set(self.stream)) == len(self.stream)
                and not set(self.history) & set(self.stream), "Invalid replay account")
        self._means = {name: values[:, self.history].mean(axis=(0, 1))
                       for name, values in train["cost"].items()}
        self._current_q = None
        self._current_main = None
        self._position = 0
        self._executed = {"members": set(), "aggregators": set()}
        self._gold = set()
        self._scored = set()
        self.evaluation = []
        self.ledger = {"service": zero_resources(), "audit": zero_resources(), "total": zero_resources()}
        self.audit_skips = []
        audit_members = (3 if diagnostic else 1) * self.gold_budget
        audit_aggregators = (19 if diagnostic else 1) * self.gold_budget
        self.service_caps = self._caps(3 * len(self.stream), len(self.stream))
        self.audit_caps = self._caps(audit_members, audit_aggregators)
        self._served_mains = {}

    @staticmethod
    def _caps(members, aggregates):
        inputs = members * MEMBER_INPUT_CAP + aggregates * AGGREGATOR_INPUT_CAP
        outputs = (members + aggregates) * OUTPUT_CAP
        return {"member_calls": members, "aggregator_calls": aggregates,
                "input_tokens": inputs, "output_tokens": outputs, "tokens": inputs + outputs}

    def executed_current(self):
        return {name: set(values) for name, values in self._executed.items()}

    @staticmethod
    def _new_executions(actions, executed):
        actions = tuple(sorted(set(map(int, actions))))
        require(actions and all(0 <= action < 20 for action in actions), "Invalid combination actions")
        members = set().union(*(set(TEAMS[action]) for action in actions)) - set(executed["members"])
        aggregates = set(actions) - set(executed["aggregators"])
        return actions, members, aggregates

    def get_actual(self, q, g, actions, executed):
        """Environment-internal current-action lookup, never exposed to the planner."""
        require(q == self._current_q and g == self.g, "Actual costs are private to current execution")
        _, members, aggregates = self._new_executions(actions, executed)
        result = zero_resources()
        for stem, selected, call_key in (("member", members, "member_calls"),
                                          ("aggregate", aggregates, "aggregator_calls")):
            indices = sorted(selected)
            for suffix, resource in (("execution_counts", call_key), ("prompt_tokens", "input_tokens"),
                                     ("completion_tokens", "output_tokens"), ("wall_time_s", "source_model_wall_s")):
                value = self._train["cost"][stem + "_" + suffix][g, q, indices].sum()
                result[resource] += float(value) if suffix == "wall_time_s" else int(value)
        result["tokens"] = result["input_tokens"] + result["output_tokens"]
        return result

    def predict_incremental(self, actions, executed, *, include_gold=True):
        """Historical per-member/per-combination means only, with no hidden query lookup."""
        actions, members, aggregates = self._new_executions(actions, executed)
        result = zero_resources()
        for stem, selected, call_key in (("member", members, "member_calls"),
                                          ("aggregate", aggregates, "aggregator_calls")):
            indices = sorted(selected)
            for suffix, resource in (("execution_counts", call_key), ("prompt_tokens", "input_tokens"),
                                     ("completion_tokens", "output_tokens"), ("wall_time_s", "source_model_wall_s")):
                result[resource] += float(self._means[stem + "_" + suffix][indices].sum())
        result["tokens"] = result["input_tokens"] + result["output_tokens"]
        result["gold"] = int(include_gold)
        result["scoring_ops"] = len(actions)
        result["priced_units"] = priced_units(result)
        return result

    def _commit(self, account, resources):
        updated = add_resources(self.ledger[account], resources)
        caps = self.service_caps if account == "service" else self.audit_caps
        require(all(updated[name] <= bound for name, bound in caps.items()), "Resource cap exceeded")
        self.ledger[account] = updated
        self.ledger["total"] = add_resources(self.ledger["total"], resources)

    def service(self, q, main):
        """Freeze current service, execute its three members/aggregator, return resources only."""
        q, main = int(q), int(main)
        if q in self._served_mains:
            require(q == self._current_q and self._served_mains[q] == main, "Service decision already frozen")
            return zero_resources()
        require(self._position < len(self.stream) and q == self.stream[self._position]
                and 0 <= main < 20, "Service must follow the frozen stream order")
        self._current_q, self._current_main = q, main
        self._executed = {"members": set(), "aggregators": set()}
        resources = self.get_actual(q, self.g, [main], self._executed)
        self._commit("service", resources)
        self._executed = {"members": set(TEAMS[main]), "aggregators": {main}}
        self._served_mains[q] = main
        self._position += 1
        self.evaluation.append({"question_index": q, "generation_index": self.g,
                                "generation_seed": SEEDS[self.g], "main": main,
                                "quality": float(self._train["Y"][q, self.g, main])})
        return resources

    def buy(self, q, actions, *, diagnostic=False):
        """Reveal only purchased original scores after service, with shared gold and execution reuse."""
        q = int(q)
        require(q == self._current_q and q in self._served_mains, "Feedback requires frozen current service")
        actions, members, aggregates = self._new_executions(actions, self._executed)
        require(self._current_main in actions, "Feedback bundle must include frozen service action")
        if diagnostic:
            require(self.diagnostic and set(actions) == set(range(20)), "Full information is diagnostic-only")
        else:
            require(not self.diagnostic and len(actions) <= 2, "Primary feedback supports main plus one neighbor")
            require(all(len(set(TEAMS[action]) ^ set(TEAMS[self._current_main])) == 2
                        for action in actions if action != self._current_main), "Alternate must replace one member")
        canonical = str(self._train["canonical"][q])
        gold = int(canonical not in self._gold)
        observed = [action for action in actions if (q, self.g, action) not in self._scored]
        reservation = self._caps(len(members), len(aggregates))
        enough_gold = len(self._gold) + gold <= self.gold_budget
        enough_audit = all(self.ledger["audit"][name] + value <= self.audit_caps[name]
                           for name, value in reservation.items())
        future = self._caps(3 * (len(self.stream) - self._position), len(self.stream) - self._position)
        total_caps = {name: self.service_caps[name] + self.audit_caps[name] for name in future}
        protected = all(self.ledger["total"][name] + reservation[name] + future[name] <= total_caps[name]
                        for name in future)
        if not (enough_gold and enough_audit and protected):
            reason = "gold_cap" if not enough_gold else "audit_cap" if not enough_audit else "remaining_service_reserve"
            self.audit_skips.append({"question_index": q, "actions": list(actions), "reason": reason})
            return {"accepted": False, "observations": [], "resources": zero_resources(), "reason": reason}
        resources = self.get_actual(q, self.g, actions, self._executed)
        resources["gold"] = gold
        resources["scoring_ops"] = len(observed)
        self._commit("audit", resources)
        self._executed["members"].update(members)
        self._executed["aggregators"].update(aggregates)
        self._gold.add(canonical)
        self._scored.update((q, self.g, action) for action in observed)
        return {"accepted": True, "observations": [(action, float(self._train["Y"][q, self.g, action]))
                                                     for action in observed],
                "resources": resources, "reason": None}

    def summary(self):
        """Serializable full account. Source wall sums are proxies, not measured GPU kernel time."""
        return {"ledger": self.ledger, "service_caps": self.service_caps, "audit_caps": self.audit_caps,
                "gold_budget": self.gold_budget, "gold_revealed": len(self._gold),
                "served_questions": self._position, "stream_questions": len(self.stream),
                "original_score_observations": len(self._scored), "audit_skips": self.audit_skips,
                "priced_units": priced_units(self.ledger["total"]),
                "new_physical_model_calls": 0, "new_physical_gold_acquisition_calls": 0,
                "gpu_time_s": None, "gpu_time_note": "Only source call wall sums are available",
                "physical_preparation_is_separate": True, "diagnostic": self.diagnostic}
