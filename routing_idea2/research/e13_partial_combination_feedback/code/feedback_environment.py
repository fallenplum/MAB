"""Real sparse combination feedback with execution reuse and an explicit ledger.

Only the cache manager owns dense outcomes. Learners receive purchased tuples.
Each account uses one generation repeat and only its allowed historical pool.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np


DEFAULT_ASSET_ROOT = Path(r"E:\语言模型组合路由")
E3_ASSETS_SHA256 = "64f01de3e2e007f50c8ed6e5f690ffd147f6354eed808d83fce1f0fc5892cf64"
REFERENCE_VERSION = "safe_exact_v2_0_terminal_final_heading_fix"
_E3 = None


def _load_e3(asset_root=DEFAULT_ASSET_ROOT):
    """Load the old execution primitives without opening any dataset payload."""
    global _E3
    path = Path(asset_root) / "research/decision_acquisition_e3/public/assets.py"
    if _E3 is None or Path(_E3.__file__) != path:
        if hashlib.sha256(path.read_bytes()).hexdigest() != E3_ASSETS_SHA256:
            raise ValueError("Frozen original E3 execution primitives changed")
        spec = importlib.util.spec_from_file_location("e13_original_e3_assets", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _E3 = module
    return _E3


def source_identity(asset_root=DEFAULT_ASSET_ROOT):
    source = _load_e3(asset_root)
    return {"path": str(Path(source.__file__)), "sha256": source.sha256(source.__file__),
            "execution_primitives": ["TEAMS", "SEEDS", "zero_resources", "add_resources", "Env._new_executions"],
            "raw_cache_loader": "load_assets reads selector_train only"}


def load_train600(asset_root=DEFAULT_ASSET_ROOT):
    """Reuse the original source hashes, final-score provenance, axes and bounds."""
    return _load_e3(asset_root).load_assets(Path(asset_root))


def make_logger_plan(question_indices, question_ids, logger_seed, fold=0,
                     asset_root=DEFAULT_ASSET_ROOT):
    """Reward/cost-blind plan, invariant to budget and question traversal order.

    The same plan is reused across all three generation-repeat accounts. The
    logger is unrelated to a learner's eventual evaluation-time service action.
    """
    source = _load_e3(asset_root)
    indices = np.asarray(question_indices, dtype=int)
    ids = np.asarray(question_ids, dtype=str)
    if indices.ndim != 1 or len(indices) != len(set(indices.tolist())):
        raise ValueError("Logger question indices must be unique")
    if ids.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(ids)):
        raise ValueError("Logger IDs do not cover the requested indices")
    main, neighbor = [], []
    for question in indices:
        text = f"E13-logger-v1|{int(logger_seed)}|{int(fold)}|{ids[question]}"
        entropy = np.frombuffer(hashlib.sha256(text.encode("utf-8")).digest(), dtype="<u4")
        rng = np.random.default_rng(np.random.SeedSequence(entropy.tolist()))
        action = int(rng.integers(20))
        nearby = [candidate for candidate, team in enumerate(source.TEAMS)
                  if len(set(team) ^ set(source.TEAMS[action])) == 2]
        if len(nearby) != 9:
            raise ValueError("Frozen 6-choose-3 action order has the wrong neighbor count")
        main.append(action)
        neighbor.append(int(nearby[int(rng.integers(len(nearby)))]))
    return {"question_indices": indices, "main_actions": np.asarray(main, dtype=int),
            "neighbor_actions": np.asarray(neighbor, dtype=int),
            "logger_seed": int(logger_seed), "fold": int(fold),
            "definition": "one uniform main and one uniform single-replacement neighbor per question; shared across generations"}


def _detail_zero(source):
    result = source.zero_resources()
    for stem in ("member", "aggregator"):
        for suffix in ("input_tokens", "output_tokens", "source_wall_s"):
            result[f"{stem}_{suffix}"] = 0
    return result


def _sum_resources(left, right):
    return {name: left[name] + right[name] for name in left}


class PartialFeedbackEnv:
    """One generation and historical pool, with no reward-bearing query input.

    Model code must receive only ``observations()`` and declared pool inputs.
    Hidden costs are read only after the frozen execution action is accepted.
    Optional hard limits reserve public worst-case token bounds before reads.
    """

    def __init__(self, train, allowed_questions, generation_index, *,
                 forbidden_questions=(), reference_version=REFERENCE_VERSION,
                 limits=None, asset_root=DEFAULT_ASSET_ROOT):
        self._source = _load_e3(asset_root)
        self.__rewards = train["Y"]
        self.__costs = train["cost"]
        self.__canonical = train["canonical"]
        self.allowed_questions = frozenset(map(int, allowed_questions))
        self.forbidden_questions = frozenset(map(int, forbidden_questions))
        self.generation_index = int(generation_index)
        self.reference_version = str(reference_version)
        if (not self.allowed_questions or self.allowed_questions & self.forbidden_questions
                or not 0 <= self.generation_index < len(self._source.SEEDS)):
            raise ValueError("Invalid historical feedback scope or generation repeat")
        if max(self.allowed_questions) >= len(self.__canonical) or min(self.allowed_questions) < 0:
            raise ValueError("Historical feedback scope is outside the verified cache")
        self._limits = dict(limits or {})
        allowed_limits = {"gold", "member_calls", "aggregator_calls", "input_tokens", "output_tokens", "tokens", "scoring_ops"}
        if not set(self._limits) <= allowed_limits or any(value < 0 for value in self._limits.values()):
            raise ValueError("Unknown or negative feedback resource limit")
        self._served_main = {}
        self._execution = {}
        self._gold_keys = set()
        self._scored = set()
        self._observations = []
        self._known_costs = {}
        self._ledger = {key: _detail_zero(self._source) for key in ("logger_service", "feedback", "total")}
        self.access_log = []
        self.execution_log = []

    def _check_question(self, question):
        question = int(question)
        if question not in self.allowed_questions or question in self.forbidden_questions:
            raise ValueError("Only historical purchase-pool questions may be executed or revealed")
        return question

    def _state(self, question):
        return self._execution.get(question, {"members": set(), "aggregators": set()})

    def _reservation(self, member_count, aggregate_count, gold=0, scoring=0):
        inputs = member_count * self._source.MEMBER_INPUT_CAP + aggregate_count * self._source.AGGREGATOR_INPUT_CAP
        outputs = (member_count + aggregate_count) * self._source.OUTPUT_CAP
        return {"member_calls": member_count, "aggregator_calls": aggregate_count,
                "gold": gold, "scoring_ops": scoring, "input_tokens": inputs,
                "output_tokens": outputs, "tokens": inputs + outputs}

    def _budget_check(self, reservation):
        for key, cap in self._limits.items():
            if self._ledger["total"][key] + reservation[key] > cap:
                raise ValueError("Frozen execution cannot fit public worst-case resource reservation: " + key)

    def _read_selected_costs(self, question, members, aggregates):
        resources = _detail_zero(self._source)
        reads = []
        cells = {}
        for stem, selected, count_key, detail_stem in (
            ("member", sorted(members), "member_calls", "member"),
            ("aggregate", sorted(aggregates), "aggregator_calls", "aggregator"),
        ):
            if not selected:
                continue
            for suffix, result_key, detail_key in (
                ("execution_counts", count_key, None),
                ("prompt_tokens", "input_tokens", f"{detail_stem}_input_tokens"),
                ("completion_tokens", "output_tokens", f"{detail_stem}_output_tokens"),
                ("wall_time_s", "source_model_wall_s", f"{detail_stem}_source_wall_s"),
            ):
                values = np.asarray(self.__costs[f"{stem}_{suffix}"][self.generation_index, question, selected])
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("Executed source cost is missing or invalid")
                amount = float(values.sum()) if suffix == "wall_time_s" else int(values.sum())
                resources[result_key] += amount
                if detail_key:
                    resources[detail_key] += amount
                for column, value in zip(selected, values):
                    cells[(question, self.generation_index, stem, suffix, column)] = float(value)
                reads.append({"kind": "cost", "stem": stem, "field": suffix, "columns": selected})
        resources["tokens"] = resources["input_tokens"] + resources["output_tokens"]
        return resources, reads, cells

    def _commit(self, question, phase, resources, members, aggregates, reads, cells):
        self._ledger[phase] = _sum_resources(self._ledger[phase], resources)
        self._ledger["total"] = _sum_resources(self._ledger["total"], resources)
        state = self._state(question)
        self._execution[question] = {"members": set(state["members"]) | set(members),
                                     "aggregators": set(state["aggregators"]) | set(aggregates)}
        self._known_costs.update(cells)
        for record in reads:
            self.access_log.append({"question_index": question, "generation_index": self.generation_index,
                                    "phase": phase, **record})

    def _execution_keys(self, question, members, aggregates):
        return {
            "member_keys": [(question, self.generation_index, int(member)) for member in sorted(members)],
            "aggregate_keys": [(question, self.generation_index, int(action)) for action in sorted(aggregates)],
        }

    def _standalone_tokens(self, question, action, pending_cost_cells):
        """Use only cost cells already paid or in this accepted execution."""
        total = 0.0
        for stem, columns in (("member", self._source.TEAMS[action]), ("aggregate", [action])):
            for suffix in ("prompt_tokens", "completion_tokens"):
                for column in columns:
                    key = (question, self.generation_index, stem, suffix, int(column))
                    if key in pending_cost_cells:
                        total += pending_cost_cells[key]
                    elif key in self._known_costs:
                        total += self._known_costs[key]
                    else:
                        raise ValueError("Standalone tuple cost requires executed members and aggregator")
        return int(total)

    def serve(self, question, main_action):
        """Freeze and charge logger service. It exposes no correctness value."""
        question = self._check_question(question)
        main_action = int(main_action)
        if question in self._served_main:
            if self._served_main[question] != main_action:
                raise ValueError("The logger's original service decision is already frozen")
            return _detail_zero(self._source)
        _, members, aggregates = self._source.Env._new_executions([main_action], self._state(question))
        self._budget_check(self._reservation(len(members), len(aggregates)))
        resources, reads, cells = self._read_selected_costs(question, members, aggregates)
        self._commit(question, "logger_service", resources, members, aggregates, reads, cells)
        self._served_main[question] = main_action
        self.execution_log.append({"question_index": question, "main_action": main_action,
                                   "generation_index": self.generation_index, "phase": "logger_service",
                                   "resources": dict(resources),
                                   "cumulative": dict(self._ledger["total"]),
                                   **self._execution_keys(question, members, aggregates)})
        return dict(resources)

    def buy(self, question, actions):
        """Reveal only executed main/main-plus-neighbor scores, once per tuple."""
        question = self._check_question(question)
        if question not in self._served_main:
            raise ValueError("Feedback requires a previously frozen logger service")
        actions, members, aggregates = self._source.Env._new_executions(actions, self._state(question))
        main = self._served_main[question]
        if main not in actions or len(actions) > 2:
            raise ValueError("A feedback package must be main or main plus one neighbor")
        for action in actions:
            if action != main and len(set(self._source.TEAMS[action]) ^ set(self._source.TEAMS[main])) != 2:
                raise ValueError("Comparison action must replace exactly one member")
        gold_key = (str(self.__canonical[question]), self.reference_version)
        gold = int(gold_key not in self._gold_keys)
        observed = [a for a in actions if (question, self.generation_index, a) not in self._scored]
        self._budget_check(self._reservation(len(members), len(aggregates), gold, len(observed)))
        resources, reads, cells = self._read_selected_costs(question, members, aggregates)
        resources["gold"], resources["scoring_ops"] = gold, len(observed)
        values = np.asarray(self.__rewards[question, self.generation_index, observed], dtype=float)
        if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
            raise ValueError("Purchased true final aggregate reward is missing")
        pending = [{"question_index": question, "generation_index": self.generation_index,
                    "generation_seed": int(self._source.SEEDS[self.generation_index]),
                    "action": int(action), "reward": float(value),
                    "standalone_tokens": self._standalone_tokens(question, action, cells)}
                   for action, value in zip(observed, values)]
        if observed:
            reads.append({"kind": "reward", "actions": observed})
        self._commit(question, "feedback", resources, members, aggregates, reads, cells)
        self._gold_keys.add(gold_key)
        self._scored.update((question, self.generation_index, action) for action in observed)
        self._observations.extend(pending)
        self.execution_log.append({"question_index": question, "actions": list(actions),
                                   "generation_index": self.generation_index, "phase": "feedback",
                                   "resources": dict(resources), "observations": pending,
                                   "gold_reference_keys": [gold_key] if gold else [],
                                   "cumulative": dict(self._ledger["total"]),
                                   **self._execution_keys(question, members, aggregates)})
        return {"observations": [dict(record) for record in pending], "resources": dict(resources)}

    def observations(self):
        """Return a sparse tuple list. Unobserved actions have no label row."""
        return [dict(record) for record in self._observations]

    def training_packet(self):
        """Batched observed rows for adapters, with no unobserved reward slots."""
        return {
            "query_indices": np.asarray([row["question_index"] for row in self._observations], dtype=int),
            "action_indices": np.asarray([row["action"] for row in self._observations], dtype=int),
            "rewards": np.asarray([row["reward"] for row in self._observations], dtype=float),
            "standalone_tokens": np.asarray([row["standalone_tokens"] for row in self._observations], dtype=float),
            "generation_index": np.asarray([row["generation_index"] for row in self._observations], dtype=int),
        }

    def summary(self):
        return {"ledger": {key: dict(value) for key, value in self._ledger.items()},
                "generation_index": self.generation_index,
                "generation_seed": int(self._source.SEEDS[self.generation_index]),
                "reference_version": self.reference_version,
                "gold_revealed": len(self._gold_keys), "gold_keys": sorted(self._gold_keys),
                "served_questions": len(self._served_main), "score_observations": len(self._scored),
                "member_execution_cache_entries": sum(len(state["members"]) for state in self._execution.values()),
                "aggregate_execution_cache_entries": sum(len(state["aggregators"]) for state in self._execution.values()),
                "member_execution_keys": [(question, self.generation_index, member)
                                          for question, state in sorted(self._execution.items()) for member in sorted(state["members"])],
                "aggregate_execution_keys": [(question, self.generation_index, action)
                                             for question, state in sorted(self._execution.items()) for action in sorted(state["aggregators"])],
                "new_physical_model_calls": 0, "new_physical_gold_acquisition_calls": 0,
                "gpu_time_s": None, "gpu_time_note": "Existing source-call wall sums are proxies; no new GPU inference",
                "resource_limits": dict(self._limits),
                "account_scope": "single generation-repeat historical logger; evaluation pool excluded"}
