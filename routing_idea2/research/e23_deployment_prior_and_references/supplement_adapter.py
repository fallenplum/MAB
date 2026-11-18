"""An additive E23 scope around the original P2 scientific execution methods."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import FunctionType


CODE = Path(__file__).resolve().parent
ORIGINAL_ADAPTER_SHA = "4cbba10269aea9b8bfcc84dbc6a3f709244e447a6a91d1b9a76ec9ba9cfeeff2"
PARENT_SCOPE_SHA = "b355581aaaab1a581e76dab35107a897948da7b4086a1d9e2a33be4a3e1cd1b9"
COHORT = "e23_reference_supplement"
SCOPE_SCHEMA = "e23_supplement_scope_v1"
APPROVAL_SCHEMA = "e23_supplement_approval_v1"
REQUIRED_BINDINGS = frozenset({
    "protocol_sha256", "target_inputs_sha256", "frozen_actions_sha256",
    "budget_sha256", "target_ids_sha256", "parent_scope_sha256", "cache_evidence_sha256",
})


def _original_adapter():
    candidates = (
        CODE / "p2_original_execution_adapter.py",
        CODE.parent / "e20_p2_execution/p2_execution_adapter.py",
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None or hashlib.sha256(source.read_bytes()).hexdigest() != ORIGINAL_ADAPTER_SHA:
        raise ValueError("The byte-pinned original P2 scientific adapter is required")
    name = "_e23_original_p2_execution_adapter"
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != source.resolve():
            raise ValueError("The original scientific adapter module path differs")
        return module
    sys.path.insert(0, str(source.parent))
    specification = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


original = _original_adapter()
OPERATOR_FP = original.OPERATOR_FP
GENERATION_SEEDS = original.GENERATION_SEEDS
TEAMS = original.TEAMS
TRANSPORT_SPLIT = original.TRANSPORT_SPLIT
OPERATIONS = original.OPERATIONS
canonical_digest = original.canonical_digest
read_json = original.read_json
load_original_apis = original.load_original_apis
mapped_backend_factory = original.mapped_backend_factory
original_backend_factory = original.original_backend_factory


def _keys(values, maximum, question_ids):
    result = []
    for value in values:
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ValueError("An exact supplementary question, generation and index key is required")
        question, seed, index = value
        if (question not in question_ids or type(seed) is not int or seed not in GENERATION_SEEDS
                or type(index) is not int or not 0 <= index < maximum):
            raise ValueError("A supplementary execution key has an invalid identity")
        result.append((question, seed, index))
    if len(result) != len(set(result)):
        raise ValueError("A supplementary execution key was repeated")
    return set(result)


def build_scope_contract(question_records, policy_probabilities, bindings,
                         member_execution_keys, aggregate_execution_keys):
    records = list(question_records)
    ids = [row["question_id"] for row in records]
    canonical = [row["canonical_question_sha256"] for row in records]
    if (not records or any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != len(ids) or len(set(canonical)) != len(canonical)):
        raise ValueError("Unique original supplementary question identities are required")
    if (not REQUIRED_BINDINGS <= set(bindings)
            or not all(original.valid_sha(value) for value in bindings.values())
            or bindings["parent_scope_sha256"] != PARENT_SCOPE_SHA):
        raise ValueError("Complete E23 protocol, parent scope, cache and policy bindings are required")
    if bindings["target_ids_sha256"] != canonical_digest([
            [question, hashed] for question, hashed in zip(ids, canonical)]):
        raise ValueError("Supplementary target identity binding differs")
    if not isinstance(policy_probabilities, dict) or not policy_probabilities:
        raise ValueError("A named frozen policy set is required")
    probabilities = {}
    for name, policy in policy_probabilities.items():
        if not isinstance(name, str) or not name or len(policy) != len(records):
            raise ValueError("A complete named policy on the frozen question pool is required")
        converted = [[float(value) for value in row] for row in policy]
        if any(len(row) != 20 or any(not math.isfinite(value) or value < 0 for value in row)
               or abs(sum(row)-1) > 1e-12 for row in converted):
            raise ValueError("Frozen policy probabilities must be finite normalized 20-action rows")
        probabilities[name] = converted
    support = set()
    for position, question in enumerate(ids):
        for action in range(20):
            if any(policy[position][action] > 0 for policy in probabilities.values()):
                support.update((question, seed, action) for seed in GENERATION_SEEDS)
    aggregate_calls = _keys(aggregate_execution_keys, 20, set(ids))
    if not aggregate_calls <= support:
        raise ValueError("Supplementary aggregate calls exceed the frozen new policy union")
    dependencies = {
        (question, seed, member)
        for question, seed, action in aggregate_calls
        for member in TEAMS[action]
    }
    member_calls = _keys(member_execution_keys, 6, set(ids))
    if not member_calls <= dependencies:
        raise ValueError("Supplementary member calls are not needed by a missing aggregate")
    return dict(
        schema=SCOPE_SCHEMA, role="execution", cohort=COHORT, bindings=dict(bindings),
        transport_split=TRANSPORT_SPLIT, operator_fingerprint=OPERATOR_FP,
        question_keys=[dict(
            question_id=row["question_id"], canonical_question_sha256=row["canonical_question_sha256"],
            question_sha256=hashlib.sha256(row["question"].encode()).hexdigest(),
            task=row["task"], subtask=row.get("subtask") or "",
        ) for row in records],
        frozen_policy_probabilities=probabilities,
        evaluation_union_aggregate_keys=[list(key) for key in sorted(support)],
        required_member_keys=[list(key) for key in sorted(dependencies)],
        required_aggregate_keys=[list(key) for key in sorted(aggregate_calls)],
        cached_member_dependency_keys=[list(key) for key in sorted(dependencies-member_calls)],
        member_execution_keys=[list(key) for key in sorted(member_calls)],
        aggregate_execution_keys=[list(key) for key in sorted(aggregate_calls)],
        execution_limits=dict(member_generation=len(member_calls), aggregation=len(aggregate_calls)),
        generation_seeds=list(GENERATION_SEEDS), aggregation_seed=42,
    )


def validate_scope_approval(approval_ref, scope_contract, operation, *, software_test_only=False):
    if operation not in OPERATIONS or type(software_test_only) is not bool:
        raise ValueError("Unknown supplementary operation or execution mode")
    if ((scope_contract.get("schema"), scope_contract.get("role"), scope_contract.get("cohort"))
            != (SCOPE_SCHEMA, "execution", COHORT)):
        raise ValueError("A separately frozen E23 execution scope is required")
    bindings = scope_contract.get("bindings", {})
    if (not REQUIRED_BINDINGS <= set(bindings)
            or not all(original.valid_sha(value) for value in bindings.values())
            or bindings["parent_scope_sha256"] != PARENT_SCOPE_SHA):
        raise ValueError("Supplementary scope bindings differ from the P2 parent")
    approval = read_json(original.checked_reference(approval_ref))
    if ((approval.get("schema"), approval.get("cohort"), approval.get("status"))
            != (APPROVAL_SCHEMA, COHORT, "APPROVED")):
        raise PermissionError("An explicit hash-bound E23 supplement approval is required")
    if approval.get("software_test_only") is not software_test_only:
        raise PermissionError("The supplementary approval mode differs")
    if operation not in approval.get("operations", []):
        raise PermissionError("The requested supplementary operation is not authorized")
    if (approval.get("scope_sha256_by_role", {}).get("execution") != canonical_digest(scope_contract)
            or approval.get("bindings") != bindings
            or approval.get("execution_limits") != scope_contract.get("execution_limits")):
        raise PermissionError("Supplementary approval scope, bindings or exact call limits differ")
    return approval


class SupplementExecutionAdapter(original.P2ExecutionAdapter):
    """Use original job construction and validation inside a separate paid scope."""

    def __init__(self, apis, scope_contract, question_records, member_specs, aggregator_spec):
        if ((scope_contract.get("schema"), scope_contract.get("role"), scope_contract.get("cohort"))
                != (SCOPE_SCHEMA, "execution", COHORT)):
            raise ValueError("The E23 adapter cannot reinterpret a frozen P2 scope")
        records = list(question_records)
        rebuilt = build_scope_contract(
            records, scope_contract["frozen_policy_probabilities"], scope_contract["bindings"],
            scope_contract["member_execution_keys"], scope_contract["aggregate_execution_keys"])
        if rebuilt != scope_contract:
            raise ValueError("The supplementary frozen key union or input identity changed")
        forbidden = {
            "gold", "reference", "reference_answer", "answer_key", "score", "scores",
            "label", "labels", "reward", "rewards", "answer", "answers", "Y", "target", "targets",
        }
        if any(forbidden & set(row) or row["task"] not in {"gsm8k", "bbh"}
               or original.canonical_question(row["question"]) != row["canonical_question_sha256"]
               for row in records):
            raise ValueError("Supplementary generation inputs must match the frozen input-only cohort")
        self.apis = apis
        self.scope = original.freeze_scope(scope_contract)
        original._SCOPE_DIGESTS[id(self.scope)] = (
            self.scope, canonical_digest(self.scope))
        self.questions = {row["question_id"]: dict(row) for row in records}
        self.question_metadata = {row["question_id"]: row for row in self.scope["question_keys"]}
        self.member_specs = dict(member_specs)
        pool = read_json(apis.root / "configs/pool_v1.json")["models"]
        self.members = {row["index"]: row for row in pool}
        if set(self.member_specs) != set(range(6)) or set(self.members) != set(range(6)):
            raise ValueError("Exactly the six frozen member identities are required")
        for index, spec in self.member_specs.items():
            if (apis.inference.configuration_fingerprint(spec) != self.members[index]["fingerprint"]
                    or spec.temperature != 0.7 or spec.top_p != 0.95
                    or spec.max_new_tokens != 2048 or spec.enable_thinking is not False):
                raise ValueError("Supplementary member generation changed its scientific profile")
        if apis.inference.configuration_fingerprint(aggregator_spec) != original.AGGREGATOR_MODEL_FP:
            raise ValueError("Supplementary aggregation changed the frozen model profile")
        self.aggregator_spec = aggregator_spec
        self.required_members = set(map(tuple, self.scope["required_member_keys"]))
        self.required_aggregates = set(map(tuple, self.scope["required_aggregate_keys"]))
        self.callable_members = set(map(tuple, self.scope["member_execution_keys"]))
        self.callable_aggregates = set(map(tuple, self.scope["aggregate_execution_keys"]))


P2ExecutionAdapter = SupplementExecutionAdapter
_execution_globals = dict(original.run_approved_task.__globals__)
_execution_globals.update(COHORT=COHORT, validate_scope_approval=validate_scope_approval)
_original_execution = FunctionType(
    original.run_approved_task.__code__, _execution_globals,
    name="e23_original_native_execution", argdefs=original.run_approved_task.__defaults__,
)
_original_execution.__kwdefaults__ = original.run_approved_task.__kwdefaults__


def run_approved_task(adapter, task, approval_ref, output_root, backend_factory, **kwargs):
    stage = task.get("stage")
    if stage not in {"member_generation", "aggregation"}:
        raise ValueError("Only the two supplementary execution stages are supported")
    index = task.get("member_index" if stage == "member_generation" else "combination_index")
    key = (task.get("question_id"), task.get("generation_seed"), index)
    callable_keys = adapter.callable_members if stage == "member_generation" else adapter.callable_aggregates
    if key not in callable_keys:
        raise PermissionError("This task was not purchased in the supplementary missing-key manifest")
    return _original_execution(adapter, task, approval_ref, output_root, backend_factory, **kwargs)


def __getattr__(name):
    return getattr(original, name)
