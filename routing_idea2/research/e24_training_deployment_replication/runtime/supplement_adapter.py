"""Run the original P2 scientific operator under the independent E24 scope."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
from pathlib import Path
import sys


CODE = Path(__file__).resolve().parent
E23_SHA = "29cc9b2aa47dd691ebf25cd5be04d9ee941820601cb5e33138530f60cb0e6a51"
COHORT = "e24_training_deployment_replication"
SCOPE_SCHEMA = "e24_independent_execution_scope_v1"
APPROVAL_SCHEMA = "e24_independent_execution_approval_v1"
COUNTS = {"member_generation": 20661, "aggregation": 53661}
POLICIES = frozenset({"B_U", "B_P", "C_U", "C_P"})
FROZEN = {
    "protocol_sha256": "6c30f10b6ac417e04b33170d8088230dc07d3c4893b65f2a51283ab216dca8f1",
    "target_inputs_sha256": "16c65d5f97c4bca3f3db66b3ac47944632034d84c699d9f6486bcee31bd5b435",
    "frozen_actions_sha256": "9595ee1ddf9e30e85a0c95e51af2155e6b3de4cbd36265114341f32f9c324cef",
    "budget_sha256": "f949970a26a874a28fe039ffd2c3c1945945f1379619716044c3178a1e0340ac",
    "deployed_policies_sha256": "cf3fb435718f2f5d5265fbca0cce7988a9ca0a0325877567298dca76b2f65780",
    "execution_keys_sha256": "b6288062188d2dc4f33a2b3ea85c66213938de3964bbbf29879535d25b8f6a0a",
    "reference_plan_sha256": "9ed27bb75a6be34a6888c343cae5172f860c101c63f77fed2412e75e14e657dc",
    "origin_receipt_sha256": "212e298d5f13924757875ce744bcb14ca1125f3e4843f49c022d3eb8e60f12fe",
}


def load_reference_adapter():
    local = CODE / "e23_supplement_adapter_original.py"
    source = local if local.is_file() else CODE.parents[1] / "e23_deployment_prior_and_references/supplement_adapter.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != E23_SHA:
        raise ValueError("The original E23 native adapter differs")
    name = "_e24_isolated_e23_adapter"
    spec = importlib.util.spec_from_file_location(name, source)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[name] = reference
    spec.loader.exec_module(reference)
    return reference


_reference = load_reference_adapter()
_reference.COHORT = COHORT
_reference.SCOPE_SCHEMA = SCOPE_SCHEMA
_reference.APPROVAL_SCHEMA = APPROVAL_SCHEMA
_reference.REQUIRED_BINDINGS = frozenset(FROZEN) | {
    "target_ids_sha256", "parent_scope_sha256",
}
_reference._execution_globals["COHORT"] = COHORT
_base_scope = _reference.build_scope_contract
_base_approval = _reference.validate_scope_approval

PARENT_SCOPE_SHA = _reference.PARENT_SCOPE_SHA
OPERATOR_FP = _reference.OPERATOR_FP
TEAMS = _reference.TEAMS
GENERATION_SEEDS = _reference.GENERATION_SEEDS
canonical_digest = _reference.canonical_digest


def build_scope_contract(records, policies, bindings, member_keys, aggregate_keys):
    records = list(records)
    if set(policies) != POLICIES:
        raise ValueError("Exactly the four frozen E24 policies are required")
    if Counter(row["task"] for row in records) != {"bbh": 598, "gsm8k": 598}:
        raise ValueError("The E24 independent task allocation differs")
    bbh = Counter(row.get("subtask") for row in records if row["task"] == "bbh")
    if len(bbh) != 26 or set(bbh.values()) != {23} or "penguins_in_a_table" in bbh:
        raise ValueError("The frozen common 26-subtask allocation differs")
    if any(bindings.get(key) != value for key, value in FROZEN.items()):
        raise ValueError("The E24 frozen cohort, protocol, predictions or budget differs")
    scope = _base_scope(records, policies, bindings, member_keys, aggregate_keys)
    if (scope["execution_limits"] != COUNTS or scope["cached_member_dependency_keys"]
            or scope["required_member_keys"] != scope["member_execution_keys"]
            or scope["evaluation_union_aggregate_keys"] != scope["aggregate_execution_keys"]):
        raise ValueError("E24 requires the complete from-zero support union and no cached dependencies")
    for head in ("B", "C"):
        uniform = scope["frozen_policy_probabilities"][f"{head}_U"]
        prior = scope["frozen_policy_probabilities"][f"{head}_P"]
        if any(p > 0 and u <= 0 for urow, prow in zip(uniform, prior)
               for u, p in zip(urow, prow)):
            raise ValueError("A prior deployment left its frozen uniform support")
    return scope


def validate_scope_approval(approval_ref, scope, operation, *, software_test_only=False):
    if operation not in COUNTS:
        raise PermissionError("E24 remote execution is limited to its two native inference stages")
    approval = _base_approval(approval_ref, scope, operation, software_test_only=software_test_only)
    if (scope["execution_limits"] != COUNTS or approval.get("sealed200_answer_access") is not False
            or set(approval["operations"]) != set(COUNTS)
            or any(approval["bindings"].get(key) != value for key, value in FROZEN.items())):
        raise PermissionError("E24 authorization differs from the frozen inputs and exact native call limits")
    return approval


_reference.build_scope_contract = build_scope_contract
_reference._execution_globals["validate_scope_approval"] = validate_scope_approval


def __getattr__(name):
    return getattr(_reference, name)
