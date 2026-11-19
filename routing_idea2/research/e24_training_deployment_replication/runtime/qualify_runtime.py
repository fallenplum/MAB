"""Qualify E24 manifests and dependency polling without a native model or gold."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import prepare_execution as prepare
import supplement_adapter as adapter
import supplement_worker as worker


def synthetic_scope():
    """Match the frozen count and set-overlap histograms using artificial IDs and text."""
    overlap_four = next(i for i, team in enumerate(adapter.TEAMS) if len(set(team) | {0, 1, 2}) == 4)
    overlap_five = next(i for i, team in enumerate(adapter.TEAMS) if len(set(team) | {0, 1, 2}) == 5)
    overlap_six = next(i for i, team in enumerate(adapter.TEAMS) if len(set(team) | {0, 1, 2}) == 6)
    supports = ([[0]] * 18 + [[0, overlap_four]] * 93 + [[0, overlap_five]] * 49
                + [[0, overlap_six]] * 3 + [list(range(17))] * 1015 + [list(range(18))] * 18)
    records = []
    uniforms, priors, aggregates, members = [], [], [], set()
    for index, support in enumerate(supports):
        question = f"Synthetic input {index}."
        identity = f"synthetic_{index}"
        records.append(dict(question=question, question_id=identity,
                            canonical_question_sha256=adapter.canonical_question(question),
                            task="gsm8k" if index < 598 else "bbh",
                            subtask="" if index < 598 else f"toy_{(index-598)//23}"))
        uniforms.append([1/len(support) if action in support else 0 for action in range(20)])
        priors.append([1 if action == support[0] else 0 for action in range(20)])
        for seed in adapter.GENERATION_SEEDS:
            aggregates.extend([identity, seed, action] for action in support)
            members.update((identity, seed, m) for action in support for m in adapter.TEAMS[action])
    policies = dict(B_U=uniforms, B_P=priors, C_U=priors, C_P=priors)
    bindings = dict(adapter.FROZEN, parent_scope_sha256=adapter.PARENT_SCOPE_SHA,
                    target_ids_sha256=adapter.canonical_digest([
                        [row["question_id"], row["canonical_question_sha256"]] for row in records]))
    scope = adapter.build_scope_contract(records, policies, bindings, sorted(members), aggregates)
    return scope, records


def require_failure(operation, error):
    try:
        operation()
    except error:
        return True
    raise AssertionError(f"Expected refusal with {error.__name__}")


def check_plan(root, scope, checks):
    inputs = prepare.freeze(root / "input.json", {"synthetic_only": True})
    prepare.freeze(root / "code/supplement_adapter.py", {"synthetic_only": True})
    mode = {profile: {"mode": "noop_for_software_qualification"} for profile in (*range(6), "aggregation")}
    plan, plan_ref = prepare.build_plan(root, scope, inputs, root, mode, [2, 2, 2, 2, 4, 2], 8,
                                       software_test_only=True)
    checks["plan_scope_exact"] = plan["scope_canonical_sha256"] == adapter.canonical_digest(scope)
    checks["plan_two_stages_derived_from_locked_scope"] = len(plan["manifests"]) == 22
    checks["plan_idempotent"] = prepare.build_plan(root, scope, inputs, root, mode,
                                                   [2, 2, 2, 2, 4, 2], 8, software_test_only=True)[1] == plan_ref
    assigned = {"member_generation": set(), "aggregation": set()}
    stamps = Counter()
    for row in plan["manifests"]:
        path = Path(row["manifest_ref"]["path"])
        manifest = prepare.load(path)
        key_ref = manifest["task_keys_ref"]
        raw = prepare.load(Path(key_ref["path"]))
        keys = set(map(tuple, raw))
        assert prepare.sha(path) == row["manifest_ref"]["sha256"]
        assert prepare.sha(key_ref["path"]) == key_ref["sha256"]
        assert len(keys) == len(raw) == row["task_count"]
        assert not assigned[row["stage"]].intersection(keys)
        assert manifest["software_test_only"] is True
        assigned[row["stage"]].update(keys)
        if row["stage"] == "aggregation":
            assert not prepare.load(Path(manifest["member_dependencies_ref"]["path"]))
            stamps.update(set(key[:2] for key in keys))
        else:
            assert all(key[2] == row["profile"] for key in keys)
    checks["exact_exclusive_full_stage_key_union"] = (
        assigned["member_generation"] == set(map(tuple, scope["member_execution_keys"]))
        and assigned["aggregation"] == set(map(tuple, scope["aggregate_execution_keys"])))
    checks["aggregation_keeps_question_seed_together"] = set(stamps.values()) == {1}
    approval_ref = plan["approval_ref"]
    for stage in adapter.COUNTS:
        assert adapter.validate_scope_approval(approval_ref, scope, stage,
                                               software_test_only=True)["execution_limits"] == adapter.COUNTS
    checks["synthetic_cannot_enter_production"] = require_failure(
        lambda: adapter.validate_scope_approval(approval_ref, scope, "member_generation"), PermissionError)
    checks["answer_access_refused"] = require_failure(
        lambda: adapter.validate_scope_approval(approval_ref, scope, "read_reference", software_test_only=True),
        PermissionError)
    checks["altered_scope_refused"] = require_failure(
        lambda: adapter.validate_scope_approval(approval_ref, dict(scope, aggregation_seed=41),
                                                "aggregation", software_test_only=True), PermissionError)


def check_scope_memo(root, scope, records, checks):
    reference = adapter._reference
    api = SimpleNamespace(root=root, inference=SimpleNamespace(configuration_fingerprint=lambda spec: spec.fp))
    pool = [dict(index=index, fingerprint=f"synthetic-member-{index}") for index in range(6)]
    prepare.freeze(root / "configs/pool_v1.json", {"models": pool})
    members = {index: SimpleNamespace(fp=row["fingerprint"], temperature=0.7, top_p=0.95,
                                      max_new_tokens=2048, enable_thinking=False) for index, row in enumerate(pool)}
    aggregate = SimpleNamespace(fp=reference.original.AGGREGATOR_MODEL_FP)
    scientific = adapter.P2ExecutionAdapter(api, scope, records, members, aggregate)
    expected = adapter.canonical_digest(scope)
    assert reference.original._SCOPE_DIGESTS[id(scientific.scope)][0] is scientific.scope
    assert reference._execution_globals["COHORT"] == adapter.COHORT
    assert reference._execution_globals["validate_scope_approval"] is adapter.validate_scope_approval
    with patch.object(reference.original.json, "dumps", side_effect=AssertionError("Scope cache miss")):
        checks["native_scope_authorization_uses_object_memo"] = all(
            reference._execution_globals["canonical_digest"](scientific.scope) == expected for _ in range(256))
    checks["scientific_adapter_frozen_blind_inputs"] = require_failure(
        lambda: adapter.P2ExecutionAdapter(api, scope, [dict(records[0], gold="synthetic"), *records[1:]],
                                           members, aggregate), ValueError)


def check_dependency_polling(checks):
    clock, available, calls = [0.0], {0, 1}, Counter()
    def dependency(module, adapter_obj, output_root, external, key):
        m = key[2]
        calls[m] += 1
        if m not in available:
            raise FileNotFoundError("synthetic late commit")
        return {"member": m}

    queue = worker.PacedDependencyQueue(SimpleNamespace(TEAMS=adapter.TEAMS),
             SimpleNamespace(required_members={("q", 42, m) for m in range(3)}),
             [("q", 42, 0)], Path("unused"), {}, dependency)
    with patch.object(worker.time, "monotonic", side_effect=lambda: clock[0]):
        queue.refresh()
        assert not queue.ready
        available.add(2)
        clock[0] = 9.0
        queue.refresh()
        assert not queue.ready and calls[2] == 1
        clock[0] = 10.0
        queue.refresh()
        assert tuple(queue.ready) == (("q", 42, 0),)
    checks["missing_receipt_throttling_keeps_committed_dependencies"] = calls == {0: 1, 1: 1, 2: 2}
    checks["bounded_poll_after_late_member_commit"] = queue.records(("q", 42, 0)) == {
        0: {"member": 0}, 1: {"member": 1}, 2: {"member": 2}}


def qualify(input_bundle):
    scope, records = synthetic_scope()
    checks = {}
    with TemporaryDirectory(prefix="e24_runtime_toy_") as temporary:
        root = Path(temporary)
        check_plan(root, scope, checks)
        check_scope_memo(root, scope, records, checks)
    check_dependency_polling(checks)
    partial_keys = scope["aggregate_execution_keys"][:-1]
    checks["underfilled_scope_refused"] = require_failure(
        lambda: adapter.build_scope_contract(records, scope["frozen_policy_probabilities"],
             scope["bindings"], scope["member_execution_keys"], partial_keys), ValueError)
    real = None
    if input_bundle is not None:
        admitted, real_rows, _ = prepare.input_scope(input_bundle.resolve(strict=True))
        real = dict(canonical_scope_sha256=adapter.canonical_digest(admitted), questions=len(real_rows),
                    execution_limits=admitted["execution_limits"],
                    cached_member_dependencies=len(admitted["cached_member_dependency_keys"]),
                    operator_fingerprint=admitted["operator_fingerprint"])
        checks["actual_local_input_only_bundle_identity"] = real["execution_limits"] == adapter.COUNTS
    if not all(checks.values()):
        raise AssertionError("E24 software qualification has a failing check")
    return dict(schema="e24_native_runtime_software_qualification_v1", status="PASS",
                checks=checks, checks_passed=len(checks), actual_local_bundle=real,
                source_sha256={name: prepare.sha(Path(__file__).with_name(name)) for name in (
                    "prepare_execution.py", "supplement_adapter.py", "supplement_worker.py", "qualify_runtime.py")},
                model_construction_calls=0, new_llm_calls=0, gold_reads=0, ssh_operations=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bundle", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(qualify(arguments.input_bundle), ensure_ascii=False, indent=2))
