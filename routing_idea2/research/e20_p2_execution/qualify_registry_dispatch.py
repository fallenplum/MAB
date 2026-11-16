"""Small local-only checks for evidence registration and disjoint continuation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

import launch_registered_worker_v3 as launcher
import p2_aggregation_follower_v3 as follower
import p2_manifest_registry as registry
import prepare_aggregation_shards_v3 as preparation
from qualify_registry_consumers import append_event, expect_raises, fixture, write_json


def success(stage, key):
    task = dict(stage=stage, question_id=key[0], generation_seed=key[1])
    task["member_index" if stage == "member_generation" else "combination_index"] = key[2]
    return dict(event="TASK_RESULT", status="SUCCESS", task=task, from_cache=False,
                physical_attempt_id="synthetic-" + registry.digest(task),
                actual_usage=dict(prompt_tokens=7, completion_tokens=2), at_unix=1)


def register_plan(path, plan):
    for shard in plan["shards"]:
        registry.add_manifest(path, {name: shard[name] for name in ("worker_id", "manifest_ref", "worker_source_ref")})
    return registry.load_registry(path)


def qualify():
    checks = {}
    with tempfile.TemporaryDirectory(prefix="_qd3_", dir=Path.cwd()) as folder:
        root = Path(folder)
        path, common, scope, spec = fixture(root)
        state = registry.load_registry(path)
        original = next(entry for entry in state["entries"] if entry["worker_id"] == "original-aggregation")
        member = next(entry for entry in state["entries"] if entry["worker_id"] == "original-member0")
        first_key = tuple(scope["required_aggregate_keys"][0])
        append_event(Path(original["journal_path"]), success("aggregation", first_key))
        original_prefix = path.read_bytes()

        duplicate = spec("duplicate-key", "aggregation", [list(first_key), list(first_key)])
        checks["within_manifest_duplicate_rejected"] = expect_raises(lambda: registry.add_manifest(path, duplicate))
        rebound = dict(original, manifest_ref=spec("replacement", "aggregation", [list(first_key)])["manifest_ref"])
        rebound = {name: rebound[name] for name in ("worker_id", "manifest_ref", "worker_source_ref")}
        checks["worker_identity_redefinition_rejected"] = expect_raises(lambda: registry.add_manifest(path, rebound))
        alias = dict(worker_id="manifest-alias", manifest_ref=original["manifest_ref"],
                     worker_source_ref=original["worker_source_ref"])
        checks["manifest_alias_rejected"] = expect_raises(lambda: registry.add_manifest(path, alias))

        own_source = registry.reference(Path(follower.__file__))
        prepared = preparation.prepare(path, root / "prepared2", own_source)
        plan = prepared["plan"]
        checked = preparation.validate_dispatch_plan(state, plan)
        assert checked["required"] == 3 and checked["completed"] == 1
        pieces = [set(map(tuple, registry.read_json(item["keys_ref"]["path"]))) for item in plan["shards"]]
        assert set.union({first_key}, *pieces) == set(map(tuple, scope["required_aggregate_keys"]))
        assert not pieces[0] & pieces[1] and not any(first_key in part for part in pieces)
        checks["two_shards_exact_completed_remaining_partition"] = True
        state = register_plan(path, plan)
        assert path.read_bytes().startswith(original_prefix)
        checks["registration_add_only_prefix_retained"] = True
        assert state["common"] == common
        changes = {"worker_id", "execution_plan_id", "task_keys_ref", "gpu_lock_root"}
        for item in plan["shards"]:
            child = registry.read_json(item["manifest_ref"]["path"])
            assert {key: value for key, value in child.items() if key not in changes} == {
                key: value for key, value in original["manifest"].items() if key not in changes}
        checks["same_scope_approval_owner_backend_and_manifest_payload"] = True

        # A sibling already making progress must not invalidate this same plan.
        active_shard = next(item for item in state["entries"] if item["worker_id"] == plan["shards"][0]["worker_id"])
        if not active_shard["task_keys"]:
            active_shard = next(item for item in state["entries"] if item["worker_id"] == plan["shards"][1]["worker_id"])
        later_key = active_shard["task_keys"][0]
        append_event(Path(active_shard["journal_path"]), success("aggregation", later_key))
        preparation.validate_dispatch_plan(state, plan)
        checks["same_plan_successor_progress_does_not_invalidate_sibling"] = True
        subset = follower.restored_successes(state, [first_key, later_key])
        assert subset == {first_key, later_key}
        checks["restore_union_of_old_and_new_aggregate_journals"] = True

        # Old/full evidence advancing after the snapshot is an actual stale plan.
        append_event(Path(original["journal_path"]), success("aggregation", later_key))
        checks["historical_progress_after_preparation_rejected"] = expect_raises(
            lambda: preparation.validate_dispatch_plan(state, plan), RuntimeError)

        prepared3 = preparation.prepare(path, root / "prepared3", own_source, shard_count=3)
        plan3 = prepared3["plan"]
        assert len(plan3["shards"]) == 3
        preparation.validate_dispatch_plan(state, plan3)
        checks["explicit_three_shards_supported"] = True
        artificial = [("synthetic-" + str(index), seed, action)
                      for index in range(9) for seed in (42, 137, 2024) for action in range(4)]
        for count in (2, 3):
            grouped = preparation.partition_remaining(artificial, artificial[:3], count)
            locations = {}
            for index, rows in enumerate(grouped):
                for key in rows:
                    group = key[:2]
                    assert group not in locations or locations[group] == index
                    locations[group] = index
        checks["question_seed_actions_stay_together_in_two_and_three_shards"] = True

        # Refresh adds future member journals without resetting prior offsets.
        tail = follower.RegisteredMemberTail(path, original["worker_id"], original["manifest_ref"],
                                             original["worker_source_ref"], software_test_only=True)
        member_key = tuple(scope["required_member_keys"][0])
        append_event(Path(member["journal_path"]), success("member_generation", member_key))
        assert list(tail.completed()) == [member_key]
        assert not list(tail.completed())
        extra = spec("future-member0", "member_generation", [list(member_key)], 0)
        registry.add_manifest(path, extra)
        refreshed = registry.load_registry(path)
        extra_entry = next(entry for entry in refreshed["entries"] if entry["worker_id"] == extra["worker_id"])
        extra_journal = Path(extra_entry["journal_path"])
        extra_journal.parent.mkdir(parents=True, exist_ok=True)
        row = json.dumps(success("member_generation", member_key)).encode() + b"\n"
        split = len(row) // 2
        extra_journal.write_bytes(row[:split])
        tail.refresh(force=True)
        assert not list(tail.completed())
        with extra_journal.open("ab") as stream:
            stream.write(row[split:])
        assert list(tail.completed()) == [member_key]
        assert not list(tail.completed())
        checks["dynamic_member_addition_preserves_offsets_and_partial_lines"] = True
        checks["production_follower_rejects_software_fixture"] = expect_raises(
            lambda: follower.RegisteredMemberTail(path, original["worker_id"], original["manifest_ref"],
                                                   original["worker_source_ref"]), PermissionError)

        selected = next(entry for entry in state["entries"] if entry["worker_id"] == active_shard["worker_id"])
        old_start = dict(pid=7001, manifest_sha256=original["manifest_ref"]["sha256"], at_unix=1)
        checks["live_original_full_actor_overlap_rejected"] = expect_raises(
            lambda: launcher.assert_no_live_overlap(state, selected, [old_start], alive=lambda _: True), RuntimeError)
        isolated = deepcopy(selected)
        isolated["task_keys"] = [first_key]
        other = dict(pid=7002, manifest_sha256=active_shard["manifest_ref"]["sha256"], at_unix=2)
        assert launcher.assert_no_live_overlap(state, isolated, [other], alive=lambda _: True) == [other]
        checks["disjoint_registered_actors_can_coexist"] = True
        checks["missing_graceful_stop_checkpoint_rejected"] = expect_raises(
            lambda: launcher.stopped_predecessor_barrier(state, selected, [old_start], alive=lambda _: False), FileNotFoundError)
        write_json(Path(original["journal_path"]).with_name("SUMMARY.json"),
                   dict(status="SAFELY_STOPPED", stopped=True, scope_sha256=state["scope_canonical_sha256"], pid=7001))
        barrier = launcher.stopped_predecessor_barrier(state, selected, [old_start], alive=lambda _: False)
        assert len(barrier) == 1
        checks["exact_predecessor_stop_checkpoint_accepted"] = True
        argv = launcher.build_argv(path, selected, gpu=4, python="synthetic-python-never-executed")
        assert argv[2] == selected["worker_source_ref"]["path"]
        assert argv[argv.index("--manifest") + 1] == selected["manifest_ref"]["path"]
        assert argv[argv.index("--worker-id") + 1] == selected["worker_id"]
        assert "--registry" in argv
        checks["launcher_explicit_worker_manifest_and_registry_binding"] = True

        # A partial append remains readable but cannot be followed by another ADD.
        with path.open("ab") as stream:
            stream.write(b'{"event":"ADD"')
        assert registry.load_registry(path)["trailing_partial_line"]
        checks["registry_partial_append_requires_explicit_recovery"] = expect_raises(
            lambda: registry.add_manifest(path, extra), RuntimeError)
        checks["invalid_shard_count_rejected"] = expect_raises(
            lambda: preparation.partition_remaining(artificial, [], 4))

    return dict(schema="p2_registry_dispatch_qualification_v1", status="PASS", checks=checks,
                checks_passed=len(checks), software_test_only=True, new_llm_calls=0,
                reference_values_decoded=0, ssh_operations=0, production_process_changes=0,
                production_deployed=False)


if __name__ == "__main__":
    result = qualify()
    sources = ("p2_manifest_registry.py", "p2_aggregation_follower_v3.py", "prepare_aggregation_shards_v3.py",
               "launch_registered_worker_v3.py", Path(__file__).name)
    result["source_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in sources}
    target = Path(__file__).with_name("P2_REGISTRY_DISPATCH_QUALIFICATION.json")
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
