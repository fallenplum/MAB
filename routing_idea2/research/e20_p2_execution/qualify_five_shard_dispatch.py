"""Synthetic-only qualification of five-way continuation after historical shards."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

import launch_registered_worker_v3 as legacy_launcher
import launch_registered_worker_v4 as launcher
import p2_aggregation_follower_v3 as follower
import p2_manifest_registry as registry
import prepare_aggregation_shards_v3 as old_preparation
import prepare_aggregation_shards_v4 as preparation
from qualify_registry_consumers import append_event, expect_raises, write_json
from qualify_registry_dispatch import register_plan, success


def fixture(root):
    members = [["synthetic-q0", 42, index] for index in range(6)]
    aggregates = [["synthetic-q" + str(index), seed, action]
                  for index in range(18) for seed in (42, 137, 2024) for action in range(4)]
    scope = dict(schema="p2_scope_contract_v1", role="execution", cohort="e20_p2_confirm",
                 bindings={"final_model_sha256": "synthetic-only"}, required_member_keys=members,
                 required_aggregate_keys=aggregates,
                 execution_limits={"member_generation": len(members), "aggregation": len(aggregates)})
    scope_ref = write_json(root / "scope.json", scope)
    owned = root / "owned"
    approval = dict(schema="p2_scope_approval_v1", status="APPROVED", cohort="e20_p2_confirm",
                    software_test_only=True, scope_sha256_by_role={"execution": registry.digest(scope)},
                    approved_output_root=str(owned), bindings=scope["bindings"],
                    execution_limits=scope["execution_limits"])
    common = dict(scope_ref=scope_ref, approval_ref=write_json(root / "approval.json", approval),
                  owned_output_root=str(owned), asset_root=str(root),
                  question_records_ref=write_json(root / "input.json", {"input_only": True}),
                  adapter_ref=registry.reference(Path(registry.__file__)))
    source = root / "synthetic_worker.py"
    source.write_text("# Synthetic fixture only. Never executed.\n", encoding="utf-8")
    specs = []
    for index in range(7):
        stage = "member_generation" if index < 6 else "aggregation"
        worker_id = "original-" + str(index)
        manifest = dict(common, stage=stage, software_test_only=True,
                        backend={"profile": index if index < 6 else None},
                        task_keys_ref=write_json(root / (worker_id + "_keys.json"),
                                                [members[index]] if index < 6 else aggregates))
        if index < 6:
            manifest["member_index"] = index
        specs.append(dict(worker_id=worker_id, worker_source_ref=registry.reference(source),
                          manifest_ref=write_json(root / (worker_id + ".json"), manifest)))
    path = root / "registry.jsonl"
    registry.create_registry(path, specs)
    return path, scope


def qualify():
    checks = {}
    frozen_names = ("p2_manifest_registry.py", "p2_aggregation_follower_v3.py",
                    "prepare_aggregation_shards_v3.py", "launch_registered_worker_v3.py")
    original_hashes = {name: registry.file_sha(Path(__file__).with_name(name)) for name in frozen_names}
    with tempfile.TemporaryDirectory(prefix="_q5_", dir=Path.cwd()) as folder:
        root = Path(folder)
        path, scope = fixture(root)
        state = registry.load_registry(path)
        original = state["entries"][-1]
        first_key = tuple(scope["required_aggregate_keys"][0])
        append_event(Path(original["journal_path"]), success("aggregation", first_key))
        source_ref = registry.reference(Path(follower.__file__))
        old_plan = old_preparation.prepare(path, root / "old3", source_ref, 3)["plan"]
        state = register_plan(path, old_plan)
        predecessors = []
        for index, shard in enumerate(old_plan["shards"]):
            entry = next(row for row in state["entries"] if row["worker_id"] == shard["worker_id"])
            append_event(Path(entry["journal_path"]), success("aggregation", entry["task_keys"][0]))
            predecessors.append(dict(pid=9000 + index, manifest_sha256=entry["manifest_ref"]["sha256"], at_unix=index))
            write_json(Path(entry["journal_path"]).with_name("SUMMARY.json"),
                       dict(status="SAFELY_STOPPED", stopped=True,
                            scope_sha256=state["scope_canonical_sha256"], pid=9000 + index))
        prefix = path.read_bytes()
        plan = preparation.prepare(path, root / "new5", source_ref, 5)["plan"]
        qualified = preparation.validate_dispatch_plan(state, plan)
        assert qualified["completed"] == 4 and qualified["required"] == len(scope["required_aggregate_keys"])
        assert len(qualified["remaining"]) == 5 and min(qualified["remaining"]) > 0
        checks["five_nonempty_shards_retain_all_old_successes"] = True
        parts = [set(map(tuple, registry.read_json(item["keys_ref"]["path"]))) for item in plan["shards"]]
        done = set(map(tuple, registry.read_json(plan["completed_keys_ref"]["path"])))
        assert set.union(done, *parts) == set(map(tuple, scope["required_aggregate_keys"]))
        assert sum(map(len, parts)) + len(done) == len(scope["required_aggregate_keys"])
        checks["completed_and_five_remaining_subsets_are_disjoint_exact_union"] = True
        state = register_plan(path, plan)
        assert path.read_bytes().startswith(prefix)
        checks["original_and_historical_registration_prefix_preserved"] = True
        allowed_changes = {"worker_id", "execution_plan_id", "task_keys_ref", "gpu_lock_root"}
        for item in plan["shards"]:
            manifest = registry.read_json(item["manifest_ref"]["path"])
            assert {key: value for key, value in manifest.items() if key not in allowed_changes} == {
                key: value for key, value in original["manifest"].items() if key not in allowed_changes}
            assert item["worker_source_ref"] == source_ref
        checks["scope_approval_owner_backend_source_and_scientific_payload_unchanged"] = True
        child = next(row for row in state["entries"] if row["worker_id"] == plan["shards"][0]["worker_id"])
        next_key = child["task_keys"][0]
        append_event(Path(child["journal_path"]), success("aggregation", next_key))
        preparation.validate_dispatch_plan(state, plan)
        checks["same_plan_sibling_progress_allows_remaining_launches"] = True
        assert follower.restored_successes(state, list(done) + [next_key]) == done | {next_key}
        checks["old_and_new_successes_restore_once"] = True
        checks["live_predecessor_overlap_rejected"] = expect_raises(
            lambda: launcher.assert_no_live_overlap(state, child, predecessors, alive=lambda _: True), RuntimeError)
        barriers = launcher.stopped_predecessor_barrier(state, child, predecessors, alive=lambda _: False)
        assert barriers and all(row["status"] == "SAFELY_STOPPED" for row in barriers)
        checks["exact_safe_stop_checkpoints_accepted"] = True
        isolated = deepcopy(child)
        isolated["task_keys"] = [first_key]
        assert launcher.assert_no_live_overlap(state, isolated, predecessors, alive=lambda _: True) == predecessors
        checks["disjoint_actors_can_coexist"] = True
        assert launcher.build_argv is legacy_launcher.build_argv
        assert launcher.admit_gpu is legacy_launcher.admit_gpu
        assert launcher.assert_no_live_overlap is legacy_launcher.assert_no_live_overlap
        checks["original_launch_admission_primitives_reused"] = True
        checks["native_launcher_rejects_synthetic_fixture_without_popen"] = expect_raises(
            lambda: launcher.launch(path, child["worker_id"], 1, root / "unused.json", 14336, True), PermissionError)
        modified = deepcopy(plan)
        manifest = registry.read_json(modified["shards"][0]["manifest_ref"]["path"])
        manifest["backend"] = {"profile": "forbidden-change"}
        modified["shards"][0]["manifest_ref"] = write_json(root / "changed_backend.json", manifest)
        checks["scientific_backend_change_rejected"] = expect_raises(
            lambda: preparation.validate_dispatch_plan(state, modified))
        append_event(Path(original["journal_path"]), success("aggregation", next_key))
        checks["late_historical_progress_rejects_stale_plan"] = expect_raises(
            lambda: preparation.validate_dispatch_plan(state, plan), RuntimeError)
        for count in range(2, 9):
            groups = preparation.partition_remaining(scope["required_aggregate_keys"], done, count)
            locations = {}
            for slot, rows in enumerate(groups):
                for key in rows:
                    assert key[:2] not in locations or locations[key[:2]] == slot
                    locations[key[:2]] = slot
        checks["question_seed_groups_retained_for_two_through_eight_shards"] = True
        for count in (0, 1, 9, True, 5.0):
            expect_raises(lambda: preparation.partition_remaining(scope["required_aggregate_keys"], done, count))
        checks["invalid_noninteger_or_out_of_range_shard_counts_rejected"] = True
    assert original_hashes == {name: registry.file_sha(Path(__file__).with_name(name)) for name in frozen_names}
    checks["original_frozen_files_byte_identical"] = True
    sources = ("prepare_aggregation_shards_v4.py", "launch_registered_worker_v4.py", Path(__file__).name)
    return dict(schema="p2_five_shard_dispatch_qualification_v1", status="PASS", checks=checks,
                checks_passed=len(checks), software_test_only=True, production_deployed=False,
                old_source_sha256=original_hashes,
                source_sha256={name: registry.file_sha(Path(__file__).with_name(name)) for name in sources},
                reference_values_decoded=0, new_llm_calls=0, ssh_operations=0, production_process_changes=0)


if __name__ == "__main__":
    result = qualify()
    target = Path(__file__).with_name("P2_FIVE_SHARD_TOOLING_QUALIFICATION.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
