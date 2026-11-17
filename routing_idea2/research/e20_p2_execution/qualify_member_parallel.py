"""Synthetic-only regression qualification of frozen member continuation."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

import launch_registered_member as launcher
import launch_registered_worker_v3 as legacy
import p2_manifest_registry as registry
import prepare_member_shards as preparation
import stop_registered_workers as stopper
from qualify_registry_consumers import append_event, expect_raises, write_json
from qualify_registry_dispatch import register_plan, success


def fixture(root):
    members = [["synthetic-q" + str(question), seed, index]
               for question in range(24) for seed in (42, 137, 2024) for index in range(6)]
    aggregate = [["synthetic-q0", 42, 0]]
    scope = dict(schema="p2_scope_contract_v1", role="execution", cohort="e20_p2_confirm",
                 bindings={"final_model_sha256": "synthetic-only"}, required_member_keys=members,
                 required_aggregate_keys=aggregate,
                 execution_limits={"member_generation": len(members), "aggregation": len(aggregate)})
    owned = root / "owned"
    approval = dict(schema="p2_scope_approval_v1", status="APPROVED", cohort="e20_p2_confirm",
                    software_test_only=True, scope_sha256_by_role={"execution": registry.digest(scope)},
                    approved_output_root=str(owned), bindings=scope["bindings"], execution_limits=scope["execution_limits"])
    common = dict(scope_ref=write_json(root / "scope.json", scope),
                  approval_ref=write_json(root / "approval.json", approval), owned_output_root=str(owned),
                  asset_root=str(root), question_records_ref=write_json(root / "input.json", {"input_only": True}),
                  adapter_ref=registry.reference(Path(registry.__file__)))
    source = registry.reference(Path(__file__).with_name("p2_member_worker.py"))
    specs = []
    for index in range(7):
        stage = "member_generation" if index < 6 else "aggregation"
        keys = [key for key in members if key[2] == index] if index < 6 else aggregate
        manifest = dict(common, stage=stage, software_test_only=True,
                        backend={"profile": index}, schema="e18_gpu_worker_manifest_v1",
                        task_keys_ref=write_json(root / f"keys{index}.json", keys))
        if index < 6:
            manifest["member_index"] = index
        specs.append(dict(worker_id=f"original{index}", worker_source_ref=source,
                          manifest_ref=write_json(root / f"manifest{index}.json", manifest)))
    path = root / "registry.jsonl"
    registry.create_registry(path, specs)
    return path, source


def qualify():
    checks = {}
    frozen_names = ("p2_member_worker.py", "p2_manifest_registry.py", "p2_aggregation_follower_v3.py")
    hashes = {name: registry.file_sha(Path(__file__).with_name(name)) for name in frozen_names}
    with tempfile.TemporaryDirectory(prefix="_qmember_", dir=Path.cwd()) as folder:
        root = Path(folder)
        path, source = fixture(root)
        state = registry.load_registry(path)
        original = next(entry for entry in state["entries"] if entry["member_index"] == 4)
        done = original["task_keys"][:9]
        for key in done:
            append_event(Path(original["journal_path"]), success("member_generation", key))
        start = dict(pid=904, at_unix=0, manifest_sha256=original["manifest_ref"]["sha256"])
        summary = dict(schema="e18_gpu_worker_summary_v1", manifest_sha256=start["manifest_sha256"],
                       stage="member_generation", profile=4, planned=len(original["task_keys"]),
                       completed=len(done), failures=[], stop_requested=True, software_test_only=True)
        summary_path = Path(original["journal_path"]).with_name("SUMMARY.json")
        write_json(summary_path, summary)
        checkpoint = launcher.checked_member_checkpoint(state, original, start)
        assert checkpoint["status"] == "SAFELY_STOPPED" and "pid" not in summary and "scope_sha256" not in summary
        checks["original_member_summary_without_pid_or_scope_qualified"] = True
        plan = preparation.prepare(path, root / "plans", source, 4, 4)["plan"]
        sizes = preparation.validate_dispatch_plan(state, plan)["remaining"]
        assert min(sizes) > 0 and sum(sizes) + len(done) == len(original["task_keys"])
        checks["four_nonempty_subsets_exact_profile_coverage"] = True
        parts = [set(map(tuple, registry.read_json(item["keys_ref"]["path"]))) for item in plan["shards"]]
        assert all(not part.intersection(done) for part in parts)
        assert len(set.union(*parts)) == sum(map(len, parts))
        checks["old_completed_native_cache_keys_excluded_from_new_paid_execution"] = True
        permitted = {"worker_id", "execution_plan_id", "task_keys_ref", "gpu_lock_root"}
        for item in plan["shards"]:
            manifest = registry.read_json(item["manifest_ref"]["path"])
            assert {key: value for key, value in manifest.items() if key not in permitted} == {
                key: value for key, value in original["manifest"].items() if key not in permitted}
            assert item["worker_source_ref"] == source
        checks["original_executable_scope_backend_and_scientific_payload_preserved"] = True
        state = register_plan(path, plan)
        child = next(entry for entry in state["entries"] if entry["worker_id"] == plan["shards"][0]["worker_id"])
        append_event(Path(child["journal_path"]), success("member_generation", child["task_keys"][0]))
        preparation.validate_dispatch_plan(state, plan)
        checks["same_plan_sibling_progress_does_not_block_launch"] = True
        other = next(entry for entry in state["entries"] if entry["member_index"] == 1)
        append_event(Path(other["journal_path"]), success("member_generation", other["task_keys"][0]))
        preparation.validate_dispatch_plan(state, plan)
        checks["other_member_profiles_can_continue_without_staling_plan"] = True
        checks["live_overlapping_predecessor_rejected"] = expect_raises(
            lambda: launcher.assert_no_live_overlap(state, child, [start], alive=lambda _: True), RuntimeError)
        barriers = launcher.stopped_member_predecessors(state, child, [start], alive=lambda _: False)
        assert barriers and barriers[0]["completed"] == len(done)
        checks["member_stop_barrier_uses_original_summary_schema"] = True
        changed = deepcopy(summary)
        changed["completed"] += 1
        write_json(summary_path, changed)
        checks["summary_journal_count_mismatch_rejected"] = expect_raises(
            lambda: launcher.checked_member_checkpoint(state, original, start), RuntimeError)
        changed = deepcopy(summary)
        changed["stop_requested"] = False
        write_json(summary_path, changed)
        checks["incomplete_nongraceful_member_checkpoint_rejected"] = expect_raises(
            lambda: launcher.checked_member_checkpoint(state, original, start), RuntimeError)
        write_json(summary_path, summary)
        changed = deepcopy(plan)
        manifest = registry.read_json(changed["shards"][0]["manifest_ref"]["path"])
        manifest["backend"] = {"profile": "forbidden"}
        changed["shards"][0]["manifest_ref"] = write_json(root / "bad_backend.json", manifest)
        checks["backend_change_rejected"] = expect_raises(lambda: preparation.validate_dispatch_plan(state, changed))
        append_event(Path(original["journal_path"]), success("member_generation", child["task_keys"][0]))
        checks["late_same_profile_historical_progress_rejected"] = expect_raises(
            lambda: preparation.validate_dispatch_plan(state, plan), RuntimeError)
        checks["native_member_launcher_rejects_fixture_before_popen"] = expect_raises(
            lambda: launcher.launch(path, child["worker_id"], 4, root / "unused", 14336, True), PermissionError)
        checks["stop_helper_rejects_fixture_before_any_signal"] = expect_raises(
            lambda: stopper.request_stop(path, [904], root / "stop"), PermissionError)
        assert launcher.assert_no_live_overlap is legacy.assert_no_live_overlap
        assert launcher.build_argv is legacy.build_argv and launcher.admit_gpu is legacy.admit_gpu
        checks["existing_launch_admission_primitives_reused"] = True
        for count in (0, 1, 9, True, 4.0):
            expect_raises(lambda: preparation.partition_remaining(original["task_keys"], done, count))
        checks["invalid_partition_counts_rejected"] = True
    assert hashes == {name: registry.file_sha(Path(__file__).with_name(name)) for name in frozen_names}
    checks["old_frozen_worker_registry_and_follower_byte_identical"] = True
    sources = ("prepare_member_shards.py", "launch_registered_member.py", "stop_registered_workers.py", Path(__file__).name)
    return dict(schema="p2_member_parallel_software_qualification_v1", status="PASS", checks=checks,
                checks_passed=len(checks), software_test_only=True, production_deployed=False,
                frozen_source_sha256=hashes,
                source_sha256={name: registry.file_sha(Path(__file__).with_name(name)) for name in sources},
                ssh_operations=0, production_process_changes=0, new_llm_calls=0, reference_values_decoded=0)


if __name__ == "__main__":
    result = qualify()
    Path(__file__).with_name("P2_MEMBER_PARALLEL_QUALIFICATION.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
