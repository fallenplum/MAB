"""Small synthetic tests of the four-way graceful member continuation."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import prepare_member_dispatch_v2 as dispatch


def fixture(root):
    scope = dict(synthetic=True)
    scope_sha = dispatch.canonical(scope)
    scope_ref = dispatch.freeze(root/"inputs/SCOPE.json", scope)
    approval_ref = dispatch.freeze(root/"inputs/APPROVAL.json", {"synthetic": True})
    worker = root/"code/supplement_worker.py"
    worker.parent.mkdir()
    worker.write_text("synthetic fixture only", encoding="utf-8")
    state = dict(root=root, owned=root/"owned", scope_sha=scope_sha, manifests=[],
                 required=dict(member_generation=set(), aggregation=set()))
    entries = []
    for shard in range(2):
        keys = {("synthetic"+str(index), 42, 4) for index in range(40*shard, 40*(shard+1))}
        state["required"]["member_generation"].update(keys)
        name = f"member4_shard{shard}"
        worker_id = "fixture_"+name
        task_ref = dispatch.freeze(root/"inputs"/f"{name}_tasks.json", [list(key) for key in sorted(keys)])
        manifest = dict(stage="member_generation", member_index=4, worker_id=worker_id,
                        scope_ref=scope_ref, approval_ref=approval_ref,
                        task_keys_ref=task_ref, owned_output_root=str(root/"owned"),
                        gpu_lock_root=str(root/"gpu-locks"/worker_id), backend={"synthetic": True})
        manifest_ref = dispatch.freeze(root/"manifests"/f"{name}.json", manifest)
        entry = dict(worker_id=worker_id, stage="member_generation", profile=4,
                     task_count=len(keys), manifest_ref=manifest_ref)
        entries.append(entry)
        state["manifests"].append((entry, manifest, keys))
        argv = [dispatch.PYTHON, "-B", "-u", str(worker), "--manifest", manifest_ref["path"],
                "--manifest-sha256", manifest_ref["sha256"], "--gpu", str(shard)]
        dispatch.freeze(root/"control"/f"{name}.start.json", dict(
            worker_id=worker_id, stage="member_generation", manifest_sha256=manifest_ref["sha256"],
            pid=1000000000+shard, gpu=shard, argv=argv, at_unix=1))
        completed = sorted(keys)[:4]
        journal = root/"owned/_dispatcher"/manifest_ref["sha256"]/"EVENTS.jsonl"
        journal.parent.mkdir(parents=True)
        rows = []
        for key in completed:
            native_sha = "0"*64
            rows.append(dict(event="TASK_RESULT", task=dict(
                stage="member_generation", question_id=key[0], generation_seed=key[1], member_index=key[2]),
                status="SUCCESS", actual_usage={"prompt_tokens": 1, "completion_tokens": 1},
                physical_attempt_id=key[0], native_record_sha256=native_sha))
            owner = state["owned"]/scope_sha/"member_generation"/dispatch.canonical(list(key))
            dispatch.freeze(owner/"ATTEMPT_STATE.json", dict(
                status="SUCCESS_COMMITTED", scope_sha256=scope_sha, stage="member_generation",
                approval_sha256=approval_ref["sha256"], task_key=list(key), native_record_sha256=native_sha))
            dispatch.freeze(owner/"LAST_TASK_RECEIPT.json", {})
            dispatch.freeze(owner/"records.jsonl", {})
        journal.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        dispatch.freeze(journal.with_name("SUMMARY.json"), dict(
            schema="e23_native_worker_summary_v1", status="SAFELY_STOPPED", stopped=True,
            pid=1000000000+shard, scope_sha256=scope_sha, manifest_sha256=manifest_ref["sha256"],
            stage="member_generation", profile=4, planned=40, completed=4, failures=[], at_unix=2))
    aggregate_key = ("synthetic-other", 42, 0)
    aggregate_manifest = dict(stage="aggregation", worker_id="fixture-aggregate",
                              scope_ref=scope_ref, approval_ref=approval_ref,
                              owned_output_root=str(root/"owned"))
    aggregate_ref = dispatch.freeze(root/"manifests/aggregate.json", aggregate_manifest)
    aggregate_entry = dict(worker_id="fixture-aggregate", stage="aggregation", profile=0,
                           task_count=1, manifest_ref=aggregate_ref)
    entries.append(aggregate_entry)
    state["manifests"].append((aggregate_entry, aggregate_manifest, {aggregate_key}))
    state["required"]["aggregation"].add(aggregate_key)
    plan = dict(root=str(root), source_sha256={"supplement_worker.py": dispatch.sha(worker)},
                scope_ref=scope_ref, approval_ref=approval_ref, manifests=entries)
    dispatch.freeze(root/"EXECUTION_PLAN.json", plan)
    state["plan"] = plan
    return state


def qualify():
    checks = {}
    original_loader = dispatch.load_parent
    try:
        with TemporaryDirectory(dir=Path.cwd()) as name:
            root = Path(name)
            state = fixture(root)
            dispatch.load_parent = lambda _: state
            output = root/"control/member4_dispatch_v2"
            result = dispatch.prepare(root, output, 4)
            state_after, plan, new = dispatch.validate_dispatch(root, output/"DISPATCH_V2.json")
            checks["four_nonempty_disjoint_shards"] = len(new) == 4 and all(item[2] for item in new)
            checks["completed_and_remaining_exact"] = result["completed"] == 8 and sum(result["shard_counts"]) == 72
            union = set().union(*(item[2] for item in new))
            done = set(map(tuple, dispatch.checked(plan["completed_member4_keys_ref"])))
            checks["no_success_reexecution"] = not union & done
            checks["exact_original_key_union"] = union | done == state["required"]["member_generation"]
            checks["original_scope_approval_plan_unchanged"] = (
                plan["scope_ref"] == state["plan"]["scope_ref"]
                and plan["approval_ref"] == state["plan"]["approval_ref"]
                and dispatch.checked(plan["parent_plan_ref"]) == state["plan"])
            checks["repeated_prepare_reuses_frozen_plan"] = (
                dispatch.prepare(root, output, 4)["status"] == "EXISTING_FOUR_WAY_DISPATCH_REUSED")
            other = state["manifests"][-1][0]
            journal = state["owned"]/"_dispatcher"/other["manifest_ref"]["sha256"]/"EVENTS.jsonl"
            journal.parent.mkdir(parents=True, exist_ok=True)
            journal.write_text(json.dumps(dict(
                event="TASK_RESULT", task=dict(stage="aggregation", question_id="synthetic-other",
                generation_seed=42, combination_index=0), status="SUCCESS"))+"\n", encoding="utf-8")
            dispatch.validate_dispatch(root, output/"DISPATCH_V2.json")
            checks["other_original_actor_progress_does_not_stale_dispatch"] = True
            _, _, keys = state["manifests"][0]
            remaining = sorted(keys-done)[0]
            owner = state["owned"]/state["scope_sha"]/"member_generation"/dispatch.canonical(list(remaining))
            dispatch.freeze(owner/"ATTEMPT_STATE.json", dict(
                status="RESERVED_BEFORE_BACKEND", stage="member_generation",
                approval_sha256=state["plan"]["approval_ref"]["sha256"],
                scope_sha256=state["scope_sha"], task_key=list(remaining)))
            _, _, continued = dispatch.validate_dispatch(root, output/"DISPATCH_V2.json")
            checks["assigned_successor_reservation_does_not_block_siblings"] = len(continued) == 4
            fence = dispatch.read(owner/"ATTEMPT_STATE.json")
            fence["status"] = "SUCCESS_COMMITTED"
            (owner/"ATTEMPT_STATE.json").write_text(json.dumps(fence), encoding="utf-8")
            _, _, continued = dispatch.validate_dispatch(root, output/"DISPATCH_V2.json")
            checks["assigned_successor_commit_does_not_block_siblings"] = len(continued) == 4
            checks["successor_not_reclassified_as_retired_success"] = (
                dispatch.read(output/"DISPATCH_V2.json")["completed_at_preparation"] == 8)
            fence["status"] = "STOP_REQUIRES_RECONCILIATION"
            (owner/"ATTEMPT_STATE.json").write_text(json.dumps(fence), encoding="utf-8")
            try:
                dispatch.validate_dispatch(root, output/"DISPATCH_V2.json")
            except RuntimeError:
                checks["failed_successor_fence_blocks_restart"] = True
            else:
                checks["failed_successor_fence_blocks_restart"] = False
            (owner/"ATTEMPT_STATE.json").unlink()
            try:
                dispatch.validate_dispatch(root, output/"DISPATCH_V2.json")
            except RuntimeError:
                checks["unknown_owner_without_fence_blocks_restart"] = True
            else:
                checks["unknown_owner_without_fence_blocks_restart"] = False
        with TemporaryDirectory(dir=Path.cwd()) as name:
            root = Path(name)
            state = fixture(root)
            dispatch.load_parent = lambda _: state
            first_entry, first_manifest, keys = state["manifests"][0]
            remaining = sorted(keys)[8]
            owner = state["owned"]/state["scope_sha"]/"member_generation"/dispatch.canonical(list(remaining))
            dispatch.freeze(owner/"ATTEMPT_STATE.json", dict(
                status="RESERVED_BEFORE_BACKEND", stage="member_generation",
                approval_sha256=state["plan"]["approval_ref"]["sha256"],
                scope_sha256=state["scope_sha"], task_key=list(remaining)))
            try:
                dispatch.prepare(root, root/"control/member4_dispatch_v2", 4)
            except RuntimeError:
                checks["preexisting_reserved_owner_blocks_initial_preparation"] = True
            else:
                checks["preexisting_reserved_owner_blocks_initial_preparation"] = False
    finally:
        dispatch.load_parent = original_loader
    try:
        dispatch.ensure_exited({"pid": 1}, lambda _: {"alive": True})
    except RuntimeError:
        checks["live_predecessor_blocks_preparation"] = True
    else:
        checks["live_predecessor_blocks_preparation"] = False
    if not all(checks.values()):
        raise AssertionError("Member dispatch synthetic qualification failed")
    return dict(status="PASS", checks=checks, checks_passed=len(checks), new_llm_calls=0, gold_reads=0,
                source_sha256={name: dispatch.sha(Path(__file__).with_name(name)) for name in (
                    "prepare_member_dispatch_v2.py", "stop_member4_v2.py", "launch_member_dispatch_v2.py")})


if __name__ == "__main__":
    print(json.dumps(qualify()))
