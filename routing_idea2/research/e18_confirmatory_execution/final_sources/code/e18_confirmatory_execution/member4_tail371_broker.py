"""Explicit371-key operational entry for the unchanged tail-partition class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal

import member4_tail_partition as tail


PARENT_PLAN_SHA = "a627f3094c742ec0d352582a9abe8b6338af6eaf0967333ebf30f45727c97166"
PARENT_SHA = "5fc23696b12239804d4e22eb7fa4a7c15c0c3882e8b7ebc2ec2b0c56ee762995"
CONTROL_SHA = "45c88b376059a4b390d065ef61e38db24fa1862cf8b98f7658c110d89c423f9f"


def validate_parent(plan_sha, parent_sha, count):
    if (plan_sha, parent_sha, count) != (PARENT_PLAN_SHA, PARENT_SHA, 371):
        raise PermissionError("Only the frozen371-key part15 of the original R1 plan is authorized")


def make_broker(args, dispatcher, processes, validator, scope_sha):
    if tail.base.file_sha(tail.__file__) != CONTROL_SHA:
        raise ValueError("Original tail controller source bytes differ")
    plan_sha = tail.base.file_sha(args.parent_plan)
    parents = dispatcher.admit_plan(tail.base.read_json(args.parent_plan))
    matches = [row for reference, row in parents if reference["sha256"] == PARENT_SHA]
    if len(matches) != 1:
        raise ValueError("Explicit part15 parent is absent")
    count = len(tail._keys(matches[0]["task_keys_ref"]))
    validate_parent(plan_sha, PARENT_SHA, count)
    return tail.TailPartitionBroker(
        args.parent_plan, plan_sha, PARENT_SHA, dispatcher, processes,
        args.lease_root, args.fine_root, args.fine_plan,
        former_pid=args.former_pid, former_gpu=args.former_gpu,
        validator=validator, scope_sha=scope_sha,
        chunk_size=args.chunk_size, expected_tasks=371,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--former-pid", type=int, required=True)
    parser.add_argument("--former-gpu", required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--fine-root", type=Path, required=True)
    parser.add_argument("--fine-plan", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--dispatcher", type=Path, default=Path(__file__).with_name("e18_member4_dispatcher_v3.py"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    dispatcher = tail.load_runtime(args.dispatcher)
    parents = dispatcher.admit_plan(tail.base.read_json(args.parent_plan))
    parent = next(value for reference, value in parents if reference["sha256"] == PARENT_SHA)
    module, adapter, _, _, _, _ = dispatcher.worker.load_worker(parent, software_test_only=False)
    broker = make_broker(args, dispatcher, tail.base.LinuxProcesses(), adapter.checked_member_dependency, module.canonical_digest(adapter.scope))
    if not args.execute:
        result = broker.inspect()
    else:
        receipt_path = args.fine_plan.with_name(args.fine_plan.name + ".ENTRY_ADMISSION.json")
        tail.base.write_once(receipt_path, dict(
            schema="e18_explicit_tail371_entry_admission_v1",
            entry_source_ref=tail.ref(__file__), original_controller_source_ref=tail.ref(tail.__file__),
            parent_plan_ref=tail.ref(args.parent_plan), parent_manifest_sha256=PARENT_SHA,
            expected_parent_tasks=371, former_pid=args.former_pid, former_gpu=args.former_gpu,
            signals_sent=0, model_calls=0, gold_reads=0,
        ))
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: setattr(broker, "stop_requested", True))
        result = broker.run()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
