"""Exercise an interrupted fine child through the original dispatcher."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import member4_tail_partition as tail
from qualify_member4_tail_partition import Fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification receipts are append-only")
    simulation_root = Path.cwd() / ".e18_tail_toy"
    simulation_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=simulation_root) as directory:
        fixture = Fixture(directory)
        try:
            fixture.checkpoint()

            def stop_after_one(manifest, checksum, *, software_test_only=False):
                key = list(tail._keys(manifest["task_keys_ref"])[0])
                fixture.worker.assert_lease_active({}, software_test_only=True)
                fixture.make_success(key)
                fixture.charges += 1
                return dict(planned=2, completed=1, failures=[], stop_requested=True)

            paused = tail.execute_fine_plan(
                fixture.fine_plan,
                fixture.fine_sha,
                fixture.dispatcher,
                software_test_only=True,
                processes=fixture.processes,
                execute=stop_after_one,
            )
            assert paused["status"] == "PAUSED"
            assert fixture.charges == 1
            paid_key = fixture.lease["remaining_task_keys_ref"]
            paid_key = tail.base.read_json(tail.base.checked_ref(paid_key))[0]
            native_path = fixture.owner(paid_key) / "records.jsonl"
            before = tail.base.file_sha(native_path)
            resumed = fixture.execute()
            assert resumed["status"] == "DONE"
            assert fixture.charges == 3
            assert tail.base.file_sha(native_path) == before
            fixture.broker.finish(fixture.fine_plan, fixture.lease)
            assert tail.base.read_json(fixture.broker.status_path)["status"] == "DONE"
        finally:
            fixture.close()
    report = dict(
        schema="e18_member4_tail_partial_resume_qualification_v1",
        status="PASS_ORIGINAL_DISPATCHER_PAUSED_PREFIX_RESUME",
        paused_child_committed=1,
        resumed_new_calls=2,
        total_new_synthetic_calls=3,
        repeated_paid_call_count=0,
        prefix_native_bytes_preserved=True,
        parent_native_tasks_verified=5,
        controller_source_sha256=tail.base.file_sha(tail.__file__),
        qualification_source_sha256=tail.base.file_sha(__file__),
        fixture_source_sha256=tail.base.file_sha(Path(__file__).with_name("qualify_member4_tail_partition.py")),
        software_test_only=True,
        physical_model_calls=0,
        gold_reads=0,
        process_signals_sent=0,
    )
    tail.base.write_once(args.output, report)
    print(report["status"])


if __name__ == "__main__":
    main()
