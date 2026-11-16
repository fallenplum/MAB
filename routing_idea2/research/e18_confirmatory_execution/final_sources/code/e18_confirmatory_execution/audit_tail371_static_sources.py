"""Read-only provenance checks for the explicit 371-task E18 tail partition.

Only control metadata and source bytes are decoded. No model, answer, question
text, gold, reward or native output record is opened by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import zipfile

import audit_tail_export_lineage as lineage


BUNDLE_SHA = "99dcd3dc4f7e2c701eb29e68c184405cf7bd3d55cf6734a3359eb402f13c446d"
LINEAGE_SHA = "98b3e24d9a46095b519629db2562a1d9861f9f2d7aef639d43bbf2662171dd4c"
PLAN_SHA = "af3fddaa5ec89e4724d3dbcc840c2b20abc239b8bb0192c4a60628fe075d66b3"
PARENT_SHA = "5fc23696b12239804d4e22eb7fa4a7c15c0c3882e8b7ebc2ec2b0c56ee762995"
SCOPE_SHA = "aeb2a2e9cd7324c730cce49e0b10e26b89887f5f45c9c1f49a0da9982a73d3de"
SCOPE_FILE_SHA = "cae646111612ceb26196d73ed4469165e77b163a332f4db81deeb6f66551f73e"
APPROVAL_SHA = "2fc5174d8470373319457a0b65c5a7b6b111b0930d1f01008af747ef38caa5f0"
CONFIG_SHA = "48598a3598e31a0f30de85713282f43b315e7da6d0a4671ec9d1dad96dada5c8"
ENTRY_SHA = "0da57c5f6c3c0eabdf50671a037b422956eb2cea59c7fd84b786223796708de0"
V5_SHA = "5f5176787a8a5be8db1582256b5ff5e1e7dbd8d838b0d5beddda13ae125df6df"
V6_SHA = "3a9d8ac2a3fb398f395b9de32f083d4bfb9e2c5734b089421db6c1ee1dba7aae"
SOURCE_PINS = {
    "code/e18_member4_worker_v3.py": lineage.WORKER_SHA,
    "code/e18_member4_dispatcher_v3.py": "b097a47f67c3116f5635b16d26dcb8db366aac24fde06afb925d362c45c97327",
    "code/member4_checkpoint_broker.py": "b2dc379cb05cb10bd66979fbca38a6920f2af70ba3f8f4a944d343228b7a304b",
    "code/member4_tail_partition.py": lineage.CONTROLLER_SHA,
    "code/member4_tail371_broker.py": ENTRY_SHA,
    "code/run_member4_fine_cooperative.py": "81da354bbbd3f1dafa935926653fd3f9b125d214ca85d7837e6a541473547c75",
    "code/cooperative_slot_admission_v2.py": "3d20b299e26c03bafb189be795a21b54156b4bc0bf62001da178f13b16d25d10",
    "code/qualify_tail371_extension.py": "8d53744e1fe67e6c2605df2a98877af3af469d03c452d05b48cd601b21bdad8b",
    "planner_v5_runtime/incremental_aggregation_planner_v5.py": V5_SHA,
    "planner_v6_runtime/incremental_aggregation_planner_v5.py": V5_SHA,
    "planner_v6_runtime/incremental_aggregation_planner_v6.py": V6_SHA,
}


def require(condition, description):
    if not condition:
        raise ValueError(description)


def reference(bundle, name):
    return {
        "path": str(bundle.remote_root / name),
        "sha256": hashlib.sha256(bundle.raw[name]).hexdigest(),
    }


def inspect(bundle):
    """Verify immutable sources and the exact parent/checkpoint/fine partition."""
    for name, expected in SOURCE_PINS.items():
        require(hashlib.sha256(bundle.raw[name]).hexdigest() == expected,
                "Frozen runtime/source bytes differ: " + name)
    plan_ref = reference(bundle, "MEMBER4_TAIL371_PLAN_v1.json")
    require(plan_ref["sha256"] == PLAN_SHA, "Explicit fine plan differs")
    plan = bundle.read(plan_ref)
    lease_ref = plan["tail_parent_lease_ref"]
    lease = bundle.read(lease_ref)
    parent_ref = plan["tail_parent_manifest_ref"]
    parent = bundle.read(parent_ref)
    parent_plan_ref = plan["tail_parent_plan_ref"]
    parent_plan = bundle.read(parent_plan_ref)
    upstream_ref = parent["checkpoint_lease_ref"]
    upstream = bundle.read(upstream_ref)
    parent_keys = lineage.unique_keys(bundle.read(parent["task_keys_ref"]))
    remaining = lineage.unique_keys(bundle.read(lease["remaining_task_keys_ref"]))
    upstream_keys = lineage.unique_keys(bundle.read(upstream["remaining_task_keys_ref"]))
    checkpoint_ref = lease["checkpoint_source_audit_ref"]
    checkpoint = bundle.read(checkpoint_ref)
    require(
        plan["schema"] == "e18_member4_dispatch_plan_v3"
        and plan["worker_source_sha256"] == lineage.WORKER_SHA
        and parent_ref["sha256"] == PARENT_SHA
        and parent_plan_ref["sha256"] == lineage.PARENT_PLAN_SHA
        and parent_ref in parent_plan["manifest_refs"]
        and parent["stage"] == "member_generation" and parent["member_index"] == 4
        and parent["software_test_only"] is False
        and parent["adapter_ref"]["sha256"] == lineage.ADAPTER_SHA
        and parent["scope_ref"]["sha256"] == SCOPE_FILE_SHA
        and parent["approval_ref"]["sha256"] == APPROVAL_SHA,
        "Original scientific request/scope/adapter identity differs",
    )
    original_union = set()
    for source_ref in parent_plan["manifest_refs"]:
        source = bundle.read(source_ref)
        keys = lineage.unique_keys(bundle.read(source["task_keys_ref"]))
        require(not original_union.intersection(keys), "Original R1 partition overlaps")
        require(source["checkpoint_lease_ref"] == upstream_ref,
                "Original R1 upstream lease differs")
        original_union.update(keys)
    require(len(parent_plan["manifest_refs"]) == 16 and len(original_union) == 8051
            and original_union == upstream_keys and parent_keys <= upstream_keys,
            "Original R1 partition differs from upstream remaining scope")
    require(
        lease["schema"] == "e18_member4_tail_partition_lease_v1"
        and lease["status"] == "LEASED" and lease["software_test_only"] is False
        and lease["controller_source_ref"]["sha256"] == lineage.CONTROLLER_SHA
        and lease["parent_plan_ref"] == parent_plan_ref
        and lease["parent_manifest_ref"] == parent_ref
        and lease["upstream_checkpoint_lease_ref"] == upstream_ref
        and plan["checkpoint_lease_ref"] == upstream_ref
        and lease["execution_scope_sha256"] == SCOPE_SHA
        and lease["approval_sha256"] == APPROVAL_SHA
        and lease["original_parent_tasks"] == len(parent_keys) == 371
        and lease["checkpoint_successes"] == 348
        and lease["remaining_count"] == len(remaining) == 23
        and lease["pending_or_unknown_attempts"] == 0
        and lease["maximum_child_keys"] == 4
        and remaining <= parent_keys,
        "Nested lease identity or 371=348+23 metadata differs",
    )
    expected_claim = bundle.remote_root / "owned/_dispatcher/plans" / parent_plan_ref["sha256"] / (PARENT_SHA + ".claim")
    require(PurePosixPath(lease["parent_claim_path"]) == expected_claim,
            "Parent FLOCK namespace differs")
    bundle.read(lease["controller_source_ref"], decode=False)
    rows = checkpoint["tasks"]
    by_hash = {row["task_key_sha256"]: row for row in rows}
    success = {key for key, row in by_hash.items() if row["status"] == "SUCCESS"}
    untouched = {key for key, row in by_hash.items() if row["status"] == "UNEXECUTED"}
    require(
        len(rows) == len(by_hash) == 371
        and set(by_hash) == {lineage.digest(list(key)) for key in parent_keys}
        and len(success) == 348 and len(untouched) == 23
        and not success.intersection(untouched) and success | untouched == set(by_hash)
        and untouched == {lineage.digest(list(key)) for key in remaining}
        and checkpoint["original_tasks"] == 371
        and checkpoint["checkpoint_successes"] == 348
        and checkpoint["pending_or_unknown_attempts"] == 0,
        "Checkpoint is not an exact disjoint SUCCESS/UNEXECUTED partition",
    )
    for key in success:
        row = by_hash[key]
        require(lineage.SHA.fullmatch(row["native_record_sha256"]) is not None
                and lineage.SHA.fullmatch(row["native_receipt_file_sha256"]) is not None,
                "Successful prefix lacks native/receipt hashes")
    fine_keys, sizes = set(), []
    for index, fine_ref in enumerate(plan["manifest_refs"]):
        fine = bundle.read(fine_ref)
        keys = lineage.unique_keys(bundle.read(fine["task_keys_ref"]))
        require(
            set(fine) == set(parent) | {"tail_parent_lease_ref", "tail_partition_index"}
            and fine["tail_parent_lease_ref"] == lease_ref
            and fine["tail_partition_index"] == index
            and all(fine[name] == value for name, value in parent.items() if name != "task_keys_ref")
            and 1 <= len(keys) <= 4 and keys <= remaining
            and not fine_keys.intersection(keys),
            "Fine child changes a scientific field or overlaps/escapes parent",
        )
        fine_keys.update(keys)
        sizes.append(len(keys))
    require(sizes == [4, 4, 4, 4, 4, 3] and fine_keys == remaining,
            "Six fine children do not exactly cover 23 unexecuted keys")
    entry_ref = reference(bundle, "MEMBER4_TAIL371_PLAN_v1.json.ENTRY_ADMISSION.json")
    entry = bundle.read(entry_ref)
    require(entry["schema"] == "e18_explicit_tail371_entry_admission_v1"
            and entry["entry_source_ref"]["sha256"] == ENTRY_SHA
            and entry["original_controller_source_ref"] == lease["controller_source_ref"]
            and entry["parent_plan_ref"] == parent_plan_ref
            and entry["parent_manifest_sha256"] == PARENT_SHA
            and entry["expected_parent_tasks"] == 371
            and entry["former_pid"] == lease["former_pid"] == 1172939
            and all(entry[name] == 0 for name in ("signals_sent", "model_calls", "gold_reads")),
            "Explicit371 entry admission differs")
    bundle.read(entry["entry_source_ref"], decode=False)
    sidecar_names = [name for name in bundle.raw if "/R1_EXPLICIT_TWO_PARENT_EXTENSION_" in name]
    require(len(sidecar_names) == 1, "One v6 sidecar required")
    v6_ref = reference(bundle, sidecar_names[0])
    v6 = bundle.read(v6_ref)
    v5 = bundle.read(v6["inherited_tail_extension_ref"])
    require(
        v6["schema"] == "e18_planner_explicit_two_parent_extension_v6"
        and v6["inherited_v5_source_sha256"] == V5_SHA
        and v6["extension_source_sha256"] == V6_SHA
        and v6["tail371_entry_source_ref"] == entry["entry_source_ref"]
        and v6["allowed_original_parent_counts"] == {lineage.PARENT_SHA: 512, PARENT_SHA: 371}
        and v5["schema"] == "e18_planner_r1_tail_source_extension_v1"
        and v5["extension_source_sha256"] == V5_SHA
        and v5["tail_control_source_ref"] == lease["controller_source_ref"]
        and v5["historical_source_admission_requires_live_broker"] is False
        and {row["sha256"] for row in v5["tail_source_plan_refs"]} == {lineage.PLAN_SHA, PLAN_SHA}
        and len(v5["tail_source_plan_refs"]) == 2,
        "Planner v6 is detached from exact two-parent inherited admission",
    )
    for layer in (v5, v6):
        require(layer["additional_scientific_keys"] == 0
                and layer["original_config_modified"] is False
                and layer["sqlite_schema_modified"] is False
                and layer["original_config_ref"]["sha256"] == CONFIG_SHA
                and layer["all_done_config_sha256"] == CONFIG_SHA,
                "Planner changes original config, ledger schema or task union")
    source_rows = [row for row in v5["source_audits"] if row["plan_ref"] == plan_ref]
    require(len(source_rows) == 1, "Exactly one371 source audit required")
    source_row = source_rows[0]
    for name, expected in dict(
        tail_parent_lease_ref=lease_ref, parent_plan_ref=parent_plan_ref,
        parent_manifest_ref=parent_ref, checkpoint_source_audit_ref=checkpoint_ref,
        original_tasks=371, checkpoint_successes=348, remaining_tasks=23,
        fine_manifests=6, additional_scientific_keys=0,
    ).items():
        require(source_row[name] == expected, "Planner371 source audit field differs: " + name)
    linux_ref = reference(bundle, "resource_scheduling/TAIL371_LINUX_QUALIFICATION_v1.json")
    linux = bundle.read(linux_ref)
    require(linux["status"] == "PASS" and linux["tests_run"] == 19
            and linux["errors"] == linux["failures"] == 0
            and linux["broker_entry_sha256"] == ENTRY_SHA
            and linux["planner_v6_sha256"] == V6_SHA
            and linux["original_control_sha256"] == lineage.CONTROLLER_SHA,
            "Linux metadata qualification identity differs")
    return dict(
        fine_plan_ref=plan_ref, original_parent_ref=parent_ref,
        original_parent_plan_ref=parent_plan_ref, nested_lease_ref=lease_ref,
        checkpoint_source_audit_ref=checkpoint_ref, entry_admission_ref=entry_ref,
        planner_v6_sidecar_ref=v6_ref, inherited_tail_extension_ref=v6["inherited_tail_extension_ref"],
        linux_qualification_ref=linux_ref, parent_tasks=371,
        checkpoint_successes=348, unexecuted_tasks=23, fine_child_sizes=sizes,
        original_r1_tasks=len(original_union), exact_disjoint_partition=True,
        scope_sha256=SCOPE_SHA, source_sha256s=SOURCE_PINS,
        original_config_identity=CONFIG_SHA,
    )


def audit(bundle_path, receipt_path):
    started = time.time()
    require(lineage.file_sha(Path(lineage.__file__)) == LINEAGE_SHA,
            "Pinned read-only metadata helper changed")
    bundle_ref = dict(path=str(Path(bundle_path).resolve()), sha256=BUNDLE_SHA)
    bundle = lineage.MetadataBundle(bundle_ref)
    with zipfile.ZipFile(bundle_path) as archive:
        require(archive.testzip() is None, "ZIP CRC verification failed")
    result = inspect(bundle)
    result.update(
        schema="e18_tail371_static_source_audit_v1",
        status="PASS_STATIC_TAIL371_SOURCE_BUNDLE", bundle_ref=bundle_ref,
        audit_source_sha256=lineage.file_sha(Path(__file__)),
        metadata_helper_sha256=LINEAGE_SHA, payload_files_checked=len(bundle.raw) - 1,
        zip_crc_verified=True, all_payload_sha256_and_sizes_verified=True,
        sources_used=list(bundle.used.values()), wall_time_s=time.time() - started,
        gold_reads=0, question_text_reads=0, reward_reads=0,
        native_answer_json_decodes=0, model_calls=0, scorer_calls=0,
        signals_sent=0, files_extracted=0,
        native_output_binding="PENDING_SEPARATE_CLOSED_EXPORT_LINEAGE_AUDIT",
        final_cpu_gate="UNCHANGED_9b_REQUIRED_FOR_GLOBAL_SCIENTIFIC_COMPLETION",
        limitations=[
            "Static SHA-bound checkpoint receipts do not independently prove live FLOCK ownership",
            "348 prefix native hashes and 23 fresh outputs require closed-export binding",
            "Original config identity is checked against frozen references; config bytes are outside this ZIP",
        ],
    )
    destination = Path(receipt_path)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return {"path": str(destination.resolve()), "sha256": lineage.file_sha(destination),
            "status": result["status"], "payload_files_checked": result["payload_files_checked"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.bundle, args.receipt), ensure_ascii=True))


if __name__ == "__main__":
    main()
