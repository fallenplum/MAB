"""Additional toy controls for the frozen E18 CPU scoring interface.

This reviewer does not call the production CLI or access formal references,
evaluation rewards, or model backends. Existing test fixtures create synthetic
native records and synthetic references in isolated temporary directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

import e18_final_scoring as final
import qualify_final_scoring as qualification


FINALIZER_SHA = "9b164c40ad6b3a1b21f84c7dc446438a11ce6421d56991d82d224ae2c4b0daa8"
QUALIFIER_SHA = "547adfc01e3c2e3ea4997dcde38663ad4e9292914ed25eb183ecb39ce8449299"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review():
    if sha(final.__file__) != FINALIZER_SHA or sha(qualification.__file__) != QUALIFIER_SHA:
        raise ValueError("Await the exact final source freeze before reviewing")
    guarded = []
    for declared in (47285, 47286):
        context = dict(
            gate=dict(
                status="PASS",
                original_questions=3000,
                unique_aggregate_keys=30921,
                unique_member_keys=declared,
            ),
            metadata=[None] * 3000,
            receipts=[None] * 30921,
            members={index: None for index in range(47285)},
        )
        reached = []

        def forbidden_reference(_):
            reached.append(True)
            raise AssertionError("Reference reached before complete admission")

        rejected = False
        try:
            with patch.object(final, "load_reference_api", side_effect=forbidden_reference):
                final.execute_scoring({}, context)
        except PermissionError:
            rejected = True
        if not rejected or reached:
            raise AssertionError("Incomplete member admission reached the reference phase")
        guarded.append(
            dict(declared_members=declared, actual_members=47285, reference_reached=False)
        )
    qualification.FinalScoringQualification.setUpClass()
    case = qualification.FinalScoringQualification(
        "test_complete_gate_is_reference_and_scorer_free"
    )
    case.setUp()
    try:
        module = case.module
        key = [case.records[0]["question_id"], 42, 0]
        member = (
            case.root
            / "owned"
            / module.canonical_digest(case.scope)
            / "member_generation"
            / module.canonical_digest(key)
            / "LAST_TASK_RECEIPT.json"
        )
        content = member.read_bytes()
        member.unlink()
        provider_calls = []

        def fail_provider(_):
            provider_calls.append(True)
            raise AssertionError("Reference reached before all member records")

        missing_member_blocked = False
        try:
            with patch.object(final, "load_reference_api", side_effect=fail_provider):
                case.gate()
        except FileNotFoundError:
            missing_member_blocked = True
        finally:
            member.write_bytes(content)
        if not missing_member_blocked or provider_calls:
            raise AssertionError(
                "An absent native member did not stop complete execution admission"
            )
        context = case.gate()
        reference_api = final.load_reference_api(case.manifest["reference_adapter_ref"])
        provider = reference_api.E18ReferenceProvider(
            context["reference_scope"], context["approval_ref"], software_test_only=True
        )
        denied = context["reference_scope"]["sealed_exclusions"][0]
        sealed_blocked = False
        try:
            provider.get_reference(
                denied["question_id"],
                denied["canonical_question_sha256"],
                context["reference_scope"]["reference_version"],
            )
        except PermissionError:
            sealed_blocked = True
        if not sealed_blocked or provider.stats()["reference_values_decoded"] != 0:
            raise AssertionError("A denied synthetic sealed identity decoded a reference")
        result = final.execute_scoring(case.manifest, context, software_test_only=True)
        if (
            result["physical_cost_totals"]["member_generation"]["physical_attempts"] != 48
            or result["physical_cost_totals"]["aggregation"]["physical_attempts"] != 24
            or result["distinct_gold_accounts"] != 4
        ):
            raise AssertionError("Toy execution/Gold reuse accounting differs")
        with np.load(
            case.root / "final-private/FINAL_POLICY_REWARDS.npz", allow_pickle=False
        ) as archive:
            same_action_shared = bool(np.array_equal(archive["rewards"][0], archive["rewards"][3]))
        if not same_action_shared:
            raise AssertionError(
                "Policies choosing the same physical action received different rewards"
            )
        public_text = json.dumps(result, ensure_ascii=False)
        if any(
            row["question_id"] in public_text or row["question"] in public_text
            for row in case.records
        ):
            raise AssertionError("A public result exported a toy question identity/text")
        power = final.load_statistics(case.manifest["power_source_ref"])
        toy = np.zeros((4, 3, 3000), dtype=np.int8)
        toy[0, :, :1746] = 1
        toy[1, :, ::2] = 1
        toy[2, :, 1::3] = 1
        labels = np.asarray(["gsm8k"] * 1500 + ["bbh"] * 1500)
        metadata = []
        for index, task in enumerate(labels):
            if task == "gsm8k":
                frame = "gsm8k_official_train_original_primary_frame"
            elif index < 1746:
                frame = "bbh_old_llmrouterbench_common_subpool"
            else:
                frame = "bbh_official_full_source_frame_expansion"
            metadata.append(
                dict(
                    task=str(task),
                    subtask=None if task == "gsm8k" else "synthetic_bbh",
                    source_frame=frame,
                )
            )
        statistics = final.final_statistics(toy, metadata, power)
        quality = statistics["quality"]["Ridge"]
        expected = dict(
            task_macro_accuracy=0.582,
            bbh_accuracy=246 / 1500,
            old_bbh_questions=246,
            expanded_bbh_questions=1254,
        )
        if (
            abs(quality["task_macro_accuracy"] - expected["task_macro_accuracy"]) > 1e-14
            or abs(quality["task_accuracy"]["bbh"] - expected["bbh_accuracy"]) > 1e-14
            or quality["source_frames"]["bbh_old_llmrouterbench_common_subpool"]["questions"] != 246
            or quality["source_frames"]["bbh_official_full_source_frame_expansion"]["questions"]
            != 1254
        ):
            raise AssertionError("Equal-task weights or source-frame denominators differ")
        differences = (toy[2] - toy[1]).mean(axis=0)
        manual = 0.5 * differences[:1500].mean() + 0.5 * differences[1500:].mean()
        manual_se = math.sqrt(
            0.25 * np.var(differences[:1500], ddof=1) / 1500
            + 0.25 * np.var(differences[1500:], ddof=1) / 1500
        )
        primary = statistics["primary"]
        if (
            abs(primary["estimate"] - manual) > 1e-14
            or abs(primary["standard_error"] - manual_se) > 1e-14
            or primary["original_questions"] != 3000
            or primary["generation_records"] != 9000
        ):
            raise AssertionError("Paired inference changed the original-question statistical unit")
        report = dict(
            status="PASS_INDEPENDENT_INTERFACE_PERMISSION_AND_NUMERICAL_REVIEW",
            finalizer_source_sha256=FINALIZER_SHA,
            qualifier_source_sha256=QUALIFIER_SHA,
            direct_api_member_guards=guarded,
            missing_native_member_stops_before_reference=True,
            denied_synthetic_sealed_identity_rejected=True,
            denied_reference_values_decoded=0,
            toy_union_member_attempts=48,
            toy_union_aggregation_attempts=24,
            nominal_policy_question_generation_projections=48,
            same_action_rewards_shared=True,
            toy_distinct_reference_accounts=4,
            toy_public_question_identity_or_text_matches=0,
            synthetic_weighting_control=expected,
            synthetic_question_units=3000,
            synthetic_generation_records=9000,
            paired_estimate_matches_manual=True,
            question_cluster_standard_error_matches_manual=True,
            physical_lf_jsonl_implementation="file iteration with newline='' for original inputs and native histories",
            source_scorer_sha256=context["validator"].SCORER_SOURCE_SHA256,
            source_original_scorer_sha256=context["validator"].ORIGINAL_SCORER_SOURCE_SHA256,
            formal_gold_reads=0,
            formal_scorer_calls=0,
            actual_model_calls=0,
            production_state_or_results_modified=0,
            frozen_sources_modified=0,
            boundary="A frozen software review and synthetic numerical controls. Formal scoring remains root-triggered after complete actual execution admission.",
        )
    finally:
        case.doCleanups()
    if sha(final.__file__) != FINALIZER_SHA or sha(qualification.__file__) != QUALIFIER_SHA:
        raise ValueError("Frozen sources changed during the independent review")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = review()
    result["review_source_sha256"] = sha(Path(__file__))
    with arguments.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
