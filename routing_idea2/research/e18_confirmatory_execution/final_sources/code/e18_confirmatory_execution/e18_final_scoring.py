"""CPU-only E18 final scoring after the complete sparse execution gate.

Default CLI mode validates existing execution outputs. The separate explicit
--execute-scoring switch reaches the approved reference provider and scorer.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time

import audit_startup16
import e18_aggregation_worker_v3 as worker
from e18_aggregation_dispatcher_v3 import ShardClaim


POLICIES = ("Ridge", "SPO_coordinate", "TSPO_uniform", "GlobalFixed")
REFERENCE_ADAPTER_SHA = (
    "8848aa235d98e11005a33ff969274dc0309d02228297f4242f207663e57178ea"
)
POWER_SHA = "1516045f08e475ff94d0d808ca08943819a64cb164bc26f7821b824920cc959f"
COMMON_SHA = "93c92e706e8741a6bfd47aed2c69bf3ceef4a37ebe72aeff1d3bf12b72e5e171"


def load_reference_api(reference):
    path = worker.checked_ref(reference)
    if reference["sha256"] != REFERENCE_ADAPTER_SHA:
        raise ValueError("Frozen single-reference provider source differs")
    spec = importlib.util.spec_from_file_location(
        "e18_final_frozen_reference_api", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_statistics(reference):
    path = worker.checked_ref(reference)
    if (
        reference["sha256"] != POWER_SHA
        or worker.file_sha(path.parent / "e18_common.py") != COMMON_SHA
    ):
        raise ValueError("Frozen paired-question statistical source differs")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(
        "e18_final_frozen_paired_statistics", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_reference(reference, mappings):
    logical = reference["path"].replace("\\", "/")
    matches = []
    for row in mappings:
        if (
            set(row) != {"logical_prefix", "physical_root"}
            or not Path(row["physical_root"]).is_absolute()
        ):
            raise ValueError(
                "Explicit logical-prefix/absolute-physical-root mappings required"
            )
        prefix = row["logical_prefix"].replace("\\", "/").rstrip("/")
        if logical == prefix or logical.startswith(prefix + "/"):
            matches.append((len(prefix), prefix, row["physical_root"]))
    if not matches:
        return reference
    _, prefix, target = max(matches)
    suffix = PurePosixPath(logical[len(prefix) :].lstrip("/"))
    if ".." in suffix.parts:
        raise ValueError(
            "Runtime source mapping cannot escape its explicit physical root"
        )
    physical = Path(target).resolve().joinpath(*suffix.parts)
    if not physical.resolve().is_relative_to(Path(target).resolve()):
        raise ValueError("Runtime source mapping escaped its physical root")
    if any(
        part.casefold() in {"sealed", "sealed_test", "sealed200"}
        for part in physical.resolve().parts
    ):
        raise PermissionError(
            "A runtime member-reference mapping cannot enter a sealed payload directory"
        )
    return dict(path=str(physical), sha256=reference["sha256"])


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = ShardClaim(path.with_name(path.name + ".publication.lock"))
    if not lock.acquire():
        raise RuntimeError("Another final-output publisher owns this file")
    temporary = path.with_name(
        "." + path.name + ".tmp-" + str(os.getpid()) + "-" + str(time.time_ns())
    )
    try:
        if path.exists():
            if worker.read_json(path) != value:
                raise ValueError(
                    "Existing immutable output has different contents " + str(path)
                )
            return
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        lock.release()


def write_npz_once(path, arrays):
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = ShardClaim(path.with_name(path.name + ".publication.lock"))
    if not lock.acquire():
        raise RuntimeError("Another final-array publisher owns this file")
    temporary = path.with_name(
        "." + path.name + ".tmp-" + str(os.getpid()) + "-" + str(time.time_ns())
    )
    try:
        if path.exists():
            with np.load(path, allow_pickle=False) as existing:
                if set(existing.files) != set(arrays) or any(
                    existing[key].dtype != value.dtype
                    or not np.array_equal(existing[key], value)
                    for key, value in arrays.items()
                ):
                    raise ValueError("Existing immutable result arrays differ")
            return
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        lock.release()


def question_union(scope, *, software_test_only=False):
    metadata = scope["question_keys"]
    ids = [row["question_id"] for row in metadata]
    canonical = [row["canonical_question_sha256"] for row in metadata]
    actions = scope["frozen_policy_actions"]
    if (
        len(set(ids)) != len(ids)
        or len(set(canonical)) != len(canonical)
        or len(actions) != 4
        or any(len(row) != len(ids) for row in actions)
        or any(
            type(value) is not int or not 0 <= value < 20
            for row in actions
            for value in row
        )
    ):
        raise ValueError(
            "Unique cohort and four frozen integer policy action rows required"
        )
    expected = {
        (qid, seed, action)
        for index, qid in enumerate(ids)
        for action in {row[index] for row in actions}
        for seed in (42, 137, 2024)
    }
    raw = scope["required_aggregate_keys"]
    required = set(map(tuple, raw))
    if required != expected or len(raw) != len(required):
        raise ValueError(
            "Required sparse union does not equal four policies times three repeats"
        )
    if not software_test_only:
        if (
            len(ids) != 3000
            or len(required) != 30921
            or len(scope["required_member_keys"]) != 47286
        ):
            raise ValueError(
                "The approved 3000-question / 30921-aggregate / 47286-member scope differs"
            )
        if Counter(row["task"] for row in metadata) != Counter(gsm8k=1500, bbh=1500):
            raise ValueError("The approved equal-task source composition differs")
    return metadata, required


def owned_history(
    owner, success, stage, key, apis, costs, *, external=False, software_test_only=False
):
    path = owner / "records.jsonl"
    if external:
        history = [success]
    else:
        fence = worker.read_json(owner / "ATTEMPT_STATE.json")
        if (
            fence.get("status") != "SUCCESS_COMMITTED"
            or fence.get("native_record_sha256") != success["record_sha256"]
            or fence.get("native_job_id") != success["job_id"]
            or fence.get("scope_sha256") != owner.parent.parent.name
            or fence.get("stage") != stage
            or fence.get("task_key") != list(key)
            or fence.get("software_test_only") is not software_test_only
        ):
            raise ValueError("An owned execution remains pending or unreconciled")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            history = [json.loads(line) for line in stream if line.strip()]
    for record in history:
        if record.get("job_id") != success["job_id"] or record.get(
            "record_sha256"
        ) != apis.execution._digest(
            {
                field: value
                for field, value in record.items()
                if field != "record_sha256"
            }
        ):
            raise ValueError("Native physical-cost history changed identity")
        usage = (
            record.get("result")
            if record["status"] == "success"
            else record.get("actual_usage")
        )
        usage = apis.inference._validated_usage(usage)
        if usage is None or record.get("cost_status") != "recorded_actual_usage":
            raise ValueError(
                "Unknown physical execution usage requires reconciliation before Gold"
            )
        attempt = record.get("attempt_id")
        if not isinstance(attempt, str) or not attempt:
            raise ValueError("Physical attempt identity is missing")
        measured = dict(
            stage=stage,
            task_key=list(key),
            native_record_sha256=record["record_sha256"],
            physical_attempt_id=attempt,
            status=record["status"],
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            reused_historical_source=external,
        )
        if attempt in costs and costs[attempt] != measured:
            raise ValueError(
                "One physical execution has incompatible task or cost identities"
            )
        costs[attempt] = measured


def execution_gate(
    manifest, *, software_test_only=False, fixture_factory=None, native_read_root=None
):
    if manifest.get("schema") != "e18_final_scoring_manifest_v1" or manifest.get(
        "policy_ids"
    ) != list(POLICIES):
        raise ValueError(
            "The fixed E18 final scoring manifest and policy order are required"
        )
    if manifest.get("software_test_only") is not software_test_only:
        raise PermissionError("Final evaluation mode differs from its manifest")
    if not software_test_only and fixture_factory is not None:
        raise PermissionError("Production finalization accepts no fixture backend")
    validator = audit_startup16.load_source_validator(manifest["source_validator_ref"])
    execution = worker.read_json(worker.checked_ref(manifest["execution_manifest_ref"]))
    module, adapter, _, _, _, external = worker.load_worker(
        execution,
        software_test_only=software_test_only,
        injected_factory=fixture_factory,
    )
    adapter._worker_software_mode = software_test_only
    scope = adapter.scope
    if (
        execution["question_records_ref"]["sha256"]
        != scope["bindings"]["target_inputs_sha256"]
    ):
        raise ValueError(
            "Input source metadata is detached from the frozen input binding"
        )
    metadata, required = question_union(scope, software_test_only=software_test_only)
    metadata = [
        dict(
            row,
            source_frame=adapter.questions[row["question_id"]].get("source_frame")
            or "MISSING_SOURCE_FRAME",
            cohort_role=adapter.questions[row["question_id"]].get("cohort_role")
            or "MISSING_COHORT_ROLE",
        )
        for row in metadata
    ]
    if not software_test_only:
        bbh = [row for row in metadata if row["task"] == "bbh"]
        old = sum(
            row["source_frame"] == "bbh_old_llmrouterbench_common_subpool"
            for row in bbh
        )
        if (
            old != 246
            or len(bbh) - old != 1254
            or any(row["source_frame"] == "MISSING_SOURCE_FRAME" for row in metadata)
            or Counter(row["cohort_role"] for row in metadata)
            != Counter(original_fresh400=400, approved_new2600=2600)
        ):
            raise ValueError(
                "Frozen source-frame or fresh400/new2600 composition differs"
            )
    reference_scope = worker.read_json(
        worker.checked_ref(manifest["reference_scope_ref"])
    )
    scoring_scope = worker.read_json(worker.checked_ref(manifest["scoring_scope_ref"]))
    approval_ref = execution["approval_ref"]
    module.validate_scope_approval(
        approval_ref,
        reference_scope,
        "read_reference",
        software_test_only=software_test_only,
    )
    module.validate_scope_approval(
        approval_ref,
        scoring_scope,
        "score_outputs",
        software_test_only=software_test_only,
    )
    if validator.build_scoring_scope(scope, reference_scope) != scoring_scope:
        raise ValueError(
            "Scoring/reference scope bindings differ from the exact execution scope"
        )
    exclusions = reference_scope["sealed_exclusions"]
    if not software_test_only and len(exclusions) != 200:
        raise ValueError("The original 200 sealed identity exclusions are required")
    denied_ids = {row["question_id"] for row in exclusions}
    denied_hashes = {row["canonical_question_sha256"] for row in exclusions}
    if any(
        row["question_id"] in denied_ids
        or row["canonical_question_sha256"] in denied_hashes
        for row in metadata
    ):
        raise PermissionError("A sealed identity entered the final cohort")
    for source in reference_scope["reference_sources"]:
        if any(
            part.casefold() in {"sealed", "sealed200", "sealed_test"}
            for part in Path(source["path"]).resolve().parts
        ):
            raise PermissionError("A sealed payload path cannot supply a reference")
    logical_root = execution["owned_output_root"]
    output_root = Path(
        native_read_root or manifest.get("native_read_root") or logical_root
    ).resolve()
    mappings = manifest.get("runtime_reference_path_mappings", [])
    external = {
        key: runtime_reference(reference, mappings)
        for key, reference in external.items()
    }
    scope_sha = module.canonical_digest(scope)
    costs = {}
    members = {}
    for key in sorted(adapter.required_members):
        record = worker.member_dependency(module, adapter, output_root, external, key)
        members[key] = record
        owner = (
            output_root
            / scope_sha
            / "member_generation"
            / module.canonical_digest(list(key))
        )
        owned_history(
            owner,
            record,
            "member_generation",
            key,
            adapter.apis,
            costs,
            external=key not in adapter.callable_members,
            software_test_only=software_test_only,
        )
    receipts = []
    by_key = {}
    physical = set()
    for key in sorted(required):
        owner = (
            output_root / scope_sha / "aggregation" / module.canonical_digest(list(key))
        )
        receipt = worker.read_json(owner / "LAST_TASK_RECEIPT.json")
        source = validator.verified_aggregation_source(receipt)
        qid, generation, action = key
        expected = dict(
            question_id=qid,
            canonical_id=adapter.questions[qid]["canonical_question_sha256"],
            generation_seed=generation,
            combination_index=action,
            team=list(module.TEAMS[action]),
        )
        if (
            receipt["task_key"] != expected
            or receipt["scope_sha256"] != scope_sha
            or receipt["approval_sha256"] != approval_ref["sha256"]
            or receipt["bindings"] != scope["bindings"]
            or receipt["software_test_only"] is not software_test_only
        ):
            raise ValueError(
                "A final source receipt belongs to another approved key or scope"
            )
        native = receipt["native_record"]
        adapter.apis.aggregation._validate_result(
            adapter.apis.inference.GenerationResult(**native["result"]),
            adapter.aggregator_spec,
            {"seed": 42},
        )
        expected_provenance = {}
        for index in module.TEAMS[action]:
            member = members[(qid, generation, index)]
            expected_provenance[adapter.members[index]["alias"]] = dict(
                job_id=member["job_id"],
                record_sha256=member["record_sha256"],
                model_id=member["result"]["model_id"],
                revision=member["result"]["model_revision"],
                model_fingerprint=member["model_fingerprint"],
            )
        provenance = dict(native["member_provenance"])
        contract = provenance.pop("output_contract", None)
        suffix = adapter.apis.typed.output_contract(native["task"], native["subtask"])
        if (
            not isinstance(contract, dict)
            or set(contract) != {"version", "base_messages_sha256", "suffix_sha256"}
            or contract["version"] != adapter.apis.typed.CONTRACT_VERSION
            or contract["suffix_sha256"] != adapter.apis.execution._text_hash(suffix)
            or not module.valid_sha(contract["base_messages_sha256"])
        ):
            raise ValueError("Native frozen typed output-contract metadata differs")
        if provenance != expected_provenance:
            raise ValueError(
                "A final aggregation is detached from its exact same-seed member sources"
            )
        if source["physical_attempt_id"] in physical:
            raise ValueError(
                "Distinct final selected keys reused one physical aggregation"
            )
        physical.add(source["physical_attempt_id"])
        owned_history(
            owner,
            native,
            "aggregation",
            key,
            adapter.apis,
            costs,
            software_test_only=software_test_only,
        )
        receipts.append(receipt)
        by_key[key] = receipt
    startup_ref = manifest["startup_audit_ref"]
    startup = worker.read_json(worker.checked_ref(startup_ref))
    if (
        startup.get("schema") != "e18_startup_aggregation_scientific_identity_audit_v1"
        or startup.get("status") != "PASS"
        or startup.get("audited_tasks", 0) < 16
        or startup.get("software_test_only") is not software_test_only
        or startup.get("operator_fingerprint") != module.OPERATOR_FP
        or startup.get("audit_source_sha256")
        != worker.file_sha(audit_startup16.__file__)
        or startup.get("scorer_calls") != 0
        or startup.get("gold_reads") != 0
        or not {row["record_sha256"] for row in startup["tasks"]}
        <= {row["native_record_sha256"] for row in receipts}
    ):
        raise ValueError(
            "Exact typed-prompt startup audit is detached from the final execution set"
        )
    gate = dict(
        schema="e18_complete_execution_admission_v1",
        status="PASS",
        original_questions=len(metadata),
        unique_aggregate_keys=len(required),
        unique_member_keys=len(members),
        policy_ids=list(POLICIES),
        generation_seeds=[42, 137, 2024],
        execution_scope_sha256=scope_sha,
        approval_sha256=approval_ref["sha256"],
        source_receipt_set_sha256=module.canonical_digest(
            sorted(row["native_record_sha256"] for row in receipts)
        ),
        physical_attempts=len(costs),
        physical_cost_ledger_sha256=module.canonical_digest(
            sorted(row["native_record_sha256"] for row in costs.values())
        ),
        failed_known_cost_attempts=sum(
            row["status"] != "success" for row in costs.values()
        ),
        reference_values_read=0,
        scorer_calls=0,
        sealed_payload_reads=0,
        startup_exact_prompt_audit_sha256=startup_ref["sha256"],
        source_frame_counts=dict(Counter(row["source_frame"] for row in metadata)),
        cohort_role_counts=dict(Counter(row["cohort_role"] for row in metadata)),
        runtime_native_transport_sha256=module.canonical_digest(
            dict(
                logical_owned_root=logical_root,
                physical_native_read_root=str(output_root),
                member_reference_mappings=mappings,
            )
        ),
    )
    return dict(
        gate=gate,
        module=module,
        adapter=adapter,
        validator=validator,
        reference_scope=reference_scope,
        scoring_scope=scoring_scope,
        approval_ref=approval_ref,
        execution=execution,
        receipts=receipts,
        by_key=by_key,
        members=members,
        costs=costs,
        metadata=metadata,
        runtime_io=dict(
            logical_owned_root=logical_root,
            physical_native_read_root=str(output_root),
            member_reference_mappings=mappings,
            original_native_bytes_and_scope_hashes_preserved=True,
        ),
    )


def holm_five(comparisons):
    values = [
        (name, 1.0 if row["p_value"] is None else row["p_value"])
        for name, row in comparisons.items()
    ]
    running = 0.0
    for rank, (name, value) in enumerate(sorted(values, key=lambda item: item[1])):
        running = max(running, (len(values) - rank) * value)
        comparisons[name]["holm_adjusted_p"] = (
            None if comparisons[name]["p_value"] is None else min(1.0, running)
        )
    return comparisons


def final_statistics(rewards, metadata, power_api):
    import numpy as np

    labels = np.asarray([row["task"] for row in metadata])
    quality = {}
    for index, name in enumerate(POLICIES):
        groups = {
            task: float(rewards[index, :, labels == task].mean())
            for task in ("gsm8k", "bbh")
        }
        subtasks = {
            subtask: dict(
                questions=sum(
                    row["task"] == "bbh" and row["subtask"] == subtask
                    for row in metadata
                ),
                accuracy=float(
                    rewards[
                        index,
                        :,
                        np.asarray(
                            [
                                row["task"] == "bbh" and row["subtask"] == subtask
                                for row in metadata
                            ]
                        ),
                    ].mean()
                ),
            )
            for subtask in sorted(
                {row["subtask"] for row in metadata if row["task"] == "bbh"}
            )
        }
        quality[name] = dict(
            task_accuracy=groups,
            task_macro_accuracy=sum(groups.values()) / 2,
            bbh_subtask_macro_accuracy=float(
                np.mean([row["accuracy"] for row in subtasks.values()])
            ),
            by_generation_seed={
                str(seed): float(
                    np.mean(
                        [
                            rewards[index, generation, labels == task].mean()
                            for task in ("gsm8k", "bbh")
                        ]
                    )
                )
                for generation, seed in enumerate((42, 137, 2024))
            },
            bbh_subtasks=subtasks,
            source_frames={
                frame: dict(
                    questions=sum(row["source_frame"] == frame for row in metadata),
                    accuracy=float(
                        rewards[
                            index,
                            :,
                            np.asarray(
                                [row["source_frame"] == frame for row in metadata]
                            ),
                        ].mean()
                    ),
                )
                for frame in sorted({row["source_frame"] for row in metadata})
            },
        )
    primary = power_api.paired_question_test(rewards[2], rewards[1], labels)
    primary["corrected_generation_outcomes"] = int(
        ((rewards[2] == 1) & (rewards[1] == 0)).sum()
    )
    primary["damaged_generation_outcomes"] = int(
        ((rewards[2] == 0) & (rewards[1] == 1)).sum()
    )
    primary["net_corrected_generation_outcomes"] = (
        primary["corrected_generation_outcomes"]
        - primary["damaged_generation_outcomes"]
    )
    primary["by_generation_seed_gain"] = {
        str(seed): float(
            np.mean(
                [
                    (
                        rewards[2, generation, labels == task]
                        - rewards[1, generation, labels == task]
                    ).mean()
                    for task in ("gsm8k", "bbh")
                ]
            )
        )
        for generation, seed in enumerate((42, 137, 2024))
    }
    secondary_pairs = ((2, 0), (1, 0), (2, 3), (1, 3), (0, 3))
    secondary = {
        POLICIES[a] + " minus " + POLICIES[b]: power_api.paired_question_test(
            rewards[a], rewards[b], labels
        )
        for a, b in secondary_pairs
    }
    return dict(
        quality=quality,
        primary_comparison="TSPO_uniform minus SPO_coordinate",
        primary=primary,
        secondary=holm_five(secondary),
        primary_alpha=0.05,
        secondary_family_size=5,
        statistical_unit="original question; average three generation repeats before paired inference",
        policy_exception="explicit E18 two-sided paired-question inference approval; E17 unchanged",
    )


def execute_scoring(manifest, context, *, software_test_only=False, runtime_root=None):
    import numpy as np

    gate = context.get("gate", {})
    declared_counts = (
        gate.get("original_questions"),
        gate.get("unique_aggregate_keys"),
        gate.get("unique_member_keys"),
    )
    actual_counts = (
        len(context.get("metadata", ())),
        len(context.get("receipts", ())),
        len(context.get("members", {})),
    )
    if (
        gate.get("status") != "PASS"
        or declared_counts != actual_counts
        or (not software_test_only and declared_counts != (3000, 30921, 47286))
    ):
        raise PermissionError(
            "Complete execution admission is required before reference access"
        )

    reference_api = load_reference_api(manifest["reference_adapter_ref"])
    provider = reference_api.E18ReferenceProvider(
        context["reference_scope"],
        context["approval_ref"],
        software_test_only=software_test_only,
    )
    private = Path(
        runtime_root
        or manifest.get("cpu_output_root")
        or manifest["private_output_root"]
    ).resolve()
    private.mkdir(parents=True, exist_ok=True)
    scorer = context["validator"].SparseScoringAdapter(
        context["approval_ref"],
        context["scoring_scope"],
        provider,
        private / "SCORING_JOURNAL.sqlite3",
        private / "scoring_receipts",
        software_test_only=software_test_only,
        scorer_source_root=context["adapter"].apis.root / "src",
    )
    sparse = []
    try:
        for receipt in context["receipts"]:
            sparse.append(
                scorer.score_one(
                    receipt, context["reference_scope"]["reference_version"]
                )["sparse_row"]
            )
        scoring_ledger = scorer.ledger()
    finally:
        scorer.close()
    metadata = context["metadata"]
    projection = sys.modules["sparse_policy_projection"].project_frozen_policy_scores(
        list(POLICIES),
        np.asarray(context["adapter"].scope["frozen_policy_actions"], dtype=np.int64),
        [row["question_id"] for row in metadata],
        [row["canonical_question_sha256"] for row in metadata],
        sparse,
    )
    rewards = projection["rewards"]
    if rewards.shape != (4, 3, len(metadata)) or projection[
        "unique_required_scored_keys"
    ] != len(context["receipts"]):
        raise ValueError("Final sparse-to-policy projection axes or counts differ")
    power_api = load_statistics(manifest["power_source_ref"])
    statistics = final_statistics(rewards, metadata, power_api)
    costs = list(context["costs"].values())
    totals = {
        stage: dict(
            physical_attempts=sum(row["stage"] == stage for row in costs),
            prompt_tokens=sum(
                row["prompt_tokens"] for row in costs if row["stage"] == stage
            ),
            completion_tokens=sum(
                row["completion_tokens"] for row in costs if row["stage"] == stage
            ),
        )
        for stage in ("member_generation", "aggregation")
    }
    service = {}
    actions = context["adapter"].scope["frozen_policy_actions"]
    for policy_index, policy in enumerate(POLICIES):
        prompt = completion = 0
        for question, row in enumerate(metadata):
            for seed in (42, 137, 2024):
                action = actions[policy_index][question]
                selected = [
                    context["by_key"][(row["question_id"], seed, action)][
                        "native_record"
                    ]
                ]
                selected.extend(
                    context["members"][(row["question_id"], seed, index)]
                    for index in context["module"].TEAMS[action]
                )
                prompt += sum(item["result"]["prompt_tokens"] for item in selected)
                completion += sum(
                    item["result"]["completion_tokens"] for item in selected
                )
        service[policy] = dict(
            member_calls=9 * len(metadata),
            aggregation_calls=3 * len(metadata),
            measured_prompt_tokens=prompt,
            measured_completion_tokens=completion,
            accounting="selected single-policy service projection; study-wide actual payment is separately deduplicated",
        )
    public = dict(
        schema="e18_final_confirmation_results_v1",
        status="COMPLETE",
        original_questions=len(metadata),
        generation_seeds=[42, 137, 2024],
        unique_aggregate_keys=len(context["receipts"]),
        unique_member_keys=len(context["members"]),
        execution_gate=context["gate"],
        results=statistics,
        physical_cost_totals=totals,
        selected_policy_service_costs=service,
        distinct_gold_accounts=scoring_ledger[
            "confirmed_reference_keys_in_scoring_journal"
        ],
        original_source_cached_member_records=sum(
            row["reused_historical_source"] for row in costs
        ),
        reference_provider_stats=provider.stats(),
        scoring_model_calls=0,
        sealed_reference_values_decoded=0,
        sealed_input_values_decoded=0,
        reference_source_access_policy="approved raw-byte SHA/structure scans with selected-reference decoding only",
        finalizer_source_sha256=worker.file_sha(__file__),
        sources={
            key: manifest[key]["sha256"]
            for key in (
                "execution_manifest_ref",
                "reference_scope_ref",
                "scoring_scope_ref",
                "source_validator_ref",
                "reference_adapter_ref",
                "power_source_ref",
                "startup_audit_ref",
            )
        },
        final_manifest_canonical_sha256=context["module"].canonical_digest(manifest),
        operator_fingerprint=context["validator"].OPERATOR_FP,
        aggregator_model_fingerprint=context["validator"].AGGREGATOR_MODEL_FP,
        scorer_version=context["validator"].SCORER_VERSION,
        frozen_sources=dict(
            execution_adapter=worker.ADAPTER_SHA,
            aggregation_worker=worker.file_sha(worker.__file__),
            projection=audit_startup16.PROJECTION_SHA,
            power=POWER_SHA,
            power_common=COMMON_SHA,
            deterministic_scorer=context["validator"].SCORER_SOURCE_SHA256,
            original_scorer=context["validator"].ORIGINAL_SCORER_SOURCE_SHA256,
        ),
        runtime_storage_strategy="explicit native read view and CPU small-file workspace; canonical private archive coordinated externally",
    )
    if public["distinct_gold_accounts"] != len(metadata):
        raise ValueError(
            "Final distinct reference accounts differ from original questions"
        )
    frozen_result = dict(
        schema="e18_completed_scoring_publication_snapshot_v1",
        finalizer_source_sha256=worker.file_sha(__file__),
        manifest_sha256=context["module"].canonical_digest(manifest),
        gate_sha256=context["module"].canonical_digest(context["gate"]),
        reward_sha256=context["module"].canonical_digest(
            rewards.astype(np.int8).tolist()
        ),
        sparse_rows_sha256=context["module"].canonical_digest(sparse),
        public=public,
        scoring_ledger=scoring_ledger,
        reference_access_events=provider.audit_events(),
    )
    snapshot_path = private / "PUBLICATION_SNAPSHOT.json"
    if snapshot_path.exists():
        previous = worker.read_json(snapshot_path)
        fixed = (
            "schema",
            "finalizer_source_sha256",
            "manifest_sha256",
            "gate_sha256",
            "reward_sha256",
            "sparse_rows_sha256",
        )
        if any(previous.get(key) != frozen_result[key] for key in fixed):
            raise ValueError(
                "Completed publication snapshot no longer matches its frozen inputs and scores"
            )
        frozen_result = previous
    else:
        write_new_json(snapshot_path, frozen_result)
    public = frozen_result["public"]
    arrays = dict(
        rewards=rewards.astype(np.int8),
        policy_ids=np.asarray(POLICIES, dtype="U"),
        question_ids=np.asarray([row["question_id"] for row in metadata], dtype="U"),
        canonical_hashes=np.asarray(
            [row["canonical_question_sha256"] for row in metadata], dtype="U"
        ),
        task_labels=np.asarray([row["task"] for row in metadata], dtype="U"),
        subtask_labels=np.asarray(
            [row.get("subtask") or "" for row in metadata], dtype="U"
        ),
        source_frames=np.asarray([row["source_frame"] for row in metadata], dtype="U"),
        frozen_policy_actions=np.asarray(
            context["adapter"].scope["frozen_policy_actions"], dtype=np.int64
        ),
    )
    write_npz_once(private / "FINAL_POLICY_REWARDS.npz", arrays)
    write_new_json(private / "SPARSE_SCORED_ROWS.json", sparse)
    write_new_json(private / "SCORING_LEDGER.json", frozen_result["scoring_ledger"])
    write_new_json(
        private / "PHYSICAL_COST_LEDGER.json", dict(rows=costs, totals=totals)
    )
    write_new_json(
        private / "REFERENCE_ACCESS_EVENTS.json",
        frozen_result["reference_access_events"],
    )
    write_new_json(
        private / "RUNTIME_IO_METADATA.json",
        dict(
            context["runtime_io"],
            physical_cpu_output_root=str(private),
            canonical_private_output_root=manifest["private_output_root"],
            archive_copy_decision="root archives immutable scorer/native small files to canonical private storage after completion",
        ),
    )
    write_new_json(private / "FINAL_RESULTS_PRIVATE.json", public)
    write_new_json(
        Path(manifest["public_output_root"]) / "E18_FINAL_RESULTS.json", public
    )
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--native-read-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute-scoring", action="store_true")
    args = parser.parse_args()
    if worker.file_sha(args.manifest) != args.manifest_sha256:
        raise ValueError("Final manifest bytes differ")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    manifest = worker.read_json(args.manifest)
    started = time.perf_counter()
    context = execution_gate(manifest, native_read_root=args.native_read_root)
    private = Path(
        args.runtime_root
        or manifest.get("cpu_output_root")
        or manifest["private_output_root"]
    )
    write_new_json(
        private / ("PRE_SCORING_ADMISSION_" + str(time.time_ns()) + ".json"),
        context["gate"],
    )
    if args.preflight_only:
        print(
            json.dumps(
                dict(
                    status="EXECUTION_COMPLETE_REFERENCE_UNREAD",
                    gate=context["gate"],
                    wall_seconds=time.perf_counter() - started,
                )
            )
        )
        return
    result = execute_scoring(manifest, context, runtime_root=args.runtime_root)
    print(
        json.dumps(
            dict(
                status=result["status"],
                original_questions=result["original_questions"],
                wall_seconds=time.perf_counter() - started,
            )
        )
    )


if __name__ == "__main__":
    main()
