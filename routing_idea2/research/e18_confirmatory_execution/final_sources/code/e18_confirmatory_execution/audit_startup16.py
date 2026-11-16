"""Read-only startup aggregation identity audit without scoring or Gold access."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import e18_aggregation_worker_v3 as worker
import e18_aggregation_dispatcher_v3 as dispatcher


SCORING_ADAPTER_SHA = "7eb390f0424f2f18fda806c188e0f3e8e7ed2fc633605c5a3d170d271ca06d8a"
PROJECTION_SHA = "098df2f6a5ec6e014473516bab1fa136efbe35a9258803f2ec25c9e43cb9cb6f"


def load_source_validator(reference):
    path = worker.checked_ref(reference)
    if reference["sha256"] != SCORING_ADAPTER_SHA:
        raise ValueError("Frozen execution-source validator identity differs")
    projection = path.parent / "sparse_policy_projection.py"
    if worker.file_sha(projection) != PROJECTION_SHA:
        raise ValueError("Frozen validator constant dependency identity differs")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(
        "e18_readonly_execution_source_validator", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (
        Path(sys.modules["sparse_policy_projection"].__file__).resolve()
        != projection.resolve()
    ):
        raise ValueError("Another validator constant module is already loaded")
    return module


def tokenizer_only(adapter_module, adapter, manifest):
    config = manifest["backend"]
    if config["mode"] != "mapped":
        raise ValueError(
            "This startup audit requires the actual mapped deployment manifest"
        )
    helper = worker.checked_ref(config["helper_ref"])
    if worker.file_sha(helper) != adapter_module.MAPPING_HELPER_SHA:
        raise ValueError("Original audited model loading-path helper differs")
    mapping = worker.checked_ref(config["mapping_ref"])
    spec = importlib.util.spec_from_file_location(
        "e18_startup_mapping_metadata", helper
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mapper = module.MappedBackend(
        config["physical_root"],
        mapping,
        config["mapping_ref"]["sha256"],
        adapter.apis.inference,
    )
    if mapper.logical != adapter.aggregator_spec:
        raise ValueError("Mapped tokenizer belongs to another logical aggregator")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        mapper.physical.model_path, local_files_only=True, trust_remote_code=False
    )
    alias = worker.restore_logical_tokenizer_path(
        tokenizer,
        adapter.aggregator_spec,
        mapper.receipt,
        worker.frozen_tokenizer_files(adapter.apis),
        adapter.apis.execution,
        mapping_provenance=dict(
            mapping_sha256=config["mapping_ref"]["sha256"],
            adapter_source_sha256=config["helper_ref"]["sha256"],
        ),
    )
    if mapper.constructed:
        raise RuntimeError("A read-only audit must never construct a language model")
    return tokenizer, alias


def audit_plan(
    plan_ref,
    validation_ref,
    *,
    expected_tasks=16,
    software_test_only=False,
    fixture_factory=None,
    fixture_tokenizer=None,
):
    if not software_test_only and (
        fixture_factory is not None or fixture_tokenizer is not None
    ):
        raise PermissionError(
            "Production audit uses original metadata and verified HF tokenizer only"
        )
    validator = load_source_validator(validation_ref)
    plan = worker.read_json(worker.checked_ref(plan_ref))
    manifests = dispatcher.admit_plan(plan)
    rows = []
    receipts = []
    dependencies = {}
    tokenizer = fixture_tokenizer
    alias = None
    identity = None
    scope = None
    for _, manifest in manifests:
        module, adapter, keys, _, _, external = worker.load_worker(
            manifest,
            software_test_only=software_test_only,
            injected_factory=fixture_factory,
        )
        adapter._worker_software_mode = software_test_only
        current = (
            module.canonical_digest(adapter.scope),
            manifest["approval_ref"]["sha256"],
            str(Path(manifest["owned_output_root"]).resolve()),
            manifest["backend"],
        )
        if identity is None:
            identity, scope = current, adapter.scope
        elif identity != current:
            raise ValueError("Startup manifests have different execution identities")
        if tokenizer is None:
            tokenizer, alias = tokenizer_only(module, adapter, manifest)
        output_root = Path(manifest["owned_output_root"]).resolve()
        for key in keys:
            qid, generation, combination = key
            selected = {
                index: worker.member_dependency(
                    module, adapter, output_root, external, (qid, generation, index)
                )
                for index in module.TEAMS[combination]
            }
            job = adapter.aggregation_job(
                qid, generation, combination, selected, tokenizer
            )
            owner = (
                output_root
                / current[0]
                / "aggregation"
                / module.canonical_digest(list(key))
            )
            receipt = worker.read_json(owner / "LAST_TASK_RECEIPT.json")
            source = validator.verified_aggregation_source(receipt)
            source.pop("prediction")
            if (
                receipt["software_test_only"] is not software_test_only
                or receipt["scope_sha256"] != current[0]
                or receipt["approval_sha256"] != current[1]
                or receipt["task_key"]
                != dict(
                    question_id=qid,
                    canonical_id=adapter.questions[qid]["canonical_question_sha256"],
                    generation_seed=generation,
                    combination_index=combination,
                    team=list(module.TEAMS[combination]),
                )
            ):
                raise ValueError(
                    "Startup receipt does not belong to its approved sparse key"
                )
            native = receipt["native_record"]
            if job["job_id"] != native["job_id"]:
                raise ValueError(
                    "Rebuilt typed prompt and anonymous order differ from paid job identity"
                )
            adapter.apis.aggregation.ExecutionStore._cached(
                SimpleNamespace(completed={native["job_id"]: native}),
                job,
                adapter.aggregator_spec,
            )
            adapter.apis.aggregation._validate_result(
                adapter.apis.inference.GenerationResult(**native["result"]),
                adapter.aggregator_spec,
                job,
            )
            for index, record in selected.items():
                if (
                    record["seed"] != generation
                    or record["sample_index"] != 0
                    or record["sampling_control"] is not False
                ):
                    raise ValueError(
                        "Member generation or ordinary sample protocol differs"
                    )
                dependencies[record["record_sha256"]] = record
            rows.append(
                dict(
                    task_key_sha256=module.canonical_digest(list(key)),
                    generation_seed=generation,
                    combination_index=combination,
                    aggregation_seed=job["seed"],
                    job_id=job["job_id"],
                    record_sha256=native["record_sha256"],
                    actual_prompt_tokens=source["prompt_tokens"],
                    actual_completion_tokens=source["completion_tokens"],
                    member_record_sha256=[
                        selected[index]["record_sha256"]
                        for index in module.TEAMS[combination]
                    ],
                    typed_prompt_exact_rebuild=True,
                    anonymous_order_exact_rebuild=True,
                )
            )
            receipts.append(receipt)
    if len(rows) != expected_tasks or len(
        {row["task_key_sha256"] for row in rows}
    ) != len(rows):
        raise ValueError(
            "Startup task count or uniqueness differs from the audit request"
        )
    groups = {}
    for qid, seed, combination in scope["required_aggregate_keys"]:
        groups.setdefault((qid, combination), set()).add(seed)
    if any(
        seeds
        != set(worker.load_adapter(manifests[0][1]["adapter_ref"]).GENERATION_SEEDS)
        for seeds in groups.values()
    ):
        raise ValueError(
            "Approved aggregate union does not preserve all three generation repeats"
        )
    ledger = validator.resource_ledger_from_execution_receipts(receipts)
    if (
        ledger["failed_attempts"]
        or ledger["unknown_usage_attempts"]
        or ledger["unique_physical_attempts"] != len(rows)
    ):
        raise ValueError(
            "Startup physical usage has duplicate or unresolved executions"
        )
    return dict(
        schema="e18_startup_aggregation_scientific_identity_audit_v1",
        status="PASS",
        plan_ref=plan_ref,
        source_validator_sha256=validation_ref["sha256"],
        audited_tasks=len(rows),
        operator_fingerprint=worker.load_adapter(
            manifests[0][1]["adapter_ref"]
        ).OPERATOR_FP,
        aggregator_model_fingerprint=validator.AGGREGATOR_MODEL_FP,
        startup_generation_seed_counts={
            str(seed): sum(row["generation_seed"] == seed for row in rows)
            for seed in (42, 137, 2024)
        },
        full_scope_three_repeat_pairs=len(groups),
        full_scope_aggregate_keys=len(scope["required_aggregate_keys"]),
        unique_reused_member_records=len(dependencies),
        member_sample_index=0,
        aggregation_seed=42,
        aggregate_physical_attempts=ledger["unique_physical_attempts"],
        actual_aggregate_prompt_tokens=ledger["total_actual_input_tokens"],
        actual_aggregate_completion_tokens=ledger["total_actual_output_tokens"],
        tokenizer_loading_path_qualification=alias,
        tasks=rows,
        correctness_evaluation="NOT_PERFORMED",
        gold_reads=0,
        scorer_calls=0,
        language_model_constructions=0,
        language_model_generations=0,
        existing_model_predictions_read_only=True,
        software_test_only=software_test_only,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--source-validator", type=Path, required=True)
    parser.add_argument("--source-validator-sha256", required=True)
    parser.add_argument("--expected-tasks", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output is append-only")
    result = audit_plan(
        dict(path=str(args.plan), sha256=args.plan_sha256),
        dict(path=str(args.source_validator), sha256=args.source_validator_sha256),
        expected_tasks=args.expected_tasks,
    )
    result["audit_source_sha256"] = worker.file_sha(__file__)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                status=result["status"],
                audited_tasks=result["audited_tasks"],
                scorer_calls=0,
            )
        )
    )


if __name__ == "__main__":
    main()
