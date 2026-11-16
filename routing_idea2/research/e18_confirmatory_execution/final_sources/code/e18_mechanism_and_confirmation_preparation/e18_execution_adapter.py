"""Isolated E18 scope factories and an approval-gated single-task worker.

The source V2 modules, global split manifest and ALLOWED_SPLITS are untouched.
New tasks use the existing online_stream transport while their owning cohort is
e18_confirm. Real backend construction is deferred until explicit approval.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
from itertools import combinations
import json
from numbers import Integral
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import unicodedata


OPERATOR_FP = "9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e"
AGGREGATOR_MODEL_FP = "32aed6079298ebcf8eee99add17dd5c4a6828a504f13dd5d9236b234776861d3"
MAPPING_HELPER_SHA = "4ea0b3f213e4abe70a388e5f3d86cfe92b86eabb1db4f550ea0c57758febb845"
COHORT = "e18_confirm"
TRANSPORT_SPLIT = "online_stream"
GENERATION_SEEDS = (42, 137, 2024)
TEAMS = tuple(combinations(range(6), 3))
OPERATIONS = frozenset(
    {"member_generation", "aggregation", "read_reference", "score_outputs"}
)
REQUIRED_BINDINGS = frozenset(
    {
        "protocol_sha256",
        "target_inputs_sha256",
        "frozen_actions_sha256",
        "budget_sha256",
        "target_ids_sha256",
    }
)
CORE_SHAS = {
    "src/budget_router/member_execution.py": "b1eac1cb346f7c67f14de398df130c644193c4f527a33b00ab18de1844f6dfb2",
    "src/budget_router/execution.py": "2dd0fc4bf9e36eb63e4ade5143d07505e14780e59338230ea86817ddd8eae17f",
    "scripts/v2/typed_aggregation_execution.py": "acbc9cd911635f26da0e23cce4174bf6660a7604d29a5d7224f1df08e083f368",
}


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def checked_reference(reference):
    if (
        not isinstance(reference, dict)
        or set(reference) != {"path", "sha256"}
        or not valid_sha(reference["sha256"])
    ):
        raise ValueError("An explicit path and SHA256 reference is required")
    path = Path(reference["path"]).resolve(strict=True)
    if not path.is_file() or sha256(path) != reference["sha256"]:
        raise ValueError("Referenced artifact bytes differ")
    return path


def validate_scope_approval(
    approval_ref, scope_contract, operation, *, software_test_only=False
):
    """Validate one component of a hash-bound explicit approval receipt.

    The trusted caller supplies approval_ref after the user's approval. This
    routine does not infer approval from elapsed time, a plan or a PASS status.
    It performs no Gold, input-source or model access.
    """
    if operation not in OPERATIONS or type(software_test_only) is not bool:
        raise ValueError("Unknown approval operation or test mode")
    if (
        scope_contract.get("schema") != "e18_scope_contract_v2"
        or scope_contract.get("cohort") != COHORT
    ):
        raise ValueError("Unexpected E18 scope contract")
    role = scope_contract.get("role")
    if role not in {"execution", "reference", "scoring"}:
        raise ValueError("Unknown E18 scope role")
    bindings = scope_contract.get("bindings", {})
    if not REQUIRED_BINDINGS <= set(bindings) or not all(
        map(valid_sha, bindings.values())
    ):
        raise ValueError(
            "Complete protocol/input/action/budget/target bindings required"
        )
    approval = read_json(checked_reference(approval_ref))
    if (
        approval.get("schema") != "e18_scope_approval_v2"
        or approval.get("status") != "APPROVED"
        or approval.get("cohort") != COHORT
    ):
        raise PermissionError("Explicit E18 approval receipt required")
    if approval.get("software_test_only") is not software_test_only:
        raise PermissionError(
            "Synthetic approval cannot authorize production execution"
        )
    if operation not in approval.get("operations", []):
        raise PermissionError("Requested operation has not been approved")
    if approval.get("scope_sha256_by_role", {}).get(role) != canonical_digest(
        scope_contract
    ):
        raise PermissionError("Approved component scope differs")
    approved_bindings = approval.get("bindings", {})
    if any(approved_bindings.get(key) != value for key, value in bindings.items()):
        raise PermissionError("Approved protocol or source bindings differ")
    return approval


def load_original_apis(asset_root):
    """Load byte-pinned factories and Stores without constructing a model."""
    root = Path(asset_root).resolve(strict=True)
    for relative, expected in CORE_SHAS.items():
        if sha256(root / relative) != expected:
            raise ValueError("Original scientific source changed " + relative)
    sys.path[:0] = [str(root / "src"), str(root / "scripts/v2")]
    import budget_router.inference as inference
    import budget_router.execution as execution
    import budget_router.member_execution as member
    import typed_aggregation_execution as typed

    for module, relative in (
        (member, "src/budget_router/member_execution.py"),
        (execution, "src/budget_router/execution.py"),
        (typed, "scripts/v2/typed_aggregation_execution.py"),
    ):
        if Path(module.__file__).resolve() != (root / relative).resolve():
            raise ValueError("Loaded another scientific source tree")
    aggregation = typed.load_execution_api()
    # Runtime identity checking uses original read-only helpers only. Importing
    # CPU preparation modules here could alter CUDA visibility before loading.
    from stage1_review_common import operator_identity
    from typed_operator_contract import validate_typed_freeze_header
    from budget_router_v2.scoring import BBH_TASKS, SCORER_VERSION

    lock_path = (
        root
        / "results/decision_feedback_v2/s1_typed_aggcal_generation42_9ef6c02bc9fb/OPERATOR_FREEZE_LOCK_v1.json"
    )
    if (
        sha256(lock_path)
        != "66d77c2ab5eef0c6a360cfe2d8df01429369927c96fe3ae7b6a63697f4fc09b0"
    ):
        raise ValueError("Complete operator lock bytes differ")
    frozen = validate_typed_freeze_header(read_json(lock_path))
    current = operator_identity(
        root, root / "artifacts/private/routing_redesign_20261104/stage1_run_20261104"
    )
    current["schema"] = "cross_machine_typed_aggregator_candidate_identity_v1"
    current["prompt"]["base_version"] = current["prompt"]["version"]
    current["prompt"]["version"] = typed.PROMPT_VERSION
    current["prompt"]["contract_version"] = typed.CONTRACT_VERSION
    contracts = {"gsm8k": typed.output_contract("gsm8k", None)}
    contracts.update(
        {
            "bbh/" + name: typed.output_contract("bbh", name)
            for name in sorted(BBH_TASKS)
        }
    )
    current["prompt"]["contracts_by_task_subtask"] = contracts
    current["prompt"]["contracts_sha256"] = canonical_digest(contracts)
    current["prompt"]["typed_adapter_source_sha256"] = sha256(
        root / "scripts/v2/typed_aggregation_execution.py"
    )
    current["scorer"] = {
        "version": SCORER_VERSION,
        "source_sha256": sha256(root / "src/budget_router_v2/scoring.py"),
    }
    if current != frozen or canonical_digest(current) != OPERATOR_FP:
        raise ValueError("Runtime typed operator differs from its full frozen identity")
    for relative, expected in frozen["original_source_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError("Runtime original scientific source changed")
    qualified = {
        "active_typed_operator_fingerprint": canonical_digest(current),
        "original_scientific_sources_verified": len(frozen["original_source_sha256"]),
        "typed_contracts": len(contracts),
    }
    return SimpleNamespace(
        root=root,
        inference=inference,
        member=member,
        execution=execution,
        typed=typed,
        aggregation=aggregation,
        operator_qualification=qualified,
    )


def canonical_question(question):
    normalized = " ".join(unicodedata.normalize("NFKC", question).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def build_scope_contract(
    question_records,
    policy_actions,
    bindings,
    member_execution_keys,
    aggregate_execution_keys,
):
    """Build metadata only. Questions and native prompt/job payloads stay out."""
    records = list(question_records)
    ids = [row["question_id"] for row in records]
    canonical = [row["canonical_question_sha256"] for row in records]
    if (
        not records
        or len(set(ids)) != len(ids)
        or len(set(canonical)) != len(canonical)
    ):
        raise ValueError("Unique target question identities required")
    actions = [list(row) for row in policy_actions]
    if len(actions) != 4 or any(len(row) != len(records) for row in actions):
        raise ValueError("Exactly four frozen policy action rows required")
    if any(
        not isinstance(action, Integral)
        or isinstance(action, bool)
        or not 0 <= action < 20
        for row in actions
        for action in row
    ):
        raise ValueError("Frozen combination indices must be integers in0..19")
    actions = [[int(action) for action in row] for row in actions]
    target_ids_sha = canonical_digest(
        [[qid, digest] for qid, digest in zip(ids, canonical)]
    )
    if bindings.get("target_ids_sha256") != target_ids_sha:
        raise ValueError("Target identity binding differs")
    members = set()
    aggregates = set()
    for index, record in enumerate(records):
        for action in {row[index] for row in actions}:
            for seed in GENERATION_SEEDS:
                aggregates.add((record["question_id"], seed, action))
                members.update(
                    (record["question_id"], seed, member) for member in TEAMS[action]
                )
    member_rows = list(map(tuple, member_execution_keys))
    aggregate_rows = list(map(tuple, aggregate_execution_keys))
    member_calls = set(member_rows)
    aggregate_calls = set(aggregate_rows)
    if len(member_calls) != len(member_rows) or len(aggregate_calls) != len(
        aggregate_rows
    ):
        raise ValueError("Duplicate execution ownership keys are forbidden")
    if not member_calls <= members or not aggregate_calls <= aggregates:
        raise ValueError("Execution keys exceed the frozen policy union")
    return dict(
        schema="e18_scope_contract_v2",
        role="execution",
        cohort=COHORT,
        bindings=dict(bindings),
        transport_split=TRANSPORT_SPLIT,
        operator_fingerprint=OPERATOR_FP,
        question_keys=[
            dict(
                question_id=row["question_id"],
                canonical_question_sha256=row["canonical_question_sha256"],
                question_sha256=hashlib.sha256(row["question"].encode()).hexdigest(),
                task=row["task"],
                subtask=row.get("subtask") or "",
            )
            for row in records
        ],
        frozen_policy_actions=actions,
        required_member_keys=[list(key) for key in sorted(members)],
        required_aggregate_keys=[list(key) for key in sorted(aggregates)],
        member_execution_keys=[list(key) for key in sorted(member_calls)],
        aggregate_execution_keys=[list(key) for key in sorted(aggregate_calls)],
        execution_limits=dict(
            member_generation=len(member_calls), aggregation=len(aggregate_calls)
        ),
        generation_seeds=list(GENERATION_SEEDS),
        aggregation_seed=42,
        authorization_state="PREPARATION_ONLY_UNTIL_APPROVAL",
    )


class E18ExecutionAdapter:
    def __init__(
        self, apis, scope_contract, question_records, member_specs, aggregator_spec
    ):
        self.apis = apis
        self.scope = json.loads(json.dumps(scope_contract))
        if (
            self.scope.get("schema") != "e18_scope_contract_v2"
            or self.scope.get("role") != "execution"
            or self.scope.get("cohort") != COHORT
        ):
            raise ValueError("Exact E18 execution scope required")
        if (
            self.scope.get("transport_split") != TRANSPORT_SPLIT
            or self.scope.get("operator_fingerprint") != OPERATOR_FP
        ):
            raise ValueError("Transport or scientific operator differs")
        self.questions = {row["question_id"]: dict(row) for row in question_records}
        expected = self.scope["question_keys"]
        self.question_metadata = {row["question_id"]: row for row in expected}
        if len(self.questions) != len(expected) or set(self.questions) != {
            row["question_id"] for row in expected
        }:
            raise ValueError("Input cohort differs from approved target identities")
        forbidden = {
            "gold",
            "reference",
            "reference_answer",
            "answer_key",
            "score",
            "scores",
            "label",
            "labels",
            "reward",
        }
        for row in expected:
            actual = self.questions[row["question_id"]]
            if forbidden & set(actual) or actual["task"] not in {"gsm8k", "bbh"}:
                raise ValueError(
                    "Generation inputs must contain input-only task metadata"
                )
            if (
                canonical_question(actual["question"])
                != row["canonical_question_sha256"]
                or actual["canonical_question_sha256"]
                != row["canonical_question_sha256"]
            ):
                raise ValueError("Original question bytes do not match canonical scope")
        rebuild = build_scope_contract(
            list(self.questions.values()),
            self.scope["frozen_policy_actions"],
            self.scope["bindings"],
            self.scope["member_execution_keys"],
            self.scope["aggregate_execution_keys"],
        )
        if rebuild != self.scope:
            raise ValueError("Frozen union or execution limits were altered")
        self.member_specs = dict(member_specs)
        if set(self.member_specs) != set(range(6)):
            raise ValueError("Exactly six member specs are required")
        pool = read_json(apis.root / "configs/pool_v1.json")["models"]
        self.members = {entry["index"]: entry for entry in pool}
        for index, spec in self.member_specs.items():
            if (
                apis.inference.configuration_fingerprint(spec)
                != self.members[index]["fingerprint"]
            ):
                raise ValueError("Original member model profile differs")
            if (
                spec.temperature != 0.7
                or spec.top_p != 0.95
                or spec.max_new_tokens != 2048
                or spec.enable_thinking is not False
            ):
                raise ValueError("Original member sampling profile differs")
        if (
            apis.inference.configuration_fingerprint(aggregator_spec)
            != AGGREGATOR_MODEL_FP
        ):
            raise ValueError("Original aggregator logical model profile differs")
        self.aggregator_spec = aggregator_spec
        self.required_members = set(map(tuple, self.scope["required_member_keys"]))
        self.required_aggregates = set(
            map(tuple, self.scope["required_aggregate_keys"])
        )
        self.callable_members = set(map(tuple, self.scope["member_execution_keys"]))
        self.callable_aggregates = set(
            map(tuple, self.scope["aggregate_execution_keys"])
        )

    def member_job(self, question_id, generation_seed, member_index):
        if type(generation_seed) is not int or type(member_index) is not int:
            raise ValueError("Explicit integer member generation and index required")
        key = (question_id, generation_seed, member_index)
        if key not in self.required_members:
            raise ValueError("Member key is outside the frozen policy union")
        record = self.questions[question_id]
        metadata = self.question_metadata[question_id]
        if (
            hashlib.sha256(record["question"].encode()).hexdigest()
            != metadata["question_sha256"]
            or record["task"] != metadata["task"]
            or (record.get("subtask") or "") != metadata["subtask"]
        ):
            raise ValueError("Frozen raw input or task metadata changed")
        if (
            self.apis.inference.configuration_fingerprint(
                self.member_specs[member_index]
            )
            != self.members[member_index]["fingerprint"]
        ):
            raise ValueError("Frozen member profile changed")
        return self.apis.member.make_member_job(
            question_id=question_id,
            task=record["task"],
            question=record["question"],
            subtask=record.get("subtask") or None,
            model_alias=self.members[member_index]["alias"],
            seed=generation_seed,
            model_spec=self.member_specs[member_index],
            split=TRANSPORT_SPLIT,
            sample_index=0,
            sampling_control=False,
        )

    def checked_member_dependency(
        self, question_id, generation_seed, member_index, record
    ):
        expected = self.member_job(question_id, generation_seed, member_index)
        if (
            record.get("status") != "success"
            or record.get("cost_status") != "recorded_actual_usage"
        ):
            raise ValueError("Successful fully recorded member dependencies required")
        if record.get("record_sha256") != canonical_digest(
            {key: value for key, value in record.items() if key != "record_sha256"}
        ):
            raise ValueError("Member execution record bytes differ")
        # A paid fresh400 cache has its original transport split and job ID.
        # Rebuild that source job without changing any prompt/scientific field.
        if record.get("split") not in {TRANSPORT_SPLIT, "fresh_dev_test"}:
            raise ValueError("Undeclared source transport split")
        source_job = dict(expected)
        source_job["split"] = record["split"]
        source_job["job_id"] = canonical_digest(
            {key: value for key, value in source_job.items() if key != "job_id"}
        )
        if source_job["job_id"] != record.get("job_id"):
            raise ValueError("Paid source member job differs beyond transport metadata")
        spec = self.member_specs[member_index]
        cached = self.apis.member.MemberStore._cached(
            SimpleNamespace(completed={record["job_id"]: record}), source_job, spec
        )
        self.apis.execution._validate_result(
            self.apis.inference.GenerationResult(**cached["result"]), spec, source_job
        )
        return cached

    def aggregation_job(
        self, question_id, generation_seed, combination_index, member_records, tokenizer
    ):
        if type(generation_seed) is not int or type(combination_index) is not int:
            raise ValueError(
                "Explicit integer member generation and combination required"
            )
        key = (question_id, generation_seed, combination_index)
        if key not in self.required_aggregates:
            raise ValueError("Aggregation key is outside the frozen policy union")
        if (
            self.apis.inference.configuration_fingerprint(self.aggregator_spec)
            != AGGREGATOR_MODEL_FP
        ):
            raise ValueError("Frozen aggregator profile changed")
        team = TEAMS[combination_index]
        if set(member_records) != set(team):
            raise ValueError(
                "Exactly the selected three member dependencies are required"
            )
        chosen = {
            index: self.checked_member_dependency(
                question_id, generation_seed, index, member_records[index]
            )
            for index in team
        }
        answers = {
            self.members[index]["alias"]: row["result"]["text"]
            for index, row in chosen.items()
        }
        provenance = {
            self.members[index]["alias"]: dict(
                job_id=row["job_id"],
                record_sha256=row["record_sha256"],
                model_id=row["result"]["model_id"],
                revision=row["result"]["model_revision"],
                model_fingerprint=row["model_fingerprint"],
            )
            for index, row in chosen.items()
        }
        item = self.questions[question_id]
        job = self.apis.aggregation.make_job(
            question_id=question_id,
            task=item["task"],
            subtask=item.get("subtask") or None,
            question=item["question"],
            answers=answers,
            seed=42,
            model_spec=self.aggregator_spec,
            split=TRANSPORT_SPLIT,
            member_provenance=provenance,
            tokenizer=tokenizer,
        )
        self.apis.aggregation.validate_job(
            job, self.aggregator_spec, require_token_policy=True
        )
        return job


def _atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_approved_task(
    adapter,
    task,
    approval_ref,
    output_root,
    backend_factory,
    *,
    tokenizer=None,
    member_records=None,
    software_test_only=False,
):
    """Run or reuse one exact task after approval; never retry a failed owner.

    backend_factory(logical_spec) is lazy and must retain the logical spec. A
    A production caller supplies the original source-validated local factory
    or the audited loading-path mapping factory.
    The future multiworker/native transport dispatcher is a separate layer.
    """
    stage = task.get("stage")
    if stage not in {"member_generation", "aggregation"}:
        raise ValueError("Single-stage member or aggregation task required")
    required_fields = {
        "stage",
        "question_id",
        "generation_seed",
        "member_index" if stage == "member_generation" else "combination_index",
    }
    if set(task) != required_fields:
        raise ValueError("Exact label-free single-task request fields required")
    approval = validate_scope_approval(
        approval_ref, adapter.scope, stage, software_test_only=software_test_only
    )
    required_cap = adapter.scope["execution_limits"][stage]
    if approval.get("execution_limits", {}).get(stage, -1) < required_cap:
        raise PermissionError(
            "Approval does not cover the frozen stage execution budget"
        )
    approved_root = approval.get("approved_output_root")
    if not isinstance(approved_root, str) or not Path(approved_root).is_absolute():
        raise PermissionError("Approval must bind an absolute execution output root")
    owned_root = Path(output_root).resolve()
    if owned_root != Path(approved_root).resolve():
        raise PermissionError("Execution output root differs from the approved owner")
    qid = task["question_id"]
    generation = task["generation_seed"]
    index = (
        task["member_index"]
        if stage == "member_generation"
        else task["combination_index"]
    )
    key = (qid, generation, index)
    if stage == "member_generation":
        job = adapter.member_job(*key)
        spec = adapter.member_specs[index]
        store_type = adapter.apis.member.MemberStore
        callable_keys = adapter.callable_members
    else:
        job = adapter.aggregation_job(*key, member_records or {}, tokenizer)
        spec = adapter.aggregator_spec
        store_type = adapter.apis.aggregation.ExecutionStore
        callable_keys = adapter.callable_aggregates
    scope_sha = canonical_digest(adapter.scope)
    owner = owned_root / scope_sha / stage / canonical_digest(list(key))
    owner.mkdir(parents=True, exist_ok=True)
    store = store_type(owner / "records.jsonl")
    fence = owner / "ATTEMPT_STATE.json"
    cached = store._cached(job, spec)
    if cached is None:
        if key not in callable_keys:
            raise PermissionError(
                "Existing-cache key is not admitted for a new paid call"
            )
        if store.unreconciled or fence.exists():
            raise RuntimeError(
                "Prior reserved/failed/unknown-cost task requires reconciliation; no retry"
            )
        reservation = dict(
            status="RESERVED_BEFORE_BACKEND",
            scope_sha256=scope_sha,
            approval_sha256=approval_ref["sha256"],
            task_key=list(key),
            stage=stage,
            native_job_id=job["job_id"],
            software_test_only=software_test_only,
        )
        try:
            with fence.open("x", encoding="utf-8") as stream:
                json.dump(reservation, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            raise RuntimeError(
                "Another owner already reserved this semantic task"
            ) from None
        try:
            backend = backend_factory(spec)
            if backend.spec != spec:
                raise ValueError("Backend must expose the original logical ModelSpec")
            if not software_test_only:
                provenance = getattr(backend, "server_runtime_provenance", {})
                receipt = getattr(backend, "source_receipt", {})
                original_backend = (
                    type(backend) is adapter.apis.inference.TransformersBackend
                )
                direct_original_source = (
                    original_backend
                    and receipt.get("repo_id") == spec.model_id
                    and receipt.get("revision") == spec.revision
                    and isinstance(receipt.get("path"), str)
                    and Path(receipt["path"]).resolve()
                    == Path(spec.model_path).resolve()
                )
                mapped_original_source = (
                    original_backend
                    and receipt.get("repo_id") == spec.model_id
                    and receipt.get("revision") == spec.revision
                    and provenance.get("logical_fingerprint")
                    == adapter.apis.inference.configuration_fingerprint(spec)
                    and provenance.get("adapter_source_sha256") == MAPPING_HELPER_SHA
                    and valid_sha(provenance.get("mapping_sha256"))
                    and valid_sha(provenance.get("physical_loading_fingerprint"))
                    and provenance.get("original_source_validator_unchanged") is True
                    and provenance.get("generation_method_unchanged") is True
                )
                if not (direct_original_source or mapped_original_source):
                    raise ValueError(
                        "Production requires the original source-validated backend or its audited loading-path mapping"
                    )
            validate_scope_approval(
                approval_ref,
                adapter.scope,
                stage,
                software_test_only=software_test_only,
            )
            row, reused = store.run(backend, job)
        except BaseException:
            _atomic_json(
                fence, {**reservation, "status": "STOP_REQUIRES_RECONCILIATION"}
            )
            raise
        _atomic_json(
            fence,
            {
                **reservation,
                "status": "SUCCESS_COMMITTED"
                if row["status"] == "success"
                else "STOP_REQUIRES_RECONCILIATION",
                "native_record_sha256": row["record_sha256"],
            },
        )
    else:
        if not fence.is_file():
            raise RuntimeError("Cached owner lacks its committed execution fence")
        committed = read_json(fence)
        expected_fence = {
            "status": "SUCCESS_COMMITTED",
            "scope_sha256": scope_sha,
            "task_key": list(key),
            "stage": stage,
            "native_job_id": job["job_id"],
            "native_record_sha256": cached["record_sha256"],
            "software_test_only": software_test_only,
        }
        if any(
            committed.get(field) != value for field, value in expected_fence.items()
        ):
            raise RuntimeError(
                "Cached owner fence has incompatible task, mode or commit"
            )
        row, reused = cached, True
    item = adapter.questions[qid]
    task_key = dict(
        question_id=qid,
        canonical_id=item["canonical_question_sha256"],
        generation_seed=generation,
    )
    task_key.update(
        member_index=index
    ) if stage == "member_generation" else task_key.update(
        combination_index=index, team=list(TEAMS[index])
    )
    result = dict(
        schema="e18_execution_task_receipt_v2",
        status="SUCCESS"
        if row["status"] == "success"
        else "FAILED_REQUIRES_RECONCILIATION",
        scope_sha256=scope_sha,
        approval_sha256=approval_ref["sha256"],
        approved_output_root=str(owned_root),
        bindings=adapter.scope["bindings"],
        cohort=COHORT,
        transport_split=TRANSPORT_SPLIT,
        stage=stage,
        task_key=task_key,
        aggregator_fingerprint=OPERATOR_FP,
        aggregation_seed=42,
        prompt_version=job["prompt_version"],
        native_job_id=job["job_id"],
        native_record_sha256=row["record_sha256"],
        native_record=row,
        physical_attempt_id=row["attempt_id"],
        from_cache=reused,
        software_test_only=software_test_only,
        actual_gpu_runtime_status="NOT_RUNTIME_TESTED"
        if software_test_only
        else "CALLER_QUALIFIED_BACKEND",
        root_scope_dispatcher_status="NOT_IMPLEMENTED",
    )
    _atomic_json(owner / "LAST_TASK_RECEIPT.json", result)
    return result


def mapped_backend_factory(asset_root, physical_root, mapping_ref, helper_ref):
    """Return a lazy wrapper around the original private MappedBackend.

    Its original constructor validates E0, physical source receipts and loading
    path mapping. A mapping/helper reference is required per model profile.
    This function constructs no backend until the returned callable is invoked.
    """
    root = Path(asset_root).resolve(strict=True)
    helper = checked_reference(helper_ref)
    original = root / "artifacts/private/server_migration_20261102/mapped_backend_v1.py"
    if sha256(helper) != MAPPING_HELPER_SHA or sha256(original) != MAPPING_HELPER_SHA:
        raise ValueError(
            "Mapping helper differs from the audited original implementation"
        )
    mapping = checked_reference(mapping_ref)

    def create(logical_spec):
        apis = load_original_apis(root)
        spec = importlib.util.spec_from_file_location(
            "e18_original_mapping_backend", helper
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        mapper = module.MappedBackend(
            physical_root, mapping, mapping_ref["sha256"], apis.inference
        )
        return mapper.make_backend(logical_spec)

    return create


def original_backend_factory(asset_root):
    """Return the lazy original validator-backed local logical-path factory."""

    def create(logical_spec):
        apis = load_original_apis(asset_root)
        return apis.inference.make_backend(logical_spec)

    return create


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe-only", action="store_true", required=True)
    parser.parse_args()
    print(
        json.dumps(
            dict(
                scope_factory="E18ExecutionAdapter",
                single_task_entry="run_approved_task",
                approval_gate="validate_scope_approval",
                operations=sorted(OPERATIONS),
                native_gpu_runtime_status="NOT_RUNTIME_TESTED",
                root_scope_dispatcher_status="NOT_IMPLEMENTED",
                actual_model_calls=0,
                does_not_create_job_queues=True,
            )
        )
    )


if __name__ == "__main__":
    main()
