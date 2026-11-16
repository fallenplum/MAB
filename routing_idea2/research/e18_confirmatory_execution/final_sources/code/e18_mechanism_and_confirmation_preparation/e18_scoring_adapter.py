"""Approval-gated deterministic sparse scoring for the frozen E18 operator.

The adapter consumes selected aggregation execution receipts and a gated
single-reference provider. It never reads a dense reward matrix or initiates
model inference. Qualification uses synthetic references and FakeBackend only.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
import uuid

from sparse_policy_projection import FIELDS, GENERATION_SEEDS, OPERATOR_FP, SCORER_VERSION, TEAMS


SCORER_SOURCE_SHA256 = "6fc1d2f3cc81b3e954350143abd1071d6556e1cfe4f321d9333f2016baf2150d"
ORIGINAL_SCORER_SOURCE_SHA256 = "a9d3a7c7015ef542ff3240d5b638b1239ff5d7b1c4c1c19db81ffc34f0aeba1d"
TYPED_PROMPT = "anonymous_final_answer_typed_v3"
AGGREGATOR_MODEL_FP = "32aed6079298ebcf8eee99add17dd5c4a6828a504f13dd5d9236b234776861d3"
MEMBER_ALIASES = ("Qwen3-8B", "Qwen2.5-Coder-7B-Instruct", "Granite-3.3-8b-instruct",
    "DeepHermes-3-Llama-3-8B-Preview", "DeepSeek-R1-Distill-Qwen-7B", "Cogito-v1-preview-llama-8B")


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def load_frozen_scorer(source_root=None):
    """Import the byte-frozen scorer and its byte-frozen original dependency."""
    if source_root is None:
        source_root = Path(os.environ.get("MAB_ASSET_ROOT", "E:/语言模型组合路由")) / "src"
    source_root = Path(source_root).resolve()
    expected = ((source_root / "budget_router_v2/scoring.py", SCORER_SOURCE_SHA256),
                (source_root / "budget_router/scoring.py", ORIGINAL_SCORER_SOURCE_SHA256))
    for path, fingerprint in expected:
        if hashlib.sha256(path.read_bytes()).hexdigest() != fingerprint:
            raise ValueError("The frozen deterministic scorer source does not match")
    source_name = str(source_root)
    if source_name not in sys.path:
        sys.path.insert(0, source_name)
    scorer = importlib.import_module("budget_router_v2.scoring")
    if Path(scorer.__file__).resolve() != expected[0][0] or scorer.SCORER_VERSION != SCORER_VERSION:
        raise ValueError("Another scorer is already imported under the frozen package name")
    if Path(scorer.original.__file__).resolve() != expected[1][0]:
        raise ValueError("Another original scorer dependency is already imported")
    return scorer


def build_scoring_scope(execution_scope, reference_scope):
    """Bind scorer identity and the complete frozen chosen-action union."""
    if execution_scope.get("schema") != "e18_scope_contract_v2" or execution_scope.get("role") != "execution":
        raise ValueError("The frozen execution component scope is required")
    if reference_scope.get("schema") != "e18_scope_contract_v2" or reference_scope.get("role") != "reference":
        raise ValueError("The frozen reference component scope is required")
    if execution_scope.get("cohort") != "e18_confirm" or reference_scope.get("cohort") != "e18_confirm":
        raise ValueError("Only the explicit E18 confirmation cohort is permitted")
    if any(reference_scope.get("bindings", {}).get(key) != value
        for key, value in execution_scope.get("bindings", {}).items()):
        raise ValueError("Execution and reference have different frozen source bindings")
    return dict(schema="e18_scope_contract_v2", role="scoring", cohort="e18_confirm",
        bindings=dict(reference_scope["bindings"]), execution_bindings=dict(execution_scope["bindings"]),
        execution_scope_sha256=digest(execution_scope),
        reference_scope_sha256=digest(reference_scope),
        question_keys=json.loads(json.dumps(execution_scope["question_keys"])),
        required_aggregate_keys=json.loads(json.dumps(execution_scope["required_aggregate_keys"])),
        scorer_version=SCORER_VERSION, scorer_source_sha256=SCORER_SOURCE_SHA256,
        original_scorer_source_sha256=ORIGINAL_SCORER_SOURCE_SHA256,
        operator_fingerprint=OPERATOR_FP, prompt_version=TYPED_PROMPT, aggregation_seed=42,
        generation_seeds=list(GENERATION_SEEDS),
        authorization_state="PREPARATION_ONLY_UNTIL_APPROVAL")


def verified_aggregation_source(receipt):
    """Validate identity and measured usage from one successful native receipt."""
    if not isinstance(receipt, dict) or receipt.get("schema") != "e18_execution_task_receipt_v2":
        raise ValueError("A selected E18 execution task receipt is required")
    if receipt.get("status") != "SUCCESS" or receipt.get("stage") != "aggregation" or receipt.get("cohort") != "e18_confirm":
        raise ValueError("A successful E18 aggregation stage receipt is required")
    for name in ("scope_sha256", "approval_sha256", "native_record_sha256"):
        if not _sha(receipt.get(name)):
            raise ValueError("Execution receipt source hashes are required")
    if receipt.get("aggregator_fingerprint") != OPERATOR_FP or receipt.get("aggregation_seed") != 42:
        raise ValueError("Aggregation operator or seed differs from the frozen V2 protocol")
    if receipt.get("prompt_version") != TYPED_PROMPT:
        raise ValueError("Aggregation prompt identity differs")
    key = receipt.get("task_key")
    if not isinstance(key, dict):
        raise ValueError("A frozen sparse task key is required")
    question_id = key.get("question_id")
    canonical_id = key.get("canonical_id")
    generation_seed, action = key.get("generation_seed"), key.get("combination_index")
    if not isinstance(question_id, str) or not question_id or not _sha(canonical_id):
        raise ValueError("A question and canonical identity are required")
    if type(generation_seed) is not int or generation_seed not in GENERATION_SEEDS:
        raise ValueError("Member generation seed differs")
    if type(action) is not int or not 0 <= action < len(TEAMS) or key.get("team") != list(TEAMS[action]):
        raise ValueError("Selected combination index and semantic team differ")
    row = receipt.get("native_record")
    if not isinstance(row, dict) or row.get("status") != "success" or row.get("cost_status") != "recorded_actual_usage":
        raise ValueError("A successful settled native aggregation output is required")
    checksum = row.get("record_sha256")
    if checksum != receipt["native_record_sha256"] or digest({k: v for k, v in row.items() if k != "record_sha256"}) != checksum:
        raise ValueError("Native aggregation record hash differs")
    if row.get("job_id") != receipt.get("native_job_id") or row.get("attempt_id") != receipt.get("physical_attempt_id"):
        raise ValueError("Physical execution identity differs")
    if row.get("question_id") != question_id or row.get("seed") != 42 or row.get("prompt_version") != TYPED_PROMPT:
        raise ValueError("Native question, prompt or aggregation seed differs")
    if row.get("model_fingerprint") != AGGREGATOR_MODEL_FP or row.get("members") != sorted(MEMBER_ALIASES[index] for index in TEAMS[action]):
        raise ValueError("Native served-model profile or selected member identities differ")
    result = row.get("result")
    if not isinstance(result, dict) or result.get("model_identity_verified") is not True or result.get("seed") != 42:
        raise ValueError("Verified served-model output identity is required")
    if not isinstance(result.get("text"), str) or not result["text"].strip():
        raise ValueError("A nonempty final aggregate prediction is required")
    for name in ("prompt_tokens", "completion_tokens"):
        if type(result.get(name)) is not int or result[name] < 0:
            raise ValueError("Measured input/output tokens are required")
    if type(result.get("wall_time_s")) not in (int, float) or not math.isfinite(result["wall_time_s"]) or result["wall_time_s"] < 0:
        raise ValueError("Actual source execution wall time is required")
    return dict(question_id=question_id, canonical_id=canonical_id, generation_seed=generation_seed,
        combination_index=action, team=list(TEAMS[action]), prediction=result["text"],
        physical_attempt_id=receipt["physical_attempt_id"], aggregation_record_sha256=checksum,
        prompt_tokens=result["prompt_tokens"], completion_tokens=result["completion_tokens"],
        tokens=result["prompt_tokens"] + result["completion_tokens"], source_wall_time_s=result["wall_time_s"],
        source_execution_receipt_sha256=digest(receipt))


def resource_ledger_from_execution_receipts(receipts):
    """Deduplicate actual member/aggregation costs without interpreting answers.

    Unknown failed-call usage remains unknown in total-cost fields. Source
    execution receipts must already have passed their component scope checks.
    Backend failures before a native record exists remain the execution owner's
    unresolved reservation, outside the materialized receipt list.
    """
    unique = {}
    receipt_count = 0
    for receipt in receipts:
        receipt_count += 1
        if receipt.get("schema") != "e18_execution_task_receipt_v2" or receipt.get("stage") not in {"member_generation", "aggregation"}:
            raise ValueError("Native E18 execution receipts are required for resource accounting")
        native = receipt.get("native_record")
        if not isinstance(native, dict) or native.get("record_sha256") != receipt.get("native_record_sha256"):
            raise ValueError("A complete native physical record is required")
        if digest({key: value for key, value in native.items() if key != "record_sha256"}) != native["record_sha256"]:
            raise ValueError("Native resource record hash differs")
        physical_id = receipt.get("physical_attempt_id")
        if not isinstance(physical_id, str) or not physical_id or physical_id != native.get("attempt_id"):
            raise ValueError("Physical attempt identity is missing or differs")
        if native.get("job_id") != receipt.get("native_job_id"):
            raise ValueError("Resource receipt and native job differ")
        if native.get("status") not in {"success", "failure"}:
            raise ValueError("An actual success/failure native execution is required")
        usage = native.get("result") if native["status"] == "success" else native.get("actual_usage")
        known_usage = isinstance(usage, dict) and all(type(usage.get(key)) is int and usage[key] >= 0
            for key in ("prompt_tokens", "completion_tokens"))
        if known_usage and native.get("cost_status") != "recorded_actual_usage":
            raise ValueError("Measured usage and native cost settlement status disagree")
        if native["status"] == "success" and (native.get("cost_status") != "recorded_actual_usage" or not known_usage):
            raise ValueError("Successful execution lacks measured token usage")
        row = dict(physical_attempt_id=physical_id, stage=receipt["stage"], task_key=receipt["task_key"],
            native_record_sha256=native["record_sha256"], status=native["status"],
            usage_status="MEASURED" if known_usage else "UNKNOWN_REQUIRES_RECONCILIATION",
            prompt_tokens=usage["prompt_tokens"] if known_usage else None,
            completion_tokens=usage["completion_tokens"] if known_usage else None,
            actual_wall_s=usage.get("wall_time_s") if native["status"] == "success" else native.get("failure_wall_time_s"),
            software_test_only=receipt.get("software_test_only"), from_cache=bool(receipt.get("from_cache")))
        if physical_id in unique:
            old = unique[physical_id]
            if {key: value for key, value in old.items() if key != "from_cache"} != {key: value for key, value in row.items() if key != "from_cache"}:
                raise ValueError("One physical execution has conflicting resource or task identities")
            old["from_cache"] = old["from_cache"] and row["from_cache"]
        else:
            unique[physical_id] = row
    rows = list(unique.values())
    unknown = sum(row["usage_status"] != "MEASURED" for row in rows)
    known_input = sum(row["prompt_tokens"] for row in rows if row["prompt_tokens"] is not None)
    known_output = sum(row["completion_tokens"] for row in rows if row["completion_tokens"] is not None)
    return dict(schema="e18_physical_execution_cost_ledger_v2", receipt_rows=receipt_count,
        physical_execution_rows=rows, unique_physical_attempts=len(rows),
        member_attempts=sum(row["stage"] == "member_generation" for row in rows),
        aggregation_attempts=sum(row["stage"] == "aggregation" for row in rows),
        failed_attempts=sum(row["status"] == "failure" for row in rows), unknown_usage_attempts=unknown,
        known_actual_input_tokens=known_input, known_actual_output_tokens=known_output,
        total_actual_input_tokens=None if unknown else known_input,
        total_actual_output_tokens=None if unknown else known_output,
        total_actual_tokens=None if unknown else known_input + known_output,
        paid_resource_deduplication="physical_attempt_id; score/reference reuse does not create another model call",
        excludes_unmaterialized_reservations=True)


class ScoringJournal:
    """Persistent score outcomes, failed attempts and deduplicated resource refs."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY, logical_key TEXT NOT NULL, status TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scores (
                logical_key TEXT PRIMARY KEY, source_record_sha256 TEXT NOT NULL,
                sparse_json TEXT NOT NULL, receipt_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS execution_resources (
                physical_attempt_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS revealed_references (
                gold_key TEXT PRIMARY KEY, source_identity_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                setting_key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
        """)
        self.connection.commit()

    def close(self):
        self.connection.close()

    def bind(self, scope_sha, approval_sha, test_mode):
        values = {"scope_sha256": scope_sha, "approval_sha256": approval_sha,
                  "software_test_only": str(test_mode)}
        for key, value in values.items():
            old = self.connection.execute("SELECT value FROM settings WHERE setting_key=?", (key,)).fetchone()
            if old is not None and old[0] != value:
                raise ValueError("Existing scoring journal belongs to another approved scope")
            self.connection.execute("INSERT OR IGNORE INTO settings VALUES (?,?)", (key, value))
        self.connection.commit()

    def cached(self, key, source_sha):
        result = self.connection.execute("SELECT source_record_sha256,sparse_json,receipt_json FROM scores WHERE logical_key=?", (key,)).fetchone()
        if result is None:
            unresolved = self.connection.execute("SELECT COUNT(*) FROM attempts WHERE logical_key=? AND status!='SUCCESS'", (key,)).fetchone()[0]
            if unresolved:
                raise RuntimeError("Prior interrupted or failed scoring requires review before another attempt")
            return None
        if result[0] != source_sha:
            raise ValueError("The same selected scoring key now refers to another physical output")
        return dict(sparse_row=json.loads(result[1]), scoring_receipt=json.loads(result[2]), from_scoring_cache=True)

    def attempt(self, attempt_id, key, status, metadata):
        self.connection.execute("INSERT INTO attempts VALUES (?,?,?,?) ON CONFLICT(attempt_id) DO UPDATE SET status=excluded.status,metadata_json=excluded.metadata_json",
            (attempt_id, key, status, canonical_bytes(metadata).decode()))
        self.connection.commit()

    def execution_resource(self, source):
        selected = {key: source[key] for key in ("physical_attempt_id", "aggregation_record_sha256", "prompt_tokens", "completion_tokens", "tokens", "source_wall_time_s")}
        selected.update(resource_kind="aggregation_source_reference", charged_by_execution_adapter=True,
            incremental_member_calls_here=0, incremental_aggregation_calls_here=0)
        old = self.connection.execute("SELECT metadata_json FROM execution_resources WHERE physical_attempt_id=?", (source["physical_attempt_id"],)).fetchone()
        encoded = canonical_bytes(selected).decode()
        if old is not None and old[0] != encoded:
            raise ValueError("One physical execution has conflicting resource metadata")
        self.connection.execute("INSERT OR IGNORE INTO execution_resources VALUES (?,?)", (source["physical_attempt_id"], encoded))
        self.connection.commit()

    def reference(self, canonical_id, reference_version, source_identity_hash):
        key = digest([canonical_id, reference_version])
        old = self.connection.execute("SELECT source_identity_hash FROM revealed_references WHERE gold_key=?", (key,)).fetchone()
        if old is not None and old[0] != source_identity_hash:
            raise ValueError("One canonical/reference-version key changed its reference source")
        self.connection.execute("INSERT OR IGNORE INTO revealed_references VALUES (?,?)", (key, source_identity_hash))
        self.connection.commit()
        return old is None

    def succeed(self, key, source_sha, sparse, receipt):
        self.connection.execute("INSERT INTO scores VALUES (?,?,?,?)", (key, source_sha,
            canonical_bytes(sparse).decode(), canonical_bytes(receipt).decode()))
        self.connection.commit()

    def ledger(self):
        resources = [json.loads(row[0]) for row in self.connection.execute("SELECT metadata_json FROM execution_resources ORDER BY physical_attempt_id")]
        attempts = [dict(attempt_id=row[0], logical_key=row[1], status=row[2], **json.loads(row[3]))
            for row in self.connection.execute("SELECT attempt_id,logical_key,status,metadata_json FROM attempts ORDER BY rowid")]
        return dict(schema="e18_sparse_scoring_resource_ledger_v2", source_execution_resources=resources,
            physical_aggregation_sources=len(resources), actual_source_input_tokens=sum(row["prompt_tokens"] for row in resources),
            actual_source_output_tokens=sum(row["completion_tokens"] for row in resources),
            confirmed_reference_keys_in_scoring_journal=self.connection.execute("SELECT COUNT(*) FROM revealed_references").fetchone()[0],
            successful_sparse_keys=self.connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0],
            attempts=attempts, deterministic_scoring_model_calls=0, incremental_member_calls=0,
            incremental_aggregation_calls=0, execution_resources_already_charged_upstream=True)


class SparseScoringAdapter:
    """One frozen chosen-combination score, with shared approval enforcement."""

    def __init__(self, approval_ref, scope_contract, reference_provider, journal_path,
                 receipt_directory, *, software_test_only=False, scorer_source_root=None):
        if scope_contract.get("schema") != "e18_scope_contract_v2" or scope_contract.get("role") != "scoring" or scope_contract.get("cohort") != "e18_confirm":
            raise ValueError("The explicit E18 scoring component scope is required")
        if not (scope_contract.get("scorer_version") == SCORER_VERSION
            and scope_contract.get("scorer_source_sha256") == SCORER_SOURCE_SHA256
            and scope_contract.get("original_scorer_source_sha256") == ORIGINAL_SCORER_SOURCE_SHA256
            and scope_contract.get("operator_fingerprint") == OPERATOR_FP
            and scope_contract.get("prompt_version") == TYPED_PROMPT
            and scope_contract.get("aggregation_seed") == 42
            and scope_contract.get("generation_seeds") == list(GENERATION_SEEDS)):
            raise ValueError("Frozen scorer/operator source identities are required")
        self.approval_ref = approval_ref
        self.scope_contract = json.loads(json.dumps(scope_contract))
        self.reference_provider = reference_provider
        if digest(reference_provider.scope) != self.scope_contract.get("reference_scope_sha256"):
            raise ValueError("The supplied reference provider has another approved scope")
        self.software_test_only = software_test_only
        pairs = [(row["question_id"], row["canonical_question_sha256"]) for row in self.scope_contract["question_keys"]]
        if len({row[0] for row in pairs}) != len(pairs) or len({row[1] for row in pairs}) != len(pairs):
            raise ValueError("Scoring targets must have unique question/canonical identities")
        self.question_keys = dict(pairs)
        self.required_keys = set(map(tuple, self.scope_contract["required_aggregate_keys"]))
        if len(self.required_keys) != len(self.scope_contract["required_aggregate_keys"]):
            raise ValueError("Scoring union contains duplicate selected keys")
        if any(qid not in self.question_keys or type(seed) is not int or seed not in GENERATION_SEEDS
            or type(action) is not int or not 0 <= action < 20 for qid, seed, action in self.required_keys):
            raise ValueError("Scoring union contains malformed selected keys")
        self.scorer = load_frozen_scorer(scorer_source_root)
        self.journal = ScoringJournal(journal_path)
        self.receipt_directory = Path(receipt_directory)
        self.lock = threading.RLock()

    def close(self):
        self.journal.close()

    def _authorize(self):
        from e18_execution_adapter import validate_scope_approval
        return validate_scope_approval(self.approval_ref, self.scope_contract, "score_outputs",
            software_test_only=self.software_test_only)

    def _validate_upstream(self, execution_receipt):
        if execution_receipt.get("bindings") != self.scope_contract.get("execution_bindings"):
            raise ValueError("Scoring and execution have different frozen source bindings")
        if execution_receipt.get("software_test_only") is not self.software_test_only:
            raise ValueError("Synthetic and production execution modes cannot be mixed")
        expected_scope = self.scope_contract.get("execution_scope_sha256")
        if not _sha(expected_scope) or execution_receipt.get("scope_sha256") != expected_scope:
            raise ValueError("Scoring scope must bind the exact approved execution scope")

    def score_one(self, execution_receipt, reference_version):
        """Return a sparse row and an immutable scorer receipt, or fail closed.

        Approval is checked before any provider access, including cached scores.
        The execution adapter owns paid resource charges. This ledger retains
        actual upstream resources by physical ID and records scoring/gold work.
        """
        with self.lock:
            authorization = self._authorize()
            if authorization.get("scope_sha256_by_role", {}).get("execution") != self.scope_contract["execution_scope_sha256"]:
                raise PermissionError("The exact upstream execution scope was not approved")
            if authorization.get("scope_sha256_by_role", {}).get("reference") != self.scope_contract["reference_scope_sha256"]:
                raise PermissionError("The exact upstream reference scope was not approved")
            if execution_receipt.get("approval_sha256") != self.approval_ref["sha256"]:
                raise PermissionError("Execution and scoring refer to different approval receipts")
            self._validate_upstream(execution_receipt)
            source = verified_aggregation_source(execution_receipt)
            if self.question_keys.get(source["question_id"]) != source["canonical_id"] or (
                source["question_id"], source["generation_seed"], source["combination_index"]) not in self.required_keys:
                raise PermissionError("Scored execution lies outside the frozen chosen-combination union")
            if not isinstance(reference_version, str) or not reference_version:
                raise ValueError("A pinned reference version is required")
            self.journal.bind(digest(self.scope_contract), self.approval_ref["sha256"], self.software_test_only)
            key = digest([source["canonical_id"], source["generation_seed"], source["combination_index"],
                reference_version, OPERATOR_FP, SCORER_VERSION])
            cached = self.journal.cached(key, source["aggregation_record_sha256"])
            if cached is not None:
                receipt_path = self.receipt_directory / (cached["sparse_row"]["scoring_receipt_sha256"] + ".json")
                if receipt_path.read_bytes() != canonical_bytes(cached["scoring_receipt"]):
                    raise ValueError("Cached scoring receipt bytes differ or are missing")
                return cached
            attempt_id = uuid.uuid4().hex
            started_wall, started_cpu = time.perf_counter(), time.process_time()
            metadata = dict(reference_version=reference_version,
                source_execution_receipt_sha256=source["source_execution_receipt_sha256"],
                source_aggregation_record_sha256=source["aggregation_record_sha256"],
                physical_attempt_id=source["physical_attempt_id"], software_test_only=self.software_test_only,
                reference_provider_called=False, gold_access_may_have_occurred=False,
                reference_provider_stats_before=self.reference_provider.stats())
            self.journal.attempt(attempt_id, key, "STARTED", metadata)
            try:
                self.journal.execution_resource(source)
                metadata["reference_provider_called"] = True
                metadata["gold_access_may_have_occurred"] = True
                self.journal.attempt(attempt_id, key, "REFERENCE_ACCESS_STARTED", metadata)
                reference = self.reference_provider.get_reference(source["question_id"], source["canonical_id"], reference_version)
                metadata["reference_provider_stats_after"] = self.reference_provider.stats()
                if not isinstance(reference, dict) or not (reference.get("question_id") == source["question_id"]
                    and reference.get("canonical_question_sha256") == source["canonical_id"]
                    and reference.get("reference_version") == reference_version and _sha(reference.get("source_identity_hash"))):
                    raise ValueError("Projected reference identity differs from the selected execution")
                gold_new = self.journal.reference(source["canonical_id"], reference_version, reference["source_identity_hash"])
                metadata.update(gold_new_in_scoring_journal=gold_new, reference_source_identity_hash=reference["source_identity_hash"],
                    reference_provider_cache_hit=bool(reference.get("cache_hit", False)),
                    reference_provider_revealed_new=bool(reference.get("reference_revealed_new", False)))
                metadata["deterministic_scoring_ops"] = 1
                scorer_cpu, scorer_wall = time.process_time(), time.perf_counter()
                result = self.scorer.score_answer(reference["task"], source["prediction"], reference["reference"],
                    subtask=reference.get("subtask"), choices=reference.get("choices"))
                scorer_cpu_s, scorer_wall_s = time.process_time() - scorer_cpu, time.perf_counter() - scorer_wall
                if result.scorer_version != SCORER_VERSION or result.score not in (0, 1):
                    raise ValueError("Frozen deterministic scoring result differs")
                receipt = dict(schema="e18_sparse_scoring_receipt_v2", scoring_attempt_id=attempt_id,
                    scope_sha256=digest(self.scope_contract), approval_sha256=self.approval_ref["sha256"],
                    bindings=self.scope_contract["bindings"], software_test_only=self.software_test_only,
                    question_id=source["question_id"], canonical_id=source["canonical_id"],
                    generation_seed=source["generation_seed"], aggregation_seed=42,
                    combination_index=source["combination_index"], team=source["team"],
                    reference_version=reference_version, reference_source_identity_hash=reference["source_identity_hash"],
                    gold_new_in_scoring_journal=gold_new, scorer_version=SCORER_VERSION,
                    reference_provider_cache_hit=bool(reference.get("cache_hit", False)),
                    reference_provider_revealed_new=bool(reference.get("reference_revealed_new", False)),
                    scorer_source_sha256=SCORER_SOURCE_SHA256, aggregator_fingerprint=OPERATOR_FP,
                    source_execution_receipt_sha256=source["source_execution_receipt_sha256"],
                    aggregation_record_sha256=source["aggregation_record_sha256"], physical_attempt_id=source["physical_attempt_id"],
                    actual_source_input_tokens=source["prompt_tokens"], actual_source_output_tokens=source["completion_tokens"],
                    execution_resources_already_charged_upstream=True, incremental_member_calls=0,
                    incremental_aggregation_calls=0, scoring_model_calls=0,
                    score=float(result.score), parsed=bool(result.parsed), reason=result.reason,
                    scorer_cpu_s=scorer_cpu_s, scorer_wall_s=scorer_wall_s,
                    scoring_cpu_s=time.process_time() - started_cpu, scoring_wall_s=time.perf_counter() - started_wall)
                receipt_bytes = canonical_bytes(receipt)
                receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
                self.receipt_directory.mkdir(parents=True, exist_ok=True)
                receipt_path = self.receipt_directory / (receipt_sha + ".json")
                with receipt_path.open("xb") as stream:
                    stream.write(receipt_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                sparse = {name: receipt[name] for name in FIELDS if name != "scoring_receipt_sha256"}
                sparse["scoring_receipt_sha256"] = receipt_sha
                self.journal.succeed(key, source["aggregation_record_sha256"], sparse, receipt)
                metadata.update(scoring_receipt_sha256=receipt_sha, scoring_receipt_path=str(receipt_path),
                    scoring_cpu_s=receipt["scoring_cpu_s"], scoring_wall_s=receipt["scoring_wall_s"],
                    deterministic_scoring_ops=1)
                self.journal.attempt(attempt_id, key, "SUCCESS", metadata)
                return dict(sparse_row=sparse, scoring_receipt=receipt, from_scoring_cache=False)
            except Exception as error:
                metadata["reference_provider_stats_after"] = self.reference_provider.stats()
                metadata.update(error_type=type(error).__name__, scoring_cpu_s=time.process_time() - started_cpu,
                    scoring_wall_s=time.perf_counter() - started_wall)
                metadata.setdefault("deterministic_scoring_ops", 0)
                self.journal.attempt(attempt_id, key, "FAILED_REQUIRES_REVIEW", metadata)
                raise

    def ledger(self):
        result = self.journal.ledger()
        result["reference_provider_stats"] = self.reference_provider.stats()
        result["reference_cost_owner"] = "reference provider canonical-question/reference-version ledger, including failures after value access"
        return result
