"""Metadata-only, globally deduplicated E18 aggregation-ready planner.

SQLite reserves the immutable frozen keys before queue publication. Every
artifact is recoverable from the committed batch bytes. Native answers, Gold,
scores and model backends are never opened or constructed by this module.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time


TEAMS = tuple(combinations(range(6), 3))
SEEDS = (42, 137, 2024)
QUEUES = ("gpu5_slot0", "gpu5_slot1", "gpu6_slot0")
AGG_WORKER_SHA = "de5c3cc8a509be10e66b3290899862bab73a2784a62aa743e7c48d41a7842495"
MAPPING_SHA = "dfae3e84c1988616775d7ccb0385189ebd0112db6e152dc52c2400366be15217"
DEPENDENCIES_SHA = "d67d177688c45c0b5d933fd32f918e6efb2d6b4d719e1dd8c38df0b05f840556"
STARTUP_PLAN_SHA = "84ba35cd7c98aefdcee077281b547280c49e473c543767e017d0f1eaa50c1985"
SOURCE_TYPES = {
    "e18_gpu_dispatch_plan_v1": (
        "e18_gpu_worker_manifest_v1",
        "member_generation",
        "e18_gpu_worker.py",
    ),
    "e18_member4_dispatch_plan_v3": (
        "e18_member4_worker_manifest_v3",
        "member_generation",
        "e18_member4_worker_v3.py",
    ),
    "e18_aggregation_dispatch_plan_v3": (
        "e18_aggregation_worker_manifest_v3",
        "aggregation",
        "e18_aggregation_worker_v3.py",
    ),
}


def sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            checksum.update(block)
    return checksum.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def checked_ref(reference):
    if (
        not isinstance(reference, dict)
        or set(reference) != {"path", "sha256"}
        or not valid_sha(reference["sha256"])
    ):
        raise ValueError("An exact path/SHA artifact reference is required")
    path = Path(reference["path"]).resolve(strict=True)
    if not path.is_file() or sha(path) != reference["sha256"]:
        raise ValueError("Referenced metadata bytes differ")
    return path


def publish_exact(path, data):
    """Publish complete bytes without replacing a prior artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("An immutable queue artifact differs " + str(path))
        return
    temporary = path.with_name(
        "." + path.name + ".tmp-" + str(os.getpid()) + "-" + str(time.time_ns())
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("Another publisher committed different queue bytes")
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class IncrementalPlanner:
    def __init__(self, config_ref, *, qualification_only=False):
        self.config_path = checked_ref(config_ref)
        self.config_sha = config_ref["sha256"]
        self.config = read_json(self.config_path)
        if self.config.get("schema") != "e18_incremental_aggregation_planner_config_v1":
            raise ValueError("Unknown incremental-planner configuration schema")
        self.server = Path(self.config["server_root"]).resolve()
        self.template = read_json(checked_ref(self.config["template_manifest_ref"]))
        self.scope = read_json(checked_ref(self.config["scope_ref"]))
        if (
            self.scope.get("schema") != "e18_scope_contract_v2"
            or self.scope.get("role") != "execution"
            or self.scope.get("cohort") != "e18_confirm"
        ):
            raise ValueError("The formal E18 execution scope is required")
        self.scope_sha = digest(self.scope)
        self.output = Path(self.template["owned_output_root"]).resolve()
        if (
            self.output != self.server / "owned"
            or self.template.get("scope_ref") != self.config["scope_ref"]
        ):
            raise ValueError("Formal scope or owned output root differs")
        if (
            self.template.get("schema") != "e18_aggregation_worker_manifest_v3"
            or self.template.get("stage") != "aggregation"
            or self.template.get("software_test_only") is not False
        ):
            raise ValueError("A production aggregation-v3 profile template is required")
        if (
            self.template.get("member_dependencies_ref")
            != self.config["historical_dependencies_ref"]
        ):
            raise ValueError("The historical dependency index differs from the template")
        self.approval_sha = self.template["approval_ref"]["sha256"]
        self.worker_sha = sha(
            Path(self.template["adapter_ref"]["path"]).parent / "e18_aggregation_worker_v3.py"
        )
        self.history = read_json(checked_ref(self.config["historical_dependencies_ref"]))
        self.startup = read_json(checked_ref(self.config["startup_keys_ref"]))
        self.startup_plan_path = checked_ref(self.config["startup_plan_ref"])
        self.startup_plan = read_json(self.startup_plan_path)
        self.questions = {
            row["question_id"]: index for index, row in enumerate(self.scope["question_keys"])
        }
        self.aggregate_keys = {tuple(key) for key in self.scope["required_aggregate_keys"]}
        self.member_keys = {tuple(key) for key in self.scope["required_member_keys"]}
        self.startup_keys = {tuple(key) for key in self.startup}
        if (
            len(self.startup_keys) != len(self.startup)
            or not self.startup_keys <= self.aggregate_keys
        ):
            raise ValueError("Startup reservation contains duplicate or foreign aggregation keys")
        if set(map(tuple, self.scope["aggregate_execution_keys"])) != self.aggregate_keys:
            raise ValueError("The full frozen aggregation target cannot be reduced")
        self._validate_scope_keys()
        if not qualification_only:
            if (
                len(self.aggregate_keys) != 30921
                or len(self.history) != 4717
                or len(self.startup_keys) != 16
            ):
                raise ValueError("Frozen E18 target/cache/startup cardinalities differ")
            if (
                self.worker_sha != AGG_WORKER_SHA
                or self.template["backend"]["mapping_ref"]["sha256"] != MAPPING_SHA
            ):
                raise ValueError("Frozen aggregation worker or loading map identity differs")
            if (
                self.config["historical_dependencies_ref"]["sha256"] != DEPENDENCIES_SHA
                or self.config["startup_plan_ref"]["sha256"] != STARTUP_PLAN_SHA
            ):
                raise ValueError("Frozen historical-index or startup-plan identity differs")
            if (
                self.config["scope_ref"]["sha256"]
                != "cae646111612ceb26196d73ed4469165e77b163a332f4db81deeb6f66551f73e"
            ):
                raise ValueError("The original formal3000 execution-scope bytes differ")
        self.batch_max = self.config.get("batch_max", 512)
        if type(self.batch_max) is not int or not 1 <= self.batch_max <= 512:
            raise ValueError("Aggregation batches must contain at most512 complete keys")
        queue_rows = self.config["queues"]
        if [row["name"] for row in queue_rows] != list(QUEUES):
            raise ValueError("Exactly the three root-admitted aggregation queues are required")
        self.queues = {row["name"]: row for row in queue_rows}
        for name, row in self.queues.items():
            if (
                Path(row["queue_root"]).resolve() != self.server / "aggregation_queues" / name
                or Path(row["gpu_lock_root"]).resolve()
                != self.server / "gpu-locks/aggregation" / name
            ):
                raise ValueError("Queue or independent GPU-owner lock path differs")
        self.database = Path(self.config["database_path"]).resolve()
        if not self.database.is_relative_to(self.server):
            raise ValueError("Planner ledger escaped the approved E18 root")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.stop_path = self.database.parent / "STOP_REQUIRES_RECONCILIATION.json"
        self.connection = sqlite3.connect(self.database, timeout=60, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._initialize()

    def _validate_scope_keys(self):
        if len(self.questions) != len(self.scope["question_keys"]):
            raise ValueError("Duplicate frozen question identities")
        for qid, generation, action in self.aggregate_keys:
            if (
                qid not in self.questions
                or type(generation) is not int
                or generation not in SEEDS
                or type(action) is not int
                or not 0 <= action < 20
            ):
                raise ValueError("Invalid frozen aggregation key")
            if any((qid, generation, member) not in self.member_keys for member in TEAMS[action]):
                raise ValueError("Frozen aggregation lacks its three-member dependency union")

    def _initialize(self):
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS members(qid TEXT,generation INTEGER,member INTEGER,proof_sha TEXT,attempt TEXT,source TEXT,
          PRIMARY KEY(qid,generation,member));
        CREATE TABLE IF NOT EXISTS aggregates(qid TEXT,generation INTEGER,action INTEGER,
          question_index INTEGER,status TEXT,batch_id INTEGER,
          proof_sha TEXT,attempt TEXT,PRIMARY KEY(qid,generation,action));
        CREATE TABLE IF NOT EXISTS batches(id INTEGER PRIMARY KEY AUTOINCREMENT,queue TEXT,status TEXT,key_rows TEXT,files TEXT,
          ready_path TEXT,manifest_sha TEXT,manifest_path TEXT);
        CREATE TABLE IF NOT EXISTS journals(path TEXT PRIMARY KEY,device INTEGER,inode INTEGER,offset INTEGER,prefix_sha TEXT);
        """)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            previous = self.connection.execute(
                "SELECT value FROM metadata WHERE key='config_sha'"
            ).fetchone()
            if previous is not None:
                if previous[0] != self.config_sha:
                    raise ValueError("Planner configuration changed after ledger initialization")
            else:
                if any(
                    list(Path(row["queue_root"]).glob("*.ready.json"))
                    for row in self.queues.values()
                ):
                    raise RuntimeError(
                        "Existing queue entries cannot be adopted by a recreated empty ledger"
                    )
                self.connection.execute(
                    "INSERT INTO metadata VALUES('config_sha',?)", (self.config_sha,)
                )
                self.connection.executemany(
                    "INSERT INTO aggregates VALUES(?,?,?,?,?,NULL,NULL,NULL)",
                    [
                        (
                            qid,
                            generation,
                            action,
                            self.questions[qid],
                            "EXTERNAL_RESERVED"
                            if (qid, generation, action) in self.startup_keys
                            else "WAITING",
                        )
                        for qid, generation, action in sorted(
                            self.aggregate_keys, key=self.key_order
                        )
                    ],
                )
                seen = set()
                for row in self.history:
                    if set(row) != {"task_key", "record_ref"}:
                        raise ValueError("Unknown historical dependency-index schema")
                    key = tuple(row["task_key"])
                    reference = row["record_ref"]
                    if (
                        key not in self.member_keys
                        or key in seen
                        or not isinstance(reference, dict)
                        or set(reference) != {"path", "sha256"}
                        or not valid_sha(reference["sha256"])
                    ):
                        raise ValueError(
                            "Foreign, duplicate or unbound historical member dependency"
                        )
                    seen.add(key)
                    # This pinned index has already been independently bridged.
                    # Native member payloads are deliberately not opened here.
                    self.connection.execute(
                        "INSERT INTO members VALUES(?,?,?,?,NULL,'HISTORICAL_INDEX')",
                        (*key, reference["sha256"]),
                    )
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def key_order(self, key):
        return self.questions[key[0]], key[1], key[2]

    def _validate_manifest(self, reference, manifest, expected_schema=None):
        schema = manifest.get("schema")
        if schema not in {item[0] for item in SOURCE_TYPES.values()} or (
            expected_schema is not None and schema != expected_schema
        ):
            raise ValueError("Unknown or mismatched execution manifest schema")
        for key in (
            "scope_ref",
            "approval_ref",
            "question_records_ref",
            "adapter_ref",
            "asset_root",
            "owned_output_root",
            "software_test_only",
        ):
            if manifest.get(key) != self.template.get(key):
                raise ValueError("Journal manifest has another frozen scope/input/approval/owner")
        stage = manifest.get("stage")
        if stage not in {"member_generation", "aggregation"}:
            raise ValueError("Unknown execution stage")
        rows = read_json(checked_ref(manifest["task_keys_ref"]))
        keys = {tuple(key) for key in rows}
        permitted = self.member_keys if stage == "member_generation" else self.aggregate_keys
        if len(keys) != len(rows) or not keys <= permitted:
            raise ValueError("Journal manifest contains duplicate or out-of-scope task keys")
        if stage == "member_generation":
            if manifest.get("member_index") not in range(6) or any(
                key[2] != manifest["member_index"] for key in keys
            ):
                raise ValueError("Member profile differs from its manifest task keys")
        elif manifest.get("backend") != self.template.get("backend") or manifest.get(
            "member_dependencies_ref"
        ) != self.template.get("member_dependencies_ref"):
            raise ValueError("Aggregation manifest changes the frozen loading/dependency profile")
        return dict(reference=reference, manifest=manifest, stage=stage, keys=keys)

    def registry(self):
        result = {}
        references = self.config["source_plan_refs"] + [self.config["startup_plan_ref"]]
        for reference in references:
            plan = read_json(checked_ref(reference))
            if plan.get("schema") not in SOURCE_TYPES:
                raise ValueError("Unknown source-plan schema")
            if not plan.get("manifest_refs"):
                raise ValueError("A source plan has no manifest references")
            manifest_schema, stage, worker_name = SOURCE_TYPES[plan["schema"]]
            worker_path = Path(self.template["adapter_ref"]["path"]).parent / worker_name
            if plan.get("worker_source_sha256") != sha(worker_path):
                raise ValueError("A source plan changes its frozen worker source")
            for manifest_ref in plan.get("manifest_refs", []):
                manifest = read_json(checked_ref(manifest_ref))
                item = self._validate_manifest(manifest_ref, manifest, manifest_schema)
                if item["stage"] != stage:
                    raise ValueError("A source-plan stage differs")
                existing = result.get(manifest_ref["sha256"])
                if existing is not None and existing["reference"] != manifest_ref:
                    raise ValueError("One manifest SHA has ambiguous source paths")
                result[manifest_ref["sha256"]] = item
        for row in self.connection.execute("SELECT manifest_path,manifest_sha FROM batches"):
            reference = dict(path=row["manifest_path"], sha256=row["manifest_sha"])
            if Path(reference["path"]).exists():
                result[reference["sha256"]] = self._validate_manifest(
                    reference, read_json(checked_ref(reference))
                )
        return result

    def _native_commit_proof(self, event, key, stage):
        proof = event.get("native_record_sha256")
        attempt = event.get("physical_attempt_id")
        usage = event.get("actual_usage")
        if (
            not valid_sha(proof)
            or not isinstance(attempt, str)
            or not attempt
            or not isinstance(usage, dict)
            or any(
                type(usage.get(name)) is not int or usage[name] < 0
                for name in ("prompt_tokens", "completion_tokens")
            )
        ):
            raise RuntimeError("A SUCCESS event has unsettled native identity or usage")
        owner = self.output / self.scope_sha / stage / digest(list(key))
        fence = read_json(owner / "ATTEMPT_STATE.json")
        expected = dict(
            status="SUCCESS_COMMITTED",
            scope_sha256=self.scope_sha,
            approval_sha256=self.approval_sha,
            task_key=list(key),
            stage=stage,
            native_record_sha256=proof,
            software_test_only=False,
        )
        if any(fence.get(name) != value for name, value in expected.items()):
            raise RuntimeError("A native execution fence differs or remains unsettled")
        if (
            not (owner / "LAST_TASK_RECEIPT.json").is_file()
            or not (owner / "records.jsonl").is_file()
        ):
            raise RuntimeError("A SUCCESS event lacks its committed native files")
        return proof, attempt

    def _event(self, event, item):
        label = event.get("event")
        if label not in {
            "TASK_ENTER",
            "TASK_RESULT",
            "DEPENDENCY_NOT_READY",
            "STOP_REQUIRES_RECONCILIATION",
        }:
            raise ValueError("Unknown worker journal event schema")
        task = event.get("task")
        if not isinstance(task, dict):
            raise ValueError("Worker journal lacks task metadata")
        stage = task.get("stage")
        index = "member_index" if stage == "member_generation" else "combination_index"
        key = (task.get("question_id"), task.get("generation_seed"), task.get(index))
        if stage != item["stage"] or key not in item["keys"]:
            raise ValueError("Journal task exceeds its bound manifest or scope")
        if event.get("scope_sha256", self.scope_sha) != self.scope_sha:
            raise ValueError("Explicit journal scope differs")
        if label == "STOP_REQUIRES_RECONCILIATION":
            raise RuntimeError("An execution journal requires failure/usage reconciliation")
        if label != "TASK_RESULT":
            return
        if event.get("status") != "SUCCESS":
            raise RuntimeError("A failed or unknown native execution cannot be skipped")
        proof, attempt = self._native_commit_proof(event, key, stage)
        table = "members" if stage == "member_generation" else "aggregates"
        field = "member" if stage == "member_generation" else "action"
        existing = self.connection.execute(
            f"SELECT proof_sha,attempt FROM {table} WHERE qid=? AND generation=? AND {field}=?", key
        ).fetchone()
        if (
            existing is not None
            and existing["proof_sha"] is not None
            and (existing["proof_sha"], existing["attempt"]) != (proof, attempt)
        ):
            raise RuntimeError("Conflicting native successes require explicit reconciliation")
        if stage == "member_generation":
            self.connection.execute(
                "INSERT INTO members VALUES(?,?,?,?,?,'NATIVE_EVENT') ON CONFLICT(qid,generation,member) DO NOTHING",
                (*key, proof, attempt),
            )
        else:
            row = self.connection.execute(
                "SELECT status FROM aggregates WHERE qid=? AND generation=? AND action=?", key
            ).fetchone()
            if row["status"] == "WAITING":
                raise RuntimeError(
                    "Aggregation success has no startup reservation or planner batch"
                )
            self.connection.execute(
                "UPDATE aggregates SET status='SUCCESS',proof_sha=?,attempt=? WHERE qid=? AND generation=? AND action=?",
                (proof, attempt, *key),
            )

    def _scan_file(self, path, item):
        stat = path.stat()
        previous = self.connection.execute(
            "SELECT * FROM journals WHERE path=?", (str(path),)
        ).fetchone()
        offset = previous["offset"] if previous else 0
        if previous and (
            (stat.st_dev, stat.st_ino) != (previous["device"], previous["inode"])
            or stat.st_size < offset
        ):
            raise ValueError("A previously ingested worker journal was replaced or truncated")
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            remaining = offset
            while remaining:
                block = stream.read(min(1048576, remaining))
                if not block:
                    raise ValueError("Journal truncated during prefix verification")
                checksum.update(block)
                remaining -= len(block)
            if previous and checksum.hexdigest() != previous["prefix_sha"]:
                raise ValueError("Previously ingested journal bytes changed")
            while True:
                start = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    # An active append's unfinished tail stays unread/unbought.
                    stream.seek(start)
                    break
                event = json.loads(raw)
                if item is None:
                    if event.get("event") == "TASK_RESULT" or "task" in event:
                        raise ValueError("Unregistered journal contains worker execution events")
                else:
                    self._event(event, item)
                checksum.update(raw)
                offset = stream.tell()
        self.connection.execute(
            "INSERT INTO journals VALUES(?,?,?,?,?) ON CONFLICT(path) DO UPDATE "
            "SET device=excluded.device,inode=excluded.inode,offset=excluded.offset,"
            "prefix_sha=excluded.prefix_sha",
            (str(path), stat.st_dev, stat.st_ino, offset, checksum.hexdigest()),
        )

    def refresh(self):
        registry = self.registry()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            for path in sorted((self.output / "_dispatcher").glob("*/EVENTS.jsonl")):
                item = registry.get(path.parent.name)
                self._scan_file(path, item)
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def _batch_files(self, number, queue, keys):
        base = Path(self.queues[queue]["queue_root"]) / "batches" / f"{number:08d}"
        payloads = []

        def artifact(path, value):
            data = encoded(value)
            reference = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())
            payloads.append(
                dict(
                    path=str(path), sha256=reference["sha256"], data=base64.b64encode(data).decode()
                )
            )
            return reference

        keys_ref = artifact(base / "keys.json", keys)
        manifest = dict(
            self.template, task_keys_ref=keys_ref, gpu_lock_root=self.queues[queue]["gpu_lock_root"]
        )
        manifest_ref = artifact(base / "manifest.json", manifest)
        plan_ref = artifact(
            base / "plan.json",
            dict(
                schema="e18_aggregation_dispatch_plan_v3",
                worker_source_sha256=self.worker_sha,
                manifest_refs=[manifest_ref],
            ),
        )
        ready_path = Path(self.queues[queue]["queue_root"]) / f"{number:08d}.ready.json"
        artifact(ready_path, dict(schema="e18_aggregation_queue_entry_v1", plan_ref=plan_ref))
        return payloads, ready_path, manifest_ref

    def reserve(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            members = {
                tuple(row)
                for row in self.connection.execute("SELECT qid,generation,member FROM members")
            }
            waiting = self.connection.execute(
                "SELECT qid,generation,action FROM aggregates WHERE status='WAITING' ORDER BY question_index,generation,action"
            ).fetchall()
            ready = [
                list(row)
                for row in waiting
                if all((row[0], row[1], member) in members for member in TEAMS[row[2]])
            ]
            reserved = 0
            for offset in range(0, len(ready), self.batch_max):
                keys = ready[offset : offset + self.batch_max]
                pending = {name: 0 for name in QUEUES}
                for row in self.connection.execute(
                    "SELECT b.queue,COUNT(*) FROM aggregates a JOIN batches b ON a.batch_id=b.id "
                    "WHERE a.status!='SUCCESS' GROUP BY b.queue"
                ):
                    pending[row[0]] = row[1]
                queue = min(QUEUES, key=lambda name: (pending[name], QUEUES.index(name)))
                cursor = self.connection.execute(
                    "INSERT INTO batches(queue,status,key_rows,files,ready_path,manifest_sha,"
                    "manifest_path) VALUES(?,'RESERVED',?,'[]','','','')",
                    (queue, json.dumps(keys)),
                )
                number = cursor.lastrowid
                files, ready_path, manifest_ref = self._batch_files(number, queue, keys)
                self.connection.execute(
                    "UPDATE batches SET files=?,ready_path=?,manifest_sha=?,manifest_path=? WHERE id=?",
                    (
                        json.dumps(files),
                        str(ready_path),
                        manifest_ref["sha256"],
                        manifest_ref["path"],
                        number,
                    ),
                )
                for key in keys:
                    changed = self.connection.execute(
                        "UPDATE aggregates SET status='RESERVED',batch_id=? "
                        "WHERE qid=? AND generation=? AND action=? AND status='WAITING'",
                        (number, *key),
                    ).rowcount
                    if changed != 1:
                        raise RuntimeError("An aggregation key was reserved concurrently twice")
                reserved += len(keys)
            self.connection.execute("COMMIT")
            return reserved
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def publish_reserved(self):
        for queue in self.queues.values():
            for path in Path(queue["queue_root"]).glob("*.ready.json"):
                row = self.connection.execute(
                    "SELECT files FROM batches WHERE ready_path=?", (str(path),)
                ).fetchone()
                if row is None:
                    raise RuntimeError(
                        "An aggregation ready entry has no central-ledger reservation"
                    )
                reference = next(item for item in json.loads(row[0]) if item["path"] == str(path))
                if sha(path) != reference["sha256"]:
                    raise ValueError("A published ready entry changed")
        published = 0
        for row in self.connection.execute("SELECT * FROM batches ORDER BY id").fetchall():
            for artifact in json.loads(row["files"]):
                data = base64.b64decode(artifact["data"], validate=True)
                if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
                    raise ValueError("Committed batch bytes were changed in the ledger")
                publish_exact(artifact["path"], data)
            if row["status"] == "RESERVED":
                self.connection.execute(
                    "UPDATE batches SET status='PUBLISHED' WHERE id=? AND status='RESERVED'",
                    (row["id"],),
                )
                published += 1
        return published

    def mark_all_done(self):
        completed = self.connection.execute(
            "SELECT COUNT(*) FROM aggregates WHERE status='SUCCESS'"
        ).fetchone()[0]
        if completed != len(self.aggregate_keys):
            return False
        startup_state = (
            self.output / "_dispatcher/plans" / self.config["startup_plan_ref"]["sha256"]
        )
        for reference in self.startup_plan["manifest_refs"]:
            path = startup_state / (reference["sha256"] + ".status.json")
            if not path.is_file():
                return False
            status = read_json(path)
            if (
                status.get("status") != "DONE"
                or status.get("manifest_sha256") != reference["sha256"]
            ):
                return False
        proofs = [
            list(row)
            for row in self.connection.execute(
                "SELECT qid,generation,action,proof_sha FROM aggregates ORDER BY question_index,generation,action"
            )
        ]
        payload = encoded(
            dict(
                schema="e18_incremental_aggregation_all_done_v1",
                status="ALL_FROZEN_NATIVE_COMMITS_SUCCESS",
                config_sha256=self.config_sha,
                execution_scope_sha256=self.scope_sha,
                total_frozen_aggregate_keys=len(self.aggregate_keys),
                startup_external_keys=len(self.startup_keys),
                startup_plan_sha256=self.config["startup_plan_ref"]["sha256"],
                native_success_metadata_digest=digest(proofs),
                gold_or_native_answer_payloads_read=0,
            )
        )
        for queue in self.queues.values():
            publish_exact(Path(queue["queue_root"]) / "ALL_DONE", payload)
        return True

    def record_stop(self, error):
        """Retain the first failure without replacing another planner's proof."""
        if self.stop_path.exists():
            return
        payload = encoded(
            dict(
                status="STOP_REQUIRES_RECONCILIATION",
                error_type=type(error).__name__,
                error=str(error),
                config_sha256=self.config_sha,
                published_queue_batches_retained=True,
                at_unix=time.time(),
            )
        )
        try:
            publish_exact(self.stop_path, payload)
        except ValueError:
            if not self.stop_path.exists():
                raise

    def once(self, *, after_reserve_hook=None):
        if self.stop_path.exists():
            raise RuntimeError("Planner STOP evidence requires root reconciliation")
        try:
            recovered = self.publish_reserved()
            self.refresh()
            newly_reserved = self.reserve()
            if after_reserve_hook is not None:
                after_reserve_hook()
            published = self.publish_reserved()
            done = self.mark_all_done()
            counts = dict(
                self.connection.execute(
                    "SELECT status,COUNT(*) FROM aggregates GROUP BY status"
                ).fetchall()
            )
            result = dict(
                schema="e18_incremental_aggregation_planner_once_v1",
                status="ALL_DONE" if done else "RUNNING",
                frozen_aggregate_target=len(self.aggregate_keys),
                startup_external_reserved=len(self.startup_keys),
                aggregate_status_counts=counts,
                historical_and_native_ready_members=self.connection.execute(
                    "SELECT COUNT(*) FROM members"
                ).fetchone()[0],
                newly_reserved_keys=newly_reserved,
                recovered_reserved_batches=recovered,
                newly_published_batches=published,
                queues_pending_keys={
                    name: self.connection.execute(
                        "SELECT COUNT(*) FROM aggregates a JOIN batches b ON a.batch_id=b.id "
                        "WHERE b.queue=? AND a.status!='SUCCESS'",
                        (name,),
                    ).fetchone()[0]
                    for name in QUEUES
                },
                physical_model_calls=0,
                gold_values_read=0,
                native_answer_payloads_read=0,
            )
            publish_exact(
                self.database.parent / ("ONCE_" + str(time.time_ns()) + ".json"), encoded(result)
            )
            return result
        except BaseException as error:
            self.record_stop(error)
            raise

    def follow(self, interval_seconds=60, *, sleeper=time.sleep):
        """Keep supplying metadata batches, stopping on completion or any error."""
        try:
            if not math.isfinite(interval_seconds) or interval_seconds <= 0:
                raise ValueError("A finite positive planner interval is required")
            while True:
                result = self.once()
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if result["status"] == "ALL_DONE":
                    return result
                sleeper(interval_seconds)
        except BaseException as error:
            self.record_stop(error)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--follow", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=60)
    args = parser.parse_args()
    planner = IncrementalPlanner(dict(path=str(args.config), sha256=args.config_sha256))
    try:
        if args.once:
            print(json.dumps(planner.once(), ensure_ascii=False, indent=2))
        else:
            planner.follow(args.interval_seconds)
    finally:
        planner.connection.close()


if __name__ == "__main__":
    main()
