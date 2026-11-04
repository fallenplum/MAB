"""Persistent question-level reference permissions and output-level cost receipts.

Each SQLite database is one logical feedback account. Benchmark gold_reveals
count authorized reference reveals, never actual paid human annotations.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from pathlib import Path
import sqlite3


class LedgerError(ValueError):
    """Invalid permission, conflicting identity or receipt."""


class UnknownCostError(RuntimeError):
    """A persisted failed attempt has unresolved actual resource usage."""


@dataclass(frozen=True)
class ExecutionReceipt:
    execution_id: str
    attempt_id: str
    canonical_question: str
    role: str
    model_fingerprint: str
    generation_seed: int
    output_hash: str | None
    status: str
    input_tokens: int | None
    output_tokens: int | None
    wall_time_s: float | None


@dataclass(frozen=True)
class ScoreObservation:
    execution_id: str
    scorer_version: str
    reference_version: str
    value: float
    available_at: int
    judge_execution_id: str | None = None
    score_cpu_s: float = 0.0
    score_wall_s: float = 0.0


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(name + " must be a nonempty identity")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise LedgerError(name + " must be a nonnegative integer")
    return int(value)


def _duration(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise LedgerError(name + " must be a finite nonnegative duration")
    return float(value)


class BudgetFeedbackLedger:
    """Atomic, restart-safe accounting. Raw gold and output text stay external.

    Use has_success before invoking a backend. A successful execution_id is
    cached, while failed attempts retain separate actual costs. Unknown failed
    usage is committed before UnknownCostError is raised and blocks new work.
    """
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        try:
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            tables = self.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if version not in (0, 2) or (version == 0 and tables):
                raise LedgerError("Database belongs to another schema")
            self.db.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS gold (
                    canonical_question TEXT NOT NULL, reference_version TEXT NOT NULL,
                    requested_at INTEGER NOT NULL, available_at INTEGER NOT NULL, arrived_at INTEGER,
                    PRIMARY KEY(canonical_question, reference_version));
                CREATE TABLE IF NOT EXISTS attempts (
                    execution_id TEXT NOT NULL, attempt_id TEXT PRIMARY KEY,
                    canonical_question TEXT NOT NULL, role TEXT NOT NULL,
                    model_fingerprint TEXT NOT NULL, generation_seed INTEGER NOT NULL,
                    output_hash TEXT, status TEXT NOT NULL,
                    input_tokens INTEGER, output_tokens INTEGER, wall_time_s REAL);
                CREATE UNIQUE INDEX IF NOT EXISTS successful_execution
                    ON attempts(execution_id) WHERE status='success';
                CREATE TABLE IF NOT EXISTS observations (
                    execution_id TEXT NOT NULL, scorer_version TEXT NOT NULL,
                    reference_version TEXT NOT NULL, value REAL NOT NULL,
                    available_at INTEGER NOT NULL, judge_execution_id TEXT,
                    score_cpu_s REAL NOT NULL, score_wall_s REAL NOT NULL,
                    PRIMARY KEY(execution_id, scorer_version));
                CREATE TABLE IF NOT EXISTS overheads (
                    event_id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                    cpu_s REAL NOT NULL, wall_s REAL NOT NULL);
                PRAGMA user_version=2;
                COMMIT;
            """)
        except BaseException:
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def close(self):
        self.db.close()

    @contextmanager
    def _transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def _unknown_count(self):
        return self.db.execute("""SELECT COUNT(*) FROM attempts WHERE status='failure'
            AND (input_tokens IS NULL OR output_tokens IS NULL OR wall_time_s IS NULL)""").fetchone()[0]

    def _assert_settled(self):
        if self._unknown_count():
            raise UnknownCostError("Reconcile all unknown failed costs before new work")

    def request_gold(self, canonical_question, reference_version, *, now, available_at):
        _identity(canonical_question, "canonical_question")
        _identity(reference_version, "reference_version")
        now = _integer(now, "now")
        available_at = _integer(available_at, "available_at")
        with self._transaction():
            self._assert_settled()
            existing = self.db.execute("SELECT 1 FROM gold WHERE canonical_question=? AND reference_version=?",
                                       (canonical_question, reference_version)).fetchone()
            if existing:
                return False
            if available_at < now:
                raise LedgerError("Reference availability cannot precede its request")
            self.db.execute("INSERT INTO gold VALUES (?, ?, ?, ?, NULL)",
                            (canonical_question, reference_version, now, available_at))
        return True

    def arrive_gold(self, canonical_question, reference_version, *, now):
        now = _integer(now, "now")
        with self._transaction():
            self._assert_settled()
            row = self.db.execute("SELECT * FROM gold WHERE canonical_question=? AND reference_version=?",
                                  (canonical_question, reference_version)).fetchone()
            if row is None or now < row["available_at"]:
                raise LedgerError("Reference is unrequested or has not arrived")
            if row["arrived_at"] is not None:
                if now < row["arrived_at"]:
                    raise LedgerError("Cannot replay arrival before its recorded time")
                return False
            self.db.execute("UPDATE gold SET arrived_at=? WHERE canonical_question=? AND reference_version=?",
                            (now, canonical_question, reference_version))
        return True

    def has_success(self, execution_id):
        _identity(execution_id, "execution_id")
        with self._transaction():
            self._assert_settled()
            return self.db.execute("SELECT 1 FROM attempts WHERE execution_id=? AND status='success'",
                                   (execution_id,)).fetchone() is not None

    def record_execution(self, receipt):
        if not isinstance(receipt, ExecutionReceipt):
            raise LedgerError("ExecutionReceipt required")
        data = asdict(receipt)
        for name in ("execution_id", "attempt_id", "canonical_question", "model_fingerprint"):
            _identity(data[name], name)
        if data["role"] not in ("member", "aggregator", "judge") or data["status"] not in ("success", "failure"):
            raise LedgerError("Unknown execution role or status")
        data["generation_seed"] = _integer(data["generation_seed"], "generation_seed")
        for name in ("input_tokens", "output_tokens"):
            if data[name] is not None:
                data[name] = _integer(data[name], name)
        if data["wall_time_s"] is not None:
            data["wall_time_s"] = _duration(data["wall_time_s"], "wall_time_s")
        unknown = any(data[k] is None for k in ("input_tokens", "output_tokens", "wall_time_s"))
        if data["status"] == "success":
            _identity(data["output_hash"], "successful output_hash")
            if unknown:
                raise LedgerError("Successful executions require actual complete usage")
        with self._transaction():
            existing = self.db.execute("SELECT * FROM attempts WHERE attempt_id=?", (data["attempt_id"],)).fetchone()
            if existing is not None:
                if dict(existing) != data:
                    raise LedgerError("Conflicting attempt receipt")
                if unknown:
                    raise UnknownCostError("Existing failed attempt still has unknown cost")
                return False
            self._assert_settled()
            previous = self.db.execute("SELECT * FROM attempts WHERE execution_id=? LIMIT 1",
                                       (data["execution_id"],)).fetchone()
            identity = ("canonical_question", "role", "model_fingerprint", "generation_seed")
            if previous is not None and any(previous[k] != data[k] for k in identity):
                raise LedgerError("Execution identity changed across attempts")
            if self.db.execute("SELECT 1 FROM attempts WHERE execution_id=? AND status='success'",
                               (data["execution_id"],)).fetchone():
                raise LedgerError("Execution already succeeded; reuse its cache")
            columns = tuple(data)
            self.db.execute("INSERT INTO attempts (" + ",".join(columns) + ") VALUES (" +
                            ",".join("?" for _ in columns) + ")", tuple(data.values()))
        if unknown:
            raise UnknownCostError("Failed attempt persisted with unknown cost; reconcile before continuing")
        return True

    def reconcile_execution(self, attempt_id, *, input_tokens, output_tokens, wall_time_s):
        values = {"input_tokens": _integer(input_tokens, "input_tokens"),
                  "output_tokens": _integer(output_tokens, "output_tokens"),
                  "wall_time_s": _duration(wall_time_s, "wall_time_s")}
        with self._transaction():
            row = self.db.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None or row["status"] != "failure":
                raise LedgerError("Only recorded failed attempts can be reconciled")
            if any(row[k] is not None and row[k] != value for k, value in values.items()):
                raise LedgerError("Reconciliation cannot overwrite known actual usage")
            if all(row[k] is not None for k in values):
                return False
            self.db.execute("UPDATE attempts SET input_tokens=?,output_tokens=?,wall_time_s=? WHERE attempt_id=?",
                            (*values.values(), attempt_id))
        return True

    def record_score(self, observation, *, now):
        if not isinstance(observation, ScoreObservation):
            raise LedgerError("ScoreObservation required")
        data = asdict(observation)
        for name in ("execution_id", "scorer_version", "reference_version"):
            _identity(data[name], name)
        now = _integer(now, "now")
        data["available_at"] = _integer(data["available_at"], "available_at")
        if data["available_at"] > now:
            raise LedgerError("Score target is still pending")
        if isinstance(data["value"], bool) or not isinstance(data["value"], Real) or not math.isfinite(data["value"]) or not 0 <= data["value"] <= 1:
            raise LedgerError("Score must be finite and in [0,1]")
        data["value"] = float(data["value"])
        for name in ("score_cpu_s", "score_wall_s"):
            data[name] = _duration(data[name], name)
        with self._transaction():
            self._assert_settled()
            execution = self.db.execute("SELECT * FROM attempts WHERE execution_id=? AND status='success'",
                                        (data["execution_id"],)).fetchone()
            if execution is None or execution["role"] not in ("member", "aggregator"):
                raise LedgerError("Only a successful generated task output may be scored")
            gold = self.db.execute("SELECT * FROM gold WHERE canonical_question=? AND reference_version=?",
                                  (execution["canonical_question"], data["reference_version"])).fetchone()
            if gold is None or gold["arrived_at"] is None or gold["arrived_at"] > data["available_at"]:
                raise LedgerError("Matching question reference has not arrived for this score")
            if data["judge_execution_id"] is not None:
                judge = self.db.execute("SELECT * FROM attempts WHERE execution_id=? AND status='success'",
                                       (data["judge_execution_id"],)).fetchone()
                if judge is None or judge["role"] != "judge" or judge["canonical_question"] != execution["canonical_question"]:
                    raise LedgerError("Paid judge must have a successful receipt for the same question")
            existing = self.db.execute("SELECT * FROM observations WHERE execution_id=? AND scorer_version=?",
                                      (data["execution_id"], data["scorer_version"])).fetchone()
            if existing is not None:
                stable = ("reference_version", "value", "judge_execution_id")
                if any(existing[k] != data[k] for k in stable):
                    raise LedgerError("Conflicting original score observation")
                return False
            columns = tuple(data)
            self.db.execute("INSERT INTO observations (" + ",".join(columns) + ") VALUES (" +
                            ",".join("?" for _ in columns) + ")", tuple(data.values()))
        return True

    def record_overhead(self, event_id, kind, *, cpu_s, wall_s):
        """Persist actual bookkeeping overhead even when unknown inference cost blocks work."""
        _identity(event_id, "event_id")
        if kind not in ("planning", "update"):
            raise LedgerError("Overhead kind must be planning or update")
        values = (event_id, kind, _duration(cpu_s, "cpu_s"), _duration(wall_s, "wall_s"))
        with self._transaction():
            existing = self.db.execute("SELECT * FROM overheads WHERE event_id=?", (event_id,)).fetchone()
            if existing is not None:
                if tuple(existing) != values:
                    raise LedgerError("Conflicting overhead event")
                return False
            self.db.execute("INSERT INTO overheads VALUES (?,?,?,?)", values)
        return True

    def summary(self):
        with self._transaction():
            roles = {}
            for role in ("member", "aggregator", "judge"):
                rows = self.db.execute("SELECT * FROM attempts WHERE role=?", (role,)).fetchall()
                totals = {"calls": len(rows), "successful_executions": sum(r["status"] == "success" for r in rows),
                          "failure_attempts": sum(r["status"] == "failure" for r in rows)}
                for field in ("input_tokens", "output_tokens", "wall_time_s"):
                    totals[field] = None if any(r[field] is None for r in rows) else sum(r[field] for r in rows)
                roles[role] = totals
            overheads = {}
            for kind in ("planning", "update"):
                rows = self.db.execute("SELECT * FROM overheads WHERE kind=?", (kind,)).fetchall()
                overheads[kind] = {"events": len(rows), "cpu_s": sum(r["cpu_s"] for r in rows),
                                   "wall_s": sum(r["wall_s"] for r in rows)}
            scores = self.db.execute("SELECT COUNT(*),COALESCE(SUM(score_cpu_s),0),COALESCE(SUM(score_wall_s),0) FROM observations").fetchone()
            gold = self.db.execute("SELECT COUNT(*),COUNT(arrived_at) FROM gold").fetchone()
            unknown = self._unknown_count()
            return {"gold_reveals": gold[0], "gold_pending": gold[0] - gold[1], "gold_arrived": gold[1],
                    "scoring_ops": scores[0], "score_cpu_s": scores[1], "score_wall_s": scores[2],
                    "roles": roles, "overheads": overheads, "unknown_cost_attempts": unknown,
                    "blocked": bool(unknown), "budget_semantics": "Logical benchmark question/reference reveals; no claim of paid human annotation"}
