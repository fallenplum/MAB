"""Recoverably collect closed P2 native TARs without extracting small files.

One SQLite writer transaction owns the two NTFS JSONL append positions. Bytes
are fsynced before the committed offsets move. Recovery truncates only bytes
beyond those offsets. Gold is never imported, read or scored by this entry.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile
import uuid


CODE = Path(__file__).resolve().parent
ASSETS = Path(r"E:\语言模型组合路由")
PRIVATE = ASSETS / "research/e20_p2_execution/private"
WORK = Path(r"D:\develop\CV\多臂老虎机\.e20_p2_cpu_runtime")
FILES = {"member_generation": "MEMBER_NATIVE_RECEIPTS.jsonl", "aggregation": "AGGREGATION_NATIVE_RECEIPTS.jsonl"}
OWNER_FILES = {"ATTEMPT_STATE.json", "LAST_TASK_RECEIPT.json", "records.jsonl"}
SCOPE_FILE_SHA = "835dc9fa321ff3b20c95d5a9410ce18e79084e01453e66a94b6e3501e9ca0d06"
SCOPE_CANONICAL_SHA = "b355581aaaab1a581e76dab35107a897948da7b4086a1d9e2a33be4a3e1cd1b9"
PREDICTION_SHA = "b790643119b6d85b370d34b6547e89e1316c26b26ad0d4a005e047babc521c75"
MODEL_SHA = "b8b786eeb0aaac6f869136f5353186c13f78aba0a716ebf0bcad145532d01f92"
REFERENCE_PLAN_SHA = "b24b870647fcd70c0b9479375e2eed2317e76f3dc963a21c830f231e9ae18060"


def encoded(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):
            checksum.update(block)
    return checksum.hexdigest()


def stamp(path):
    value = Path(path).stat()
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns)


def atomic_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name("."+path.name+"-"+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded(value)+b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def load_auditor():
    path = CODE.parent/"e18_confirmatory_execution/audit_closed_native_archives.py"
    spec = importlib.util.spec_from_file_location("p2_native_readonly_auditor",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module,path


class Context:
    """P2 metadata with the already qualified E18 native owner validator."""

    def __init__(self,args):
        self.audit,self.audit_path = load_auditor()
        self.scope = self.audit.read_json(args.scope)
        self.scope_file_sha = file_sha(args.scope)
        self.scope_sha = digest(self.scope)
        if (self.scope_file_sha != SCOPE_FILE_SHA or self.scope_sha != SCOPE_CANONICAL_SHA
                or self.scope.get("schema") != "p2_scope_contract_v1"
                or self.scope.get("cohort") != "e20_p2_confirm"):
            raise ValueError("The frozen P2 runtime scope differs")
        self.approval = self.audit.read_json(args.approval)
        self.approval_sha = file_sha(args.approval)
        if (self.approval.get("schema") != "p2_scope_approval_v1"
                or self.approval.get("status") != "APPROVED"
                or self.approval.get("cohort") != "e20_p2_confirm"
                or self.approval.get("software_test_only") is not False
                or not {"member_generation","aggregation"} <= set(self.approval.get("operations",[]))
                or self.approval.get("scope_sha256_by_role",{}).get("execution") != self.scope_sha
                or self.approval.get("bindings") != self.scope["bindings"]):
            raise PermissionError("P2 approval does not match the frozen scope")
        if (self.scope["bindings"]["final_model_sha256"] != MODEL_SHA
                or self.scope["bindings"]["final_prediction_sha256"] != PREDICTION_SHA
                or file_sha(args.predictions) != PREDICTION_SHA
                or file_sha(args.reference_plan) != REFERENCE_PLAN_SHA):
            raise ValueError("The fixed P2 model, prediction or reference locator identity differs")
        self.owned_root = PurePosixPath(self.approval["approved_output_root"])
        self.identities = {row["question_id"]:row["canonical_question_sha256"] for row in self.scope["question_keys"]}
        self.question_metadata = {row["question_id"]:row for row in self.scope["question_keys"]}
        self.allowed = {stage:set(map(tuple,self.scope[field])) for stage,field in
                        (("member_generation","required_member_keys"),("aggregation","required_aggregate_keys"))}
        if {stage:len(keys) for stage,keys in self.allowed.items()} != {"member_generation":41289,"aggregation":106536}:
            raise ValueError("The fixed P2 execution union differs")
        pool = self.audit.read_json(args.asset_root/"configs/pool_v1.json")
        self.profiles,self.aliases = {},{}
        for row in pool["models"]:
            profile,fingerprint = self.audit.model_fingerprint(self.audit.read_json(args.asset_root/row["config"]))
            if fingerprint != row["fingerprint"]:
                raise ValueError("Frozen member ModelSpec fingerprint differs")
            self.profiles["member_generation",row["index"]] = profile,fingerprint
            self.aliases[row["index"]] = row["alias"]
        if set(self.aliases) != set(range(6)):
            raise ValueError("Exactly six frozen member identities are required")
        profile,fingerprint = self.audit.model_fingerprint(self.audit.read_json(
            args.asset_root/"configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json"))
        if fingerprint != "32aed6079298ebcf8eee99add17dd5c4a6828a504f13dd5d9236b234776861d3":
            raise ValueError("Frozen aggregator ModelSpec fingerprint differs")
        self.profiles["aggregation",0] = profile,fingerprint
        self.bindings = dict(scope_sha256=self.scope_file_sha,scope_canonical_sha256=self.scope_sha,
            approval_sha256=self.approval_sha,prediction_sha256=PREDICTION_SHA,
            final_model_sha256=MODEL_SHA,reference_plan_sha256=REFERENCE_PLAN_SHA)

    def audit_owner(self,stage,key_hash,files):
        receipt = self.audit.strict_json(files["LAST_TASK_RECEIPT.json"])
        if receipt.get("schema") != "p2_execution_task_receipt_v1" or receipt.get("cohort") != "e20_p2_confirm":
            raise ValueError("Closed owner has another execution namespace")
        if receipt.get("stage") != stage:
            raise ValueError("Archived stage and native receipt differ")
        namespace_view = dict(files)
        namespace_view["LAST_TASK_RECEIPT.json"] = encoded(dict(
            receipt,schema="e18_execution_task_receipt_v2",cohort="e18_confirm"))
        checked = self.audit.audit_owner(self,stage,key_hash,namespace_view)
        native = receipt["native_record"]
        metadata = self.question_metadata[checked["task_key"][0]]
        if (native.get("task") != metadata["task"]
                or (native.get("subtask") or "") != metadata["subtask"]):
            raise ValueError("Native task metadata differs from the frozen question")
        if stage == "member_generation" and native.get("source_question_sha256") != metadata["question_sha256"]:
            raise ValueError("Native member input bytes differ from the frozen question")
        return receipt,checked


class Collector:
    def __init__(self,context,work_root):
        self.context = context
        self.root = Path(work_root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.recovered_uncommitted_bytes = 0
        self.connection = sqlite3.connect(self.root/"NATIVE_COLLECTION.sqlite",timeout=180,isolation_level=None)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS offsets (stage TEXT PRIMARY KEY, filename TEXT NOT NULL, committed_offset INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS outputs (
                stage TEXT NOT NULL, question_id TEXT NOT NULL, seed INTEGER NOT NULL, action_index INTEGER NOT NULL,
                native_sha TEXT NOT NULL, receipt_sha TEXT NOT NULL, physical_id TEXT NOT NULL UNIQUE,
                prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL, wall_seconds REAL NOT NULL,
                line_offset INTEGER NOT NULL, line_bytes INTEGER NOT NULL,
                PRIMARY KEY(stage,question_id,seed,action_index), UNIQUE(stage,native_sha));
            CREATE TABLE IF NOT EXISTS archives (
                archive_sha TEXT PRIMARY KEY, batch TEXT NOT NULL, manifest_sha TEXT NOT NULL,
                tasks INTEGER NOT NULL, new_outputs INTEGER NOT NULL, duplicate_outputs INTEGER NOT NULL,
                compressed_bytes INTEGER NOT NULL, committed_utc TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS archive_observations (
                path TEXT PRIMARY KEY, device INTEGER NOT NULL, inode INTEGER NOT NULL,
                bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, archive_sha TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS import_failures (
                attempt_id TEXT PRIMARY KEY, source_path TEXT NOT NULL, error_type TEXT NOT NULL, error_message TEXT NOT NULL);
        """)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            for key,value in context.bindings.items():
                old = self.connection.execute("SELECT value FROM settings WHERE key=?",(key,)).fetchone()
                if old is not None and old[0] != value:
                    raise ValueError("Existing collection belongs to another frozen P2 scope")
                self.connection.execute("INSERT OR IGNORE INTO settings VALUES (?,?)",(key,value))
            for stage,filename in FILES.items():
                self.connection.execute("INSERT OR IGNORE INTO offsets VALUES (?,?,0)",(stage,filename))
            self._recover()
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def _recover(self):
        recovered = 0
        for stage,filename,offset in self.connection.execute("SELECT stage,filename,committed_offset FROM offsets"):
            if FILES.get(stage) != filename or offset < 0:
                raise ValueError("Invalid committed file state")
            path = self.root/filename
            if not path.exists() and offset:
                raise ValueError("A committed native JSONL file is missing")
            mode = "r+b" if path.exists() else "w+b"
            with path.open(mode,buffering=0) as stream:
                size = stream.seek(0,os.SEEK_END)
                if size < offset:
                    raise ValueError("Native JSONL is shorter than its committed durable offset")
                if size != offset:
                    stream.truncate(offset)
                    os.fsync(stream.fileno())
                    recovered += size-offset
        self.recovered_uncommitted_bytes += recovered
        return recovered

    def _append(self,streams,receipt,checked):
        stage = checked["stage"]
        key = checked["task_key"]
        row = self.connection.execute("SELECT native_sha,physical_id FROM outputs WHERE stage=? AND question_id=? AND seed=? AND action_index=?",(stage,*key)).fetchone()
        if row is not None:
            if row != (checked["native_record_sha256"],checked["physical_attempt_id"]):
                raise ValueError("A semantic task has conflicting successful native outputs")
            return False
        line = encoded(receipt)+b"\n"
        offset = streams[stage].tell()
        remaining = memoryview(line)
        while remaining:
            written = streams[stage].write(remaining)
            if not written:
                raise OSError("Native JSONL append made no progress")
            remaining = remaining[written:]
        self.connection.execute("INSERT INTO outputs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",(
            stage,*key,checked["native_record_sha256"],digest(receipt),checked["physical_attempt_id"],
            checked["prompt_tokens"],checked["completion_tokens"],checked["wall_time_s"],offset,len(line)))
        return True

    def _consume_archive(self,path,streams):
        pending,owners,observed = {},{},{}
        manifest = None
        new = duplicates = 0
        with path.open("rb") as raw, gzip.GzipFile(fileobj=raw,mode="rb") as unpacked:
            with tarfile.open(fileobj=unpacked,mode="r|") as archive:
                for member in archive:
                    name = member.name
                    if not member.isfile() or name in observed or member.size < 0 or member.size > 64*1024*1024:
                        raise ValueError("Unexpected or excessive native TAR member")
                    parts = PurePosixPath(name).parts
                    if PurePosixPath(name).is_absolute() or ".." in parts:
                        raise ValueError("Unsafe native archive path")
                    with archive.extractfile(member) as stream:
                        payload = stream.read()
                    if len(payload) != member.size:
                        raise ValueError("Truncated native TAR payload")
                    observed[name] = dict(path=name,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
                    if name == "P2_NATIVE_EXPORT_MANIFEST.json":
                        if manifest is not None:
                            raise ValueError("Duplicate export manifest")
                        manifest = self.context.audit.strict_json(payload)
                        continue
                    if (len(parts) != 5 or parts[0] != "owned" or parts[1] != self.context.scope_sha
                            or parts[2] not in FILES or parts[4] not in OWNER_FILES):
                        raise ValueError("Archive contains an undeclared native owner file")
                    owner = parts[:4]
                    files = pending.setdefault(owner,{})
                    if parts[4] in files:
                        raise ValueError("Duplicate native owner file")
                    files[parts[4]] = payload
                    if set(files) == OWNER_FILES:
                        receipt,checked = self.context.audit_owner(parts[2],parts[3],files)
                        if owner in owners:
                            raise ValueError("Repeated closed native owner")
                        owners[owner] = checked
                        if self._append(streams,receipt,checked):
                            new += 1
                        else:
                            duplicates += 1
                        del pending[owner]
            # Read the gzip trailer to validate compressed-stream CRC and closure.
            while unpacked.read(1048576):
                pass
        if pending or manifest is None:
            raise ValueError("Closed archive has incomplete native owners or no manifest")
        if (manifest.get("schema") != "p2_closed_native_export_v1"
                or manifest.get("scope_sha256") != self.context.scope_sha or manifest.get("gold_reads") != 0):
            raise ValueError("Closed export manifest has another scope or generation isolation")
        expected_files = {"P2_NATIVE_EXPORT_MANIFEST.json"}
        seen_owners = set()
        prompt = completion = 0
        for item in manifest["tasks"]:
            owner = ("owned",self.context.scope_sha,item["stage"],item["digest"])
            checked = owners.get(owner)
            if (owner in seen_owners or checked is None or item["task_key"] != checked["task_key"]
                    or item["digest"] != digest(item["task_key"])
                    or item["native_record_sha256"] != checked["native_record_sha256"]):
                raise ValueError("Export manifest owner identity differs")
            expected_owner_files = {"/".join((*owner,name)) for name in OWNER_FILES}
            if {row["path"] for row in item["files"]} != expected_owner_files or len(item["files"]) != 3:
                raise ValueError("Export manifest owner payload coverage differs")
            for row in item["files"]:
                if observed.get(row["path"]) != row:
                    raise ValueError("Closed export payload bytes differ from its manifest")
                expected_files.add(row["path"])
            seen_owners.add(owner)
            prompt += checked["prompt_tokens"]
            completion += checked["completion_tokens"]
        if (seen_owners != set(owners) or expected_files != set(observed)
                or prompt != manifest["prompt_tokens"] or completion != manifest["completion_tokens"]):
            raise ValueError("Closed export task count or successful usage differs")
        return manifest,new,duplicates

    def import_archive(self,path):
        path = Path(path).resolve()
        before = stamp(path)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            recovered = self._recover()
            seen = self.connection.execute("SELECT device,inode,bytes,mtime_ns,archive_sha FROM archive_observations WHERE path=?",(str(path),)).fetchone()
            if seen is not None and tuple(seen[:4]) == before:
                self.connection.commit()
                return dict(status="ALREADY_IMPORTED_UNCHANGED",new_outputs=0,recovered_uncommitted_bytes=recovered)
            archive_sha = file_sha(path)
            if stamp(path) != before:
                raise ValueError("Archive changed during compressed-byte identification")
            prior = self.connection.execute("SELECT archive_sha FROM archives WHERE archive_sha=?",(archive_sha,)).fetchone()
            if prior is not None:
                self.connection.execute("INSERT OR REPLACE INTO archive_observations VALUES (?,?,?,?,?,?)",(str(path),*before,archive_sha))
                self.connection.commit()
                return dict(status="ALREADY_IMPORTED_SAME_BYTES",new_outputs=0,recovered_uncommitted_bytes=recovered)
            with ExitStack() as stack:
                streams = {stage:stack.enter_context((self.root/filename).open("r+b",buffering=0)) for stage,filename in FILES.items()}
                for stage,offset in self.connection.execute("SELECT stage,committed_offset FROM offsets"):
                    streams[stage].seek(offset)
                manifest,new,duplicates = self._consume_archive(path,streams)
                if stamp(path) != before:
                    raise ValueError("Closed archive changed during import")
                for stage,stream in streams.items():
                    stream.flush()
                    os.fsync(stream.fileno())
                    self.connection.execute("UPDATE offsets SET committed_offset=? WHERE stage=?",(stream.tell(),stage))
                self.connection.execute("INSERT INTO archives VALUES (?,?,?,?,?,?,?,?)",(
                    archive_sha,manifest["batch"],digest(manifest),len(manifest["tasks"]),new,duplicates,before[2],datetime.now(timezone.utc).isoformat()))
                self.connection.execute("INSERT OR REPLACE INTO archive_observations VALUES (?,?,?,?,?,?)",(str(path),*before,archive_sha))
                # Both native files are durable before their offsets and keys commit.
                self.connection.commit()
            return dict(status="ARCHIVE_COMMITTED",tasks=len(manifest["tasks"]),new_outputs=new,
                        duplicate_outputs=duplicates,recovered_uncommitted_bytes=recovered)
        except BaseException as error:
            self.connection.rollback()
            # Reacquire ownership before recovery, including an ambiguous commit.
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self._recover()
                self.connection.execute("INSERT INTO import_failures VALUES (?,?,?,?)",(uuid.uuid4().hex,str(path),type(error).__name__,str(error)))
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise
            raise

    def snapshot(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            recovered = self._recover()
            stages = {}
            for stage in FILES:
                count,prompt,completion,wall = self.connection.execute(
                    "SELECT COUNT(*),COALESCE(SUM(prompt_tokens),0),COALESCE(SUM(completion_tokens),0),COALESCE(SUM(wall_seconds),0) FROM outputs WHERE stage=?",(stage,)).fetchone()
                stages[stage] = dict(successful_outputs=count,expected_outputs=len(self.context.allowed[stage]),
                    prompt_tokens=prompt,completion_tokens=completion,total_successful_output_tokens=prompt+completion,
                    sum_successful_call_wall_seconds=wall)
            complete = all(row["successful_outputs"] == row["expected_outputs"] for row in stages.values())
            summary = dict(schema="p2_native_collection_progress_v1",status="COMPLETE_NATIVE_OUTPUTS" if complete else "PARTIAL_NATIVE_OUTPUTS_IMPORTED",
                utc=datetime.now(timezone.utc).isoformat(),stages=stages,
                successful_outputs=sum(row["successful_outputs"] for row in stages.values()),
                successful_output_prompt_tokens=sum(row["prompt_tokens"] for row in stages.values()),
                successful_output_completion_tokens=sum(row["completion_tokens"] for row in stages.values()),
                successful_output_total_tokens=sum(row["total_successful_output_tokens"] for row in stages.values()),
                imported_archives=self.connection.execute("SELECT COUNT(*) FROM archives").fetchone()[0],
                technical_import_failures=self.connection.execute("SELECT COUNT(*) FROM import_failures").fetchone()[0],
                recovered_uncommitted_bytes=self.recovered_uncommitted_bytes,all_required_outputs_frozen=complete,
                failure_closure_pending=True,failed_or_unknown_native_attempts_audited=False,
                all_attempt_total_tokens=None,
                cost_scope="Committed successful outputs only; failed and unknown native attempts require their separate ledger closure",
                new_llm_calls=0,reference_values_read=0,deterministic_scores_computed=0,
                scope_sha256=self.context.scope_file_sha,scope_canonical_sha256=self.context.scope_sha)
            freeze_path = self.root/"OUTPUTS_FROZEN.json"
            if complete and not freeze_path.exists():
                freeze = dict(status="ALL_REQUIRED_OUTPUTS_FROZEN",**self.context.bindings,
                    completed_required_member_calls=stages["member_generation"]["successful_outputs"],
                    completed_required_aggregate_calls=stages["aggregation"]["successful_outputs"],
                    member_records_sha256=file_sha(self.root/FILES["member_generation"]),
                    aggregate_records_sha256=file_sha(self.root/FILES["aggregation"]),
                    member_records_path=str(self.root/FILES["member_generation"]),
                    aggregate_records_path=str(self.root/FILES["aggregation"]),
                    successful_output_total_tokens=summary["successful_output_total_tokens"],
                    failure_closure_pending=True,all_attempt_total_tokens=None,
                    reference_values_read=0,new_llm_calls=0)
                atomic_json(freeze_path,freeze)
            elif complete:
                existing = self.context.audit.read_json(freeze_path)
                expected = dict(status="ALL_REQUIRED_OUTPUTS_FROZEN",**self.context.bindings,
                    completed_required_member_calls=stages["member_generation"]["successful_outputs"],
                    completed_required_aggregate_calls=stages["aggregation"]["successful_outputs"])
                if any(existing.get(key) != value for key,value in expected.items()):
                    raise ValueError("Existing final output freeze has incompatible bindings or counts")
            if not complete and freeze_path.exists():
                raise ValueError("Incomplete collection cannot have a final output freeze")
            atomic_json(self.root/"COLLECTION_PROGRESS.json",summary)
            self.connection.commit()
            return summary
        except BaseException:
            self.connection.rollback()
            raise

    def close(self):
        self.connection.close()


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root",type=Path,default=ASSETS)
    parser.add_argument("--scope",type=Path,default=PRIVATE/"runtime_launch_v2/inputs/EXECUTION_SCOPE_RUNTIME.json")
    parser.add_argument("--approval",type=Path,default=PRIVATE/"runtime_launch_v2/inputs/AUTHORIZATION_v2.json")
    parser.add_argument("--predictions",type=Path,default=PRIVATE/"P2_PREDICTIONS_FROZEN.npz")
    parser.add_argument("--reference-plan",type=Path,default=PRIVATE/"P2_REFERENCE_LOCATORS_FROZEN.json")
    parser.add_argument("--exports",type=Path,default=PRIVATE/"native_exports")
    parser.add_argument("--work-root",type=Path,default=WORK)
    parser.add_argument("--archive",type=Path,action="append")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    collector = Collector(Context(args),args.work_root)
    try:
        results = [collector.import_archive(path) for path in (args.archive or sorted(args.exports.glob("*.tar.gz")))]
        summary = collector.snapshot()
    finally:
        collector.close()
    print(json.dumps(dict(import_results=results,progress=summary),ensure_ascii=False))


if __name__ == "__main__":
    main()
