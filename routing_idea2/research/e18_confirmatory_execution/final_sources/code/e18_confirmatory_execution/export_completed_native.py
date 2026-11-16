"""Close completed native E18 owners into verified, resumable TAR.GZ batches.

This tool reads only explicit metadata references and three allowed files per
scope-owned task. Original owners stay in place. No model or scorer is called.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sqlite3
import stat
import sys
import tarfile
import time
import uuid


INFERENCE_SHA = "3be35e1be1d30487e10b45fabbeae7254a5931c297359d9ef37adf80dd16d0ef"
OWNER_FILES = ("ATTEMPT_STATE.json", "LAST_TASK_RECEIPT.json", "records.jsonl")
STAGES = ("member_generation", "aggregation")
FORBIDDEN_KEYS = {"gold", "gold_answer", "reference", "references", "reference_answer", "reference_answers",
    "ground_truth", "target", "targets", "label", "labels", "reward", "rewards", "score", "scores", "is_correct"}


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def _signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _safe_path(root, path):
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(root):
        raise ValueError("File escaped the explicit E18 root")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if any(word in part.casefold() for word in ("sealed", "gold", "reward", "credential", "secret")):
            raise ValueError("Protected path refused")
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024):
            raise ValueError("Symlink/reparse file or ancestor refused")
    return path


def stable_file(root, path, maximum=64 * 1024 * 1024):
    path = _safe_path(root, path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise ValueError("Only bounded regular metadata/native output files may be read")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if _signature(opened) != _signature(before):
            raise ValueError("Source changed before stable read")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if len(raw) != before.st_size or _signature(before) != _signature(after) or _signature(before) != _signature(after_open):
        raise ValueError("Source changed during stable read")
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
        stat_signature=list(_signature(after)), data=raw)


def safe_json(raw):
    # Property-name inspection precedes value decoding, including nested rows.
    position = 0
    while position < len(raw):
        if raw[position] != 34:
            position += 1
            continue
        start = position
        position += 1
        while position < len(raw):
            if raw[position] == 92:
                position += 2
            elif raw[position] == 34:
                position += 1
                break
            else:
                position += 1
        end = position
        while position < len(raw) and raw[position] in b" \r\n\t":
            position += 1
        if position < len(raw) and raw[position] == 58 and json.loads(raw[start:end]).casefold() in FORBIDDEN_KEYS:
            raise ValueError("Protected payload property refused before value decoding")
    return json.loads(raw)


def checked_json(root, reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("An exact metadata path/SHA reference is required")
    source = stable_file(root, reference["path"])
    if source["sha256"] != reference["sha256"]:
        raise ValueError("Explicit metadata reference hash differs")
    return safe_json(source["data"]), source


@contextmanager
def export_lock(path):
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        if os.name == "nt":
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


class Context:
    def __init__(self, root, scope_ref, references, software_test_only):
        self.root = root
        self.scope, scope_source = checked_json(root, scope_ref)
        self.scope_ref = dict(scope_ref)
        self.scope_sha = digest(self.scope)
        if self.scope.get("schema") != "e18_scope_contract_v2" or self.scope.get("role") != "execution" or self.scope.get("cohort") != "e18_confirm":
            raise ValueError("Exact E18 execution scope required")
        self.mode = software_test_only
        self.sources = {scope_source["path"]: scope_source}
        self.manifests, self.task_plans, self.profiles = {}, {}, {}
        self.identities = {row["question_id"]:row["canonical_question_sha256"] for row in self.scope["question_keys"]}
        self.allowed = {stage:set(map(tuple, self.scope["required_member_keys" if stage == "member_generation" else "required_aggregate_keys"])) for stage in STAGES}
        self.owned = None
        self.inference = None
        self.approval_shas = set()
        self.model_sources = {}
        for reference in references:
            self.visit(reference)
        if self.owned is None:
            raise ValueError("At least one explicit worker manifest/plan is required")

    def visit(self, reference):
        if reference["path"] in self.sources:
            if reference["sha256"] != self.sources[reference["path"]]["sha256"]:
                raise ValueError("Conflicting metadata source references")
            return
        value, source = checked_json(self.root, reference)
        self.sources[source["path"]] = source
        if isinstance(value, list):
            for item in value:
                self.visit(item)
            return
        if not isinstance(value, dict):
            raise ValueError("Unknown metadata registry value")
        if "manifest_refs" in value:
            for item in value["manifest_refs"]:
                self.visit(item)
            return
        if "plan_ref" in value:
            self.visit(value["plan_ref"])
            return
        if value.get("stage") not in STAGES or "task_keys_ref" not in value:
            raise ValueError("Unknown plan/registry schema")
        self.add_manifest(value, source)

    def add_manifest(self, manifest, source):
        if manifest.get("software_test_only") is not self.mode:
            raise ValueError("Worker manifest software mode differs")
        if manifest.get("scope_ref") != self.scope_ref:
            declared, _ = checked_json(self.root, manifest["scope_ref"])
            if digest(declared) != self.scope_sha:
                raise ValueError("Worker manifest references another scope")
        owned = Path(manifest["owned_output_root"])
        owned = _safe_path(self.root, owned)
        if self.owned is not None and owned != self.owned:
            raise ValueError("Worker manifests use different owned roots")
        self.owned = owned
        approval, approval_source = checked_json(self.root, manifest["approval_ref"])
        if not (approval.get("status") == "APPROVED" and approval.get("software_test_only") is self.mode
            and approval.get("cohort") == "e18_confirm" and approval.get("scope_sha256_by_role", {}).get("execution") == self.scope_sha
            and Path(approval.get("approved_output_root", "")) == owned and manifest["stage"] in approval.get("operations", [])):
            raise ValueError("Worker approval/scope/output boundary differs")
        if any(approval.get("bindings", {}).get(key) != value for key,value in self.scope["bindings"].items()):
            raise ValueError("Approval bindings differ")
        self.approval_shas.add(approval_source["sha256"])
        for ref in (manifest["adapter_ref"], manifest["scope_ref"], manifest["approval_ref"], manifest["task_keys_ref"]):
            metadata = stable_file(self.root, ref["path"])
            if metadata["sha256"] != ref["sha256"]:
                raise ValueError("Worker metadata/code source hash differs")
            self.sources[metadata["path"]] = metadata
        keys, _ = checked_json(self.root, manifest["task_keys_ref"])
        stage = manifest["stage"]
        keys = [tuple(key) for key in keys]
        if len(keys) != len(set(keys)) or not set(keys) <= self.allowed[stage]:
            raise ValueError("Worker task list exceeds the frozen union")
        asset_root = _safe_path(self.root, manifest["asset_root"])
        inference_source = stable_file(self.root, asset_root / "src/budget_router/inference.py")
        if inference_source["sha256"] != INFERENCE_SHA:
            raise ValueError("Original ModelSpec implementation differs")
        if self.inference is None:
            name = "_e18_export_pinned_inference"
            spec = importlib.util.spec_from_file_location(name, inference_source["path"])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            self.inference = module
        self.model_sources[inference_source["path"]] = inference_source["sha256"]
        core_sources = {"src/budget_router/member_execution.py":"b1eac1cb346f7c67f14de398df130c644193c4f527a33b00ab18de1844f6dfb2",
            "src/budget_router/execution.py":"2dd0fc4bf9e36eb63e4ade5143d07505e14780e59338230ea86817ddd8eae17f",
            "scripts/v2/typed_aggregation_execution.py":"acbc9cd911635f26da0e23cce4174bf6660a7604d29a5d7224f1df08e083f368"}
        for relative,checksum in core_sources.items():
            scientific_source=stable_file(self.root,asset_root/relative)
            if scientific_source["sha256"]!=checksum:
                raise ValueError("Original native execution source differs")
            self.model_sources[scientific_source["path"]]=checksum
        pool_path = asset_root / "configs/pool_v1.json"
        pool_source = stable_file(self.root, pool_path)
        pool = safe_json(pool_source["data"])["models"]
        self.model_sources[pool_source["path"]]=pool_source["sha256"]
        for entry in pool:
            key = "member_generation", entry["index"]
            config_source = stable_file(self.root, asset_root / entry["config"])
            profile = self.inference.ModelSpec(**safe_json(config_source["data"]))
            if self.inference.configuration_fingerprint(profile) != entry["fingerprint"]:
                raise ValueError("Original member pool/configuration differs")
            self.profiles[key] = (profile, entry["fingerprint"], entry["alias"])
            self.model_sources[config_source["path"]] = config_source["sha256"]
        agg_source = stable_file(self.root, asset_root / "configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json")
        agg = self.inference.ModelSpec(**safe_json(agg_source["data"]))
        agg_fp = self.inference.configuration_fingerprint(agg)
        if agg_fp != "32aed6079298ebcf8eee99add17dd5c4a6828a504f13dd5d9236b234776861d3":
            raise ValueError("Frozen Qwen3 aggregator configuration differs")
        self.profiles["aggregation", 0] = agg, agg_fp, None
        self.model_sources[agg_source["path"]] = agg_source["sha256"]
        self.manifests[source["sha256"]] = dict(reference={"path":source["path"], "sha256":source["sha256"]}, manifest=manifest)
        for key in keys:
            self.task_plans.setdefault((stage,key), set()).add(source["sha256"])


def validate_payload(context, stage, key, payloads):
    fence, receipt = safe_json(payloads["ATTEMPT_STATE.json"]), safe_json(payloads["LAST_TASK_RECEIPT.json"])
    rows = [safe_json(line) for line in payloads["records.jsonl"].splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("A closed owner must contain exactly one committed native attempt")
    native = rows[0]
    if native.get("record_sha256") != digest({name:value for name,value in native.items() if name != "record_sha256"}):
        raise ValueError("Native record semantic checksum differs")
    if not (native.get("status") == "success" and native.get("cost_status") == "recorded_actual_usage"):
        raise ValueError("Native output is not a settled success")
    if not (receipt.get("schema") == "e18_execution_task_receipt_v2" and receipt.get("status") == "SUCCESS"
        and receipt.get("scope_sha256") == context.scope_sha and receipt.get("software_test_only") is context.mode
        and receipt.get("stage") == stage and receipt.get("approval_sha256") in context.approval_shas
        and receipt.get("bindings") == context.scope["bindings"] and Path(receipt.get("approved_output_root", "")) == context.owned
        and receipt.get("cohort")=="e18_confirm" and receipt.get("transport_split")==context.scope["transport_split"]
        and native.get("split")==context.scope["transport_split"]):
        raise ValueError("Task receipt scope, mode, approval or status differs")
    task = receipt["task_key"]
    index_field = "member_index" if stage == "member_generation" else "combination_index"
    if not (task.get("question_id") == key[0] and task.get("generation_seed") == key[1]
        and task.get(index_field) == key[2] and task.get("canonical_id") == context.identities[key[0]]):
        raise ValueError("Semantic task identity differs")
    expected = dict(status="SUCCESS_COMMITTED",scope_sha256=context.scope_sha,task_key=list(key),stage=stage,
        native_job_id=native["job_id"],native_record_sha256=native["record_sha256"],software_test_only=context.mode)
    if any(fence.get(name) != value for name,value in expected.items()):
        raise ValueError("Task fence is not the exact committed success")
    if fence.get("approval_sha256")!=receipt["approval_sha256"]:
        raise ValueError("Committed fence approval differs")
    if receipt.get("native_record") != native or receipt.get("native_record_sha256") != native["record_sha256"] or receipt.get("native_job_id") != native["job_id"] or receipt.get("physical_attempt_id") != native["attempt_id"]:
        raise ValueError("Native row, receipt and physical execution differ")
    profile, fingerprint, alias = context.profiles[stage, key[2] if stage == "member_generation" else 0]
    result = native["result"]
    if not (native.get("question_id") == key[0] and native.get("model_fingerprint") == fingerprint
        and result.get("model_id") == profile.model_id and result.get("model_revision") == profile.revision
        and result.get("backend") == profile.backend and result.get("model_identity_verified") is True
        and result.get("seed") == native.get("seed") == (key[1] if stage == "member_generation" else 42)):
        raise ValueError("Native result does not match frozen ModelSpec/seed")
    if stage == "member_generation" and (native.get("model_alias") != alias or native.get("sample_index") != 0 or native.get("sampling_control") is not False or native.get("prompt_version") != "independent_user_only_brief_final_v2"):
        raise ValueError("Original ordinary member protocol differs")
    if stage == "aggregation" and (receipt.get("aggregator_fingerprint") != context.scope["operator_fingerprint"] or receipt.get("aggregation_seed") != 42 or native.get("prompt_version") != "anonymous_final_answer_typed_v3"):
        raise ValueError("Frozen typed aggregation protocol differs")
    for name, maximum in (("prompt_tokens",profile.max_input_tokens),("completion_tokens",profile.max_new_tokens)):
        if type(result.get(name)) is not int or not 0 <= result[name] <= maximum:
            raise ValueError("Actual token usage is absent or outside frozen limits")
    if type(result.get("wall_time_s")) not in (int,float) or not math.isfinite(result["wall_time_s"]) or result["wall_time_s"] < 0:
        raise ValueError("Actual native execution time is absent")
    return dict(task_id=digest([context.scope_sha,stage,list(key)]),stage=stage,task_key=list(key),
        native_job_id=native["job_id"],native_record_sha256=native["record_sha256"],physical_attempt_id=native["attempt_id"],
        model_fingerprint=fingerprint,prompt_tokens=result["prompt_tokens"],completion_tokens=result["completion_tokens"],
        manifest_sha256s=sorted(context.task_plans.get((stage,key),set())))


def snapshot_task(context, stage, key):
    owner = context.owned / context.scope_sha / stage / digest(list(key))
    if not owner.exists():
        return None
    files = {name:stable_file(context.root,owner/name) for name in OWNER_FILES}
    task = validate_payload(context,stage,key,{name:item["data"] for name,item in files.items()})
    # Close the cross-file observation only if all three sources remain equal.
    for name,item in files.items():
        current = stable_file(context.root,item["path"])
        if current["sha256"] != item["sha256"] or current["stat_signature"] != item["stat_signature"]:
            raise ValueError("Native owner changed across the closed snapshot")
        item["archive_path"] = str(Path(item["path"]).relative_to(context.owned)).replace("\\","/")
    return task, files


def tar_member(archive, name, raw):
    info=tarfile.TarInfo(name)
    info.size=len(raw)
    info.mode=0o600
    info.mtime=0
    archive.addfile(info,io.BytesIO(raw))


def validate_archive(path, context):
    with tarfile.open(path,"r:gz") as archive:
        members=archive.getmembers()
        if any(not member.isfile() or member.size>64*1024*1024 or PurePosixPath(member.name).is_absolute() or ".." in PurePosixPath(member.name).parts for member in members):
            raise ValueError("Archive contains symlink/directory/unsafe entries")
        names=[member.name for member in members]
        if len(set(names)) != len(names) or "EXPORT_MANIFEST.json" not in names:
            raise ValueError("Archive manifest or unique paths are absent")
        manifest=safe_json(archive.extractfile("EXPORT_MANIFEST.json").read())
        if manifest.get("schema") != "e18_closed_native_export_manifest_v1" or manifest.get("scope_sha256") != context.scope_sha or manifest.get("software_test_only") is not context.mode:
            raise ValueError("Closed archive scope/mode differs")
        expected={item["path"]:item for item in manifest["files"]}
        if not 1<=len(manifest["tasks"])<=1000 or len(expected)!=len(manifest["files"]):
            raise ValueError("Archive must contain nonempty bounded tasks and unique files")
        if set(names) != set(expected)|{"EXPORT_MANIFEST.json"}:
            raise ValueError("Archive file inventory is incomplete")
        payloads={}
        for name,item in expected.items():
            if not name.startswith(context.scope_sha+"/") or PurePosixPath(name).name not in OWNER_FILES:
                raise ValueError("Archive contains a non-native/protected file")
            raw=archive.extractfile(name).read()
            if len(raw)!=item["bytes"] or hashlib.sha256(raw).hexdigest()!=item["sha256"]:
                raise ValueError("Closed archive file bytes differ")
            payloads[name]=raw
        seen=set()
        for declared in manifest["tasks"]:
            stage,key=declared["stage"],tuple(declared["task_key"])
            if stage not in STAGES or key not in context.allowed[stage]:
                raise ValueError("Closed archive contains a foreign task")
            prefix=context.scope_sha+"/"+stage+"/"+digest(list(key))+"/"
            task=validate_payload(context,stage,key,{name:payloads[prefix+name] for name in OWNER_FILES})
            if any(task[name]!=declared[name] for name in task if name!="manifest_sha256s") or task["task_id"] in seen:
                raise ValueError("Closed archive task/native identity differs")
            seen.add(task["task_id"])
        if len(expected)!=3*len(seen):
            raise ValueError("Closed archive lacks exactly three files per task")
        return manifest


def register_archive(database, path, archive_sha, manifest):
    database.execute("BEGIN IMMEDIATE")
    try:
        previous=database.execute("SELECT archive_sha256 FROM archives WHERE archive_id=?",(manifest["archive_id"],)).fetchone()
        if previous:
            if previous[0]!=archive_sha:
                raise ValueError("One closed archive ID has different bytes")
        else:
            database.execute("INSERT INTO archives VALUES (?,?,?,?)",(manifest["archive_id"],str(path),archive_sha,digest(manifest)))
            for task in manifest["tasks"]:
                database.execute("INSERT INTO tasks VALUES (?,?,?,?)",(task["task_id"],manifest["archive_id"],task["native_record_sha256"],task["physical_attempt_id"]))
        database.commit()
    except BaseException:
        database.rollback()
        raise


def export_completed(root,scope_ref,references,*,batch_size=750,event_paths=(),software_test_only=False,crash_hook=None):
    started=time.perf_counter()
    root=Path(root).resolve(strict=True)
    minimum=1 if software_test_only else 500
    if type(batch_size) is not int or not minimum<=batch_size<=1000:
        raise ValueError("Batch size must be bounded by 1,000 tasks")
    context=Context(root,scope_ref,references,software_test_only)
    export_root=root/"exports"
    export_root.mkdir(exist_ok=True)
    _safe_path(root,export_root)
    for name in ("EXPORT.lock","EXPORT_REGISTRY.sqlite3"):
        path=export_root/name
        if path.exists() or path.is_symlink(): _safe_path(root,path)
    result=dict(schema="e18_completed_native_export_result_v1",scope_sha256=context.scope_sha,
        new_archives=[],recovered_archives=[],deferred=[],event_metadata=[],new_model_calls=0,scorer_calls=0,gold_reads=0,
        original_execution_files_deleted=0,software_test_only=software_test_only)
    with export_lock(export_root/"EXPORT.lock"):
        database=sqlite3.connect(export_root/"EXPORT_REGISTRY.sqlite3")
        database.execute("PRAGMA synchronous=FULL")
        database.executescript("CREATE TABLE IF NOT EXISTS archives(archive_id TEXT PRIMARY KEY,path TEXT NOT NULL,archive_sha256 TEXT NOT NULL,manifest_sha256 TEXT NOT NULL);CREATE TABLE IF NOT EXISTS tasks(task_id TEXT PRIMARY KEY,archive_id TEXT NOT NULL,native_record_sha256 TEXT NOT NULL,physical_attempt_id TEXT NOT NULL UNIQUE);")
        try:
            # Existing registry claims are valid only while their exact archive exists.
            for archive_id,path,checksum,_ in database.execute("SELECT * FROM archives").fetchall():
                if file_sha(_safe_path(root,path))!=checksum:
                    raise ValueError("Registered archive is missing/corrupt; original owners must be reconciled")
            for path in sorted(export_root.glob("native-*.tar.gz")):
                _safe_path(root,path)
                manifest=validate_archive(path,context)
                present=database.execute("SELECT 1 FROM archives WHERE archive_id=?",(manifest["archive_id"],)).fetchone()
                if not present:
                    checksum=file_sha(path)
                    register_archive(database,path,checksum,manifest)
                    result["recovered_archives"].append(dict(path=str(path),sha256=checksum,tasks=len(manifest["tasks"])))
            for path in event_paths:
                try:
                    source=stable_file(root,path)
                    for line in source["data"].splitlines():
                        if line.strip(): safe_json(line)
                    result["event_metadata"].append({key:value for key,value in source.items() if key!="data"})
                except (OSError,ValueError) as error:
                    result["event_metadata"].append(dict(path=str(path),status="UNSTABLE_OR_REFUSED",error_type=type(error).__name__))
            done={row[0] for row in database.execute("SELECT task_id FROM tasks")}
            pending=[]
            for (stage,key) in sorted(context.task_plans):
                task_id=digest([context.scope_sha,stage,list(key)])
                if task_id in done: continue
                try:
                    snapshot=snapshot_task(context,stage,key)
                    if snapshot is not None: pending.append(snapshot)
                except (OSError,ValueError,KeyError,TypeError) as error:
                    result["deferred"].append(dict(stage=stage,task_key=list(key),error_type=type(error).__name__,
                        reason=str(error) if isinstance(error,ValueError) else type(error).__name__))
            for begin in range(0,len(pending),batch_size):
                batch=pending[begin:begin+batch_size]
                archive_id=uuid.uuid4().hex
                files=[dict(path=item["archive_path"],bytes=item["bytes"],sha256=item["sha256"],source_stat_signature=item["stat_signature"]) for _,source_files in batch for item in source_files.values()]
                manifest=dict(schema="e18_closed_native_export_manifest_v1",archive_id=archive_id,
                    scope_sha256=context.scope_sha,scope_ref=scope_ref,owned_root=str(context.owned),
                    scope_bindings=context.scope["bindings"],
                    software_test_only=software_test_only,tasks=[task for task,_ in batch],files=files,
                    metadata_sources=[dict(path=value["path"],bytes=value["bytes"],sha256=value["sha256"]) for value in context.sources.values()],
                    scientific_model_source_hashes=context.model_sources,exporter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    worker_sources=[dict(manifest_sha256=checksum,adapter_ref=value["manifest"]["adapter_ref"],
                        task_keys_ref=value["manifest"]["task_keys_ref"],backend=value["manifest"].get("backend"))
                        for checksum,value in context.manifests.items()],
                    original_files_retained=True,new_model_calls=0,scorer_calls=0,gold_reads=0)
                temporary=export_root/("native-"+archive_id+".tar.gz.partial")
                with temporary.open("xb") as stream:
                    with gzip.GzipFile(filename="",mode="wb",fileobj=stream,mtime=0) as compressed:
                        with tarfile.open(fileobj=compressed,mode="w") as archive:
                            for _,source_files in batch:
                                for item in source_files.values(): tar_member(archive,item["archive_path"],item["data"])
                            tar_member(archive,"EXPORT_MANIFEST.json",encoded(manifest))
                    stream.flush();os.fsync(stream.fileno())
                if crash_hook: crash_hook("after_archive_build_before_source_recheck",temporary)
                changed=False
                for task,source_files in batch:
                    try:
                        for original in source_files.values():
                            current=stable_file(root,original["path"])
                            if current["sha256"]!=original["sha256"] or current["stat_signature"]!=original["stat_signature"]:
                                raise ValueError("Source changed while archive was being built")
                    except (OSError,ValueError) as error:
                        changed=True
                        result["deferred"].append(dict(stage=task["stage"],task_key=task["task_key"],
                            error_type=type(error).__name__,reason="SOURCE_CHANGED_DURING_ARCHIVE_BATCH_NOT_PUBLISHED"))
                if changed:
                    continue
                validate_archive(temporary,context)
                if crash_hook: crash_hook("before_archive_publish",temporary)
                destination=export_root/("native-"+archive_id+".tar.gz")
                os.replace(temporary,destination)
                if os.name!="nt":
                    descriptor=os.open(export_root,os.O_RDONLY)
                    try: os.fsync(descriptor)
                    finally: os.close(descriptor)
                checksum=file_sha(destination)
                if crash_hook: crash_hook("after_archive_publish_before_registry",destination)
                register_archive(database,destination,checksum,manifest)
                result["new_archives"].append(dict(path=str(destination),sha256=checksum,bytes=destination.stat().st_size,
                    manifest_sha256=digest(manifest),tasks=len(batch)))
            result.update(status="CLOSED_NATIVE_ARCHIVES_COMPLETE" if result["new_archives"] else "NO_NEW_STABLE_COMPLETED_TASKS",
                newly_closed_tasks=sum(row["tasks"] for row in result["new_archives"]),
                registered_tasks=database.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
                unclosed_partial_archives=len(list(export_root.glob("*.partial"))),wall_s=time.perf_counter()-started)
        finally:
            database.close()
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--scope-ref",nargs=2,required=True,metavar=("PATH","SHA256"))
    parser.add_argument("--plan-ref",nargs=2,action="append",default=[],metavar=("PATH","SHA256"))
    parser.add_argument("--manifest-ref",nargs=2,action="append",default=[],metavar=("PATH","SHA256"))
    parser.add_argument("--registry-ref",nargs=2,action="append",default=[],metavar=("PATH","SHA256"))
    parser.add_argument("--event-path",type=Path,action="append",default=[])
    parser.add_argument("--batch-size",type=int,default=750)
    args=parser.parse_args()
    refs=[dict(path=path,sha256=checksum) for path,checksum in args.plan_ref+args.manifest_ref+args.registry_ref]
    print(json.dumps(export_completed(args.root,dict(path=args.scope_ref[0],sha256=args.scope_ref[1]),refs,
        batch_size=args.batch_size,event_paths=args.event_path),ensure_ascii=False,indent=2,allow_nan=False))
