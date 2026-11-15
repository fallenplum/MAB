"""E18 additive, bounded SFTP audit of existing unscored aggregation caches.

No remote shell, scorer, generation process, database or write operation is
used. Property names are checked before JSON values, and job messages and
question text are never decoded. Credentials exist only in process memory.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import datetime
import getpass
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import time

import paramiko

from cache_audit import FORBIDDEN_KEYS, RESULT_FIELDS, digest, json_property_names
from remote_cache_audit_v2 import ALIASES, EXPECTED_PIN, REMOTE_ROOT, pinned_transport


OPERATOR = "9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e"
PROMPT = "anonymous_final_answer_typed_v3"
SCORER_SHA = "6fc1d2f3cc81b3e954350143abd1071d6556e1cfe4f321d9333f2016baf2150d"
SCORER_VERSION = "safe_exact_v2_0_terminal_final_heading_fix"
ADAPTER_SHA = "acbc9cd911635f26da0e23cce4174bf6660a7604d29a5d7224f1df08e083f368"
NAMESPACE = "stage2_task_typed_final_answer_v2"
PROTECTED = ("sealed", "gold", "reference", "reward", "score", "credential", "secret")
SKIP_DIRECTORIES = {".git", "__pycache__", "runtime", "runtime_cache", "models", "envs", "cache", "tmp", "datasets", "data"}
METADATA_FILES = {"RUN_PLAN.json", "context.json", "EXPORT_MANIFEST.json", "EXPORT_RECEIPT.json", "OPERATOR_FREEZE_LOCK.json"}
META_KEYS = frozenset({
    "schema", "kind", "cohort", "job_split", "split", "generation_seed", "member_generation_seed",
    "aggregation_seed", "seed", "expected_aggregator_fingerprint", "aggregator_fingerprint", "operator_fingerprint",
    "execution_namespace", "prompt_version", "scorer_sha256", "scorer_version", "typed_adapter_sha256",
    "logical_fingerprint", "physical_fingerprint", "original_logical_fingerprint", "plan_sha256", "context_sha256",
    "software_test_only", "expected_jobs", "cohort_question_count", "teams_per_question", "member_count",
    "files_sha256", "files", "output", "output_dir", "output_root", "bundle", "remote_bundle", "owned_output_root", "jobs_path", "logical_jobs_path",
    "physical_jobs_path", "mapping_sha256", "raw_source_absent", "member_generation_seed", "run_status",
    "actual_model_spec", "runtime_versions", "gpu", "operator_lock_sha256", "fresh_admission_sha256",
    "source_receipt_sha256", "session_id", "role", "operator_identity", "aggregator_contract_sha256",
})
JOB_KEYS = frozenset({"job_id", "question_id", "task", "subtask", "split", "seed", "members", "candidate_order",
    "member_provenance", "prompt_version", "model_fingerprint", "source_hashes", "output_contract"})
ROW_KEYS = frozenset({"schema", "job_id", "logical_job_id", "question_id", "task", "split", "attempt_id",
    "request_sha256", "plan_sha256", "context_sha256", "status", "response", "cost_status", "record_sha256"})
RESPONSE_KEYS = frozenset({"status", "attempt_id", "job_id", "request_sha256", "model_fingerprint", "context_sha256",
    "result", "cost_status", "actual_usage"})


def _value_end(raw, start):
    """Locate one JSON value without interpreting string values."""
    position = start
    if raw[position] == 34:
        position += 1
        while position < len(raw):
            if raw[position] == 92:
                position += 2
            elif raw[position] == 34:
                return position + 1
            else:
                position += 1
        raise ValueError("Unterminated JSON string")
    if raw[position] in (91, 123):
        stack = [raw[position]]
        position += 1
        while stack and position < len(raw):
            value = raw[position]
            if value == 34:
                position = _value_end(raw, position)
                continue
            if value in (91, 123):
                stack.append(value)
            elif value in (93, 125):
                opening = stack.pop()
                if (opening, value) not in ((91, 93), (123, 125)):
                    raise ValueError("Mismatched JSON container")
            position += 1
        if stack:
            raise ValueError("Unterminated JSON container")
        return position
    while position < len(raw) and raw[position] not in b",}] \r\n\t":
        position += 1
    if position == start:
        raise ValueError("Empty JSON value")
    return position


def project_object(raw, allowed, nested=None):
    """Decode selected top-level fields after a whole-container key guard.

    Excluded values, notably messages and source question text, stay bytes.
    A reference/reward property anywhere rejects the complete container first.
    """
    names = json_property_names(raw)
    if any(name.casefold() in FORBIDDEN_KEYS for name in names):
        raise ValueError("PROTECTED_SCHEMA_VALUE_DECODE_REFUSED")
    raw = raw.strip()
    if not raw.startswith(b"{") or not raw.endswith(b"}"):
        raise ValueError("Expected JSON object")
    result, top_keys = {}, set()
    position = 1
    while position < len(raw) - 1:
        while raw[position] in b" \r\n\t,":
            position += 1
        if raw[position] == 125:
            break
        if raw[position] != 34:
            raise ValueError("Expected property name")
        end = _value_end(raw, position)
        key = json.loads(raw[position:end])
        if key in top_keys:
            raise ValueError("Duplicate JSON property")
        top_keys.add(key)
        position = end
        while raw[position] in b" \r\n\t":
            position += 1
        if raw[position] != 58:
            raise ValueError("Expected JSON colon")
        position += 1
        while raw[position] in b" \r\n\t":
            position += 1
        end = _value_end(raw, position)
        if key in allowed:
            if nested and key in nested:
                selected_keys, deeper = nested[key]
                result[key], selected_names = project_object(raw[position:end], selected_keys, deeper)
                if selected_names != set(result[key]):
                    raise ValueError("Nested object contains unapproved fields")
            else:
                result[key] = json.loads(raw[position:end])
        position = end
    return result, top_keys


def _safe_path(path):
    parts = PurePosixPath(path).parts
    if not path.startswith(REMOTE_ROOT + "/") and path != REMOTE_ROOT:
        raise ValueError("Outside inspected project root")
    if ".." in parts or any(term in part.casefold() for part in parts for term in PROTECTED):
        raise ValueError("Protected path")


def _read(sftp, path, limit):
    _safe_path(path)
    before = sftp.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("Only bounded regular JSON metadata or unscored output may be read")
    with sftp.open(path, "rb") as stream:
        raw = stream.read(limit + 1)
    after = sftp.lstat(path)
    if len(raw) != before.st_size or len(raw) > limit or (before.st_size, before.st_mtime) != (after.st_size, after.st_mtime):
        raise ValueError("File changed during read")
    return raw


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity_ok(item):
    return (item.get("aggregator_fingerprint", item.get("expected_aggregator_fingerprint")) == OPERATOR
        and item.get("prompt_version") == PROMPT and item.get("execution_namespace") == NAMESPACE
        and item.get("typed_adapter_sha256") == ADAPTER_SHA and item.get("scorer_sha256") == SCORER_SHA
        and item.get("scorer_version") == SCORER_VERSION)


def _fresh(item):
    return item.get("cohort") == "fresh_dev_test" or item.get("job_split", item.get("split")) == "fresh_dev_test"


def _directory_relevant(path, name):
    lower = name.casefold()
    if lower in SKIP_DIRECTORIES or any(term in lower for term in PROTECTED):
        return False
    if "member" in lower and "aggregation" not in lower:
        return False
    parent = PurePosixPath(path)
    if path == REMOTE_ROOT:
        return any(term in lower for term in ("agg", "typed", "stage2", "parallel_workers", "artifacts"))
    if parent.name == "parallel_workers_20261103":
        return lower.startswith("primary-aggregator-slot") or lower == "qwen3-8b"
    if parent.name in ("private", "preparation"):
        return any(term in lower for term in ("agg", "typed", "stage2", "fresh", "export"))
    return True


def _root_candidates(sftp, max_depth, max_entries):
    """Enumerate actual aggregation roots, worker slots, preparation and exports."""
    queue = deque([(REMOTE_ROOT, 0)])
    listings, metadata_files, output_files, other_jsonl, boundaries = [], [], [], [], []
    entries_seen = 0
    while queue:
        path, depth = queue.popleft()
        try:
            _safe_path(path)
            entries = sorted(sftp.listdir_attr(path), key=lambda entry: entry.filename)
        except (OSError, ValueError) as error:
            boundaries.append(dict(path=path, state="UNAVAILABLE_OR_PROTECTED", error_type=type(error).__name__))
            continue
        accepted = []
        for entry in entries:
            entries_seen += 1
            if entries_seen > max_entries:
                boundaries.append(dict(path=path, state="ENTRY_LIMIT_REACHED"))
                boundaries.extend(dict(path=p, state="NOT_VISITED_ENTRY_LIMIT") for p, _ in queue)
                return listings, metadata_files, output_files, other_jsonl, boundaries, entries_seen
            child = path + "/" + entry.filename
            try:
                _safe_path(child)
            except ValueError:
                boundaries.append(dict(path=child, state="PROTECTED_NAME_SKIPPED_NO_OPEN"))
                continue
            directory = stat.S_ISDIR(entry.st_mode)
            regular = stat.S_ISREG(entry.st_mode)
            accepted.append(dict(name=entry.filename, kind="directory" if directory else "regular" if regular else "symlink_or_special",
                bytes=entry.st_size, mtime=entry.st_mtime))
            if directory:
                if not _directory_relevant(path, entry.filename):
                    boundaries.append(dict(path=child, state="OUTSIDE_AGGREGATION_SCOPE_NO_OPEN"))
                elif depth >= max_depth:
                    boundaries.append(dict(path=child, state="DEPTH_LIMIT_NO_OPEN"))
                else:
                    queue.append((child, depth + 1))
            elif regular and entry.filename in METADATA_FILES:
                metadata_files.append(child)
            elif regular and entry.filename == "outputs.jsonl":
                output_files.append(child)
            elif regular and entry.filename.endswith(".jsonl"):
                other_jsonl.append(child)
        listings.append(dict(path=path, depth=depth, entries=accepted, directory_listing_complete=True))
    return listings, metadata_files, output_files, other_jsonl, boundaries, entries_seen


def _metadata(sftp, paths, limit):
    records = []
    for path in paths:
        item = dict(path=path)
        try:
            raw = _read(sftp, path, limit)
            projected, keys = project_object(raw, META_KEYS)
            item.update(state="SAFE_IDENTITY_METADATA_PROJECTED", sha256=_sha(raw), bytes=len(raw),
                top_level_keys=sorted(keys), identity=projected, frozen_operator_match=_identity_ok(projected),
                fresh_scope=_fresh(projected))
        except (OSError, ValueError, TypeError) as error:
            item.update(state="METADATA_REFUSED_OR_UNAVAILABLE", reason=str(error) if isinstance(error, ValueError) else type(error).__name__)
        records.append(item)
    return records


def _find_near(path, records, name):
    folder = str(PurePosixPath(path).parent)
    candidates = [row for row in records if PurePosixPath(row["path"]).name == name and "identity" in row
        and (str(PurePosixPath(row["path"]).parent) == folder
             or str(PurePosixPath(row["path"]).parent) == folder + "/inputs")]
    return candidates[0] if len(candidates) == 1 else None


def _dependency_jobs(sftp, output_path, plan_row, discovered_jsonl, limit):
    """Only open projected jobs in a verified fresh plan's own bundle/export."""
    plan = plan_row["identity"]
    files = plan.get("files_sha256", {})
    base = str(PurePosixPath(plan_row["path"]).parent)
    jobs, provenance, bridge = {}, [], {}
    for filename, expected_sha in files.items():
        if not isinstance(filename, str) or not filename.endswith(".jsonl") or "job" not in filename.casefold():
            continue
        if "/" in filename or "\\" in filename:
            continue
        path = base + "/" + filename
        if path not in discovered_jsonl:
            provenance.append(dict(path=path, state="NOT_IN_ENUMERATED_BOUNDARY"))
            continue
        try:
            raw = _read(sftp, path, limit)
            if _sha(raw) != expected_sha:
                raise ValueError("Job source file hash differs from RUN_PLAN")
            if any(key.casefold() in FORBIDDEN_KEYS for key in json_property_names(raw)):
                raise ValueError("PROTECTED_SCHEMA_VALUE_DECODE_REFUSED")
            count = 0
            for line in raw.splitlines():
                if not line.strip():
                    continue
                row, _ = project_object(line, JOB_KEYS)
                if row.get("split") != "fresh_dev_test":
                    raise ValueError("Fresh bundle contains another split")
                jobs[row["job_id"]] = row
                count += 1
            provenance.append(dict(path=path, state="FROZEN_HASH_PROJECTED_JOBS", sha256=_sha(raw), rows=count))
        except (OSError, KeyError, ValueError, TypeError) as error:
            provenance.append(dict(path=path, state="DEPENDENCY_REFUSED", reason=str(error) if isinstance(error, ValueError) else type(error).__name__))
    bridge_path = base + "/JOB_ID_MAP.json"
    try:
        raw = _read(sftp, bridge_path, limit)
        if _sha(raw) != files.get("JOB_ID_MAP.json"):
            raise ValueError("Bridge source file hash differs from RUN_PLAN")
        projected, _ = project_object(raw, {"schema", "jobs"})
        if projected.get("schema") != "stage1_logical_physical_job_bridge_v1":
            raise ValueError("Unrecognized immutable job bridge")
        bridge_fields = {"logical_job_id", "logical_full_job_sha256", "physical_job_id", "physical_full_job_sha256", "question_id", "task"}
        for row in projected["jobs"]:
            if set(row) != bridge_fields:
                raise ValueError("Unrecognized job bridge row schema")
            bridge[row["physical_job_id"]] = row
        provenance.append(dict(path=bridge_path, state="FROZEN_HASH_IDENTITY_BRIDGE", sha256=_sha(raw), rows=len(bridge)))
    except (OSError, KeyError, ValueError, TypeError) as error:
        provenance.append(dict(path=bridge_path, state="BRIDGE_REFUSED", reason=str(error) if isinstance(error, ValueError) else type(error).__name__))
    return jobs, bridge, provenance


def _verify_outputs(raw, plan_row, context_row, jobs, bridge, members):
    """Return only fully identity-closed success keys and physical cost rows."""
    if any(key.casefold() in FORBIDDEN_KEYS for key in json_property_names(raw)):
        raise ValueError("PROTECTED_SCHEMA_VALUE_DECODE_REFUSED")
    context = context_row["identity"]
    context_complete = set(context) == set(context_row["top_level_keys"])
    if not context_complete:
        raise ValueError("Context contains unprojected metadata; digest closure unavailable")
    context_sha = digest(context)
    plan = plan_row["identity"]
    generation_seed = plan.get("generation_seed", context.get("member_generation_seed"))
    member_index = {alias: index for index, alias in enumerate(ALIASES)}
    member_lookup = {(row["question_id"], row["member_index"], row["generation_seed"]): row for row in members}
    identities = {row["question_id"]: row["canonical_id"] for row in members}
    verified, rejected = [], []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row, keys = project_object(line, ROW_KEYS, {"response": (RESPONSE_KEYS, {"result": (RESULT_FIELDS, None)})})
            if keys != ROW_KEYS or row["schema"] != "v2_stage2_physical_generation_record_v1":
                raise ValueError("Unrecognized physical aggregation record")
            if digest({k: v for k, v in row.items() if k != "record_sha256"}) != row["record_sha256"]:
                raise ValueError("Record digest mismatch")
            if row["status"] != "success" or row["cost_status"] != "recorded_actual_usage":
                raise ValueError("Non-success or unsettled execution")
            response = row["response"]
            if not isinstance(response, dict) or not set(response) <= RESPONSE_KEYS:
                raise ValueError("Unrecognized response schema")
            result = response.get("result")
            if not isinstance(result, dict) or set(result) != RESULT_FIELDS:
                raise ValueError("Unrecognized generation result schema")
            job = jobs.get(row["job_id"])
            if job is None:
                raise ValueError("Immutable physical job dependency unavailable")
            link = bridge.get(row["job_id"])
            if link is None or not (link["logical_job_id"] == row["logical_job_id"]
                    and link["question_id"] == row["question_id"] and link["task"] == row["task"]
                    and link["logical_job_id"] in jobs):
                raise ValueError("Immutable logical/physical job bridge unavailable or inconsistent")
            qid = row["question_id"]
            if not (qid in identities and row["split"] == job["split"] == "fresh_dev_test"
                    and row["plan_sha256"] == plan_row["sha256"] and row["context_sha256"] == context_sha
                    and response.get("context_sha256") == context_sha and response.get("status") == "SUCCESS"
                    and response.get("job_id") == row["job_id"] and response.get("attempt_id") == row["attempt_id"]
                    and response.get("request_sha256") == row["request_sha256"]
                    and response.get("model_fingerprint") == plan.get("physical_fingerprint")
                    and job.get("model_fingerprint") == plan.get("physical_fingerprint")
                    and job.get("seed") == result.get("seed") == 42 and job.get("prompt_version") == PROMPT
                    and result.get("model_identity_verified") is True):
                raise ValueError("Generation/context/plan identity mismatch")
            aliases = job.get("members", [])
            if len(aliases) != 3 or len(set(aliases)) != 3 or not all(alias in member_index for alias in aliases):
                raise ValueError("Unexpected combination identity")
            dependencies = []
            source_hashes = job.get("source_hashes", {})
            for alias in aliases:
                prior = member_lookup.get((qid, member_index[alias], generation_seed))
                declared = job.get("member_provenance", {}).get(alias, {})
                if prior is None or not (source_hashes.get("question_sha256") == prior["source_question_sha256"]
                    and source_hashes.get("member_answers_sha256", {}).get(alias) == prior["answer_sha256"]
                    and declared.get("job_id") == prior["job_id"] and declared.get("record_sha256") == prior["record_sha256"]
                    and declared.get("model_fingerprint") == prior["model_fingerprint"]
                    and declared.get("model_id") == prior["model_id"] and declared.get("revision") == prior["model_revision"]):
                    raise ValueError("Member-answer dependency does not match verified cache")
                dependencies.append(prior["record_sha256"])
            if not isinstance(result.get("text"), str) or not result["text"].strip():
                raise ValueError("Empty aggregation prediction")
            if any(isinstance(result[k], bool) or not isinstance(result[k], int) or result[k] < 0
                   for k in ("prompt_tokens", "completion_tokens")):
                raise ValueError("Token measurement unavailable")
            verified.append(dict(question_id=qid, canonical_id=identities[qid], generation_seed=generation_seed,
                members=sorted(member_index[alias] for alias in aliases), aggregator_operator_fingerprint=OPERATOR,
                prompt_version=PROMPT, scorer_version=SCORER_VERSION, scorer_sha256=SCORER_SHA,
                physical_attempt_id=row["attempt_id"], physical_job_id=row["job_id"], logical_job_id=row["logical_job_id"],
                record_sha256=row["record_sha256"], answer_sha256=_sha(result["text"].encode()),
                member_record_sha256=dependencies, prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"], tokens=result["prompt_tokens"] + result["completion_tokens"],
                source_model_wall_s=result["wall_time_s"], storage="REMOTE_VERIFIED_NOT_LOCALLY_MIRRORED"))
        except (KeyError, ValueError, TypeError) as error:
            rejected.append(dict(line=number, reason=str(error) if isinstance(error, ValueError) else type(error).__name__))
    return verified, rejected


def inventory(transport, member_path, max_depth=9, max_entries=20000, max_bytes=64 * 1024 * 1024):
    members = json.loads(Path(member_path).read_text(encoding="utf-8"))["keys"]
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.get_channel().settimeout(20)
    try:
        listings, meta_paths, outputs, jobs_paths, boundaries, count = _root_candidates(sftp, max_depth, max_entries)
        metadata = _metadata(sftp, meta_paths, max_bytes)
        output_audits, all_keys, job_audits = [], [], []
        for path in outputs:
            context_row = _find_near(path, metadata, "context.json")
            plan_row = _find_near(path, metadata, "RUN_PLAN.json")
            # Original output dirs may bind a plan in a preparation bundle.
            if plan_row is None and context_row is not None:
                plan_hash = context_row["identity"].get("plan_sha256")
                matches = [row for row in metadata if PurePosixPath(row["path"]).name == "RUN_PLAN.json"
                    and row.get("sha256") == plan_hash]
                if matches:
                    plan_row = matches[0]
            if not (context_row and plan_row and _fresh(context_row["identity"]) and _fresh(plan_row["identity"])):
                output_audits.append(dict(path=path, state="NOT_VERIFIED_FRESH_SCOPE_PAYLOAD_NOT_OPENED"))
                continue
            if not (_identity_ok(context_row["identity"]) and _identity_ok(plan_row["identity"])):
                output_audits.append(dict(path=path, state="FROZEN_OPERATOR_MISMATCH_PAYLOAD_NOT_OPENED"))
                continue
            try:
                if plan_row["identity"].get("software_test_only", False):
                    raise ValueError("Software-test-only output")
                jobs, bridge, sources = _dependency_jobs(sftp, path, plan_row, jobs_paths, max_bytes)
                job_audits.extend(sources)
                raw = _read(sftp, path, max_bytes)
                verified, rejected = _verify_outputs(raw, plan_row, context_row, jobs, bridge, members)
                for row in verified:
                    row.update(source_path=path, source_sha256=_sha(raw))
                all_keys.extend(verified)
                output_audits.append(dict(path=path, state="FRESH_UNSCORED_IDENTITY_AUDITED", sha256=_sha(raw),
                    bytes=len(raw), verified_outputs=len(verified), rejected_rows=rejected))
            except (OSError, ValueError, KeyError, TypeError) as error:
                output_audits.append(dict(path=path, state="FRESH_CONTAINER_REFUSED_OR_UNAVAILABLE",
                    reason=str(error) if isinstance(error, ValueError) else type(error).__name__))
    finally:
        sftp.close()
    unique, conflicts = {}, []
    for row in all_keys:
        key = (row["canonical_id"], row["generation_seed"], tuple(row["members"]))
        if key in unique and unique[key]["record_sha256"] != row["record_sha256"]:
            conflicts.append(list(key))
        else:
            unique[key] = row
    for key in conflicts:
        unique.pop((key[0], key[1], tuple(key[2])), None)
    physical = {row["physical_attempt_id"]: row for row in all_keys}
    return dict(directory_metadata=listings, inspected_metadata=metadata, output_containers=output_audits,
        projected_job_sources=job_audits, boundaries=boundaries, entries_seen=count,
        max_depth=max_depth, max_entries=max_entries, max_file_bytes=max_bytes,
        reusable_aggregation_keys=list(unique.values()), physical_aggregation_ledger=list(physical.values()),
        conflicting_logical_keys=conflicts, verified_logical_key_count=len(unique),
        aggregate_payloads_opened=sum(row["state"] in ("FRESH_UNSCORED_IDENTITY_AUDITED", "FRESH_CONTAINER_REFUSED_OR_UNAVAILABLE") for row in output_audits),
        unscored_outputs_known_by_metadata=len(outputs), other_jsonl_files_known_by_metadata=len(jobs_paths),
        metadata_state_counts=dict(Counter(row["state"] for row in metadata)),
        scope="Known project routing root, aggregation roots, primary slots, preparation bundles and exports; listed omissions stay UNKNOWN",
        fresh_gold_or_reward_value_decodes=0, sealed_paths_opened=0, remote_commands_started=0, remote_writes=0,
        model_training_jobs=0, generation_jobs=0, scoring_jobs=0)


def qualify_projection():
    projected, _ = project_object(b'{"job_id":"x","messages":[{"content":"do not decode"}],"seed":42}', {"job_id", "seed"})
    assert projected == {"job_id": "x", "seed": 42}
    try:
        project_object(b'{"prediction":"safe","reference":"forbidden"}', {"prediction"})
    except ValueError as error:
        assert str(error) == "PROTECTED_SCHEMA_VALUE_DECODE_REFUSED"
    else:
        raise AssertionError("Reference value guard failed")
    projected, _ = project_object(b'{"prediction":"escaped \\"reference\\": remains prediction"}', {"prediction"})
    assert "reference" in projected["prediction"]
    return dict(schema_guard="PASS", field_projection="PASS", no_real_data_in_qualification=True)


def run(args, password_provider):
    qualification = qualify_projection()
    if args.output.exists():
        raise FileExistsError("Additive report refuses overwrite")
    started, transport = time.perf_counter(), None
    result = dict(schema="E18_remote_aggregation_cache_inventory_v3", host=args.host, port=args.port,
        host_hash=_sha(args.host.encode()), expected_pin=EXPECTED_PIN, authenticated=False,
        reported_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), projection_qualification=qualification,
        source_sha256=_sha(Path(__file__).read_bytes()), fresh_gold_or_reward_value_decodes=0,
        sealed_paths_opened=0, new_generation_or_scoring_jobs=0)
    try:
        transport, pin = pinned_transport(args.host, args.port)
        result["observed_pin"] = pin
        password = password_provider()
        try:
            transport.auth_password(args.user, password, fallback=False)
        finally:
            password = None
        if not transport.is_authenticated():
            raise paramiko.AuthenticationException("Authentication was not accepted")
        result.update(authenticated=True, status="AUTHENTICATED_READ_ONLY_AGGREGATION_INVENTORY")
        result["inventory"] = inventory(transport, args.local_cache_keys, args.max_depth, args.max_entries, args.max_bytes)
    except Exception as error:
        result.update(status="READ_ONLY_AGGREGATION_INVENTORY_FAILED", error_type=type(error).__name__)
    finally:
        if transport is not None:
            transport.close()
    result["elapsed_s"] = time.perf_counter() - started
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=10021)
    parser.add_argument("--user", default="chen")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-cache-keys", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--max-entries", type=int, default=20000)
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--password-stdin", action="store_true")
    arguments = parser.parse_args()
    provider = lambda: getpass.getpass("Existing MAB SSH authentication: ")
    if arguments.password_stdin:
        provider = lambda: sys.stdin.readline().rstrip("\r\n")
    report = run(arguments, provider)
    print(json.dumps(dict(status=report["status"], authenticated=report["authenticated"],
        host_hash=report["host_hash"], elapsed_s=report["elapsed_s"],
        verified_aggregation_keys=report.get("inventory", {}).get("verified_logical_key_count"))))
