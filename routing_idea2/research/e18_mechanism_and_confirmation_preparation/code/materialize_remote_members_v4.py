"""Mirror only eight hash-audited, member-only E18 remote cache source files.

This is read-only SFTP transport of existing generation outputs. Local copies
remain private. No remote shell, generation, scoring or write API is invoked.
Files changed since V2 postcheck are explicitly refused, preserving old keys.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime
import getpass
import hashlib
import json
from pathlib import Path
import sys
import time

import paramiko

from remote_cache_audit_v2 import EXPECTED_PIN, _member_rows, _read_bytes, pinned_transport


IDENTITY_FIELDS = (
    "question_id", "canonical_id", "member_index", "generation_seed", "physical_attempt_id",
    "job_id", "record_sha256", "source_question_sha256", "answer_sha256", "model_fingerprint",
    "model_id", "model_revision", "member_prompt_version", "member_mode", "sample_index", "sampling_control",
    "prompt_tokens", "completion_tokens", "tokens", "source_model_wall_s",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def mirror(transport, keys_path, mirror_root):
    key_bytes = Path(keys_path).read_bytes()
    keys = json.loads(key_bytes)["keys"]
    wanted = [row for row in keys if row.get("storage") == "REMOTE_VERIFIED_NOT_LOCALLY_MIRRORED"]
    if len(wanted) != 2130:
        raise ValueError("Frozen V2 newly available cache scope differs from 2,130 keys")
    by_file = defaultdict(list)
    for row in wanted:
        by_file[row["source_path"], row["source_sha256"]].append(row)
    if len(by_file) != 8:
        raise ValueError("Frozen V2 newly available scope differs from eight member sources")
    expected = {row["member_index"]: row for row in keys}
    identities = {row["question_id"]: row["canonical_id"] for row in keys}
    files, materialized = [], []
    mirror_root.mkdir(parents=True, exist_ok=True)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.get_channel().settimeout(20)
    try:
        for (remote_path, source_sha), expected_rows in sorted(by_file.items()):
            entry = dict(remote_path=remote_path, audited_source_sha256=source_sha,
                expected_key_count=len(expected_rows))
            try:
                raw = _read_bytes(sftp, remote_path)
                observed_sha = sha(raw)
                entry.update(observed_source_sha256=observed_sha, observed_bytes=len(raw))
                if observed_sha != source_sha:
                    entry["status"] = "SOURCE_CHANGED_SINCE_POSTCHECK_NO_MIRROR_NO_KEY_ASSUMPTION"
                    files.append(entry)
                    continue
                scope = {(row["member_index"], row["generation_seed"]) for row in expected_rows}
                if len(scope) != 1:
                    raise ValueError("Member-source scope is ambiguous")
                member, seed = next(iter(scope))
                observed_rows, state = _member_rows(raw, expected[member], identities, member, seed)
                if observed_rows is None:
                    raise ValueError(state)
                expected_by_question = {row["question_id"]: row for row in expected_rows}
                if len(observed_rows) != len(expected_rows) or {row["question_id"] for row in observed_rows} != set(expected_by_question):
                    raise ValueError("Complete source question-key coverage differs from frozen V2")
                for row in observed_rows:
                    original = expected_by_question[row["question_id"]]
                    if any(row[field] != original[field] for field in IDENTITY_FIELDS):
                        raise ValueError("A materialized record differs from its V2 identity or measured cost")
                target = mirror_root / (source_sha + ".jsonl")
                if target.exists():
                    if sha(target.read_bytes()) != source_sha:
                        raise ValueError("Pre-existing private mirror has a different SHA")
                    copy_status = "EXISTING_EXACT_PRIVATE_MIRROR_REUSED"
                else:
                    with target.open("xb") as stream:
                        stream.write(raw)
                    if sha(target.read_bytes()) != source_sha:
                        raise ValueError("Private byte mirror verification failed")
                    copy_status = "EXACT_PRIVATE_BYTE_MIRROR_CREATED"
                for original in expected_rows:
                    item = dict(original)
                    item.update(storage="LOCAL_PRIVATE_VERIFIED_REMOTE_MIRROR", local_source_path=str(target),
                        local_source_sha256=source_sha, original_remote_storage=original["storage"])
                    materialized.append(item)
                entry.update(status=copy_status, local_source_path=str(target),
                    local_source_sha256=source_sha, verified_keys=len(expected_rows),
                    prompt_tokens=sum(row["prompt_tokens"] for row in expected_rows),
                    completion_tokens=sum(row["completion_tokens"] for row in expected_rows))
            except (OSError, ValueError, KeyError, TypeError) as error:
                entry.update(status="REFUSED_OR_UNAVAILABLE_NO_CACHE_ADDED",
                    reason=str(error) if isinstance(error, ValueError) else type(error).__name__)
            files.append(entry)
    finally:
        sftp.close()
    return dict(source_keys_sha256=sha(key_bytes), expected_new_remote_keys=len(wanted),
        expected_member_source_files=len(by_file), source_files=files,
        materialized_member_keys=materialized, materialized_key_count=len(materialized),
        materialized_source_files=sum(row["status"] in {"EXACT_PRIVATE_BYTE_MIRROR_CREATED", "EXISTING_EXACT_PRIVATE_MIRROR_REUSED"} for row in files),
        source_changed_files=sum(row["status"] == "SOURCE_CHANGED_SINCE_POSTCHECK_NO_MIRROR_NO_KEY_ASSUMPTION" for row in files),
        all_expected_keys_materialized=len(materialized) == len(wanted),
        historical_prompt_tokens_materialized=sum(row["prompt_tokens"] for row in materialized),
        historical_completion_tokens_materialized=sum(row["completion_tokens"] for row in materialized),
        new_inference_token_cost=0, fresh_gold_or_reward_value_decodes=0, sealed_paths_opened=0,
        remote_commands_started=0, remote_writes=0, generation_jobs=0, scoring_jobs=0,
        original_local_member_cache_files_modified=0)


def run(args, provider):
    if args.output.exists() or args.keys_output.exists():
        raise FileExistsError("Additive materialization receipts refuse overwrite")
    started, transport = time.perf_counter(), None
    result = dict(schema="E18_existing_member_cache_materialization_v4", authenticated=False,
        host_hash=sha(args.host.encode()), expected_pin=EXPECTED_PIN,
        reported_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha(Path(__file__).read_bytes()),
        fresh_gold_or_reward_value_decodes=0, sealed_paths_opened=0,
        new_generation_jobs=0, new_scoring_jobs=0)
    try:
        transport, pin = pinned_transport(args.host, args.port)
        result["observed_pin"] = pin
        password = provider()
        try:
            transport.auth_password(args.user, password, fallback=False)
        finally:
            password = None
        if not transport.is_authenticated():
            raise paramiko.AuthenticationException("Authentication was not accepted")
        result.update(authenticated=True, status="AUTHENTICATED_EXISTING_MEMBER_CACHE_MATERIALIZATION")
        result["materialization"] = mirror(transport, args.local_cache_keys, args.mirror_root)
    except Exception as error:
        result.update(status="MEMBER_CACHE_MATERIALIZATION_FAILED", error_type=type(error).__name__)
    finally:
        if transport is not None:
            transport.close()
    result["elapsed_s"] = time.perf_counter() - started
    save(args.output, result)
    materialized = result.get("materialization", {}).get("materialized_member_keys", [])
    save(args.keys_output, dict(schema="E18_local_private_member_mirror_keys_v4",
        materialization_receipt_sha256=sha(args.output.read_bytes()), keys=materialized))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=10021)
    parser.add_argument("--user", default="chen")
    parser.add_argument("--local-cache-keys", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keys-output", type=Path, required=True)
    parser.add_argument("--password-stdin", action="store_true")
    args = parser.parse_args()
    provider = lambda: getpass.getpass("Existing MAB SSH authentication: ")
    if args.password_stdin:
        provider = lambda: sys.stdin.readline().rstrip("\r\n")
    report = run(args, provider)
    print(json.dumps(dict(status=report["status"], authenticated=report["authenticated"],
        elapsed_s=report["elapsed_s"],
        materialized_keys=report.get("materialization", {}).get("materialized_key_count", 0),
        source_changed_files=report.get("materialization", {}).get("source_changed_files", 0))))
