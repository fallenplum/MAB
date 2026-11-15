"""Pinned-host, password-in-memory, read-only E18 member-cache inventory.

The endpoint is explicit. Host-key verification precedes credential collection
and authentication. The only remote interface is SFTP list/stat/read. No shell,
process, inference, scoring or write operation is exposed by this module.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import datetime
import getpass
import hashlib
import json
from pathlib import Path
import socket
import stat
import sys
import time

import paramiko

from cache_audit import FORBIDDEN_KEYS, ROW_FIELDS, RESULT_FIELDS, digest, json_property_names


EXPECTED_PIN = "SHA256:js0KwclB1fmgjvrw/R8vkXitaQEl/xQAfLM2Yck+pzs"
ALIASES = (
    "qwen3-8b", "qwen2.5-coder-7b-instruct", "granite-3.3-8b-instruct",
    "deephermes-3-llama-3-8b-preview", "deepseek-r1-distill-qwen-7b",
    "cogito-v1-preview-llama-8b",
)
REMOTE_ROOT = "/data1/data2/chendong/routing"
MAIN_WORKSPACE = "routing_v2_stage2_native_members_20261104_r4"
PROTECTED_TERMS = ("sealed", "gold", "reference", "reward", "score", "credential", "secret")


def pinned_transport(host, port=10021):
    connection = socket.create_connection((host, port), timeout=10)
    transport = paramiko.Transport(connection)
    try:
        transport.get_security_options().key_types = ("ssh-ed25519",)
        transport.start_client(timeout=10)
        key = transport.get_remote_server_key()
        fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
        if key.get_name() != "ssh-ed25519" or fingerprint != EXPECTED_PIN:
            raise ValueError("Remote host pin differs from the authenticated historical server")
        return transport, fingerprint
    except BaseException:
        transport.close()
        raise


def _allowed_path(path):
    if not path.startswith(REMOTE_ROOT + "/"):
        raise ValueError("Remote path is outside the frozen project root")
    if any(term in part.casefold() for part in path.split("/") for term in PROTECTED_TERMS):
        raise ValueError("Protected remote path rejected")


def _read_bytes(sftp, path, maximum=32 * 1024 * 1024):
    _allowed_path(path)
    info = sftp.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise ValueError("Only bounded regular unscored-output files can be read")
    with sftp.open(path, "rb") as stream:
        raw = stream.read(maximum + 1)
    after = sftp.lstat(path)
    if len(raw) != info.st_size or len(raw) > maximum or after.st_size != info.st_size or after.st_mtime != info.st_mtime:
        raise ValueError("Remote file changed or exceeded the read bound")
    return raw


def _member_rows(raw, expected, identities, member, seed):
    keys = json_property_names(raw)
    if any(key.casefold() in FORBIDDEN_KEYS for key in keys):
        return None, "PROTECTED_SCHEMA_VALUE_DECODE_REFUSED"
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    verified = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("result"), dict):
            return None, "INCOMPLETE_OR_UNRECOGNIZED_MEMBER_RESULT"
        if set(row) != ROW_FIELDS or set(row["result"]) != RESULT_FIELDS:
            return None, "UNRECOGNIZED_MEMBER_SCHEMA"
        if row["status"] != "success" or row["cost_status"] != "recorded_actual_usage":
            return None, "INCOMPLETE_OR_UNSETTLED_MEMBER_PHASE"
        body = {key: value for key, value in row.items() if key != "record_sha256"}
        if digest(body) != row["record_sha256"]:
            return None, "RECORD_HASH_MISMATCH"
        result = row["result"]
        if row["question_id"] not in identities or row["split"] != "fresh_dev_test":
            return None, "UNEXPECTED_QUESTION_IDENTITY"
        checks = (
            row["seed"] == seed, result["seed"] == seed,
            row["model_fingerprint"] == expected["model_fingerprint"],
            result["model_id"] == expected["model_id"],
            result["model_revision"] == expected["model_revision"],
            row["prompt_version"] == expected["member_prompt_version"],
            row["member_mode"] == expected["member_mode"], row["sample_index"] == 0,
            row["sampling_control"] is False, result["model_identity_verified"] is True,
            row["scoring_isolated_outside_generation_runner"] is True,
        )
        if not all(checks):
            return None, "FROZEN_GENERATION_IDENTITY_MISMATCH"
        if any(isinstance(result[name], bool) or not isinstance(result[name], int) or result[name] < 0
               for name in ("prompt_tokens", "completion_tokens")):
            return None, "UNKNOWN_EXECUTION_TOKEN_COST"
        if not isinstance(result["text"], str) or not result["text"].strip():
            return None, "EMPTY_OUTPUT"
        verified.append(dict(question_id=row["question_id"], canonical_id=identities[row["question_id"]],
            member_index=member, generation_seed=seed, physical_attempt_id=row["attempt_id"],
            job_id=row["job_id"], record_sha256=row["record_sha256"],
            source_question_sha256=row["source_question_sha256"],
            answer_sha256=hashlib.sha256(result["text"].encode()).hexdigest(),
            model_fingerprint=row["model_fingerprint"], model_id=result["model_id"],
            model_revision=result["model_revision"], member_prompt_version=row["prompt_version"],
            member_mode=row["member_mode"], sample_index=0, sampling_control=False,
            prompt_tokens=result["prompt_tokens"], completion_tokens=result["completion_tokens"],
            tokens=result["prompt_tokens"] + result["completion_tokens"],
            source_model_wall_s=result["wall_time_s"]))
    if len({row["canonical_id"] for row in verified}) != len(verified):
        return None, "DUPLICATE_LOGICAL_OUTPUT"
    return verified, "COMPLETE400" if len(verified) == 400 else "PARTIAL_VERIFIED_OUTPUTS"


def inventory(transport, local_cache_path):
    local = json.loads(Path(local_cache_path).read_text(encoding="utf-8"))["keys"]
    expected = {row["member_index"]: row for row in local}
    identities = {row["question_id"]: row["canonical_id"] for row in local}
    local_by_key = {(row["canonical_id"], row["member_index"], row["generation_seed"]): row for row in local}
    sftp = paramiko.SFTPClient.from_transport(transport)
    records, verified, metadata = [], [], []
    try:
        for member, alias in enumerate(ALIASES):
            base = f"{REMOTE_ROOT}/parallel_workers_20261103/{alias}/artifacts/private/"
            workspaces = [base + MAIN_WORKSPACE]
            if member == 4:
                workspaces += [base + MAIN_WORKSPACE + suffix for suffix in ("_variant_fg137", "_variant_fg2024")]
            for workspace in workspaces:
                try:
                    attrs = sftp.listdir_attr(workspace)
                    metadata.append(dict(workspace=workspace, entries=[dict(name=x.filename, bytes=x.st_size,
                        kind="directory" if stat.S_ISDIR(x.st_mode) else "file", mtime=x.st_mtime)
                        for x in attrs if not any(term in x.filename.casefold() for term in PROTECTED_TERMS)]))
                except OSError as error:
                    metadata.append(dict(workspace=workspace, state="UNAVAILABLE", error_type=type(error).__name__))
                    continue
                for seed in (42, 137, 2024):
                    path = f"{workspace}/fresh_g{seed}.jsonl"
                    row = dict(member_index=member, generation_seed=seed, path=path)
                    try:
                        raw = _read_bytes(sftp, path)
                    except OSError as error:
                        state = "NO_FILE" if error.errno == 2 else "REMOTE_FILE_UNAVAILABLE"
                        records.append(dict(**row, state=state, error_type=type(error).__name__))
                        continue
                    except ValueError as error:
                        records.append(dict(**row, state="UNSAFE_FILE_REFUSED", error_type=type(error).__name__))
                        continue
                    try:
                        outputs, state = _member_rows(raw, expected[member], identities, member, seed)
                    except (ValueError, TypeError, KeyError):
                        outputs, state = None, "UNSETTLED_OR_UNRECOGNIZED_JSONL"
                    summary = dict(**row, state=state, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                    if outputs is not None:
                        for output in outputs:
                            output["source_path"] = path
                            output["source_sha256"] = summary["sha256"]
                        summary.update(successful_outputs=len(outputs), prompt_tokens=sum(v["prompt_tokens"] for v in outputs),
                            completion_tokens=sum(v["completion_tokens"] for v in outputs),
                            matches_local_record_count=sum(
                                local_by_key.get((v["canonical_id"], member, seed), {}).get("record_sha256") == v["record_sha256"]
                                for v in outputs))
                        verified.extend(outputs)
                    records.append(summary)
        # Top-level project metadata can locate aggregate-output directories.
        # No aggregate payload is opened without a separately reviewed schema.
        for alias_root in (REMOTE_ROOT, REMOTE_ROOT + "/parallel_workers_20261103"):
            try:
                attrs = sftp.listdir_attr(alias_root)
                metadata.append(dict(project_root=alias_root, entries=[dict(name=x.filename, bytes=x.st_size,
                    kind="directory" if stat.S_ISDIR(x.st_mode) else "file", mtime=x.st_mtime)
                    for x in attrs if not any(term in x.filename.casefold() for term in PROTECTED_TERMS)]))
            except OSError as error:
                metadata.append(dict(project_root=alias_root, state="UNAVAILABLE", error_type=type(error).__name__))
    finally:
        sftp.close()
    candidates = {}
    physical = {}
    physical_conflicts = []
    for row in verified:
        key = (row["canonical_id"], row["member_index"], row["generation_seed"])
        candidates.setdefault(key, []).append(row)
        attempt = row["physical_attempt_id"]
        if attempt in physical and physical[attempt]["record_sha256"] != row["record_sha256"]:
            physical_conflicts.append(attempt)
        else:
            physical[attempt] = row
    unique = {}
    conflicts = []
    for key, rows in candidates.items():
        declared = local_by_key.get(key, {}).get("record_sha256")
        matching = [row for row in rows if row["record_sha256"] == declared]
        if matching:
            unique[key] = matching[0]
        elif len({row["record_sha256"] for row in rows}) == 1:
            unique[key] = rows[0]
        else:
            conflicts.append(key)
    counts = Counter((row["member_index"], row["generation_seed"]) for row in unique.values())
    return dict(member_files=records, directory_metadata=metadata,
        unique_member_cache_keys=list(unique.values()),
        physical_execution_ledger=list(physical.values()),
        physical_attempt_conflicts=physical_conflicts,
        conflicting_logical_keys=[list(key) for key in conflicts],
        coverage=[dict(member_index=m, generation_seed=g, verified_outputs=counts[m, g])
                  for m in range(6) for g in (42, 137, 2024)],
        aggregate_payloads_opened=0, fresh_gold_or_reward_reads=0, sealed_paths_opened=0,
        remote_commands_started=0, remote_writes=0)


def run(host, output, local_cache_path, password_provider, user="chen", port=10021):
    """Call a memory-only provider only after the reviewed host pin matches."""
    started = time.perf_counter()
    transport = None
    result = dict(schema="E18_remote_cache_inventory_new_endpoint_v2", host=host, port=port,
        host_hash=hashlib.sha256(host.encode()).hexdigest(), expected_pin=EXPECTED_PIN,
        reported_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authenticated=False, fresh_gold_or_reward_reads=0, sealed_paths_opened=0,
        new_generation_or_scoring_jobs=0)
    try:
        transport, pin = pinned_transport(host, port)
        result["observed_pin"] = pin
        password = password_provider()
        try:
            transport.auth_password(user, password, fallback=False)
        finally:
            password = None
        if not transport.is_authenticated():
            raise paramiko.AuthenticationException("Existing credential was not accepted")
        result.update(authenticated=True, status="AUTHENTICATED_READ_ONLY_INVENTORY")
        result["inventory"] = inventory(transport, local_cache_path)
    except Exception as error:
        result.update(status="READ_ONLY_INVENTORY_FAILED", error_type=type(error).__name__)
    finally:
        if transport is not None:
            transport.close()
    result["elapsed_s"] = time.perf_counter() - started
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-cache-keys", type=Path, required=True)
    parser.add_argument("--password-stdin", action="store_true")
    arguments = parser.parse_args()
    provider = lambda: getpass.getpass("Existing MAB SSH authentication: ")
    if arguments.password_stdin:
        provider = lambda: sys.stdin.readline().rstrip("\r\n")
    result = run(arguments.host, arguments.output, arguments.local_cache_keys, provider)
    print(json.dumps(dict(status=result["status"], authenticated=result["authenticated"],
                          host_hash=result["host_hash"], elapsed_s=result["elapsed_s"])))
