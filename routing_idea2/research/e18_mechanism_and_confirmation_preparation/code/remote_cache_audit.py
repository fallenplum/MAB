"""Read-only E18 SSH handshake and directory-metadata audit.

The default mode prints a local review plan.  Probe opens an SSH connection
without requesting a remote shell or command.  Inventory uses only directory
listing and lstat, and requires explicit remote roots.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


CONNECTION_HELPER = Path(
    r"E:\语言模型组合路由\artifacts\private\server_migration_20261102"
    r"\ssh_monitor_transport_verified_newhost_20261104.py"
)
KNOWN_HOSTS = Path(r"C:\Users\23301\.ssh\known_hosts")
EXPECTED_PIN = "SHA256:js0KwclB1fmgjvrw/R8vkXitaQEl/xQAfLM2Yck+pzs"
EXPECTED_ALGORITHM = "ssh-ed25519"

REMOTE_INVENTORY = r'''
import json, os, stat, sys
from pathlib import PurePosixPath

request = json.loads(sys.argv[1])
blocked_terms = ("sealed", "gold", "reference", "reward", "score", "credential", "secret")
results = []
for root in request["roots"]:
    parts = PurePosixPath(root).parts
    if not root.startswith("/") or any(term in part.lower() for part in parts for term in blocked_terms):
        raise ValueError("Remote root is outside the requested metadata scope")
    record = {"root": root, "entries": [], "skipped_protected": 0, "symlinks": 0,
              "errors": 0, "truncated": False}
    pending = [(root, 0)]
    while pending and len(record["entries"]) < request["max_entries"]:
        directory, depth = pending.pop()
        try:
            directory_mode = os.lstat(directory).st_mode
            if stat.S_ISLNK(directory_mode):
                record["symlinks"] += 1
                continue
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    if len(record["entries"]) >= request["max_entries"]:
                        record["truncated"] = True
                        break
                    if any(term in entry.name.lower() for term in blocked_terms):
                        record["skipped_protected"] += 1
                        continue
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        record["symlinks"] += 1
                        continue
                    is_directory = stat.S_ISDIR(info.st_mode)
                    record["entries"].append({
                        "path": os.path.relpath(entry.path, root),
                        "kind": "directory" if is_directory else "file",
                        "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
                    })
                    if is_directory and depth < request["max_depth"]:
                        pending.append((entry.path, depth + 1))
        except OSError:
            record["errors"] += 1
    record["truncated"] = record["truncated"] or bool(pending)
    results.append(record)
print(json.dumps({"schema": "E18_remote_directory_metadata_v1", "results": results,
                  "file_contents_opened": 0}, sort_keys=True))
'''


def emit(value):
    print(json.dumps(value, ensure_ascii=True, sort_keys=True))


def connection_metadata():
    source = CONNECTION_HELPER.read_bytes()
    tree = ast.parse(source.decode("utf-8-sig"))
    selected = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"HOST", "PORT", "USER"}:
                selected[target.id] = node.value.value
    if set(selected) != {"HOST", "PORT", "USER"}:
        raise ValueError("SSH helper endpoint metadata is incomplete")
    host, port, user = selected["HOST"], selected["PORT"], selected["USER"]
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9.-]+", host):
        raise ValueError("SSH host metadata is invalid")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("SSH port metadata is invalid")
    if not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", user):
        raise ValueError("SSH user metadata is invalid")
    endpoint = f"[{host}]:{port}"
    pins = []
    for line in KNOWN_HOSTS.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 3 or endpoint not in fields[0].split(","):
            continue
        if fields[1] == EXPECTED_ALGORITHM:
            digest = hashlib.sha256(base64.b64decode(fields[2])).digest()
            pins.append("SHA256:" + base64.b64encode(digest).decode().rstrip("="))
    if pins != [EXPECTED_PIN]:
        raise ValueError("Saved endpoint pin differs from the reviewed E18 pin")
    return dict(host=host, port=port, user=user, host_key_algorithm=EXPECTED_ALGORITHM,
                expected_pin=EXPECTED_PIN, helper=str(CONNECTION_HELPER),
                helper_sha256=hashlib.sha256(source).hexdigest(), known_hosts=str(KNOWN_HOSTS))


def ssh_arguments(metadata, timeout, identity_file):
    executable = shutil.which("ssh")
    if executable is None:
        raise RuntimeError("OpenSSH executable is unavailable")
    arguments = [executable, "-F", os.devnull, "-T", "-p", str(metadata["port"])]
    options = {
        "BatchMode": "yes", "StrictHostKeyChecking": "yes",
        "UserKnownHostsFile": KNOWN_HOSTS.as_posix(),
        "GlobalKnownHostsFile": os.devnull, "HostKeyAlgorithms": EXPECTED_ALGORITHM,
        "PasswordAuthentication": "no", "KbdInteractiveAuthentication": "no",
        "ConnectionAttempts": "1", "ConnectTimeout": str(timeout),
    }
    for key, value in options.items():
        arguments.extend(["-o", f"{key}={value}"])
    if identity_file:
        arguments.extend(["-i", str(Path(identity_file)), "-o", "IdentitiesOnly=yes"])
    arguments.append(f"{metadata['user']}@{metadata['host']}")
    return arguments


def classify_failure(stderr):
    value = stderr.lower()
    if "host key verification failed" in value or "remote host identification has changed" in value:
        return "SSH_HOST_KEY_REJECTED"
    if "permission denied" in value or "no more authentication methods" in value:
        return "SSH_AUTHENTICATION_FAILED"
    if "connection refused" in value:
        return "SSH_CONNECTION_REFUSED"
    if "timed out" in value or "operation timed out" in value:
        return "SSH_CONNECT_TIMEOUT"
    if "could not resolve hostname" in value:
        return "SSH_NAME_RESOLUTION_FAILED"
    if "no route to host" in value or "network is unreachable" in value:
        return "SSH_NETWORK_UNREACHABLE"
    return "SSH_FAILED"


def run_ssh(arguments, timeout, probe=False):
    try:
        completed = subprocess.run(arguments, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace",
                                   timeout=timeout, check=False)
        stdout, stderr = completed.stdout, completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        returncode = None
    match = re.search(r"Server host key: (\S+) (SHA256:[A-Za-z0-9+/]+)", stderr)
    observed = dict(algorithm=match[1], fingerprint=match[2]) if match else None
    if observed and observed != {"algorithm": EXPECTED_ALGORITHM, "fingerprint": EXPECTED_PIN}:
        return dict(status="SSH_PIN_MISMATCH", observed_host_key=observed)
    if probe:
        authenticated = "Authenticated to " in stderr
        status = "SSH_AUTHENTICATED" if authenticated else classify_failure(stderr)
        if not authenticated and returncode is None:
            status = "SSH_PROBE_TIMEOUT"
        return dict(status=status, observed_host_key=observed, authenticated=authenticated,
                    remote_shell_requested=False, remote_command_requested=False,
                    remote_data_reads=0, returncode=returncode)
    if returncode != 0:
        return dict(status=classify_failure(stderr), observed_host_key=observed, returncode=returncode)
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return dict(status="REMOTE_METADATA_PROTOCOL_ERROR", observed_host_key=observed)
    return dict(status="REMOTE_METADATA_COMPLETE", inventory=result, observed_host_key=observed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("plan", "probe", "inventory"), default="plan")
    parser.add_argument("--identity-file", help="Existing key path passed directly to OpenSSH")
    parser.add_argument("--remote-root", action="append", default=[])
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-entries", type=int, default=10000)
    parser.add_argument("--connect-timeout", type=int, default=8)
    parser.add_argument("--total-timeout", type=int, default=12)
    args = parser.parse_args()
    if min(args.max_depth, args.max_entries, args.connect_timeout, args.total_timeout) < 0:
        parser.error("Audit bounds must be nonnegative")
    metadata = connection_metadata()
    if args.mode == "plan":
        emit(dict(status="PLAN_ONLY", connection=metadata,
                  modes={"probe": "SSH -N handshake with no remote command",
                         "inventory": "explicit roots, scandir/lstat metadata only"}))
        return
    arguments = ssh_arguments(metadata, args.connect_timeout, args.identity_file)
    if args.mode == "probe":
        arguments[1:1] = ["-v", "-N"]
        result = run_ssh(arguments, args.total_timeout, probe=True)
    else:
        if not args.remote_root:
            parser.error("inventory requires at least one explicit --remote-root")
        request = json.dumps(dict(roots=args.remote_root, max_depth=args.max_depth,
                                  max_entries=args.max_entries))
        arguments.append(shlex.join(["python3", "-B", "-c", REMOTE_INVENTORY, request]))
        result = run_ssh(arguments, args.total_timeout)
    emit(dict(schema="E18_remote_cache_audit_v1", connection=metadata, result=result))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        emit(dict(status="LOCAL_PREFLIGHT_FAILED", error_type=type(error).__name__))
        sys.exit(1)
