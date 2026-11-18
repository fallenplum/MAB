#!/usr/bin/env python3
"""Apply the reviewed MAB branch-name migration; the default is read-only.

Usage from the repository root:
    python scripts/migrate_branch_names.py
    python scripts/migrate_branch_names.py --apply

Only GH_TOKEN or GITHUB_TOKEN supplies authentication, at process runtime.
The sole write operation is GitHub's documented branch-rename endpoint.
No redirects are followed; no response bodies or exception details are logged.

GitHub's rename API has no expected-SHA/CAS parameter. Immediate pre-write
checks and post-write verification detect a concurrent push, but cannot make
the GET-to-POST interval atomic. On any failure, stop and review; a later run
can resume from verified old-absent/new-same-SHA pairs. POST is never retried.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable, Sequence
import urllib.error
import urllib.parse
import urllib.request

REPOSITORY = "fallenplum/MAB"
API_ROOT = "https://api.github.com/repos/" + REPOSITORY
API_VERSION = "2026-03-10"
EXPECTED_COUNT = 31
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
REF_RE = re.compile(r"[A-Za-z0-9._/-]+\Z")
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "reproducibility" / "branch_migration.json"


class MigrationError(Exception):
    """Only a fixed local code and optional numeric HTTP status are printable."""

    def __init__(self, code: str, http_status: int | None = None) -> None:
        self.code = code
        self.http_status = http_status
        super().__init__(code)

    def as_record(self) -> dict:
        record = {"error": self.code}
        if self.http_status is not None:
            record["http_status"] = self.http_status
        return record


@dataclass(frozen=True)
class Entry:
    old: str
    new: str
    sha: str


@dataclass(frozen=True)
class Branch:
    name: str
    sha: str


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise MigrationError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _valid_ref(name: object) -> bool:
    if not isinstance(name, str) or not REF_RE.fullmatch(name):
        return False
    if len(name.encode("ascii")) > 255 or ".." in name or "//" in name:
        return False
    return all(
        part and not part.startswith(".") and not part.endswith((".", ".lock"))
        for part in name.split("/")
    )


def parse_manifest(value: object) -> tuple[Entry, ...]:
    if not isinstance(value, dict):
        raise MigrationError("INVALID_MANIFEST")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise MigrationError("INVALID_SCHEMA_VERSION")
    if value.get("repository") != REPOSITORY:
        raise MigrationError("REPOSITORY_NOT_ALLOWED")
    if value.get("operation") != "remove_codex_prefix_preserve_commit":
        raise MigrationError("OPERATION_NOT_ALLOWED")
    expected_count = value.get("expected_count", EXPECTED_COUNT)
    if type(expected_count) is not int or not 1 <= expected_count <= 100:
        raise MigrationError("INVALID_EXPECTED_COUNT")
    rows = value.get("branches")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise MigrationError("INVALID_BRANCH_COUNT")
    entries = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"old", "new", "sha"}:
            raise MigrationError("INVALID_BRANCH_ENTRY")
        old, new, sha = row["old"], row["new"], row["sha"]
        if not _valid_ref(old) or not _valid_ref(new):
            raise MigrationError("INVALID_BRANCH_NAME")
        if old == "main" or new == "main":
            raise MigrationError("MAIN_NOT_ALLOWED")
        if not old.startswith("codex/"):
            raise MigrationError("SOURCE_PREFIX_NOT_ALLOWED")
        if new != old[len("codex/"):] or "codex/" in new:
            raise MigrationError("DESTINATION_NOT_ALLOWED")
        if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
            raise MigrationError("INVALID_EXPECTED_SHA")
        entries.append(Entry(old, new, sha))
    old_names = {entry.old for entry in entries}
    new_names = {entry.new for entry in entries}
    if len(old_names) != len(entries) or len(new_names) != len(entries):
        raise MigrationError("DUPLICATE_BRANCH_MAPPING")
    if old_names & new_names:
        raise MigrationError("CHAINED_BRANCH_MAPPING")
    if any(a.startswith(b + "/") for a in new_names for b in new_names if a != b):
        raise MigrationError("DESTINATION_MAPPING_REF_CONFLICT")
    # main_before, main_backup_branch and unchanged_branches are metadata only.
    return tuple(entries)


def load_manifest(path: Path) -> tuple[Entry, ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise MigrationError("MANIFEST_READ_FAILED") from None
    return parse_manifest(value)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHubAPI:
    def __init__(self, token: str | None, *, allow_write: bool = False) -> None:
        self._token = token
        self._allow_write = allow_write
        self._opener = urllib.request.build_opener(NoRedirect())

    def _request(self, method: str, path: str, payload: dict | None = None):
        if method not in {"GET", "POST"}:
            raise MigrationError("HTTP_METHOD_NOT_ALLOWED")
        if method == "POST" and not self._allow_write:
            raise MigrationError("WRITE_DISABLED")
        if method == "POST" and not self._token:
            raise MigrationError("TOKEN_REQUIRED")
        allowed = (
            re.fullmatch(r"/branches\?per_page=100&page=[1-9][0-9]*", path)
            or re.fullmatch(r"/branches/[A-Za-z0-9._%\-]+", path)
        ) if method == "GET" else re.fullmatch(r"/branches/[A-Za-z0-9._%\-]+/rename", path)
        if not allowed:
            raise MigrationError("API_PATH_NOT_ALLOWED")
        if method == "GET" and payload is not None:
            raise MigrationError("API_PAYLOAD_NOT_ALLOWED")
        if path.startswith("/branches/"):
            encoded = path[len("/branches/"):]
            if method == "POST":
                encoded = encoded[:-len("/rename")]
            branch_name = urllib.parse.unquote(encoded)
            if not _valid_ref(branch_name) or urllib.parse.quote(branch_name, safe="") != encoded:
                raise MigrationError("API_PATH_NOT_ALLOWED")
            if method == "POST":
                if (not isinstance(payload, dict) or set(payload) != {"new_name"}
                        or not branch_name.startswith("codex/")):
                    raise MigrationError("API_MUTATION_NOT_ALLOWED")
                destination = payload["new_name"]
                if (not _valid_ref(destination) or destination == "main"
                        or destination != branch_name[len("codex/"):]
                        or "codex/" in destination):
                    raise MigrationError("API_MUTATION_NOT_ALLOWED")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "MAB-reviewed-branch-migration",
        }
        if self._token:
            headers["Authorization"] = "Bearer " + self._token
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(API_ROOT + path, data=data, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=30) as response:
                status = response.status
                if status != (201 if method == "POST" else 200):
                    raise MigrationError("HTTP_STATUS", status)
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            raise MigrationError("HTTP_STATUS", status) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise MigrationError("NETWORK_ERROR") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise MigrationError("API_RESPONSE_TOO_LARGE")
        try:
            return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        except (UnicodeError, json.JSONDecodeError):
            raise MigrationError("INVALID_API_JSON") from None

    @staticmethod
    def _branch(value: object) -> Branch:
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise MigrationError("INVALID_BRANCH_RESPONSE")
        name, commit = value["name"], value.get("commit")
        if not name or not isinstance(commit, dict):
            raise MigrationError("INVALID_BRANCH_RESPONSE")
        sha = commit.get("sha")
        if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
            raise MigrationError("INVALID_BRANCH_RESPONSE")
        return Branch(name, sha)

    def list_branches(self) -> dict[str, Branch]:
        result = {}
        # Numeric, same-origin pagination; never follow an arbitrary Link URL.
        for page in range(1, 1001):
            values = self._request("GET", f"/branches?per_page=100&page={page}")
            if not isinstance(values, list) or len(values) > 100:
                raise MigrationError("INVALID_BRANCH_LIST")
            for value in values:
                branch = self._branch(value)
                if branch.name in result:
                    raise MigrationError("UNSTABLE_BRANCH_PAGINATION")
                result[branch.name] = branch
            if len(values) < 100:
                return result
        raise MigrationError("BRANCH_PAGINATION_LIMIT")

    def get_branch(self, name: str) -> Branch | None:
        try:
            value = self._request("GET", "/branches/" + urllib.parse.quote(name, safe=""))
        except MigrationError as error:
            if error.http_status == 404:
                return None
            # A 301 is not interpreted as an existing branch or as a 404.
            raise
        return self._branch(value)

    def rename(self, entry: Entry) -> Branch:
        value = self._request(
            "POST",
            "/branches/" + urllib.parse.quote(entry.old, safe="") + "/rename",
            {"new_name": entry.new},
        )
        return self._branch(value)


def classify(entry: Entry, snapshot: dict[str, Branch]) -> str:
    old, new = snapshot.get(entry.old), snapshot.get(entry.new)
    if old is not None:
        if old.name != entry.old or old.sha != entry.sha:
            raise MigrationError("SOURCE_SHA_OR_NAME_CHANGED")
        if new is not None:
            # Both names present is a collision even when their SHAs are equal.
            raise MigrationError("DESTINATION_ALREADY_EXISTS")
        if any(name.startswith(entry.new + "/") or entry.new.startswith(name + "/")
               for name in snapshot):
            raise MigrationError("DESTINATION_REF_CONFLICT")
        return "pending"
    if new is None:
        raise MigrationError("SOURCE_AND_DESTINATION_MISSING")
    if new.name != entry.new or new.sha != entry.sha:
        raise MigrationError("RESUME_DESTINATION_CHANGED")
    return "already_completed"


def preflight(entries: Sequence[Entry], snapshot: dict[str, Branch]) -> list[str]:
    # Complete the entire pass before making any write.
    return [classify(entry, snapshot) for entry in entries]


def assert_branch(branch: Branch | None, name: str, sha: str) -> None:
    if branch is None:
        raise MigrationError("BRANCH_MISSING")
    if branch.name != name:
        raise MigrationError("BRANCH_NAME_MISMATCH")
    if branch.sha != sha:
        raise MigrationError("BRANCH_SHA_MISMATCH")


def verify_rename(api: GitHubAPI, entry: Entry, sleep: Callable[[float], None]) -> None:
    # GitHub can acknowledge a rename before background processing has finished.
    # Only read requests are retried; POST is never retried.
    for attempt in range(6):
        branch = api.get_branch(entry.new)
        if branch is not None:
            assert_branch(branch, entry.new, entry.sha)
        snapshot = api.list_branches()
        old, new = snapshot.get(entry.old), snapshot.get(entry.new)
        if old is not None and old.sha != entry.sha:
            raise MigrationError("SOURCE_CHANGED_AFTER_RENAME")
        if new is not None:
            assert_branch(new, entry.new, entry.sha)
        if branch is not None and new is not None and old is None:
            return
        if attempt < 5:
            sleep(min(2 ** attempt, 4))
    raise MigrationError("RENAME_VERIFICATION_INCOMPLETE")


def final_verification(entries: Sequence[Entry], snapshot: dict[str, Branch]) -> None:
    for entry in entries:
        if entry.old in snapshot:
            raise MigrationError("FINAL_SOURCE_STILL_PRESENT")
        assert_branch(snapshot.get(entry.new), entry.new, entry.sha)


def emit_record(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=True, sort_keys=True), flush=True)


def migrate(
    entries: Sequence[Entry],
    api: GitHubAPI,
    *,
    apply: bool = False,
    emit: Callable[[dict], None] = emit_record,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    states = preflight(entries, api.list_branches())
    emit({
        "mode": "apply" if apply else "dry-run",
        "repository": REPOSITORY,
        "entries": len(entries),
        "pending": states.count("pending"),
        "already_completed": states.count("already_completed"),
    })
    if not apply:
        for entry, state in zip(entries, states):
            emit({"status": state, "old": entry.old, "new": entry.new, "sha": entry.sha})
        return {"status": "dry-run", "renamed": 0, "entries": len(entries)}

    renamed = 0
    skipped = 0
    for index, entry in enumerate(entries):
        # Recheck all collisions/SHAs, including future items, before this write.
        current_states = preflight(entries, api.list_branches())
        if current_states[index] == "already_completed":
            assert_branch(api.get_branch(entry.new), entry.new, entry.sha)
            skipped += 1
            emit({"status": "already_completed", "old": entry.old, "new": entry.new, "sha": entry.sha})
            continue
        # Last API read before POST: exact current source name and source SHA.
        assert_branch(api.get_branch(entry.old), entry.old, entry.sha)
        renamed_branch = api.rename(entry)
        assert_branch(renamed_branch, entry.new, entry.sha)
        verify_rename(api, entry, sleep)
        renamed += 1
        emit({"status": "renamed_verified", "old": entry.old, "new": entry.new, "sha": entry.sha})

    # A listing reports real names; old-name HTTP redirects cannot satisfy this.
    final_verification(entries, api.list_branches())
    result = {"status": "verified", "renamed": renamed, "already_completed": skipped, "entries": len(entries)}
    emit(result)
    return result


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise MigrationError("INVALID_ARGUMENTS")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        parser = SafeArgumentParser(description=__doc__)
        parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument("--apply", action="store_true", dest="apply",
                          help="execute the branch renames in the selected manifest")
        mode.add_argument("--dry-run", action="store_false", dest="apply",
                          help="read and validate only (default)")
        parser.set_defaults(apply=False)
        args = parser.parse_args(argv)
        entries = load_manifest(args.manifest)
        token = os.environ.get("GH_TOKEN", "").strip() or os.environ.get("GITHUB_TOKEN", "").strip()
        if args.apply and not token:
            raise MigrationError("TOKEN_REQUIRED")
        api = GitHubAPI(token or None, allow_write=args.apply)
        migrate(entries, api, apply=args.apply)
        return 0
    except MigrationError as error:
        print(json.dumps(error.as_record(), sort_keys=True), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('{"error": "INTERRUPTED"}', file=sys.stderr)
        return 130
    except Exception:
        # Never print exceptions that may contain an HTTP body, headers or token.
        print('{"error": "UNEXPECTED_ERROR"}', file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

