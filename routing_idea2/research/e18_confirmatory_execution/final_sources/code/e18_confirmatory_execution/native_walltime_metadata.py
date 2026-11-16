"""Extract private recorded call-wall metadata from closed native archives.

Only whitelisted JSON metadata strings/scalars are decoded. Generated text and
other unselected string values are structurally skipped as bytes. Upstream
native integrity audits remain responsible for canonical record-body validity.
Archive extraction, inference, scoring and scheduling are outside this module.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
import time


NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
HEX_SHA = re.compile(r"[0-9a-f]{64}")
MAX_ENTRY_BYTES = 128 * 1024 * 1024
FIELDS = {
    "native": {"attempt_id", "record_sha256", "status", "cost_status"},
    "result": {"wall_time_s", "backend"},
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


class MetadataProjection:
    """Byte parser that never JSON-decodes unselected answer string values."""

    def __init__(self, raw):
        self.raw = raw
        self.position = 0

    def whitespace(self):
        while self.position < len(self.raw) and self.raw[self.position] in b" \r\n\t":
            self.position += 1

    def expect(self, token):
        self.whitespace()
        if not self.raw.startswith(token, self.position):
            raise ValueError("Malformed native JSON metadata structure")
        self.position += len(token)

    def string_end(self):
        start = self.position
        self.expect(b'"')
        while self.position < len(self.raw):
            value = self.raw[self.position]
            self.position += 1
            if value == 34:
                return start, self.position
            if value < 32:
                raise ValueError("Unescaped control byte in native JSON string")
            if value == 92:
                if self.position >= len(self.raw):
                    raise ValueError("Truncated native JSON escape")
                escaped = self.raw[self.position]
                self.position += 1
                if escaped == 117:
                    digits = self.raw[self.position : self.position + 4]
                    if len(digits) != 4 or any(
                        ch not in b"0123456789abcdefABCDEF" for ch in digits
                    ):
                        raise ValueError("Malformed native JSON Unicode escape")
                    self.position += 4
                elif escaped not in b'"\\/bfnrt':
                    raise ValueError("Malformed native JSON escape")
        raise ValueError("Unterminated native JSON string")

    def skip_value(self, depth=0):
        if depth > 64:
            raise ValueError("Native JSON exceeds projection nesting limit")
        self.whitespace()
        if self.position >= len(self.raw):
            raise ValueError("Truncated native JSON value")
        value = self.raw[self.position]
        if value == 34:
            self.string_end()
            return
        if value in (123, 91):
            closing = b"}" if value == 123 else b"]"
            self.position += 1
            self.whitespace()
            if self.raw.startswith(closing, self.position):
                self.position += 1
                return
            while True:
                if value == 123:
                    self.whitespace()
                    self.string_end()
                    self.expect(b":")
                self.skip_value(depth + 1)
                self.whitespace()
                if self.raw.startswith(closing, self.position):
                    self.position += 1
                    return
                self.expect(b",")
        for literal in (b"true", b"false", b"null"):
            if self.raw.startswith(literal, self.position):
                self.position += len(literal)
                return
        match = NUMBER.match(self.raw, self.position)
        if match is None:
            raise ValueError("Malformed native JSON scalar")
        self.position = match.end()

    def object(self, kind):
        self.expect(b"{")
        result, seen = {}, set()
        self.whitespace()
        if self.raw.startswith(b"}", self.position):
            self.position += 1
            return result
        while True:
            self.whitespace()
            start, end = self.string_end()
            key = json.loads(self.raw[start:end])
            if key in seen:
                raise ValueError("Duplicate native top/result JSON key")
            seen.add(key)
            self.expect(b":")
            self.whitespace()
            if kind == "native" and key == "result" and self.raw.startswith(b"{", self.position):
                result["result"] = self.object("result")
            elif key in FIELDS[kind]:
                start = self.position
                if self.raw[self.position : self.position + 1] in (b"{", b"["):
                    raise ValueError("Selected native metadata must be scalar")
                self.skip_value()
                result[key] = json.loads(self.raw[start : self.position])
            else:
                self.skip_value()
            self.whitespace()
            if self.raw.startswith(b"}", self.position):
                self.position += 1
                return result
            self.expect(b",")

    def parse(self):
        result = self.object("native")
        self.whitespace()
        if self.position != len(self.raw):
            raise ValueError("Trailing data after native JSON record")
        return result


def project_native(raw):
    projected = MetadataProjection(raw).parse()
    attempt, checksum = projected.get("attempt_id"), projected.get("record_sha256")
    if (
        not isinstance(attempt, str)
        or not attempt
        or not isinstance(checksum, str)
        or HEX_SHA.fullmatch(checksum) is None
    ):
        raise ValueError("Explicit native attempt and record SHA identities required")
    wall = projected.get("result", {}).get("wall_time_s")
    if wall is not None and (type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0):
        raise ValueError("Recorded native call wall must be finite nonnegative seconds")
    return dict(
        physical_attempt_id=attempt,
        native_record_sha256=checksum,
        wall_time_s=wall,
        measurement_status="RECORDED" if wall is not None else "UNMEASURED_MISSING_FIELD",
        status=projected.get("status"),
        backend=projected.get("result", {}).get("backend"),
    )


def ledger_from_metadata(records):
    attempts, by_sha, occurrences = {}, {}, 0
    for record in records:
        occurrences += 1
        attempt = record["physical_attempt_id"]
        identity = (
            record["native_record_sha256"],
            record["wall_time_s"],
            record["measurement_status"],
        )
        if attempt in attempts and attempts[attempt][0] != identity:
            raise ValueError(
                "One physical attempt has conflicting native identity or wall metadata"
            )
        previous_attempt = by_sha.get(record["native_record_sha256"])
        if previous_attempt is not None and previous_attempt != attempt:
            raise ValueError("One native record SHA has multiple physical attempts")
        by_sha[record["native_record_sha256"]] = attempt
        attempts[attempt] = (identity, record)
    measured, missing = [], []
    for attempt in sorted(attempts):
        record = attempts[attempt][1]
        row = {
            key: record[key]
            for key in ("physical_attempt_id", "native_record_sha256", "wall_time_s")
        }
        if record["wall_time_s"] is None:
            missing.append(
                {key: row[key] for key in ("physical_attempt_id", "native_record_sha256")}
            )
        else:
            measured.append(row)
    return dict(
        schema="e18_native_call_walltime_metadata_v1",
        rows=measured,
        missing_measurements=missing,
        measurement_status="RECORDED_SUBSET" if measured else "UNMEASURED",
        summary=dict(
            native_record_occurrences=occurrences,
            unique_physical_attempts=len(attempts),
            duplicate_occurrences_removed=occurrences - len(attempts),
            measured_attempts=len(measured),
            unmeasured_attempts=len(missing),
            recorded_call_wall_seconds=math.fsum(row["wall_time_s"] for row in measured)
            if measured
            else None,
        ),
        measurement_field="result.wall_time_s",
        measurement_unit="seconds",
        boundary=(
            "Recorded generation/request call wall only. Parallel call-wall sums are not exclusive "
            "GPU kernel time, scheduler duration or model-loading time. Missing values stay unmeasured."
        ),
        generated_text_values_decoded=0,
        canonical_record_body_revalidated=False,
        private_artifact=True,
    )


def read_archive_metadata(path, expected):
    if path.stat().st_size != expected["bytes"] or file_sha(path) != expected["sha256"]:
        raise ValueError("Closed native archive external bytes differ")
    records, file_records, seen, manifest = [], {}, set(), None
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if manifest is not None:
                raise ValueError("Archive contains entries after its closing manifest")
            key = PurePosixPath(member.name)
            if not member.isfile() or key.is_absolute() or ".." in key.parts or member.name in seen:
                raise ValueError("Unsafe or repeated closed-native archive entry")
            seen.add(member.name)
            if member.size > MAX_ENTRY_BYTES:
                raise ValueError("Closed-native entry exceeds extraction metadata bound")
            if member.name == "EXPORT_MANIFEST.json":
                with archive.extractfile(member) as stream:
                    manifest = json.load(stream)
                continue
            if key.name != "records.jsonl":
                continue
            digest, size = hashlib.sha256(), 0
            with archive.extractfile(member) as stream:
                for line in stream:
                    digest.update(line)
                    size += len(line)
                    if line.strip():
                        records.append(project_native(line))
            file_records[member.name] = dict(bytes=size, sha256=digest.hexdigest())
    if manifest is None or manifest.get("schema") != "e18_closed_native_export_manifest_v1":
        raise ValueError("A closed native export manifest is required")
    if canonical_digest(manifest) != expected["manifest_sha256"]:
        raise ValueError("Closed native metadata manifest digest differs")
    declared = {row["path"]: row for row in manifest["files"]}
    for name, actual in file_records.items():
        if any(declared.get(name, {}).get(key) != value for key, value in actual.items()):
            raise ValueError("Native record file SHA or size differs from its closed manifest")
    observed = Counter((row["physical_attempt_id"], row["native_record_sha256"]) for row in records)
    required = Counter(
        (row["physical_attempt_id"], row["native_record_sha256"]) for row in manifest["tasks"]
    )
    if observed != required or len(records) != expected["tasks"]:
        raise ValueError(
            "Projected native attempt/record identities differ from the closed task inventory"
        )
    return records


def inspect_one(path, expected_sha, expected_bytes):
    if file_sha(path) != expected_sha or path.stat().st_size != expected_bytes:
        raise ValueError("Single-record inspection archive bytes differ")
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if PurePosixPath(member.name).name != "records.jsonl":
                continue
            with archive.extractfile(member) as stream:
                for line in stream:
                    if line.strip():
                        metadata = project_native(line)
                        return dict(
                            schema="e18_native_walltime_single_record_inspection_v1",
                            metadata=metadata,
                            native_records_projected=1,
                            generated_text_values_decoded=0,
                            archive_members_extracted=0,
                            wall_field_present=metadata["wall_time_s"] is not None,
                            field="result.wall_time_s",
                            unit="seconds",
                            archive_sha256=expected_sha,
                        )
    raise ValueError("No native records.jsonl found in the selected archive")


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    single = modes.add_parser("inspect-one")
    single.add_argument("--archive", type=Path, required=True)
    single.add_argument("--archive-sha256", required=True)
    single.add_argument("--archive-bytes", type=int, required=True)
    single.add_argument("--output", type=Path, required=True)
    extract = modes.add_parser("extract")
    extract.add_argument("--source-inventory", type=Path, required=True)
    extract.add_argument("--source-inventory-sha256", required=True)
    extract.add_argument("--archive-directory", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Private wall-time ledgers are append-only")
    start = time.perf_counter()
    if args.mode == "inspect-one":
        result = inspect_one(args.archive, args.archive_sha256, args.archive_bytes)
        summary = dict(
            native_records_projected=1,
            wall_field_present=result["wall_field_present"],
            wall_time_s=result["metadata"]["wall_time_s"],
            field=result["field"],
            unit=result["unit"],
        )
    else:
        if file_sha(args.source_inventory) != args.source_inventory_sha256:
            raise ValueError("Closed-export source inventory bytes differ")
        source = json.loads(args.source_inventory.read_text(encoding="utf-8-sig"))
        records = []
        for entry in source["new_archives"]:
            path = args.archive_directory / PurePosixPath(entry["path"]).name
            records.extend(read_archive_metadata(path, entry))
        result = ledger_from_metadata(records)
        result["source_inventory_sha256"] = args.source_inventory_sha256
        result["archives_checked"] = len(source["new_archives"])
        summary = result["summary"]
    result["metadata_tool_source_sha256"] = file_sha(__file__)
    result["extraction_wall_seconds"] = time.perf_counter() - start
    result["new_model_calls"] = 0
    result["gold_or_scoring_reads"] = 0
    write_new_json(args.output, result)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
