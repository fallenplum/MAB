"""Verified, process-resumable HTTP Range downloads without HF UUID temp files.

The caller obtains expected_size and authoritative LFS SHA256 (or small Git blob
SHA1) from pinned official repository metadata. This module performs no metadata
inference. Stable paths are exactly str(dest)+'.part' and str(dest)+'.lock'.
Locks are O_EXCL and never auto-stolen; a killed owner's stale lock requires
explicit process-state reconciliation. Interruptions preserve and fsync all
received prefix bytes. Hash failures preserve the part for inspection.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib
import hmac
import json
import os
import re
import time
import uuid
import requests

RANGE_BYTES = 64 * 1024 * 1024
STREAM_BYTES = 64 * 1024


class DownloadError(RuntimeError):
    pass


class DownloadLockedError(DownloadError):
    pass


class RangeProtocolError(DownloadError):
    pass


class DownloadIntegrityError(DownloadError):
    pass


class DownloadInterruptedError(DownloadError):
    pass


def _expected_digest(value, length, name):
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{" + str(length) + r"}", value):
        raise ValueError(f"Invalid authoritative {name}")
    return value.lower()


@contextmanager
def _exclusive_lock(path, expected_size, expected_sha256, expected_git_sha1):
    try:
        descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise DownloadLockedError("Destination is locked; reconcile its owner before retrying") from None
    token = uuid.uuid4().hex
    try:
        metadata = {"pid": os.getpid(), "created_at_unix": time.time(), "owner_token": token,
                    "expected_size": expected_size, "sha256": expected_sha256,
                    "git_blob_sha1": expected_git_sha1}
        os.write(descriptor, json.dumps(metadata).encode("utf8"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            # Never unlink a replacement lock belonging to someone else.
            current = json.loads(path.read_text(encoding="utf8"))
            if current.get("owner_token") == token:
                path.unlink()
        except (FileNotFoundError, ValueError, OSError):
            pass


def _verified_hashes(path, expected_size, expected_sha256, expected_git_sha1):
    if path.stat().st_size != expected_size:
        raise DownloadIntegrityError("File size differs from authoritative metadata")
    sha = hashlib.sha256()
    git = hashlib.sha1()
    git.update(f"blob {expected_size}\0".encode("ascii"))
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
            git.update(block)
    actual_sha, actual_git = sha.hexdigest(), git.hexdigest()
    if expected_sha256 is not None and not hmac.compare_digest(actual_sha, expected_sha256):
        raise DownloadIntegrityError("Complete SHA256 verification failed; partial preserved")
    if expected_git_sha1 is not None and not hmac.compare_digest(actual_git, expected_git_sha1):
        raise DownloadIntegrityError("Complete Git blob SHA1 verification failed; partial preserved")
    return actual_sha, actual_git


def _response_length(response, expected):
    raw = response.headers.get("Content-Length")
    if raw is not None and (not raw.isdigit() or int(raw) != expected):
        raise RangeProtocolError("HTTP content length differs from the exact requested range")
    encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
    if encoding not in ("", "identity"):
        raise RangeProtocolError("Encoded range payload rejected; identity encoding required")


def resumable_download(url, dest, expected_size, sha256=None, git_blob_sha1=None,
                       session=None, retries=10, *, range_bytes=RANGE_BYTES,
                       timeout=(20, 120), progress_callback=None, retry_backoff=0.5):
    """Download one metadata-locked artifact and atomically publish after hashing.

    Healthy ranges do not consume retries. retries limits consecutive failures
    WITHOUT durable byte progress; productive interruptions reset that counter.
    retries=0 disables all retries, including productive interruptions, useful
    for explicit pause/resume. Every retry begins at the current .part size.

    A 200 response is accepted only for a fresh offset-zero request covering the
    WHOLE file (small Git blobs); it is written as a fresh file, never appended.
    A nonzero Range ignored with 200 always fails with the old partial intact.
    progress_callback(event_dict) receives bytes/total_bytes/resumed_bytes.
    Return includes the full actual sha256 even when Git SHA1 is authoritative.
    """
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        raise ValueError("expected_size must be a nonnegative metadata integer")
    if not isinstance(retries, int) or retries < 0 or not isinstance(range_bytes, int) or range_bytes <= 0:
        raise ValueError("Nonnegative retries and positive range_bytes required")
    if not isinstance(retry_backoff, (int, float)) or not 0 <= retry_backoff <= 30:
        raise ValueError("retry_backoff must be between zero and thirty seconds")
    expected_sha = _expected_digest(sha256, 64, "SHA256")
    expected_git = _expected_digest(git_blob_sha1, 40, "Git blob SHA1")
    if expected_sha is None and expected_git is None:
        raise ValueError("At least one authoritative complete-file digest is required")
    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(destination) + ".part")
    lock = Path(str(destination) + ".lock")
    stats = {"attempts": 0, "failures": 0, "resumed_bytes": 0, "downloaded_bytes": 0}

    def event(kind, size):
        if progress_callback is not None:
            progress_callback({"event": kind, "path": str(destination), "bytes": size,
                               "total_bytes": expected_size, **stats})

    def verified_result(path, cache_hit):
        actual_sha, actual_git = _verified_hashes(path, expected_size, expected_sha, expected_git)
        return {"status": "VERIFIED", "verified": True, "path": str(destination),
                "size": expected_size, "bytes": expected_size, "total_bytes": expected_size,
                "sha256": actual_sha, "git_blob_sha1": actual_git,
                "cache_hit": cache_hit, **stats}

    with _exclusive_lock(lock, expected_size, expected_sha, expected_git):
        if destination.exists():
            # Never replace an existing unverified destination.
            result = verified_result(destination, True)
            event("verified", expected_size)
            return result
        if partial.exists() and not partial.is_file():
            raise DownloadIntegrityError("Stable partial path is not a regular file")
        current = partial.stat().st_size if partial.exists() else 0
        if current > expected_size:
            raise DownloadIntegrityError("Partial exceeds metadata size; refusing append/truncation")
        stats["resumed_bytes"] = current
        owned_session = session is None
        client = requests.Session() if owned_session else session
        consecutive_failures = 0
        try:
            if not partial.exists():
                with partial.open("wb") as stream:
                    stream.flush()
                    os.fsync(stream.fileno())
            event("start", current)
            while current < expected_size:
                start = current
                end = min(start + range_bytes - 1, expected_size - 1)
                response = None
                stats["attempts"] += 1
                try:
                    response = client.get(url, headers={"Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity"}, stream=True, timeout=timeout, allow_redirects=True)
                    status = response.status_code
                    if status in (408, 425, 429, 500, 502, 503, 504):
                        raise DownloadInterruptedError(f"Retryable HTTP status {status}")
                    if status == 206:
                        header = response.headers.get("Content-Range", "").strip()
                        match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", header, re.IGNORECASE)
                        if match is None or tuple(map(int, match.groups())) != (start, end, expected_size):
                            raise RangeProtocolError("HTTP Content-Range start/end/total mismatch")
                        expected_response = end - start + 1
                    elif status == 200:
                        if start != 0 or end != expected_size - 1:
                            raise RangeProtocolError("Server ignored Range; existing partial is never appended or discarded")
                        expected_response = expected_size
                    else:
                        raise RangeProtocolError(f"Unexpected HTTP status {status}")
                    _response_length(response, expected_response)
                    received = 0
                    # Explicit seek, never append mode: even fresh HTTP 200 is not appended.
                    with partial.open("r+b") as stream:
                        stream.seek(start)
                        try:
                            for block in response.iter_content(chunk_size=STREAM_BYTES):
                                if not block:
                                    continue
                                if received + len(block) > expected_response or current + len(block) > expected_size:
                                    raise RangeProtocolError("HTTP body exceeds exact range or metadata size")
                                written = stream.write(block)
                                if written != len(block):
                                    raise OSError("Short local file write")
                                received += len(block)
                                current += len(block)
                                stats["downloaded_bytes"] += len(block)
                                event("progress", current)
                        finally:
                            stream.flush()
                            os.fsync(stream.fileno())
                    if received != expected_response:
                        raise DownloadInterruptedError("HTTP range ended before declared length")
                    consecutive_failures = 0
                except (requests.exceptions.RequestException, DownloadInterruptedError) as error:
                    stats["failures"] += 1
                    # Re-read the fsynced checkpoint, never assume a complete range.
                    current = partial.stat().st_size
                    productive = current > start
                    consecutive_failures = 0 if productive else consecutive_failures + 1
                    event("interrupted", current)
                    if retries == 0 or consecutive_failures > retries:
                        raise DownloadInterruptedError("Retry limit reached; durable partial preserved") from error
                    if consecutive_failures and retry_backoff:
                        time.sleep(min(30., retry_backoff * (2 ** min(consecutive_failures - 1, 6))))
                finally:
                    if response is not None:
                        response.close()
            result = verified_result(partial, False)
            # Same-directory atomic promotion, only after full authoritative hash.
            if destination.exists():
                raise DownloadIntegrityError("Destination appeared while its download lock was held")
            os.replace(partial, destination)
            event("verified", expected_size)
            return result
        finally:
            if owned_session:
                client.close()
