from pathlib import Path, PurePosixPath
import hashlib, json, os, shutil, tarfile, time
import requests

REPO = "NPULH/LLMRouterBench"
REVISION = "0e5af1b84bf73437a01a1849c0f1d2468baa93fc"
ARCHIVE = "bench-release.tar.gz"
EXPECTED_SHA256 = "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1"
EXPECTED_BYTES = 1283503080

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)

def within(root, candidate):
    root, candidate = Path(root).resolve(), Path(candidate).resolve()
    candidate.relative_to(root)
    return candidate

def metadata(root):
    url = "https://huggingface.co/api/datasets/" + REPO + "/tree/" + REVISION + "?recursive=true&expand=false"
    r = requests.get(url, timeout=(15, 45))
    r.raise_for_status()
    rows = r.json()
    item = next(x for x in rows if x.get("path") == ARCHIVE)
    expected = item.get("lfs", {}).get("oid")
    if expected != EXPECTED_SHA256 or item["size"] != EXPECTED_BYTES:
        raise ValueError("Pinned source metadata differs from reviewed size/SHA256")
    pinned_url = "https://huggingface.co/datasets/" + REPO + "/resolve/" + REVISION + "/" + ARCHIVE
    head = requests.head(pinned_url, allow_redirects=True, timeout=(15, 45))
    head.raise_for_status()
    length = int(head.headers.get("Content-Length", EXPECTED_BYTES))
    if length != EXPECTED_BYTES:
        raise ValueError("HEAD size mismatch")
    result = dict(repo=REPO, revision=REVISION, archive=ARCHIVE, bytes=EXPECTED_BYTES,
                  sha256=EXPECTED_SHA256, pinned_url=pinned_url, gated=False,
                  verified_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  head=dict(status=head.status_code, host=requests.utils.urlparse(head.url).hostname,
                            accept_ranges=head.headers.get("Accept-Ranges"), content_length=length),
                  source_file_metadata=item)
    write_json(Path(root) / "data/raw/llmrouterbench/source_metadata.json", result)
    return result

def download(root, max_attempts=3):
    root = Path(root).resolve()
    target = within(root, root / "data/raw/llmrouterbench" / ARCHIVE)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if target.exists():
        if target.stat().st_size != EXPECTED_BYTES or digest(target) != EXPECTED_SHA256:
            raise ValueError("Existing complete archive fails size/SHA256; retained untouched")
        print("REUSED_VERIFIED_ARCHIVE", flush=True)
        return target
    info = metadata(root)
    for attempt in range(1, max_attempts + 1):
        start = partial.stat().st_size if partial.exists() else 0
        if start > EXPECTED_BYTES:
            raise ValueError("Partial archive exceeds source length; retained untouched")
        if start == EXPECTED_BYTES:
            break
        try:
            headers = {"Accept-Encoding": "identity"}
            if start:
                headers["Range"] = "bytes=" + str(start) + "-"
            with requests.get(info["pinned_url"], headers=headers, stream=True, timeout=(20, 60)) as r:
                r.raise_for_status()
                if start:
                    if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith("bytes " + str(start) + "-"):
                        raise ValueError("Server did not honor requested byte range; refusing overwrite")
                elif r.status_code != 200:
                    raise ValueError("Unexpected initial response status")
                with partial.open("ab" if start else "wb") as f:
                    written = start
                    last = time.monotonic()
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        if written + len(chunk) > EXPECTED_BYTES:
                            raise ValueError("Download exceeds pinned length")
                        f.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last > 5:
                            f.flush()
                            print("DOWNLOAD_BYTES", written, "/", EXPECTED_BYTES, flush=True)
                            last = time.monotonic()
            if partial.stat().st_size == EXPECTED_BYTES:
                break
        except (requests.RequestException, OSError) as exc:
            print("DOWNLOAD_RETRY", attempt, type(exc).__name__, "partial_bytes", partial.stat().st_size if partial.exists() else 0, flush=True)
            if attempt == max_attempts:
                raise RuntimeError("Public download failed after bounded attempts; partial retained") from None
            time.sleep(min(attempt * 2, 6))
    if not partial.exists() or partial.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("Incomplete download retained for resume")
    actual = digest(partial)
    if actual != EXPECTED_SHA256:
        raise ValueError("Downloaded archive SHA256 mismatch; retained for inspection")
    os.replace(partial, target)
    write_json(target.parent / "download_receipt.json",
               dict(status="VERIFIED", repo=REPO, revision=REVISION, bytes=EXPECTED_BYTES,
                    sha256=actual, path=str(target), verified_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    print("DOWNLOAD_SHA256_VERIFIED", actual, flush=True)
    return target

def validate_member(member):
    name = member.name
    p = PurePosixPath(name)
    if not name or "\\" in name or "\x00" in name or ":" in name or p.is_absolute() or ".." in p.parts:
        raise ValueError("Unsafe archive member path")
    reserved = {"CON","PRN","AUX","NUL"} | {"COM"+str(i) for i in range(1,10)} | {"LPT"+str(i) for i in range(1,10)}
    if any(part.rstrip(" .") != part or part.split(".")[0].upper() in reserved for part in p.parts):
        raise ValueError("Unsafe Windows reserved archive path")
    if not (member.isdir() or member.isfile()):
        raise ValueError("Archive links/devices and unsupported entries are rejected")
    return p

def safe_extract(archive, destination, max_unpacked_bytes=32 * 1024**3):
    archive, destination = Path(archive), Path(destination)
    root = destination.parent.resolve()
    within(root, destination)
    receipt = destination / ".cache_extraction_receipt.json"
    if receipt.exists():
        info = json.loads(receipt.read_text(encoding="utf-8"))
        if info.get("archive_sha256") != digest(archive):
            raise ValueError("Existing extraction source hash differs")
        for item in info["files"]:
            path = within(destination, destination / item["path"])
            if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError("Existing extraction member verification failed")
        print("REUSED_VERIFIED_EXTRACTION", flush=True)
        return destination
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Unmarked extraction directory is nonempty; retained, refusing overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    files, seen, total = [], set(), 0
    with tarfile.open(archive, "r|gz") as tf:
        for member in tf:
            rel = validate_member(member)
            canonical = rel.as_posix()
            if canonical in seen:
                raise ValueError("Duplicate tar member path")
            seen.add(canonical)
            target = within(destination, destination / Path(*rel.parts))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total += member.size
            if total > max_unpacked_bytes:
                raise ValueError("Archive unpacked size exceeds audited limit")
            if shutil.disk_usage(destination).free < member.size + 128 * 1024**2:
                raise OSError("Insufficient free space for next archive member")
            target.parent.mkdir(parents=True, exist_ok=True)
            h, copied = hashlib.sha256(), 0
            source = tf.extractfile(member)
            if source is None:
                raise ValueError("Missing regular member stream")
            with source, target.open("xb") as f:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    copied += len(chunk)
                    h.update(chunk)
                    f.write(chunk)
            if copied != member.size:
                raise ValueError("Truncated tar member")
            files.append(dict(path=canonical, bytes=copied, sha256=h.hexdigest()))
            if len(files) % 50 == 0:
                print("EXTRACTED_FILES", len(files), "bytes", total, flush=True)
    write_json(receipt, dict(status="VERIFIED", archive_sha256=digest(archive),
                            archive_bytes=archive.stat().st_size, unpacked_bytes=total,
                            file_count=len(files), files=files, no_links_or_traversal=True))
    print("EXTRACTION_VERIFIED", len(files), total, flush=True)
    return destination
