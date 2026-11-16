"""Stage P2 source, aggregate receipts and live activation status for publication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


CODE = Path(__file__).resolve().parent
WORKSPACE = CODE.parents[1]
STUDY = Path(r"E:\语言模型组合路由\research\e20_p2_execution")
STAGE = WORKSPACE / ".safe_e20_p2_delivery"
PREFIX = "routing_idea2/research/e20_p2_execution/"
JSON_ALLOWLIST = {
    "P2_PRESELECTION_PROTOCOL.json", "P2_INPUT_SELECTION_FROZEN.json",
    "P2_INPUT_ADMISSION_REPORT.json", "P2_ENCODER_ENVIRONMENT.json",
    "P2_ENCODER_RECEIPT.json", "P2_FROZEN_EXECUTION_UNION.json",
    "P2_REAL_SCOPE_QUALIFICATION.json", "P2_REFERENCE_LOCATOR_QUALIFICATION.json",
    "P2_REAL_REFERENCE_PLAN_QUALIFICATION.json", "P2_REAL_RUNTIME_SCOPE_QUALIFICATION.json",
    "P2_BLIND_AND_SCOPE_QUALIFICATION.json", "P2_PAID_SCORER_QUALIFICATION.json",
    "P2_RUNTIME_SOURCE_ADAPTATION.json", "P2_FIRST_REAL_AGGREGATION_QUALIFICATION.json",
    "P2_ACTIVATION_STATUS.json",
    "P2_FIRST_NATIVE_EXPORT_QUALIFICATION.json",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def stage():
    import numpy as np

    status = json.loads((CODE / "P2_ACTIVATION_STATUS.json").read_text(encoding="utf-8-sig"))
    paths = sorted(path for path in CODE.iterdir() if path.is_file() and (
        path.suffix in {".py", ".md", ".txt"} or path.name in JSON_ALLOWLIST
    ) and "PRIVATE" not in path.name.upper())
    forbidden = set()
    with np.load(STUDY / "private/P2_PREDICTIONS_FROZEN.npz", allow_pickle=False) as archive:
        for key in ("question_ids", "canonical_hashes"):
            forbidden.update(map(str, archive[key].tolist()))
    for line in (STUDY / "private/ADMITTED2400_INPUTS_ONLY.jsonl").read_bytes().splitlines():
        question = json.loads(line)["question"]
        if len(question) >= 60:
            forbidden.add(question)
    for path in paths:
        content = path.read_bytes().decode("utf-8-sig")
        if any(value in content for value in forbidden):
            raise ValueError("Private input content in public artifact " + path.name)
        if path.suffix == ".json":
            json.loads(content)
        if path.suffix == ".py":
            compile(content, str(path), "exec")
    STAGE.mkdir(parents=True, exist_ok=True)
    public = STUDY / "public"
    public.mkdir(parents=True, exist_ok=True)
    files = []
    for path in paths:
        data = path.read_bytes()
        shutil.copy2(path, STAGE / path.name)
        shutil.copy2(path, public / path.name)
        files.append(dict(package_path=path.name, publication_path=PREFIX + path.name,
                          bytes=len(data), sha256=digest(data), git_sha1=git_sha(data)))
    index = dict(schema="p2_activation_public_file_index_v1",
                 branch="codex/e20-p2-independent-confirmation",
                 parent_commit="031d32ee34911f58c535803c078584c424c8f6e3",
                 stage="PRODUCTION_EXECUTION_STARTED", files=files,
                 private_identity_or_text_matches=0, private_arrays_or_weights_uploaded=0,
                 activation_snapshot_sha256=digest((CODE / "P2_ACTIVATION_STATUS.json").read_bytes()))
    data = (json.dumps(index, ensure_ascii=False, indent=2) + "\n").encode()
    (STAGE / "FILE_INDEX.json").write_bytes(data)
    (public / "FILE_INDEX.json").write_bytes(data)
    print(json.dumps({"status":"STAGED", "files":len(files)+1,
                      "bytes":sum(row["bytes"] for row in files)+len(data),
                      "source_status":status.get("status"), "private_matches":0}, ensure_ascii=False))


def chunks():
    result, current, length = [], [], 0
    for path in sorted(STAGE.iterdir()):
        if not path.is_file() or path.name == "REMOTE_DELIVERY_VERIFIED.json":
            continue
        text = path.read_bytes().decode("utf-8")
        for offset in range(0, len(text), 18000):
            part = text[offset:offset + 18000]
            if current and length + len(part) > 30000:
                result.append(current)
                current, length = [], 0
            current.append(dict(path=PREFIX + path.name, offset=offset, text=part))
            length += len(part)
    if current:
        result.append(current)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--chunk", type=int)
    args = parser.parse_args()
    if args.stage:
        stage()
    elif args.chunk is not None:
        print(json.dumps(chunks()[args.chunk], ensure_ascii=False))
    else:
        print(json.dumps({"chunks":len(chunks())}))
