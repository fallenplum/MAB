"""Stage only de-identified E20 preparation artifacts and emit API-sized text chunks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys


CODE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\develop\CV\多臂老虎机")
STAGE = WORKSPACE / ".safe_e20_final_delivery"
STUDY = Path(r"E:\语言模型组合路由\research\e20_independent_confirmation_preparation")
PREFIX = "routing_idea2/research/e20_independent_confirmation_preparation/"
PRIVATE_ONLY = {"E20_EXACT_EXECUTION_BUDGET.json"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def blob_sha(data):
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def public_files():
    return sorted(path for path in CODE.iterdir()
                  if path.is_file() and path.suffix in {".py", ".md", ".json", ".txt"}
                  and path.name not in PRIVATE_ONLY)


def stage():
    import numpy as np

    sources = public_files()
    model_audit = json.loads((CODE / "FINAL_MODEL_NUMERICAL_AUDIT.json").read_text(encoding="utf-8"))
    if digest((CODE / "fit_final_models.py").read_bytes()) != model_audit["fit_source_sha256"]:
        raise ValueError("Frozen fitting source changed")
    historical_index = json.loads((WORKSPACE / ".safe_e19_final_delivery/FILE_INDEX.json").read_text(encoding="utf-8"))
    checked = []
    for record in historical_index["files"]:
        previous = WORKSPACE / "experiments/e19_symmetric_tie_constraint" / record["package_path"]
        if previous.is_file():
            if digest(previous.read_bytes()) != record["sha256"]:
                raise ValueError("Historical E19 workspace asset changed")
            checked.append(record["package_path"])

    final_root = STUDY / "private/final_models"
    forbidden = set()
    for pool in ("FRESH400", "SEALED200"):
        with np.load(final_root / f"{pool}_INPUT_ONLY_PREDICTIONS_ORIGINAL_OPERATOR_FROZEN.npz", allow_pickle=False) as archive:
            for key in ("question_ids", "canonical_hashes"):
                forbidden.update(map(str, archive[key].tolist()))
    text_path = STUDY / "private/inputs/SEALED200_INPUTS_ONLY.jsonl"
    for line in text_path.read_text(encoding="utf-8").splitlines():
        question = json.loads(line)["question"]
        if len(question) >= 60:
            forbidden.add(question)
    for path in sources:
        content = path.read_bytes().decode("utf-8")
        if any(value in content for value in forbidden):
            raise ValueError("Private input identity or text appears in public artifact " + path.name)
        if path.suffix == ".py":
            compile(content, str(path), "exec")
        if path.suffix == ".json":
            json.loads(content)

    receipt_path = CODE / "E20_PROTOCOL_FREEZE_RECEIPT.json"
    receipt = dict(schema="e20_protocol_freeze_receipt_v1", utc=datetime.now(timezone.utc).isoformat(),
                   protocol_sha256=digest((CODE / "E20_PROTOCOL.json").read_bytes()),
                   prefit_contract_sha256=digest((CODE / "E20_PREPARATION_CONTRACT.json").read_bytes()),
                   final_model_sha256=model_audit["frozen_model_sha256"],
                   source_hashes={path.name: digest(path.read_bytes()) for path in sources if path.suffix == ".py"},
                   evaluation_started=False, execution_approval="PENDING")
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text(encoding="utf-8"))
        for key in ("protocol_sha256", "prefit_contract_sha256", "final_model_sha256", "source_hashes"):
            if old[key] != receipt[key]:
                raise ValueError("Existing immutable final preparation receipt differs")
    else:
        write(receipt_path, receipt)
    write(CODE / "E20_PUBLIC_QUALIFICATION.json", dict(
        status="PASS", public_private_identity_or_text_matches=0,
        public_arrays_or_weights=0, code_syntax_and_json_parsing="PASS",
        preserved_E19_workspace_files=checked, new_llm_calls=0,
        fresh_reward_values_decoded=0, sealed_answer_values_decoded=0,
        ci_or_bootstrap_executed=False,
        remote_tree_rule="New paths under E20 only, parent E19 commit preserved",
        privacy_scan_scope="Current input identities, canonical hashes and protected input plaintext; extension allowlist excludes private arrays and weights"))
    sources = public_files()
    STAGE.mkdir(parents=True, exist_ok=True)
    (STUDY / "public").mkdir(parents=True, exist_ok=True)
    draft = CODE / "E20_EXACT_EXECUTION_BUDGET.json"
    if draft.exists():
        private_copy = STUDY / "private/budget/E20_INITIAL_UNQUALIFIED_BUDGET_RETAINED.json"
        if not private_copy.exists():
            shutil.copy2(draft, private_copy)
    records = []
    for path in sources:
        data = path.read_bytes()
        shutil.copy2(path, STAGE / path.name)
        shutil.copy2(path, STUDY / "public" / path.name)
        records.append(dict(package_path=path.name, publication_path=PREFIX + path.name,
                            bytes=len(data), sha256=digest(data), git_sha1=blob_sha(data)))
    index = dict(schema="e20_public_file_index_v1", source_commit="07b01361d425b089e49643f1ec5282eae0ac1ec0",
                 branch="codex/e20-independent-confirmation-preparation", publication_prefix=PREFIX,
                 files=records, private_artifacts_uploaded=0)
    write(STAGE / "FILE_INDEX.json", index)
    shutil.copy2(STAGE / "FILE_INDEX.json", STUDY / "public/FILE_INDEX.json")
    print(json.dumps({"status": "STAGED", "files": len(records) + 1,
                      "bytes": sum(item["bytes"] for item in records) + (STAGE / "FILE_INDEX.json").stat().st_size,
                      "preserved_E19_files_checked": len(checked), "private_matches": 0}, ensure_ascii=False))


def chunks():
    paths = sorted(path for path in STAGE.iterdir() if path.is_file()
                   and path.name not in {"REMOTE_DELIVERY_VERIFIED.json"})
    payloads = []
    current, size = [], 0
    for path in paths:
        content = path.read_bytes().decode("utf-8")
        for offset in range(0, len(content), 18000):
            piece = content[offset:offset + 18000]
            if current and size + len(piece) > 30000:
                payloads.append(current)
                current, size = [], 0
            current.append(dict(path=PREFIX + path.name, offset=offset, text=piece))
            size += len(piece)
    if current:
        payloads.append(current)
    return payloads


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
        print(json.dumps({"chunks": len(chunks())}))
