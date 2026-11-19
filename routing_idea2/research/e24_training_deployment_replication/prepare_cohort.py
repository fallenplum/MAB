"""Draw the prespecified E24 cohort from audited input-only rows and encode it."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from importlib import metadata
import hashlib
import json
from pathlib import Path
import sys

from replication import CODE, labels_for, np, read, sha, write

REQUIRED_FIELDS = {"question", "question_id", "task", "subtask", "source_split", "source_file",
                   "source_index", "source_frame", "canonical_question_sha256"}
OPTIONAL_FIELDS = {"source_revision"}


def audited_groups(path):
    sys.path.insert(0, str(CODE.parent / "e18_mechanism_and_confirmation_preparation"))
    from audit_sources import canonical_hash

    groups = defaultdict(list)
    seen_ids, seen_hashes = set(), set()
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            keys = set(row)
            if not REQUIRED_FIELDS <= keys <= REQUIRED_FIELDS | OPTIONAL_FIELDS:
                raise ValueError("Unexpected field in previously audited input-only candidates")
            identity, digest = row["question_id"], row["canonical_question_sha256"]
            if identity in seen_ids or digest in seen_hashes or digest != canonical_hash(row["question"]):
                raise ValueError("Repeated or inconsistent audited input identity")
            seen_ids.add(identity)
            seen_hashes.add(digest)
            task = row["task"]
            if (task, row["source_split"]) not in {("gsm8k", "train"), ("bbh", "test")}:
                raise ValueError("Evaluation row changed its frozen original source frame")
            key = "gsm8k" if task == "gsm8k" else f"bbh/{row['subtask']}"
            groups[key].append(row)
    return groups


def select(groups, spec):
    design = spec["cohort"]
    seed = design["selection_seed"]

    def rank(group, row):
        return hashlib.sha256(f"{seed}|{group}|{row['canonical_question_sha256']}".encode()).digest()

    chosen = []
    for group, count in design["stratum_counts"].items():
        if len(groups[group]) < count:
            raise ValueError(f"Insufficient audited input capacity for {group}")
        chosen.extend(sorted(groups[group], key=lambda row: rank(group, row))[:count])
    return chosen


def encode(rows, assets):
    source = read(assets / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/INPUT_ASSETS_RECEIPT.json")
    versions = {package: metadata.version(package) for package in ("numpy", "torch", "transformers", "tokenizers", "safetensors")}
    if any(value != source["runtime_versions"][name] for name, value in versions.items()):
        raise ValueError("Frozen MiniLM runtime differs from the original input receipt")
    sys.path.insert(0, str(assets / "src"))
    from budget_router.encoding import FrozenQuestionEncoder
    import torch

    torch.set_num_threads(2)
    revision = source["feature_encoder"]["revision"]
    encoder = FrozenQuestionEncoder(assets / "models/sentence-transformers--all-MiniLM-L6-v2",
                                    revision=revision, receipt=assets / "artifacts/audit/encoder_model_download.json",
                                    device="cpu", max_length=256)
    embedding = encoder.encode([row["question"] for row in rows], batch_size=8)
    if embedding.shape != (len(rows), 384) or not np.isfinite(embedding).all():
        raise ValueError("Invalid frozen MiniLM input-side encoding")
    return embedding, revision, versions


def main(args):
    audit = read(args.audit)
    audit_hash = sha(args.audit)
    spec = read(args.protocol)
    if (audit_hash != spec["cohort"]["source_audit_sha256"]
            or audit.get("private_unselected_input_rows_sha256") != sha(args.candidates)
            or audit.get("gold_or_reward_values_decoded_this_audit") != 0
            or audit.get("protected_input_values_decoded_this_audit") != 0):
        raise ValueError("The protocol-bound source audit must certify these input-only candidate rows")
    groups = audited_groups(args.candidates)
    rows = select(groups, spec)
    task = np.array([row["task"] for row in rows])
    subtask = np.array([row["subtask"] or "" for row in rows])
    if Counter(labels_for(task, subtask).tolist()) != spec["cohort"]["stratum_counts"]:
        raise ValueError("Admitted question counts differ from the frozen quotas")
    args.output.mkdir(parents=True, exist_ok=True)
    question_path = args.output / "INPUTS_ONLY.jsonl"
    if question_path.exists():
        with question_path.open(encoding="utf-8") as stream:
            if [json.loads(line) for line in stream] != rows:
                raise ValueError("Previously frozen selection differs from the audited deterministic cohort")
    else:
        with question_path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    selection = dict(schema="e24_input_selection_v1",
          protocol_sha256=sha(args.protocol), source_audit_sha256=audit_hash, input_sha256=sha(question_path),
          candidate_inputs_sha256=sha(args.candidates), question_counts=spec["cohort"]["stratum_counts"],
          model_outputs_used_in_selection=0, answer_values_decoded=0, reward_values_decoded=0)
    selection_path = args.output / "INPUT_SELECTION_RECEIPT.json"
    if selection_path.exists():
        if read(selection_path) != selection:
            raise ValueError("Previously frozen selection receipt differs")
    else:
        write(selection_path, selection)
    vector_path = args.output / "INPUTS_ONLY.npz"
    if vector_path.exists():
        raise FileExistsError("The frozen encoder output already exists")
    vectors, revision, versions = encode(rows, args.assets)
    with vector_path.open("xb") as stream:
        np.savez_compressed(stream, embeddings=vectors, question_ids=np.array([row["question_id"] for row in rows]),
                            canonical_hashes=np.array([row["canonical_question_sha256"] for row in rows]),
                            task_labels=task, subtask_labels=subtask)
    receipt = dict(schema="e24_input_cohort_admission_v1", status="INPUT_ONLY_ADMITTED_AND_ENCODED",
                   protocol_sha256=sha(args.protocol), input_sha256=sha(question_path),
                   source_audit_sha256=audit_hash, vector_packet_sha256=sha(vector_path),
                   source_code_sha256=sha(Path(__file__)), encoder_revision=revision, runtime_versions=versions,
                   question_counts=spec["cohort"]["stratum_counts"], questions=len(rows), new_encoder_questions=len(rows),
                   source_frame_counts=dict(Counter(row["source_frame"] for row in rows)),
                   model_outputs_used_in_selection=0, answer_values_decoded=0, reward_values_decoded=0, new_llm_calls=0)
    write(args.output / "INPUT_COHORT_ADMISSION.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=CODE / "E24_PROTOCOL.json")
    for name in ("audit", "candidates", "assets", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    main(parser.parse_args())
