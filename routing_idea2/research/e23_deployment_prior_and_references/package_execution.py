"""Package only the frozen approved reference-supplement inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np


CODE = Path(__file__).resolve().parent
ASSETS = Path(r"E:\语言模型组合路由")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n").encode()


def package(args):
    private = args.root / "private"
    freeze = read(private/"FREEZE.json")
    if sha(private/"POLICIES_FROZEN.npz") != freeze["policy_sha256"]:
        raise ValueError("Supplemental policies changed")
    missing = read(private/"MISSING_KEYS.json")["union"]
    if (len(missing["missing_member_keys"]) != 1770 or len(missing["missing_aggregate_keys"]) != 4263):
        raise ValueError("Exact approved supplement call counts differ")
    selected_ids = set(missing["missing_question_ids"])
    if len(selected_ids) != 840:
        raise ValueError("Approved missing-call question union differs")
    with np.load(private/"POLICIES_FROZEN.npz", allow_pickle=False) as archive:
        ids = archive["question_ids"].astype(str)
        selected = np.array([item in selected_ids for item in ids])
        policies = {name: probability[selected].tolist() for name, probability in
                    zip(archive["names"].tolist(), archive["probabilities"])}
    parent = ASSETS / "research/e20_p2_execution/private"
    source = parent / "ADMITTED2400_INPUTS_ONLY.jsonl"
    if sha(source) != "6192d8cdf7f5945b3b642069e16d9234e7f3f86805c5a7a36d31f9a8cee489d8":
        raise ValueError("Original input identity changed")
    rows = [row for row in (json.loads(line) for line in source.read_text(encoding="utf-8").splitlines())
            if row["question_id"] in selected_ids]
    if [row["question_id"] for row in rows] != ids[selected].tolist():
        raise ValueError("Supplemental policy and input order differs")
    body = b"".join(encode(row) for row in rows)
    budget = dict(member_generations=1770, aggregations=4263, questions=840,
                  max_incremental_tokens=45797376, actual_new_tokens=0)
    origin = dict(protocol_sha256=sha(CODE/"PROTOCOL.json"),
                  frozen_actions_sha256=freeze["policy_sha256"],
                  target_inputs_sha256=hashlib.sha256(body).hexdigest(),
                  budget_sha256=hashlib.sha256(encode(budget)).hexdigest(),
                  cache_evidence_sha256=sha(CODE.parents[1]/".e20_p2_cpu_runtime/OUTPUTS_FROZEN.json"),
                  parent_scope_sha256="b355581aaaab1a581e76dab35107a897948da7b4086a1d9e2a33be4a3e1cd1b9",
                  all_policies_frozen_before_supplement_execution=True,
                  user_approval="只补齐这些缺失调用", approved_member_calls=1770, approved_aggregation_calls=4263)
    files = {"INPUTS.jsonl": body, "POLICIES.json": encode(policies), "MISSING.json": encode(missing),
             "ORIGIN.json": encode(origin), "BUDGET.json": encode(budget)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "x:gz") as stream:
        for name, data in files.items():
            entry = tarfile.TarInfo("inputs/"+name)
            entry.size = len(data)
            stream.addfile(entry, io.BytesIO(data))
    print(json.dumps(dict(status="APPROVED_MISSING_UNION_PACKAGED", bytes=args.output.stat().st_size,
        sha256=sha(args.output), member_profiles=dict(Counter(key[2] for key in missing["missing_member_keys"])),
        members=1770, aggregations=4263, questions=840)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    package(parser.parse_args())
