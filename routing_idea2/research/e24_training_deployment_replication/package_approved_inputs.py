"""Package the approved E24 input-only cohort and the exact frozen call keys."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from plan_design import budget

CODE = Path(__file__).resolve().parent
NAMES = ("B_U", "B_P", "C_U", "C_P")
FIELDS = {"question", "question_id", "task", "subtask", "source_split", "source_file",
          "source_index", "source_frame", "source_revision", "canonical_question_sha256"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def package(cohort, output, reference_plan_sha256):
    protocol = (CODE / "E24_PROTOCOL.json").read_bytes()
    budget_bytes = (CODE / "INPUT_ONLY_BUDGET.json").read_bytes()
    admission = read(cohort / "INPUT_COHORT_ADMISSION.json")
    receipt = read(cohort / "E24_POLICY_FREEZE_RECEIPT.json")
    raw = (cohort / "INPUTS_ONLY.jsonl").read_bytes()
    keys_raw = (cohort / "FROM_ZERO_EXECUTION_KEYS.json").read_bytes()
    keys = json.loads(keys_raw)
    cohort_predictions = cohort / "E24_POLICIES_FROZEN.npz"
    prediction_digest = digest(cohort_predictions.read_bytes())
    if (digest(raw) != admission["input_sha256"] or digest(protocol) != receipt["protocol_sha256"]
            or receipt["prediction_sha256"] != prediction_digest
            or keys["prediction_sha256"] != prediction_digest or keys["from_zero"] is not True
            or json.loads(budget_bytes)["prediction_sha256"] != prediction_digest
            or digest((cohort / "INPUT_COHORT_ADMISSION.json").read_bytes()) != receipt["input_admission_sha256"]):
        raise ValueError("Frozen E24 input, policy, budget and approval scope disagree")

    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 1196 or any(set(row) != FIELDS for row in rows):
        raise ValueError("Require exactly the admitted 1,196 input-only records")
    ids = [row["question_id"] for row in rows]
    with np.load(cohort_predictions, allow_pickle=False) as packet:
        if tuple(packet["names"].tolist()) != NAMES or packet["question_ids"].tolist() != ids:
            raise ValueError("Policy and input identities differ")
        if packet["canonical_question_sha256"].tolist() != [row["canonical_question_sha256"] for row in rows]:
            raise ValueError("Canonical question identities differ")
        probability = dict(zip(NAMES, packet["probabilities"]))
    computed, private = budget(ids, probability, (), ())
    if (computed["union"]["missing_member_calls"], computed["union"]["missing_aggregate_calls"]) != (20661, 53661):
        raise ValueError("Reconstructed support differs from the authorized union count")
    for name in ("missing_member_keys", "missing_aggregate_keys"):
        expected = set(map(tuple, private["union"][name]))
        if expected != set(map(tuple, keys[name])) or len(expected) != len(keys[name]):
            raise ValueError("Existing frozen call keys differ from the independently reconstructed support")

    policies = json_bytes({name: value.tolist() for name, value in probability.items()})
    origin = dict(protocol_sha256=digest(protocol), prediction_sha256=prediction_digest,
                  budget_sha256=digest(budget_bytes), input_sha256=digest(raw), policies_sha256=digest(policies),
                  execution_keys_sha256=digest(keys_raw), authorization="继续开始实验验证啊")
    if reference_plan_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", reference_plan_sha256):
            raise ValueError("Invalid local scoring locator digest")
        origin["reference_plan_sha256"] = reference_plan_sha256
    files = {"INPUTS.jsonl": raw, "POLICIES.json": policies, "EXECUTION_KEYS.json": keys_raw,
             "ORIGIN.json": json_bytes(origin), "E24_PROTOCOL.json": protocol, "INPUT_ONLY_BUDGET.json": budget_bytes}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    print(json.dumps(dict(questions=len(rows), member_calls=20661, aggregations=53661,
                          member_profiles=dict(Counter(key[2] for key in keys["missing_member_keys"])),
                          files={name: dict(sha256=digest(data), bytes=len(data)) for name, data in files.items()})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-plan-sha256")
    args = parser.parse_args()
    package(args.cohort, args.output, args.reference_plan_sha256)
