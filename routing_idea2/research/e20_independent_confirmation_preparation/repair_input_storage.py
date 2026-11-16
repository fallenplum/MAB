"""Repackage safe sealed input metadata as non-pickle Unicode arrays.

The initial input encoder inherited JSON null GSM subtask labels. Preserve its
original file, reuse only its numeric embedding member and reconstruct metadata
from the isolated JSON input projection. No mixed protected source is opened.
"""
from __future__ import annotations

import json
from fit_final_models import DEFAULT_ROOT, now, sha, write_json, write_npz
import numpy as np


def main():
    inputs = DEFAULT_ROOT.parent / "inputs"
    original = inputs / "SEALED200_INPUTS_ONLY.npz"
    text_path = inputs / "SEALED200_INPUTS_ONLY.jsonl"
    destination = inputs / "SEALED200_INPUTS_ONLY_UNICODE.npz"
    rows = [json.loads(line) for line in text_path.read_text(encoding="utf-8-sig").splitlines()]
    with np.load(original, allow_pickle=False) as archive:
        vectors = archive["embeddings"].copy()
    def strings(key):
        return np.array(["" if row[key] is None else str(row[key]) for row in rows], dtype=str)
    write_npz(destination, embeddings=vectors, question_ids=strings("question_id"),
              canonical_hashes=strings("canonical_question_sha256"), task_labels=strings("task"), subtask_labels=strings("subtask"))
    with np.load(destination, allow_pickle=False) as archive:
        for key in archive.files:
            if archive[key].dtype.hasobject:
                raise ValueError("Unexpected object array after storage repair")
        if not np.array_equal(archive["embeddings"], vectors):
            raise AssertionError("Frozen numeric embeddings changed")
    receipt = dict(
        schema="e20_input_storage_repair_v1", utc=now(), status="PASS_NON_PICKLE_UNICODE",
        reason="Initial GSM JSON-null subtask metadata produced an object dtype; prediction loader correctly rejected it",
        source_sha256=sha(__file__), original_npz_sha256=sha(original), original_retained=True,
        isolated_input_json_sha256=sha(text_path), repaired_npz_sha256=sha(destination),
        failed_prediction_attempts=1, failed_training_attempts=0,
        embedding_values_changed=0, new_encoder_forwards=0, new_llm_calls=0,
        answer_values_decoded=0, reward_values_decoded=0,
        metadata_rule="JSON null subtask converted to empty Unicode string; not a model feature",
    )
    write_json(inputs / "SEALED200_INPUT_STORAGE_REPAIR_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
