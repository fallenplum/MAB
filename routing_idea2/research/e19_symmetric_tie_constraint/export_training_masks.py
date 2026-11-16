"""Export the exact history-only labels and beta masks for the frozen E19 fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_e19 import DEFAULT_REWARDS, REWARD_SHA, read_reward_rows, sha
from weighted_objectives import uniform_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    if sha(DEFAULT_REWARDS) != REWARD_SHA:
        raise ValueError("The frozen history reward source differs")
    rows = []
    for fold in range(5):
        folder = args.study_root / "private" / f"fold{fold}"
        target = folder / "TRAINING_LABELS_AND_MASK.npz"
        if target.exists():
            raise FileExistsError("Training label exports are append-only")
        with np.load(folder / "B_PREDICTIONS.npz", allow_pickle=False) as file:
            history = file["H"]
        observations, access = read_reward_rows(DEFAULT_REWARDS, history)
        means = observations.mean(axis=0)
        tied = np.all(means == means[:, :1], axis=1)
        alpha = uniform_weights(means)
        log = json.loads((folder / "C_FIT_LOG.json").read_text(encoding="utf-8"))["diagnostics"]
        if int(tied.sum()) != log["all_action_tied_questions"] or int((~tied).sum()) != log["effective_question_weight_sum"]:
            raise ValueError("Exported T or C weights differ from the completed fit")
        if not np.allclose(alpha.sum(axis=1), 1.0, atol=1e-14):
            raise ValueError("Uniform empirical maximizer weights are not normalized")
        np.savez_compressed(target, history_indices=history, empirical_mean_rewards=means,
            empirical_max_uniform_weights=alpha, all_action_tied_mask=tied,
            row_weights_B=np.ones(480), row_weights_C=(~tied).astype(float),
            original_normalization_count=np.asarray(480))
        rows.append(dict(fold=fold, original_rows=480, original_denominator=480,
                         T_questions=int(tied.sum()), B_weight_sum=480,
                         C_weight_sum=int((~tied).sum()), source_reward_sha256=REWARD_SHA,
                         private_export_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    public = dict(schema="e19_history_label_mask_export_v1", status="COMPLETE", folds=rows,
                  scope="History-only post-fit reproducibility export; fit artifacts unchanged",
                  new_fits=0, new_llm_calls=0, evaluation_rows_decoded=0)
    with (args.study_root / "public" / "TRAINING_MASK_EXPORT_SUMMARY.json").open("x", encoding="utf-8") as stream:
        json.dump(public, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(status="COMPLETE", history_masks=5, additional_fits=0)))


if __name__ == "__main__":
    main()
