"""Small reward-access and frozen-policy regression checks, without real fits."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import fit_final_models as models


def main():
    checks = {}
    calls = []

    class ContaminatedArchive:
        files = sorted(models.SAFE_INPUT_KEYS | {"reward"})

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, key):
            calls.append(key)
            raise AssertionError("Contaminated packet was decoded")

    with patch.object(models.np, "load", return_value=ContaminatedArchive()):
        try:
            models.load_inputs(Path("unused-contaminated-input.npz"))
        except ValueError:
            checks["contaminated_input_rejected_before_any_member_decode"] = not calls
        else:
            raise AssertionError("Reward-bearing evaluation packet accepted")

    scores = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 2.0 - 5e-9], [0.0, 1.0, 2.0]])
    probability, count, margin = models.policy(scores, 1e-8)
    expected = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
    checks["tau_uniform_policy"] = bool(np.array_equal(probability, expected))
    checks["all_policy_rows_sum_to_one"] = bool(np.all(probability.sum(axis=1) == 1))
    permutation = np.array([2, 0, 1])
    renamed, _, _ = models.policy(scores[:, permutation], 1e-8)
    mapped = np.empty_like(renamed)
    mapped[:, permutation] = renamed
    checks["policy_action_renaming_equivariance"] = bool(np.array_equal(mapped, probability))
    exact, exact_count, _ = models.policy(scores, 0.0)
    checks["raw_exact_sensitivity_kept_separate"] = bool(
        np.array_equal(exact_count, [2, 1, 1]) and np.array_equal(count, [2, 2, 1])
    )
    checks["exact_support_nested_in_tau_support"] = bool(np.all((exact > 0) <= (probability > 0)))
    with tempfile.TemporaryDirectory(prefix="e20-guard-") as folder:
        with patch.object(models, "sha", side_effect=AssertionError("Training source accessed")):
            try:
                models.fit(Path(folder))
            except FileExistsError:
                checks["existing_final_model_directory_rejected_before_training_access"] = True
            else:
                raise AssertionError("Immutable model root accepted for overwrite")
    if not all(checks.values()):
        raise AssertionError(checks)
    print(json.dumps({"status": "PASS", "checks": checks, "new_real_fits": 0,
                      "new_llm_calls": 0, "fresh400_reward_values_read": 0,
                      "sealed200_answer_values_read": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
