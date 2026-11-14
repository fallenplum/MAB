"""Independent synthetic checks of the exact helpers used by E10 training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(source, output):
    source = Path(source)
    initial_hash = sha256(source)
    spec = importlib.util.spec_from_file_location("e10_train_reviewed", source)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    checks = []

    def add(name, residual, tolerance, details):
        residual = float(residual)
        passed = residual <= tolerance
        checks.append(dict(name=name, passed=passed, maximum_absolute_residual=residual,
                           tolerance=tolerance, details=details))
        if not passed:
            raise AssertionError(f"{name} failed with residual {residual}")

    rows = np.arange(48, dtype=float)[:, None]
    dimensions = np.arange(24, dtype=float)[None, :]
    features = np.sin((rows+1)*(dimensions+1)/23) + np.cos((rows+2)*(dimensions+3)/31)
    eval_features = np.sin((rows[:9]+51)*(dimensions+1)/23) + np.cos((rows[:9]+52)*(dimensions+3)/31)
    outputs = np.arange(20)[None, None, :]
    generations = np.arange(3)[None, :, None]
    reward_blocks = (1 + np.cos((rows[:, None, :]+3)*(outputs+2)/17 + generations/7))/2
    labels = reward_blocks.mean(axis=1)
    feature_map = training.fit_features(features)
    phi = feature_map.phi(features)
    eval_phi = feature_map.phi(eval_features)
    coefficient = training.ridge_weights(phi, labels, raw_alpha=1000.)
    direct = np.linalg.solve(phi.T@phi+(1000./3)*np.eye(17), phi.T@labels)
    add("fixed_m10_ridge_matches_independent_normal_equation",
        abs(coefficient-direct).max(), 1e-12,
        {"fit_questions": 48, "outputs": 20, "parameters": 340,
         "normal_equation_penalty": "1000/3 times I17"})
    residuals = phi@coefficient-labels
    regularization = 1000./(3*len(phi)*20)
    gradient = 2*phi.T@residuals/(len(phi)*20) + 2*regularization*coefficient
    add("literal_sum_squared_error_over_n_times20_gradient",
        abs(gradient).max(), 1e-12,
        {"lambda": regularization, "data_loss": float(np.sum(residuals**2)/(len(phi)*20))})
    constant = training.ridge_weights(phi, np.ones((48, 20)), raw_alpha=1000.)
    expected_constant = 48/(48+1000./3)
    add("constant_feature_receives_same_penalty",
        max(abs(constant[0]-expected_constant).max(), abs(constant[1:]).max()), 1e-12,
        {"expected_intercept_coefficient": expected_constant})

    archive = io.BytesIO()
    np.savez_compressed(archive, coefficient=coefficient, **feature_map.arrays())
    archive.seek(0)
    with np.load(archive, allow_pickle=False) as saved:
        restored = ((eval_features.astype(float)-saved["pca_mean"])@saved["pca_components"].T
                    -saved["scaler_mean"])/saved["scaler_scale"]
        restored_scores = np.column_stack((np.ones(len(restored)), restored))@saved["coefficient"]
    scores = eval_phi@coefficient
    add("in_memory_saved_state_restores_scores", abs(restored_scores-scores).max(), 1e-12,
        {"actions_equal": bool(np.array_equal(training.semantic_argmax(scores, training.ACTION_KEYS),
                                               training.semantic_argmax(restored_scores, training.ACTION_KEYS)))})
    keys = training.ACTION_KEYS
    ties = np.zeros((2, 20))
    expected = min(range(20), key=lambda index: keys[index])
    selected = training.semantic_argmax(ties, keys)
    permutation = np.roll(np.arange(20), 7)
    renamed = permutation[training.semantic_argmax(ties[:, permutation], [keys[i] for i in permutation])]
    add("semantic_argmax_exact_ties_and_coordinate_relabeling",
        max(abs(selected-expected).max(), abs(renamed-selected).max()), 0,
        {"semantic_first_action_index": expected, "all_twenty_outputs_tied": True})

    history = {"ids": np.array([f"synthetic_{i}" for i in range(48)]),
               "task": np.array(["bbh"]*24+["gsm8k"]*24),
               "sub": np.array(["alpha"]*12+["beta"]*12+["gsm8k"]*24),
               "Y": reward_blocks}
    permutation_errors = []
    for seed in (20261114091, 20261114092, 20261114093):
        shuffled, mapping = training.permute_region(history, seed)
        permutation_errors.append(abs(shuffled["Y"]-reward_blocks[mapping]).max())
        if not np.array_equal(history["task"], history["task"][mapping]):
            raise AssertionError("Content permutation crossed task")
        if not np.array_equal(history["sub"], history["sub"][mapping]):
            raise AssertionError("Content permutation crossed BBH subtask")
    add("three_fixed_seed_controls_permute_whole_reward_blocks_within_groups",
        max(permutation_errors), 0, {"fixed_seeds": [20261114091, 20261114092, 20261114093]})
    if sha256(source) != initial_hash:
        raise RuntimeError("Training source changed during qualification")
    result = dict(status="PASS_INDEPENDENT_SYNTHETIC_HEAD_QUALIFICATION",
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  source_path=str(source), source_sha256=initial_hash,
                  real_train600_reward_values_decoded=False, real_models_fitted=0,
                  synthetic_ridge_solutions=2, checks=checks)
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
