"""Fit four fixed regularization contrasts and replay E24 input-only scores."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve()
EXPERIMENTS = HERE.parents[1]
E20 = EXPERIMENTS / "e20_independent_confirmation_preparation"
E24 = "research/e24_training_deployment_replication/private/cohort_v1"
E25 = "research/e25_tie_mechanism/private"
MODEL_SHA = "b8b786eeb0aaac6f869136f5353186c13f78aba0a716ebf0bcad145532d01f92"
INPUTS_SHA = "0b0b959297e51f29280c3ed410bdfef7072da34d238659d5c21a114c7a787fcc"
POLICIES_SHA = "9595ee1ddf9e30e85a0c95e51af2155e6b3de4cbd36265114341f32f9c324cef"
LAMBDA0 = 0.003472222222222222
FACTORS = (0.49, 1.0, 1.0 / 0.49)
BETAS = (1, 0)


def run(assets, report_path):
    os.environ["MAB_ASSET_ROOT"] = str(assets)
    sys.path.insert(0, str(E20))
    import fit_final_models as frozen
    from predict_final_inputs import features
    from weighted_objectives import objective_components
    import clarabel
    import numpy as np
    import scipy
    import sklearn

    output = assets / E25 / "LINEAR_HEADS.npz"
    for destination in (output, report_path):
        if destination.exists():
            raise FileExistsError(f"Use a fresh destination: {destination.name}")

    model_path = frozen.DEFAULT_ROOT / "FINAL600_MODELS_FROZEN.npz"
    model_config = frozen.DEFAULT_ROOT / "FINAL600_PREFIT_CONFIG_FROZEN.json"
    inputs_path = assets / E24 / "INPUTS_ONLY.npz"
    policies_path = assets / E24 / "E24_POLICIES_FROZEN.npz"
    audit = json.loads((E20 / "FINAL_MODEL_NUMERICAL_AUDIT.json").read_text(encoding="utf-8"))
    source_hashes = {}
    for path, expected in (
        (model_path, MODEL_SHA),
        (model_config, audit["prefit_sha256"]),
        (inputs_path, INPUTS_SHA),
        (policies_path, POLICIES_SHA),
        (frozen.TRAIN_INPUTS, frozen.TRAIN_INPUT_SHA),
        (frozen.TRAIN_OUTCOMES, frozen.TRAIN_OUTCOME_SHA),
        (E20 / "fit_final_models.py", audit["fit_source_sha256"]),
        (frozen.E19_CODE / "weighted_objectives.py", frozen.WEIGHTED_SHA),
    ):
        observed = frozen.sha(path)
        if observed != expected:
            raise ValueError(f"Frozen byte identity differs: {path.name}")
        source_hashes[path.name] = observed
    if audit["final600_regularization"] != LAMBDA0:
        raise ValueError("Unexpected archived base regularization")
    profile = json.loads(model_config.read_text(encoding="utf-8"))
    if profile["regularization"] != LAMBDA0 or not profile["train600_only_preprocessing"]:
        raise ValueError("Unexpected E20 training specification")
    source_hashes["predict_final_inputs.py"] = frozen.sha(E20 / "predict_final_inputs.py")

    with np.load(model_path, allow_pickle=False) as archive:
        state = {key: archive[key].copy() for key in archive.files}
    train = frozen.load_inputs(frozen.TRAIN_INPUTS)
    if len(train["embeddings"]) != 600 or not all(np.sum(train["task_labels"] == t) == 300 for t in ("gsm8k", "bbh")):
        raise ValueError("Original balanced train600 required")
    with np.load(frozen.TRAIN_OUTCOMES, allow_pickle=False) as archive:
        outcomes = {key: archive[key].copy() for key in
                    ("question_ids", "canonical_hashes", "generation_seeds", "aggregate_scores")}
    if not np.array_equal(outcomes["generation_seeds"], [42, 137, 2024]):
        raise ValueError("Historical generation seed order differs")
    for key in ("question_ids", "canonical_hashes"):
        if not np.array_equal(outcomes[key], train[key]) or not np.array_equal(state[key], train[key]):
            raise ValueError(f"Frozen train600 identities differ: {key}")
    rewards = outcomes["aggregate_scores"]
    if rewards.shape != (3, 600, 20) or not np.isin(rewards, [0, 1]).all():
        raise ValueError("Train600 binary scores differ")
    target = rewards.mean(axis=0)
    if np.count_nonzero(np.ptp(target, axis=1) == 0) != 306:
        raise ValueError("Unexpected number of all-action tied rows")
    train_phi = state["phi_h"]
    if train_phi.shape != (600, 17) or not np.array_equal(train_phi[:, 0], np.ones(600)):
        raise ValueError("Unexpected frozen training design")
    recovered_train_phi = features(train["embeddings"], state)
    preprocessing_error = float(np.max(np.abs(recovered_train_phi - train_phi)))
    if preprocessing_error > 1e-13:
        raise ValueError("Training preprocessing replay differs")

    inputs = frozen.load_inputs(inputs_path)
    if len(inputs["question_ids"]) != 1196:
        raise ValueError("Require original E24 evaluation input count")
    for key in ("question_ids", "canonical_hashes"):
        if set(state[key].tolist()).intersection(inputs[key].tolist()):
            raise ValueError("E24 inputs overlap train600 identities")
    with np.load(policies_path, allow_pickle=False) as archive:
        for key in ("question_ids", "task_labels", "subtask_labels"):
            if not np.array_equal(archive[key], inputs[key]):
                raise ValueError(f"E24 alignment differs: {key}")
        original_scores = archive["scores"].copy()
    design = features(inputs["embeddings"], state)
    weights = np.empty((len(FACTORS), len(BETAS), 17, 20))
    scores = np.empty((len(FACTORS), len(BETAS), len(design), 20))
    cell_reports = []
    for lambda_index, factor in enumerate(FACTORS):
        regularization = factor * LAMBDA0
        at_original = factor == 1.0
        for beta_index, beta in enumerate(BETAS):
            if at_original:
                head = state["weights"][beta_index]
                diagnostics = objective_components(head, train_phi, target, regularization, beta)
            else:
                head, diagnostics = frozen.fit_weighted_tspo(
                    train_phi, target, regularization, beta, gamma=0, strict_tolerances=True)
            weights[lambda_index, beta_index] = head
            predicted = design @ np.asfortranarray(head)
            if at_original:
                expected_scores = original_scores[beta_index]
                replay_error = float(np.max(np.abs(predicted - expected_scores)))
                same_policy = np.array_equal(frozen.policy(predicted, frozen.TAU)[0],
                                             frozen.policy(expected_scores, frozen.TAU)[0])
                if replay_error > 1e-13 or not same_policy:
                    raise ValueError("Frozen E24 prediction replay differs")
                diagnostics.update(original_score_replay_max_abs_error=replay_error,
                                   original_score_bytes_equal=(predicted.tobytes() == expected_scores.tobytes()))
                predicted = expected_scores
            scores[lambda_index, beta_index] = predicted
            cell_reports.append(dict(lambda_factor=factor, regularization=regularization, beta=beta,
                                     original_head_reused=at_original, coefficient_l2=float(np.linalg.norm(head)),
                                     **{key: diagnostics[key] for key in diagnostics
                                        if key not in ("regularization", "beta")}))
        b, c = cell_reports[-2:]
        objective_gap = b["F0"] - c["F0"]
        tie_gap = c["J_T"] - b["J_T"]
        if not (-1e-7 <= objective_gap <= tie_gap + 1e-7):
            raise ValueError("Weighted TSPO endpoint optimality qualification failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_npz(output, scores=scores, question_ids=inputs["question_ids"], weights=weights,
                     betas=np.array(BETAS), lambda_factors=np.array(FACTORS),
                     lambdas=np.array(FACTORS) * LAMBDA0)
    frozen.write_json(report_path, dict(
        schema="e25_linear_regularization_fit_v1", status="COMPLETE",
        training_questions=600, cached_input_questions=1196, all_action_tied_training_questions=306,
        non_all_action_tied_training_questions=294, new_head_fits=4, frozen_heads_reused=2,
        lambda0=LAMBDA0, factors=list(FACTORS), input_scores_sha256=frozen.sha(output),
        own_source_sha256=frozen.sha(HERE), train_preprocessing_replay_max_abs_error=preprocessing_error,
        source_sha256=source_hashes, python=sys.version.split()[0], numpy=np.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__, clarabel=clarabel.__version__,
        training_reward_reads="historical train600 aggregate_scores",
        evaluation_reward_reads=0, standard_answers_read=0, new_encoder_calls=0, new_llm_calls=0,
        cells=cell_reports,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run(args.assets.resolve(), args.report.resolve())
