"""Train paired E25 nonlinear heads with exact symmetric TSPO+ on train600."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for thread_env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_env] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

WIDTH = 16
LAMBDA = 0.003472222222222222
BETA_ORDER = (1, 0)
SEEDS = (1701, 1702, 1703)
STEPS = 2400
LR_MAX = 0.01
LR_MIN = 0.001
TAU = 1e-8
E24_ROOT = "research/e24_training_deployment_replication/private/cohort_v1"
MODEL_FILE = "research/e20_independent_confirmation_preparation/private/final_models/FINAL600_MODELS_FROZEN.npz"
SOURCES = {
    "model": "b8b786eeb0aaac6f869136f5353186c13f78aba0a716ebf0bcad145532d01f92",
    "inputs": "0b0b959297e51f29280c3ed410bdfef7072da34d238659d5c21a114c7a787fcc",
    "frozen_policies": "9595ee1ddf9e30e85a0c95e51af2155e6b3de4cbd36265114341f32f9c324cef",
}


def sha(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        return h.hexdigest()


def hidden_features(phi, hidden):
    values = np.tanh(phi @ hidden)
    return np.column_stack((np.ones(len(values)), values))


def objective_and_gradient(phi, reward, alpha, omega, hidden, output):
    activation = hidden_features(phi, hidden)
    scores = activation @ output
    adversarial_scores = 2 * scores - reward
    row_max = adversarial_scores.max(axis=1, keepdims=True)
    adversarial_max = adversarial_scores == row_max
    adversarial_mass = adversarial_max / adversarial_max.sum(axis=1, keepdims=True)
    row_loss = row_max[:, 0] - np.sum(alpha * adversarial_scores, axis=1)
    data = float(omega @ row_loss / len(phi))
    objective = data + LAMBDA * (np.sum(hidden**2) + np.sum(output**2))
    scores_grad = (2 / len(phi)) * omega[:, None] * (adversarial_mass - alpha)
    output_grad = activation.T @ scores_grad + 2 * LAMBDA * output
    hidden_grad = ((scores_grad @ output[1:].T) * (1 - activation[:, 1:]**2))
    hidden_grad = phi.T @ hidden_grad + 2 * LAMBDA * hidden
    return float(objective), hidden_grad, output_grad


def global_objective(phi, reward, alpha, omega, hidden, output):
    objective, _, _ = objective_and_gradient(phi, reward, alpha, omega, hidden, output)
    return objective


def check_gradient():
    rng = np.random.default_rng(911)
    phi = np.column_stack((np.ones(7), rng.standard_normal((7, 3))))
    reward = rng.integers(0, 4, size=(7, 5)) / 3
    reward[0] = 1
    maxima = reward == reward.max(axis=1, keepdims=True)
    alpha = maxima / maxima.sum(axis=1, keepdims=True)
    hidden = rng.normal(0, 0.4, (4, 3))
    output = rng.normal(0, 0.2, (4, 5))
    max_error = 0.0
    for beta in BETA_ORDER:
        omega = np.where(np.all(reward == reward[:, :1], axis=1), beta, 1.0)
        _, grad_hidden, grad_output = objective_and_gradient(phi, reward, alpha, omega, hidden, output)
        for values, gradient in ((hidden, grad_hidden), (output, grad_output)):
            for index in np.ndindex(values.shape):
                original = values[index]
                step = 1e-6
                values[index] = original + step
                plus = global_objective(phi, reward, alpha, omega, hidden, output)
                values[index] = original - step
                minus = global_objective(phi, reward, alpha, omega, hidden, output)
                values[index] = original
                max_error = max(max_error, abs((plus - minus) / (2 * step) - gradient[index]))
    if max_error > 2e-8:
        raise AssertionError(f"MLP objective finite differences disagree: {max_error}")
    return max_error


def initialization(seed):
    rng = np.random.default_rng(seed)
    hidden = np.vstack((np.zeros(WIDTH), rng.normal(0, 0.25, (16, WIDTH))))
    output = np.vstack((np.zeros(20), rng.normal(0, 0.25, (WIDTH, 20))))
    return hidden, output


def train_pair_member(phi, reward, seed, beta, weighted):
    alpha = weighted.uniform_weights(reward)
    tied = np.all(reward == reward[:, :1], axis=1)
    omega = np.where(tied, beta, 1.0)
    hidden, output = initialization(seed)
    initial_hidden = hidden.copy()
    parameters = (hidden, output)
    first_moments = [np.zeros_like(v) for v in parameters]
    second_moments = [np.zeros_like(v) for v in parameters]
    best_objective = np.inf
    best_parameters = None
    best_step = -1
    history = {}
    for step in range(STEPS + 1):
        value, *gradients = objective_and_gradient(phi, reward, alpha, omega, *parameters)
        if not np.isfinite(value):
            raise FloatingPointError("Training objective became nonfinite")
        if value < best_objective:
            best_objective = value
            best_parameters = tuple(parameter.copy() for parameter in parameters)
            best_step = step
        if step in (0, 300, 600, 1200, 1800, 2400):
            history[str(step)] = value
        if step == STEPS:
            break
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(np.pi * step / (STEPS - 1)))
        for parameter, gradient, first, second in zip(parameters, gradients, first_moments, second_moments):
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient**2
            corrected_first = first / (1 - 0.9**(step + 1))
            corrected_second = second / (1 - 0.999**(step + 1))
            parameter -= lr * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    hidden, adam_output = best_parameters
    design = hidden_features(phi, hidden)
    refined_output, qp = weighted.fit_weighted_tspo(
        design, reward, LAMBDA, beta, gamma=0, strict_tolerances=True
    )
    hidden_penalty = LAMBDA * float(np.sum(hidden**2))
    refined_objective = float(qp["objective"] + hidden_penalty)
    independent = global_objective(phi, reward, alpha, omega, hidden, refined_output)
    if refined_objective > best_objective + 1e-7 or abs(independent - refined_objective) > 1e-11:
        raise AssertionError("Output-only exact refinement did not match the declared objective")
    training_activation = design[:, 1:]
    singular = np.linalg.svd(training_activation - training_activation.mean(axis=0), compute_uv=False)
    diagnostics = {
        "optimization_seed": seed,
        "beta": beta,
        "adam_train_objectives": history,
        "best_train_adam_step": best_step,
        "best_train_adam_objective": best_objective,
        "refined_train_objective": refined_objective,
        "hidden_regularization_penalty": hidden_penalty,
        "hidden_weight_norm": float(np.linalg.norm(hidden)),
        "hidden_distance_from_initialization": float(np.linalg.norm(hidden - initial_hidden)),
        "hidden_centered_singular_values": singular.tolist(),
        "hidden_saturation_fraction": float(np.mean(np.abs(training_activation) > 0.95)),
        "refined_output_weight_norm": float(np.linalg.norm(refined_output)),
        "refinement_output_distance_l2": float(np.linalg.norm(adam_output - refined_output)),
        "conditional_output_certificate": qp,
    }
    return hidden, refined_output, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        raise FileExistsError("Use a new output path to preserve experiment artifacts")
    assets = args.assets.resolve(strict=True)
    os.environ["MAB_ASSET_ROOT"] = str(assets)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e20_independent_confirmation_preparation"))
    import fit_final_models as original
    import predict_final_inputs as original_prediction
    import weighted_objectives as weighted

    paths = {
        "model": assets / MODEL_FILE,
        "inputs": assets / E24_ROOT / "INPUTS_ONLY.npz",
        "frozen_policies": assets / E24_ROOT / "E24_POLICIES_FROZEN.npz",
        "train_inputs": original.TRAIN_INPUTS,
        "train_outcomes": original.TRAIN_OUTCOMES,
        "weighted_objectives": Path(weighted.__file__),
    }
    expected = {
        **SOURCES,
        "train_inputs": original.TRAIN_INPUT_SHA,
        "train_outcomes": original.TRAIN_OUTCOME_SHA,
        "weighted_objectives": original.WEIGHTED_SHA,
    }
    digests = {name: sha(path) for name, path in paths.items()}
    if digests != expected:
        raise ValueError("Frozen E20 or E24 inputs or operator differ from their original bytes")
    train = original.load_inputs(paths["train_inputs"])
    pending = original.load_inputs(paths["inputs"])
    with np.load(paths["model"], allow_pickle=False) as archive:
        state = {name: archive[name] for name in ("pca_mean", "pca_components", "scaler_mean",
                                                 "scaler_scale", "phi_h", "question_ids", "canonical_hashes")}
    with np.load(paths["train_outcomes"], allow_pickle=False) as outcomes:
        for key in ("question_ids", "canonical_hashes"):
            if not np.array_equal(outcomes[key], train[key]) or not np.array_equal(state[key], train[key]):
                raise ValueError("Frozen training inputs, targets, and features do not align")
            if set(train[key]) & set(pending[key]):
                raise ValueError("E24 inputs overlap train600")
        aggregate_scores = outcomes["aggregate_scores"]
        if (aggregate_scores.shape != (3, 600, 20) or not np.isin(aggregate_scores, (0, 1)).all()
                or not np.array_equal(outcomes["generation_seeds"], [42, 137, 2024])):
            raise ValueError("Unexpected original train600 score matrix")
        train_rewards = aggregate_scores.mean(axis=0)
    if len(pending["question_ids"]) != 1196 or int(np.all(train_rewards == train_rewards[:, :1], axis=1).sum()) != 306:
        raise ValueError("Unexpected E24 or train600 counts")
    with np.load(paths["frozen_policies"], allow_pickle=False) as frozen:
        if not np.array_equal(frozen["question_ids"], pending["question_ids"]):
            raise ValueError("E24 output alignment differs from archived evaluation")
    phi_train = original_prediction.features(train["embeddings"], state)
    phi_pending = original_prediction.features(pending["embeddings"], state)
    recovered = float(np.max(np.abs(phi_train - state["phi_h"])))
    if recovered > 1e-13:
        raise AssertionError("Original train600 design did not replay")
    max_gradient_error = check_gradient()
    fits = []
    scores, hidden_parameters, output_parameters = [], [], []
    for seed in SEEDS:
        seed_scores, seed_hidden, seed_output = [], [], []
        for beta in BETA_ORDER:
            hidden, output, fitted = train_pair_member(phi_train, train_rewards, seed, beta, weighted)
            fitted_scores = hidden_features(phi_pending, hidden) @ np.asfortranarray(output)
            sorted_scores = np.sort(fitted_scores, axis=1)
            margins = sorted_scores[:, -1] - sorted_scores[:, -2]
            active = fitted_scores >= fitted_scores.max(axis=1, keepdims=True) - TAU
            fitted["e24_input_only_margin_quantiles"] = np.quantile(margins, (0, 0.1, 0.5, 0.9, 1)).tolist()
            fitted["e24_input_only_unique_winners"] = int((active.sum(axis=1) == 1).sum())
            fits.append(fitted)
            seed_scores.append(fitted_scores)
            seed_hidden.append(hidden)
            seed_output.append(output)
        scores.append(seed_scores)
        hidden_parameters.append(seed_hidden)
        output_parameters.append(seed_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, scores=np.asarray(scores), question_ids=pending["question_ids"],
                            seeds=np.asarray(SEEDS), beta=np.asarray(BETA_ORDER),
                            hidden=np.asarray(hidden_parameters), output=np.asarray(output_parameters))
    audit = {
        "method": "E25 one-hidden-layer nonlinear head sensitivity",
        "shape": [16, WIDTH, 20],
        "nonlinearity": "tanh",
        "parameter_count_per_head": 17 * WIDTH + (WIDTH + 1) * 20,
        "train_questions": 600,
        "train_tied_questions": 306,
        "e24_cached_input_questions": 1196,
        "loss": "exact symmetric TSPO+ with retained 600 denominator and all coefficients including biases L2 penalized",
        "regularization": LAMBDA,
        "beta_order": list(BETA_ORDER),
        "gamma": 0,
        "adam_full_batch_steps": STEPS,
        "adam_precision": "float64",
        "adam_learning_rate_cosine_schedule": [LR_MAX, LR_MIN],
        "adam_exact_max_subgradient": "uniform on exact equal maxima",
        "checkpoint_selection": "lowest train objective only, earliest exact tie",
        "output_refinement": "frozen trained hidden state plus exact original conditional Clarabel QP",
        "solver_scope": "conditional output solution; the joint nonlinear objective is nonconvex",
        "optimization_seeds": list(SEEDS),
        "primary_optimization_seed": SEEDS[0],
        "train_design_max_abs_recovery_error": recovered,
        "central_difference_gradient_max_abs_error": max_gradient_error,
        "source_sha256": digests,
        "mlp_source_sha256": sha(Path(__file__)),
        "private_mlp_output_sha256": sha(args.output),
        "evaluation_reward_reads": 0,
        "new_encoder_forwards": 0,
        "new_llm_calls": 0,
        "fits": fits,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"fit_count": len(fits), "optimizer_seeds": SEEDS, "gradient_check": max_gradient_error,
                      "e24_cached_scores_shape": list(np.asarray(scores).shape)}))


if __name__ == "__main__":
    main()
