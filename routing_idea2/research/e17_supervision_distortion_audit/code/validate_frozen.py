"""Independently qualify frozen E17 heads without opening outer rewards.

The source receipt must be written before running the audit. Historical labels
are sliced to H480 before they are used. No outer accuracy is computed here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from e17_common import (
    AUDIT, E8, MC_COUNTS, METHODS, NS, PRIVATE, PUBLIC,
    array_hash, input_packet, labels, load_npz, posterior_path, read_json,
    sha256, timing, verify_protocol, write_json,
)
import numpy as np


SOURCE_RECEIPT = PUBLIC / "E17_VALIDATOR_SOURCE_FREEZE.json"
FEATURE_KEYS = ("pca_mean", "pca_components", "scaler_mean", "scaler_scale")


class Checks:
    def __init__(self):
        self.count = 0
        self.maximum_errors = {}

    def require(self, condition, label):
        self.count += 1
        if not bool(condition):
            raise AssertionError(label)

    def equal(self, actual, expected, label):
        self.require(np.array_equal(actual, expected), label)

    def close(self, actual, expected, tolerance, label, group=None):
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.require(actual.shape == expected.shape, label + " shape")
        self.require(np.isfinite(actual).all() and np.isfinite(expected).all(), label + " finite")
        error = float(np.max(np.abs(actual - expected)))
        self.require(error <= tolerance, label + " error " + str(error))
        if group is not None:
            self.maximum_errors[group] = max(self.maximum_errors.get(group, 0.0), error)
        return error


def direct_spo(coefficient, phi, samples, regularization, uniform=False):
    """Independent direct objective with one penalty and mean over n and B."""
    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = samples[:, None, :]
    scores = phi @ coefficient
    if uniform:
        target = samples == samples.max(axis=2, keepdims=True)
        target = target / target.sum(axis=2, keepdims=True)
    else:
        target = np.eye(samples.shape[2])[samples.argmax(axis=2)]
    transformed = 2 * scores[:, None, :] - samples
    data = np.mean(transformed.max(axis=2) - np.sum(target * transformed, axis=2))
    return float(data + regularization * np.sum(coefficient * coefficient))


def calibration(phi, coefficient, reward):
    prediction = phi @ coefficient
    prediction -= prediction.mean(axis=1, keepdims=True)
    centered = reward - reward.mean(axis=1, keepdims=True)
    energy = float(np.sum(prediction * prediction))
    scale = max(0.0, float(np.sum(prediction * centered)) / energy) if energy > 1e-20 else 0.0
    return scale, energy


def check_calibration(checks, saved, phi, coefficient, reward, label):
    scale, energy = calibration(phi, coefficient, reward)
    checks.close(saved["nonnegative_scale"], scale, 1e-10, label + " scale", "calibration")
    checks.close(saved["score_energy"], energy, 1e-10, label + " energy", "calibration")


def check_old_ns(checks, fold, packet, post, regularization):
    folder = NS / f"private/runtime/training/private/fold{fold}/NS_delta0"
    receipt = read_json(folder / "FROZEN_RECEIPT.json")
    log = read_json(folder / "FIT_LOG.json")
    model = load_npz(folder / "MODEL.npz")
    checks.require(receipt["source_sha256"] == sha256(NS / "public/code/ns_training.py"), "NS source")
    for name, field in (("MODEL.npz", "model_sha256"), ("FIT_LOG.json", "fit_log_sha256"),
                        ("PREDICTIONS_FROZEN.npz", "predictions_sha256")):
        checks.require(receipt[field] == sha256(folder / name), "NS source file " + name)
    checks.require(receipt["posterior_sha256"] == sha256(posterior_path(fold)), "NS actual draw source")
    history_path = E8 / f"private/qualification/fold_packs/fold{fold}/history480.npz"
    inputs_path = E8 / f"private/qualification/fold_packs/fold{fold}/evaluation_inputs_only120.npz"
    checks.require(receipt["history_sha256"] == sha256(history_path), "NS historical question source")
    checks.require(receipt["evaluation_inputs_sha256"] == sha256(inputs_path), "NS outer input source")
    checks.require(receipt["evaluation_rewards_opened"] is False, "NS outer reward boundary")
    checks.require(log["delta"] == 0 and log["samples_per_question"] == 8, "NS delta0/B8")
    checks.close(log["regularization"], regularization, 1e-15, "NS regularization")
    checks.require(log["status"] == "Solved", "NS solver solved")
    for key in FEATURE_KEYS:
        checks.equal(model[key], packet[key], "NS feature map " + key)
    predictions = load_npz(folder / "PREDICTIONS_FROZEN.npz")
    checks.equal(predictions["evaluation_indices"], packet["E"], "NS original E identity")
    checks.close(predictions["scores"], packet["phi_e"] @ model["coefficient"], 1e-12,
                 "NS original score restoration", "cached_ns_scores")
    draws = post["mu_samples"]
    unique = (draws == draws.max(axis=2, keepdims=True)).sum(axis=2) == 1
    checks.require(unique.all(), "B8 NS uniform and plain champion equivalence")
    value = direct_spo(model["coefficient"], packet["phi_h"], draws, regularization)
    checks.close(value, log["objective"], 1e-7, "NS cached exact objective", "cached_ns_objective")
    return model["coefficient"]


def check_posterior(checks, fold, packet):
    post = load_npz(posterior_path(fold))
    checks.equal(post["history_indices"], packet["H"], "posterior H identity")
    for key in ("mu_samples", "mu_samples_alt"):
        checks.require(post[key].shape == (480, 8, 20), "posterior sample shape")
        checks.require(np.isfinite(post[key]).all() and np.all((post[key] >= 0) & (post[key] <= 1)), "posterior bounded")
    checks.require(not post["fallback_mask"].any() and not post["numeric_failed"].any(), "posterior successful rows")
    receipt = read_json(posterior_path(fold).with_name("POSTERIOR_FREEZE_RECEIPT.json"))
    checks.require(receipt["posterior_sha256"] == sha256(posterior_path(fold)), "posterior immutable source")
    checks.require(receipt["outer_evaluation_rewards_opened"] is False, "posterior no outer rewards")
    checks.require(receipt["hyperparameter_crossfit"] is True, "posterior inner crossfit")
    history_path = E8 / f"private/qualification/fold_packs/fold{fold}/history480.npz"
    checks.require(receipt["input_history_sha256"] == sha256(history_path), "posterior source H480")
    for name, digest in receipt["source_sha256"].items():
        checks.require(digest == sha256(NS / "public/code" / name), "posterior generator source " + name)
    return post


def qualify():
    cpu, wall = time.process_time(), time.perf_counter()
    checks = Checks()
    frozen_source = read_json(SOURCE_RECEIPT)
    checks.require(frozen_source["validator_sha256"] == sha256(__file__), "validator source freeze")
    protocol = verify_protocol()
    protocol_hash = sha256(PUBLIC / "E17_PROTOCOL.json")
    checks.require(frozen_source["training_protocol_sha256"] == protocol_hash, "validator protocol context")
    checks.require(protocol["confidence_intervals"] == 0 and protocol["bootstrap_runs"] == 0, "no CI/bootstrap contract")
    final = read_json(PRIVATE / "ALL_PREDICTIONS_FROZEN.json")
    checks.require(final["protocol_sha256"] == protocol_hash, "global frozen context")
    checks.require(final["status"] == "FROZEN_BEFORE_OUTER_REWARD_EVALUATION", "global frozen status")
    permutation_file = PRIVATE / "PERMUTATIONS_FROZEN.npz"
    checks.require(sha256(permutation_file) == protocol["permutation_sha256"], "permutation source hash")
    permutations = load_npz(permutation_file)["permutations"]
    checks.require(permutations.shape == (101, 20), "100 random plus identity permutation budget")
    checks.equal(permutations[0], np.arange(20), "identity permutation")
    checks.require(len(set(map(tuple, permutations))) == 101, "unique permutations")
    rng = np.random.default_rng(20261115017)
    expected = [np.arange(20)]
    while len(expected) < 101:
        proposed = rng.permutation(20)
        if not any(np.array_equal(proposed, previous) for previous in expected):
            expected.append(proposed)
    checks.equal(permutations, np.asarray(expected), "predeclared permutation RNG")

    expected_files, fold_summaries, all_evaluation = [], [], []
    for fold in range(5):
        packet = input_packet(fold)
        history, evaluation = packet["H"], packet["E"]
        checks.require(history.shape == (480,) and evaluation.shape == (120,), "fold question counts")
        checks.require(len(np.unique(history)) == 480 and len(np.unique(evaluation)) == 120, "fold unique IDs")
        checks.equal(np.sort(np.r_[history, evaluation]), np.arange(600), "fold H/E disjoint partition")
        all_evaluation.append(evaluation)
        reward = labels(history)
        checks.require(reward.shape == (480, 20), "H-only target shape")
        m4_log = read_json(E8 / f"private/training/fold{fold}/M4/FIT_LOG.json")
        ridge_log = read_json(E8 / f"private/training/fold{fold}/M0_tuned/FIT_LOG.json")
        regularization = float(m4_log["final_diagnostics"]["regularization"])
        ridge_alpha = 100 * float(ridge_log["selected_parameters"]["lambda_multiplier"]) / 3
        post = check_posterior(checks, fold, packet)
        ns_coefficient = check_old_ns(checks, fold, packet, post, regularization)
        for index, permutation in enumerate(permutations):
            folder = PRIVATE / f"permutations/fold{fold}/p{index:03d}"
            model_path, receipt_path = folder / "MODEL.npz", folder / "FROZEN.json"
            expected_files.append(model_path)
            saved, receipt = load_npz(model_path), read_json(receipt_path)
            checks.require(receipt["protocol_sha256"] == protocol_hash, "permutation protocol")
            checks.require(receipt["sha256"] == sha256(model_path), "permutation model hash")
            checks.require(receipt["fold"] == fold and receipt["permutation_index"] == index, "permutation context")
            checks.equal(saved["permutation"], permutation, "renaming identity")
            checks.equal(saved["history"], history, "permutation H")
            checks.equal(saved["evaluation"], evaluation, "permutation E")
            checks.require(saved["weights"].shape == (3, 17, 20), "permutation coefficient shape")
            checks.require(saved["scores"].shape == (3, 120, 20), "permutation score shape")
            for method_index, method in enumerate(METHODS):
                coefficient = saved["weights"][method_index]
                diagnostic = receipt["diagnostics"][method_index]
                checks.require(diagnostic["method"] == method, "method axis")
                checks.close(packet["phi_e"] @ coefficient, saved["scores"][method_index], 1e-12,
                             "mapped W to E score", "W_to_scores")
                check_calibration(checks, receipt["calibrations"][method_index], packet["phi_h"],
                                  coefficient, reward, "permutation calibration")
                renamed = coefficient[:, permutation]
                if method == "Ridge":
                    normal = packet["phi_h"].T @ packet["phi_h"] + ridge_alpha * np.eye(17)
                    residual = normal @ renamed - packet["phi_h"].T @ reward[:, permutation]
                    checks.close(residual, np.zeros_like(residual), 1e-8, "Ridge exact mean-unit normal equation", "ridge_normal")
                    checks.close(diagnostic["alpha"], ridge_alpha, 1e-15, "Ridge alpha raw3 / 3")
                else:
                    direct = direct_spo(renamed, packet["phi_h"], reward[:, permutation], regularization,
                                        uniform=method == "TSPO_uniform")
                    checks.close(direct, diagnostic["objective"], 1e-7, "permuted training loss", "spo_objective")
                    checks.close(diagnostic["regularization"], regularization, 1e-15, "SPO frozen regularization")
                    checks.require(diagnostic["status"] == "Solved", "SPO solver status")
                if index == 0:
                    if method == "Ridge":
                        old_path = E8 / f"private/training/fold{fold}/M0_tuned/model_state.npz"
                    else:
                        rule = "lexicographic" if method == "SPO_coordinate" else "uniform"
                        old_path = AUDIT / f"private/distribution/models/fold{fold}_full_spo_{rule}.npz"
                    old_coefficient = load_npz(old_path)["coefficient"]
                    checks.close(packet["phi_e"] @ coefficient, packet["phi_e"] @ old_coefficient,
                                 1e-7, "canonical historical recovery", "canonical_recovery")
            # Undoing renaming is a pure column permutation, checked without E labels.
            checks.close(saved["weights"][:, :, permutation][:, :, np.argsort(permutation)],
                         saved["weights"], 0, "physical/numeric inverse mapping")

        pool = np.concatenate((post["mu_samples"], post["mu_samples_alt"]), axis=1)
        for count in MC_COUNTS:
            samples = pool[:, :count]
            sample_hash = array_hash(samples)
            for operator in ("LossThenMean", "MeanThenLoss"):
                model_path = PRIVATE / f"loss_order/fold{fold}/B{count}/{operator}.npz"
                expected_files.append(model_path)
                saved, receipt = load_npz(model_path), read_json(model_path.with_suffix(".json"))
                checks.require(receipt["protocol_sha256"] == protocol_hash, "order protocol context")
                checks.require(receipt["sha256"] == sha256(model_path), "order model hash")
                checks.require(receipt["fold"] == fold and receipt["MC_samples"] == count and receipt["operator"] == operator,
                               "order fold/count/operator context")
                checks.require(receipt["sample_sha256"] == sample_hash and saved["sample_hash"].item() == sample_hash,
                               "order actual joint draw hash")
                checks.equal(saved["history"], history, "order H")
                checks.equal(saved["evaluation"], evaluation, "order E")
                coefficient = saved["weights"]
                checks.require(coefficient.shape == (17, 20), "order coefficient shape")
                checks.close(packet["phi_e"] @ coefficient, saved["scores"], 1e-12, "order W to score", "W_to_scores")
                check_calibration(checks, receipt["calibration"], packet["phi_h"], coefficient, reward, "order calibration")
                direct_samples = samples if operator == "LossThenMean" else samples.mean(axis=1)
                direct = direct_spo(coefficient, packet["phi_h"], direct_samples, regularization)
                if count == 8 and operator == "LossThenMean":
                    checks.equal(coefficient, ns_coefficient, "exact NS B8 cache coefficient")
                    checks.require(receipt["reused"] and receipt["reuse_role"] == "frozen NS_delta0", "NS reuse label")
                    source = NS / f"private/runtime/training/private/fold{fold}/NS_delta0/MODEL.npz"
                    checks.require(receipt["diagnostics"]["reused_source_sha256"] == sha256(source), "B8 source receipt")
                elif count == 1 and operator == "MeanThenLoss":
                    source = model_path.with_name("LossThenMean.npz")
                    checks.equal(coefficient, load_npz(source)["weights"], "B1 operator exact identity")
                    checks.require(receipt["diagnostics"]["reused_source_sha256"] == sha256(source), "B1 source receipt")
                else:
                    diagnostic = receipt["diagnostics"]
                    checks.close(direct, diagnostic["objective"], 1e-7, "order 1/(nB) objective", "order_objective")
                    checks.require(diagnostic["status"] == "Solved", "order solver status")
                    checks.require(diagnostic["samples_per_question"] == (count if operator == "LossThenMean" else 1), "order normalized draw count")
                    checks.close(diagnostic["regularization"], regularization, 1e-15, "order regularization once")
                checks.close(receipt["mean_MC_vs_empirical_MAE"], np.abs(samples.mean(axis=1) - reward).mean(),
                             1e-12, "MC empirical diagnostic")
                checks.close(receipt["mean_MC_vs_512_summary_MAE"], np.abs(samples.mean(axis=1) - post["posterior_mean"]).mean(),
                             1e-12, "MC frozen512 summary diagnostic")
        fold_summaries.append(dict(fold=fold, history_questions=480, evaluation_inputs=120,
                                   permutation_files=101, loss_order_files=10, actual_primary_draws=8,
                                   outer_reward_reads=0, history_target_rows=480))
    checks.equal(np.sort(np.concatenate(all_evaluation)), np.arange(600), "five outer folds cover 600 original questions")
    actual_paths = set((PRIVATE / "permutations").rglob("MODEL.npz")) | set((PRIVATE / "loss_order").rglob("*.npz"))
    checks.require(actual_paths == set(expected_files), "no missing or extra frozen model artifacts")
    final_files = {Path(item["path"]): item["sha256"] for item in final["files"]}
    checks.require(set(final_files) == set(expected_files), "global manifest exact 555-file set")
    for path in expected_files:
        checks.require(final_files[path] == sha256(path), "global file digest")
    result = dict(status="PASS", validator_sha256=sha256(__file__), training_protocol_sha256=protocol_hash,
                  frozen_predictions_sha256=sha256(PRIVATE / "ALL_PREDICTIONS_FROZEN.json"),
                  checks=checks.count, maximum_errors=checks.maximum_errors, folds=fold_summaries,
                  permutation_files=505, permutation_heads=1515, loss_order_files=50,
                  random_permutations=100, identity_controls=1, history_rewards_only=True,
                  outer_reward_reads=0, fresh400_reward_reads=0, sealed200_reward_reads=0,
                  confidence_intervals=0, bootstrap_runs=0, new_physical_llm_calls=0,
                  scope="Frozen artifact and exact-objective qualification, before outer reward evaluation",
                  **timing(cpu, wall))
    write_json(PRIVATE / "FROZEN_ARTIFACT_QUALIFICATION.json", result)
    write_json(PUBLIC / "FROZEN_ARTIFACT_QUALIFICATION.json", result)
    print("E17_FROZEN_ARTIFACT_QUALIFICATION_PASS", checks.count, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze-source", "qualify"))
    args = parser.parse_args()
    if args.command == "freeze-source":
        write_json(SOURCE_RECEIPT, dict(status="FROZEN_BEFORE_OUTER_REWARD_EVALUATION",
                                      validator_sha256=sha256(__file__),
                                      training_protocol_sha256=sha256(PUBLIC / "E17_PROTOCOL.json")))
        print("E17_VALIDATOR_SOURCE_FROZEN", flush=True)
    else:
        qualify()


if __name__ == "__main__":
    main()
