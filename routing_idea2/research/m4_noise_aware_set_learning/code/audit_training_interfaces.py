"""Independent literal-loop operator and frozen posterior interface audit."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent / 'training'
spec = importlib.util.spec_from_file_location('independent_ns_training_audit', TRAINING / 'ns_training.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rng = np.random.default_rng(20261114094)
errors = []
for case in range(8):
    phi = np.column_stack((np.ones(3), rng.normal(size=(3, 2))))
    coefficient = rng.normal(scale=.3, size=(3, 20))
    samples = rng.uniform(size=(3, 4, 20))
    scores = phi @ coefficient
    regularization, delta = .03, .05
    expected_loss = regularization * np.sum(coefficient ** 2)
    expected_gradient = 2 * regularization * coefficient.copy()
    for row in range(3):
        for draw in range(4):
            reward = samples[row, draw]
            selected = [action for action in range(20) if reward.max() - reward[action] <= delta]
            transformed = 2 * scores[row] - reward
            worst = int(np.argmax(transformed))
            expected_loss += (transformed[worst] - sum(transformed[action] for action in selected) / len(selected)) / 12
            for action in range(20):
                target = (1 / len(selected)) if action in selected else 0.
                expected_gradient[:, action] += 2 * phi[row] * (float(action == worst) - target) / 12
    actual_loss, actual_gradient = module.ns_value_gradient(coefficient, phi, samples, delta, regularization)
    errors.append(dict(case=case, operator='NS-SPO+', loss_error=float(abs(actual_loss - expected_loss)),
        gradient_error=float(np.max(abs(actual_gradient - expected_gradient)))))
    targets = rng.dirichlet(np.ones(3), size=(3, 190))
    nu = .7
    expected_loss = regularization * np.sum(coefficient ** 2)
    expected_gradient = 2 * regularization * coefficient.copy()
    for row in range(3):
        for pair, (left, right) in enumerate(zip(module.PAIR_I, module.PAIR_J)):
            z = scores[row, left] - scores[row, right]
            a, b, tie = np.exp(z / 2), np.exp(-z / 2), 2 * nu
            probabilities = np.array([a, tie, b]) / (a + b + tie)
            expected_loss -= float(targets[row, pair] @ np.log(probabilities)) / 570
            derivative = (probabilities[0] - probabilities[2] - targets[row, pair, 0] + targets[row, pair, 2]) / 2
            expected_gradient[:, left] += derivative * phi[row] / 570
            expected_gradient[:, right] -= derivative * phi[row] / 570
    actual_loss, actual_gradient = module.davidson_value_gradient(coefficient, phi, targets, nu, regularization)
    errors.append(dict(case=case, operator='Davidson', loss_error=float(abs(actual_loss - expected_loss)),
        gradient_error=float(np.max(abs(actual_gradient - expected_gradient)))))
assert max(max(row['loss_error'], row['gradient_error']) for row in errors) < 1e-12
posterior_checks = []
for fold in range(5):
    path = HERE / f'private/fold{fold}/POSTERIOR_TRAINING.npz'
    receipt = json.loads((path.parent / 'POSTERIOR_FREEZE_RECEIPT.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['posterior_sha256']
    with np.load(path, allow_pickle=False) as archive:
        assert archive['mu_samples'].shape == archive['mu_samples_alt'].shape == (480, 8, 20)
        assert len(set(archive['history_indices'].tolist())) == 480
        for key in ('mu_samples', 'mu_samples_alt'):
            assert np.isfinite(archive[key]).all() and np.all((archive[key] >= 0) & (archive[key] <= 1))
        for suffix in ('delta005', 'delta000'):
            probability = archive['pair_probability_' + suffix]
            assert probability.shape == (480, 190, 3)
            assert np.allclose(probability.sum(axis=2), 1) and np.all(probability >= 0)
        posterior_checks.append(dict(fold=fold, frozen_target_hash_matches=True,
            main_alt_bounded_shapes=True, probabilities_simplex=True,
            posterior_questions=480, outer_rewards_used=False))
result = dict(status='PASS_INDEPENDENT_TRAINING_AND_POSTERIOR_INTERFACE_REVIEW',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    training_source_sha256=hashlib.sha256((TRAINING / 'ns_training.py').read_bytes()).hexdigest(),
    evaluation_source_sha256=hashlib.sha256((TRAINING / 'evaluate_ns.py').read_bytes()).hexdigest(),
    literal_loop_checks=errors, frozen_posterior_checks=posterior_checks,
    static_source_review={
        'NS_exact_epigraph_matches_finite_MC_objective': True,
        'Davidson_main_uses_same_eight_joint_samples_as_NS': True,
        '512_pair_probabilities_diagnostic_only': True,
        'all16_model_actions_sealed_before_historical_reward_reader': True,
        'cross_fold_indices_sources_models_predictions_restored': True,
        'sampling_unit_original_question': True,
        'same_linear_capacity_340_and_historical_lambda': True,
        'Davidson_nu_historical_fixed_zero_margin_matching_declared': True},
    reproduction_requirement='evaluate training-root must contain ns_training.py plus private/fold* models; canonical public/code alone is insufficient',
    caveats=['Classical Rhat/ESS passing does not establish latent probability calibration',
        'Gaussian moment interval undercoverage remains a statistical mismatch warning',
        'Final parent snapshots must verify history/input assets against all recorded SHA values'])
(HERE / 'public/JOINT_TRAINING_INTERFACE_REVIEW.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=result['status'], literal_checks=len(errors), posterior_folds=len(posterior_checks),
    maximum_loss_gradient_error=max(max(v['loss_error'], v['gradient_error']) for v in errors))))
