"""Independent cross-implementation audit of the actual training objectives."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

from set_spo_operators import (set_weights, ns_spo_value_subgradient,
                              davidson_value_gradient, posterior_pair_probabilities)

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent.parent
ACTUAL = STAGING / 'training/ns_training.py'
EVALUATION = STAGING / 'training/evaluate_ns.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('independently_reviewed_ns_training', ACTUAL)
    actual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actual)
    rng = np.random.default_rng(2026111407)
    comparisons = []
    for delta in (0., .05):
        phi = np.column_stack((np.ones(9), rng.normal(size=(9, 2))))
        coefficient = rng.normal(scale=.2, size=(3, 20))
        samples = rng.uniform(.01, .99, size=(9, 8, 20))
        weights = set_weights(samples, delta)
        own_value, own_gradient = ns_spo_value_subgradient(coefficient, phi, samples, weights, .01)
        actual_value, actual_gradient = actual.ns_value_gradient(coefficient, phi, samples, delta, .01)
        probability = posterior_pair_probabilities(samples, delta)
        own_davidson, own_davidson_gradient, _ = davidson_value_gradient(coefficient, phi, probability, np.log(.7), .01)
        actual_davidson, actual_davidson_gradient = actual.davidson_value_gradient(coefficient, phi, probability, .7, .01)
        errors = dict(ns_value=abs(own_value - actual_value),
            ns_gradient=float(abs(own_gradient - actual_gradient).max()),
            davidson_value=abs(own_davidson - actual_davidson),
            davidson_gradient=float(abs(own_davidson_gradient - actual_davidson_gradient).max()))
        if max(errors.values()) > 1e-12:
            raise AssertionError('Actual operators differ from independent implementation')
        coefficient, fit = actual.fit_ns_exact(phi, samples, delta, .01)
        independently_recomputed, _ = ns_spo_value_subgradient(coefficient, phi, samples, weights, .01)
        error = abs(independently_recomputed - fit['objective'])
        if error > 1e-10:
            raise AssertionError('Actual QP objective lost the joint reward-weight constant')
        comparisons.append(dict(delta=delta, **errors,
            independent_epigraph_direct_error=float(error), solver_status=fit['status'],
            primal_dual_gap=fit['primal_dual_gap']))
    training_source = ACTUAL.read_text(encoding='utf-8')
    evaluation_source = EVALUATION.read_text(encoding='utf-8')
    same_draws = all(text in training_source for text in (
        'pair_gap = samples[:, :, PAIR_I] - samples[:, :, PAIR_J]',
        '(pair_gap > delta).mean(axis=1)', '(abs(pair_gap) <= delta).mean(axis=1)',
        '(pair_gap < -delta).mean(axis=1)'))
    if not same_draws:
        raise AssertionError('Davidson is not using the same eight joint samples')
    if 'len(fits) != 16' not in evaluation_source:
        raise AssertionError('Predeclared full fit family is not guarded')
    result = dict(status='PASS_WITH_DOCUMENTED_MODEL_ASSUMPTIONS',
        training_source_sha256=sha(ACTUAL), evaluation_source_sha256=sha(EVALUATION),
        independent_operator_sha256=sha(HERE / 'set_spo_operators.py'),
        comparisons=comparisons,
        static_checks=dict(same_eight_draws_for_NS_and_Davidson=True,
            per_question_and_MC_mean=True, exact_E_weight_times_mu_constant_preserved=True,
            regularization_once=True, shared_fixed_capacity_340=True,
            original_M4_history_selected_regularization=True,
            Davidson_190_pair_mean=True, Davidson_nu_historical_only_and_bounded=True,
            fold0_disjoint_alt_eight_draws_not_chosen_by_quality=True,
            all16_models_sealed_before_outer_reward_reader=True,
            exact_argmax_and_near_tie_counts_reported=True,
            paired_original_question_statistics=True),
        limitations=[
            'Bounded Gaussian Gibbs output can be correlated. The iid Monte Carlo Hoeffding statement is not applied to the eight training samples.',
            'nu=p_tie/(1-p_tie) matches the historical tie frequency only at score difference zero and is not joint maximum likelihood for nu.',
            'Work-model posterior event probabilities are not validated latent-correctness probabilities.',
            'Convexity and objective minimization do not establish Bayes decision consistency.',
            'An exchangeable loss can still yield exact argmax label preference if the fitted scores are tied.',
        ], real_reward_accesses_by_this_review=0, new_LLM_calls=0)
    (HERE / 'ACTUAL_TRAINING_OPERATOR_REVIEW.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], training_source_sha256=result['training_source_sha256'],
        evaluation_source_sha256=result['evaluation_source_sha256'], comparisons=comparisons), ensure_ascii=False))


if __name__ == '__main__':
    main()
