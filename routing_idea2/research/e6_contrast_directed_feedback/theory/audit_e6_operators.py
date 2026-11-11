"""Independent exact-subset and training-isolation audits for E6."""

import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from scipy.optimize import minimize


ROOT = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由')) / 'research/e6_contrast_directed_feedback'
STAGING = ROOT / 'public'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def audit_subset():
    sampling = module('independent_sampling_target', STAGING / 'c/finite_pool_sampling.py')
    rng = np.random.default_rng(20261113)
    maximum_inclusion_error = 0.
    maximum_unbiased_error = 0.
    maximum_variance_error = 0.
    zero_support_cases = 0
    cases = 0
    for n in range(3, 10):
        for _ in range(12):
            count = int(rng.integers(1, n))
            costs = rng.integers(0, 5, n)
            minimum_cost = int(np.sort(costs)[:count].sum())
            cap = int(rng.integers(minimum_cost, int(np.sort(costs)[-count:].sum()) + 1))
            odds = np.exp(rng.uniform(-4, 4, n))
            design = sampling.CappedSubsetDesign(odds, count, costs, cap)
            masks, masses = [], []
            for chosen in itertools.combinations(range(n), count):
                mask = np.zeros(n, bool)
                mask[list(chosen)] = True
                if costs @ mask <= cap:
                    masks.append(mask)
                    masses.append(np.prod(odds[mask]))
            masks, masses = np.asarray(masks), np.asarray(masses)
            masses /= masses.sum()
            pi = masses @ masks
            maximum_inclusion_error = max(maximum_inclusion_error, float(np.max(abs(pi - design.pi))))
            values = rng.normal(size=n)
            if np.any(pi == 0):
                zero_support_cases += 1
                values[pi == 0] = 0.
            scaled = np.divide(values, pi, out=np.zeros(n), where=pi > 0)
            estimates = masks @ scaled / n
            mean = masses @ estimates
            maximum_unbiased_error = max(maximum_unbiased_error, abs(float(mean - values.mean())))
            second = (masks.T * masses) @ masks
            general_variance = scaled @ (second - np.outer(pi, pi)) @ scaled / n**2
            exact_variance = masses @ (estimates - mean)**2
            maximum_variance_error = max(maximum_variance_error, abs(float(general_variance - exact_variance)))
            for _ in range(30):
                sampled = design.sample(rng.random(n + 1))
                assert sampled.sum() == count
                assert costs @ sampled <= cap
            cases += 1
    assert maximum_inclusion_error < 1e-11
    assert maximum_unbiased_error < 1e-11
    assert maximum_variance_error < 1e-10
    return dict(status='PASS', cases=cases, independent_exhaustive_inclusion_max_error=maximum_inclusion_error,
                conditional_HT_bias_max_error=maximum_unbiased_error,
                general_design_variance_max_error=maximum_variance_error,
                zero_support_cases=zero_support_cases,
                zero_support_interpretation='These qualification cases require zero contribution on excluded coordinates; actual experiment must prove positivity or input-only fallback.')


def audit_logistic_and_isolation():
    target = ROOT / 'public/a/run_e6_a.py'
    learning = module('independent_learning_target', target)
    rng = np.random.default_rng(20261114)
    max_objective_difference = 0.
    max_predict_difference = 0.
    max_weight_error = 0.
    for auxiliary in (False, True):
        n, d, repeats = 15, 46, 3
        x = rng.normal(size=(n, 1, d))
        current = rng.choice([-1., 0., 1.], size=(n, repeats), p=[.1, .7, .2])
        aux_x = rng.normal(size=(n, 19, d)) if auxiliary else None
        aux_d = rng.choice([-1., 0., 1.], size=(n, 19, repeats)) if auxiliary else None
        features, positive, negative = learning.weighted_training(x, current, aux_x, aux_d)
        if auxiliary:
            expected = .5 * np.mean(current != 0) + .5 * np.mean(aux_d != 0)
        else:
            expected = np.mean(current != 0)
        max_weight_error = max(max_weight_error, abs(float(positive.sum() + negative.sum() - expected)))
        for lam in (.01, .1, 1.):
            head = learning.fit_head(features, positive, negative, lam, n)
            design = np.column_stack((np.ones(len(features)), (features - head['mean']) / head['scale']))
            def objective(beta):
                score = design @ beta
                return positive @ np.logaddexp(0, -score) + negative @ np.logaddexp(0, score) + lam * np.dot(beta, beta)
            independent = minimize(objective, np.zeros(d + 1), method='BFGS', options={'gtol': 1e-9, 'maxiter': 500})
            max_objective_difference = max(max_objective_difference, abs(float(objective(independent.x) - objective(head['beta']))))
            max_predict_difference = max(max_predict_difference, float(np.max(abs(design @ (independent.x - head['beta'])))))
    assert max_objective_difference < 1e-10
    assert max_predict_difference < 2e-5
    data, _ = learning.load_train()
    with np.load(ROOT / 'private/a/FROZEN_A_f0.npz') as original:
        expected_masks = original['masks'].copy()
        expected_scores = original['scores'].copy()
        expected_pca = original['pca_components'].copy()
        expected_combo = original['combo_embedding'].copy()
        E = original['E'].copy()
    poisoned = dict(data)
    poisoned['Y'] = data['Y'].copy()
    poisoned['Y'][E] = np.nan
    learning.PRIVATE = ROOT / 'private/audit_replays/training_nan'
    learning.train_fold(0, poisoned)
    with np.load(learning.PRIVATE / 'FROZEN_A_f0.npz') as changed:
        assert np.array_equal(expected_masks, changed['masks'])
        assert np.array_equal(expected_scores, changed['scores'])
        assert np.array_equal(expected_pca, changed['pca_components'])
        assert np.array_equal(expected_combo, changed['combo_embedding'])
    return dict(status='PASS', independent_logistic_fits=6, objective_difference_max=max_objective_difference,
                prediction_difference_max=max_predict_difference, normalization_error_max=max_weight_error,
                outer_fold=0, poisoned_E_questions=len(E), poison='all E rewards replaced by NaN',
                frozen_masks_exactly_identical=True, frozen_scores_exactly_identical=True,
                F_PCA_exactly_identical=True, F_combo_embedding_exactly_identical=True,
                scope='Training read isolation for E reward array and independent convex objective qualification, not OS file-access forensics')


def main():
    result = dict(subset_sampling=audit_subset(), logistic_and_reward_isolation=audit_logistic_and_isolation(), status='PASS')
    path = ROOT / 'public/theory/INDEPENDENT_OPERATOR_AUDIT.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
