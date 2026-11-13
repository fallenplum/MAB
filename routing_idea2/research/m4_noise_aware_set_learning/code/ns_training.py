"""Fixed-capacity NS-SPO+ and posterior-target Davidson mechanism fits.

All feature and regularization parameters come from frozen E8 M4 histories.
Only history480 targets are accepted while fitting. Evaluation is a separate
entry point that refuses to run until every predefined outer prediction exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp

ASSETS = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
E8 = ASSETS / 'research/e8_decision_focused_combination_learning'
sys.path[:0] = [str(E8 / 'private/dependencies'), str(E8 / 'public/qualification')]
from e8_assets import FeatureMap, matched_mean_loss_lambda
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import clarabel

K = 20
PAIR_I, PAIR_J = np.array(list(itertools.combinations(range(K), 2))).T
INCIDENCE = np.zeros((190, K))
INCIDENCE[np.arange(190), PAIR_I] = 1.
INCIDENCE[np.arange(190), PAIR_J] = -1.
CLARABEL_SETTINGS = dict(tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9,
    max_iter=1000, max_threads=1, verbose=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item.item()) + '\n', encoding='utf-8')


def archive(path):
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key].copy() for key in value.files}


def feature_map(state):
    pca = PCA(16, svd_solver='full')
    pca.components_, pca.mean_ = state['pca_components'], state['pca_mean']
    pca.n_features_in_, pca.n_components_ = 384, 16
    pca.explained_variance_, pca.whiten = np.ones(16), False
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = state['scaler_mean'], state['scaler_scale']
    scaler.n_features_in_ = 16
    return FeatureMap(pca, scaler)


def near_optimal_weights(samples, delta):
    if delta < 0 or not np.isfinite(samples).all():
        raise ValueError('Finite samples and nonnegative practical delta required')
    mask = samples >= samples.max(axis=-1, keepdims=True) - delta
    return mask / mask.sum(axis=-1, keepdims=True)


def ns_value_gradient(coefficient, phi, samples, delta, regularization):
    """Exact finite-Monte-Carlo objective and one valid max subgradient."""
    n, repetitions, actions = samples.shape
    if actions != K or len(phi) != n:
        raise ValueError('One joint twenty-action sample block per question required')
    scores = phi @ coefficient
    weights = near_optimal_weights(samples, delta)
    transformed = 2 * scores[:, None, :] - samples
    chosen = transformed.argmax(axis=2)
    maximum = np.take_along_axis(transformed, chosen[:, :, None], axis=2)[:, :, 0]
    loss = np.mean(maximum - np.sum(weights * transformed, axis=2))
    direction = -weights.copy()
    direction[np.arange(n)[:, None], np.arange(repetitions)[None, :], chosen] += 1.
    gradient = (2 / n) * phi.T @ direction.mean(axis=1) + 2 * regularization * coefficient
    return float(loss + regularization * np.sum(coefficient ** 2)), gradient


def fit_ns_exact(phi, samples, delta, regularization):
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    n, repetitions, actions = samples.shape
    dimension, sample_count = phi.shape[1], n * repetitions
    if actions != K or samples.min() < 0 or samples.max() > 1:
        raise ValueError('Joint bounded probability samples required')
    flat_phi = np.repeat(phi, repetitions, axis=0)
    flat_samples = samples.reshape(sample_count, K)
    weights = near_optimal_weights(flat_samples, delta)
    coefficient_count = dimension * K
    quadratic = sparse.diags(np.r_[np.full(coefficient_count, 2 * regularization),
        np.zeros(sample_count)], format='csc')
    linear = np.r_[(-(2 / sample_count) * flat_phi.T @ weights).reshape(-1, order='F'),
        np.full(sample_count, 1 / sample_count)]
    action_blocks = sparse.kron(sparse.eye(K, format='csc'), 2 * sparse.csc_matrix(flat_phi), format='csc')
    epigraph_blocks = sparse.vstack([-sparse.eye(sample_count, format='csc')] * K, format='csc')
    constraint = sparse.hstack([action_blocks, epigraph_blocks], format='csc')
    upper = flat_samples.reshape(-1, order='F')
    settings = clarabel.DefaultSettings()
    for name, value in CLARABEL_SETTINGS.items():
        setattr(settings, name, value)
    solver = clarabel.DefaultSolver(quadratic, linear, constraint, upper,
        [clarabel.NonnegativeConeT(sample_count * K)], settings)
    solved = solver.solve()
    if solved.status != clarabel.SolverStatus.Solved:
        raise RuntimeError('NS-SPO+ solver status ' + str(solved.status))
    primal, dual = np.asarray(solved.x), np.asarray(solved.z)
    coefficient = primal[:coefficient_count].reshape(dimension, K, order='F')
    direct, _ = ns_value_gradient(coefficient, phi, samples, delta, regularization)
    constant = np.mean(np.sum(weights * flat_samples, axis=1))
    epigraph = float(solved.obj_val + constant)
    error = abs(direct - epigraph)
    violation = float(np.maximum(constraint @ primal - upper, 0).max())
    gap = float(abs(solved.obj_val - solved.obj_val_dual))
    if error > 1e-7 or violation > 1e-7 or gap > 1e-7:
        raise AssertionError('Exact Monte Carlo epigraph qualification failed')
    scores = phi @ coefficient
    diagnostic = dict(status=str(solved.status), solver='Clarabel', solver_version=clarabel.__version__,
        objective=direct, epigraph_objective=epigraph, objective_error=error,
        constraint_violation=violation, primal_dual_gap=gap,
        stationarity=float(abs(quadratic @ primal + linear + constraint.T @ dual).max()),
        iterations=int(solved.iterations), cpu_seconds=time.process_time() - start_cpu,
        wall_seconds=time.perf_counter() - start_wall, questions=n, samples_per_question=repetitions,
        variables=coefficient_count + sample_count, constraints=sample_count * K,
        regularization=regularization, delta=delta, coefficients=coefficient_count,
        sparse_constraint_nonzeros=int(constraint.nnz),
        sparse_constraint_bytes=int(constraint.data.nbytes + constraint.indices.nbytes + constraint.indptr.nbytes),
        training_min_score_gap=float(np.min(np.diff(np.sort(scores, axis=1)[:, -2:], axis=1))))
    try:
        import psutil
        memory = psutil.Process().memory_info()
        diagnostic['current_rss_bytes'] = int(memory.rss)
        diagnostic['process_peak_working_set_bytes'] = int(getattr(memory, 'peak_wset', memory.rss))
    except ImportError:
        pass
    return coefficient, diagnostic


def davidson_value_gradient(coefficient, phi, probability, nu, regularization):
    if probability.shape != (len(phi), 190, 3) or nu <= 0:
        raise ValueError('Davidson needs question/pair/(plus,tie,minus) probabilities and nu>0')
    scores = phi @ coefficient
    gap = scores[:, PAIR_I] - scores[:, PAIR_J]
    logits = np.stack([gap / 2, -gap / 2, np.full_like(gap, np.log(2 * nu))], axis=-1)
    log_normalizer = logsumexp(logits, axis=-1)
    target = probability[:, :, [0, 2, 1]]
    loss = np.mean(log_normalizer - np.sum(target * logits, axis=-1))
    prediction = np.exp(logits - log_normalizer[:, :, None])
    gap_gradient = .5 * (prediction[:, :, 0] - prediction[:, :, 1] - probability[:, :, 0] + probability[:, :, 2])
    gradient = phi.T @ (gap_gradient @ INCIDENCE) / (len(phi) * 190) + 2 * regularization * coefficient
    return float(loss + regularization * np.sum(coefficient ** 2)), gradient


def fit_davidson(phi, probability, regularization):
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    tie_rate = float(probability[:, :, 1].mean())
    # The floor is declared solely to define a finite working likelihood at an
    # empirical rate of zero or one, without using any evaluation outcome.
    raw_nu = tie_rate / (1 - tie_rate) if tie_rate < 1 else np.inf
    nu = float(np.clip(raw_nu, 1e-6, 1e6))
    bounded_tie_rate = nu / (1 + nu)
    shape = (phi.shape[1], K)
    def objective(flat):
        value, gradient = davidson_value_gradient(flat.reshape(shape), phi, probability, nu, regularization)
        return value, gradient.ravel()
    fitted = minimize(objective, np.zeros(shape).ravel(), jac=True, method='L-BFGS-B',
        options=dict(maxiter=2000, ftol=1e-12, gtol=1e-7, maxls=50))
    if not fitted.success:
        raise RuntimeError('Davidson optimization did not converge ' + str(fitted.message))
    return fitted.x.reshape(shape), dict(status='CONVERGED', solver='L-BFGS-B',
        objective=float(fitted.fun), iterations=int(fitted.nit),
        gradient_infinity_norm=float(np.max(abs(fitted.jac))),
        fixed_nu=nu, original_posterior_tie_rate=tie_rate, effective_tie_rate=float(bounded_tie_rate),
        probability_order=['plus', 'tie', 'minus'], regularization=regularization,
        cpu_seconds=time.process_time() - start_cpu, wall_seconds=time.perf_counter() - start_wall,
        questions=len(phi), coefficients=int(np.prod(shape)),
        mechanism='fixed historical nu from posterior mean tie frequency, no hyperparameter search')


def finite_difference(function, coefficient, step=1e-6):
    value, gradient = function(coefficient)
    numerical = np.empty_like(coefficient)
    for index in np.ndindex(coefficient.shape):
        positive, negative = coefficient.copy(), coefficient.copy()
        positive[index] += step
        negative[index] -= step
        numerical[index] = (function(positive)[0] - function(negative)[0]) / (2 * step)
    return float(np.max(abs(gradient - numerical)))


def qualify(out):
    rng = np.random.default_rng(20261114031)
    phi = np.column_stack((np.ones(12), rng.normal(size=(12, 2))))
    samples = rng.uniform(.05, .95, size=(12, 4, K))
    coefficient = rng.normal(size=(3, K))
    ns_error = finite_difference(lambda w: ns_value_gradient(w, phi, samples, .05, .01), coefficient)
    p = rng.dirichlet(np.ones(3), size=(12, 190))
    davidson_error = finite_difference(lambda w: davidson_value_gradient(w, phi, p, .7, .01), coefficient)
    if max(ns_error, davidson_error) > 1e-6:
        raise AssertionError('Finite difference gradient qualification failed')
    fitted, diagnostic = fit_ns_exact(phi, samples, .05, .01)
    # Independent generic constrained optimization of the explicit epigraph.
    n, b, _ = samples.shape
    start = np.r_[fitted.ravel(), np.max(2 * (phi @ fitted)[:, None, :] - samples, axis=2).ravel()]
    def direct_epigraph(flat):
        w, t = flat[:60].reshape(3, K), flat[60:].reshape(n, b)
        weights = near_optimal_weights(samples, .05)
        return float(t.mean() - np.mean(np.sum(weights * (2 * (phi @ w)[:, None, :] - samples), axis=2)) + .01 * np.sum(w * w))
    def feasible(flat):
        w, t = flat[:60].reshape(3, K), flat[60:].reshape(n, b)
        return (t[:, :, None] - 2 * (phi @ w)[:, None, :] + samples).ravel()
    alternative = minimize(direct_epigraph, start, method='SLSQP', constraints=[dict(type='ineq', fun=feasible)],
        options=dict(maxiter=200, ftol=1e-10))
    if not alternative.success or abs(alternative.fun - diagnostic['objective']) > 1e-6:
        raise AssertionError('Independent epigraph comparison failed')
    # The precise-posterior two-action counterexample demonstrates that the
    # supplied convex surrogate does not generally preserve the Bayes action.
    # This is an analytical qualification of scope, not a model tuning signal.
    differences = np.linspace(-.1, .2, 3001)
    losses = .9 * np.maximum(0., .1 - 2 * differences) + .1 * np.maximum(0., 1 + 2 * differences)
    optimum_difference = float(differences[np.argmin(losses)])
    if abs(optimum_difference - .05) > 1e-10:
        raise AssertionError('Known decision-consistency counterexample did not reproduce')
    result = dict(status='PASS', source_sha256=sha(__file__), ns_gradient_max_error=ns_error,
        davidson_gradient_max_error=davidson_error, exact_epigraph=diagnostic,
        independent_SLSQP_objective_error=float(abs(alternative.fun - diagnostic['objective'])),
        decision_consistency_counterexample=dict(probabilities=[.9, .1], reward_vectors=[[.6, .5], [0., 1.]],
            mean_rewards=[.54, .55], surrogate_optimal_score_gap=optimum_difference,
            surrogate_selected_action=0, Bayes_selected_action=1,
            scope='Two-action exact posterior, delta in[0,.05], no general Bayes consistency guarantee'),
        Monte_Carlo_unit='one question with jointly sampled twenty-action vector')
    put(out / 'public/TRAINING_OPERATOR_QUALIFICATION.json', result)
    print(json.dumps(result, ensure_ascii=False))


def fit_fold(fold, posterior_root, out, delta):
    if delta != .05:
        raise ValueError('This frozen study uses practical delta=.05; zero is a named NS sensitivity')
    out.mkdir(parents=True, exist_ok=True)
    design_path = out / 'private/TRAINING_SOURCE_LOCK.json'
    design = dict(source_sha256=sha(__file__), delta=delta, samples_per_question=8,
        primary='NS_main', sensitivity='NS_delta0', statistical_control='Davidson_main',
        MC_qualification='fold0 NS_main_mc_alt from a disjoint eight-draw batch, never selected by outer outcome',
        Davidson_probability='plus/tie/minus frequencies from exactly the same eight mu_samples as NS_main',
        coefficients=340, regularization='reuse each fold original M4 selected lambda',
        solver=CLARABEL_SETTINGS, model_selection='none', new_LLM_calls=0,
        parent_commit='0dece5c291c5ec644f690f892638f81543aed691')
    if design_path.exists():
        if read(design_path) != design:
            raise AssertionError('Training source/config changed')
    else:
        put(design_path, design)
    folder = E8 / f'private/training/fold{fold}/M4'
    history_path = E8 / f'private/qualification/fold_packs/fold{fold}/history480.npz'
    input_path = E8 / f'private/qualification/fold_packs/fold{fold}/evaluation_inputs_only120.npz'
    posterior_path = posterior_root / f'fold{fold}/POSTERIOR_TRAINING.npz'
    state, history, inputs, posterior = map(archive, [folder / 'model_state.npz', history_path, input_path, posterior_path])
    if 'Y' in inputs or set(history['global_indices']) & set(inputs['global_indices']):
        raise AssertionError('Illegal outer fitting split')
    if not np.array_equal(history['global_indices'], posterior['history_indices']):
        raise AssertionError('Posterior training identities differ')
    samples, diagnostic_probability = posterior['mu_samples'], posterior['pair_probability_delta005']
    if samples.shape != (480, 8, K) or diagnostic_probability.shape != (480, 190, 3):
        raise AssertionError('Frozen posterior target shape differs')
    pair_gap = samples[:, :, PAIR_I] - samples[:, :, PAIR_J]
    probability = np.stack([(pair_gap > delta).mean(axis=1),
        (abs(pair_gap) <= delta).mean(axis=1), (pair_gap < -delta).mean(axis=1)], axis=2)
    if np.any(probability < 0) or not np.allclose(probability.sum(axis=2), 1):
        raise AssertionError('Three-state probabilities are invalid')
    fmap = feature_map(state)
    phi, evaluation_phi = fmap.phi(history['E']), fmap.phi(inputs['E'])
    multiplier = read(folder / 'FIT_LOG.json')['selected_parameters']['lambda_multiplier']
    regularization = matched_mean_loss_lambda(480, 100 * multiplier)
    methods = [('NS_main', delta), ('NS_delta0', 0.), ('Davidson_main', delta)]
    if fold == 0:
        methods.append(('NS_main_mc_alt', delta))
    for name, method_delta in methods:
        target = out / f'private/fold{fold}/{name}'
        if (target / 'FROZEN_RECEIPT.json').exists():
            previous = read(target / 'FROZEN_RECEIPT.json')
            if previous['source_sha256'] != sha(__file__) or previous['posterior_sha256'] != sha(posterior_path):
                raise AssertionError('Existing frozen fit belongs to a different source')
            print(f'skip frozen fold{fold} {name}', flush=True)
            continue
        target.mkdir(parents=True, exist_ok=False)
        if name.startswith('NS_'):
            fitting_samples = posterior['mu_samples_alt'] if name == 'NS_main_mc_alt' else samples
            coefficient, diagnostic = fit_ns_exact(phi, fitting_samples, method_delta, regularization)
            if fold == 0:
                reference = state['coefficient']
                main_value = ns_value_gradient(reference, phi, samples, delta, regularization)[0]
                alt_value = ns_value_gradient(reference, phi, posterior['mu_samples_alt'], delta, regularization)[0]
                diagnostic['MC_reference_M4_objective_main'] = main_value
                diagnostic['MC_reference_M4_objective_alternate'] = alt_value
                diagnostic['MC_reference_objective_difference'] = alt_value - main_value
        else:
            coefficient, diagnostic = fit_davidson(phi, probability, regularization)
            diagnostic['delta'] = delta
            diagnostic['posterior_probability_source'] = 'same eight joint mu_samples used by NS_main'
            diagnostic['diagnostic_512_probability_mean_absolute_difference'] = float(abs(probability - diagnostic_probability).mean())
            diagnostic['diagnostic_512_probability_maximum_difference'] = float(abs(probability - diagnostic_probability).max())
        scores = evaluation_phi @ coefficient
        np.savez_compressed(target / 'MODEL.npz', coefficient=coefficient, **fmap.arrays())
        np.savez_compressed(target / 'PREDICTIONS_FROZEN.npz', scores=scores,
            actions=scores.argmax(axis=1), evaluation_indices=inputs['global_indices'])
        put(target / 'FIT_LOG.json', diagnostic)
        put(target / 'FROZEN_RECEIPT.json', dict(fold=fold, method=name,
            frozen_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
            model_sha256=sha(target / 'MODEL.npz'), predictions_sha256=sha(target / 'PREDICTIONS_FROZEN.npz'),
            fit_log_sha256=sha(target / 'FIT_LOG.json'), posterior_sha256=sha(posterior_path),
            history_sha256=sha(history_path), evaluation_inputs_sha256=sha(input_path),
            evaluation_rewards_opened=False, coefficients=340))
        print(json.dumps(dict(fold=fold, method=name, **diagnostic), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['qualify', 'fit-fold'])
    parser.add_argument('--out', type=Path, default=Path(__file__).parent)
    parser.add_argument('--posterior-root', type=Path)
    parser.add_argument('--fold', type=int, choices=range(5))
    parser.add_argument('--delta', type=float, choices=[.05], default=.05)
    args = parser.parse_args()
    if args.mode == 'qualify':
        qualify(args.out)
    else:
        if args.posterior_root is None or args.fold is None:
            parser.error('fit-fold requires --posterior-root and --fold')
        fit_fold(args.fold, args.posterior_root, args.out, args.delta)


if __name__ == '__main__':
    main()
