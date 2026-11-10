"""E5 paired-gain gates with question-held-out fitting and matched controls.

This module never receives evaluation rewards during score fitting. Repeated
generations are averaged inside question-level losses and switching decisions.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


SCORE_NAMES = ('gap', 'factorized', 'disagreement', 'direction')
REGULARIZATION_GRID = (.1, 1., 10.)
TARGET_COVERAGES = (.1, .25, .5, .75)
SMOOTHING_PSEUDO_SUCCESSES = 1.
SMOOTHING_PSEUDO_FAILURES = 1.
SELECTION_GAIN_TOLERANCE = 1e-12


def _paired_array(values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] < 1 or not np.isin(result, (-1, 0, 1)).all():
        raise ValueError('Paired feedback must have shape questions by repetitions and values -1,0,1')
    return result


def _prediction_features(predictions, default_actions, candidate_actions):
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] != 20 or not np.isfinite(predictions).all():
        raise ValueError('Expected finite question by twenty P2 predictions')
    default_actions = np.asarray(default_actions)
    candidate_actions = np.asarray(candidate_actions)
    n = len(predictions)
    for name, actions in [('default', default_actions), ('candidate', candidate_actions)]:
        if actions.shape != (n,) or actions.dtype.kind not in 'iu' or np.any((actions < 0) | (actions >= 20)):
            raise ValueError(name + ' actions must be aligned integer team indices')
    if np.any(default_actions == candidate_actions):
        raise ValueError('The frozen proposal must differ from the default')
    rows = np.arange(n)
    features = np.column_stack((predictions[rows, default_actions], predictions[rows, candidate_actions]))
    return features, features[:, 1] - features[:, 0]


def main_weights(task):
    """GSM/BBH each one half, within each broad task question-uniform."""
    task = np.asarray(task).astype(str)
    if task.ndim != 1 or not np.isin(task, ('gsm8k', 'bbh')).all():
        raise ValueError('Expected one GSM8K/BBH task per original question')
    result = np.zeros(len(task))
    for name in ('gsm8k', 'bbh'):
        mask = task == name
        if not mask.any():
            raise ValueError('The primary balanced mixture needs both task groups')
        result[mask] = .5 / mask.sum()
    return result


def _question_splits(n, seed, strata=None, folds=3):
    if n < folds:
        raise ValueError('Too few original questions for the fixed split count')
    labels = None if strata is None else np.asarray(strata).astype(str)
    if labels is not None:
        if labels.shape != (n,):
            raise ValueError('CV strata must align with original questions')
        _, counts = np.unique(labels, return_counts=True)
        if counts.min() >= folds:
            return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.arange(n), labels))
    return list(KFold(folds, shuffle=True, random_state=seed).split(np.arange(n)))


def _training_observations(features, D, kind):
    n, repeats = D.shape
    question = np.repeat(np.arange(n), repeats)
    weights = np.full(n * repeats, 1. / repeats)
    if kind == 'r':
        labels = (D != 0).ravel().astype(int)
        keep = np.ones(len(labels), dtype=bool)
    elif kind == 'u':
        keep = (D != 0).ravel()
        labels = (D > 0).ravel().astype(int)
    else:
        raise ValueError('Unknown probability head')
    return features[question[keep]], labels[keep], weights[keep], question[keep]


def _support(D, kind):
    _, y, weights, question = _training_observations(np.zeros((len(D), 2)), D, kind)
    return dict(
        original_questions=len(D),
        active_original_questions=int(len(np.unique(question))),
        positive_question_support=int(len(np.unique(question[y == 1]))),
        negative_question_support=int(len(np.unique(question[y == 0]))),
        positive_generation_records=int(np.sum(y == 1)),
        negative_generation_records=int(np.sum(y == 0)),
        total_training_weight=float(weights.sum()),
        positive_training_weight=float(weights[y == 1].sum()),
        per_question_training_weight=np.bincount(question, weights=weights, minlength=len(D)).tolist(),
        question_support_overlap_note='A question with different generation outcomes can support both labels',
        conditional_weight_note='u retains original 1/L weights after D!=0 filtering; active rows are not re-normalized per question')


@dataclass
class ProbabilityHead:
    kind: str
    C: float
    scaler: StandardScaler
    estimator: LogisticRegression | None
    constant: float | None
    fit_info: dict

    def predict(self, features):
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != 2 or not np.isfinite(features).all():
            raise ValueError('Probability heads use exactly the two frozen value features')
        if self.constant is not None:
            return np.full(len(features), self.constant)
        return self.estimator.predict_proba(self.scaler.transform(features))[:, 1]

    def descriptor(self):
        return dict(kind=self.kind, C=self.C, constant=self.constant,
                    feature_names=['P2_default_raw_value', 'P2_candidate_raw_value'],
                    scaler_mean=self.scaler.mean_.tolist(), scaler_scale=self.scaler.scale_.tolist(),
                    coefficient=None if self.estimator is None else self.estimator.coef_.ravel().tolist(),
                    intercept=None if self.estimator is None else float(self.estimator.intercept_[0]),
                    fit_info=self.fit_info)


def _fit_head(features, D, kind, C):
    scaler = StandardScaler().fit(features)
    observed, y, weights, _ = _training_observations(features, D, kind)
    support = _support(D, kind)
    if not len(y) or len(np.unique(y)) < 2:
        probability = float((np.dot(weights, y) + SMOOTHING_PSEUDO_SUCCESSES) /
                            (weights.sum() + SMOOTHING_PSEUDO_SUCCESSES + SMOOTHING_PSEUDO_FAILURES))
        return ProbabilityHead(kind, float(C), scaler, None, probability,
                               dict(support=support, fallback='Beta(1,1) weighted constant', solver_iterations=0))
    model = LogisticRegression(C=float(C), solver='lbfgs', max_iter=2000, tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(scaler.transform(observed), y, sample_weight=weights)
    return ProbabilityHead(kind, float(C), scaler, model, None,
                           dict(support=support, fallback=None, solver_iterations=int(model.n_iter_.max())))


def _head_loss(probability, D, kind):
    probability = np.clip(np.asarray(probability), 1e-12, 1 - 1e-12)
    if kind == 'r':
        target = (D != 0).astype(float)
        mask = np.ones_like(target, dtype=bool)
    else:
        target = (D > 0).astype(float)
        mask = D != 0
    loss = -(target * np.log(probability[:, None]) + (1 - target) * np.log1p(-probability[:, None]))
    numerator = float(np.sum(loss * mask) / D.shape[1])
    denominator = float(mask.sum() / D.shape[1])
    return numerator, denominator


def _choose_C(features, D, kind, seed, strata=None):
    splits = _question_splits(len(features), seed, strata)
    candidates = []
    for C in REGULARIZATION_GRID:
        numerator = denominator = 0.
        fold_information = []
        for train, validation in splits:
            model = _fit_head(features[train], D[train], kind, C)
            loss, weight = _head_loss(model.predict(features[validation]), D[validation], kind)
            numerator += loss
            denominator += weight
            fold_information.append(dict(training_questions=train.tolist(), validation_questions=validation.tolist(),
                                         loss_weight=weight, fallback=model.fit_info['fallback']))
        candidates.append(dict(C=C, loss=None if denominator == 0 else numerator / denominator,
                               loss_weight=denominator, folds=fold_information))
    chosen = min(candidates, key=lambda row: (float('inf') if row['loss'] is None else row['loss'], row['C']))
    return float(chosen['C']), dict(candidates=candidates, chosen_C=float(chosen['C']),
                                   objective='weighted binary log loss with original-question grouped folds',
                                   undefined_objective_rule='smallest C when no authorized direction outcomes exist')


def fit_gate_scores(pred_G, base_G, candidate_G, D_G, pred_E, base_E, candidate_E,
                    strata_G=None, seed=20261110):
    """Fit r/u using G only. Nested OOF G scores support frozen threshold selection.

    ``pred_E`` contains frozen input-only P2 predictions. Evaluation labels are
    deliberately absent from this API. Metadata can stratify CV, never feature
    columns. With ``strata_G=None`` plain shuffled question folds are used.
    """
    features_G, gap_G = _prediction_features(pred_G, base_G, candidate_G)
    features_E, gap_E = _prediction_features(pred_E, base_E, candidate_E)
    D_G = _paired_array(D_G)
    if len(features_G) != len(D_G):
        raise ValueError('G feedback and predictions must refer to the same original questions')
    strata = None if strata_G is None else np.asarray(strata_G)
    outer = _question_splits(len(D_G), seed, strata)
    oof = {kind: np.full(len(D_G), np.nan) for kind in ('r', 'u')}
    heads, details = {}, {}
    for kind in ('r', 'u'):
        outer_information = []
        for fold, (train, validation) in enumerate(outer):
            train_strata = None if strata is None else strata[train]
            C, selection = _choose_C(features_G[train], D_G[train], kind, seed + 100 + fold, train_strata)
            head = _fit_head(features_G[train], D_G[train], kind, C)
            oof[kind][validation] = head.predict(features_G[validation])
            outer_information.append(dict(fold=fold, training_questions=train.tolist(),
                held_out_questions=validation.tolist(), regularization=selection, fitted_head=head.descriptor()))
        C, selection = _choose_C(features_G, D_G, kind, seed + 200, strata)
        heads[kind] = _fit_head(features_G, D_G, kind, C)
        details[kind] = dict(OOF_folds=outer_information, full_G_regularization=selection,
                             full_G_fitted_head=heads[kind].descriptor())
    r_E, u_E = heads['r'].predict(features_E), heads['u'].predict(features_E)
    for values in [*oof.values(), r_E, u_E]:
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Gate probability scores failed finite/range qualification')
    scores_G = dict(gap=gap_G, factorized=oof['r'] * (2 * oof['u'] - 1),
                    disagreement=oof['r'], direction=2 * oof['u'] - 1)
    scores_E = dict(gap=gap_E, factorized=r_E * (2 * u_E - 1),
                    disagreement=r_E, direction=2 * u_E - 1)
    return dict(scores_G_oof=scores_G, scores_E=scores_E,
                raw_G_oof=dict(r=oof['r'], u=oof['u']), raw_E=dict(r=r_E, u=u_E),
                model_heads=heads, fit_metadata=details,
                configuration=dict(seed=seed, regularization_grid=list(REGULARIZATION_GRID),
                    CV='3 outer G question folds with 3-fold training-only inner tuning',
                    deployment_features=['P2_default_raw_value', 'P2_candidate_raw_value'],
                    repetitions=D_G.shape[1], repeat_weight=1. / D_G.shape[1],
                    direction_score='2*u-1, monotone transform of u, zero means u=.5',
                    disagreement_zero='zero score has no gain-sign interpretation',
                    E_reward_permission=False, model_or_task_label_features=False))


def freeze_thresholds(scores_G_oof, D_G, task_G):
    """Freeze five finite thresholds per score and choose each primary using G."""
    D_G = _paired_array(D_G)
    weights = main_weights(task_G)
    mean_D = D_G.mean(axis=1)
    thresholds, main_indices, gates_G = {}, {}, {}
    for name in SCORE_NAMES:
        score = np.asarray(scores_G_oof[name], dtype=float)
        if score.shape != (len(D_G),) or not np.isfinite(score).all():
            raise ValueError('G score must align with its original questions')
        rules = []
        masks = []
        for index, coverage in enumerate(TARGET_COVERAGES):
            value = float(np.quantile(score, 1 - coverage, method='linear'))
            mask = score > value
            masks.append(mask)
            rules.append(dict(rule_id=name + '_q' + str(int(100 * coverage)), index=index,
                target_G_coverage=coverage, threshold=value, threshold_origin='G OOF score quantile',
                acceptance='score > threshold', G_coverage=float(weights @ mask),
                G_paired_gain=float(weights @ (mask * mean_D)),
                raw_u_threshold=(value + 1) / 2 if name == 'direction' else None))
        mask = score > 0
        masks.append(mask)
        rules.append(dict(rule_id=name + '_zero', index=4, target_G_coverage=None,
            threshold=0., threshold_origin='zero_score' if name == 'disagreement' else 'zero_gain_or_direction',
            acceptance='score > threshold', G_coverage=float(weights @ mask),
            G_paired_gain=float(weights @ (mask * mean_D)),
            raw_u_threshold=.5 if name == 'direction' else None))
        maximum_gain = max(row['G_paired_gain'] for row in rules)
        tied = [j for j in range(5) if rules[j]['G_paired_gain'] >= maximum_gain - SELECTION_GAIN_TOLERANCE]
        best = min(tied, key=lambda j: (rules[j]['G_coverage'], -rules[j]['threshold'], j))
        thresholds[name], main_indices[name], gates_G[name] = rules, best, np.asarray(masks)
    unique = {(name, row['threshold']) for name in SCORE_NAMES for row in thresholds[name]}
    return dict(thresholds=thresholds, main_indices=main_indices, gates_G=gates_G,
                nondefault_family_size=20, unique_score_threshold_pairs=len(unique),
                unique_G_gate_patterns=int(len(np.unique(np.vstack(list(gates_G.values())), axis=0))),
                unique_pattern_scope='Observed G patterns are diagnostic, not a proof of equal functions on unseen inputs',
                default_reference='exact zero-gain policy, excluded from finite five-threshold selection',
                primary_score='factorized', selection_gain_tolerance=SELECTION_GAIN_TOLERANCE,
                selection='maximum G OOF paired gain within tolerance, then lower coverage/higher numeric threshold/lower index')


def apply_thresholds(scores_E, frozen):
    return {name: np.asarray([np.asarray(scores_E[name]) > row['threshold']
                              for row in frozen['thresholds'][name]], dtype=bool) for name in SCORE_NAMES}


def gate_statistics(gate, D, task, sub=None):
    D = _paired_array(D)
    gate = np.asarray(gate, dtype=bool)
    task = np.asarray(task).astype(str)
    if gate.shape != (len(D),) or task.shape != gate.shape:
        raise ValueError('One shared generation-independent gate is required per original question')
    weights = main_weights(task)
    z = gate[:, None] * D
    corrected = gate[:, None] * (D == 1)
    harmed = gate[:, None] * (D == -1)
    rho = float(weights @ np.mean(z ** 2, axis=1))
    correction = float(weights @ corrected.mean(axis=1))
    harm = float(weights @ harmed.mean(axis=1))
    gain = float(weights @ z.mean(axis=1))
    eta = correction / rho if rho > 0 else None
    identity_error = abs(gain - (rho * (2 * eta - 1) if eta is not None else 0.))
    if identity_error > 1e-12:
        raise ArithmeticError('Paired disagreement-direction identity failed')
    coverage = float(weights @ gate)
    result = dict(coverage=coverage, switched_questions=int(gate.sum()), independent_questions=len(D),
        gain=gain, rho=rho, eta=eta, correction_rate=correction, harm_rate=harm,
        gain_among_switched=gain / coverage if coverage > 0 else None,
        expected_corrected_questions=float(corrected.mean(axis=1).sum()),
        expected_harmed_questions=float(harmed.mean(axis=1).sum()),
        question_generation_corrections=int(corrected.sum()), question_generation_harms=int(harmed.sum()),
        selected_nonzero_feedback_questions=int(np.sum(gate & np.any(D != 0, axis=1))),
        selected_positive_mean_questions=int(np.sum(gate & (D.mean(axis=1) > 0))),
        selected_negative_mean_questions=int(np.sum(gate & (D.mean(axis=1) < 0))),
        per_generation_gain=(weights @ z).tolist(),
        per_generation_correction=(weights @ corrected).tolist(),
        per_generation_harm=(weights @ harmed).tolist(), identity_error=identity_error,
        per_question_gain=z.mean(axis=1), per_question_correction=corrected.mean(axis=1),
        per_question_harm=harmed.mean(axis=1), per_question_rho=(z ** 2).mean(axis=1),
        main_weighting='GSM8K/BBH each .5, BBH question average')
    result['by_task'] = {name: dict(gain=float(z[task == name].mean()),
                                  coverage=float(gate[task == name].mean())) for name in ('gsm8k', 'bbh')}
    if sub is not None:
        sub = np.asarray(sub).astype(str)
        if sub.shape != gate.shape:
            raise ValueError('Subtask labels are evaluator-only aligned metadata')
        names = sorted(set(sub[task == 'bbh']))
        bbh_values = [float(z[(task == 'bbh') & (sub == name)].mean()) for name in names]
        result['bbh_subtask_macro_gain'] = float(np.mean(bbh_values))
        result['balanced_subtask_gain'] = .5 * (result['by_task']['gsm8k']['gain'] + np.mean(bbh_values))
        result['by_bbh_subtask'] = {name: dict(gain=float(z[(task == 'bbh') & (sub == name)].mean()),
            coverage=float(gate[(task == 'bbh') & (sub == name)].mean()),
            original_questions=int(np.sum((task == 'bbh') & (sub == name)))) for name in names}
    return result


def _matching_groups(task, sub, scheme):
    if scheme == 'global':
        return [('all', np.arange(len(task)))]
    if scheme == 'task':
        return [(name, np.flatnonzero(task == name)) for name in ('gsm8k', 'bbh')]
    if sub is None:
        raise ValueError('Subtask-matched evaluator requires true diagnostic labels')
    return [('gsm8k', np.flatnonzero(task == 'gsm8k'))] + [
        ('bbh/' + name, np.flatnonzero((task == 'bbh') & (sub == name)))
        for name in sorted(set(sub[task == 'bbh']))]


def matched_controls(gate, D, task, sub=None, reps=1000, seed=20261110, return_masks=False):
    """Evaluator-side matched gates, preserving every question's fixed proposal."""
    D = _paired_array(D)
    gate = np.asarray(gate, dtype=bool)
    task = np.asarray(task).astype(str)
    sub = None if sub is None else np.asarray(sub).astype(str)
    if gate.shape != (len(D),) or task.shape != gate.shape or (sub is not None and sub.shape != gate.shape):
        raise ValueError('Matching requires aligned original-question metadata')
    if reps < 1:
        raise ValueError('Randomization repeat count must be positive')
    weights = main_weights(task)
    mean_D = D.mean(axis=1)
    observed = float(weights @ (gate * mean_D))
    result = {}
    for scheme_index, scheme in enumerate(('global', 'task', 'subtask')):
        groups = _matching_groups(task, sub, scheme)
        random_gate = np.zeros((reps, len(gate)), dtype=bool)
        inclusion = np.zeros(len(gate))
        rng = np.random.default_rng(np.random.SeedSequence([seed, scheme_index]))
        group_info = []
        for name, indices in groups:
            switches = int(gate[indices].sum())
            inclusion[indices] = switches / len(indices)
            if switches == len(indices):
                random_gate[:, indices] = True
            elif switches > 0:
                # A row-wise random-key sample gives an exact uniform subset of
                # the specified size, without replacement inside each question group.
                order = np.argsort(rng.random((reps, len(indices))), axis=1)
                chosen = indices[order[:, :switches]]
                random_gate[np.arange(reps)[:, None], chosen] = True
            if not np.all(random_gate[:, indices].sum(axis=1) == switches):
                raise ArithmeticError('A matched control changed its group switch count')
            group_info.append(dict(group=name, original_questions=len(indices), switches=switches,
                                   has_within_group_randomization=0 < switches < len(indices)))
        if not np.all(random_gate.sum(axis=1) == gate.sum()):
            raise ArithmeticError('A matched control changed total switched questions')
        gains_by_seed = (random_gate * weights[None, :]) @ D
        gains = gains_by_seed.mean(axis=1)
        expected = inclusion * mean_D
        value = dict(scheme=scheme, observed_gain=observed,
            exact_expected_random_gain=float(weights @ expected),
            observed_minus_expected_random_gain=float(weights @ (gate * mean_D - expected)),
            expected_per_question_random_gain=expected,
            observed_minus_expected_per_question=gate * mean_D - expected,
            expected_random_inclusion=inclusion, randomization_gains=gains,
            randomization_gain_by_generation=gains_by_seed,
            randomization_mean=float(gains.mean()),
            randomization_mean_MC_standard_error=float(gains.std(ddof=1) / np.sqrt(reps)) if reps > 1 else None,
            randomization_distribution_quantile95=np.quantile(gains, [.025, .975]).tolist(),
            one_sided_randomization_tail_probability=float((1 + np.sum(gains >= observed - 1e-15)) / (reps + 1)),
            repeats=reps, seed=seed, scheme_seed_index=scheme_index, groups=group_info,
            observed_question_switches=int(gate.sum()),
            observed_macro_coverage=float(weights @ gate), expected_random_macro_coverage=float(weights @ inclusion),
            independent_question_count=len(gate), candidate_actions_changed=False,
            repeated_randomizations_are_new_questions=False,
            inference_scope='Conditional evaluator-side matched permutation distribution, not new independent questions')
        if return_masks:
            value['randomization_masks'] = random_gate
        result[scheme] = value
    return result


def qualification():
    rng = np.random.default_rng(20261110)
    n_G, n_E = 60, 36
    q_G, q_E = rng.uniform(.1, .9, (n_G, 20)), rng.uniform(.1, .9, (n_E, 20))
    b_G, b_E = np.zeros(n_G, dtype=int), np.zeros(n_E, dtype=int)
    c_G = np.argmax(np.where(np.arange(20)[None] == 0, -np.inf, q_G), axis=1)
    c_E = np.argmax(np.where(np.arange(20)[None] == 0, -np.inf, q_E), axis=1)
    D_G = rng.choice([-1., 0., 1.], size=(n_G, 3), p=[.15, .65, .20])
    D_E = rng.choice([-1., 0., 1.], size=(n_E, 3), p=[.20, .55, .25])
    task_G = np.asarray(['gsm8k'] * (n_G // 2) + ['bbh'] * (n_G // 2))
    task_E = np.asarray(['gsm8k'] * (n_E // 2) + ['bbh'] * (n_E // 2))
    sub_E = np.asarray([''] * (n_E // 2) + ['logic_a'] * 9 + ['logic_b'] * 9)
    fitted = fit_gate_scores(q_G, b_G, c_G, D_G, q_E, b_E, c_E, seed=17)
    changed_E = fit_gate_scores(q_G, b_G, c_G, D_G, 1 - q_E, b_E, c_E, seed=17)
    for kind in ('r', 'u'):
        assert np.array_equal(fitted['raw_G_oof'][kind], changed_E['raw_G_oof'][kind])
        assert fitted['model_heads'][kind].descriptor() == changed_E['model_heads'][kind].descriptor()
        outer = fitted['fit_metadata'][kind]['OOF_folds']
        held = np.concatenate([row['held_out_questions'] for row in outer])
        assert np.array_equal(np.sort(held), np.arange(n_G))
        for row in outer:
            assert not set(row['training_questions']) & set(row['held_out_questions'])
    frozen = freeze_thresholds(fitted['scores_G_oof'], D_G, task_G)
    gates = apply_thresholds(fitted['scores_E'], frozen)
    assert frozen['nondefault_family_size'] == 20 and all(len(frozen['thresholds'][s]) == 5 for s in SCORE_NAMES)
    max_identity = 0.
    for name in SCORE_NAMES:
        for gate in gates[name]:
            metrics = gate_statistics(gate, D_E, task_E, sub_E)
            max_identity = max(max_identity, metrics['identity_error'])
            controls = matched_controls(gate, D_E, task_E, sub_E, reps=1000, seed=19, return_masks=True)
            for control in controls.values():
                masks = control['randomization_masks']
                assert np.all(masks.sum(1) == gate.sum())
                assert masks.ndim == 2 and masks.shape[1] == n_E
                error = abs(control['randomization_mean'] - control['exact_expected_random_gain'])
                assert error <= 5 * control['randomization_mean_MC_standard_error'] + 1e-12
    for value in (0., 1., -1.):
        D = np.full((n_G, 3), value)
        fallback = fit_gate_scores(q_G, b_G, c_G, D, q_E, b_E, c_E, seed=23)
        assert fallback['model_heads']['r'].constant is not None
        assert fallback['model_heads']['u'].constant is not None
    # Conditional row weights remain inherited 1/L, including mixed-seed questions.
    mixed = np.asarray([[0., 0., 1.], [0., -1., 1.], [0., 0., 0.]])
    support = _support(mixed, 'u')
    assert np.allclose(support['per_question_training_weight'], [1 / 3, 2 / 3, 0])
    return dict(status='PASS', checks=['G_only_score_fitting', 'nested_OOF_question_disjointness',
        'twenty_frozen_rules', 'direction_probability_zero_semantics', 'conditional_repeat_weights',
        'missing_class_constant_fallback', 'paired_gain_identity', 'matched_counts_and_shared_generation_decision',
        'analytical_random_control_mean_matches_MC'],
        maximum_gain_identity_error=max_identity, new_llm_calls=0)


if __name__ == '__main__':
    import json
    print(json.dumps(qualification(), ensure_ascii=False, indent=2))
