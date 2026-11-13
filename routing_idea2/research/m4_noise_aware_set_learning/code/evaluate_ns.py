"""Seal all new outer predictions before opening cached train600 outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

from ns_training import archive, feature_map, put, read, sha, ASSETS, E8, PAIR_I, PAIR_J


def distribution(scores):
    action = scores.argmax(axis=1)
    counts = np.bincount(action, minlength=20)
    nonzero = counts[counts > 0] / len(action)
    gaps = np.diff(np.sort(scores, axis=1)[:, -2:], axis=1)[:, 0]
    return dict(counts=counts, top_action=int(counts.argmax()), top_fraction=float(counts.max() / len(action)),
        entropy_effective_actions=float(np.exp(-np.sum(nonzero * np.log(nonzero)))),
        exact_score_ties=int(np.sum(np.sum(scores == scores.max(axis=1, keepdims=True), axis=1) > 1)),
        score_gap_min=float(gaps.min()), score_gap_median=float(np.median(gaps)),
        score_gap_below_1e8=int(np.sum(gaps < 1e-8)), score_gap_below_1e12=int(np.sum(gaps < 1e-12)))


def collect(training, posterior_root):
    source = training / 'ns_training.py'
    lock = read(training / 'private/TRAINING_SOURCE_LOCK.json')
    if lock['source_sha256'] != sha(source):
        raise AssertionError('Frozen training source changed')
    old_path = E8 / 'private/evaluation/PREDICTIONS_AND_CALIBRATION_FROZEN.npz'
    old_receipt = read(E8 / 'private/evaluation/CALIBRATION_FROZEN_RECEIPT.json')
    if old_receipt['prediction_and_calibration_sha256'] != sha(old_path):
        raise AssertionError('Frozen baseline prediction hash differs')
    old = archive(old_path)
    prior_path = ASSETS / 'research/e8_m4_mechanism_audit/private/distribution/INTERVENTIONS_FROZEN.npz'
    prior_receipt = read(prior_path.parent / 'INTERVENTIONS_FROZEN_RECEIPT.json')
    if prior_receipt['prediction_sha256'] != sha(prior_path):
        raise AssertionError('Frozen TSPO uniform scores differ')
    prior = archive(prior_path)
    q = {'M0_Ridge': old['raw__M0_original'], 'M4_lex': old['raw__M4'],
        'TSPO_empirical': prior['raw__full_spo_uniform']}
    for name in ('NS_main', 'NS_delta0', 'Davidson_main'):
        q[name] = np.empty((600, 20))
    seen, fits, restores = [], [], []
    alt = None
    for fold in range(5):
        inputs_path = E8 / f'private/qualification/fold_packs/fold{fold}/evaluation_inputs_only120.npz'
        inputs = archive(inputs_path)
        indices = inputs['global_indices']
        if 'Y' in inputs:
            raise AssertionError('Input pack contains outer rewards')
        if not np.all(old['fold'][indices] == fold):
            raise AssertionError('Frozen fold identities differ')
        seen.extend(indices.tolist())
        posterior = posterior_root / f'fold{fold}/POSTERIOR_TRAINING.npz'
        variants = ['NS_main', 'NS_delta0', 'Davidson_main'] + (['NS_main_mc_alt'] if fold == 0 else [])
        for variant in variants:
            folder = training / f'private/fold{fold}/{variant}'
            receipt = read(folder / 'FROZEN_RECEIPT.json')
            if receipt['source_sha256'] != sha(source) or receipt['posterior_sha256'] != sha(posterior):
                raise AssertionError('Frozen model source or posterior differs')
            for filename, key in [('MODEL.npz', 'model_sha256'), ('PREDICTIONS_FROZEN.npz', 'predictions_sha256'),
                                  ('FIT_LOG.json', 'fit_log_sha256')]:
                if sha(folder / filename) != receipt[key]:
                    raise AssertionError('Frozen fitted artifact changed')
            predictions, state = archive(folder / 'PREDICTIONS_FROZEN.npz'), archive(folder / 'MODEL.npz')
            if not np.array_equal(indices, predictions['evaluation_indices']):
                raise AssertionError('Frozen score identities differ')
            restored = feature_map(state).phi(inputs['E']) @ state['coefficient']
            maximum_error = float(abs(restored - predictions['scores']).max())
            if maximum_error > 1e-12 or not np.array_equal(restored.argmax(1), predictions['actions']):
                raise AssertionError('Saved model cannot reproduce its frozen actions')
            if variant == 'NS_main_mc_alt':
                alt = dict(indices=indices, scores=predictions['scores'])
            else:
                q[variant][indices] = predictions['scores']
            fit = read(folder / 'FIT_LOG.json')
            fits.append(dict(fold=fold, variant=variant, **fit))
            restores.append(dict(fold=fold, variant=variant, score_error=maximum_error))
    if sorted(seen) != list(range(600)) or len(fits) != 16:
        raise AssertionError('All sixteen predefined fits must exist before scoring')
    return q, old, alt, fits, restores


def holm(values):
    names = sorted(values, key=values.get)
    result, previous = {}, 0.
    for position, name in enumerate(names):
        previous = max(previous, min(1., (len(names) - position) * values[name]))
        result[name] = previous
    return result


def evaluate(training, posterior_root, output):
    if (output / 'public/NS_TRAINING_RESULTS.json').exists():
        raise FileExistsError('Evaluation output already exists')
    q, old, alt, fits, restores = collect(training, posterior_root)
    frozen_path = output / 'private/ALL_PREDICTIONS_FROZEN.npz'
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(frozen_path, **{'scores__' + key: value for key, value in q.items()},
        fold=old['fold'], default_actions=old['default_actions'],
        alt_fold0_indices=alt['indices'], alt_fold0_scores=alt['scores'])
    put(output / 'private/ALL_PREDICTIONS_FROZEN_RECEIPT.json', dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        predictions_sha256=sha(frozen_path), evaluated_outer_rewards=False,
        models_restored=restores, training_source_sha256=sha(training / 'ns_training.py')))
    # The reward reader is invoked only after every method's actions are sealed.
    sys.path.insert(0, str(E8 / 'public/qualification'))
    from e8_assets import load_historical
    data, refs, _ = load_historical(ASSETS)
    y, task, sub = data['Y'], data['task'], data['sub']
    means = y.mean(axis=1)
    n = len(y)
    rng = np.random.default_rng(20261114091)
    sampled = []
    for fold in range(5):
        for name in ('gsm8k', 'bbh'):
            rows = np.flatnonzero((old['fold'] == fold) & (task == name))
            sampled.append(rng.choice(rows, size=(10000, len(rows)), replace=True))
    bootstrap = np.concatenate(sampled, axis=1)
    rewards = {name: y[np.arange(n), :, scores.argmax(axis=1)] for name, scores in q.items()}
    rewards['train_global_fixed'] = y[np.arange(n), :, old['default_actions']]

    def pair(left, right):
        delta = rewards[left] - rewards[right]
        question = delta.mean(axis=1)
        draws = question[bootstrap].mean(axis=1)
        gain = float(question.mean())
        p = (1 + int(np.sum(abs(draws - gain) >= abs(gain)))) / 10001
        return dict(gain=gain, ci95=np.quantile(draws, [.025, .975]),
            raw_two_sided_centered_bootstrap_p=p,
            corrected_events=int(np.sum(delta > 0)), destroyed_events=int(np.sum(delta < 0)),
            net_events=int(delta.sum()), seed_gain=delta.mean(axis=0),
            task_gain={t: float(delta[task == t].mean()) for t in ('gsm8k', 'bbh')},
            folds=[float(delta[old['fold'] == f].mean()) for f in range(5)])

    metrics = {}
    for name, scores in q.items():
        values = rewards[name]
        action = scores.argmax(axis=1)
        true_gap = means[:, PAIR_I] - means[:, PAIR_J]
        predicted_gap = scores[:, PAIR_I] - scores[:, PAIR_J]
        valid = abs(true_gap) > 0
        correct = (np.sign(predicted_gap) == np.sign(true_gap)).astype(float)
        correct[abs(predicted_gap) <= 1e-12] = .5
        rank_per_question = np.divide((correct * valid).sum(axis=1), valid.sum(axis=1),
            out=np.full(n, np.nan), where=valid.sum(axis=1) > 0)
        metrics[name] = dict(accuracy=float(values.mean()), seed_accuracy=values.mean(axis=0),
            task_accuracy={t: float(values[task == t].mean()) for t in ('gsm8k', 'bbh')},
            fold_accuracy=[float(values[old['fold'] == f].mean()) for f in range(5)],
            subtask={str(s): dict(n=int(np.sum((task == 'bbh') & (sub == s))),
                accuracy=float(values[(task == 'bbh') & (sub == s)].mean())) for s in sorted(set(sub[task == 'bbh']))},
            distribution=distribution(scores),
            empirical_three_repeat_selection_loss=float((means.max(axis=1) - means[np.arange(n), action]).mean()),
            strict_pair_rank_accuracy=float(np.nanmean(rank_per_question)),
            strict_pair_support_questions=int(np.sum(valid.any(axis=1))),
            versus_M4=pair(name, 'M4_lex'), versus_Ridge=pair(name, 'M0_Ridge'),
            versus_fixed=pair(name, 'train_global_fixed'))
    primary = {'NS_main_minus_M4': metrics['NS_main']['versus_M4'],
               'NS_main_minus_Ridge': metrics['NS_main']['versus_Ridge']}
    adjusted = holm({key: value['raw_two_sided_centered_bootstrap_p'] for key, value in primary.items()})
    for key, value in primary.items():
        value['Holm_p'] = adjusted[key]
        value['exploratory_positive_support'] = bool(value['gain'] > 0 and adjusted[key] < .05)
    alt_indices = alt['indices']
    alt_reward = y[alt_indices, :, alt['scores'].argmax(axis=1)]
    primary_fold0 = rewards['NS_main'][alt_indices]
    alt_gap = np.diff(np.sort(alt['scores'], axis=1)[:, -2:], axis=1)[:, 0]
    mc = dict(questions=120, scope='prespecified disjoint eight-draw integration diagnostic, no winner selection',
        action_changes=int(np.sum(alt['scores'].argmax(1) != q['NS_main'][alt_indices].argmax(1))),
        primary_accuracy=float(primary_fold0.mean()), alternative_accuracy=float(alt_reward.mean()),
        alternative_minus_primary=float((alt_reward - primary_fold0).mean()),
        max_raw_score_difference=float(abs(alt['scores'] - q['NS_main'][alt_indices]).max()),
        alternative_score_gap_min=float(alt_gap.min()), alternative_near_ties_1e8=int(np.sum(alt_gap < 1e-8)))
    result = dict(status='COMPLETE_EXPLORATORY_FIXED_CAPACITY', methods=metrics,
        primary_statistics=primary, MC_qualification=mc, fits=fits,
        statistics=dict(unit='600 original question clusters, three generations attached',
            bootstrap_repetitions=10000, bootstrap_seed=20261114091,
            strata='outerfold x task', primary_weights='GSM8K50% BBH50%',
            Holm_family=['NS_main_minus_M4', 'NS_main_minus_Ridge'],
            scope='reused development CV, overlapping training histories'),
        computational_cost=dict(successful_fit_cpu_seconds=sum(v['cpu_seconds'] for v in fits),
            successful_fit_wall_seconds=sum(v['wall_seconds'] for v in fits),
            new_fits=len(fits), new_LLM_calls=0),
        limitations=['Joint posterior is a working approximation', 'Eight training MC samples are finite integration',
            'Raw ranking scores have no calibrated probability interpretation',
            'Near-tied predicted argmax can depend on numerical residuals',
            'Empirical selection loss is not population regret'])
    put(output / 'public/NS_TRAINING_RESULTS.json', result)
    put(output / 'public/NS_PRIMARY_STATISTICS.json', primary)
    np.savez_compressed(output / 'private/PAIRED_OUTCOMES.npz', task=task, sub=sub, fold=old['fold'],
        **{'rewards__' + key: value for key, value in rewards.items()})
    put(output / 'private/REWARD_ACCESS_RECEIPT.json', dict(
        decoded_split='selector_train600', source_refs=refs, forbidden_outcomes_accessed=[],
        all_predictions_frozen_sha256=sha(frozen_path), new_LLM_calls=0))
    print(json.dumps(dict(status='COMPLETE', accuracy={name: m['accuracy'] for name, m in metrics.items()},
        primary_statistics=primary, MC_qualification=mc), default=lambda a: a.tolist(), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-root', type=Path, default=Path(__file__).parent)
    parser.add_argument('--posterior-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    options = parser.parse_args()
    evaluate(options.training_root, options.posterior_root, options.output)
