"""Reconstruct legal E6-C observations, HT estimates, choices and hard ledgers."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np


ROOT = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由')) / 'research/e6_contrast_directed_feedback'
HERE = ROOT / 'public/theory'
STAGING = ROOT / 'public'


def main():
    sys.path.insert(0, str(STAGING / 'c'))
    spec = importlib.util.spec_from_file_location('independent_C_actor', STAGING / 'c/run_real_pool.py')
    actor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor)
    results = []
    minimum_pi = 1.
    maximum_ht_error = 0.
    maximum_budget_violation = 0.
    minimum_partition = float('inf')
    maximum_pi_error = 0.
    design_count = 0
    clipped_coordinates = 0
    cap_fallbacks = 0
    no_pilot_cap_fallbacks = 0
    conditional_token_excess = []
    for fold in range(5):
        with np.load(ROOT / f'private/c/C_f{fold}.npz') as a:
            saved = {key: a[key].copy() for key in a.files}
        metadata = json.loads((ROOT / f'private/c/PLANNING_C_f{fold}.json').read_text(encoding='utf-8'))
        decisions = json.loads((ROOT / f'private/c/DECISIONS_C_f{fold}.json').read_text(encoding='utf-8'))
        assert decisions['decision_sha256'] == hashlib.sha256((ROOT / f'private/c/C_f{fold}.npz').read_bytes()).hexdigest()
        masks = saved['masksP']
        assert np.all((masks == 0) | (masks == 1))
        assert np.all(masks[0] == 0)
        assert any(np.all(row == 1) for row in masks)
        purchases = saved['purchases']
        inclusion = saved['inclusion']
        dbar = saved['DP'].mean(axis=1)
        assert np.all(inclusion > 0)
        minimum_pi = min(minimum_pi, float(inclusion.min()))
        reconstructed = np.einsum('bmri,ji,i->bmrj', purchases / inclusion, masks, dbar) / 60
        error = float(np.max(abs(reconstructed - saved['estimates'])))
        maximum_ht_error = max(maximum_ht_error, error)
        assert error < 1e-12
        choices = np.argmax(reconstructed >= reconstructed.max(axis=-1, keepdims=True) - 1e-12, axis=-1)
        assert np.array_equal(choices, saved['selected'])
        assert np.array_equal(saved['selected'][:, 1], saved['selected'][:, 2])
        assert np.array_equal(purchases[:, 1], purchases[:, 2])
        assert not set(saved['P']) & set(saved['Q'])
        actual_calls = purchases @ saved['calls']
        actual_tokens = purchases.astype(float) @ saved['costmatrix'][:, list(saved['costnames']).index('tokens')]
        for bi, budget in enumerate(saved['budgets']):
            assert np.all(purchases[bi].sum(axis=-1) == budget)
            assert np.all(actual_tokens[bi] <= 159744 * budget)
            for pilot_index in range(50):
                meta = next(row for row in metadata if row['pilot_index'] == pilot_index and row['gold_budget'] == budget)
                bounds = np.r_[np.full(4, meta['cap']), meta['no_pilot_cap']]
                block = slice(pilot_index * 40, (pilot_index + 1) * 40)
                violation = float(np.max(actual_calls[bi, :, block] - bounds[:, None]))
                maximum_budget_violation = max(maximum_budget_violation, violation)
                assert violation <= 0
                pilot = saved['pilot_ids'][pilot_index]
                remaining = np.setdiff1d(np.arange(60), pilot)
                cap_fallbacks += int(meta['fallback'])
                no_pilot_cap_fallbacks += int(meta['no_pilot_fallback'])
                for mi, name in enumerate(('Uniform', 'Marginal', 'Contrast', 'GapContrast')):
                    proposal = saved['proposal'][bi, mi, pilot_index, remaining]
                    clipped_coordinates += int(np.sum((proposal < 1e-6) | (proposal > 1 - 1e-6)))
                    design = actor.make_design(proposal, saved['calls'][remaining], int(budget - 12),
                                               int(meta['cap'] - saved['calls'][pilot].sum()))
                    minimum_partition = min(minimum_partition, design.partition)
                    pi_error = float(np.max(abs(design.pi - inclusion[bi, mi, pilot_index * 40, remaining])))
                    maximum_pi_error = max(maximum_pi_error, pi_error)
                    assert pi_error < 1e-12
                    design_count += 1
                    conditional_token_excess.append(float(meta['conditional_expected_tokens'][name] - meta['predicted_token_budget']))
        results.append(dict(fold=fold, policy_selections=int(saved['selected'].size), ht_max_error=error,
                            distinct_P_questions=len(saved['P']), distinct_Q_questions=len(saved['Q']),
                            paired_marginal_contrast_identical=True))
    data, refs = actor.load_train()
    with np.load(ROOT / 'private/c/C_f0.npz') as source:
        Q = source['Q'].copy()
        original = {key: source[key].copy() for key in ('selected', 'estimates', 'purchases', 'inclusion')}
    poisoned = dict(data)
    poisoned['Y'] = data['Y'].copy()
    poisoned['Y'][Q] = np.nan
    temp = ROOT / 'private/audit_replays/C_Q_nan'
    for relative in ('public/E6_RUN_CONFIG.json', 'private/PROTOCOL_FROZEN.json', 'private/E6_FGE_PQ_FROZEN.npz', 'private/a/fold0_policies.npz'):
        target = temp / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    actor.WORK = temp
    actor.PILOTS = 2
    actor.DRAWS = 3
    actor.load_train = lambda: (poisoned, refs)
    try:
        actor.run_fold(0)
    except ValueError as exc:
        # The evaluation-only NaN must be rejected when writing the final JSON.
        assert 'Out of range float' in str(exc) or 'JSON compliant' in str(exc), str(exc)
    with np.load(temp / 'private/c/C_f0.npz') as altered:
        selected_replicates = np.asarray([0, 1, 2, 40, 41, 42])
        poison_checks = {}
        for key, expected in original.items():
            poison_checks[key] = bool(np.array_equal(expected[:, :, selected_replicates], altered[key]))
            assert poison_checks[key], key
    record = dict(status='PASS', folds=results, independently_reconstructed_policy_selections=sum(row['policy_selections'] for row in results),
                  minimum_actual_inclusion=minimum_pi, maximum_HT_reconstruction_error=maximum_ht_error,
                  maximum_hard_call_cap_violation=max(0., maximum_budget_violation),
                  conditional_design_stability=dict(reconstructed_designs=design_count,
                      minimum_partition=minimum_partition, actual_inclusion_reconstruction_max_error=maximum_pi_error,
                      proposal_clip_bounds=[1e-6, 1 - 1e-6], clipped_proposal_coordinates=clipped_coordinates,
                      hard_call_cap_fallback_designs=cap_fallbacks, no_pilot_hard_call_cap_fallback_designs=no_pilot_cap_fallbacks,
                      conditional_predicted_token_excess_count=int(np.sum(np.asarray(conditional_token_excess) > 1e-7)),
                      conditional_predicted_token_excess_max=max(0., max(conditional_token_excess)),
                      conditional_predicted_token_excess_interpretation='Conditioning changes actual inclusion; preconditioning token proxy constraint is not inherited. Gold/call and conservative frozen token hard caps are checked separately.'),
                  gold_count_exact=True, token_caps_pass=True, finite_pool_masks_default_all=True,
                  Q_reward_isolation=dict(outer_fold=0, poisoned_Q_questions=len(Q),
                      pilot_draws=2, purchase_draws_per_pilot=3, total_decisions=3 * 5 * 6,
                      all_Q_rewards_replaced_by_NaN=True, unchanged=poison_checks,
                      evaluation_JSON_rejects_NaN=True),
                  source_sha256=hashlib.sha256((STAGING / 'c/run_real_pool.py').read_bytes()).hexdigest(),
                  caveat='Conditional on fixed gates and F/G history; code-array perturbation is not operating-system access forensics.')
    output = HERE / 'INDEPENDENT_REAL_POOL_AUDIT.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
