"""Historical nested-OOF calibration and input-only projected-cost diagnostics."""
from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=r'E:\语言模型组合路由')
    args = parser.parse_args()
    work = Path(args.root) / 'research/b1_reliable_paired_evaluation'
    private = work / 'private/assets'
    with np.load(private / 'TRAIN_OOF.npz', allow_pickle=False) as archive:
        active = archive['active']
        targets = np.column_stack((archive['residual'] ** 2, archive['d'] ** 2))[active]
        predictions = archive['nested_moment_predictions'][active]
    calibration = {}
    for j, name in enumerate(('paired_residual_second_moment', 'paired_raw_second_moment')):
        order = np.argsort(predictions[:, j], kind='stable')
        bins = []
        for ids in np.array_split(order, 5):
            bins.append(dict(questions=len(ids), predicted_mean=float(predictions[ids, j].mean()),
                             observed_mean=float(targets[ids, j].mean())))
        correlation = float(np.corrcoef(predictions[:, j], targets[:, j])[0, 1])
        calibration[name] = dict(questions=int(active.sum()),
            observed_mean=float(targets[:, j].mean()), predicted_mean=float(predictions[:, j].mean()),
            mean_squared_error=float(np.mean((predictions[:, j] - targets[:, j]) ** 2)),
            correlation=correlation, prediction_quintiles=bins,
            questions_below_floor=int(np.sum(predictions[:, j] <= 1e-6)),
            confidence_claim='working second-moment prediction; no calibrated conditional guarantee')
    with np.load(private / 'FROZEN_INPUTS.npz', allow_pickle=False) as archive:
        active = archive['active']
        cost = archive['cost_prediction'][active]
        actor = dict(active_questions=int(active.sum()), structural_zero_questions=int((~active).sum()),
            score_gap_summary={name: float(value) for name, value in
                zip(('min', 'median', 'max'), np.quantile(archive['m'][active], (0, .5, 1)))},
            auxiliary_raw_negative_counts=dict(residual=int(np.sum(archive['vhat_raw'][active] < 0)),
                                                raw_moment=int(np.sum(archive['rawmom_raw'][active] < 0))),
            projected_pair_token_cost=dict(min=float(cost.min()), median=float(np.median(cost)),
                max=float(cost.max()), mean=float(cost.mean()), std=float(cost.std()),
                coefficient_of_variation=float(cost.std() / cost.mean()),
                maximum_over_minimum=float(cost.max() / cost.min()),
                source='train600 average usage and frozen validation actions only'))
    result = dict(schema='B1_train_only_auxiliary_diagnostic_v1',
                  historical_nested_OOF_calibration=calibration, input_only_actor_diagnostic=actor,
                  validation_scores_or_actual_costs_used=False, parameter_selection_after_diagnostic=False)
    output = work / 'public/assets/AUXILIARY_DIAGNOSTICS.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
