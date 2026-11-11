"""Independent re-evaluation of the primary A question bootstrap and matches."""

import json
import os
from pathlib import Path

import numpy as np


def main():
    root = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由')) / 'research/e6_contrast_directed_feedback'
    with np.load(root / 'private/a/PAIRED_E_RESULTS.npz') as a:
        saved = {key: a[key].copy() for key in a.files}
    indices = saved['bootstrap_indices']
    masks = saved['masks']
    d = saved['D'].mean(axis=1)
    reward = saved['Rb'].mean(axis=1)[None] + masks * d
    bootstrap_accuracy = reward[:, indices].mean(axis=2).T
    bootstrap_gain = (masks[:, indices] * d[indices]).mean(axis=2).T
    errors = dict(accuracy=float(np.max(abs(bootstrap_accuracy - saved['bootstrap_accuracy']))),
                  gain=float(np.max(abs(bootstrap_gain - saved['bootstrap_gain']))))
    matched = np.zeros_like(bootstrap_gain)
    # The original broad-task x fold strata have fixed bootstrap size60 each.
    for fold in range(5):
        for task in ('gsm8k', 'bbh'):
            is_group = (saved['folds'][indices] == fold) & (saved['tasks'][indices] == task)
            n = is_group.sum(axis=1)
            assert np.all(n == 60)
            acceptance = np.sum(masks[:, indices] * is_group[None], axis=2).T / n[:, None]
            group_d = np.sum(d[indices] * is_group, axis=1) / n
            matched += acceptance * group_d[:, None] * (60 / 600)
    errors['task_matched'] = float(np.max(abs(matched - saved['bootstrap_matched_task'])))
    # Smaller subtask bootstrap cells vary in size and must be recalculated.
    rng = np.random.default_rng(20261116)
    probes = rng.choice(len(indices), 100, replace=False)
    subtask_error = 0.
    for b in probes:
        chosen = indices[b]
        groups = [(int(saved['folds'][q]), str(saved['subtasks'][q]) if saved['tasks'][q] == 'bbh' else 'gsm8k') for q in chosen]
        expected = np.zeros(len(masks))
        for group in set(groups):
            local = np.asarray([item == group for item in groups])
            expected += masks[:, chosen[local]].mean(axis=1) * d[chosen[local]].mean() * local.mean()
        subtask_error = max(subtask_error, float(np.max(abs(expected - saved['bootstrap_matched_subtask'][b]))))
    errors['subtask_matched_100_probes'] = subtask_error
    assert max(errors.values()) < 1e-12
    output = dict(status='PASS', original_questions=600, paired_generation_repeats=3,
                  all_accuracy_gain_taskmatch_bootstraps=5000, subtask_match_bootstrap_probes=100,
                  maximum_errors=errors,
                  primary_recomputed_on_each_question_bootstrap=True,
                  inference_scope='Frozen cross-fitted development predictions, overlapping training histories retained as limitation')
    target = root / 'public/theory/INDEPENDENT_A_STATISTICS_AUDIT.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
