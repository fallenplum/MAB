"""Independent convex-design checks against exhaustive coarse feasible grids."""

import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np


def main():
    work = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由')) / 'research/e6_contrast_directed_feedback'
    path = work / 'public/b/contrast_design.py'
    spec = importlib.util.spec_from_file_location('design_target', path)
    target = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target)
    rng = np.random.default_rng(20261115)
    rows = []
    for case in range(40):
        n = 3
        coefficient = rng.uniform(.001, .3, size=(4, n))
        annotation = rng.dirichlet(np.ones(n))
        execution = annotation * rng.uniform(.3, 4, size=n)
        ann_budget = float(rng.uniform(.2, .8) * annotation.sum())
        exec_budget = float(rng.uniform(.2, .8) * execution.sum())
        offset = -coefficient.sum(axis=1) if case % 2 else np.zeros(4)
        probability, diagnostic = target.optimize_design(coefficient, annotation, execution, ann_budget, exec_budget, offset=offset)
        grid = np.asarray(list(itertools.product(np.linspace(.05, 1, 25), repeat=n)))
        feasible = (grid @ annotation <= ann_budget) & (grid @ execution <= exec_budget)
        feasible_grid = grid[feasible]
        grid_best = float(np.min(np.max((1 / feasible_grid) @ coefficient.T + offset, axis=1)))
        objective = float(np.max(coefficient @ (1 / probability) + offset))
        assert objective <= grid_best + 1e-7
        assert annotation @ probability <= ann_budget + 1e-7
        assert execution @ probability <= exec_budget + 1e-7
        assert diagnostic['duality_gap'] >= -1e-7
        rows.append(dict(case=case, objective=objective, feasible_grid_best=grid_best,
                         gap=float(diagnostic['duality_gap']), violation=float(diagnostic['max_constraint_violation'])))
    # Under the all/default theorem, independently supplied row systems have the same convex optimum.
    equivalence = []
    for _ in range(20):
        gates = np.concatenate([np.zeros((1, 5)), np.ones((1, 5)), rng.integers(0, 2, size=(4, 5))])
        moments = rng.uniform(.01, .2, size=5)
        pair = np.asarray([(gates[j] - gates[k])**2 for j, k in itertools.combinations(range(len(gates)), 2)])
        aw = np.ones(5) / 5
        ew = aw * rng.uniform(.5, 2, size=5)
        p_marginal, m = target.optimize_design(gates * moments, aw, ew, .4, float(.5 * ew.sum()))
        p_contrast, c = target.optimize_design(pair * moments, aw, ew, .4, float(.5 * ew.sum()))
        equivalence.append(float(np.max(abs(p_marginal - p_contrast))))
        assert abs(m['objective'] - c['objective']) < 1e-7
    result = dict(status='PASS', source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  coarse_grid_cases=len(rows), coarse_grid_points_per_case=25**3,
                  max_duality_gap=max(row['gap'] for row in rows),
                  max_constraint_violation=max(row['violation'] for row in rows),
                  equivalent_all_switch_families=len(equivalence),
                  max_equal_objective_probability_difference=max(equivalence),
                  cases=rows)
    output = work / 'public/theory/INDEPENDENT_SOLVER_AUDIT.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}))


if __name__ == '__main__':
    main()
