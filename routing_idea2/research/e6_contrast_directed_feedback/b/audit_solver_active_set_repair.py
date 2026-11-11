"""Record exact primal invariance of the KKT active-set bookkeeping repair."""
from pathlib import Path
import sys
import json
import hashlib
import importlib.util
import numpy as np
from contrast_design import optimize_design
import run_contrast_synthetic as run


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    folder = run.OUTPUT / 'private/b/solver_qualification'
    old_path = folder / 'contrast_design_before_active_set_fix.py'
    new_path = run.OUTPUT / 'public/b/contrast_design.py'
    original_text = old_path.read_text(encoding='utf-8')
    revised_text = new_path.read_text(encoding='utf-8')
    old_line = 'tolerance = 2e-6 * max(1., abs(t), float(budgets.max()))'
    new_line = 'tolerance = 2e-8 * max(1., abs(t))'
    assert original_text.replace(old_line, new_line) == revised_text
    spec = importlib.util.spec_from_file_location('e6_old_solver_for_qualification', old_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    captured = {}

    def trace(frame, event, arg):
        if frame.f_code.co_filename == str(old_path) and frame.f_code.co_name == 'optimize_design' and event == 'exception':
            if isinstance(arg[1], RuntimeError) and 'p' in frame.f_locals:
                captured['p'] = frame.f_locals['p'].copy()
                captured['gap_scaled'] = float(frame.f_locals['gap_scaled'])
                captured['violation'] = float(frame.f_locals['violation'])
                captured['objective'] = float(frame.f_locals['t'] * frame.f_locals['scale'])
                captured['error'] = str(arg[1])
        return trace

    failure = folder / 'failure_C_f2.npz'
    with np.load(failure) as a:
        arguments = (a['C'], a['aw'], a['ew'], float(a['ba']), float(a['be']))
        kwargs = dict(pmin=a['pmin'], offset=a['offset'])
        sys.settrace(trace)
        try:
            legacy.optimize_design(*arguments, **kwargs)
        except RuntimeError:
            pass
        finally:
            sys.settrace(None)
        probabilities, diagnostic = optimize_design(*arguments, **kwargs)
    assert captured and np.array_equal(captured['p'], probabilities)
    assert captured['objective'] == diagnostic['objective']
    np.savez_compressed(folder / 'FAILED_AND_REPAIRED_PRIMAL.npz',
                        before=captured['p'], after=probabilities)
    result_path = run.OUTPUT / 'public/b/E6_B_RESULTS.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    checked = 0
    max_p_error = 0.
    for case in result['cases']:
        gates = np.asarray(case['gates'])
        pairs = np.asarray(case['pairs'])
        contrast = gates[pairs[:, 0]] - gates[pairs[:, 1]]
        for label in ['Estimated', 'Oracle']:
            a = run.old.true_moments() if label == 'Oracle' else np.asarray(case['estimated_moments'])
            gain = np.asarray(case['true_policy_gains'] if label == 'Oracle' else case['estimated_policy_gains'])
            for objective in ['Marginal', 'GapContrast']:
                if objective == 'Marginal':
                    C = (run.old.WEIGHTS * a)[None, :6]
                else:
                    gap = np.maximum(np.abs(gain[pairs[:, 0]] - gain[pairs[:, 1]]), .01)
                    C = contrast[:, :6] ** 2 * (run.old.WEIGHTS * a)[None, :6] / gap[:, None] ** 2
                p, _ = optimize_design(C, run.old.WEIGHTS[:6], (run.old.WEIGHTS * run.old.COSTS)[:6],
                                       case['phase_annotation'], case['phase_execution'])
                old_p = np.asarray(case['probabilities'][label + '-' + objective])[:6]
                np.testing.assert_array_equal(p, old_p)
                max_p_error = max(max_p_error, float(np.max(np.abs(p - old_p))))
                checked += 1
    if 'original_execution_source_sha256' not in result:
        result['original_execution_source_sha256'] = result['source_sha256'].copy()
        run.save(folder / 'E6_B_RESULTS_BEFORE_PROVENANCE_REPAIR.json', result)
    result['source_sha256']['contrast_design.py'] = sha(new_path)
    result['source_repair'] = dict(
        changed_only='post-fit KKT active-set identification tolerance',
        primal_training_optimization_and_all_sampling_probabilities_unchanged=True,
        original_solver_sha256=sha(old_path), delivery_solver_sha256=sha(new_path),
        B_unique_designs_replayed_bitwise_equal=checked,
        B_sampling_not_repeated=True)
    run.save(result_path, result)
    report = dict(status='PASS', failing_case=dict(fold=2, pilot_index=38, gold_budget=48),
        old_solver_sha256=sha(old_path), repaired_solver_sha256=sha(new_path),
        private_failure_input_sha256=sha(failure),
        old_active_identification_tolerance='2e-6 * max(1, abs(scaled_objective), maximum_resource_budget)',
        new_active_identification_tolerance='2e-8 * max(1, abs(scaled_objective))',
        root_cause='Resource-budget scaling admitted strictly slack objective rows into the KKT multiplier reconstruction.',
        original_primal_objective=captured['objective'], repaired_primal_objective=diagnostic['objective'],
        old_scaled_duality_gap=captured['gap_scaled'], repaired_scaled_duality_gap=diagnostic['scaled_duality_gap'],
        original_constraint_violation=captured['violation'], repaired_constraint_violation=diagnostic['max_constraint_violation'],
        primal_probability_arrays_bitwise_identical=True,
        primal_probability_max_abs_difference=float(np.max(np.abs(captured['p'] - probabilities))),
        original_duality_acceptance_threshold_unchanged=True,
        B_unique_designs_replayed=checked, B_probability_max_abs_difference=max_p_error,
        unchanged_completed_C_folds=[0, 1, 3, 4],
        C_fold2_status='COMPLETE_AFTER_REPAIR', actual_new_LLM_calls=0)
    run.save(folder / 'SOLVER_ACTIVE_SET_REPAIR.json', report)
    run.save(run.OUTPUT / 'public/b/SOLVER_ACTIVE_SET_REPAIR.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
