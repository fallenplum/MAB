"""Freeze the user's E9 contract before cached development experiments."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

ASSET = Path(r'E:\语言模型组合路由')
ROOT = ASSET / 'research/e9_consistent_cost_sensitive_learning'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if ROOT.exists():
        raise FileExistsError('E9 root already exists')
    (ROOT / 'public').mkdir(parents=True)
    (ROOT / 'private').mkdir()
    protocol = {
        'status': 'FROZEN_BEFORE_E9_MODEL_FITS',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'branch': 'codex/e9-consistent-cost-sensitive-learning',
        'parent_commit': '19ba4fd5316bd897036622dab6b02fcfd0e036fd',
        'target': 'maximize conditional expected final aggregation correctness from question input only',
        'data': 'original train600, GSM8K300 BBH300, six models K3 twenty combinations, V2 frozen aggregation/scoring',
        'outer_folds': 5, 'history_questions': 480, 'outer_evaluation_questions': 120,
        'inner_folds': 3, 'inner_split_seed': '20261113+outer, exact original E8 saved rows',
        'features': 'same frozen MiniLM384, fitting-region PCA16 and StandardScaler, one penalized constant,17x20 head',
        'families': {'A': 'Empirical-Ridge, original M0_tuned complete-process verified reuse',
            'B': 'PosteriorMean-Ridge', 'C': 'Empirical-CSCE', 'D': 'PosteriorMean-CSCE'},
        'anchor': 'M0_original', 'background': ['M4', 'TSPO', 'NS', 'Davidson'],
        'lambda_multipliers': [.1, 1., 10.], 'regularization': '100*multiplier/(3*n_fit*20)',
        'data_normalization': 'n_original_questions*20 for both MSE and CSCE',
        'CSCE': 'w=r-min(r), Z=sum(w), Z*logsumexp(q)-sum(w*q), never row-normalized',
        'constant_reward_rows': 'zero data loss/gradient, retained in denominator',
        'model_selection': 'inner unseen true routed accuracy, GSM/BBH50:50, tie1e-12 stronger regularization',
        'teacher': '512-draw bounded joint Gaussian working mean, not eight-draw mean or winner probability',
        'outer_teacher': 'verified immutable NS posterior_mean reuse',
        'inner_teacher': 'rebuild each inner320 with its own internal threefold crossfit supporting unequal sizes',
        'teacher_numerics': {'shrinkage': .2, 'ridge': 1e-6, 'burn': 128, 'thin': 2,
            'chains': 2, 'draws_per_chain': 256, 'max_Rhat': 1.1, 'min_ESS': 32},
        'teacher_split_seed': 20261113,
        'teacher_sample_seed': '2026111400+1000*outer+100*teacher_internal_fold+3, shared common random numbers across fitting regions',
        'permutation_seeds': [20261114091, 20261114092, 20261114093],
        'permutation_rule': 'whole20-action x3-generation blocks within task/BBH subtask, input fixed, B/D teachers rebuilt from permuted raw rewards',
        'controls': 'reuse each main selected hyperparameter, no retuning',
        'teacher_sharing': 'B/D share legal targets',
        'planned_new_head_slots': 195, 'planned_new_teacher_regions': 30,
        'Ridge_equivalence_cache': 'one B fit per inner, other grid coefficients derived only after Gram=nI and actual-gradient verification; fallback actual solves logged',
        'selection_ties': 'fixed canonical three-member semantic identity keys, independent of output coordinate order',
        'primary_comparisons': ['C-A', 'B-A'],
        'mechanism_comparisons': ['D-C', 'D-B', 'D-A', 'each vs M0_original', 'each vs frozen NS'],
        'bootstrap': {'repetitions': 10000, 'seed': 20261114091,
            'unit': 'original question retaining all generations and all methods', 'strata': 'outerfold x task'},
        'multiplicity': 'Holm exactly two primary tests, ordinary paired95 intervals separately labeled',
        'permutation_inference': 'three permutations are sensitivity diagnostics only, minimum valid Monte Carlo p1/4',
        'minimum_practical_gain': .01,
        'go_scope': 'independent confirmation planning only, review primary effects, practical margin, controls and candidate-specific paired power',
        'post_evaluation_search': False, 'new_LLM_calls': 0,
        'forbidden_outcomes': ['validation200', 'fresh400', 'sealed200'],
        'all_controls_frozen_before_outer_scoring': True,
        'scientific_scope': 'exploratory repeatedly exposed development CV with overlapping training histories',
    }
    path = ROOT / 'public/E9_PROTOCOL.json'
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    snapshots = {}
    for name in ('e8_decision_focused_combination_learning', 'e8_m4_mechanism_audit', 'm4_noise_aware_set_learning'):
        parent = ASSET / 'research' / name
        snapshots[name] = {p.relative_to(parent).as_posix(): sha(p) for p in sorted(parent.rglob('*')) if p.is_file()}
    (ROOT / 'private/PARENT_SNAPSHOTS.json').write_text(json.dumps(snapshots, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'private/PROTOCOL_FREEZE_RECEIPT.json').write_text(json.dumps({
        'protocol_sha256': sha(path), 'created_utc': protocol['created_utc'],
        'immutable_protocol': True, 'parent_file_counts': {k:len(v) for k,v in snapshots.items()}}, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'root': str(ROOT), 'protocol_sha256': sha(path), 'parent_files': sum(map(len,snapshots.values()))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
