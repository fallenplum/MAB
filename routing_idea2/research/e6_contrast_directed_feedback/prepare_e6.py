import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
WORK = ROOT / 'research/e6_contrast_directed_feedback'
OLD = ROOT / 'research/e5_paired_gain_validation'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    for part in ['public', 'private']:
        (WORK / part).mkdir(parents=True, exist_ok=True)
    target = WORK / 'private/PROTOCOL_FROZEN.json'
    if target.exists():
        raise RuntimeError('Existing frozen protocol must be preserved')
    bundle = json.loads((ROOT / 'artifacts/private/routing_redesign_20261104/ff_v1/BUNDLE_TRAINVAL.json').read_text('utf-8-sig'))
    ref = bundle['inputs']['selector_train']
    inputs_path = Path(ref['path'])
    if not inputs_path.is_absolute():
        inputs_path = ROOT / inputs_path
    assert sha(inputs_path) == ref['sha256']
    with np.load(inputs_path, allow_pickle=False) as a:
        ids = a['question_ids'].astype(str)
        tasks = a['task_labels'].astype(str)
    splits = {}
    sources = {}
    for f in range(5):
        source = OLD / f'private/base/BASE_f{f}.npz'
        with np.load(source, allow_pickle=False) as a:
            F, G, E = (a[x].copy() for x in ['F', 'G', 'E'])
        P = []
        for task in ['gsm8k', 'bbh']:
            ordered = sorted(E[tasks[E] == task], key=lambda q: hashlib.sha256(('E6-PQ-v1|' + ids[q]).encode()).hexdigest())
            P.extend(ordered[:30])
        P = np.array(sorted(P), dtype=int)
        Q = np.setdiff1d(E, P)
        assert len(P) == len(Q) == 60
        assert np.sum(tasks[P] == 'gsm8k') == np.sum(tasks[Q] == 'gsm8k') == 30
        assert not set(P) & set(Q)
        for name, rows in [('F', F), ('G', G), ('E', E), ('P', P), ('Q', Q)]:
            splits[f'f{f}_{name}'] = rows
        sources[f'BASE_f{f}'] = sha(source)
    splitpath = WORK / 'private/E6_FGE_PQ_FROZEN.npz'
    np.savez_compressed(splitpath, **splits)
    config = dict(
        schema='mab_e6_v1', frozen_utc=datetime.now(timezone.utc).isoformat(),
        task_sha256=sha(ROOT / '新建 文本文档.txt'),
        branch='codex/e6-contrast-directed-feedback',
        base_commit='7e06c632787f1de093732aed0250c8a30d3035b7',
        population='train600 only; fresh400 and sealed200 not evaluated',
        seeds=[42,137,2024], N=6, K=3, combinations=20, physical_new_llm_calls=0,
        A=dict(main='A3', lambdas=[0.01,0.1,1], cv='three grouped original-question folds inside G, broad-task stratification',
               cv_objective='mean paired service gain; tie stronger regularization', threshold=0,
               logistic_intercept='penalized as part of beta', normalization='all original n*L, D=0 remains in denominator',
               one_class_fallback='Beta(1,1) weighted constant', auto_class_balance=False,
               A3_features=46, PCA='8 components fit on F only', combo_embedding='first 2 right singular vectors of row-centered F average rewards, no singular value scaling, deterministic sign',
               feature_scaling='G training fold only', A4_weights=[0.5,0.5/19]),
        B=dict(pilots_per_active_stratum=[30,5], budgets=[[.25,.5],[.5,.25],[.5,.5]], repeats=2000,
               epsilon=.01, preserve_default_and_all=True, old_worlds_unchanged=True),
        C=dict(main_gold=36, gold_budgets=[24,36,48], pilot_questions=12,
               pilot_draws=50, purchase_draws_per_pilot=40, repeats_per_fold=2000,
               methods=['Uniform','Marginal','Contrast','GapContrast','NoPilotUniform'],
               sampler='conditional Bernoulli with exact gold count and execution-call cap, exact first-order inclusion probabilities',
               independent_Bernoulli='supplementary formula and budget-overrun diagnostic; not treated as hard-cap deployment',
               execution_units='one member or aggregator call, all 3 generation repeats charged',
               execution_cap='3*floor(7.5*B_gold), maximum 22.5 calls per gold; deterministic fallback to 24*B only if positivity infeasible based on inputs and pilot costs',
               token_hard_cap='159744*B_gold from frozen member4096/aggregate6144 input and2048 output caps',
               expected_cost_source='F historical per-model and per-combination tokens only',
               estimated_moment='pilot gate-signature mean Dbar squared, 4 pseudo-observations at pooled pilot mean',
               estimated_gain='pilot mean g*Dbar, no Q rewards', epsilon=.01,
               tie='estimate tolerance 1e-12, default then fixed listed order',
               unit='question, shared three-repeat average', Q_only_evaluator=True),
        statistics=dict(bootstrap_reps=5000, seed=20261112, question_strata='outer fold x GSM/BBH',
                        primary_comparisons=['A3 minus task-matched random','C Contrast minus Marginal at gold36','C Contrast minus Uniform at gold36'],
                        family='Holm three two-sided centered bootstrap tests; unadjusted95CI plus simultaneous Bonferroni98.3333CI',
                        MC='paired replicate distributions, clustered pilot sensitivity separately',
                        outer_fold_overlap='conditional development inference, not independent training replications'),
        frozen_split_sha256=sha(splitpath), inputs_sha256=sha(inputs_path), base_sources=sources,
        E6_A_E_scores_read_before_freeze=False, Q_rewards_available_to_algorithms=False)
    write(WORK / 'public/E6_RUN_CONFIG.json', config)
    write(target, dict(status='FROZEN_BEFORE_E6_EVALUATION', utc=config['frozen_utc'],
                       config_sha256=sha(WORK / 'public/E6_RUN_CONFIG.json'), split_sha256=sha(splitpath),
                       decoded_payloads=['train600 inputs', 'E5 F/G/E indices only'], outcomes_not_decoded=True))
    (WORK / 'private/E6_USER_TASK.txt').write_bytes((ROOT / '新建 文本文档.txt').read_bytes())
    print(json.dumps(dict(status='FROZEN', protocol_sha256=sha(target), split_sha256=sha(splitpath))))

if __name__ == '__main__':
    main()
