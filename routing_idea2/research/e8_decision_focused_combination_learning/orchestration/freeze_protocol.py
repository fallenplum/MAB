"""Freeze E8 training and statistical choices before new outer evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import sys

ROOT=Path(r'E:\语言模型组合路由')
BASE=ROOT/'research/e8_decision_focused_combination_learning'
sys.path[:0]=[str(BASE/'public/qualification'),str(BASE/'public/training')]
from e8_training_v1 import DEFAULT_SETTINGS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    protocol_path=BASE/'public/E8_PROTOCOL.json'
    if protocol_path.exists():raise FileExistsError('Existing E8 protocol is immutable')
    prerequisite=json.loads((BASE/'public/qualification/E8_QUALIFICATION.json').read_text())
    actual=json.loads((BASE/'public/qualification/ACTUAL_OPERATOR_REVIEW.json').read_text())
    if not prerequisite['A1']['status']=='PASS' or not prerequisite['A2']['status']=='PASS':
        raise AssertionError('The original Ridge and equal-gap equivalence must pass')
    if not str(actual.get('status','')).startswith('PASS'):raise AssertionError('Actual operators must pass')
    sources=[BASE/'public/qualification/e8_assets.py',BASE/'public/training/e8_training_v1.py',
             BASE/'public/training/run_e8_training_v1.py',BASE/'public/evaluation/e8_metrics.py']
    source_lock=dict(schema='E8_immutable_source_lock_v1',files={str(p):sha(p) for p in sources})
    lock_path=BASE/'private/E8_SOURCE_LOCK.json';put(lock_path,source_lock)
    protocol=dict(schema='MAB_E8_fixed_objectives_protocol_v1',
        status='FROZEN_BEFORE_NEW_OUTER_EVALUATION',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        branch='codex/e8-decision-focused-combination-learning',
        parent_branch='codex/b1-reliable-paired-evaluation',parent_commit='a8ca3eb660136de0cc0793d153e3561b8315283b',
        target='same-feature same-linear-capacity relative-value and exact decision-focused training',
        asset=dict(original_questions=600,tasks=dict(gsm8k=300,bbh=300),members=6,K=3,combinations=20,
            generations=[42,137,2024],aggregator='Qwen/Qwen3-8B',
            operator_fingerprint='9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e',
            scorer='safe_exact_v2_0_terminal_final_heading_fix',question_is_independent_unit=True),
        data=dict(outer_folds=5,history_per_fold=480,evaluation_per_fold=120,inner_folds=3,
            historical_fold_sha256='6b433dcd2d8e6cd9938d2dd2d97261e18c3ae3d18325d7a04c6590e4d0600d50',
            fold_pack_manifest_sha256=sha(BASE/'private/qualification/FOLD_PACKS_MANIFEST.json'),
            data_builder='privileged fold-scoped pack constructor',
            training_access='history480 rewards and evaluation_inputs_only120 only',
            new_validation200_outcome_access=False,fresh400_outcome_access=False,sealed200_outcome_access=False),
        features=dict(encoder='frozen all-MiniLM-L6-v2 cached384',PCA=16,svd_solver='full',
            standardization='historical fitting region only',constant='one penalized coordinate',
            head_parameters=17*20,task_and_subtask_not_predictor_inputs=True),
        methods=dict(M0_original='absolute mean-squared loss; fixed original raw-alpha100',
            M0_tuned='same M0 loss; same three regularization candidates fairness comparator',
            M1='absolute-gap-weighted RankNet, fixed190 denominator then mean over questions',
            M2='ListNet cross entropy, mean over questions',
            M3='OOF Top3 competitor weighted squared gaps, 1/400 then mean over questions',
            M4='exact epigraph convex SPO+ QP, lexicographic true argmax'),
        settings=DEFAULT_SETTINGS,
        regularization='lambda=(100*multiplier)/(3*n_fitting_questions*20), constant penalized',
        model_selection=dict(primary_baseline='M0_original',fairness_baseline='M0_tuned',
            criterion='GSM8K and BBH equally weighted historical inner OOF actual routed accuracy',
            ties='stronger regularization, then smaller M3 alpha or larger ListNet target temperature',
            routing_ties='original exact numpy argmax, first lexicographic action',base_seed=20261113),
        sensitivity=dict(M3_top5='reuse selected Top3 alpha and regularization',
            M4_uniformtie='reuse selected lexicographic regularization; uniform average over true reward-maximizers',
            shuffled_control='one fixed task/subtask-block permutation recipe within each legal fitting region; frozen selected hyperparameters; no retuning',
            shuffle_oof='one regional permuted dataset, plain OOF holds its current reassigned reward block out'),
        calibration=dict(purpose='numerical value-error diagnostics only',
            mapping='one nonnegative scalar slope and shared intercept, no clipping, all20 outputs',
            fit='selected historical inner-OOF scores and original historical mean rewards',
            raw_scores_select_service_action=True),
        statistics=dict(primary_comparisons=['M3 minus M0_original','M4 minus M0_original'],
            bootstrap_repetitions=10000,bootstrap_seed=20261113,
            resampling='paired original questions within outerfold x task strata, all methods and3 repeats attached',
            primary_weights='GSM8K50% BBH50%, BBH internally by question',
            bbh_subtask_macro='secondary descriptive',
            pvalues='two-sided centered paired-bootstrap Monte Carlo test, plus-one correction',
            multiplicity='Holm across exactly two primary objective comparisons',
            evidence_scope='conditional reused-development cross-validation with overlapping training histories'),
        metrics=dict(primary='real aggregated correctness at raw-score argmax',
            pair_rank='true nonzero mean gap, numerical predicted tie within1e-12 earns half credit; question average and support reported',
            competition='common outer M0_original predicted Top3; any pair touching that set',
            numeric_errors='raw and historical-affine-calibrated both reported',
            selection_loss='max three-repeat empirical mean minus selected empirical mean, not population regret'),
        stopping='positive primary Holm-controlled evidence required for next-phase protocol only; no automatic feedback execution',
        physical_llm_calls=0,prior_results_modified=False,
        solver=dict(exact_SPO='OSQP1.0.4 epigraph convex quadratic program',
            dependency_location='private/dependencies, isolated from shared environment',workers=3,BLAS_threads_per_worker=1),
        qualification_hashes={str(p.relative_to(BASE)):sha(p) for p in [
            BASE/'public/qualification/E8_QUALIFICATION.json',
            BASE/'public/qualification/ACTUAL_OPERATOR_REVIEW.json',
            BASE/'public/E8_AUTHOR_SPO_OPERATOR_CHECK.json']},
        source_sha256={str(p.relative_to(BASE)):sha(p) for p in sources},source_lock_sha256=sha(lock_path))
    put(protocol_path,protocol)
    put(BASE/'private/E8_PROTOCOL_FREEZE_RECEIPT.json',dict(protocol_sha256=sha(protocol_path),
        source_lock_sha256=sha(lock_path),new_outer_reward_evaluation_started=False,training_started=False))
    print(json.dumps(dict(status=protocol['status'],protocol_sha256=sha(protocol_path),sources=len(sources))))


if __name__=='__main__':main()
