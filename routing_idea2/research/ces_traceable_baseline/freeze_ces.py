"""Freeze E7 and a bounded author-code CES migration before cache evaluation."""
from pathlib import Path
import os
import json
from datetime import datetime,timezone
import sys
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/ces_traceable_baseline'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
sys.path.insert(0,str(PUBLIC))
from ces_common import load_train,load_member_scores,manual_inputs,make_features,resources,put,sha,E7,SOURCE_PINS


def main():
    if (PUBLIC/'CES_PROTOCOL.json').exists():
        raise RuntimeError('Protocol already frozen')
    data,refs=load_train()
    member,member_source=load_member_scores(data)
    manual,text_source=manual_inputs(data)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(PRIVATE/'MANUAL_INPUTS.npz',features=manual,question_ids=data['ids'])
    protected=['public/c/active_feedback.py','public/E7_PROTOCOL.json','public/E7_PRIMARY_STATISTICS.json',
        'public/E7_COMPLETE_RESULTS.json','public/05_E7_RESEARCH_DECISION.md','public/PUBLIC_DELIVERY_MANIFEST.json',
        'private/c/ALL_PATHS_FROZEN.json','private/REMOTE_DELIVERY_VERIFIED.json']
    freeze=dict(status='E7_DECISION_AWARE_AND_CONCLUSIONS_FROZEN',commit='a7bc3d9a72160b3e6134cae76a1a13f47b8b992b',
        files={name:sha(E7/name) for name in protected},no_parameter_or_conclusion_changes=True)
    put(PUBLIC/'E7_FREEZE_RECEIPT.json',freeze)
    config=dict(schema='ces_traceable_baseline_v1',frozen_utc=datetime.now(timezone.utc).isoformat(),
        branch='codex/ces-traceable-baseline-20261112',base_commit=freeze['commit'],
        upstream='JinkunXu/CES@9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7',source_pins=SOURCE_PINS,
        native_performance_scope='Original reported evaluation cannot yet be recomputed without original trajectories; author core executed, migration separately identified',
        pool=dict(N=6,K=3,teams=20,generation_seeds=[42,137,2024],reward='real frozen Qwen3 final aggregation scores'),
        trials=dict(folds=5,algorithm_seeds=[17,29,43],tracks=['ProfileOnly-SingleCombo','HistoryFit-SingleCombo','HistoryFit-Full20'],
            formal_paths=45,checkpoints=[0,20,40,60],primary_gold=60,split_sha256=sha(E7/'private/E7_SPLITS_FROZEN.npz')),
        features=dict(d_x=32,d_c=8,phi_dimension=96,beta=3.,gamma=3.,
            query='existing normalized MiniLM-L6 cached384 vector replaces native multilingual-L12 query encoder; explicit adaptation',
            manual='exact author FeatEncoder with character length, digit fraction, codefence; text-only ID filtered extraction',
            profile='I240 six-member actualGSM/BBH correctness plus I historical token cost normalized within6; 3D replaces native10MMLU+cost11D',
            profile_apply='exact native SVD initializer and raw s application, no extra standardization',
            projectors='original pca warmstart function extracted AST, same seed as algorithm repeat; no learned new projection'),
        learner=dict(alpha=3.5,ridge=.1,lam_cost=0.,main_service='original greedy marginal set-UCB select_topk k3',
            diagnostic_service='alpha0 same state, author select_topk, explicitly a mean-only controlled variant',
            update='original author update(z, r), z=sum of selected memberphi',
            observation_unit='three-repeat final reward average, one update per delivered combination; original numeric parameters fixed, changed unit disclosed',
            history_fit='I240 full20 mean rewards =>4800 labeled combination updates, native profile-only has no aggregate pretraining',
            order='fixed ID-sorted I order, then lexicographic20 combination IDs per question',
            hyperparameter_search=False),
        feedback=dict(question_schedule='exact E7 Random purchased U order for samefold/seed, first60, no E rewards',
            SingleCombo='freeze CES UCB selection before reveal, execute selected3 members and1 aggregator in3 generation repeats, deliver one mean reward',
            Full20='same E7 full20 package, deliver all20 mean scores; off-policy/shared history extension of author estimator',
            Gold_per_question=1,initial_profile=dict(member_calls=4320,aggregator_calls=0,gold=240,member_scoring_ops=4320),
            initial_history_extra=dict(member_calls=0,aggregator_calls=14400,gold=0,aggregate_scoring_ops=14400),
            SingleCombo_packet=dict(member_calls=9,aggregator_calls=3,gold=1,scoring_ops=3),
            Full20_packet=dict(member_calls=18,aggregator_calls=60,gold=1,scoring_ops=60),
            compare_scope='full-system comparison, not same-predictor acquisition attribution; actual Tokens differ'),
        primary_comparisons=['HistoryFit-Full20 UCB minus frozenE7 Random PCA16 Ridge atGold60',
            'HistoryFit-Full20 UCB minus frozenE7 Random RBF atGold60'],
        stats=dict(reps=5000,seed=20261112,unit='original question with3 LLM repeats and3 algorithm seeds kept together',
            strata='outerfold x GSM/BBH',weights='GSM .5 BBH .5 question mean;27BBH subtask macro supplementary',
            multiplicity='Holm2 over primary contrasts; all bridge/mean-only contrasts exploratory',
            scope='existing train600 development, overlapping training histories'),
        member_scores=member_source,manual_text_source=text_source,train_sources=refs,
        upstream_data_gaps='Native capability inconsistency, code/paper differences and unavailable raw trajectories documented by audits; no silent fixes',
        physical_new_LLM_calls=0,new_data_generation=False,fresh_or_sealed_evaluated=False)
    put(PUBLIC/'CES_PROTOCOL.json',config)
    put(PRIVATE/'PROTOCOL_FROZEN.json',dict(status='FROZEN_BEFORE_CES_MIGRATION_EVALUATION',
        protocol_sha256=sha(PUBLIC/'CES_PROTOCOL.json'),manual_sha256=sha(PRIVATE/'MANUAL_INPUTS.npz'),
        E7_freeze_sha256=sha(PUBLIC/'E7_FREEZE_RECEIPT.json')))
    print(json.dumps(dict(status='FROZEN',protocol_sha256=sha(PUBLIC/'CES_PROTOCOL.json'),text_rows=600,
        paths=45,new_LLM_calls=0),ensure_ascii=False))


if __name__=='__main__':
    main()
