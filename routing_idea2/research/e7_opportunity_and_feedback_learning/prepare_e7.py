"""Persist all E7 roles and operational choices before new evaluation."""
import os
from pathlib import Path
import hashlib
import json
from datetime import datetime,timezone
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e7_opportunity_and_feedback_learning'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    protocol=WORK/'public/E7_PROTOCOL.json'
    if protocol.exists():
        raise RuntimeError('Frozen protocol already exists')
    bundle=json.loads((ROOT/'artifacts/private/routing_redesign_20261104/ff_v1/BUNDLE_TRAINVAL.json').read_text('utf-8-sig'))
    source=Path(bundle['inputs']['selector_train']['path'])
    if not source.is_absolute():
        source=ROOT/source
    assert sha(source)==bundle['inputs']['selector_train']['sha256']
    with np.load(source,allow_pickle=False) as a:
        ids=a['question_ids'].astype(str)
        task=a['task_labels'].astype(str)
        sub=a['subtask_labels'].astype(str)
    frozen={}
    def ordered(rows,prefix):
        return sorted(map(int,rows),key=lambda q:hashlib.sha256((prefix+'|'+ids[q]).encode()).hexdigest())
    for f in range(5):
        with np.load(ROOT/f'research/e5_paired_gain_validation/private/base/BASE_f{f}.npz') as a:
            E=a['E'].copy()
        H=np.setdiff1d(np.arange(600),E)
        gsm=ordered(H[task[H]=='gsm8k'],'E7-I-v1')
        groups=[ordered(H[(task[H]=='bbh')&(sub[H]==s)],'E7-I-v1') for s in sorted(set(sub[H][task[H]=='bbh']))]
        bbh=[group[i] for i in range(max(map(len,groups))) for group in groups if i<len(group)]
        I=np.sort(np.r_[gsm[:120],bbh[:120]]).astype(int)
        Ug=ordered(np.setdiff1d(gsm,I),'E7-U-v1')
        Ub=ordered(np.setdiff1d(bbh,I),'E7-U-v1')
        U=np.asarray([q for pair in zip(Ug,Ub) for q in pair],int)
        refgsm=ordered(I[task[I]=='gsm8k'],'E7-ref-v1')[:27]
        refbbh=[ordered(I[(task[I]=='bbh')&(sub[I]==s)],'E7-ref-v1')[0] for s in sorted(set(sub[I][task[I]=='bbh']))]
        refs=np.asarray(refgsm+refbbh,int)
        weights=np.r_[np.full(27,.5/27),[np.sum((task[I]=='bbh')&(sub[I]==sub[q]))/240 for q in refbbh]]
        assert len(I)==len(U)==240 and len(E)==120 and len(refs)==54
        assert len(set(I)|set(U)|set(E))==600
        assert set(refs)<=set(I)
        assert np.isclose(weights.sum(),1.) and np.isclose(weights[:27].sum(),.5)
        for key,value in dict(H=H,I=I,U=U,E=E,refs=refs,weights=weights).items():
            frozen[f'f{f}_{key}']=value
    path=WORK/'private/E7_SPLITS_FROZEN.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**frozen)
    config=dict(schema='mab_e7_v1',frozen_utc=datetime.now(timezone.utc).isoformat(),
        branch='codex/e7-opportunity-and-feedback-learning',base_commit='0c34a38bede9fcf46359316058759b27fa9f5fea',
        N=6,K=3,actions=20,generation_seeds=[42,137,2024],new_llm_calls=0,
        input_scope='train600 only, fixed MiniLM embeddings and true V2 aggregate scores',
        split='five original E120 outer folds; H480 complement; I240 deterministic ID-hash broad-task/subtask-balanced; U240 deterministic ID-hash task-interleaved nested order',
        split_sha256=sha(path),input_sha256=sha(source),
        A=dict(candidate_m=[1,3,5,19],random_sets_per_question=200,
            tolerance=1e-10,tie='lexicographic combination ID; uniform-among-maximizers supplementary',
            reference='E5 F360-only default and frozen P2 predictions, not the new I default',
            random_seed=20261112,opportunity='generation-record denominator, original-question counts separately; undefined if zero'),
        B=dict(models=['P1_PCA16_Ridge','P2_Full384_Ridge','P3_RBF_KRR'],sizes=[240,300,360,480],
            default='I240-only combination mean argmax, remains fixed on curve',
            ridge_alpha_raw=100.,ridge_alpha_mean=100/3,
            ridge_gp=dict(kernel='linear design dot product /100',noise_mean=1/3,offset=0.),
            full_ridge_features='unchanged normalized full MiniLM vector plus penalized constant, no extra scaling',
            P3=dict(kernel='exp(-squared_distance/(2*sigma^2))',sigma='median positive Euclidean distance inside I',
                lambdas=[.1,1.,10.],selection='I240 three question-grouped broad-task-stratified folds, route reward, tie stronger lambda',
                CV='inner-training bandwidth and mean; outer final refit on I',offset='I240 combination means fixed thereafter',
                observation='one original question three-repeat average'),
            permutations=list(range(100,120)),shuffle='within semantic task/subtask; I and current U prefix separately, whole20x3 reward block',
            primary_content='P3 original480 minus mean of20 shuffled480',
            synthetic_controls=['all combinations equal population means','known feature determines optimal arm'],
            synthetic_role='software qualification only'),
        C=dict(learners=['P1_PCA16_Ridge','P3_RBF_KRR'],strategies=['Random','ParameterIG','DecisionAware'],
            algorithm_seeds=[17,29,43],checkpoints=[0,20,40,60],primary_gold=60,qualification_packets=10,
            qualification='outerfold0, full three-strategy pool/update test, no E reward scoring before actor freeze',
            full_feedback_packet=dict(gold=1,member_calls=18,aggregate_calls=60,raw_scores=60,mean_targets=20),
            reference_count=54,reference_inputs='I240 only,27 GSM and1 per27BBHsubtask',
            reference_weights='GSM .5/27; BBH each subtask I frequency/240, conditional approximation to broad-task question mixture',
            fantasies=64,antithetic_pairs=32,normal_dimension=20,
            fantasy_seed='SeedSequence([20261112,fold,algorithm_seed,step,731]); same20-vector across all references and candidates within step',
            KG='E weighted max updated mean on frozen I reference minus current max; same posterior/noise used by real update',
            IG='10*log1p(latent_variance/noise_mean)',
            prior='independent20 outputs, same scalar kernel/noise; work model, no coverage claim',
            tie='score tolerance1e-12 then fixed seed-specific input-only U priority',
            cost='I-only historical estimates if needed; actual source lengths only accessed after paid purchase; no cost normalization',
            path_limit=90,zero_feedback='initial predictions shared across methods',
            update='one full20-component package per new original question, no repeated purchase'),
        statistics=dict(primary_comparisons=['P3 480 minus240','P3 original480 minus shuffled480 mean',
                         'P3 DecisionAware minus Random at gold60','P3 DecisionAware minus ParameterIG at gold60'],
            bootstrap_reps=5000,seed=20261112,strata='outerfold x GSM/BBH',
            weight='GSM andBBH .5 each; BBH question mean primary,27subtask macro supplementary',
            multiplicity='Holm4 centered two-sided paired bootstrap p, percentile95CI and Bonferroni98.75CI',
            repeated_measure='original question includes3 generation repetitions and all algorithm/permutation repetitions, never extra independent n',
            limitation='conditional development inference, overlapping outer histories, no fresh independent confirmation'),
        boundaries=['No LLM training or generation','No aggregator/scorer change','No old report edits','No MAML/transformer/Mamba/large tuning','No frozen-policy-family selection','No fresh400/sealed200 outcomes'],
        report_files=['01_CANDIDATE_OPPORTUNITY.md','02_INPUT_SIGNAL_AND_LEARNING_CURVES.md','03_ACTIVE_FEEDBACK_LEARNING.md',
            '04_COST_AND_STATISTICS.md','05_E7_RESEARCH_DECISION.md','RELATED_WORK_MAPPING.md','E7_PROTOCOL.json','E7_COMPLETE_RESULTS.json','REPRODUCE.md'])
    put(protocol,config)
    put(WORK/'private/PROTOCOL_FROZEN.json',dict(status='FROZEN_BEFORE_E7_EVALUATION',
        protocol_sha256=sha(protocol),split_sha256=sha(path),utc=config['frozen_utc'],
        decoded_payloads=['train600 inputs only','E5 E fold indices only'],E7_evaluation_rewards_read=False))
    print(json.dumps(dict(status='FROZEN',protocol_sha256=sha(protocol),split_sha256=sha(path))))


if __name__=='__main__':
    main()
