"""E7-A input-only candidate construction and evaluator-only opportunity checks.

All candidate sets are persisted before the evaluator indexes outer-E rewards.
Every collection contains the original F360 default. K remains three members.
"""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys
import json
import argparse
from time import perf_counter, process_time
import numpy as np

ROOT = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
WORK = ROOT / 'research/e7_opportunity_and_feedback_learning'
E5 = ROOT / 'research/e5_paired_gain_validation'
E6 = ROOT / 'research/e6_contrast_directed_feedback'
PUBLIC = WORK / 'public/a'
PRIVATE = WORK / 'private/a'
sys.path.insert(0, str(E5 / 'public'))
from e5_common import load_train, sha, put, argmax

MS = (1,3,5,19)
RANDOM_REPETITIONS = 200
TOLERANCE = 1e-10
BOOTSTRAPS = 5000
SEED = 20261112


def ranking(values, excluded):
    """Lexicographic choices among values within fixed tolerance of each max."""
    x = np.asarray(values, dtype=float).copy()
    if x.ndim == 1:
        x = x[None]
    out = np.zeros((len(x),19), dtype=np.int16)
    x[:,int(excluded)] = -np.inf
    for position in range(19):
        selected = np.argmax(x >= x.max(axis=1,keepdims=True)-TOLERANCE,axis=1)
        out[:,position] = selected
        x[np.arange(len(x)),selected] = -np.inf
    return out


def make_mask(ranks, default, m):
    shape = ranks.shape[:-1] + (20,)
    mask = np.zeros(shape,dtype=bool)
    mask[...,int(default)] = True
    np.put_along_axis(mask,ranks[...,:m],True,axis=-1)
    return mask


def freeze_candidates():
    # Presence of the parent's outcome-independent protocol is a dependency.
    if not (WORK/'public/E7_PROTOCOL.json').exists():
        raise RuntimeError('E7 protocol must be frozen before candidate construction')
    data,refs = load_train()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    for f in range(5):
        source = E5/f'private/base/BASE_f{f}.npz'
        with np.load(source) as a:
            F,E = a['F'],a['E']
            prediction = a['pred_E'][1]
            default = int(a['base'])
            candidate = a['candidate_E']
        ranks = ranking(prediction,default)
        assert np.array_equal(ranks[:,0],candidate)
        history_mean = data['Y'][F].mean(axis=(0,1))
        fixed_order = ranking(history_mean,default)[0]
        fixed_ranks = np.broadcast_to(fixed_order,(len(E),19)).copy()
        rng = np.random.default_rng(np.random.SeedSequence([SEED,f,751]))
        choices = np.array([s for s in range(20) if s!=default])
        random_ranks = np.asarray([[rng.permutation(choices) for _ in range(RANDOM_REPETITIONS)] for q in E],dtype=np.int16)
        p2_masks = np.asarray([make_mask(ranks,default,m) for m in MS])
        fixed_masks = np.asarray([make_mask(fixed_ranks,default,m) for m in MS])
        random_masks = np.asarray([make_mask(random_ranks,default,m) for m in MS])
        for index,m in enumerate(MS):
            assert np.all(p2_masks[index].sum(axis=-1)==m+1)
            assert np.all(fixed_masks[index].sum(axis=-1)==m+1)
            assert np.all(random_masks[index].sum(axis=-1)==m+1)
            assert np.all(p2_masks[index,:,default])
        with np.load(E6/f'private/a/fold{f}_policies.npz') as a:
            assert np.array_equal(E,a['E'])
            names = a['names'].tolist()
            A3 = a['masks'][names.index('A3')]
        target = PRIVATE/f'CANDIDATES_f{f}.npz'
        np.savez_compressed(target,F=F,E=E,prediction=prediction,default=default,candidate=candidate,
            p2_rank=ranks,fixed_rank=fixed_ranks,random_rank=random_ranks,
            p2_masks=p2_masks,fixed_masks=fixed_masks,random_masks=random_masks,
            direct_p2=argmax(prediction),A3_mask=A3,history_mean=history_mean)
        put(PRIVATE/f'FROZEN_CANDIDATES_f{f}.json',dict(status='FROZEN_BEFORE_E_REWARD_INDEXING',
            fold=f,source_base_sha256=sha(source),candidate_sha256=sha(target),
            protocol_sha256=sha(WORK/'public/E7_PROTOCOL.json'),source_sha256=sha(__file__),
            candidate_sizes=[m+1 for m in MS],random_repetitions=RANDOM_REPETITIONS,
            random_seed=[SEED,f,751],F_reward_only_for_fixed_candidates=True,E_rewards_used=False,
            tie='within1e-10 of current maximum, lowest action index first',
            random_design='independent per original input, one rank shared across candidate sizes and all generation repeats'))
    print(json.dumps(dict(status='CANDIDATES_FROZEN',folds=5)),flush=True)


def cross_repeat(rewards,masks):
    """Return held-out rewards, chosen action, and uniform tie expectation."""
    masks = np.asarray(masks,dtype=bool)
    repeat_axis = masks.ndim==3
    if not repeat_axis:
        masks = masks[:,None,:]
    n,draws,_ = masks.shape
    outcomes = np.zeros((n,draws,3))
    uniform = np.zeros_like(outcomes)
    actions = np.zeros_like(outcomes,dtype=np.int16)
    tie_counts = np.zeros_like(actions)
    for held in range(3):
        mean = rewards[:,[g for g in range(3) if g!=held]].mean(axis=1)
        allowed = np.where(masks,mean[:,None,:],-np.inf)
        tied = masks & (allowed >= allowed.max(axis=-1,keepdims=True)-TOLERANCE)
        selected = np.argmax(tied,axis=-1)
        test = rewards[:,held,:]
        outcomes[:,:,held] = np.take_along_axis(test[:,None,:],selected[:,:,None],axis=2)[:,:,0]
        uniform[:,:,held] = np.sum(tied*test[:,None,:],axis=-1)/tied.sum(axis=-1)
        actions[:,:,held] = selected
        tie_counts[:,:,held] = tied.sum(axis=-1)
    if not repeat_axis:
        return outcomes[:,0],actions[:,0],uniform[:,0],tie_counts[:,0]
    return outcomes,actions,uniform,tie_counts


def evaluate():
    started,cpu = perf_counter(),process_time()
    data,_ = load_train()
    allrows=[]
    Y=[]
    default_reward=[]
    direct=[]
    alternative=[]
    A3_rewards=[]
    A3_masks=[]
    datasets={name:[] for name in ('P2','Fixed','Random')}
    all_masks={name:[] for name in datasets}
    choices={name:[] for name in datasets}
    uniform={name:[] for name in datasets}
    ties={name:[] for name in datasets}
    for f in range(5):
        path=PRIVATE/f'CANDIDATES_f{f}.npz'
        receipt=json.loads((PRIVATE/f'FROZEN_CANDIDATES_f{f}.json').read_text())
        assert receipt['candidate_sha256']==sha(path)
        with np.load(path) as a:
            E=a['E']
            b=int(a['default'])
            candidate=a['candidate']
            p2=a['direct_p2']
            A3=a['A3_mask']
            masks={name:a[key] for name,key in [('P2','p2_masks'),('Fixed','fixed_masks'),('Random','random_masks')]}
        put(PRIVATE/f'EVALUATOR_ACCESS_f{f}.json',dict(role='post-freeze outer-E diagnostic evaluator',
            operation='index existing complete20 x3 score block',question_indices=E.tolist(),
            candidate_sha256=receipt['candidate_sha256'],candidate_freeze_verified_before_access=True,
            application_array_access_log=True,OS_file_access_audit=False))
        rewards=data['Y'][E]
        allrows.append(E)
        Y.append(rewards)
        default_reward.append(rewards[:,:,b])
        direct.append(rewards[np.arange(len(E))[:,None],np.arange(3)[None,:],p2[:,None]])
        alternative.append(rewards[np.arange(len(E))[:,None],np.arange(3)[None,:],candidate[:,None]])
        A3_rewards.append(np.where(A3[:,None],alternative[-1],default_reward[-1]))
        A3_masks.append(A3)
        for name,collection in masks.items():
            scores=[]; selected=[]; tieuniform=[]; counts=[]
            for mask in collection:
                score,choice,ut,tc=cross_repeat(rewards,mask)
                scores.append(score);selected.append(choice);tieuniform.append(ut);counts.append(tc)
            datasets[name].append(np.asarray(scores))
            choices[name].append(np.asarray(selected))
            uniform[name].append(np.asarray(tieuniform))
            ties[name].append(np.asarray(counts))
            all_masks[name].append(collection)
    rows=np.concatenate(allrows)
    assert len(set(rows))==600
    Y=np.concatenate(Y)
    baseline=np.concatenate(default_reward)
    direct=np.concatenate(direct)
    alternative=np.concatenate(alternative)
    A3_reward=np.concatenate(A3_rewards)
    A3=np.concatenate(A3_masks)
    for group in datasets:
        datasets[group]=np.concatenate(datasets[group],axis=1)
        choices[group]=np.concatenate(choices[group],axis=1)
        uniform[group]=np.concatenate(uniform[group],axis=1)
        ties[group]=np.concatenate(ties[group],axis=1)
        all_masks[group]=np.concatenate(all_masks[group],axis=1)
    opportunity=(baseline==0)&np.any(Y==1,axis=-1)
    retention={}
    for group,masks in all_masks.items():
        rr=[]
        for mask in masks:
            if group=='Random':
                covered=np.einsum('nra,nla->nrl',mask.astype(np.int16),Y)>0
                rr.append((covered*opportunity[:,None,:]).mean(axis=1))
            else:
                covered=np.einsum('na,nla->nl',mask.astype(np.int16),Y)>0
                rr.append(covered*opportunity)
        retention[group]=np.asarray(rr)
    scores={group:array.mean(axis=2) if group=='Random' else array for group,array in datasets.items()}
    uniform_scores={group:array.mean(axis=2) if group=='Random' else array for group,array in uniform.items()}
    labels=[f'{group}-Top{m}' for group in ('P2','Fixed','Random') for m in MS]
    outcomes=np.concatenate([scores[group] for group in ('P2','Fixed','Random')],axis=0)
    uniform_outcomes=np.concatenate([uniform_scores[group] for group in ('P2','Fixed','Random')],axis=0)
    retained=np.concatenate([retention[group] for group in ('P2','Fixed','Random')],axis=0)
    full=outcomes[3]
    assert np.array_equal(outcomes[3],outcomes[7]) and np.array_equal(outcomes[3],outcomes[11])
    task=data['task'][rows].astype(str)
    sub=data['sub'][rows].astype(str)
    fold=np.repeat(np.arange(5),120)
    np.savez_compressed(PRIVATE/'OPPORTUNITY_PAIRED.npz',rows=rows,Y=Y,baseline=baseline,direct=direct,alternative=alternative,
        A3_reward=A3_reward,A3_mask=A3,opportunity=opportunity,outcomes=outcomes,uniform_outcomes=uniform_outcomes,
        labels=np.asarray(labels),retained=retained,task=task,sub=sub,fold=fold,
        P2_choices=choices['P2'],Fixed_choices=choices['Fixed'],Random_choices=choices['Random'],
        P2_tie_counts=ties['P2'],Fixed_tie_counts=ties['Fixed'],Random_tie_counts=ties['Random'],
        random_draw_scores=datasets['Random'],random_draw_uniform_scores=uniform['Random'])
    bootstrap=[]
    rng=np.random.default_rng(np.random.SeedSequence([SEED,752]))
    for rep in range(BOOTSTRAPS):
        ix=np.concatenate([rng.choice(cell,len(cell),replace=True) for f in range(5) for t in ('gsm8k','bbh')
            for cell in [np.flatnonzero((fold==f)&(task==t))]])
        acc=outcomes[:,ix].mean(axis=(1,2))
        base=baseline[ix].mean()
        fullacc=full[ix].mean()
        denominator=opportunity[ix].sum()
        retain=retained[:,ix].sum(axis=(1,2))/denominator if denominator else np.full(len(labels),np.nan)
        bootstrap.append(np.column_stack((acc,acc-base,acc-fullacc,retain)))
    bootstrap=np.asarray(bootstrap)
    np.savez_compressed(PRIVATE/'OPPORTUNITY_BOOTSTRAP.npz',labels=labels,bootstrap=bootstrap,
        seed=[SEED,752],strata='fold x GSM/BBH, original questions, all3 repeat outcomes travel together')
    denominator=int(opportunity.sum())
    result=dict(status='COMPLETE',N_original_questions=600,LLM_generation_repeats=3,
        generation_record_count=1800,random_sets_per_question=RANDOM_REPETITIONS,candidate_sizes=[m+1 for m in MS],
        source='E5 original F360 predictor/default, complete train600 E folds; no new gate fitting',
        main_mixture='GSM/BBH each50%, BBH question mean',
        baseline_accuracy=float(baseline.mean()),full20_cross_repeat_accuracy=float(full.mean()),
        observable_opportunity_records=denominator,original_questions_with_any_opportunity=int(np.any(opportunity,axis=1).sum()),
        mixture_roles='Candidate oracle coverage and cross-repeat feedback diagnostics, not input-only deployment accuracy',
        input_only_routes=dict(P2_direct_argmax20_accuracy=float(direct.mean()),
            always_execute_nondefault_candidate_accuracy=float(alternative.mean()),E6_A3_accuracy=float(A3_reward.mean())),
        configurations=[],baselines={},stage_cpu_s=process_time()-cpu,stage_wall_s=perf_counter()-started,new_llm_calls=0)
    for li,label in enumerate(labels):
        row=scope(outcomes[li],retained[li],opportunity,baseline,full)
        row.update(name=label,scope='main',uniform_tie_accuracy=float(uniform_outcomes[li].mean()),
            uniform_tie_gain_default=float(uniform_outcomes[li].mean()-baseline.mean()),
            accuracy_ci95=np.quantile(bootstrap[:,li,0],[.025,.975]).tolist(),
            gain_default_ci95=np.quantile(bootstrap[:,li,1],[.025,.975]).tolist(),
            gap_to_full20_ci95=np.quantile(bootstrap[:,li,2],[.025,.975]).tolist(),
            retention_ci95=np.nanquantile(bootstrap[:,li,3],[.025,.975]).tolist(),
            task={t:scope(outcomes[li,task==t],retained[li,task==t],opportunity[task==t],baseline[task==t],full[task==t]) for t in ('gsm8k','bbh')},
            BBH_subtask={s:scope(outcomes[li,(task=='bbh')&(sub==s)],retained[li,(task=='bbh')&(sub==s)],
                opportunity[(task=='bbh')&(sub==s)],baseline[(task=='bbh')&(sub==s)],full[(task=='bbh')&(sub==s)]) for s in sorted(set(sub[task=='bbh']))})
        row['GSM_BBH_subtask_macro_accuracy']=float(.5*row['task']['gsm8k']['accuracy']+.5*np.mean([v['accuracy'] for v in row['BBH_subtask'].values()]))
        if label.startswith('Random'):
            ri=MS.index(int(label.split('Top')[1]))
            draw_acc=datasets['Random'][ri].mean(axis=(0,2))
            row['random_set_MC']=dict(draws=RANDOM_REPETITIONS,conditional_fixed_cache_mean=float(draw_acc.mean()),
                conditional_MC_SE=float(draw_acc.std(ddof=1)/np.sqrt(RANDOM_REPETITIONS)),
                unit='sampling-set draw aggregated over fixed600 inputs, not new independent questions')
        result['configurations'].append(row)
    for name,values in [('default',baseline),('P2_direct_argmax20',direct),('always_nondefault_candidate',alternative),('E6_A3',A3_reward)]:
        result['baselines'][name]=dict(accuracy=float(values.mean()),gain_default=float(values.mean()-baseline.mean()),
            seed_accuracy=values.mean(axis=0).tolist(),task={t:float(values[task==t].mean()) for t in ('gsm8k','bbh')},
            BBH_subtask={s:float(values[(task=='bbh')&(sub==s)].mean()) for s in sorted(set(sub[task=='bbh']))})
    lost=opportunity&(alternative==0)
    retained_opportunity=opportunity&(alternative==1)
    missed=retained_opportunity&(~A3[:,None])
    fixed=retained_opportunity&A3[:,None]
    harm=(baseline==1)&(alternative==0)&A3[:,None]
    assert np.array_equal(lost.astype(int)+missed+fixed,opportunity.astype(int))
    result['two_choice_decomposition']=dict(all20_opportunity_records=int(opportunity.sum()),
        lost_before_gate_records=int(lost.sum()),retained_in_candidate_records=int(retained_opportunity.sum()),
        retained_but_A3_declined_records=int(missed.sum()),A3_corrected_records=int(fixed.sum()),A3_harmed_records=int(harm.sum()),
        lost_before_gate_fraction=float(lost.sum()/opportunity.sum()) if opportunity.sum() else None,
        conditional_A3_acceptance_of_correct_alternative=float(fixed.sum()/retained_opportunity.sum()) if retained_opportunity.sum() else None,
        task={t:dict(opportunity_records=int(opportunity[task==t].sum()),lost_before_gate=int(lost[task==t].sum()),
                     retained_but_declined=int(missed[task==t].sum()),corrected=int(fixed[task==t].sum()),harmed=int(harm[task==t].sum())) for t in ('gsm8k','bbh')},
        normalization='Counts over600 questions x3 frozen generation repetitions, divide3 for expected-question equivalents')
    put(PUBLIC/'E7_A_RESULTS.json',result)
    report(result)
    print(json.dumps(dict(status='COMPLETE',opportunities=denominator,
        retention=[r['opportunity_retention'] for r in result['configurations'][:4]],
        cross_repeat_accuracy=[r['accuracy'] for r in result['configurations'][:4]],
        default=result['baseline_accuracy'],decomposition=result['two_choice_decomposition'])),flush=True)


def scope(outcomes,retained,opportunity,baseline,full):
    denominator=float(opportunity.sum())
    return dict(original_questions=len(baseline),generation_records=int(np.size(baseline)),
        original_questions_with_opportunity=int(np.any(opportunity,axis=1).sum()),
        original_questions_with_positive_retention_support=int(np.any(retained>0,axis=1).sum()),
        observable_opportunity_records=int(denominator),retained_opportunity_records=float(retained.sum()),
        opportunity_retention=float(retained.sum()/denominator) if denominator else None,
        accuracy=float(outcomes.mean()),gain_default=float(outcomes.mean()-baseline.mean()),
        gap_to_full20=float(outcomes.mean()-full.mean()),seed_accuracy=outcomes.mean(axis=0).tolist(),
        correction=float(np.mean((outcomes-baseline).clip(0))),harm=float(np.mean((baseline-outcomes).clip(0))))


def pp(value):
    return f'{100*value:+.4f}'


def report(result):
    rows=[]
    for row in result['configurations']:
        retention='未定义' if row['opportunity_retention'] is None else f"{100*row['opportunity_retention']:.2f}%"
        lo,hi=row['gain_default_ci95']
        rows.append(f"| {row['name']} | {retention} | {100*row['accuracy']:.4f}% | {pp(row['gain_default'])} [{pp(lo)}, {pp(hi)}] | {pp(row['gap_to_full20'])} | {100*row['uniform_tie_accuracy']:.4f}% |")
    d=result['two_choice_decomposition']
    text=f'''# E7-A 候选机会保留

## 研究问题

20个三模型组合的可观察纠错机会有多少进入默认＋P2候选，候选范围内剩余机会又有多少被E6的A3判断遗漏。

## 实验设置

沿用E5五折F360训练的默认及P2冻结预测，600道不同外折题，三个生成种子42、137、2024。每题始终执行一个三成员组合。候选集合大小2、4、6、20，默认始终在集合中。P2排列采用每次最高分1e-10内取最小组合索引。固定集合按F区真实平均成绩选取，随机集合每题200份、与标签无关，同一随机排列在候选规模间嵌套。

候选集合、完整P2排序及随机数配置先持久保存，评价器随后读取E区奖励。本文计数包含600道原题和1800条生成重复记录，重复记录没有作为1800道独立题用于统计。

## 对照与定义

当前纠错机会指默认答错且20组合至少一个答对，保留表示集合中至少一个答对。跨重复诊断用两个种子在集合内选最高平均分组合，再于第三种子评价，轮流留出三次。补充对全部并列最优组合均匀选择的期望成绩。选择规则不读取留出重复。两类诊断都使用真实同题反馈，具有部署前不可得的信息。

## 数值结果

全集存在{result['observable_opportunity_records']}条可纠错生成记录，涉及{result['original_questions_with_any_opportunity']}道不同题。默认准确率{100*result['baseline_accuracy']:.4f}%。

| 候选集合 | 当前纠错机会保留率 | 跨重复准确率 | 相对默认及95%CI，百分点 | 候选减全20，百分点 | 并列均匀期望准确率 |
|---|---:|---:|---|---:|---:|
{chr(10).join(rows)}

输入直接P2在20组合取最高分的准确率为{100*result['input_only_routes']['P2_direct_argmax20_accuracy']:.4f}%，始终执行冻结非默认候选c为{100*result['input_only_routes']['always_execute_nondefault_candidate_accuracy']:.4f}%。两者单列，避免将非默认候选与实际直接选队混同。E6-A3为{100*result['input_only_routes']['E6_A3_accuracy']:.4f}%。

## 机会在哪里丢失

{d['all20_opportunity_records']}条全集机会中，{d['lost_before_gate_records']}条在进入二选一前已丢失，比例{100*d['lost_before_gate_fraction']:.2f}%。其余{d['retained_in_candidate_records']}条候选内机会，A3拒绝{d['retained_but_A3_declined_records']}条并纠正{d['A3_corrected_records']}条。另有{d['A3_harmed_records']}条默认正确记录被破坏。三类机会分解逐记录精确覆盖全集机会，破坏另行统计。以上为生成记录数，除以3可转为期望题数。

这些量区分候选遗漏与判断遗漏，无法直接把某一种诊断的事后潜力当成题目输入模型已经学到的收益。GSM8K、BBH及27个BBH子任务全部保留于E7_A_RESULTS.json。

## 统计结论

区间来自5000次原题配对bootstrap，按外折×GSM/BBH分层，同题三生成、全部候选及对照一起抽样。保留率在每次抽样中重新计算机会分母，分母为0记未定义。主表为描述性95%区间，不属于E7四项确认比较。随机集合结果先按200份集合取期望，再以原题统计，其Monte Carlo误差另外保存。

更大候选集合的当前覆盖机会单调不减，跨重复评价不保证单调。并列规则可能改变有限重复诊断，因此均匀并列期望同时报告。全20跨重复结果不等于总体能力上界。

## 优势与风险

优势是完全复用冻结预测、精确记录候选遗漏、匹配规模及同题留种子评价。风险包括默认和预测器来自F360、外折训练历史重叠、旧开发题复用、有限三种子、同题排序并列，以及后验反馈诊断和真实输入路由的信息权限差异。

## 失败原因与下一步

候选范围与换队判断分别造成可观察损失。扩大候选集合本身没有改变P2分数的argmax，也没有形成新算法。E7-B将直接检查输入信息与更多独立题目的学习能力，E7-C按固定学习器检验真实反馈更新，均按原协议继续执行。
'''
    (WORK/'public/01_CANDIDATE_OPPORTUNITY.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('freeze','evaluate'))
    args=parser.parse_args()
    {'freeze':freeze_candidates,'evaluate':evaluate}[args.stage]()
