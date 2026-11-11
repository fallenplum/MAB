"""E6-A report and independent, post-freeze descriptive summaries."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from time import process_time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import run_e6_a as a


def format_ci(record):
    lo, hi = record['interval']
    return f"{100 * record['point']:+.4f} [{100 * lo:+.4f}, {100 * hi:+.4f}]"


def centered_p(draws, point):
    return float((1 + np.sum(np.abs(draws - point) >= abs(point) - 1e-15)) / (len(draws) + 1))


def holm(values):
    order = np.argsort(values)
    result = np.zeros(len(values))
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(values) - rank) * values[index]))
        result[index] = previous
    return result.tolist()


def main():
    summary_path = a.PUBLIC / 'A_RESULTS.json'
    result = json.loads(summary_path.read_text(encoding='utf-8'))
    by_name = {r['configuration']: r for r in result['configurations']}
    data, refs = a.load_train()
    seen_questions = set()
    direction_questions = set()
    auxiliary_questions = set()
    per_fold = []
    for fold in range(5):
        with np.load(a.E5 / f'private/base/BASE_f{fold}.npz') as archive:
            F, G, E = archive['F'], archive['G'], archive['E']
            base = int(archive['base'])
            candidate_g, candidate_e = archive['candidate_G'], archive['candidate_E']
            pred_e = archive['pred_E'][1]
        d = data['Y'][G[:, None], np.arange(3)[None, :], candidate_g[:, None]] - data['Y'][G, :, base]
        aux = data['Y'][G] - data['Y'][G, :, base][:, :, None]
        seen_questions.update(G.tolist())
        direction_questions.update(G[np.any(d != 0, axis=1)].tolist())
        auxiliary_questions.update(G[np.any(aux != 0, axis=(1, 2))].tolist())
        receipt = json.loads((a.PRIVATE / f'FROZEN_A_f{fold}.json').read_text(encoding='utf-8'))
        with np.load(a.PRIVATE / f'FROZEN_A_f{fold}.npz') as archive:
            phi_e = (data['E'][E] - archive['pca_mean']) @ archive['pca_components'].T
            embedding = archive['combo_embedding']
            expected_scores = archive['scores']
        cpu = {}
        for index, name in enumerate(('A1', 'A2', 'A3', 'A4')):
            head = receipt['heads'][name]
            head['beta'] = np.asarray(head['beta'])
            head['mean'] = np.asarray(head['mean'])
            head['scale'] = np.asarray(head['scale'])
            repeats = 1000
            started = process_time()
            for _ in range(repeats):
                # Existing PCA8 and P2 values are shared inputs to the incremental head.
                features = a.features(pred_e, base, candidate_e, phi_e, embedding, name)[:, 0]
                predicted = a.predict_head(head, features)
            elapsed = process_time() - started
            assert np.max(np.abs(predicted - expected_scores[index])) < 1e-12
            cpu[name] = dict(batch_repetitions=repeats, original_questions_per_batch=len(E),
                process_cpu_s=elapsed, cpu_s_per_question=elapsed / repeats / len(E),
                scope='gate feature assembly and direction head from saved PCA8 and saved P2 values',
                excludes=['MiniLM encoder', 'P2 prediction', 'PCA8 transform'])
        current = receipt['G_pair_feedback_cost']
        full = receipt['G_all_feedback_cost']
        delta = {key: full[key] - current[key] for key in current}
        per_fold.append(dict(fold=fold, shared_F360_full_information=a.resources(data, F, [range(20)] * len(F), gold=len(F)),
            G120_pair_feedback=current, G120_all_feedback=full, A4_increment_over_pair=delta,
            gate_cpu=cpu, effective_candidate_direction_questions=int(np.any(d != 0, axis=1).sum()),
            effective_auxiliary_direction_questions=int(np.any(aux != 0, axis=(1, 2)).sum())))
    with np.load(a.PRIVATE / 'PAIRED_E_RESULTS.npz') as archive:
        names = list(archive['names'])
        gain = archive['bootstrap_gain']
        acc = archive['bootstrap_accuracy']
        task = archive['bootstrap_matched_task']
        index = names.index('A3')
        extras = [
            ('A3_minus_default', by_name['A3']['gain']['point'], gain[:, index]),
            ('A3_minus_A1', result['contrasts']['A3_minus_A1']['point'], acc[:, index] - acc[:, names.index('A1')])]
        raw = [centered_p(draws, point) for _, point, draws in extras]
        adjusted = holm(raw)
        exploratory = {name: dict(point=point, raw_two_sided_bootstrap_p=p, holm_two_comparisons=q)
            for (name, point, draws), p, q in zip(extras, raw, adjusted)}
        mask = archive['masks'][index]
        expected = np.zeros(len(mask))
        for fold in range(5):
            for task_name in ('gsm8k', 'bbh'):
                cell = (archive['folds'] == fold) & (archive['tasks'] == task_name)
                expected[cell] = np.mean(mask[cell])
        contribution = (mask - expected) * archive['D'].mean(axis=1)
        point = by_name['A3']['matched_excess']['task']['point']
        assert abs(contribution.mean() - point) < 1e-14
        a.atomic_npz(a.PRIVATE / 'PRIMARY_A_CONTRAST.npz', rows=archive['rows'],
            question_contribution=contribution, observed=point,
            bootstrap=gain[:, index] - task[:, index],
            raw_switch_gain_contribution=mask * archive['D'].mean(axis=1),
            matched_task_acceptance_probability=expected)
    diagnostics = dict(status='COMPLETE', unique_G_questions_across_overlapping_folds=len(seen_questions),
        unique_G_candidate_direction_questions_across_folds=len(direction_questions),
        unique_G_auxiliary_direction_questions_across_folds=len(auxiliary_questions),
        per_fold=per_fold, supplementary_key_contrasts=exploratory,
        gate_cpu_note='batch CPU of incremental gate only. P2 and encoder excluded, source inference costs counterfactual historic executions.',
        fresh400_evaluated=False, sealed200_evaluated=False, new_physical_llm_calls=0)
    a.put(a.PUBLIC / 'A_SUPPORT_AND_COST_DIAGNOSTICS.json', diagnostics)
    rows = []
    for name in ('A0', 'A1', 'A2', 'A3', 'A4', 'gap', 'default', 'all'):
        r = by_name[name]
        support = []
        for fold in result['folds']:
            field = fold['training']['candidate_direction_support']
            key = 'all_nineteen_nondefault' if name == 'A4' else 'current_candidate'
            support.append(field[key]['effective_direction_questions'])
        support_text = f"{min(support)}—{max(support)}/120" if name not in ('default', 'all', 'gap') else '固定规则'
        rows.append(f"| {name} | {support_text} | {100 * r['accuracy']['point']:.4f}% | {r['switch_questions']} | {r['corrected']:.3f}/{r['harmed']:.3f} | {format_ci(r['gain'])} | {format_ci(r['matched_excess']['task'])} |")
    a3, a4 = by_name['A3'], by_name['A4']
    positive = sum(value > 0 for value in a3['seed_gain'])
    main_supported = a3['matched_excess']['task']['interval'][0] > 0
    text = f"""# E6-A 换队信息与配对监督

## 研究问题

固定默认组合与P2候选，检验题目信息、组合身份和更多同题组合监督能否改善有益切换判断。A3是预先指定主候选，A4是增加监督的消融。

## 实验设置

沿用E5五折F360/G120/E120。每折仅F拟合MiniLM-PCA8和组合表示，仅G选择正则与训练方向头。E不参与拟合、正则或阈值选择。每道题的三个生成重复始终共用一个换队决定。全部E决策在独立评价器访问评分前持久保存。

A1使用两个P2值，A2增加20维组合差身份，A3增加8维文本和16维交互，总维度46。A4维持A3结构，训练目标为当前候选损失的0.5加19个非默认候选平均损失的0.5。其推理候选仍是原P2推荐。

Logistic严格以原题数和三个重复归一化，零差值保留在分母中。截距并入参数且一并施加正则。λ取0.01、0.1、1，G内原题分组、任务分层三折，以配对服务收益选择，并列取较强正则。服务阈值固定0。缺类采用固定Beta(1,1)常数回退。没有类别平衡。

组合表示取F行中心化平均奖励的前两维右奇异向量，单位正交尺度，固定最大绝对载荷符号。主模型不输入真实任务或子任务身份。标准化只拟合内训练题，A4按0.5及0.5/19的特征混合权重拟合标准化器。

## 对照

A0与gap复用E5冻结决策。默认与全切换均保留。随机对照保持外折内总体、GSM/BBH或BBH子任务切换数，只随机化换队题目。报告条件随机门控收益的精确期望，候选组合保持逐题不变。

## 数值结果

下列收益与区间单位为百分点。纠错和破坏为三个生成重复的题数均值，因此允许非整数。独立方向支持范围为每折G120中的原题数。

| 配置 | 有效方向支持题 | E准确率 | 切换题数 | 纠错/破坏 | 相对默认及95%CI | 超额任务匹配随机及95%CI |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

A3相对A1的准确率差为 {format_ci(result['contrasts']['A3_minus_A1'])} 个百分点。A4相对A3为 {format_ci(result['contrasts']['A4_minus_A3'])} 个百分点。

A3三生成种子的相对默认收益依次为 {', '.join(f'{100*v:+.4f}' for v in a3['seed_gain'])} 个百分点，其中{positive}个方向为正。GSM8K收益为 {100*a3['tasks']['gsm8k']['gain']:+.4f}，BBH收益为 {100*a3['tasks']['bbh']['gain']:+.4f} 个百分点。全部27个BBH子任务及各方法逐种子结果保存在A_RESULTS.json。

A3的ρ为 {a3['rho']:.6f}，η为 {a3['eta'] if a3['eta'] is not None else '未定义'}。A4的ρ为 {a4['rho']:.6f}，η为 {a4['eta'] if a4['eta'] is not None else '未定义'}。ρ按三个重复的原题平均计算，η条件于已切换且配对输出评分不相同。

## 统计结论

题级bootstrap重复5000次，分层为外折×GSM/BBH，同题三个生成结果和全部策略一起重抽样。每次重新计算匹配门控切换数。A3相对任务匹配随机的单项95%CI{'严格高于零' if main_supported else '包含零或低于零'}。统一三项主比较的多重校正由E6总统计报告完成，此表为单项95%区间。

A3相对默认及A1作为两个额外关键比较，单列Holm结果，原始结果见A_SUPPORT_AND_COST_DIAGNOSTICS.json。当前区间是给定已训练策略的条件开发区间，没有重训重叠外折历史，不能代表独立确认性实验。

H1目前没有得到支持。A3增加题目信息后的准确率与两分数A1相同，相对任务匹配随机的超额收益点估计为负。H2增加了有效方向支持，但A4相对A3的收益区间跨零，尚未建立更多同题组合标签能够改善实际换队决策。此结论限于已冻结的线性头、特征和当前开发样本，不能据此断言问题文本不存在可利用信息。

## 反馈支持与成本

每折G固定120道不同问题。五折G合计使用{len(seen_questions)}道不同题，当前候选产生方向反馈的独立问题并集为{len(direction_questions)}，使用19个组合后为{len(auxiliary_questions)}。外折G有重叠，不能把各折支持数相加充当独立样本。

A1—A3每折有120×3=360条候选配对记录。A4同样只有120道原题，另有120×19×3=6840条辅助配对记录，其余组合与默认的同题比较可共享成员、聚合和一次gold。A4总监督包括当前记录与辅助记录的加权重复，不能把条数当作独立题数。

对应源执行成本逐折见A_SUPPORT_AND_COST_DIAGNOSTICS.json。A4每折比候选配对需要额外120×18×3=6480次组合聚合，Gold新增为0，成员只按实际并集计费。现有缓存已经完成这些调用，本轮新增物理LLM调用为0。F360完全监督基础预测器与全部门控共享，不能反复计为各模型新支出。

增量门控CPU计时从已有PCA8和P2预测开始，包含特征组装和Logistic头，排除编码器、PCA变换及P2预测，所有范围在成本JSON中明确。源模型wall-time仅为历史执行墙钟代理，未冒充GPU kernel时间。

## 失败原因与局限

增加组合身份和文本特征仅扩充可观测信息，不能保证未见题收益。A4增加了同题组合标签，独立问题数没有增加，新增标签中的高度相关性继续存在。收益区间与任务匹配超额需要共同判断，训练损失下降不足以建立有益换队的泛化。五折训练历史重叠，全部600题是旧开发题，本轮没有评价fresh400或sealed200。

## 下一步

E6-C只使用已冻结的小型策略族执行有限池反馈分配，按任务书完整保留默认和全切换。A4不进入主预算族。A阶段结果与差值分配机制分别评价，随后在总报告中联合解释H1、H2、H3。
"""
    (a.PUBLIC / '01_SWITCH_INFORMATION_AND_SUPERVISION.md').write_text(text, encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE', report='01_SWITCH_INFORMATION_AND_SUPERVISION.md',
        unique_training_questions=len(seen_questions), current_direction=len(direction_questions),
        auxiliary_direction=len(auxiliary_questions))), flush=True)


if __name__ == '__main__':
    main()
