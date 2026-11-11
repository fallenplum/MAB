"""Independent algebra, saved-array checks, and Chinese E6-B result tables."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json
import hashlib
import numpy as np
import run_contrast_synthetic as run


def main():
    public = run.OUTPUT / 'public/b'
    private = run.OUTPUT / 'private/b'
    path = public / 'E6_B_RESULTS.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    checks = 0
    max_mean_error = 0.
    max_cov_error = 0.
    max_objective_error = 0.
    max_scaled_dual_gap = 0.
    dominance_cases = 0
    rows = []
    for case in result['cases']:
        family, ann, exe, pilot = (case[k] for k in
                                  ['family', 'annotation_fraction', 'execution_fraction', 'pilot_per_stratum'])
        name = f'{family}_ann{int(ann*100)}_exec{int(exe*100)}_pilot{pilot}'
        with np.load(private / 'cases' / (name + '.npz')) as arrays:
            estimates = arrays['estimates']
            resources = arrays['realized_resources']
            truth = arrays['true_gains']
            masks = arrays['gate_masks']
            selection = run.select(estimates)
            np.testing.assert_array_equal(selection, arrays['selected'])
            np.testing.assert_allclose(arrays['regret'], truth.max() - truth[selection], atol=1e-15, rtol=0)
            np.testing.assert_array_equal(estimates[:, 1], estimates[:, 2])
            np.testing.assert_array_equal(estimates[:, 4], estimates[:, 5])
            if family == 'nested':
                np.testing.assert_array_equal(selection[:, 1], selection[:, 3])
            checks += selection.size * 2 + estimates[:, 1].size * 2
            for m, method in enumerate(run.METHODS):
                saved = case['methods'][method]
                np.testing.assert_allclose(saved['marginal_mse'], np.mean((estimates[:, m] - truth) ** 2, axis=0), atol=1e-15)
                np.testing.assert_allclose(saved['policy_selection']['mean_value_regret'], np.mean(truth.max() - truth[selection[:, m]]), atol=1e-15)
                checks += truth.size + 1
                p = np.asarray(case['probabilities'][method])
                # Exact H,D,U finite-cell enumeration. This reconstructs both
                # means and cross-policy covariance without using iid-MC fits.
                mean = np.zeros(len(masks))
                second = np.zeros((len(masks), len(masks)))
                for h in range(6):
                    support, law = run.cluster_distribution(run.old.RHO[h], run.old.MU[h], 3, 'conditional_independent')
                    for d, mass in zip(support, law):
                        value = d * masks[:, h] / p[h]
                        probability = run.old.WEIGHTS[h] * mass * p[h]
                        mean += probability * value
                        second += probability * np.outer(value, value)
                theoretical_covariance = (second - np.outer(mean, mean)) / run.old.T
                np.testing.assert_allclose(mean, truth, atol=1e-15, rtol=0)
                np.testing.assert_allclose(theoretical_covariance, saved['theoretical_covariance'], atol=1e-15, rtol=0)
                max_mean_error = max(max_mean_error, float(np.max(np.abs(mean - truth))))
                max_cov_error = max(max_cov_error, float(np.max(np.abs(theoretical_covariance - saved['theoretical_covariance']))))
                checks += len(mean) + len(mean) ** 2
            for method, diagnostic in case['solver'].items():
                p = np.asarray(case['probabilities'][method])[:6]
                oracle = method.startswith('Oracle')
                a = run.old.true_moments() if oracle else np.asarray(case['estimated_moments'])
                gain = truth if oracle else np.asarray(case['estimated_policy_gains'])
                pair = np.asarray(case['pairs'])
                if method.endswith('GapContrast'):
                    C = (masks[pair[:, 0], :6] - masks[pair[:, 1], :6]) ** 2
                    C *= (run.old.WEIGHTS[:6] * a[:6])[None]
                    gap = np.maximum(np.abs(gain[pair[:, 0]] - gain[pair[:, 1]]), run.EPSILON)
                    C /= gap[:, None] ** 2
                else:
                    C = (run.old.WEIGHTS[:6] * a[:6])[None]
                objective = float((C @ (1 / p)).max())
                max_objective_error = max(max_objective_error, abs(objective - diagnostic['objective']))
                np.testing.assert_allclose(objective, diagnostic['objective'], rtol=1e-12, atol=1e-10)
                assert run.old.WEIGHTS[:6] @ p <= case['phase_annotation'] + 1e-7
                assert (run.old.WEIGHTS * run.old.COSTS)[:6] @ p <= case['phase_execution'] + 1e-7
                max_scaled_dual_gap = max(max_scaled_dual_gap, abs(diagnostic['scaled_duality_gap']))
                checks += C.size + 3
            assert case['exact_marginal_contrast_identity_max_error'] < 1e-14
            dominance_cases += 1
        uniform = case['methods']['Uniform']['policy_selection']['mean_value_regret']
        marginal = case['methods']['Estimated-Marginal']['policy_selection']['mean_value_regret']
        gap = case['methods']['Estimated-GapContrast']['policy_selection']['mean_value_regret']
        no_pilot = case['methods']['NoPilot-Uniform']['policy_selection']['mean_value_regret']
        diff = case['paired_comparisons']['Estimated-GapContrast_minus_Uniform']['regret']
        vsold = case['paired_comparisons']['Estimated-GapContrast_minus_Estimated-Marginal']['regret']
        rows.append(dict(name=name, family=family, pilot=pilot, ann=ann, exe=exe,
                         uniform=uniform, marginal=marginal, gap=gap, no_pilot=no_pilot,
                         delta=diff['mean'], ci=diff['paired_mc_ci95'],
                         delta_old=vsold['mean'], ci_old=vsold['paired_mc_ci95']))
    assert result['status'] == 'COMPLETE' and len(result['cases']) == 12
    assert len(result['old_world_reproduction']) == 6
    assert all(w['original_estimate_max_abs_error'] == 0 for w in result['old_world_reproduction'])
    assert all(w['original_resource_max_abs_error'] == 0 for w in result['old_world_reproduction'])
    audit = dict(status='PASS', scalar_checks=checks, cases=len(rows),
                 original_worlds_reproduced_exactly=6, objective_identity_cases=dominance_cases,
                 max_exact_mean_error=max_mean_error, max_exact_covariance_formula_error=max_cov_error,
                 max_solver_objective_reconstruction_error=max_objective_error,
                 maximum_scaled_duality_gap=max_scaled_dual_gap,
                 actual_LLM_calls=0, results_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    run.save(public / 'E6_B_AUDIT.json', audit)
    smaller = sum(r['delta'] < -1e-12 for r in rows)
    ci_better = sum(r['ci'][1] < 0 for r in rows)
    ci_worse = sum(r['ci'][0] > 0 for r in rows)
    worse_nopilot = sum(r['gap'] > r['no_pilot'] + 1e-12 for r in rows)
    report = [
        '# E6-B 策略差值导向分配的合成验证', '',
        '## 研究问题', '',
        '在相同冻结候选策略、反馈共享和双资源预算下，按策略差值分配反馈能否减少策略选择损失。', '',
        '## 实验设置', '',
        '完整复用E5的分层权重、条件奖励分布、三次生成重复、400道输入及原始六个预算世界。六个旧世界的7种方法估计量和费用均逐样本精确复现，再条件细分其共同采集随机数区间供新概率使用。每格2000次配对Monte Carlo，保留30道/层及预先固定5道/层两种先导规模。', '',
        '候选策略保留默认和全切换。旧非嵌套候选族增加全切换，奖励世界保持原样，原/扩展候选映射均保存。默认在选择并列时优先，分数并列容差为1e-12。结构性零奖励层保留在400道输入分母中，其配对差恒为0，无需购买。', '',
        '先导分别付费180/30道gold、945/157.5执行单位、2700/450次成员调用、1080/180次聚合。条件按层招募假设对应预计1277.78/212.96次输入筛选。先导只用于设计，主估计器沿用E5的独立400题IPW。无先导Uniform把这些资源全部归还后续采集。', '',
        '## 对照及目标', '',
        'Uniform、估计量Marginal、估计量Contrast、估计量GapContrast，以及三种明确标注的Oracle诊断和无先导Uniform。GapContrast的收益间距下限固定为0.01。主分析已知分层权重和期望成本，只从合法先导估计奖励二阶矩及收益差。未知权重/成本另列敏感性分析。', '',
        '默认0与全切换1同时存在时，对任何p都有 F_marginal(p)=F_contrast(p)=sum_h w_h a_h/p_h。两种未归一化目标的求解概率和逐次估计均完全相同。该退化由候选族结构决定。GapContrast仍可能因收益差归一化而改变分配。', '',
        '## 数值结果', '',
        '下表均为最终选中策略的真实价值损失，单位为百分点。Δ为GapContrast减Uniform，负值表示选择损失降低。区间为配对Monte Carlo 95%区间，尚未作为跨格统一多重比较的确认性结论。', '',
        '| 世界 | 先导/层 | Ann/Exec比例 | Uniform | Marginal=Contrast | GapContrast | 无先导Uniform | Δ及95%CI |',
        '|---|---:|---|---:|---:|---:|---:|---|'
    ]
    for row in rows:
        lo, hi = row['ci']
        report.append(f"| {row['family']} | {row['pilot']} | {row['ann']:.0%}/{row['exe']:.0%} | {100*row['uniform']:.5f} | {100*row['marginal']:.5f} | {100*row['gap']:.5f} | {100*row['no_pilot']:.5f} | {100*row['delta']:+.5f} [{100*lo:+.5f}, {100*hi:+.5f}] |")
    report += ['', f'GapContrast对Uniform的点估计在{smaller}/12格较低，未校正区间在{ci_better}/12格完全低于0、{ci_worse}/12格完全高于0。相对无先导Uniform，GapContrast在{worse_nopilot}/12格选择损失更高。嵌套族中两个先导规模的所有格，GapContrast与Marginal/Contrast的选择结果相同。', '',
               '| 世界 | 先导/层 | Ann/Exec | GapContrast减Marginal的损失及95%CI，百分点 |',
               '|---|---:|---|---|']
    for row in rows:
        lo, hi = row['ci_old']
        report.append(f"| {row['family']} | {row['pilot']} | {row['ann']:.0%}/{row['exe']:.0%} | {100*row['delta_old']:+.5f} [{100*lo:+.5f}, {100*hi:+.5f}] |")
    report += ['',
               '| 世界 | 先导/层 | Ann/Exec | Uniform选错/损失>1pp | Marginal选错/损失>1pp | GapContrast选错/损失>1pp | 无先导Uniform选错/损失>1pp |',
               '|---|---:|---|---|---|---|---|']
    for case in result['cases']:
        cells = []
        for method in ['Uniform', 'Estimated-Marginal', 'Estimated-GapContrast', 'NoPilot-Uniform']:
            selection = case['methods'][method]['policy_selection']
            cells.append(f"{selection['nonoptimal_probability']:.2%}/{selection['probability_regret_above_epsilon']:.2%}")
        report.append(f"| {case['family']} | {case['pilot_per_stratum']} | {case['annotation_fraction']:.0%}/{case['execution_fraction']:.0%} | " + ' | '.join(cells) + ' |')
    report += ['',
               '每种方法的选错概率、损失超过0.01的概率、选择分布、全部策略差值MSE、完整跨策略协方差、实际费用及预算溢出概率见 E6_B_RESULTS.json。每次配对的选择及费用保存在private/b/cases的NPZ中。', '',
               '## 统计结论', '',
               '主未归一化Contrast相对旧Marginal的全部配对差均为精确0，无法提供独立机制增益。GapContrast在部分格改善相对Uniform的策略选择，结论随预算、先导规模和候选族改变。付费先导全部计入后，当前设计未超过无先导Uniform。', '',
               '结果同时包含已实现的统计优势与失效范围。二阶矩、MSE和真实选择损失分别报告，未将代理目标降低解释为普遍决策收益。', '',
               '## 优势与风险', '',
               '优势为原始世界逐样本复现、所有方法共同输入/奖励/采集随机数、原始反馈在全部冻结策略之间共享，以及精确保留协方差。', '',
               '风险包括单个固定先导的条件性、候选族含全切换导致目标恒等、条件分层招募的可实现性，以及期望预算允许单次实现超支。未知权重/成本的估计不继承真实资源可行性保证。', '',
               '## 失败原因与下一步', '',
               '未归一化目标受all-switch支配，单纯改写为两两差值无法改变最坏方向。间距归一化又依赖少量先导对真实收益差的估计，小先导在非嵌套50%/50%格产生显著更高选择损失。先导资源回收后的Uniform表现说明先导开销可能抵消分配优势。', '',
               '真实缓存回放按已冻结策略与有限池权限继续执行。合成收益保持为条件诊断，不用于重新调节已冻结E6-A或E6-C。', '',
               '## 数值校验', '',
               f"独立复算通过{checks:,}项标量校验。六个旧E5世界的估计和费用最大差均为0。精确联合分布复算均值最大误差{max_mean_error:.3g}，共享反馈协方差公式最大误差{max_cov_error:.3g}，最大缩放原始对偶间隙{max_scaled_dual_gap:.3g}。", '',
               '本报告的置信度为高，范围限于保存的合成世界、冻结先导和指定算法。真实任务泛化由E6-C另行检验。', '']
    report += ['## 未知权重与费用的敏感性', '',
               '每个世界、先导规模使用50份独立iid输入先导，三组预算及两个不同分配目标，共1200次求解。这里的5/30为每层规模的标签，实际敏感性先导随机抽取30/180道输入，不保证每层配额。Contrast与Marginal恒等，因此只需重复一种。', '',
               '| 世界 | 先导输入数 | 目标 | 真实期望Gold超支率 | 真实期望执行超支率 |',
               '|---|---:|---|---:|---:|']
    for sensitivity in result['estimated_feasible_set_sensitivity']:
        for count in run.PILOTS:
            for method in ['Marginal', 'GapContrast']:
                subset = [r for r in sensitivity['rows'] if r['pilot_per_stratum_label'] == count and r['method'] == method]
                ann_rate = np.mean([r['annotation_violation'] > 1e-7 for r in subset])
                exec_rate = np.mean([r['execution_violation'] > 1e-7 for r in subset])
                report.append(f"| {sensitivity['family']} | {6*count} | {method} | {ann_rate:.2%} | {exec_rate:.2%} |")
    report += ['', '这些比率为真实期望成本相对预算的违反频率，与主实验中Bernoulli实现波动导致的单次超支不同。没有据此宣称未知费用情形满足硬预算。', '']
    repair_path = public / 'SOLVER_ACTIVE_SET_REPAIR.json'
    if repair_path.exists():
        repair = json.loads(repair_path.read_text(encoding='utf-8'))
        report += ['## 共享求解器数值修复', '',
                   'E6-C第2折先导编号38、Gold48的凸程序在后置KKT核验时失败。预算数值被用于放大活跃约束识别容差，导致严格松弛的策略行进入对偶乘子重建。修复只收紧活跃行识别，原始优化与验收阈值保持原值。', '',
                   f"失败输入修复前后的概率数组和原始目标逐位相同。缩放对偶间隙从{repair['old_scaled_duality_gap']:.6g}降至{repair['repaired_scaled_duality_gap']:.6g}。48个E6-B独立设计的概率逐位复核相同，Monte Carlo结果复用原保存数组，源码新旧哈希均保留。", '']
    (public / 'E6_B_SYNTHETIC_ALLOCATION.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
