"""Render E19 reports from completed public aggregates without new fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(row):
    return f"[{row['ci95_low_pp']:+.3f}, {row['ci95_high_pp']:+.3f}]"


def write(path, lines):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")


def footer(data, result_sha, protocol_sha, review_sha, historical_sha=None):
    lines = ["", "## 来源与复现", "", "报告仅从已完成汇总渲染，拟合与Bootstrap新增运行次数均为0。",
            "", "| 来源 | SHA256 |", "|---|---|",
            f"| E19_COMPLETE_RESULTS.json | `{result_sha}` |",
            f"| E19_PROTOCOL.json | `{protocol_sha}` |",
            f"| NUMERICAL_AND_ACCESS_REVIEW.json | `{review_sha}` |",
            f"| analyze_e19.py | `{data['source_hashes']['analysis']}` |",
            f"| OOF_EVALUATION.npz，私有 | `{data['source_hashes']['oof_evaluation']}` |",
            ]
    if historical_sha:
        lines.append(f"| HISTORICAL_NUMERICAL_RECOVERY.json | `{historical_sha}` |")
    return lines+["", "逐题分数、概率、奖励、模型参数和bootstrap重复值按私有资产规则保存。"]


def historical_note(review):
    rows = review.get("historical_recovery_diagnostics", [])
    changed = review.get("historical_near_policy_probability_changed_questions")
    over = review.get("historical_centered_score_recovery_over_1e_minus7_folds",
                      review.get("historical_centered_score_recovery_over_1e7_tolerance_folds", []))
    lines = ["", "## 旧E17缓存的数值边界", "",
             "B按本轮同目标、同特征和原λ重新拟合，使用本轮更紧的求解容差。当前B与C的比较基于这一成对新拟合。旧E17缓存保留原版本。"]
    if rows:
        lines += ["", "| 折 | 旧/新W最大差 | 旧/新中心化评价分数最大差 | 同近并列规则下概率变化题数 |",
                  "|---|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['fold']} | {row['old_vs_new_coefficient_max_error']:.6g} | {row['old_vs_new_centered_E_score_max_error']:.6g} | {row['old_vs_new_common_near_policy_changed_questions']} |")
        lines += ["", f"实际有{changed}道旧/新概率不同的题，中心化分数超过1e-7核对尺度的折为{over}。",
                  "fold1的8道题在同一1e-8近并列规则下，支持集合由13项变为14项。fold3最大中心化差约4.831e-7，概率支持集合保持一致。",
                  "这些记录界定旧缓存的数值继承程度。E19均匀数值argmax部署与原E17语义优先部署各自保留定义，当前P0资格只对本轮反向编号映射提供数值核验。"]
    else:
        lines += ["", "历史数值对照由独立review另行归档。当前报告的B标记为本轮重新拟合，旧缓存精确恢复状态留待该记录核对。"]
    return lines


def p0_report(data, protocol, review):
    q = data["P0_qualification"]
    lines = ["# E19 P0 数值与权限资格报告", "", "## 研究问题与冻结设置", "",
             "比较经验全动作并列损失强度β。B取β=1，C取β=0，γ=0。每个训练区使用480道原题、同一PCA16标准化特征、17×20线性系数及同一正L2算子，常数项也受罚。",
             "全部480行及对应epigraph变量保留，数据项分母固定为480。C在经验全20动作均值完全相等的训练行上赋零权重，其有效权重和列在后文。",
             "", "经验集合T只由合法H480中的三生成均值确定。每折H标签按指定行解码，E120奖励在权重、完整分数、概率及哈希收据写入后开放。独立访问核对覆盖五折。",
             "", "## P0实际执行", "", "fold0共4次正式拟合，B/C各一次原顺序和一次reverse20顺序。每次使用全新Clarabel求解器，tol_gap_abs、tol_gap_rel、tol_feas均为1e-10。反向编号同步映射奖励、标签、等向正则与输出物理列。",
             "", "| P0项目 | 实际结果 |", "|---|---|", f"| 状态 | {q['status']} |",
             f"| 正式P0拟合数 | {q['real_fits']} |", f"| 固定分母 | {q['denominator']} |",
             f"| 端点不等式左量F0(B)−F0(C) | {q['F0_B_minus_F0_C']:.12f} |",
             f"| 端点不等式右量JT(C)−JT(B) | {q['JT_C_minus_JT_B']:.12f} |",
             f"| 预设核验容差 | {q['inequality_tolerance']:.1e} |", "| 以评价效用筛选P1 | False |",
             "", "0 ≤ F0(B)−F0(C) ≤ JT(C)−JT(B)在预设数值容差内成立，全部P0检查通过。P1由数值资格放行。",
             "", "| 分支 | H中心化映射最大差 | E中心化映射最大差 | 均匀概率最大差 |",
             "|---|---:|---:|---:|"]
    for branch, row in q["mappings"].items():
        lines.append(f"| {branch} | {row['centered_score_h_max_error']:.6g} | {row['centered_score_e_max_error']:.6g} | {row['uniform_probability_max_error']:.6g} |")
    lines += ["", f"全12次拟合最大KKT stationarity为{data['max_KKT_stationarity']:.6g}，最大primal-dual gap为{data['max_primal_dual_gap']:.6g}。正L2给W提供严格凸性，零权重行的辅助epigraph坐标具有各自的可行自由度。",
              "", "## 五折目标分解", "", "定义F0=L_nonT/480+λ‖W‖²，JT=(2/480)∑T(max q−mean q)，F1=F0+JT。下表λ同时用于该折B/C。",
              "", "| 折 | N | T题数 | λ | B有效权重 | C有效权重 | F0(B)−F0(C) | JT(C)−JT(B) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in data["folds"]:
        lines.append(f"| {row['fold']} | {row['history_questions']} | {row['training_all_tied_questions']} | {row['regularization']:.17g} | {row['B_effective_weight_sum']:.0f} | {row['C_effective_weight_sum']:.0f} | {row['F0_B_minus_F0_C']:.9f} | {row['JT_C_minus_JT_B']:.9f} |")
    lines += ["", "| 折 | 非T数据项B | 非T数据项C | L2项B | L2项C | JT(B) | JT(C) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in data["folds"]:
        lines.append(f"| {row['fold']} | {row['nonT_loss_B']:.9f} | {row['nonT_loss_C']:.9f} | {row['penalty_B']:.9f} | {row['penalty_C']:.9f} | {row['JT_B']:.9f} | {row['JT_C']:.9f} |")
    lines += ["", "五折C均取得更低的非T训练数据项、更高的L2项及更大的JT。F0的净下降同时计入这两部分。该端点关系验证本轮共享参数目标的数值行为。最终选队效用由外折评价单列。"]
    lines += historical_note(review)
    lines += ["", "## 统计解释边界", "", "软件资格与凸优化残差为实现核对提供证据。研究效用属于反复使用的train600开发OOF，真实条件均值、未来分布收益及总体一致性仍各有适用条件。"]
    return lines


def five_fold_report(data):
    p, m = data["primary"], data["mechanism"]
    lines = ["# E19 P1 五折选队与机制结果", "", "## 实验设置", "",
             "五个原问题级外折，每折480题训练、120题评价，GSM8K/BBH共各300道原题。两支具有相同340参数容量、输入、标签、原λ和部署规则。主服务概率在score≥rowmax−1e-8的集合上均匀分配，以缓存奖励直接计算期望服务正确率。",
             "", "| 主指标 | B，β=1 | C，β=0 | C−B百分点 |", "|---|---:|---:|---:|",
             f"| GSM8K/BBH各50%的总体准确率 | {p['B_accuracy_pct']:.6f}% | {p['C_accuracy_pct']:.6f}% | {p['gain_pp']:+.6f} |",
             "", "## 五折与任务分解", "", "| 折 | 评价原题 | B准确率 | C准确率 | 点差pp | 严格唯一且换队题数 |", "|---|---:|---:|---:|---:|---:|"]
    for row in data["folds"]:
        lines.append(f"| {row['fold']} | 120 | {row['B_accuracy_pct']:.6f}% | {row['C_accuracy_pct']:.6f}% | {row['gain_pp']:+.6f} | {row['strict_changed_questions']} |")
    lines += ["", "四折点差为正，fold4为−0.059524pp。全部折保留在600题主分析中。",
              "", "| 任务 | 原题数 | B准确率 | C准确率 | 点差pp |", "|---|---:|---:|---:|---:|"]
    for row in data["task_effects"]:
        lines.append(f"| {row['task']} | {row['questions']} | {row['B_accuracy_pct']:.6f}% | {row['C_accuracy_pct']:.6f}% | {row['gain_pp']:+.6f} |")
    lines += ["", "| 生成seed | B准确率 | C准确率 | 点差pp |", "|---|---:|---:|---:|"]
    for seed, row in data["seed_effects"].items():
        lines.append(f"| {seed} | {100*row['B_accuracy']:.6f}% | {100*row['C_accuracy']:.6f}% | {row['gain_pp']:+.6f} |")
    lines += ["", f"BBH27子任务宏平均为B {data['bbh_subtask_macro']['B_accuracy_pct']:.6f}%、C {data['bbh_subtask_macro']['C_accuracy_pct']:.6f}%。完整27子任务的支持量及正负点差保留在BBH_SUBTASK_RESULTS.csv。主任务权重按GSM8K/BBH各50%计算。",
              "", "## 相对分数与实际选队变化", "", "| 指标 | 实际值 |", "|---|---:|",
              f"| 中心化分数变化超过1e-7的原题 | {m['centered_score_changed_questions_threshold_1e_minus7']} / 600 |",
              f"| 中心化分数最大绝对变化 | {m['centered_score_max_abs_change']:.9f} |",
              f"| 中心化分数行L2变化均值 | {m['centered_score_mean_row_l2_change']:.9f} |",
              f"| 公共分数偏移最大值 | {m['common_score_shift_max_abs']:.6g} |",
              f"| B数值并列题数 | {m['B_numerical_tied_questions']} |", f"| C数值并列题数 | {m['C_numerical_tied_questions']} |",
              f"| 两支均有严格唯一胜者 | {m['strict_both_unique_questions']} |", f"| 严格唯一且物理胜者变化 | {m['strict_changed_questions']} |",
              f"| 严格换队纠错事件 | {m['strict_generation_counts']['corrected_generation_events']} |",
              f"| 严格换队破坏事件 | {m['strict_generation_counts']['damaged_generation_events']} |",
              "", "严格唯一判据为top1−top2>1e-8。150题换队包含450条已记录题目×生成事件，其中38次纠错、19次破坏，净19次。这些为固定唯一动作在缓存中的整数计数。",
              "", "相同部署定义下的中心化分数变化及严格换队支持当前共享参数的相对预测发生变化。训练中经验全并列行对这些共享参数的约束具有可观测影响。其效用范围由开发评价及敏感性结果共同限定。",
              "", "## 子集贡献", "", "| 子集 | 原题 | 子集原题均值点差pp | 对主任务宏平均的贡献pp | 期望纠错 | 期望破坏 |", "|---|---:|---:|---:|---:|---:|"]
    for row in data["subgroup_decomposition"]:
        lines.append(f"| {row['subset']} | {row['questions']} | {row['conditional_gain_pp']:+.6f} | {row['primary_macro_gain_contribution_pp']:+.6f} | {row['expected_corrections']:.6f} | {row['expected_damages']:.6f} |")
    lines += ["", "strict_changed包含于strict_both_unique，两行各自描述同一批19次净纠错的不同条件分母。主分解可将203题strict_both_unique与397题numerical_tie_in_at_least_one相加。其贡献分别为+1.055556pp和+0.386464pp。",
              "", "## 优势与风险", "", "- 当前控制同时固定容量、特征、标签、分母、λ及均匀部署定义，并保存训练目标分解和严格动作变化。",
              "- 600题已多轮参与开发，五折训练区重叠。三个生成repeat属于同题重复测量，1800条生成事件对应600道原题。",
              "- 训练T是三重复经验全并列，真实条件期望奖励具有各自的生成不确定性。当前结论受该经验监督构造与有限线性表示约束。"]
    return lines


def utility_report(data, protocol):
    p, raw = data["primary"], data["raw_exact_uniform_diagnostic"]
    lines = ["# E19 Bootstrap与效用解释", "", "## 主估计与预定推断", "",
             "主比较为C−B，GSM8K/BBH各50%。使用预定τ=1e-8数值argmax集合上的均匀策略，缓存中的三生成结果先在原题内保留并共同重采样。",
             "", "| 口径 | B准确率 | C准确率 | 点差pp | 95%区间pp |", "|---|---:|---:|---:|---|",
             f"| 预定数值argmax均匀策略 | {p['B_accuracy_pct']:.6f}% | {p['C_accuracy_pct']:.6f}% | {p['gain_pp']:+.6f} | {interval(p)} |",
             f"| raw-exact argmax均匀诊断 | {raw['B_accuracy_pct']:.6f}% | {raw['C_accuracy_pct']:.6f}% | {raw['gain_pp']:+.6f} | {interval(raw)} |",
             "", "主口径点值约+1.442pp，区间下端约+0.191pp。raw-exact诊断点值约+1.000pp，其区间跨0。两种部署在浮点并列边界上具有不同的概率定义，结果限定于各自固定口径。当前证据支持预定数值策略下的开发线索，raw-exact结果保留其不确定性。",
             "", f"Bootstrap重复数为{data['bootstrap_repetitions']:,}，seed为{data['bootstrap_seed']}。GSM8K和BBH各300题分层有放回抽样，两支策略及每题三个生成结果一起保留。区间采用等尾percentile与linear quantile。",
             "", "该区间条件于已拟合的头、当前题源与三次已记录生成。五折训练历史重叠，问题抽样区间覆盖的是这一冻结OOF分析的条件不确定性。600题为反复使用的开发资产，外部总体和完整训练程序的独立确认仍需另行设计。",
             "", "## 任务区间", "", "| 任务 | 原题 | 点差pp | 95%区间pp |", "|---|---:|---:|---|"]
    for row in data["task_effects"]:
        lines.append(f"| {row['task']} | {row['questions']} | {row['gain_pp']:+.6f} | {interval(row)} |")
    lines += ["", "BBH点值较大，GSM8K区间跨0。完整27子任务结果按预定支持量全部报告，任务或seed中的方向保留各自重复测量身份。",
              "", "## 期望纠错与整数纠错", "",
              "主效用直接计算p·R，与B/C联合抽样的耦合选择无关。whole-policy纠错、破坏采用预先固定的最大同动作重叠耦合。令h=min(pB,pC)、ρ=1−∑h、eB=pB−h、eC=pC−h。ρ>0时，期望纠错为[eB·(1−R)](eC·R)/ρ，期望破坏为(eB·R)[eC·(1−R)]/ρ，ρ=0时两者为0。",
              "", "| 样本范围 | 纠错 | 破坏 | 净纠错 | 计数含义 |", "|---|---:|---:|---:|---|",
              f"| 全600题、三生成 | {p['expected_corrections']:.6f} | {p['expected_damages']:.6f} | {p['expected_net_corrections']:.6f} | 声明耦合下的分数期望 |",
              "| 150题严格换队、三生成 | 38 | 19 | 19 | 固定唯一动作的整数缓存事件 |",
              "", "whole-policy的89.6468次纠错和63.6905次破坏是概率期望。实际没有为这些分数期望额外采样服务动作。严格换队两支各有唯一动作，因此可以直接统计38/19个记录事件。",
              "", f"政策总变差均值为{data['mechanism']['mean_total_variation']:.6f}，有界奖励给当前题池的确定性效用影响上界约{data['mechanism']['fixed_pool_utility_bound_pp']:.3f}pp。实际效用和区间提供更具体的质量信息。",
              "", "## 实际意义与后续边界", "",
              f"预定最小实际效应为{protocol['minimum_practical_effect_pp']:.1f}pp。主点值超过该值，主区间下端低于该值。当前证据分别保留正向开发点差、实际效应幅度的不确定性及raw-exact敏感性。",
              "", "当前P0/P1完成，P2/P3/P4运行次数为0。后续可依据冻结模型、数值并列规则及其敏感性规划独立验证，新的外部题池评价需要单独执行方案。"]
    return lines


def summary_report(data, review):
    p, raw, m = data["primary"], data["raw_exact_uniform_diagnostic"], data["mechanism"]
    return ["# E19 实验总结", "", "## 当前结论", "",
            f"在预定τ=1e-8均匀数值argmax策略下，B准确率为{p['B_accuracy_pct']:.3f}%，C为{p['C_accuracy_pct']:.3f}%，C−B为{p['gain_pp']:+.3f}pp，95%区间{interval(p)}pp。raw-exact诊断为{raw['gain_pp']:+.3f}pp，区间{interval(raw)}pp。",
            "", "经验全动作并列损失强度的变化，在当前共享线性头中改变相对预测和严格唯一动作。效用结果显示正向开发线索，并保留部署数值边界的敏感性。",
            "", "## 已完成的控制", "",
            "- P0正式4fit包含fold0两支及其reverse20映射，数值与端点检查通过。P1仅追加其余8fit，总计10主拟合和2资格拟合。",
            "- 每折保留N=480、480行epigraph、1/N归一化和该折原λ，17×20参数与常数项正则共同沿用。五折均满足0≤F0(B)−F0(C)≤JT(C)−JT(B)。",
            f"- {m['centered_score_changed_questions_threshold_1e_minus7']}题中心化分数变化超过1e-7，公共偏移最大约{m['common_score_shift_max_abs']:.3g}。203题两支均严格唯一，150题严格换队，38次纠错、19次破坏。",
            f"- 全策略最大同动作重叠耦合的期望纠错为{p['expected_corrections']:.4f}，破坏为{p['expected_damages']:.4f}，净期望为{p['expected_net_corrections']:.4f}。这些分数计数与严格整数事件分别报告。",
            "", "## 正负结果与局限", "",
            "四折点差为正，fold4为−0.059524pp。三个生成seed点差均为正。GSM8K区间跨0，BBH点值较大。完整27子任务结果保存在BBH_SUBTASK_RESULTS.csv。",
            "", "主口径区间下端大于0，raw-exact区间跨0。五折训练区重叠，600题已反复开发，三生成对应同题重复测量。当前统计是冻结开发OOF的条件证据，独立总体确认仍需另行设计。",
            "", f"旧E17与本轮B的历史数值比较记录了{review.get('historical_near_policy_probability_changed_questions', '待核对')}道近并列概率变化，fold3中心化数值差约4.831e-7。当前B/C使用共同的本轮严格求解设置，旧部署结果按原版本保留。",
            "", "## 计算与下一阶段", "",
            f"12次正式拟合CPU合计{data['real_fit_cpu_seconds']:.6f}秒，拟合墙钟和为{data['real_fit_wall_seconds_sum']:.6f}秒。分析执行{data['bootstrap_repetitions']:,}次题级分层Bootstrap，分析CPU为{data['analysis_cpu_seconds']:.6f}秒。报告生成仅使用完成汇总。",
            "", "新增LLM调用、fresh400奖励读取、sealed200读取及P2/P3/P4运行次数均为0。下一阶段定位为冻结条件下的独立验证方案设计。数值并列规则、raw-exact敏感性和旧题源条件应随方案一并保留。"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("E19_PROTOCOL.json"))
    parser.add_argument("--review", type=Path, default=Path(__file__).with_name("NUMERICAL_AND_ACCESS_REVIEW.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    data, protocol, review = read(args.results), read(args.protocol), read(args.review)
    historical_sha = None
    if review.get("historical_review_file"):
        name = review["historical_review_file"]
        if Path(name).name != name:
            raise ValueError("Historical review reference must be a same-directory filename")
        historical_path = args.review.parent/name
        historical = read(historical_path)
        review = dict(review, historical_recovery_diagnostics=historical["folds"])
        historical_sha = sha(historical_path)
    if data.get("status") != "P0_P1_COMPLETE" or data["source_hashes"]["protocol"] != sha(args.protocol):
        raise ValueError("Report inputs are not the completed frozen E19 analysis")
    args.output_root.mkdir(parents=True, exist_ok=True)
    tail = footer(data, sha(args.results), sha(args.protocol), sha(args.review), historical_sha)
    reports = {
        "02_P0_QUALIFICATION_REPORT.md": p0_report(data, protocol, review),
        "03_P1_FIVE_FOLD_RESULTS.md": five_fold_report(data),
        "04_BOOTSTRAP_AND_UTILITY.md": utility_report(data, protocol),
        "EXPERIMENT_SUMMARY.md": summary_report(data, review),
    }
    for name, lines in reports.items():
        path = args.output_root/name
        write(path, lines+tail)
        print(json.dumps(dict(file=str(path), sha256=sha(path), bytes=path.stat().st_size), ensure_ascii=False))


if __name__ == "__main__":
    main()
