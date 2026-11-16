"""Render E18.1 aggregate reports from the completed statistical analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def number(value, digits=3):
    if value is None:
        return "未定义"
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def ci(row):
    return "[" + number(row["ci95_low_pp"]) + ", " + number(row["ci95_high_pp"]) + "]"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write(path, text):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text.rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    if result["status"] != "COMPLETE":
        raise ValueError("Statistical results are incomplete")
    output = args.output_root
    main_table = table(
        ("TSPO比较对象", "点差pp", "题级Bootstrap95%CI", "原配对t p", "原次要Holm p", "纠错", "破坏"),
        [(row["comparison"].split(" minus ")[1], number(row["gain_pp"]), ci(row),
          number(row["original_t_p"], 6), number(row["original_secondary_holm_p"], 6),
          row["corrected_events"], row["damaged_events"]) for row in result["overall"]])
    missing = result["deterministic_missing_gain_bounds"]["TSPO_uniform minus SPO_coordinate"]
    statistical = f"""# E18.1统计分析报告

## 研究问题与数据

在冻结的六模型、K=3、V2聚合器与四套路由策略下，补充TSPO相对SPO、Ridge和GlobalFixed的题级配对区间。原计划3,000题，可评分2,998题，包括GSM8K1,500题与BBH1,498题。每题三次既有生成保留为一个问题簇。

本轮是在查看E18结果之后执行的统计补充。用户明确授权本轮置信区间与Bootstrap，E18原始报告和统计记录完整保留。

## 主要统计结果

{main_table}

单位为准确率百分点。主比较TSPO−SPO为+0.011749pp，净增加1条已记录生成事件。三个区间均跨越零，当前结果没有提供TSPO总体质量优于相应对照的支持。主区间上端约0.690pp，低于此前讨论的1pp实际意义标准。该观测也须结合本轮事后统计补充、问题代表性与有限生成重复解释。

## Bootstrap方法

固定20,000次重采样和随机种子202611160181。每次在GSM8K和BBH内部各自有放回抽取相同数量的原题。被抽中的题同时携带四策略和三个生成重复。任务均值固定各占50%，BBH内部按题平均。采用百分位法的2.5%与97.5%分位，线性插值，批大小256。

对每道题先平均三次生成，再按题重采样，与把该题全部策略及生成记录整块抽取后的均值运算等价。实现使用共享问题索引配对重采样，独立复核还重建了各任务前3个抽样。配对重采样的索引规则及百分位区间定义可见SciPy一手文档。[SciPy Bootstrap文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)

这里已全评分的有限题池均值是已知点值。Bootstrap量化问题抽样泛化不确定性的近似，条件包括冻结模型、当前任务题源混合及三个已记录生成。题内未独立重抽seed，也未重新拟合训练程序。覆盖解释依赖任务内问题可代表目标分布的假设。

这些95%区间逐项具有边际口径，原E18五项次要比较的Holm调整p单独保留。当前区间没有同时覆盖保证。BBH子任务各自区间仅作异质性诊断。

## 置信区间与两道缺失题

可评分2,998题上的主Bootstrap95%CI为{ci(result['overall'][0])}pp。完整原计划3,000题分母上的缺失结果确定性范围为[{number(missing['lower_pp'])}, {number(missing['upper_pp'])}]pp。

后者只允许两道未知奖励在合法取值内变化，且相同冻结动作具有相同结果。它没有概率覆盖水平。两道BBH题继续保持不可评分，所有已执行费用保留，缺失奖励未填0。两种范围分别回答抽样泛化不确定性和缺失结果敏感性。

## 优势、风险与结论

优势是同题配对、生成簇保留、固定任务权重和原检验完整继承。风险包括数据已被E18查看、两个缺失题、题源构成与有限生成重复，以及边际区间的多重比较限制。未拒绝零差不能证明绝对等价。任务分析显示总体近零伴随方向抵消，需结合下一份异质性报告解释。
"""
    write(output / "01_STATISTICAL_REPORT.md", statistical)

    task_table = table(("任务", "独立题数", "TSPO−SPO pp", "题级95%CI", "配对t p", "纠错", "破坏"),
        [(row["task"], row["questions"], number(row["gain_pp"]), ci(row),
          number(row["t_p"], 6), row["corrected_events"], row["damaged_events"])
         for row in result["task_effects"]])
    interaction = result["interaction"]
    subtask_table = table(("BBH子任务", "原题数", "点差pp", "题级95%CI", "原始p", "Holm27 p", "净纠错事件"),
        [(row["subtask"], row["questions"], number(row["gain_pp"]), ci(row),
          number(row["t_p"], 5), number(row["diagnostic_holm27_p"], 5), row["net_corrected_events"])
         for row in result["bbh_subtask_effects"]])
    seed_table = table(("评价范围", "生成seed", "点差pp", "纠错", "破坏"),
        [(row["scope"], row["generation_seed"], number(row["gain_pp"]),
          row["corrected_events"], row["damaged_events"])
         for row in result["generation_seed_effects"]])
    heterogeneity = f"""# E18.1任务异质性分析

## 任务级结果

{task_table}

GSM8K上的下降区间位于零以下，BBH上升的区间跨越零。两任务各占50%后，方向抵消形成总体接近零的点差。单个任务是否跨零与两任务效应是否不同，是各自独立的统计问题。

## Task×Policy交互

对每题定义d为三次生成的TSPO−SPO平均正确率差。交互参数为BBH的平均d减去GSM8K的平均d，估计{number(interaction['estimate_pp'])}pp，题级Bootstrap95%CI为{ci(interaction)}pp。双侧Welch检验t={number(interaction['t_statistic'], 6)}，自由度{number(interaction['degrees_of_freedom'], 3)}，p={number(interaction['two_sided_welch_p'], 8)}。

该检验等价于在题级配对差上比较两个独立任务组的均值，并使用各组样本方差与Satterthwaite自由度。两个策略在同题内的依赖已包含在d中。SciPy的Welch实现通过equal_var=False选择不等方差检验。[SciPy Welch文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)

这是看过E18后的探索性任务异质性证据。任务类别属于观测条件，任务差异还可能关联题源、难度与推理形式。现有结果不足以识别某个训练机制的单独因果效应。模型、阈值与任务权重均维持冻结记录。

## 生成重复方向

{seed_table}

三个seed共享同一原题集合，各自结果属于题内重复测量。任务宏平均有两个正方向和一个负方向，未形成一致的总体正增益。

## 27个BBH子任务完整结果

{subtask_table}

每个子任务在其原题内独立执行20,000次配对Bootstrap，保持当前样本量。全部子任务支持数合计1,498，单个子任务只有54至57题。表中为边际95%区间，Holm27 p另列。没有子任务通过该诊断性Holm27检验。

分位点接近零的数值按三位小数显示为0.000，避免将浮点误差符号解释为证据。全部子任务和负结果均保留。后续模型选择未使用这些结果。
"""
    write(output / "02_TASK_HETEROGENEITY.md", heterogeneity)

    distributions = [row for row in result["distributions"] if row["scope"] == "all_evaluable"]
    distribution_table = table(("策略", "准确率%", "熵nats", "HHI", "最高频组合", "最高频占比%", "组合0占比%"),
        [(row["policy"], number(row["accuracy_pct"]), number(row["entropy_nats"], 6),
          number(row["hhi"], 6), row["dominant_combination"], number(100 * row["dominant_share"], 2),
          number(100 * row["combination0_share"], 2)) for row in distributions])
    selection = f"""# E18.1组合选择与实际质量

## 冻结动作分布

{distribution_table}

熵使用自然对数，H=−sum(p log p)。HHI为sum(p²)。频率分母为2,998道可评分原题，每题一个冻结动作。报告准确率仍使用GSM8K与BBH各50%的权重。任务内的完整分布另保留在POLICY_DISTRIBUTIONS.csv。

SPO在2,489题上选择组合0，占83.02%。TSPO在139题选择组合0，占4.64%，最高频组合5占9.27%。E18可观察到SPO的组合0集中持续出现，以及TSPO集中度明显下降。

## 同题动作差异

SPO与TSPO在2,778题上选择不同组合，其中GSM8K1,405题、BBH1,373题。剩余220题共享相同动作，三个生成重复的复用成绩逐项相同。20×20真实动作转移及纠错、破坏记录在SPO_TSPO_ACTION_TRANSITIONS.csv。

尽管动作变化覆盖大量问题，TSPO−SPO总体净增加1条正确生成记录。E17训练规则干预与E18服务质量评价分别提供偏好形成机制和冻结决策表现的证据。

## 分散度与质量的描述性关系

TSPO的熵更高、HHI更低，同时相对SPO的总体质量接近零差。GlobalFixed只选组合5，熵为0、HHI为1，却取得四策略最高的当前点值86.863%。因此，本轮数据没有建立更分散的组合分布带来更高服务质量的关系。

这里只有四套冻结策略，而且它们在同一批题上评价。并列表描述这些系统的共同结果，无法识别熵或HHI的因果效应。更均匀的选择也不足以证明问题内容已被有效用于选队。

## 优势与风险

优势是以真实组合身份与同题最终聚合成绩联合核对，保持推理期并列规则的冻结版本。风险是E17还发现近并列推理规则会改变实际动作，训练与推理的规则影响需要分别审计。当前分布只能回答给定冻结模型与给定规则在本次题池中的表现。
"""
    write(output / "03_SELECTION_AND_QUALITY.md", selection)

    departures = result["tspo_global_fixed_decomposition"]
    top = [row for row in departures if row["group"] in ("all_evaluable", "gsm8k", "bbh")]
    top_table = table(("范围", "动作", "题数", "该范围占比%", "条件点差pp", "纠错", "破坏", "净纠错"),
        [(row["group"], row["relation"], row["questions"], number(100 * row["subset_fraction"], 2),
          number(row["conditional_gain_pp"]), row["corrected_events"], row["damaged_events"],
          row["net_corrected_events"]) for row in top])
    detail_table = table(("BBH子任务", "离开5题数", "占子任务%", "纠错", "破坏", "净纠错", "条件点差pp"),
        [(row["group"].split("/", 1)[1], row["questions"], number(100 * row["subset_fraction"], 2),
          row["corrected_events"], row["damaged_events"], row["net_corrected_events"],
          number(row["conditional_gain_pp"])) for row in departures
         if row["group"].startswith("bbh/") and row["relation"] == "leave_combination5"])
    mechanism = f"""# E18.1 TSPO与GlobalFixed机制分解

## 主对照结果

GlobalFixed始终选择训练区确定的组合5。训练来源与排除题成本见05_ACCOUNTING_AND_GLOBAL_FIXED.md。TSPO相对GlobalFixed的任务宏平均差为−0.455570pp，题级95%CI为{ci(result['overall'][2])}pp，原配对t检验p=0.200390。

TSPO共有344次纠错、385次破坏，净少41条正确生成事件。两种策略在相同题和相同生成重复上配对。

## 何时离开组合5

{top_table}

TSPO离开组合5共2,720题，占90.73%。留在组合5的278题上，两策略完全复用相同执行结果，纠错和破坏均为0。全部344次纠错和385次破坏均发生在离开组合5的题上。

表中条件点差使用当前子集原题的自然平均。离开集合中GSM8K与BBH题数不相等，其条件点差−0.502451pp具有该子集口径。主结果−0.455570pp继续使用全体可评分题及固定任务50/50权重。

GSM8K离开组合5后的64次纠错伴随104次破坏，净−40。BBH为280次纠错和281次破坏，净−1。整体质量代价主要由当前GSM8K样本贡献，BBH内部还存在相互抵消的子任务差异。

## BBH全部子任务分解

{detail_table}

所有任务组均保留，表格用于解释已冻结决策的得失分布。它未参与本轮模型或默认组合选择。

## 条件分组的局限

留在组合5的题与离开组合5的题由冻结路由器选择，两个子集的难度与内容可能不同。其绝对准确率差不能归因于离开动作本身。可解释的服务差是同题TSPO与GlobalFixed的配对结果，现有结果尚未支持离开组合5带来总体正收益。
"""
    write(output / "04_TSPO_GLOBAL_FIXED.md", mechanism)
    summary = f"""# E18.1统计与机制分析总结

## 主要结论

{main_table}

主比较的总体点差接近零，区间包含零。任务交互为{number(interaction['estimate_pp'])}pp，95%CI{ci(interaction)}，p={number(interaction['two_sided_welch_p'], 6)}。GSM8K下降、BBH上升的抵消需要与总体结果同时呈现。

SPO组合0占比83.02%，TSPO降至4.64%，但这种集中度变化尚未表现为总体服务质量改善。相对GlobalFixed，TSPO离开组合5的2,720题中纠错344次、破坏385次。

## 统计与数据边界

20,000次按任务分层的题级配对Bootstrap保留同题三seed，主权重50/50。子任务全部报告，边际区间和Holm调整p分别呈现。E18.1是观察E18后进行的统计补充。E17的100次重编号是训练规则重复，独立评价支持数仍为2,998个可评分问题。

原计划另有两道不可评分BBH题，全部执行支出保留。原E18统计、确定性缺失范围和负面结果完整继承。新增LLM调用、模型训练与调参均为0。

## 文件

- 01_STATISTICAL_REPORT.md给出主区间与推断口径。
- 02_TASK_HETEROGENEITY.md给出交互、seed和27子任务。
- 03_SELECTION_AND_QUALITY.md给出熵、HHI与真实动作变化。
- 04_TSPO_GLOBAL_FIXED.md给出344/385的任务分解。
- 05_ACCOUNTING_AND_GLOBAL_FIXED.md核对执行账目与训练来源。
- 06_JOINT_EVIDENCE.md和07_RESULTS_AND_DISCUSSION.md组织论文证据与草稿。
- E18_1_COMPLETE_RESULTS.json、全部CSV及私有BOOTSTRAP_REPLICATES.npz保存结果。
"""
    write(output / "EXPERIMENT_SUMMARY.md", summary)
    print(json.dumps(dict(status="REPORTS_RENDERED", markdown_reports=5), ensure_ascii=False))


if __name__ == "__main__":
    main()
