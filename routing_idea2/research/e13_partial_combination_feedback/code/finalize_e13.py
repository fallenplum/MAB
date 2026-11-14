"""Report the fixed E13 comparison and its fully charged feedback ledger."""
from __future__ import annotations

import csv
from pathlib import Path
import time

from common import (
    BUDGETS, CODE, METHODS, PRIVATE, PUBLIC, SCENARIOS, Timer,
    load_npz, read_json, sha256, verify_source_lock, write_json,
)
import numpy as np


MAIN_PLOT_METHODS = ("Raw", "RawCentered", "Group", "GroupResidual", "SaveRouterFixedMask",
                     "GlobalFixed", "TaskFixed", "SubtaskFixed")
COLORS = {"Raw": "#2E5983", "RawCentered": "#8DB0CE", "Group": "#735499", "GroupResidual": "#B07AA1",
          "SaveRouterFixedMask": "#CC8A3E", "GlobalFixed": "#989898", "TaskFixed": "#555555", "SubtaskFixed": "#B5B5B5"}


def mechanism_diagnostics():
    frozen = read_json(PRIVATE / "ALL_PREDICTIONS_FROZEN.json")
    grouped = {(scenario, budget): [] for scenario in SCENARIOS for budget in BUDGETS}
    for world in frozen["worlds"]:
        for snapshot in world["snapshots"]:
            grouped[(world["scenario"], snapshot["budget"])].append(snapshot)
    conditions = {}
    for scenario in SCENARIOS:
        conditions[scenario] = {}
        for budget in BUDGETS:
            rows = grouped[(scenario, budget)]
            counts = np.array([r["observed_group_action_cells"] for r in rows])
            conditions[scenario][str(budget)] = dict(
                world_paths=len(rows), observed_group_action_cells_mean=float(counts.mean()),
                observed_group_action_cells_min=int(counts.min()), observed_group_action_cells_max=int(counts.max()),
                total_group_action_cells=560, observed_group_action_fraction=float(counts.mean()/560),
                scalar_feedback_per_world=budget*(2 if scenario == "Pair" else 1),
                original_questions_with_feedback=budget,
                rewarded_action_fraction_per_revealed_question=(2 if scenario == "Pair" else 1)/20,
                native_residual_actions_mean=float(np.mean([len(r["saverouter_diagnostics"]["residual_action_models"]) for r in rows])),
            )
    mapping, correct = [], []
    for fold in range(5):
        packet = load_npz(PRIVATE / f"inputs/fold{fold}/INPUTS_ONLY.npz")
        equal = packet["predicted_group_e"] == packet["oracle_group_e"]
        correct.extend(equal.tolist())
        mapping.append(dict(fold=fold, questions=120, group_identification_accuracy=float(equal.mean())))
    result = dict(status="COMPLETE", conditions=conditions, group_mapping_by_fold=mapping,
                  heldout_group_identification_accuracy=float(np.mean(correct)),
                  interpretation="Coverage and reward-free group identification are descriptive mechanisms, not a causal attribution of all performance changes",
                  old_E12_difference="E12 propagated group targets through a linear student; E13 uses a common hard predicted group to match the native public baseline",
                  unseen_cells="Missing reward cells carry no training rows; their predictions come from stated observed-only priors")
    write_json(PUBLIC / "MECHANISM_DIAGNOSTICS.json", result)
    return result


def plots(result, costs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "legend.frameon": False})
    paths = []
    with Timer() as timer:
        for x_kind in ("gold", "tokens"):
            fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
            for axis, scenario in zip(axes, SCENARIOS):
                xs = list(BUDGETS) if x_kind == "gold" else [costs["conditions"][scenario][str(b)]["single_generation_path"]["mean"]["tokens"]/1e6 for b in BUDGETS]
                for name in MAIN_PLOT_METHODS:
                    ys = [result["scenarios"][scenario][str(b)]["methods"][name]["accuracy_pct"] for b in BUDGETS]
                    fixed = name.endswith("Fixed")
                    axis.plot(xs, ys, color=COLORS[name], label=name,
                              linestyle="--" if fixed or name == "GroupResidual" else "-",
                              marker="s" if name == "GroupResidual" else "." if fixed else "o",
                              linewidth=1.2 if fixed else 1.7, markersize=3.5)
                axis.set_title("One observed combination" if scenario == "Main" else "Main + one neighboring combination", loc="left")
                axis.set_xlabel("Revealed original questions per generation trajectory" if x_kind == "gold" else
                                "Full feedback execution tokens (millions)")
                axis.grid(axis="y", color="#EEEEEE", linewidth=.7)
                if x_kind == "gold":
                    axis.set_xticks(BUDGETS)
                axis.set_ylim(83.5, 86.5)
            axes[0].set_ylabel("Final heldout aggregation accuracy (%)")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .06), ncol=4, fontsize=8)
            fig.subplots_adjust(left=.07, right=.98, top=.88, bottom=.31, wspace=.10)
            fig.text(.5, .015, "600 reused development questions; 3 generation x 3 logger trajectories retained within each question.\n"
                     "All methods share each full feedback ledger. Policy-specific final service tokens are reported separately. Oracle rows are excluded.",
                     ha="center", va="bottom", fontsize=7.5)
            path = PUBLIC / ("quality_gold.png" if x_kind == "gold" else "quality_feedback_tokens.png")
            fig.savefig(path, dpi=220)
            plt.close(fig)
            paths.append(path.name)
    return dict(backend="Python matplotlib Agg", font="Times New Roman", files=paths, timing=timer.record,
                contract="retain all fixed six-point quality curves and strong public controls; no uncertainty bands or best-envelope selection",
                data_sha256=sha256(PUBLIC / "E13_RESULTS.json"), cost_sha256=sha256(PUBLIC / "COST.json"))


def service_and_total_table(result, costs):
    rows = []
    for scenario in SCENARIOS:
        for budget in BUDGETS:
            feedback = costs["conditions"][scenario][str(budget)]["single_generation_path"]["mean"]
            for method, values in result["scenarios"][scenario][str(budget)]["methods"].items():
                service600 = values["service_execution_cost"]["mean_per_600_question_generation_trajectory"]
                service120 = {key: value/5 for key, value in service600.items()}
                rows.append(dict(scenario=scenario, gold_budget=budget, method=method,
                                 same_permission_main=not method.endswith("Oracle"),
                                 feedback_gold=feedback["gold"], feedback_members=feedback["member_calls"],
                                 feedback_aggregates=feedback["aggregator_calls"], feedback_scores=feedback["scoring_ops"],
                                 feedback_input_tokens=feedback["input_tokens"], feedback_output_tokens=feedback["output_tokens"],
                                 feedback_tokens=feedback["tokens"], service_questions=120,
                                 service_member_calls=service120["member_calls"], service_aggregate_calls=service120["aggregator_calls"],
                                 service_input_tokens=service120["input_tokens"], service_output_tokens=service120["output_tokens"],
                                 service_tokens=service120["tokens"],
                                 feedback_plus_120_service_tokens=feedback["tokens"]+service120["tokens"],
                                 aggregation_accuracy_pct=values["accuracy_pct"]))
    path = PUBLIC / "FEEDBACK_AND_SERVICE_COSTS.csv"
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def report(result, costs, primary, mechanism, service_rows):
    comp = primary["primary_comparisons"]
    lines = ["# E13 真正部分组合反馈下的组能力共享", "", "## 研究问题与结论", "",
             "本轮固定六个冻结模型、每次三个成员、20种组合与V2聚合器，比较同一部分反馈在不同学习器中的利用效果。", "",
             "**主预算Pair、Gold120未提供组共享独立优于强逐题控制或SaveRouter的统计支持。** Group相对原Raw有正向点值，RawCentered与公开基线仍具有竞争力。全部三个预定主比较的普通95%区间跨零，Holm调整p均为1。", "",
             "| 预定主比较 | 差值pp | 普通95%配对区间pp | 纠错 | 破坏 | 净事件 | Holm p |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for label, row in comp.items():
        lo, hi = row["ci95_pp"]
        lines.append(f"| {label} | {row['difference_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {row['corrections']} | {row['destructions']} | {row['net_events']:+d} | {row['p_holm']:.6f} |")
    lines += ["", "## 固定设置与合法反馈", "",
              "原train600为GSM8K和BBH各300题。沿用五个H480/E120外折，六个累计预算60、120、240、300、360、480。三个固定logger种子17/29/43与成员生成42/137/2024分别记录，形成90条轨迹和540个预算快照。",
              "Main每题购买一个三成员组合的单生成评分。Pair增加一个单成员替换邻居，仍然每次只执行三成员组合。反馈包由题目ID哈希确定，所有方法、各预算及三生成使用相同主动作和邻居，动作选择不读取奖励或实际费用。",
              "模型只接收已购买COO元组。缺失组合没有训练标签行。原600题完整矩阵由缓存管理器和最终评价器持有，E120不参与采购或更新。三生成分别训练、分别评价匹配生成，没有把一次购买转换为三份奖励。",
              "使用旧H480 MiniLM-PCA16与标准化状态，只读取四项预处理数组。旧全反馈拟合权重和旧C480均未进入新学习器。共同硬分组为argmax(Phi T_M)，T_M仅由训练输入和训练组metadata构建。", "",
              "## 学习器与既定选择", "",
              "Raw为按动作稀疏Ridge。单生成事件求和目标固定alpha=1000，常数受罚。完整三重复展开除3后恢复E12的1000/3均值尺度，资格检查已通过。",
              "Group固定grand→global-action→task→BBH-subtask收缩，先验伪计数为10/10/3。GSM只使用task层。GroupResidual排除当前整题的所有已购买动作后构造组能力基线，再以相同alpha拟合残差。",
              "RawCentered使用同一已观察全局动作先验加逐题回归，检查原Raw的零先验与观测数收缩是否构成混淆。这里使用既有平方损失和固定分解，没有新增损失搜索。",
              "GlobalFixed、TaskFixed、SubtaskFixed均从当前已观察反馈建立经验组合均值，缺少单元按task/global逐级回退。主任务/子任务固定策略使用共同的输入预测组。两项Oracle额外使用真实查询metadata，单独保留为诊断。",
              "**E13采用共同硬预测组，与E12的线性组监督传播学生不同。** 该选择用于与作者原生SaveRouter预测接口对齐。E12的87%端点和本轮部分反馈绝对成绩之间存在方法与监督单位差异，不能把全部点差都归因于减少标签。", "",
              "## SaveRouter可追溯适配", "",
              "官方commit固定为250b7ef35a741c598aaf1bb264930d1e0bfd1a61，原FixedKSparseRouter能力与残差估计器直接运行。LLMRouterBench profile为收缩40、先验Ridge10、残差Ridge200、残差倍率2、最少8个动作观察。主服务按质量argmax，未扫描费用权重或新超参。",
              "本轮将动作映射为20个真实组合，替换为共同fixed-mask logger和共同输入预测组。其费用模型接收已执行组合的独立完整服务Token，采购账仍按成员并集计算。作者类与适配器的分数、成本和模型恢复合成检查通过。",
              "该结果属于SaveRouter-FixedMask的同权限迁移。作者原数据、原获取器与原论文完整在线费用前沿的数值复现属于另一范围。四源码远端字节和本地缓存哈希一致，详细一手来源见LITERATURE_AND_BASELINE_MAPPING.md。", "",
              "## 主预算完整方法表", "",
              "| 方法 | 总准确率% | GSM8K% | BBH% | BBH子任务宏平均% | 服务输入权限 |",
              "|---|---:|---:|---:|---:|---|"]
    for name in METHODS:
        row = result["scenarios"]["Pair"]["120"]["methods"][name]
        permission = "真实metadata诊断" if name.endswith("Oracle") else "共同题目输入"
        lines.append(f"| {name} | {row['accuracy_pct']:.4f} | {row['gsm8k_pct']:.4f} | {row['bbh_pct']:.4f} | {row['bbh_subtask_macro_pct']:.4f} | {permission} |")
    lines += ["", "## 全部预算曲线", "", "![预算曲线](quality_gold.png)", ""]
    for scenario in SCENARIOS:
        lines += [f"### {scenario}", "", "| Gold | Raw | RawCentered | Group | GroupResidual | SaveRouter | GlobalFixed | TaskFixed | SubtaskFixed |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for budget in BUDGETS:
            methods = result["scenarios"][scenario][str(budget)]["methods"]
            values = [methods[n]["accuracy_pct"] for n in MAIN_PLOT_METHODS]
            lines.append("| " + str(budget) + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
        lines.append("")
    lines += ["", "表中单位均为%。Main与Pair的Group−RawCentered仅B60为正，其余五个预算点为负。Pair中Group−Raw有4/6正向点值，480点仍为负。GroupResidual相对Group的收益方向随预算变化。保留这些交叉，不能由主格点负结果推出任何组方法均无效。", "",
              "## 生成、logger与外折方向", "",
              "| 主比较 | seed42 pp | seed137 pp | seed2024 pp | logger17 pp | logger29 pp | logger43 pp |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in comp.items():
        vals = list(row["generation_difference_pp"].values()) + list(row["logger_difference_pp"].values())
        lines.append("| " + name + " | " + " | ".join(f"{v:+.4f}" for v in vals) + " |")
    lines += ["", "每个方法在每个预算有600个独立原题和5400个已记录生成×logger事件。所有27子任务、五外折、完整事件交叉及动作频率均在E13_RESULTS.json和E13_SUMMARY.csv。三项主比较按原题簇bootstrap5000次并统一Holm3。普通区间和调整p分别标注，训练历史重叠与已拟合模型条件性继续保留。", "",
              "## 完整反馈执行成本", "", "![实际Token曲线](quality_feedback_tokens.png)", "",
              "| 场景 | Gold | 成员生成 | 组合聚合 | 评分 | 平均实际反馈Token | Token最小–最大 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for scenario in SCENARIOS:
        for budget in BUDGETS:
            cell = costs["conditions"][scenario][str(budget)]["single_generation_path"]
            mean = cell["mean"]
            lines.append(f"| {scenario} | {budget} | {mean['member_calls']:.0f} | {mean['aggregator_calls']:.0f} | {mean['scoring_ops']:.0f} | {mean['tokens']:.2f} | {cell['min']['tokens']:.0f}–{cell['max']['tokens']:.0f} |")
    c = costs["conditions"]["Pair"]["120"]
    reuse = c["no_member_reuse_counterfactual"]["single_generation_path"]["mean"]
    saving = reuse["tokens_saved_by_shared_members"]
    ratio = 100*saving/reuse["tokens"]
    lines += ["", f"主Pair120单生成轨迹平均完整反馈为120 gold、480成员、240聚合、240评分、{c['single_generation_path']['mean']['tokens']:.2f}Token。表内分布覆盖45个外折×logger×生成路径。每个学习器完整共享这一账，采购费只记一次。",
              f"不复用共同成员的账本反事实平均为{reuse['tokens']:.2f}Token，合法复用节省{saving:.2f}Token，约{ratio:.2f}%。该项是所有方法共同拥有的执行权限。",
              "三个生成轨迹若作为一组实际执行，Gold按canonical/reference版本并集仍为B，Main成员/聚合为9B/3B，Pair为12B/6B。Pair120对应1440成员、720聚合、720评分和120 gold。三个模型仍各自只接受其生成轨迹的反馈。",
              "六个预算是嵌套累计场景，五个外折历史重叠，三个logger也是重复方案。COST.json分别保存每路径、跨生成并集与费用分布，不能将这些累计账相加视为新采购。参考版本为固定V2缓存标签，未将其冒称为Gold内容哈希。", "",
              "## 服务与总执行Token", "",
              "同预算、同mask的完整反馈成本相同。各方法在E120选择不同组合，最终服务Token可能不同。下表将一份B120反馈采购与120题服务推理按相同世界口径合计。", "",
              "| 主Pair120方法 | 共同反馈Token | 120题服务Token | 两项合计Token |",
              "|---|---:|---:|---:|"]
    for row in service_rows:
        if row["scenario"] == "Pair" and row["gold_budget"] == 120 and row["same_permission_main"]:
            lines.append(f"| {row['method']} | {row['feedback_tokens']:.2f} | {row['service_tokens']:.2f} | {row['feedback_plus_120_service_tokens']:.2f} |")
    lines += ["", "服务列包含所选三成员与一次聚合，推理本身的gold/评分为0。共同研究评测参考答案与完整历史缓存投入另列。服务表由九条600题轨迹的原调用账，除以五折得到平均120题世界。FEEDBACK_AND_SERVICE_COSTS.csv保留所有方法与预算。",
              "题目编码器的既有缓存属于共同输入准备，当前没有重新提取表征。CPU组件另列，未新测各方法独立端到端部署延迟。GlobalFixed实际可以省去编码步骤，当前Token表只比较语言模型执行。源调用wall之和保持历史代理口径，当前GPUelapsed为未测。", "",
              "## 支持量与失效原因边界", "",
              f"共同组映射的外折组识别率为{100*mechanism['heldout_group_identification_accuracy']:.2f}%。这一指标描述组识别，不等于组合选择准确率。", "",
              "| Gold | Main已观察组–组合单元比例% | Pair已观察组–组合单元比例% |",
              "|---:|---:|---:|"]
    for budget in BUDGETS:
        vals = [100*mechanism["conditions"][s][str(budget)]["observed_group_action_fraction"] for s in SCENARIOS]
        lines.append(f"| {budget} | {vals[0]:.2f} | {vals[1]:.2f} |")
    lines += ["", "28组×20组合共有560个组–组合单元。缺失单元依靠已声明的观测统计先验，未填入零奖励。覆盖不足、有限组识别误差、固定收缩强度和有限表示均是可能的限制，当前对照没有识别唯一因果原因。",
              "RawCentered使得零先验/观测数收缩的影响可单独观察。Group未超过它的主点值，因而Group相对原Raw的小幅改善不足以归因为任务组共享的独立价值。残差也未稳定超过既有公开基线。", "",
              "## 优势、风险与研究决定", "",
              "**优势**是相同稀疏反馈、完整共享执行账、奖励无关logger、单生成单位及原生公开估计器迁移。对成本的比较没有免费读出其他18或19个组合的评分。",
              "**风险**是旧600题反复开发、各外折训练池重叠、每题每轨迹仅1/2动作，hard分组与有限固定profile。SaveRouter保留作者profile，未在当前任务调参，这个迁移结果不能代表其所有可调配置的性能上限。",
              "**本轮假说尚未得到支持。** 当前可保留SaveRouter-FixedMask与RawCentered作为真正部分反馈的强参照。下一优先任务是定位低支持组–组合单元估计与输入分组误差各自造成的决策损失，再决定是否有必要改变固定logger的覆盖设计。当前不扩展新损失或主动获取网格。", "",
              "## 实际计算与交付", ""]
    cpu = costs["computation"]
    lines += [f"实际完成{cpu['actual_unique_checkpoint_fits']}个快照拟合，含1个机制预检并复用于正式训练。自己的三个Ridge右端共享{cpu['own_actual_normal_matrix_solves']}次小矩阵求解，对应{cpu['own_model_right_hand_side_fits']}个RHS目标。SaveRouter完成{cpu['native_saverouter_prior_quality_fits']}个原生先验拟合和{cpu['native_saverouter_residual_quality_models']}个残差模型拟合。",
              f"正式编排wall为{cpu['formal_orchestrator_wall_s']:.3f}秒。拟合、logger/更新、缓存读取、资格和统计CPU分别保留，嵌套计时与并行wall不相加伪装总耗时。新增LLM/API、成员推理、聚合、Gold采购和保留集载荷读取均为0。",
              "私有观察、模型、逐题预测、合法访问日志、实际费用及冻结哈希位于private。公开协议、完整结果、成本、源码和复现入口位于public。MAB工作区没有本地Git检出，本轮交付采用实际文件哈希和来源版本，未虚构新的仓库commit。", ""]
    path = PUBLIC / "EXPERIMENT_SUMMARY.md"
    if path.exists():
        raise FileExistsError(path)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    verify_source_lock()
    write_json(PRIVATE / "REPORT_SOURCE_FROZEN.json", dict(source_sha256=sha256(__file__),
               results_sha256=sha256(PUBLIC / "E13_RESULTS.json"), costs_sha256=sha256(PUBLIC / "COST.json")))
    with Timer() as timer:
        result = read_json(PUBLIC / "E13_RESULTS.json")
        costs = read_json(PUBLIC / "COST.json")
        primary = read_json(PUBLIC / "E13_PRIMARY_STATISTICS.json")
        mechanism = mechanism_diagnostics()
        service_rows = service_and_total_table(result, costs)
        figures = plots(result, costs)
        report(result, costs, primary, mechanism, service_rows)
    write_json(PRIVATE / "REPORT_RUN.json", dict(status="COMPLETE", timing=timer.record, figures=figures,
               report_sha256=sha256(PUBLIC / "EXPERIMENT_SUMMARY.md")))
    print("E13_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
