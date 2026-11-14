"""Generate the requested Chinese report, cost ledger, and quality/gain figure."""
from __future__ import annotations

import csv
import itertools
from pathlib import Path
import time

from common import (
    CODE, OUTCOMES, PERMUTATION_SEEDS, PRIVATE, RESULTS, Timer,
    load_npz, read_json, sha256, write_json,
)
import numpy as np


def service_cost(outcomes, actions):
    teams = np.array(list(itertools.combinations(range(6), 3)))
    question = np.arange(600)
    resource = {"gold": 0, "scoring_ops": 0}
    for stem in ("member", "aggregate"):
        for suffix in ("execution_counts", "prompt_tokens", "completion_tokens", "wall_time_s"):
            values = outcomes[f"{stem}_{suffix}"]
            selected = values[:, question[:, None], teams[actions]] if stem == "member" else values[:, question, actions]
            resource[f"{stem}_{suffix}"] = float(selected.sum()) if suffix == "wall_time_s" else int(selected.sum())
    resource["total_tokens"] = sum(resource[f"{stem}_{suffix}"] for stem in ("member", "aggregate")
                                    for suffix in ("prompt_tokens", "completion_tokens"))
    resource["unit"] = "execution-equivalent frozen service traces over 600 questions x 3 existing repeats"
    return resource


def build_cost(result):
    records = []
    for pattern, role in (("main/fold*/FITTING_LOG.json", "main"),
                          ("controls/seed*/fold*/FITTING_LOG.json", "control")):
        for path in PRIVATE.glob(pattern):
            log = read_json(path)
            if len(log["fits"]) != 56:
                raise ValueError("Incomplete recorded nested-fitting path")
            records.extend((role, row["diagnostics"]) for row in log["fits"])
    if len(records) != 28280:
        raise ValueError(f"Unexpected executed head count {len(records)}")
    fit_summary = {}
    for role in ("main", "control"):
        for family in ("PSMR", "HSMR"):
            selected = [row for kind, row in records if kind == role and row["family"] == family]
            fit_summary[f"{role}_{family}"] = dict(
                actual_head_fits=len(selected), cpu_s=sum(r["cpu_seconds"] for r in selected),
                wall_s_sum=sum(r["wall_seconds"] for r in selected),
                maximum_gradient_norm=max(r["gradient_l2"] for r in selected),
                optimizer_failures=sum(not r["optimizer_success"] for r in selected),
                maximum_iterations=max(r.get("iterations", 1) for r in selected),
            )
    frozen = read_json(PRIVATE / "ALL_PREDICTIONS_FROZEN.json")
    feature = [read_json(PRIVATE / f"preprocessing/fold{fold}/FROZEN.json") for fold in range(5)]
    asset = read_json(RESULTS / "00_asset_manifest.json")
    math = read_json(RESULTS / "01_math_qualification.json")
    signal = read_json(RESULTS / "02_signal_diagnostics.json")
    outcome = load_npz(OUTCOMES)
    paired = load_npz(PRIVATE / "PAIRED_OUTCOMES.npz")
    methods = paired.get("methods", paired.get("method_names"))
    if methods is None:
        raise ValueError("Private paired outcome method names missing")
    actions = paired["actions"]
    if actions.shape[0] != len(methods):
        actions = actions.T
    history = {"distinct_original_questions": 600, "gold": 600,
               "member_calls": int(outcome["member_execution_counts"].sum()),
               "aggregation_calls": int(outcome["aggregate_execution_counts"].sum()),
               "aggregate_scoring_operations": 36000,
               "input_tokens": int(outcome["member_prompt_tokens"].sum() + outcome["aggregate_prompt_tokens"].sum()),
               "output_tokens": int(outcome["member_completion_tokens"].sum() + outcome["aggregate_completion_tokens"].sum()),
               "new_physical_expenditure_this_round": 0}
    costs = dict(
        status="COMPLETE", head_fits=fit_summary, total_new_head_fits=len(records),
        main_head_fits=280, control_head_fits=28000,
        pilot_heads=18, pilot_heads_reused_in_main=18,
        pilot=read_json(PRIVATE / "RESOURCE_PILOT.json"),
        preprocessing=dict(actual_PCA16_scaler_fits=20,
                           cpu_s=sum(r["preprocessing_cost"]["cpu_s"] for r in feature),
                           wall_s_sum=sum(r["preprocessing_cost"]["wall_s"] for r in feature),
                           control_reuse="All 100 controls reuse exactly matched reward-free fitting-index preprocessing"),
        formal_training_orchestrator_wall_s=frozen["wall_s"],
        sum_worker_process_cpu_s=sum(r["time"]["cpu_s"] for r in frozen["paths"]),
        sum_worker_path_wall_s=sum(r["time"]["wall_s"] for r in frozen["paths"]),
        maximum_worker_peak_rss_bytes=max(r["time"]["peak_rss_bytes"] for r in frozen["paths"]),
        software_qualification=read_json(RESULTS / "01_software_qualification.json")["timing"],
        asset_audit={k: asset[k] for k in ("cpu_seconds", "wall_seconds")},
        mathematics_qualification={k: math[k] for k in ("cpu_seconds", "wall_seconds")},
        reward_signal_diagnostics=signal["cost"],
        qualification_development_failure=dict(phase="before source freeze", type="synthetic test expected-array shape mismatch",
            model_change=False, corrected=True, CPU_s=None, wall_s=None,
            note="Independent initial process time was not retained; recorded subsequent 20-test qualifications pass"),
        postprocessing_retry=(read_json(PRIVATE / "REPORT_SCAN_RETRY.json")
                              if (PRIVATE / "REPORT_SCAN_RETRY.json").exists() else None),
        evaluation=result["evaluation_cost"],
        historical_full_feedback=history,
        physical_new=dict(language_model_calls=0, api_calls=0, gold=0, aggregation_calls=0, new_embeddings=0),
        retained_payload_access=dict(validation200=0, fresh400=0, sealed200=0),
        service_execution_equivalent={str(name): service_cost(outcome, actions[i]) for i, name in enumerate(methods)},
        notes=["Component wall sums overlap under three workers and are not end-to-end elapsed time",
               "Original common feedback is paid once, not once per method, fold, seed or permutation",
               "Service traces are cached counterfactual accounting, no physical service inference was run",
               "No monetary currency rate or new end-to-end online MiniLM latency was measured",
               "The first suite execution and the agents' small qualification checks also used CPU; their independent logs are retained where measured"],
    )
    write_json(RESULTS / "03_cost_ledger.json", costs)
    grid_rows = []
    for fold in range(5):
        log = read_json(PRIVATE / f"main/fold{fold}/FITTING_LOG.json")
        for row in log["candidates"]:
            grid_rows.append(dict(outerfold=fold, family=row["family"], multiplier=row["multiplier"],
                                 loss=row.get("loss", "mse"), shrinkage=row.get("shrinkage", ""),
                                 inner_accuracy_pct=100*row["inner_accuracy"],
                                 selected=row == log["selected"][row["family"]]))
    with (RESULTS / "03_hyperparameter_grid.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)
    return costs


def make_plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    chinese = "Microsoft YaHei" if "Microsoft YaHei" in fonts else "DejaVu Sans"
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "savefig.facecolor": "white"})
    order = ["GlobalFixed", "TaskFixed", "SubtaskFixed", "PriorShrink", "M0_original", "M4", "NS_main", "PSMR", "HSMR", "E9Rule"]
    order = [name for name in order if name in result["methods"]]
    colors = ["#9A9A9A" if name in order[:4] else "#597FA3" if name in ("M0_original", "M4", "NS_main")
              else "#8064A2" for name in order]
    with Timer() as timer:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
        y = np.arange(len(order))
        means = np.array([result["methods"][name]["accuracy_pct"] for name in order])
        ci = np.array([result["methods"][name]["accuracy_ci95_pct"] for name in order])
        for i in range(len(order)):
            axes[0].errorbar(means[i], y[i], xerr=[[max(0, means[i]-ci[i, 0])], [max(0, ci[i, 1]-means[i])]],
                             fmt="o", color=colors[i], markersize=4, capsize=2)
        axes[0].set_yticks(y, order)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Final aggregation accuracy (%)")
        axes[0].set_title("a   Frozen development OOF quality", loc="left")
        for i, (name, row) in enumerate(result["comparisons"].items()):
            low, high = row["family_ci975_pp"]
            point = row["difference_pp"]
            axes[1].errorbar(point, i, xerr=[[max(0, point-low)], [max(0, high-point)]],
                             fmt="o", color="#8064A2", markersize=5, capsize=3)
        axes[1].set_yticks(range(len(result["comparisons"])), list(result["comparisons"]))
        axes[1].invert_yaxis()
        axes[1].axvline(0, color="#777777", linewidth=.8, linestyle="--")
        axes[1].set_xlabel("Paired accuracy difference (percentage points)")
        axes[1].set_title("b   Predeclared complete-rule comparisons", loc="left")
        fig.subplots_adjust(left=.13, right=.98, top=.87, bottom=.23, wspace=.85)
        fig.text(.5, .06, "600 original questions; three repeats clustered per question; 10,000 stratified paired bootstraps.\n"
                 "Left: ordinary 95% intervals. Right: 97.5% Bonferroni intervals for a two-comparison family.\n"
                 "Fixed/prior/fallback strategies use declared task/subtask metadata; learned heads use text only.",
                 ha="center", va="center", fontsize=8)
        fig.savefig(RESULTS / "05_figure_quality_and_gain.png", dpi=220)
        plt.close(fig)
    return dict(backend="Python matplotlib Agg", font="Times New Roman", chinese_font_available=chinese,
                title_language="English", conclusion="Compare frozen rule quality and predeclared gains, retaining zero/negative results",
                timing=timer.record, source=sha256(RESULTS / "03_main_results.json"))


def make_report(result, gate, costs, plot, output_path=None):
    comparisons = result["comparisons"]
    next_plan = ("冻结完整E9规则，单独制定独立问题确认方案，重新估算配对分歧、所需题量与最小调用并集，再申请相应授权。"
                 if gate["decision"] == "GO_FRESH_REQUEST" else
                 "优先推进低预算任务/子任务组能力估计，在固定反馈掩码下研究同一监督的利用效率，并与已有SaveRouter收缩基线做同权限对照。")
    lines = ["# E9 Bayes决策对齐、奖励可辨识性与保守组合路由", "", "## 两项主比较与研究决策", ""]
    for key, row in comparisons.items():
        lo, hi = row["ci95_pp"]
        flo, fhi = row["family_ci975_pp"]
        lines.append(f"- **{key}** 为{row['difference_pp']:+.4f}个百分点，普通95%区间[{lo:+.4f}, {hi:+.4f}]，"
                     f"双比较家族区间[{flo:+.4f}, {fhi:+.4f}]，Holm调整p={row['p_holm']:.6f}。")
    lines += ["", f"**研究决策为 {gate['decision']}。**", "",
              "这里评价的是完整内层选模及默认回退规则。PSMR和HSMR分别成绩保留为预定开发对照，全部结果来自既有600题的开发交叉验证。", "",
              "## 实际实验设置", "",
              "六个冻结模型，每题选择三个，共20个无序组合，V2聚合器与原评分器保持冻结。生成重复42、137、2024始终作为同题簇保留。",
              "每个外折480题训练、120题评价，训练区内部沿用原三折。每个合法拟合区独立PCA16、原标准化及受罚常数，头为17×20共340参数。",
              "PSMR采用20个独立sigmoid，内选BCE或Brier。HSMR使用整题LOO的global/task/subtask平滑先验，固定task/subtask伪计数10/3，收缩候选0/1/3/9。正则multiplier为0.1/1/10。未增加可选校准参数。",
              "完整规则按内层真实未见题准确率选择PSMR或HSMR，严格优于内层子任务默认时启用，否则保留默认。平局优先HSMR，候选内并列优先更强正则。规则在外折评分前冻结。",
              "固定策略、层级先验与完整规则的回退部分使用真实task/subtask metadata。两个学习头在部署时仅使用题目向量。metadata对照的权限在表中明确列示。", "",
              "## 主实验表", "",
              "| 方法 | 任务宏平均% | GSM8K% | BBH% | BBH子任务宏平均% | 经验选队损失 | 权限 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for name in result["method_order"]:
        row = result["methods"][name]
        permission = row["input_permission"].replace("; ", "，")
        lines.append(f"| {name} | {row['accuracy_pct']:.4f} | {row['gsm8k_pct']:.4f} | {row['bbh_pct']:.4f} | "
                     f"{row['bbh_subtask_macro_pct']:.4f} | {row['empirical_regret_mean']:.6f} | {permission} |")
    lines += ["", "经验选队损失以本题三生成平均奖励最大值为参照，其总体潜在期望仍有估计误差。", "",
              "GlobalFixed遵循本轮冻结的先三生成平均、再题平均与精确argmax。数学并列值可能受浮点计算顺序影响，旧研究已记录84.9444%与当前84.7778%的默认点差。本轮保留已冻结选择，未在看到成绩后更换并列规则。", "",
              "![质量与配对收益](05_figure_quality_and_gain.png)", "", "## 各生成方向、纠错和破坏", "",
              "| 方法 | seed42% | seed137% | seed2024% | 相对M0纠错 | 破坏 | 净事件 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in result["method_order"]:
        row = result["methods"][name]
        event = row["versus"]["M0_original"]
        seed = row["by_generation_seed"]
        lines.append(f"| {name} | {seed['42']['accuracy_pct']:.4f} | {seed['137']['accuracy_pct']:.4f} | "
                     f"{seed['2024']['accuracy_pct']:.4f} | {event['corrections']} | {event['destructions']} | {event['net_events']:+d} |")
    lines += ["", "事件分母为600题×3个已记录生成，即1800。各外折和27个BBH子任务的题数与成绩完整保存在JSON。", "",
              "## 内容负对照与机制", "",
              "完成100个预先固定种子，每个合法inner/outer拟合区分别在GSM或BBH子任务内置乱整题3×20奖励块。对应模型执行相同内层网格与选模规则，特征预处理按相同合法输入缓存复用。", "",
              "| 方法 | 原始训练% | 100次控制均值% | 控制范围% | 原始减均值pp | Monte Carlo p |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, row in result["negative_controls"]["per_method"].items():
        lines.append(f"| {name} | {row['main_accuracy_pct']:.4f} | {row['control_mean_pct']:.4f} | "
                     f"{row['control_min_pct']:.4f}–{row['control_max_pct']:.4f} | {row['main_minus_control_mean_pp']:+.4f} | {row['monte_carlo_p']:.6f} |")
    lines += ["", "100次置乱属于训练机制控制，未增加独立题数。标准Monte Carlo p的最小分辨率为1/101，不能把重复置乱视为100个外部测试集。", "",
              "PSMR与HSMR的原始训练成绩均低于各自100次控制均值。这一配置下，题目内容对应关系尚未显示超越训练组先验的选队收益。PriorShrink直接使用任务/子任务信息，其较高点值保留为组能力估计线索。",
              "HSMR相对M0的正向点值是预定候选对照，完整规则按内层选出的跨折组合仍需按预定主结果报告。未根据外折成绩改选HSMR为正式获胜方案。", "",
              "| 方法 | 原始均值MSE | 中心化MSE | 非零竞争差MAE | 非零配对排序准确率 | 最频繁动作题数 | 近并列题数≤1e-10 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("M0_original", "PSMR", "HSMR", "PriorShrink", "E9Rule"):
        row = result["methods"][name]
        score = row["score_diagnostics"]
        lines.append(f"| {name} | {score['raw_mean_mse']:.6f} | {score['centered_mean_mse']:.6f} | "
                     f"{score['nonzero_competition_gap_mae']:.6f} | {score['strict_nonzero_pair_rank_accuracy']:.6f} | "
                     f"{row['action_distribution']['max_count']} | {score['near_top_tie_questions_1e_10']} |")
    lines += ["",
              "动作集中度、近并列、非零竞争差MAE与真实Brier保存在逐方法score_diagnostics。竞争组合由原M0冻结Top3相关配对定义，非零子集按本题经验奖励差确定，仅用于诊断。",
              "M4/NS输出为未校准分数。其原始数值误差与PSMR概率Brier具有不同含义。独立Bernoulli输出允许多个动作同时正确。", "",
              "数学检查确认平方损失总体最优为条件均值，真实实现后regret的总体目标同样保持均值排序。有限线性类、正则、噪声与SPO替代风险分别分析，未将它们混为MSE总体不一致。",
              "历史M4的全平局赢家约定可引入动作编号方向。新候选无赢家索引监督，软件同步重命名测试通过。数学反例和软件PASS各自只支持相应命题。", "",
              "跨生成稳定性与奖励可辨识性详见02_signal_report.md。两seed选队、第三seed评价是使用同题反馈的hindsight诊断。", "",
              "## 完整成本与复现", "",
              f"实际完成{costs['total_new_head_fits']}个新预测头拟合，主实验{costs['main_head_fits']}个，100次内容控制{costs['control_head_fits']}个。18个资源预检头全部复用于主实验。另执行20次合法PCA/Scaler拟合并在控制中复用。",
              f"正式训练编排wall为{costs['formal_training_orchestrator_wall_s']:.3f}秒，worker路径CPU合计{costs['sum_worker_process_cpu_s']:.3f}秒。每worker最大记录峰值RSS为{costs['maximum_worker_peak_rss_bytes']/1024**2:.2f}MiB。",
              "组件耗时与并行总wall分别保留，逐候选目标、梯度、收敛、内层分数及失败状态均在私有日志。全部新优化器成功，较差候选和种子完整保留。",
              "首次合成测试开发中曾修正期望数组形状断言，预测模型未因此改变，该次单独进程耗时未保留。正式源锁前后两次20项资格执行均通过，正式头拟合失败数为0。",
              "成本汇总首次逐小文件扫描受I/O限制，记录10.703125秒CPU和190.803119秒wall后，改读同一批诊断的冻结聚合日志。该次后处理保留在成本账本，模型和统计均未重跑。",
              "新增LLM/API、成员生成、聚合、gold和向量提取均为0。共同历史投入为600 gold、10800成员调用和36000聚合调用。各策略服务Token按真实缓存中的所选三成员与聚合器分账，未把训练缓存采购费用当成免费服务。",
              "实际命令、依赖及私有输入清单见[复现说明](D:/develop/CV/多臂老虎机/experiments/e9_bayes_gate/README.md)。数据hash与相对来源路径见00_asset_manifest.json。原协议和必要PASS状态加强修订均保留为不可覆盖文件。", "",
              "## 统计与证据等级", "",
              "10000次bootstrap按outerfold×task重采样原题，各方法与同题三生成一起重采样。普通95%区间逐比较报告，主族另报Bonferroni97.5%区间与Holm调整p。后者与普通区间区分标注。",
              "五折训练历史重叠，区间条件于已拟合模型。旧600题已多轮开发，bootstrap不包含整个训练过程重复不确定性。当前证据等级为数学构造、软件资格和开发OOF。未来独立泛化尚待单独设计。", "",
              "## 优势与风险", "",
              "**优势**是恢复历史冻结参照，两个新方向保持相同输入维数和预测头容量，所有选参在合法内层完成，100次内容控制保留任务先验与题内相关结构。",
              "**风险**是每题仅三次已有生成，BBH每子任务训练支持有限。层级伪计数与有限正则网格为固定实现选择，Sigmoid-Brier具有非凸性。metadata默认与text-only头的输入权限不同，需要明确区分。",
              "当前工作目录没有本地Git checkout，本轮按任务书仅保存本地新增文件，Git分支未创建。来源SHA引用已保存的E12远端回执，未在本轮重新联网确认。", "",
              "## 决策与唯一下一优先任务", "",
              f"未满足条件为 {', '.join(gate['failed_conditions']) if gate['failed_conditions'] else '无'}。",
              next_plan + "此处仅给下一阶段选择，不新增实验。", "",
              "## 给导师的摘要", "",
              "本轮在既有600道题上完成独立Bernoulli奖励预测与层级收缩监督的嵌套开发验证，保持六模型、三成员组合、原MiniLM表示和340参数头。"
              f"完整规则相对子任务默认和原Ridge分别为{comparisons['E9Rule-SubtaskFixed']['difference_pp']:+.4f}、"
              f"{comparisons['E9Rule-M0_original']['difference_pp']:+.4f}个百分点，研究决策为{gate['decision']}。"
              "历史参照均恢复，并完成100次整题内容负对照。数学核查区分了条件均值目标、替代风险和并列编号偏好。全部负向候选保留，结果限于重复开发资产，保留题与新模型调用均未使用。需要请教的关键选择是，是否将固定反馈下的低预算组能力估计作为下一主线，再依据独立协议讨论部分动作反馈。", ""]
    path = Path(output_path) if output_path is not None else RESULTS / "07_scientific_report_zh.md"
    if path.exists():
        raise FileExistsError(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    return plot


def main():
    result = read_json(RESULTS / "03_main_results.json")
    gate = read_json(RESULTS / "06_gate_decision.json")
    costs = build_cost(result)
    plot = make_plot(result)
    make_report(result, gate, costs, plot)
    write_json(PRIVATE / "REPORT_AND_FIGURE_RUN.json", plot)
    print("REPORT_AND_COST_COMPLETE")


if __name__ == "__main__":
    main()
