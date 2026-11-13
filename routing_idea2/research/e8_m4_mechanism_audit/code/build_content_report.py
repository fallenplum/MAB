"""Render reports exclusively from frozen M4 diagnostic aggregates."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
result = json.loads((HERE / "M4_CONTENT_AND_POWER_RESULTS.json").read_text(encoding="utf-8"))


def pp(value):
    return f"{100 * value:+.3f}"


def ci(value):
    return f"[{pp(value[0])}, {pp(value[1])}]"


lines = [
    "# M4题目内容负对照与独立确认功效规划", "",
    "## 研究问题", "",
    "检查原M4的正向点值对题目内容的依赖，并估计冻结单一候选后开展独立配对评价所需的题量。现有600题的结果属于开发诊断。", "",
    "## 实验设置与对照", "",
    "复用E8已冻结的五折M4、M0和SHUFFLE_M4预测。每题的三个生成重复42、137、2024一起构成一个题簇。Bootstrap在外折×任务内按原始题配对抽样10000次，固定种子2026111403。GSM8K与BBH各300题。", "",
    "SHUFFLE_M4保留每道题20组合×3重复的整个奖励块，在每个合法训练区的同一任务/BBH子任务内部置乱。全部五折的最终映射与内层拟合映射均通过组内、问题分离和完整奖励块核查。该对照继承M4已选正则，不再次调参。它提供固定配置的内容诊断。", "",
    "输入屏蔽干预直接取原模型的常数系数W[0]，将16维问题特征全部置零。五折均选择组合0。预测与动作先冻结，随后独立评价器才读取历史train600评分。本项保留原训练得到的截距，纯先验重新拟合由选队分布报告另外提供。", "",
    "## 总体数值结果", "",
    "| 比较对象 | M4准确率 | 对照准确率 | M4收益，百分点 | 配对95%CI，百分点 | 纠错/破坏，题×生成记录 |", "|---|---:|---:|---:|---|---:|",
]
for key, name in [("M0_original", "原Ridge"), ("SHUFFLE_M4", "任务内奖励块置乱M4"),
                  ("M4_input_mask", "原M4只保留常数项"), ("train_global_fixed", "训练区全局固定组合")]:
    row = result["comparisons"][key]["overall"]
    lines.append(f"| {name} | {row['accuracy_left']*100:.3f}% | {row['accuracy_right']*100:.3f}% | {pp(row['mean'])} | {ci(row['ci95'])} | {row['corrected_events']}/{row['destroyed_events']} |")
lines += ["", "输入屏蔽收益区间的下端由数值浮点误差得到−7.4×10⁻¹⁷个百分点，科学解释按0处理。所有区间为未校正的探索性区间。E8原报告的bootstrap种子与本轮不同，本轮结果独立保留，原记录保持原样。", "",
    "M4相对原Ridge有95个纠错与83个破坏事件，净12个事件。题级三重复平均差在59题为正、50题为负、491题为零。543次组合改变同时包含大量收益为零的改变。", "",
    "## 任务、生成重复与外折方向", "",
    "| 对照 | GSM8K收益，百分点 | BBH收益，百分点 | seed42/137/2024收益，百分点 | fold0—4收益，百分点 |", "|---|---:|---:|---|---|",
]
for key, name in [("SHUFFLE_M4", "任务内置乱"), ("M4_input_mask", "只保留常数项"), ("M0_original", "原Ridge")]:
    row = result["comparisons"][key]
    lines.append(f"| {name} | {pp(row['task']['gsm8k']['mean'])} | {pp(row['task']['bbh']['mean'])} | {' / '.join(pp(x) for x in row['overall']['seed_gain'])} | {' / '.join(pp(x['mean']) for x in row['fold'])} |")
lines += ["", "M4与置乱训练的差值在GSM8K为负、BBH为正，三个生成重复中两个正、一个负，五折中三个正、两个负。当前600题没有给出稳定的内容依赖收益证据。输入屏蔽后的损失主要来自BBH，提示问题特征可能对一部分题有作用。它与任务内标签置乱对照共同支持继续区分内容信息、组合先验及优化偏好。", "",
    "### BBH子任务支持量", "",
    "| 子任务 | 独立题数 | M4减任务内置乱，百分点 | M4减输入屏蔽，百分点 |", "|---|---:|---:|---:|",
]
mask_sub = {x["subtask"]: x for x in result["comparisons"]["M4_input_mask"]["subtask"]}
for row in result["comparisons"]["SHUFFLE_M4"]["subtask"]:
    lines.append(f"| {row['subtask']} | {row['questions']} | {pp(row['mean'])} | {pp(mask_sub[row['subtask']]['mean'])} |")
lines += ["", "每个子任务只有11—12道独立问题。以上27项用于支持量和异质性诊断，完整题级配对区间保存在JSON，当前结果未构成经过多重检验的子任务结论。", "",
    "## 独立确认的统计功效", "",
    "以同一道题上M4与参考方案的三生成平均差d为观测。配对标准误使用题级差的标准差除以独立题数平方根。该计算保留同题生成重复的相关性。[NIST配对观察方法](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm)", "",
    "对未来GSM8K/BBH各占50%的独立题，采用规划方差σ²=0.5(s²_GSM+s²_BBH)。原Ridge参照的σ为0.206614，全局固定参照为0.209161。已知方差的单侧正态近似为Power≈Φ(δ√N/σ−z₀.₉₅)，80%所需题数约为[σ(z₀.₉₅+z₀.₈)/δ]²。代码同时保存双侧5%与非中心t分布敏感性计算。[statsmodels功效文档](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.power.html) [作者功效实现](https://www.statsmodels.org/stable/_modules/statsmodels/stats/power.html#normal_sample_size_one_tail)", "",
    "### 单侧5%检验对原Ridge的规划功效", "",
    "| 独立题数 | 当前点值0.667pp | 真实收益1pp | 真实收益2pp | 真实收益3pp | 80%可检测最小收益 |", "|---:|---:|---:|---:|---:|---:|",
]
power = result["power"]["M0_original"]
for n in (200, 400, 600):
    rows = [x for x in power["grid"] if x["sided"] == "one" and x["questions"] == n]
    mde = next(x for x in power["minimum_detectable_gains"] if x["sided"] == "one" and x["questions"] == n and x["target_power"] == .8)
    power_cells = " | ".join(f"{100 * x['power_normal']:.1f}%" for x in rows)
    lines.append(f"| {n} | {power_cells} | {100*mde['approximate_minimum_detectable_gain']:.2f}pp |")
lines += ["", "### 所需独立题数", "",
    "| 真实收益 | 单侧5%，80%功效 | 单侧5%，90%功效 | 双侧5%，80%功效 |", "|---:|---:|---:|---:|",
]
for delta in (1 / 150, .01, .02, .03):
    table = power["approximate_required_n"]
    values = [next(x["approximate_required_questions"] for x in table if x["sided"] == side and x["target_power"] == target and x["delta"] == delta)
              for side, target in (("one", .8), ("one", .9), ("two", .8))]
    lines.append(f"| {delta*100:.3f}pp | {' | '.join(str(x) for x in values)} |")
lines += ["", "若要求80%功效检出1pp对原Ridge的正收益，规划题量约2640道。若实际题级方差比开发估计高50%，题量约3960道。相对全局固定组合的对应80%题量为2705道。200题对当前0.667pp点值的近似功效仅11.7%，对1pp为16.8%。", "",
    "这些数值依赖历史OOF差值方差和指定的真实效果。五个外折的训练历史互相重叠，最终以600题拟合的一套部署模型也会改变配对差分布。当前点值经过开发研究观察，含选择乐观性。以上功效具有规划用途，置信度为中。它没有使用保留题奖励。", "",
    "已知正态总体的软件校验使用10万次统计量模拟。零效应拒绝率4.966%，理论5%。1pp、400题拒绝率24.774%，理论24.925%，差异落在模拟误差范围内。该校验检验计算实现。", "",
    "## 保留题角色与先行注册草案", "",
    "既有协议与完成收据记录fresh400评分未读、sealed200未读。fresh400原角色是冻结后的一次独立开发评价，sealed200原角色是最终确认。此前少样本计划对fresh400拟作56道Support、344道Query，完成收据注明执行尚未开始。本轮仅读取这些角色文档和布尔标志，实际保留题奖励未解码。完整全局历史曝光审计应在将来评价前核对，现有声明证据保存在JSON的heldout_role_metadata。", "",
    "下一轮先行确认的主候选保持原字典序真实最优标签的M4。反向字典序和均匀并列版本仅用于本轮机制对照，其开发点值不进入候选版本选择。先固定单一M4配置和单一原Ridge参考，再一次性写出所有题的动作并封存。600题全量模型需采用事先记录的确定性正则选择规则，保存特征、权重、组合顺序及哈希。M4真实最优并列规则与推理并列规则必须一起冻结。届时保留题只评价这一套候选，评价结果不用于选择正则或并列版本。", "",
    "主效应采用同题三生成平均质量差，GSM8K与BBH各50%。固定评价时点的主假设为G≤0，对立为G>0，单侧α=0.05。最小实际意义收益预声明为1pp。支持正收益采用下限>0，支持至少1pp收益采用下限>1pp，并分别报告。此时若真实收益恰为1pp，达到后一标准的概率在边界上约为5%。若真实收益为2pp，后一标准的功效对应上表1pp列。", "",
    "保留200题可以给出冻结系统的效果区间，80%功效对应约3.63pp的正收益。400题对应约2.57pp。当前亚百分点点值下，优先完成机制结论和样本量决定，保留小题池供明确目标的最终确认。将两个不同角色的题池合并前，需要将使用角色和样本数写入固定协议，保留同一任务权重和无标签驱动选择的样本清单。当前研究不启动该评价。", "",
    "独立确认建议保存一次固定时点的题级分层配对区间、各生成方向、纠错/破坏与实际成本。主候选和原Ridge构成一项主要检验，全局固定组合结果作预声明次级参照。任何候选版本扩大均需要在读评价奖励前写入多重比较安排。", "",
    "## 优势、风险与下一步", "",
    "- 优势。当前分析复用全部冻结预测，题级抽样与历史合法映射可追溯，新增LLM调用为0。", "- 风险。单次置乱覆盖有限，输入屏蔽会改变拟合后的特征分布，历史CV功效估计存在外推与选择偏差。", "- 统计结论。M4相对匹配置乱的内容收益区间跨零，可靠内容贡献尚未建立。输入屏蔽提示BBH部分题的特征贡献，当前证据强度为探索性。", "- 下一步。先结合选队集中度、并列干预及跨生成稳定非零差值的结果判断机制，再冻结未来独立评价协议。200道最终保留题对当前亚百分点点值的统计功效不足。", "",
    "## 日志补记", "",
    "首次数值运行之后，代码追加四份保留集角色文档的元数据与哈希，并明确输入屏蔽收据中的读入顺序。数值结果的comparisons和power与初版逐项一致。初版源码和数值JSON由精确移除新增元数据重建，文件标有RECONSTRUCTED。初版屏蔽收据的原始时间戳未保留，private/v1_reconstructed中的v2_metadata_snapshot收据来自实际第二次运行。置乱审计在屏蔽预测冻结前已读五个history480包，五包并集覆盖原600题。屏蔽规则只取既有W[0]，整体train600源文件在该次流程内于预测冻结后读取。这里记录应用层读入顺序。", "",
    "## 可复现入口", "",
    "`run_content_and_power.py --asset-root <原有MAB资产根目录> --output <新的输出目录>`随后运行`build_content_report.py`。`private/INPUT_MASK_FREEZE_RECEIPT.json`保存输入屏蔽冻结证据，`READ_ONLY_INPUT_HASHES.json`保存输入哈希，`CONTENT_AND_POWER_PAIRED_VALUES.npz`保存私有逐题配对记录。公开数值汇总在`M4_CONTENT_AND_POWER_RESULTS.json`。", "",
]
(HERE / "03_CONTENT_AND_INDEPENDENT_CONFIRMATION_PLAN.md").write_text("\n".join(lines), encoding="utf-8")
draft = dict(status="PROPOSED_NOT_EXECUTED", target="one frozen M4 vs original Ridge",
    true_optimal_tie_rule="original_M4_lexicographic",
    reverse_and_uniform_sensitivity="mechanism_controls_only_excluded_from_main_candidate_selection",
    alpha=.05, alternative="gain > 0", practical_gain_margin=.01,
    evaluation_units="independent original question, all3 generations attached",
    primary_weights="GSM8K50% BBH50%, within BBH question mean",
    evaluation_time="once at predeclared final sample count, no adaptive stopping",
    candidate_configuration="freeze one full-train600 M4 under deterministic historical-only selection rule before heldout rewards",
    required_frozen_components=["MiniLM/PCA/Scaler", "linear coefficients", "M4 regularization",
        "true-optimum tie rule", "test-time tie rule", "six-model and20-team order",
        "all question-level actions", "aggregation/scoring operators", "question identities and roles"],
    primary_question_counts_considered=[200, 400, 600],
    power_for_200_at_current_point=.1173106210323221,
    approximate_n_for_one_pp_80pct=2640,
    approval_gain_positive="one-sided95% lower bound >0",
    evidence_for_at_least_one_pp="same lower bound >.01",
    scope="planning only, no retained rewards read, no new language-model generation",
    roles={"fresh400":"one-time independent development evaluation after freezing",
           "sealed200":"final confirmation, power limitations explicitly retained"})
(HERE / "M4_INDEPENDENT_CONFIRMATION_DRAFT.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
