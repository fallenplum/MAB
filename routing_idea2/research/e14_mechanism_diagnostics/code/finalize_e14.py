"""Render measured E14 results and verify the append-only historical inputs."""
from __future__ import annotations

import csv
from pathlib import Path
import shutil

from e14_common import CODE, E13, PRIVATE, PUBLIC, check_protocol, read_json, sha256, sources, write_json


def text(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(value.strip() + "\n")


def main():
    check_protocol()
    result = read_json(PUBLIC / "E14_COMPLETE_RESULTS.json")
    freeze = read_json(PRIVATE / "ALL_DIAGNOSTIC_PREDICTIONS_FROZEN.json")
    fits = read_json(PRIVATE / "FIT_LEDGER.json")
    if result["confidence_intervals_computed"] or result["bootstrap_runs"]:
        raise ValueError("Future-experiment no-CI rule violated")
    a = [r for r in result["A"]["summary"] if r["scenario"] == "Pair" and r["budget"] == 120 and r["task"] == "all"]
    a_table = "\n".join(f"| {r['method']} | {r['predicted_accuracy_pct']:.4f} | {r['oracle_accuracy_pct']:.4f} | {r['oracle_minus_predicted_pp']:+.4f} |" for r in a)
    a_group = next(r for r in a if r["method"] == "Group")
    b = [r for r in result["B"]["cell_support_summary"] if r["scenario"] == "Pair" and r["budget"] == 120 and r["task"] == "bbh"]
    b_table = "\n".join(f"| {r['support_bin']} | {r['measurement_rows']} | {r['group_three_mean_abs_error']:.4f} | {r['group_individual_binary_reward_mse']:.4f} |" for r in b)
    bq = [r for r in result["B"]["selected_support_summary"] if r["scenario"] == "Pair" and r["budget"] == 120 and r["task"] == "bbh" and r["method"] == "GroupOracle"]
    bq_table = "\n".join(f"| {r['selected_support_bin']} | {r['measurement_rows']} | {r['seed_reward']*100:.4f} | {r['three_mean_empirical_selection_loss']:.4f} |" for r in bq)
    low = [r for r in bq if r["selected_support_bin"] in ("0", "1", "2")]
    total_events = sum(r["measurement_rows"] for r in bq)
    low_events = sum(r["measurement_rows"] for r in low)
    total_loss = sum(r["measurement_rows"]*r["three_mean_empirical_selection_loss"] for r in bq)
    low_loss = sum(r["measurement_rows"]*r["three_mean_empirical_selection_loss"] for r in low)
    low_share = low_loss/total_loss if total_loss else None
    c = [r for r in result["C"]["summary"] if r["anchor_main_gold"] == 120 and r["task"] == "all"]
    c_table = "\n".join(f"| {r['method']} | {r['main_accuracy_pct']:.4f} | {r['pair_accuracy_pct']:.4f} | {r['pair_minus_main_pp']:+.4f} |" for r in c)
    cost = next(r for r in result["C"]["cost_summary"] if r["anchor_main_gold"] == 120)
    fit_cpu = sum(r.get("cpu_s", 0) for r in fits)
    fit_wall = sum(r.get("wall_s", 0) for r in fits)
    n_files = len(read_json(PRIVATE / "INPUT_MANIFEST.json")["files"])
    text(PUBLIC / "01_GROUP_RECOGNITION.md", f"""
# A 组识别误差

## 研究问题

固定组能力表和残差参数，仅替换评价题的组索引，量化组分类错误带来的质量变化。

## 实验设置与对照

复用E13五外折、600原题、三个logger和三个生成重复、两个反馈场景及六个预算。表内所有参数来自对应H480合法反馈。预测组来自原输入分类器。真实组来自评价题元数据，整个真实组结果均为Oracle机制诊断。

Group使用同一份own__C。GroupResidual同时固定原残差权重。SaveRouter使用原已存组能力表及残差项。TaskFixed/SubtaskFixed使用各自原经验表作为补充参照。每个快照先恢复原Group分数与原动作后实施干预。

## 数值结果

600题组分类准确率为{result['A']['group_accuracy_pct']:.4f}%。主展示为Pair、Gold120，单位为百分比及百分点。

| 固定学习器 | 预测组选队准确率 | 真实组Oracle准确率 | Oracle减预测组 |
|---|---:|---:|---:|
{a_table}

## 统计结论

本轮只报告配对点差、纠错破坏以及任务、外折、种子方向。置信区间和bootstrap执行数均为0。Group在主展示上的组索引替换效应为{a_group['oracle_minus_predicted_pp']:+.4f}个百分点。

## 失败原因与风险

Oracle提高或降低服务质量取决于原表估计误差。分类准确率不能单独说明组错误造成了多少选队损失。其他限制包括开发资产已暴露、表本身支持稀少，以及单次生成奖励噪声。

## 下一步

结合B的留出误差判断分类和能力表各自的可见问题，后续决定另列新协议。
""")
    text(PUBLIC / "02_CAPABILITY_ESTIMATION.md", f"""
# B 能力估计误差

## 研究问题

同一个反馈掩码下，支持稀少的组–组合单元是否表现出更高的留出误差或贡献更多经验选队损失。

## 实验设置与对照

固定原own__C与残差系数。只在不属于历史监督区的E120评价，能力表按真实组查表以隔离组分类错误，标记Oracle机制诊断。单元支持为单一生成轨迹中实际购买该组、该组合反馈的不同历史原题数。三个生成和logger重复分别保留。

目标分别为同seed二值奖励、E组内同seed均值、E组内三生成均值。单元误差按留出组–动作均值计算，逐题二值MSE另列。空留出组目标为null。经验最大值及经验选择损失均基于有限已存结果。

## 数值结果

Pair、Gold120、BBH，单元等权汇总。measurement_rows为重复测量单元数。

| 支持数 | 重复测量单元行 | 对三生成留出均值的MAE | 逐题二值MSE |
|---|---:|---:|---:|
{b_table}

同设置下GroupOracle实际选中单元的分解。

| 选中单元支持数 | 重复测量题事件 | 同seed准确率% | 三生成经验选择损失 |
|---|---:|---:|---:|
{bq_table}

选中支持≤2的事件为{low_events}/{total_events}，占{100*low_events/total_events:.4f}%。这些事件贡献三生成经验选择损失的{100*low_share:.4f}%。

## 统计结论

将损失占比与事件占比一起判断是否集中于低支持单元。关联结果不提供支持数量的因果效应。留出目标包含题目差异和有限生成噪声。

## 失败原因与风险

BBH每组留出题数有限，高支持主要来自不同任务分布。所有组–动作单元、选中单元与未选中单元的覆盖不同。三生成均值仍存在噪声。公开表按任务分列，私有逐题表保留完整匹配关系。

## 下一步

将低支持占比和误差、损失一起用于确定瓶颈，避免凭训练自评解释泛化。
""")
    text(PUBLIC / "03_TOKEN_MATCHED_FEEDBACK.md", f"""
# C 反馈信息效率

## 研究问题

在同一实际Token上限下，多购买一个邻居组合与评测更多Main题相比是否带来质量收益。

## 实验设置与对照

使用E13已执行完整题包的原始顺序。每个Main原检查点的实际总Token是共同上限，Pair取该上限内最长的完整已执行前缀。费用含logger服务和反馈增量，成员共享按原账本计费。每条轨迹保留gold、member、aggregate、scoring、输入输出Token及余额。没有准确率插值。额外包的完整成本也保留在匹配记录中。

原Main快照全部复用。Pair只在新增匹配前缀上，用原E13八种固定学习器、相同参数及相同预处理重算。所有预测先封存，随后读取E120评分。该过程属于回顾性成本匹配诊断。

## 数值结果

主锚点Main Gold120，跨45条fold×logger×generation轨迹平均。

- Main实际Token为{cost['main_actual_tokens']:.4f}。
- Pair实际Token为{cost['pair_actual_tokens']:.4f}，平均获得{cost['pair_questions']:.4f}道不同题的双组合反馈。
- 平均未用余额为{cost['remaining_tokens']:.4f}Token，余额比为{100*cost['remaining_token_fraction']:.4f}%。

| 固定学习器 | Main准确率% | Token匹配Pair准确率% | Pair减Main百分点 |
|---|---:|---:|---:|
{c_table}

## 统计结论

主表比较相同Token上限下的实测质量，各自实际花费逐项保留。Main/Pair题包不可分割，所以实际Token可有余额差，其他资源也不同时相等。六个锚点、各任务、种子和外折结果全部公开。

## 失败原因与风险

Token匹配读取的是已执行账本。它不证明未知未来费用下的在线硬预算策略。费用信息只决定前缀长度，评价奖励不参与选择。Pair减少了可覆盖的独立历史问题，邻居反馈是否值得取决于固定学习器与预算区域。

## 下一步

结合全部锚点的方向决定是否有证据支持额外邻居反馈；下一轮学习或采集改变需单独立协议。
""")
    text(PUBLIC / "EXPERIMENT_SUMMARY.md", f"""
# E14 三个受控机制诊断

## 主要结果

### A 组识别误差

原分类器在600题上的组识别准确率为{result['A']['group_accuracy_pct']:.4f}%。Pair Gold120下，固定Group能力表改用真实组后，准确率从{a_group['predicted_accuracy_pct']:.4f}%变为{a_group['oracle_accuracy_pct']:.4f}%，点差{a_group['oracle_minus_predicted_pp']:+.4f}个百分点。真实组仅用于Oracle机制分析。

{a_table}

### B 能力估计误差

Pair Gold120、BBH的GroupOracle选队事件中，支持≤2占{100*low_events/total_events:.4f}%，贡献{100*low_share:.4f}%三生成经验选择损失。详细误差按任务和支持分箱单列。支持稀少与误差/损失的关联受组难度和样本结构影响。

### C 反馈信息效率

主锚点同实际Token上限，Pair平均评测{cost['pair_questions']:.4f}道双组合反馈题，Main评测120道单组合反馈题。

| 学习器 | Main准确率% | Pair准确率% | Pair减Main百分点 |
|---|---:|---:|---:|
{c_table}

## 优势

同一组能力表、反馈掩码和原学习器参数可追溯复用；B完全使用合法外折留出奖励；C使用已执行总账中的实际Token和成员共享；正负结果、六个锚点和种子方向均保留。

## 劣势与风险

全部证据来自反复使用的600题开发资产。三个生成和logger是重复测量。真实组带有特权。小子任务留出均值不稳定。相同Token上限仍有完整题包造成的实际花费差，C也属于回顾性前缀对比。结果不提供独立确认或因果识别。

## 实际执行与成本

复用540个E13快照。为匹配已执行反馈前缀，调用原学习器训练{len(fits)}次，另有2次真实工件数值恢复资格拟合。每次包含原自有学习器和原SaveRouter；自有Ridge共享矩阵求解三类系数。前缀拟合CPU累计{fit_cpu:.4f}秒，拟合墙钟累计{fit_wall:.4f}秒。私有FIT_LEDGER和公开QUALIFICATION包含各次细项。新增语言模型、API和标准答案获取调用均为0。置信区间和bootstrap计算均为0。

## 置信度与下一步

工件一致性和账本核验的置信度为高，范围限于可重复恢复。机制解释属于探索性开发证据，置信度为中。跨数据集或新题的外推置信度为低。优先依据A、B、C的全部方向定位瓶颈，再另行决定后续实验。
""")
    text(PUBLIC / "README.md", """
# E14 受控机制诊断

- [研究结论](EXPERIMENT_SUMMARY.md)
- [A 组识别](01_GROUP_RECOGNITION.md)
- [B 能力估计](02_CAPABILITY_ESTIMATION.md)
- [C 实际Token匹配](03_TOKEN_MATCHED_FEEDBACK.md)
- [完整结果](E14_COMPLETE_RESULTS.json)
- [复现说明](REPRODUCE.md)

本阶段仅使用E13既有执行结果与冻结学习器。真实组结果为Oracle机制诊断。新实验遵循不计算置信区间的项目约定。
""")
    text(PUBLIC / "REPRODUCE.md", r"""
# 复现

## 依赖与私有工件

Python 3.11、NumPy、SciPy、scikit-learn、psutil，使用原MAB环境。需要E13完整private/worlds、inputs、冻结收据和public/code，以及原train600输入/输出缓存、原SaveRouter固定源码。上述私有数据不在公开仓库内。

设MAB_ASSET_ROOT指向语言模型组合路由根目录。MAB_E14_ROOT可以指向一个全新输出目录。MAB_E13_CODE可显式指向原E13源码目录。模块从E13复用原预测器与SaveRouter，需同时保留两个阶段源码。

## 执行顺序

```powershell
python -B -X utf8 code/run_e14.py prepare
python -B -X utf8 code/run_e14.py freeze
python -B -X utf8 code/evaluate_e14.py
python -B -X utf8 code/finalize_e14.py
```

prepare执行必要资格并锁定输入与源码；freeze仅使用原已购买H区反馈和实际执行账本，生成A干预动作及C原学习器前缀快照。全部动作封存之后才允许评价器读取原train600真实奖励。B仅在各折E120上计算误差。阶段文件采用不可覆盖写入。

## 成本与输出

private保留逐题诊断、原题索引、原能力表来源、重新拟合权重、反馈前缀、完整账本和所有冻结收据。public只有脱敏汇总及复现源码。C_TOKEN_MATCHES保留完整调用资源、实际Token与离散剩余，历史已执行计量成本与E14新增CPU成本分别记录。

置信区间、bootstrap、假设检验和新增物理LLM/API调用执行数均为0。fresh400和sealed200不读取。原题级相关性按600个原始问题解释，重复日志及生成seed不增加独立题量。
""")
    for path in sources():
        target = PUBLIC / "code" / path.relative_to(CODE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    public_files = [dict(path=p.relative_to(PUBLIC).as_posix(), sha256=sha256(p), bytes=p.stat().st_size)
                    for p in sorted(PUBLIC.rglob("*")) if p.is_file()]
    write_json(PUBLIC / "PUBLIC_DELIVERY_MANIFEST.json", dict(status="SEALED_FOR_PUBLICATION", files=public_files))
    check_protocol()
    write_json(PRIVATE / "FINAL_VERIFICATION.json", dict(status="PASS", protected_historical_files=n_files,
        historical_hashes_unchanged=True, public_file_count=len(public_files)+1,
        input_manifest_sha256=sha256(PRIVATE / "INPUT_MANIFEST.json"),
        predictions_frozen_before_E14_evaluation=True, confidence_intervals=0, bootstrap_runs=0,
        new_physical_llm_calls=0, fresh400_reads=0, sealed200_reads=0,
        result_sha256=sha256(PUBLIC / "E14_COMPLETE_RESULTS.json")))
    print("E14_REPORTS_AND_VERIFICATION_COMPLETE", len(public_files)+1, flush=True)


if __name__ == "__main__":
    main()
