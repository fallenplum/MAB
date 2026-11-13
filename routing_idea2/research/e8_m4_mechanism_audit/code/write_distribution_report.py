"""Render descriptive tables from the sealed distribution audit."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
RESULT = json.loads((BASE / "public/M4_DISTRIBUTION_AND_TIE_RESULTS.json").read_text(encoding="utf-8"))
MEMBERS = ["Qwen3-8B", "Qwen2.5-Coder-7B-Instruct", "Granite-3.3-8b-instruct",
           "DeepHermes-3-Llama-3-8B-Preview", "DeepSeek-R1-Distill-Qwen-7B", "Cogito-v1-preview-llama-8B"]
with np.load(BASE / "private/EVALUATED_PAIRED_ACTIONS.npz", allow_pickle=False) as source:
    z = {k: source[k].copy() for k in source.files}
action = z["actions__M4_original"]
m4, ridge, zero = [z["reward__" + key] for key in ["M4_original", "M0_original", "constant_spo_lexicographic"]]
decomposition = {}
for key, mask in [("M4_selects_combo0", action == 0), ("M4_selects_other", action != 0)]:
    d = (m4 - ridge)[mask]
    decomposition[key] = dict(questions=int(mask.sum()), corrected_events=int((d > 0).sum()),
        destroyed_events=int((d < 0).sum()), net_events=int(d.sum()))
d = m4 - zero
decomposition["versus_always_combo0"] = dict(questions_departing=int((action != 0).sum()),
    corrected_events=int((d > 0).sum()), destroyed_events=int((d < 0).sum()), net_events=int(d.sum()))
decomposition["provenance"] = dict(status="POSTHOC_DESCRIPTIVE_DECOMPOSITION",
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    paired_values_sha256=hashlib.sha256((BASE / "private/EVALUATED_PAIRED_ACTIONS.npz").read_bytes()).hexdigest(),
    no_new_model_selection=True, independent_questions=600,
    grouping="already-frozen original M4 action equals combo0 versus other")
(BASE / "public/M4_CONCENTRATION_DECOMPOSITION.json").write_text(json.dumps(decomposition, indent=2) + "\n", encoding="utf-8")

rows = []
for combo, team in enumerate(RESULT["teams_zero_based"]):
    rows.append(dict(combo_id=combo, members=" + ".join(MEMBERS[m] for m in team),
        M0_count=RESULT["distributions"]["M0_original"]["all"]["counts"][combo],
        M4_lex_count=RESULT["distributions"]["M4_original"]["all"]["counts"][combo],
        M4_reverse_count=RESULT["distributions"]["full_spo_reverselex"]["all"]["counts"][combo]))
with (BASE / "public/M4_ALL20_SELECTION_COUNTS.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def pct(value):
    return f"{100 * value:.3f}%"


def pp(value):
    return f"{100 * value:+.3f}"


def interval(pair):
    return f"[{100 * pair['ci95'][0]:+.3f}, {100 * pair['ci95'][1]:+.3f}]"


text = """# M4选队分布与真实最优并列审计

## 研究问题与实验设置

复用E8原600题五折外层预测。每折只用480道历史题训练，120道外折题用于冻结后的开发评价。原始问题是统计单位，三个生成重复保留在同一问题簇中。所有数值都是已反复使用的开发资产上的探索性机制诊断。

本模块先冻结设计和源码SHA，再完成25个精确SPO+ epigraph QP。完整线性头保持17×20参数、原PCA16和标准化，正则乘子直接复用每折原M4配置。三个训练标签规则分别取真实均值最大者中的最小编号、最大编号及均匀权重。另以仅常数项的20参数模型实现两个无内容SPO+诊断。全部拟合使用Clarabel0.11.1及1e-9绝对、相对与可行性容差。

训练区全局固定组合和仅常数Ridge先验只使用该折历史平均成绩。它们的选择完全一致。全部干预预测和模型权重已在独立评价器读取旧train600评分前保存。新增物理LLM调用为0。

## 数值结果

### 原M4高度集中于组合0

组合ID从0开始，顺序为6选3的字典序。

| 方法 | 最常选组合及次数 | 前三组合占比 | 熵有效组合数 | 最高两分差中位数 |
|---|---|---:|---:|---:|
"""
for key, label in [("M0_original", "原Ridge"), ("M4_original", "原M4"),
                   ("full_spo_reverselex", "M4反向真实并列")]:
    value = RESULT["distributions"][key]["all"]
    text += f"| {label} | {value['top1_action']}，{value['top1_count']}/600 | {pct(value['top3_fraction'])} | {value['effective_combinations']:.3f} | {value['score_gap_median']:.6g} |\n"
text += """
原M4在GSM8K的283/300题与BBH的184/300题选择组合0。该组合由Qwen3-8B、Qwen2.5-Coder-7B-Instruct、Granite-3.3-8b-instruct组成。第二和第三常选组合分别是组合2的16题与组合6的14题。

组合2由Qwen3-8B、Qwen2.5-Coder-7B-Instruct、DeepSeek-R1-Distill-Qwen-7B组成。组合6由Qwen3-8B、Granite-3.3-8b-instruct、Cogito-v1-preview-llama-8B组成。

原M4最小最高两分差为7.16×10⁻⁶，600题均高于10⁻⁸，预测最高分没有完全并列。其集中选队在数值上具有明确分差。

### 真实最优标签的方向改变会改变集中目标

相同Clarabel后端下，最小编号真实并列规则完全复现原M4的600个选队。仅把训练标签规则改为最大编号，594/600题选队变化，505/600题改为选择组合19。组合19由DeepHermes-3-Llama-3-8B-Preview、DeepSeek-R1-Distill-Qwen-7B、Cogito-v1-preview-llama-8B组成。其最高两分差中位数0.08114，最小值1.78×10⁻⁴。

五个480题训练区中，真实最优动作多重并列占462—467题，20动作完全同均值占240—254题。最小编号规则把380—386题的目标动作指定为组合0，最大编号规则把390—395题指定为组合19。五折历史相互重叠，此处各折计数分别展示。

### 区分真实奖励并列和预测分数并列

若一道题20个真实均值全为常数a，最小编号规则选择j*，精确奖励最大化SPO+可化为

\[
\ell_{lex}=\max_j(2q_j-a)-2q_{j^*}+a
=2(\max_j q_j-q_{j^*}).
\]

任意使q_j*达到最高分的向量都令这项损失为0。均匀真实最优权重的适配变体则为

\[
\ell_{uniform}=2(\max_jq_j-\operatorname{mean}_j q_j),
\]

仅当20个分数全部相等时该项为0。上述判断针对单题数据损失，完整训练还包含共享参数和正则。最小编号标签会给完全无可区分奖励的样本指定方向，反向标签干预展示了这一方向对实际选队分布的影响。组合0和19是否真实更优仍需独立题目检验。

均匀真实并列适配的完整模型有397/600题最高两分差小于10⁻⁸，分差中位数6.78×10⁻¹²。仅常数均匀模型的600题分差也全小于10⁻⁸。严格argmax会保留求解器微小差异，结果不能作为稳定优胜策略解读。

### 质量与无内容先验比较

| 方法 | 准确率 | 相对原Ridge，百分点 | 配对95%CI，百分点 | 相对原M4，百分点及区间 |
|---|---:|---:|---:|---|
"""
for key, label in [("M0_original", "原Ridge"), ("M4_original", "原M4"),
                   ("train_global_fixed", "训练区全局均值先验"),
                   ("constant_spo_lexicographic", "仅常数SPO+，最小编号"),
                   ("full_spo_reverselex", "完整SPO+，最大编号"),
                   ("full_spo_uniform", "完整SPO+，均匀真实并列"),
                   ("constant_spo_uniform", "仅常数SPO+，均匀真实并列")]:
    value = RESULT["comparisons"][key]
    text += f"| {label} | {pct(value['accuracy'])} | {pp(value['difference'])} | {interval(value)} | {pp(value['versus_M4']['difference'])}，{interval(value['versus_M4'])} |\n"
text += """
本表为按原始问题、外折×任务分层、10,000次配对bootstrap的探索性区间，未进行新一轮确认性检验。M4相对训练区全局先验提高1.000个百分点，95%CI为[-0.667,+2.667]。当前题池没有建立其质量增益，点值接近也未经过等效性检验。反向真实并列虽然点值略高，研究中保留全部结果，未用它替换原M4。

### 543次选队变化与12个净正确事件

原M4相对原Ridge共有95个纠错与83个破坏，净增加12个题目×生成事件，即整体提高0.667个百分点。

| 按原M4已冻结选队分组 | 原始题数 | 相对Ridge纠错事件 | 相对Ridge破坏事件 | 净事件 |
|---|---:|---:|---:|---:|
| 选择组合0 | 467 | 76 | 52 | +24 |
| 选择其他组合 | 133 | 19 | 31 | -12 |

相对于始终选择组合0，M4在133道偏离题上有45个纠错与26个破坏，净增加19个事件。这个分解按冻结M4动作作描述，不能把分组后的正向点值视为独立确认。M4相对Ridge的总体优势主要来自其高频组合0区域，偏离区域相对Ridge的净事件为负。

## 统计结论与机制判断

选队集中和真实标签字典序影响的证据置信度为高。其依据包括467/600与505/600的集中分布、同后端改变真实并列规则后594题改变选队、原lex同后端600题完整复现，以及实质分差。

当前质量增益的证据置信度为低。原M4、反向规则和均匀适配相对Ridge的95%区间均包含0，原E8两项主比较Holm结论保持原样。

## 优势与局限风险

本轮优势是使用完全相同历史题、标签、线性容量和已选正则进行受控干预，且全部预测先于评价冻结。训练标签方向效应有可追溯的数学与数值证据。

主要风险是大量三重复经验均值并列、均匀变体的浮点不稳定、开发题反复暴露，以及五折历史重叠。25个轻量拟合的结果只支持当前冻结开发流程中的机制分析，三重复均值仍带有生成随机性。

## 下一步

把组合分布和并列规则作为待冻结协议中的必要审计项。结合跨生成稳定非零差值分析与内容负对照判断可学习信号，再规划独立确认题量。

## 复现与产物

执行run_distribution_audit.py的freeze、train、evaluate三个模式后运行write_distribution_report.py。新目录下首次执行会建立私有设计锁、25个模型、所有逐题预测、评分访问记录及配对事件矩阵。重复执行应使用新的输出目录并保持源码SHA。

主结果为M4_DISTRIBUTION_AND_TIE_RESULTS.json，完整20组合表为M4_ALL20_SELECTION_COUNTS.csv，纠错分解为M4_CONCENTRATION_DECOMPOSITION.json。原始模型和逐题结果保存在private目录。
"""
(BASE / "public/01_SELECTION_DISTRIBUTION_AND_TIE_AUDIT.md").write_text(text, encoding="utf-8")
