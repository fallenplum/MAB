"""Render the three bounded E16 diagnostics and seal a public release."""
from __future__ import annotations

import shutil
from e16_common import CODE,PRIVATE,PUBLIC,REPS,read_json,sha256,source_files,verify_protocol,write_json


def write(path,text):
    with path.open("x",encoding="utf-8") as f:f.write(text.strip()+"\n")


def main():
    verify_protocol();r=read_json(PUBLIC/"E16_COMPLETE_RESULTS.json")
    variance="\n".join(f"| {v['representation']} / {v['scope']} | {v['mean_observed_variance']:.6f} | {v['mean_noise_of_three_mean']:.6f} | {v['mean_signal_estimate']:.6f} | {v['negative_signal_estimates']} |" for v in r["A"])
    btable="\n".join(f"| {v['n']} | {v['L']} | {v['accuracy_pct']:.4f} |" for v in r["B"]["quality"] if v["package_budget"]==480 and v["task"]=="all")
    ctable="\n".join(f"| {v['representation']} | {v['accuracy_pct']:.4f} |" for v in r["C"]["quality"] if v["task"]=="all")
    bcomp=next(v for v in r["B"]["comparisons"] if v["package_budget"]==480 and v["comparison"]=="L1-L3")
    cb="\n".join(f"| {v['comparison']} | {v['difference_pp']:+.4f} | {v['generation_direction']} |" for v in r["C"]["comparisons"])
    costs="\n".join(f"| {v['n']} | {v['L']} | {v['gold']:.0f} | {v['member_calls']:.0f} | {v['aggregator_calls']:.0f} | {v['tokens']:.2f} |" for v in r["B"]["costs"] if v["package_budget"]==480)
    a_pair=next(v for v in r["A"] if v["representation"]=="pairs" and v["scope"]=="bbh_subtask_within")
    write(PUBLIC/"01_VARIANCE_AND_STABILITY.md",f"""
# 生成内噪声与题间组合优势

## 研究问题

稳定组合差异是否存在可观察的题间变动，生成重复噪声占多大。

## 实验设置与对照

原train600、20动作、三个生成。分别计算原始动作、每题每次生成去除20动作共同偏移的中心化动作，以及190个实际配对差。配对差保留同次生成的跨动作相关性。

## 方差恒等式及解释条件

对配对差D，设题目条件均值为mu，生成内条件方差为sigma²。当原题独立、生成重复条件不相关且有共同条件均值时，`Var(mean_L D)=Var(mu)+E(sigma²)/L`。单题样本方差对sigma²无偏，因此题间均值样本方差减题内样本方差均值/L给出矩估计。

任务/子任务内用组内SSE除以总自由度，并减相同自由度权重的噪声/L。单题组自由度0不进入该项。该解释需要条件独立或条件不相关假设，单纯交换性不充分。三个seed编号自身不能证明这些假设。

## 数值结果

数值为相应动作/配对维度的均值。负矩估计保留原值。

| 表示与范围 | 观察题间方差 | 三生成均值噪声修正 | 信号矩估计 | 负估计维数 |
|---|---:|---:|---:|---:|
{variance}

## 统计结论与风险

BBH子任务内190配对差的平均信号估计为{a_pair['mean_signal_estimate']:.6f}，噪声修正为{a_pair['mean_noise_of_three_mean']:.6f}。经验正值不等于已准确识别真实条件期望，更不等于已证明这些差异能从文本预测。

跨seed严格比较同向、反转与留出seed平局均在A_CROSS_SEED_DIRECTION.csv完整保留。高同向率需要同时检查平局数量。未计算区间或假设检验。

## 下一步

与独立题/重复预算及输入表征的泛化结果共同解释。
""")
    write(PUBLIC/"02_QUESTIONS_VS_REPEATS.md",f"""
# 等题目–生成包预算的分配

## 研究问题

在相同完整组合评分包数量下，增加不同问题或增加同题生成重复哪个更有效。

## 实验设置与对照

每个包为一道题、一个生成、全部20组合的真实最终奖励，需6次成员输出、20次组合聚合和20次评分。同题多个生成共享一次gold。主预算nL=480，240为预声明曲线。原五外折，H480内按logger17/29/43冻结任务平衡嵌套题序。L1用三个单seed，L2用三个seed对，L3用全三个，等价重复拟合复用。原H480 PCA16输入预处理固定。

Ridge逐生成目标为平方误差求和加100倍全系数范数，常数项受罚。与同题均值标签训练等价时alpha=100/L。固定nL保证同数据项归一化下有效正则一致。

## 数值结果

主预算准确率百分比。

| 独立训练题数 | 每题生成次数 | 外折准确率 |
|---:|---:|---:|
{btable}

L1−L3点差为{bcomp['difference_pp']:+.4f}个百分点，评价生成净正确事件方向为{bcomp['generation_direction']}。

## 完整反馈费用

费用为每条fold/logger/训练seed选择轨迹均值。

| 题数 | 生成次数 | Gold | 成员调用 | 聚合调用 | 实际Token |
|---:|---:|---:|---:|---:|---:|
{costs}

## 统计结论与风险

相同nL保持名义完整组合反馈数量一致，Gold和实际Token分别变化。结论只对应这一反馈口径。所有评价发生在未参与当前模型拟合的外折问题上。600道题已反复用于开发，训练seed选择与评价seed分别记录，重复记录不增加独立题量。

## 下一步

依据B_COST_SUMMARY与质量一起判断实际补数据方向，后续新增物理调用另立预算。
""")
    selections=read_json(PUBLIC/"C_MODEL_SELECTION.json")
    degenerate=sum(v["inner_grid_actions_identical"] for v in selections)
    write(PUBLIC/"03_REPRESENTATION_ABLATION.md",f"""
# 压缩与互补输入表征

## 研究问题

MiniLM压缩维度和Qwen输入信息是否限制现有线性价值预测器。

## 实验设置与对照

固定同一直接20输出Ridge、H480三生成均值标签、五个外折和原E8合法三内折。每个训练区单独fit PCA与StandardScaler。三档alpha为(100/3)乘0.1/1/10，仅按内折实际路由正确率选，并列取更强正则。

MiniLM16含常数项为340参数，其余64维配置为1300参数。每PC为单位方差，融合使用MiniLM32+Qwen32直接拼接，总方差64，与单路64维一致。P16→P64同时改变维度和容量，该对照本身无法单独排除容量影响。

主Qwen向量为与MiniLM可见文本匹配的冻结输入表征，问题/ID逐项600/600对齐。QwenFull融合是预声明敏感性，只有6题的可见内容不同。QwenMatched64对照用于检查融合收益是否超出Qwen单路。

## 数值结果

外折真实聚合准确率百分比。

| 表征 | 准确率 |
|---|---:|
{ctable}

| 预声明比较 | 点差百分点 | 三评价生成净事件 |
|---|---:|---|
{cb}

## 优化和误差诊断

25个外层配置中{degenerate}个在每个内折的三个正则候选都给出完全相同动作。单路PCA标准化后的设计正交，含常数项受罚时正则可只产生共同正比例缩放。三档数值候选不自动构成三种有效选队自由度。

绝对MSE、中心化MSE和经验选择损失按折保留。后者针对三次经验均值。单独的误差下降不替代最终选队质量。

## 统计结论与风险

主比较为MiniLM64−MiniLM16及FusionMatched64−MiniLM64。完整文本融合单列，所有预定配置均报告。当前证据属于开发交叉验证，历史重叠训练和反复数据使用限制独立外推。

## 下一步

结合融合对两个单路64维的差值、生成方向和费用判断是否有互补信息线索。
""")
    write(PUBLIC/"EXPERIMENT_SUMMARY.md",f"""
# E16 三项受控诊断

## 方差分解

BBH子任务内配对差平均信号矩估计{a_pair['mean_signal_estimate']:.6f}，三生成均值的噪声修正{a_pair['mean_noise_of_three_mean']:.6f}。负分量保留，解释依赖原题独立与条件不相关生成假设。它与可从文本泛化的路由价值分别判断。

## 独立问题与重复生成

主预算nL=480，准确率百分比。

| n | L | 准确率 |
|---:|---:|---:|
{btable}

L1−L3为{bcomp['difference_pp']:+.4f}个百分点。相同nL的Gold与实际Token不同，完整费用另列。

## 表征对照

| 表征 | 准确率 |
|---|---:|
{ctable}

全部主比较、Qwen单路与Full敏感性均保留，原P16与64维模型容量差明确记录。

## 优势、风险与置信度

优势是同一真实评分矩阵、原题级分折、所有训练区内重新fit预处理、重复损失正则换算可核对，以及费用完整分账。风险是三个生成重复有限、独立性假设未由seed编号证明、数据已多轮暴露、维数变化影响容量与正则几何。工程工件一致性置信度为高，独立新题外推置信度为低。

## 实际执行与后续

B实际拟合{r['B_actual_fits']}个预测头，C内外层实际拟合{r['C_actual_fits']}个预测头，另5次原Ridge恢复资格。方差、Token和原编码投入完整保留。新增生成调用、编码器前向、置信区间和bootstrap均为0。准备/冻结/评价耗时单列。后续新增数据或算法另立协议。
""")
    write(PUBLIC/"README.md","""
# E16 噪声、重复预算和表征诊断

- [实验总结](EXPERIMENT_SUMMARY.md)
- [方差与跨生成稳定性](01_VARIANCE_AND_STABILITY.md)
- [题目数与生成次数](02_QUESTIONS_VS_REPEATS.md)
- [PCA及输入融合](03_REPRESENTATION_ABLATION.md)
- [全部结果](E16_COMPLETE_RESULTS.json)
- [计算及历史编码成本](COMPUTE_AND_HISTORY_COST.json)
- [复现](REPRODUCE.md)
""")
    write(PUBLIC/"REPRODUCE.md",r"""
# E16复现

原MAB Python3.11、NumPy、SciPy、scikit-learn和psutil。需要原train600输入与20组合×3生成输出，E13原H/E及PCA16状态，E8 M0_tuned各折FIT_LOG与model_state，E10 PREFILL_FEATURES_FROZEN.npz、原输入来源清单和编码冻结收据。

MAB_ASSET_ROOT指定原资产根，MAB_E16_ROOT可指定一个全新输出目录。复用E14 e14_common.py的不可覆盖JSON与哈希函数。

```powershell
python -B -X utf8 code/run_e16.py prepare
python -B -X utf8 code/run_e16.py freeze
python -B -X utf8 code/evaluate_e16.py
python -B -X utf8 code/finalize_e16.py
```

prepare执行所有源码AST与单元测试、Qwen行身份审计、原Ridge恢复，并封存内外层分折、嵌套样本顺序、seed子集、预算和模型选择规则。freeze仅使用各自合法H奖励，保存全部B/C模型与动作后再由evaluate读取E奖励及完整A诊断矩阵。

私有工件保留逐题预测、真实评分、采样题目身份、向量、权重及内折日志。公开文件为脱敏统计和源码。历史训练反馈和输入编码成本、当前缓存复用CPU成本分别记录。三个生成是题内重复，600独立题不扩大计数。

置信区间和bootstrap执行数为0。新语言模型生成和编码前向为0。validation200、fresh400和sealed200不读取。
""")
    for p in source_files():
        target=PUBLIC/"code"/p.relative_to(CODE);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    verify_protocol()
    files=[dict(path=p.relative_to(PUBLIC).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(PUBLIC.rglob("*")) if p.is_file()]
    write_json(PUBLIC/"PUBLIC_DELIVERY_MANIFEST.json",dict(status="SEALED_FOR_PUBLICATION",files=files))
    write_json(PRIVATE/"FINAL_VERIFICATION.json",dict(status="PASS",public_files=len(files)+1,
        original_source_hashes_verified=True,predictions_frozen_before_outer_evaluation=True,
        confidence_intervals=0,bootstrap_runs=0,new_generation_calls=0,new_encoder_forwards=0,
        result_sha256=sha256(PUBLIC/"E16_COMPLETE_RESULTS.json")))
    print("E16_REPORTS_COMPLETE",len(files)+1,flush=True)


if __name__=="__main__":main()
