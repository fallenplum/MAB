"""Create E15 reports from the entire locked grid, including failed gates."""
from __future__ import annotations

import shutil
from e15_common import CAPS,CODE,METHODS,PRIMARY_CAP,PRIVATE,PUBLIC,STRUCTURES,read_json,sha256,sources,verify_protocol,write_json


def write(path,content):
    with path.open("x",encoding="utf-8") as f:
        f.write(content.strip()+"\n")


def main():
    verify_protocol()
    r=read_json(PUBLIC/"E15_COMPLETE_RESULTS.json")
    primary=[v for v in r["quality"] if v["design"]=="TokenMatched" and v["token_cap"]==PRIMARY_CAP and v["task"]=="all"]
    table="\n".join("| "+m+" | "+" | ".join(f"{next(v['accuracy_pct'] for v in primary if v['method']==m and v['structure']==s):.4f}" for s in STRUCTURES)+" |" for m in METHODS)
    header="| 学习器 | Main | NeighborPair | RandomPair |\n|---|---:|---:|---:|\n"
    comps=[v for v in r["comparisons"] if v["token_cap"]==PRIMARY_CAP and (v["method"]=="Group" or v["method"]=="Group-minus-RawCentered")]
    comp_table="\n".join(f"| {v['method']} {v['comparison']} | {v['difference_pp']:+.4f} | {v['positive_generation_count']}/3 | {v['positive_fold_count']}/5 |" for v in comps)
    curve=[]
    for cap in CAPS:
        vals=[next(v for v in r["comparisons"] if v["token_cap"]==cap and v["method"]=="Group" and v["comparison"]==name) for name in ("NeighborPair-Main","NeighborPair-RandomPair")]
        curve.append(f"| {cap:,} | {vals[0]['difference_pp']:+.4f} | {vals[1]['difference_pp']:+.4f} |")
    costs=[v for v in r["cost_summary"] if v["design"]=="TokenMatched" and v["token_cap"]==PRIMARY_CAP]
    cost_table="\n".join(f"| {v['structure']} | {v['questions']:.4f} | {v['member_calls']:.4f} | {v['aggregator_calls']:.4f} | {v['tokens']:.4f} | {100*v['remaining_fraction']:.4f}% |" for v in costs)
    decomp=[v for v in r["coverage_decomposition"] if v["token_cap"]==PRIMARY_CAP]
    decomposition_table="\n".join(f"| {v['method']} {v['alternative']} | {v['total_quality_difference_pp']:+.4f} | {v['extra_combination_effect_at_common_questions_pp']:+.4f} | {v['alternative_added_coverage_set_pp']:+.4f} | {v['main_added_coverage_set_pp']:+.4f} |" for v in decomp)
    gate=r["primary_gate"]
    gates="\n".join(f"- {name}，{'通过' if passed else '未通过'}。" for name,passed in gate["conditions"].items())
    fits=read_json(PRIVATE/"FIT_LEDGER.json")
    fitcpu=sum(v["cpu_s"] for v in fits)
    fitwall=sum(v["wall_s"] for v in fits)
    repair=read_json(PRIVATE/"PREFLIGHT_AMENDMENT.json") if (PRIVATE/"PREFLIGHT_AMENDMENT.json").exists() else None
    repair_note=(f"此前v1预检的RandomPair gold键标签与原V2包身份不一致。已在读取E15评价成绩前归档并全量重建v2。旧预检{repair['previous_model_packets']}个模型包、CPU{repair['previous_pilot']['cpu_s']:.4f}秒、墙钟{repair['previous_pilot']['wall_s']:.4f}秒作为额外工程成本保留，未纳入结果。" if repair else "无协议修订预检成本。")
    write(PUBLIC/"01_LOCKED_DESIGN_AND_COST.md",f"""
# E15 固定协议与实际费用

## 研究问题

同一绝对Token上限下，反馈组合结构对固定学习器的收益是否存在可重复交互。

## 实验设置与对照

六个冻结模型、K=3、20组合、三个生成seed42/137/2024、原MiniLM PCA16、原五折H480/E120。logger17/29/43、原题遍历顺序、Main及邻居动作全部复用。三个学习器为原RawCentered、Group和固定作者实现SaveRouter。每条反馈包由三方法共同使用，同一问题的多动作共享一次gold及成员执行。

唯一主预算为每条生成轨迹250000实际输入加输出Token。其余预算125000/500000/750000在评价前锁定，全部报告。完整题包按原顺序取不超过上限的最长前缀，实际花费和余额保留。

RandomPair从其余19种动作均匀选择，包含9种邻居、9种只共享一个成员的组合和1种不相交组合。新账本分别按4/5/6成员并集计费，每题2聚合、2评分、1gold。随机计划在访问奖励/费用前封存，并跨生成重复共用。

## 数值结果

主预算45条world的费用均值。

| 结构 | 独立反馈题数/Gold | 成员调用 | 聚合调用 | 实际Token | 未用比例 |
|---|---:|---:|---:|---:|---:|
{cost_table}

## 统计结论与风险

同上限比较保留完整题包造成的余额差。全部计量来自已执行原始缓存，新增物理LLM调用为0。完整计划池与所选前缀成本分别保存。该设计属于回顾性资源控制，未知未来费用的在线硬预算保证需要独立分析。

## 下一步

依照锁定主比较及门槛结束本轮，完整曲线只作预算稳健性描述。
""")
    write(PUBLIC/"02_INTERACTION_RESULTS.md",f"""
# 反馈结构与学习器交互

## 研究问题与设置

固定算法、输入特征、更新参数、合法反馈轨迹，只改变反馈组合结构。主指标为独立原题外折上的真实最终聚合正确率。

## 主预算数值结果

250000Token上限，准确率单位为百分比。

{header}{table}

## 预定配对比较

差值单位为百分点。正方向由整数净正确事件判断。

| 比较 | 点差 | 正向生成seed | 正向外折 |
|---|---:|---:|---:|
{comp_table}

## 全部预算曲线

| Token上限 | Group 邻居−Main | Group 邻居−随机第二组合 |
|---|---:|---:|
{chr(10).join(curve)}

## 统计结论

全部600原题是独立单位，logger与生成重复保留在同题内。置信区间、bootstrap及假设检验运行数均为0。上述比较属于反复暴露开发资产上的描述性证据。

## 风险与下一步

主机制交互与完整系统价值分别判断。主预算Group-Neighbor相对其他两学习器所有结构的最强点值差为{gate['group_neighbor_minus_strongest_other_learner_pp']:+.4f}个百分点。未将某个有利曲线点替换为主预算。
""")
    btable=[]
    for v in r["capability_error_summary"]:
        if v["token_cap"]==PRIMARY_CAP and v["task"]=="bbh":
            btable.append(f"| {v['design']} {v['structure']} {v['method']} | {v['table_mae_three_mean']:.5f} | {v['centered_table_mse_three_mean']:.5f} |")
    write(PUBLIC/"03_CAPABILITY_AND_COVERAGE.md",f"""
# 能力误差与独立题覆盖

## 研究问题

观察更多动作与覆盖更多独立历史题，在同一学习器中分别产生什么变化。

## 实验设置与对照

每个Token上限定义N为三结构可覆盖题数的最小值，CommonCoverage使用同一个原始题目前缀。该控制匹配题目身份和数量，执行费用逐项单列。

能力误差仅用E120留出奖励。Group使用冻结C，SaveRouter单列原生组能力表，RawCentered使用同一留出组内问题得分的均值。后两者仍分别保留完整服务得分MSE。真实组只用于诊断，服务选队输入为原预测组。三生成均值是经验目标。

## 价值差分解

邻居或随机第二组合记为A，Main记为M。同一E题集上逐事件严格核验

`Afull-Mfull=(Acommon-Mcommon)+(Afull-Acommon)-(Mfull-Mcommon)`

主预算点差分解，单位为百分点。

| 学习器及结构 | 总差 | 同题覆盖额外组合效应 | 该结构新增覆盖集合效应 | Main新增覆盖集合效应 |
|---|---:|---:|---:|---:|
{decomposition_table}

## 能力估计误差

主预算BBH组–动作均值误差，组等权汇总。

| 控制、结构及学习器 | 留出均值MAE | 中心化能力MSE |
|---|---:|---:|
{chr(10).join(btable)}

## 统计结论与风险

SUPPORT_AND_COVERAGE记录独立原题、标签数、组–动作支持、同题双动作分歧率。CAPABILITY_QUALITY_LINK使用同组、同E题的配对误差变化和选队变化，相关性只作描述。

新增覆盖集合效应包含具体新增题目的任务和难度构成。Group仅使用边际sum/count，结果不能作为已实现同题相关性建模的证据。低支持分组与任务混淆继续保留。CommonCoverage和TokenMatched的费用权限分别记录。

## 下一步

依据同题覆盖分解及完整Token曲线判断哪个环节值得独立确认。
""")
    write(PUBLIC/"EXPERIMENT_SUMMARY.md",f"""
# E15 固定Token预算反馈结构与学习器交互

## 主结果

唯一主预算250000Token，准确率百分比。

{header}{table}

## 多预算结果

| Token上限 | Group 邻居−Main百分点 | Group 邻居−随机第二组合百分点 |
|---|---:|---:|
{chr(10).join(curve)}

## 锁定门槛

{gates}

总体门槛{'通过' if gate['all_passed'] else '未通过'}。执行决定为`{r['next_stage']}`。该门槛是描述性实际收益筛查，缺少错误率保证。

## 优势与风险

优势是三学习器共享完全相同的合法反馈，固定预算独立于E15结果，成员共享和聚合逐项计费，同时提供同题覆盖分解。风险是原600题已反复研究、三生成与logger相关、实际Token存在整包余量，以及新增覆盖集合还改变任务构成。机制交互与超越最强其他学习器分别报告。

## 计算与交付

唯一模型包{r['unique_learner_packets']}，实际新拟合{r['newly_fitted_packets']}包，旧缓存复用{r['legacy_reused_packets']}包。每包调用原共享Ridge及原SaveRouter，所需RawCentered与Group复用原共同计算。拟合/缓存核对CPU累计{fitcpu:.4f}秒，墙钟累计{fitwall:.4f}秒。

语言模型调用、置信区间、bootstrap和假设检验执行数均为0。评价角色为开发回放。私有逐题决策、权重、原题标识及反馈账本保存在private。

{repair_note}

## 置信度与后续决定

工件和计费恢复的置信度为高。机制结论限于锁定开发设置，外推置信度为低，依据是缺少独立新题确认。本轮完整结束后，按预设门槛执行停止或仅规划确认。
""")
    write(PUBLIC/"README.md","""
# E15 固定实际Token预算交互验证

- [实验总结](EXPERIMENT_SUMMARY.md)
- [冻结设计与实际成本](01_LOCKED_DESIGN_AND_COST.md)
- [交互及全部预算](02_INTERACTION_RESULTS.md)
- [能力误差与覆盖分解](03_CAPABILITY_AND_COVERAGE.md)
- [完整结果](E15_COMPLETE_RESULTS.json)
- [复现](REPRODUCE.md)
""")
    write(PUBLIC/"REPRODUCE.md",r"""
# E15复现

## 环境和私有资产

原MAB Python3.11、NumPy、SciPy、scikit-learn、psutil环境。需要E13完整private/worlds和inputs、原train600缓存及E3 assets；E14 public/code和private/prefixes用于逐项核对缓存复用；SaveRouter固定作者源码沿用原路径。

MAB_ASSET_ROOT设置实验资产根目录，MAB_E15_ROOT可指定全新输出目录，MAB_E14_CODE指定E14源码目录。私有权重、题目索引、反馈评分与原矩阵不包含于公开交付。

## 入口

```powershell
python -B -X utf8 code/run_e15.py prepare
python -B -X utf8 code/run_e15.py smoke
python -B -X utf8 code/run_e15.py freeze
python -B -X utf8 code/evaluate_e15.py
python -B -X utf8 code/finalize_e15.py
```

prepare先封存四个绝对预算、唯一主点、三项主要比较、停止条件、原输入来源和盲随机动作。smoke执行首条完整world资格，其模型被正式阶段复用。freeze在每种结构内为三学习器提供完全同一批反馈，封存全部模型和动作。评价器之后读取外折E奖励。

RandomPair从19候选均匀抽取，新增1/2/3成员并使用原聚合/成员成本缓存。各结构只把已选完整前缀的观测传给原学习器。完整计划池费用和被选前缀费用分开，物理调用为0。该设计属于回顾性固定缓存对比。

CommonCoverage与TokenMatched分开计费。前者匹配原题数量与身份，后者匹配Token上限。所有开发正负结果保留。原始题、答案、私有矩阵和权重留在private；public为脱敏表、协议和源码。置信区间、bootstrap和假设检验不执行，fresh400和sealed200不读取。
""")
    for p in sources():
        target=PUBLIC/"code"/p.relative_to(CODE)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    if repair:
        write_json(PUBLIC/"PREFLIGHT_REPAIR_AND_COST.json",repair)
    verify_protocol()
    files=[dict(path=p.relative_to(PUBLIC).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(PUBLIC.rglob("*")) if p.is_file()]
    write_json(PUBLIC/"PUBLIC_DELIVERY_MANIFEST.json",dict(status="SEALED_FOR_PUBLICATION",files=files))
    write_json(PRIVATE/"FINAL_VERIFICATION.json",dict(status="PASS",public_files=len(files)+1,
        original_inputs_hash_verified=True,source_hashes_verified=True,predictions_frozen_before_evaluation=True,
        confidence_intervals=0,bootstrap_runs=0,hypothesis_tests=0,new_physical_llm_calls=0,
        fresh400_reads=0,sealed200_reads=0,practical_gate_passed=gate['all_passed'],
        result_sha256=sha256(PUBLIC/"E15_COMPLETE_RESULTS.json")))
    print("E15_REPORTS_COMPLETE",len(files)+1,flush=True)


if __name__=="__main__":
    main()
