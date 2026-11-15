"""Add public semantic identities and input-only confirmation audit notes."""
from itertools import combinations
import platform
from pathlib import Path
import csv

from e17_common import ASSETS, PRIVATE, PUBLIC, read_json, sha256, verify_protocol, write_json
from objectives import clarabel
import numpy as np
import scipy
import sklearn


def main():
    verify_protocol()
    write_json(PUBLIC / "RUNTIME_VERSIONS.json", dict(
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        scikit_learn=sklearn.__version__, clarabel=clarabel.__version__,
        system=platform.system(), machine=platform.machine(),
        llm_runtime_required=False, feature_encoder_forward_required=False))
    pool_path = ASSETS / "configs/pool_v1.json"
    pool = read_json(pool_path)
    members = {int(model["index"]): model["alias"] for model in pool["models"]}
    if set(members) != set(range(6)):
        raise ValueError("Six frozen physical members required")
    actions = [dict(physical_action=index, members=list(team),
                    aliases=[members[member] for member in team])
               for index, team in enumerate(combinations(range(6), 3))]
    write_json(PUBLIC / "PHYSICAL_TEAM_MAP.json", dict(
        source_pool_sha256=sha256(pool_path), actions=actions,
        semantics="Immutable physical identities in the original experiment order; numeric renamings are mapped back here."))
    source_records = []
    for record in read_json(PRIVATE / "INPUT_MANIFEST.json")["files"]:
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError("Historical provenance changed")
        source_records.append(dict(asset_relative_path=path.relative_to(ASSETS).as_posix(),
                                   bytes=record["bytes"], sha256=record["sha256"]))
    write_json(PUBLIC / "SOURCE_PROVENANCE.json", dict(
        status="HISTORICAL_ASSETS_UNCHANGED", files=source_records,
        scope="Hash-only historical inputs, frozen configurations, original heads, posterior cache and native solver"))
    fresh = read_json(PUBLIC / "FRESH_NUMBERING_INPUT_ONLY_SUMMARY.json")
    rows = [row for row in fresh["action_consistency_and_frequencies"]
            if row["feature_map_scope"] == "all_five_feature_maps" and row["input_group"] == "all_inputs"
            and row["inference"] in ("semantic_exact", "semantic_near")]
    lines = ["# 新输入上的编号敏感性补充", "", "## 输入侧结果", "",
        "同一400道新题输入，分别经过五个冻结历史预处理与模型。不读取答案、奖励或实际输出长度。", "",
        "| 模型 | 推理规则 | 100排列相对原编号的平均身份一致率 |", "| --- | --- | --- |"]
    for row in rows:
        value = row["identity_action_consistency_across_fold_numberings"]["mean"]
        lines.append(f"| {row['method']} | {row['inference']} | {100*value:.2f}% |")
    lines.extend(["", "这些数据检查训练产生的编号敏感性是否延续到新的输入。独立正确率没有执行，400个输入也没有因为五个模型与100个编号而变为更多独立问题。", "",
        "## 物理身份与数值边界", "",
        "PHYSICAL_TEAM_MAP.json给出全部20个实际组合的成员别名。组合0由Qwen3、Coder及Granite组成，组合19由DeepHermes、DeepSeek及Cogito组成。公开频数使用该固定实际身份。", "",
        "精确最高分和10⁻⁸近并列选择同时保留。该容差位于模型分数空间，不能当作实际奖励差异阈值。分数等变误差与动作变化分别记录，固定原权重仅改变推理优先级的对照分开报告。", "",
        "## 软件与执行记录", "",
        "冻结前完成25项基础单元测试，评价器另有14项可复现单元测试。评价器14项初审和相同案例的单元测试复验分别记录，复验不增加独立科学证据。", "",
        "主实验与合法复用均保存逐模型目标、正则、求解状态及CPU时间。独立工件验证覆盖完整555个模型文件，评价开始前核验协议、预测、样本与验证器哈希绑定。", "",
        "准备阶段首次语法检查发现一个未闭合括号，修复后才成功冻结协议。该次失败没有训练结果。正式求解失败以完整模型资格和收据为准，不用软件PASS替代科学结果。", "",
        "本轮没有新增语言模型调用、后验教师拟合或后验样本。完整train600监督属于共同历史投入。输入确认新拟合3个full600头，合成例使用10个小型QP，其成本分别保存。", "",
        "## 来源复核", "",
        "INPUT_MANIFEST中的原始E8、M4和NS资产哈希在最终封存前重新核对。GitHub发布只添加E17目录，父分支历史文件树保持原SHA。", ""])
    with (PUBLIC / "FRESH_INPUT_AND_EXECUTION_NOTES.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    results = read_json(PUBLIC / "E17_COMPLETE_RESULTS.json")
    summary = read_json(PUBLIC / "PERMUTATION_SUMMARIES.json")
    selected = {(row["method"], row["inference"]): row
                for row in summary["groups"] if row["group"] == "all"}
    original = {row["method"]: row for row in results["canonical_background"]}
    with (PUBLIC / "PERMUTATION_RUNS.csv").open(encoding="utf-8-sig", newline="") as stream:
        permutation_rows = list(csv.DictReader(stream))
    first_share = np.mean([float(row["numeric_first_share"]) for row in permutation_rows
                           if row["method"] == "SPO_coordinate" and row["inference"] == "semantic_exact"
                           and int(row["permutation"]) > 0])
    spo = selected["SPO_coordinate", "semantic_exact"]
    tspo = original["TSPO_uniform"]
    numeric_near = selected["TSPO_uniform", "numeric_near"]
    decision = f"""# E17机制结论与研究决策

## 一，任意冠军规则是否进入模型

100个随机重编号中，原SPO+平均{100*first_share:.2f}%的问题选择当前数字编号0所代表的实际组合。映射回固定实际身份后，与原编号模型的一致率仅{100*spo['identity_consistency_across100']['mean']:.3f}%。相同重编号下，Ridge和均匀TSPO+的精确推理身份一致率均为100%。

原SPO+的准确率在{spo['accuracy_across100']['minimum']:.3f}%至{spo['accuracy_across100']['maximum']:.3f}%之间变化，标准差{spo['accuracy_across100']['sd']:.3f}个百分点。这是固定旧题池上的标签约定敏感性，提供直接机制证据。其均值{spo['accuracy_across100']['mean']:.3f}%不能解释成新题总体性能。

不看输入的冠军先验与SPO+平均约80%的选队一致，但后者仍有平均+.84个百分点的质量差。因而当前证据支持较强冠军先验影响，同时保留输入相关决策的贡献边界。

## 二，消除训练编号偏好是否等于成功选队

均匀TSPO+精确服务准确率为{tspo['accuracy']:.3f}%，相对Ridge仅{tspo['versus_Ridge']['accuracy_difference_pp']:+.3f}个百分点，净增加{tspo['versus_Ridge']['net_correct_events']}条题目×生成事件。其相对原SPO+的点差为{tspo['accuracy']-original['SPO_coordinate']['accuracy']:+.3f}个百分点。当前数据支持编号稳健性修复，没有建立稳定的新题质量优势。

另有重要推理边界。TSPO+在{tspo['near_score_ties']}/600道题上的最高两分之差不超过10⁻⁸，中位分差约{tspo['top_score_gap_summary']['median']:.3g}。若在这个预先固定近并列集合中重新使用当前数字优先级，100排列的身份一致率降至{100*numeric_near['identity_consistency_across100']['mean']:.3f}%，准确率范围为{numeric_near['accuracy_across100']['minimum']:.3f}%至{numeric_near['accuracy_across100']['maximum']:.3f}%。因此必须把训练冠军和推理并列分别定义。该10⁻⁸规则是数值诊断，主服务始终为冻结的语义精确最高分。

## 三，损失运算顺序是否造成现实性能改善

固定相同8个后验draw，逐draw损失后平均为86.000%，先平均draw再构造SPO+为85.111%。后者有54次纠错和70次破坏，净少16条生成事件，三个生成seed的点差均为负。它的校准相对差MAE更低，最终选队质量在此冻结设置下仍更低。

已知真值反例证明均值估计完全正确时，运算顺序本身可以产生不同的最优动作。上述真实负结果同时说明，先平均奖励也不能被直接当作实际性能修复。工作后验失配、有限线性容量与有限draw仍未被真实数据唯一识别。

## 四，独立确认现在做到哪里

fresh400只完成输入侧机制审计及策略冻结。原SPO+在新输入上的编号身份一致率为5.863%，Ridge和均匀TSPO+精确推理均为100%。这些结果提供新输入上的选择敏感性，尚未提供新题准确率。

按TSPO+相对Ridge的合法题级开发方差，单侧5%正态工作规划检出1个百分点正收益的80%功效需约2386题，400题工作功效约26.54%。该规划还假设480题OOF模型方差可迁移到full600重训模型，不能作为精确样本量保证。

## 研究决策

继续以监督审计为当前主线。将任意冠军的训练放大、训练/推理编号分离、代理目标与真实期望奖励分离作为三项机制证据。保留全部性能负结果，维持现有冻结候选及其配置。

独立准确率阶段先核对远端缓存，再确定一次性评价规模与调用预算。本轮没有自动扩题、启动新LLM队列或把fresh400用于选参。sealed200继续封存。

优势是直接干预与已知真值反例能区分机制。风险是旧开发题反复暴露、极小分数差、工作后验无法直接校准及有限题量。机制干预的证据强度为高，广泛适用性和实际质量优势仍待独立确认。文献中已有监督稳定性及并列偏好研究，当前全局新颖性尚未建立。
"""
    with (PUBLIC / "05_MECHANISM_AND_RESEARCH_DECISION.md").open("x", encoding="utf-8") as stream:
        stream.write(decision)
    write_json(PUBLIC / "EXECUTION_SOURCE_AUDIT.json", dict(
        historical_manifest_sha256=sha256(PRIVATE / "INPUT_MANIFEST.json"),
        historical_manifest_files_verified=len(read_json(PRIVATE / "INPUT_MANIFEST.json")["files"]),
        historical_assets_changed=0, physical_llm_calls=0, fresh_reward_reads=0, sealed_reward_reads=0,
        core_unit_tests=25, evaluator_unit_tests=14, evaluator_case_replay=True,
        prefreeze_incidents=[dict(type="SyntaxError", phase="pre-protocol preparation",
                                 resolution="Closed the missing parenthesis; successful protocol and all fits use corrected source",
                                 model_outputs_created=0)]))
    print("E17_SUPPLEMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
