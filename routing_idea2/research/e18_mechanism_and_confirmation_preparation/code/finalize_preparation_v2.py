"""Append E18 software-adapter qualification to the immutable v1 preparation."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import shutil

from e18_common import CODE, PRIVATE, PUBLIC, preserve_e17, read_json, sha256, write_json
from verify_and_seal_e18 import collect_private_identifiers, verify_existing


def save_markdown(path, content):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content.strip() + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--qualification", action="append", required=True)
    parser.add_argument("--failure-record", action="append", default=[])
    parser.add_argument("--supplement", action="append", default=[])
    parser.add_argument("--independent-review", required=True)
    args = parser.parse_args()
    verify_existing()
    original = preserve_e17()
    destination = PUBLIC / "v2"
    destination.mkdir(exist_ok=False)
    qualifications = {}
    runtime_records = []
    for name in args.qualification + [args.independent_review]:
        if Path(name).name != name:
            raise ValueError("A qualification filename must be local and explicit")
        value = read_json(CODE / name)
        if not str(value.get("status", "")).startswith("PASS"):
            raise ValueError("Unqualified software adapter " + name)
        shutil.copy2(CODE / name, destination / name)
        qualifications[name] = {"sha256": sha256(CODE / name), "status": value["status"]}
        runtime_records.append(dict(file=name, status=value["status"],
                                    cpu_s=value.get("cpu_s"), wall_s=value.get("wall_s", value.get("elapsed_s"))))
    for name in args.failure_record:
        if Path(name).name != name:
            raise ValueError("A failure record filename must be local and explicit")
        value = read_json(CODE / name)
        shutil.copy2(CODE / name, destination / name)
        runtime_records.append(dict(file=name, status=value.get("status"),
                                    cpu_s=value.get("cpu_s"), wall_s=value.get("wall_s", value.get("elapsed_s"))))
    for name in args.supplement:
        if Path(name).name != name or Path(name).suffix != ".md":
            raise ValueError("An additive Markdown supplement must have a local name")
        shutil.copy2(CODE / name, destination / name)
    sources = {}
    for relative in args.source + ["finalize_preparation_v2.py"]:
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Invalid additive source path")
        source = CODE / name
        ast.parse(source.read_text(encoding="utf-8-sig"))
        target = PUBLIC / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError("Version1 source is immutable " + str(name))
        shutil.copy2(source, target)
        sources[name.as_posix()] = sha256(source)

    prior_protocol = PUBLIC / "E18_EVALUATION_PROTOCOL.json"
    protocol = read_json(prior_protocol)
    protocol.update(
        schema="E18_frozen_confirmation_preparation_v2",
        previous_protocol_sha256=sha256(prior_protocol),
        status="FROZEN_PREPARATION_SOFTWARE_QUALIFIED_AWAITING_TARGET_AND_EXECUTION_APPROVAL",
        software_adapter_qualification=qualifications,
        scope_adapter_readiness=dict(
            isolated_scope_factories="SOFTWARE_QUALIFIED",
            single_task_member_and_typed_aggregation="FAKE_BACKEND_QUALIFIED",
            reference_projection="SYNTHETIC_SOURCES_QUALIFIED",
            sparse_scoring_and_resource_ledger="SYNTHETIC_RECEIPTS_QUALIFIED",
            actual_gpu_runtime="NOT_RUNTIME_TESTED",
            full_cohort_production_dispatch="NOT_STARTED",
            production_approval_receipts_created=0,
            real_fresh_gold_reads=0,
            sealed_answer_reads=0,
        ),
        input_scope_policy="Exact frozen policy union; no original split manifest or scientific source edits",
        approval_rule="Following explicit user approval the executor records hash-bound scope authorization; no extra user-authored receipt is required",
        reference_source_policy="Approved single-reference decoding, with any full-source hash or structural byte scans explicitly bound to the approved source scope",
    )
    write_json(destination / "E18_EVALUATION_PROTOCOL_v2.json", protocol)
    write_json(destination / "SOFTWARE_RUN_COSTS_v2.json", dict(
        runs=runtime_records,
        reported_cpu_s=sum(row["cpu_s"] for row in runtime_records if row["cpu_s"] is not None),
        reported_wall_s=sum(row["wall_s"] for row in runtime_records if row["wall_s"] is not None),
        complete_session_wall_s=None,
        complete_session_wall_unavailable_reason="Asynchronous preparation and source review are not covered by a single timed process",
        timing_scope="Recorded software qualification invocations; overlapping agent runs are not additive elapsed time",
        actual_new_llm_calls=0,
        actual_new_gold_scoring=0,
    ))
    budget = read_json(PUBLIC / "E18_FINAL_CALL_BUDGET.json")
    save_markdown(destination / "EXECUTION_QUALIFICATION_v2.md", """
# E18执行准备的软件资格补充

## 当前入口

`code/e18_execution_adapter.py`提供精确范围构造、审批绑定、原MemberStore和typed聚合Store的单任务入口。`code/e18_reference_adapter.py`提供批准题目集合内的单题reference投影。`code/e18_scoring_adapter.py`使用冻结确定性评分器并保存稀疏评分和执行费用账本。`code/sparse_policy_projection.py`将已执行且有合法评分收据的组合投影至四套冻结策略。

资格使用临时合成输入、FakeBackend和原Store接口。各资格JSON与源码哈希随本版封存。真实GPU加载、完整3000题调度及真实Gold投影属于批准后的运行阶段。

## 运行和审批边界

准备阶段只创建审阅用DAG和冻结预测。收到明确执行批准后，执行者将批准范围写入hash绑定收据，引用协议、题目身份、冻结动作、来源、预算及各组件scope。用户无需手工编写收据。原逻辑transport使用online_stream，外部研究身份固定为e18_confirm，原全局数据划分保持原字节。

执行前在目标主机核对模型路径映射、权重、tokenizer、运行依赖与实际GPU状态。以精确缺失key分配任务，合法历史缓存走已核验的复用桥。原成本journal及未决费用围栏持续生效。成员seed为42、137、2024，聚合decode与匿名排序使用原seed42，成员重复通过provenance绑定。

Gold投影scope按已批准源及原行定位冻结。部分格式的源SHA和结构定位需要扫描其他记录的编码字节，扫描权限在源scope中显式记录，未授权reference值保持未解码状态。格式或定位无法满足要求时入口拒绝处理。正式评分的reference按canonical_question与reference_version复用，输出分别计分。

## 优势与风险

优势是复用原提示、typed合同、实际Token记录和Store接口，同时将执行范围限制到已冻结策略并集。合成软件资格覆盖权限、身份、seed、复用及失败处理。主要风险为正式GPU环境、实际源文件与完整队列尚未进行生产端到端验证。通过软件资格只支持实现完整性结论。

BBH扩展来源1254题及3000题总体范围仍需用户确认。未知远端缓存范围、人工续跑的累计费用和语义近重复残留均沿用v1报告中的限制。
""")
    save_markdown(destination / "REPRODUCE_v2.md", r"""
# E18 v2复现补充

## 依赖与输入

使用原冻结Python3.11环境及原budget_router源码、六模型配置、typed协议和来源收据。原私有依赖清单见根目录REPRODUCE。实际运行时可通过MAB_ASSET_ROOT指向包含原src、scripts、configs、artifacts及protocol的私有资产根。测试使用临时合成题目与FakeBackend，reference测试只用临时合成JSONL/JSON。

## 纯软件资格

在本交付目录的code中运行以下入口，输出文件需使用新的路径。

```powershell
$env:MAB_ASSET_ROOT = 'E:\语言模型组合路由'
python -B -X utf8 .\e18_execution_adapter.py --describe-only
python -B -X utf8 .\qualify_execution_adapter.py --output .\execution_recheck_new.json
python -B -X utf8 .\qualify_e18_reference_adapter.py --output .\reference_recheck_new.json
python -B -X utf8 .\qualify_e18_scoring_adapter.py --asset-root $env:MAB_ASSET_ROOT --output .\scoring_recheck_new.json
```

当前最终独立复核分别为33、22、15项，通过结果及实际被测源码SHA存于本目录ROOT_FINAL2记录。初期toy来源角色、模型alias、CLI输出参数和审批拥有者路径衔接失败已保留。软件资格中的问题数不参与真实科学样本量计算。

## 批准后的执行接口

`build_scope_contract`只接受冻结输入和四策略动作并集。`E18ExecutionAdapter`构造原生job，`run_approved_task`逐key调用原Store并保存实际费用。原始本地后端工厂为`original_backend_factory`，原服务器映射后端工厂为`mapped_backend_factory`。详细参数及审批schema见EXECUTION_ADAPTER_USAGE_v2.md。

`build_reference_scope`冻结源SHA、原行定位、版本和排除身份。`E18ReferenceProvider.get_reference`提供批准范围内的单题reference。`build_scoring_scope`与`SparseScoringAdapter.score_one`连接完整执行收据和原确定性评分器。费用与失败记录读取`.ledger()`及`resource_ledger_from_execution_receipts`。稀疏评分再交给`project_frozen_policy_scores`和`power_and_statistics.paired_question_test`。

生产收据由执行者根据用户的明确批准生成，绑定固定输出拥有者根和各scope。实际主机上的模型、tokenizer映射及资源分配在执行前核对。完整多worker cohort调度尚未实现，当前交付为可审查的单任务接口及其软件资格。

## 封存核验

v1清单及其文件保持原字节。v2/PUBLIC_DELIVERY_MANIFEST_v2.json覆盖完整追加交付，逐文件检查SHA256即可回读完整性。E17公共清单及5份冻结私有工件的核验方法保留在原verify_and_seal_e18.py与e18_common.preserve_e17中。准备构建脚本采用exclusive-create模式，在同一封存目录重复执行会拒绝覆盖。
""")
    save_markdown(destination / "EXPERIMENT_SUMMARY_v2.md", """
# E18准备阶段中文汇总

## 机制整理

理论章节、机制图、E17核心表和ACL实验组织草稿已完成。受控重编号证据支持冠军约定影响共享预测头。准确率影响、训练等变性、推理并列及Bayes目标分别记录。相关工作对照包含DARS、EquiRouter和原始SPO+。

## 独立确认方案

冻结四套E17策略。推荐方案含fresh400与拟新增2600题，GSM8K和BBH各1500题。拟新增数据保持PROPOSED_NOT_ADMITTED。1254道BBH来自官方题库覆盖扩展，应单列来源层并经用户确认。旧抽样框的剩余题量无法支撑原框架内3000题。

使用旧开发配对方差作条件规划，双侧5%检验、目标收益1个百分点时，N=3000功效约82.71%。目标为0.5个百分点时约30.59%。功效依赖方差可迁移性及冻结策略表现，完整N网格和方差敏感性见统计功效报告。

## 缓存与费用

已核验5330份成员输出，4717份位于本轮策略调用并集内。已验证缓存范围内尚需42569次成员生成和30921次聚合。fresh400对应1565次成员与4086次聚合，新增2600题对应41004次成员与26835次聚合。

缺失key首次尝试的保守Token上限为514848768。该数不覆盖未知范围缓存、人工续跑或所有失败后的累计费用。实际新增推理Token为0，实际新题Gold评分为0。

## 软件资格与修正

隔离范围、单任务执行、批准reference投影、稀疏评分和费用去重已完成合成资格，真实GPU与完整队列尚待批准后预检。旧输入归档中的混合类型subtask元数据已修为Unicode v2，既有编码、模型权重、动作与预算保持逐项一致。机制图使用draw.io XML与共同几何生成SVG和PNG，PNG经人工视觉检查，当前环境缺少draw.io原生CLI渲染。

## 下一步决定

当前进入独立确认的目标范围与费用审批。批准内容应明确BBH官方覆盖扩展、固定总题数和预算。批准后按冻结并集进行主机资格检查与实际执行，沿用固定评价时点和预定主比较。E17原50份公开文件和5份冻结私有确认工件的哈希保持一致。
""")
    save_markdown(PUBLIC / "README.md", """
# E18机制整理与独立确认准备

当前版本采用[v2中文汇总](v2/EXPERIMENT_SUMMARY_v2.md)、[v2冻结评价协议](v2/E18_EVALUATION_PROTOCOL_v2.json)及[软件执行资格](v2/EXECUTION_QUALIFICATION_v2.md)。原v1准备报告与封存清单完整保留，v1中的待实现状态记录了当时的准备进度。

## 六项交付

1. [机制理论报告](E18_MECHANISM_THEORY_REPORT.md)
2. [数据来源与去重报告](E18_DATA_SOURCE_AND_DEDUP_REPORT.md)
3. [统计功效报告](E18_STATISTICAL_POWER_REPORT.md)
4. [当前冻结评价协议](v2/E18_EVALUATION_PROTOCOL_v2.json)
5. [精确调用预算](E18_EXACT_CALL_BUDGET_REPORT.md)
6. [当前执行软件资格](v2/EXECUTION_QUALIFICATION_v2.md)

理论全文、相关工作核查、ACL实验结构和可编辑机制图位于根目录与figures。私有逐题输入、预测、冻结权重、源定位、HMAC指纹及缓存保持在原私有资产目录。复现基础步骤见[REPRODUCE](REPRODUCE.md)，v2新增入口与测试见[补充执行说明](v2/EXECUTION_QUALIFICATION_v2.md)。

3000题方案包含1254道BBH官方覆盖扩展，处于等待用户批准状态。新物理LLM调用与真实新题评分均为0。
""")
    write_json(destination / "PREPARATION_COMPLETE_v2.json", dict(
        status="PASS_PREPARATION_SOFTWARE_QUALIFIED_AWAITING_APPROVAL",
        previous_release_manifest_sha256=sha256(PUBLIC / "PUBLIC_DELIVERY_MANIFEST.json"),
        previous_protocol_sha256=sha256(prior_protocol),
        protocol_sha256=sha256(destination / "E18_EVALUATION_PROTOCOL_v2.json"),
        new_sources=sources,
        qualifications=qualifications,
        E17_preserved=original,
        paid_execution_authorized=False,
        actual_new_llm_calls=0,
        actual_new_scoring=0,
        automatically_admitted_questions=0,
        original_protected_inputs_processed_once=200,
        additional_protected_inputs_read=0,
        sealed_answers_read=0,
        final_budget_sha256=sha256(PUBLIC / "E18_FINAL_CALL_BUDGET.json"),
        verified_required_cache_reuse=budget["required_verified_member_reuse"],
    ))
    forbidden = collect_private_identifiers()
    files = sorted(path for path in PUBLIC.rglob("*") if path.is_file())
    for path in files:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"))
        if path.suffix != ".png":
            content = path.read_text(encoding="utf-8-sig")
            if any(value in content for value in forbidden):
                raise ValueError("Private identifier in public additive release " + path.name)
            if re.search(r"\b59\.174\.\d+\.\d+\b", content):
                raise ValueError("Private connection information in public release")
    verify_existing()
    write_json(destination / "RELEASE_QUALIFICATION_v2.json", dict(
        status="PASS_ADDITIVE_RELEASE",
        previous_files_unchanged=True,
        privacy_matches=0,
        scanned_public_files=len(files),
        denylist_strings=len(forbidden),
        actual_gold_reads=0,
        actual_new_generations=0,
        source_ast_checks="PASS",
        E17_preserved=preserve_e17(),
    ))
    entries = [dict(path=path.relative_to(PUBLIC).as_posix(), bytes=path.stat().st_size,
                    sha256=sha256(path)) for path in sorted(PUBLIC.rglob("*")) if path.is_file()]
    manifest = dict(status="SEALED_ADDITIVE_E18_V2", files=entries,
                    previous_manifest_sha256=sha256(PUBLIC / "PUBLIC_DELIVERY_MANIFEST.json"),
                    raw_question_exports=0, gold_exports=0, weight_exports=0)
    write_json(destination / "PUBLIC_DELIVERY_MANIFEST_v2.json", manifest)
    write_json(PRIVATE / "ADDITIVE_RELEASE_v2.json", dict(
        manifest_sha256=sha256(destination / "PUBLIC_DELIVERY_MANIFEST_v2.json"),
        sources=sources, files=len(entries) + 1,
    ))
    print("E18_ADDITIVE_V2_SEALED", len(entries) + 1, flush=True)


if __name__ == "__main__":
    main()
