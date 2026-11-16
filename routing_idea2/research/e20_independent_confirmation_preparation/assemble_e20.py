"""Assemble E20 de-identified preparation reports from existing audit outputs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


CODE = Path(__file__).resolve().parent
BASE = "07b01361d425b089e49643f1ec5282eae0ac1ec0"
BRANCH = "codex/e20-independent-confirmation-preparation"


def read(name):
    return json.loads((CODE / name).read_text(encoding="utf-8-sig"))


def write(name, value):
    if isinstance(value, dict):
        content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    else:
        content = value
    (CODE / name).write_text(content, encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    models = read("FINAL_MODEL_NUMERICAL_AUDIT.json")
    access = read("E20_DATA_PERMISSIONS.json")
    near = read("E20_COHORT_NEAR_DUPLICATE_AUDIT.json")
    budget = read("E20_EXACT_EXECUTION_BUDGET_QUALIFIED.json")
    optional199 = read("E20_OPTIONAL_199_QUESTION_BUDGET.json")
    power = read("E20_POWER_RESULTS.json")
    review = read("E20_NUMERICAL_REVIEW.json")
    guards = read("E20_SOFTWARE_GUARDS.json")
    if any(item["status"] != "PASS" for item in (models, review, guards)):
        raise ValueError("Numerical and access qualifications must pass before reporting")
    fit_b, fit_c = (models["model_diagnostics"][name] for name in ("B", "C"))
    scores = models["input_only_diagnostics"]
    primary_power = power["decoders"]["tau_1e-8_uniform"]["single_primary_comparison"]
    power_1pp = {row["total_independent_questions"]: row["welch_noncentral_t_approximation_power"]
                 for row in primary_power["power_grid"] if row["effect_pp"] == 1}
    remaining = access["additional_pool_plan"]
    protocol = read("E20_PREPARATION_CONTRACT.json")
    protocol.update(
        schema="e20_p2_preparation_protocol_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="MODELS_AND_RULES_FROZEN_DATA_SELECTION_AND_EXECUTION_AWAIT_APPROVAL",
        final_model_sha256=models["frozen_model_sha256"],
        final_prefit_config_sha256=models["prefit_sha256"],
        final_fit_source_sha256=models["fit_source_sha256"],
        final_prediction_operator_source_sha256=review["reviewed_prediction_source_sha256"],
        cohort_prediction_receipts=models["frozen_input_receipts"],
        data_roles={"fresh400": "EXPOSED_DEVELOPMENT_ONLY", "sealed200": "ANSWER_SEALED_ONE_NEAR_DUPLICATE_PENDING_DISPOSITION"},
        proposed_new_pool={"status": "PROPOSAL_ONLY", "questions": 2400,
                           "gsm8k": 1200, "bbh": 1200, "selected_ids": 0,
                           "bbh_subtask_quota": "44 or45 per subtask,12 receive45; penguins_in_a_table receives44; exact allocation and seed require approval before selection",
                           "source_frame": "Remaining official GSM8K train and full official BBH test, after registered exclusions and frozen duplicate screens",
                           "power_at_1pp_conditional_on_E19_variance": power_1pp[2400],
                           "members_from_zero_upper_calls": 2400 * 18,
                           "aggregations_from_zero_upper_calls": 2400 * 60,
                           "scope": "Only eventual positive-support union; these are capacity envelopes before input-side selection"},
        statistical_reporting="No new CI or CI-only bootstrap under the latest standing instruction; two-sided power and fixed-time test definitions retained",
        linked_reports={name: sha(CODE / name) for name in (
            "FINAL_MODEL_NUMERICAL_AUDIT.json", "E20_DATA_PERMISSIONS.json",
            "E20_COHORT_NEAR_DUPLICATE_AUDIT.json", "E20_EXACT_EXECUTION_BUDGET_QUALIFIED.json",
            "E20_OPTIONAL_199_QUESTION_BUDGET.json",
            "E20_POWER_RESULTS.json", "E20_NUMERICAL_REVIEW.json", "E20_SOFTWARE_GUARDS.json",
            "05_P2_FROZEN_PROTOCOL.md")},
    )
    write("E20_PROTOCOL.json", protocol)
    model_rows = "\n".join(
        f"| {name} | {row['status']} | {row['objective']:.12f} | {row['stationarity_inf']:.3g} | {row['primal_dual_objective_gap']:.3g} | {row['wall_seconds']:.4f} |"
        for name, row in (("B", fit_b), ("C", fit_c))
    )
    pool_rows = "\n".join(
        f"| {pool} | {row['support_summary']['B']['mean']:.2f} | {row['support_summary']['B']['numerically_tied_questions']} | {row['both_strict_unique_questions']} | {row['strict_switched_questions']} |"
        for pool, row in scores.items()
    )
    write("01_FINAL_MODEL_FREEZE.md", f"""# E20最终B/C模型冻结报告

## 研究对象与实际执行

在原train600的600道独立题、每题20组合×3生成评分上，分别完成一次B和一次C拟合。B的β为1，C的β为0，两支γ均为0。预处理为原冻结MiniLM-L6的384维题目向量、train600内PCA16和StandardScaler、首列截距。每支线性头17×20，共340参数。没有OOF集成，也没有参数网格搜索。

正则系数为0.003472222222222222，来自E17最终600题的冻结五折参数中位数规则，并与其已执行两模型日志核对。全部参数包含截距均受罚。所有600条训练行及1/600归一化保持，306道全动作经验奖励相同的题仅改变β行权重，其余294行目标保持。经验最优集合使用均匀标签。

## 数值资格

| 模型 | 求解状态 | 实际目标 | KKT stationarity∞ | primal-dual gap | 拟合墙钟秒 |
|---|---|---:|---:|---:|---:|
{model_rows}

两次实际拟合共用CPU {fit_b['cpu_seconds'] + fit_c['cpu_seconds']:.4f}秒，拟合墙钟合计{fit_b['wall_seconds'] + fit_c['wall_seconds']:.4f}秒。正L2保证W唯一，C的零权重辅助上图变量允许非唯一。目标独立重算误差至多浮点舍入量。原E17 full600的PCA均值、PCA方向、标准化均值和尺度四数组逐值恢复。

端点最优性资格满足0 ≤ {models['endpoint_inequality']['F0_B_minus_F0_C']:.12f} ≤ {models['endpoint_inequality']['JT_C_minus_JT_B']:.12f}。沿用E19已通过的动作反向映射资格，本轮只新增两次主拟合。

## 冻结与输入侧结果

模型工件SHA256为`{models['frozen_model_sha256']}`。训练前配置SHA256为`{models['prefit_sha256']}`。私有模型收据先于fresh400和sealed200输入预测。主部署固定为τ=1e−8均匀argmax，同时保留raw-exact均匀策略。

| 输入池 | B平均支持组合数 | B数值并列题 | B/C均严格唯一题 | 严格换队题 |
|---|---:|---:|---:|---:|
{pool_rows}

C在两个池均为唯一动作。两池全部输入的中心化分数变化均超过1e−7。上述量均由输入和冻结模型得到，尚未计算任何新的部署正确率、纠错或破坏。

## 修正与失败记录

初版输入预测采用代数等价的中心化矩阵乘与einsum。正式版本恢复原sklearn及逐头Fortran-order矩阵乘顺序，保留初版私有记录。fresh400最大分数差为1.665×10⁻¹⁶，主策略与raw-exact策略均无行变化。正式预算只引用ORIGINAL_OPERATOR_FROZEN工件。该修正未使用评价奖励。

sealed200输入编码完成200题、25批冻结MiniLM前向，CPU {models['sealed_input_encoding']['cpu_seconds']:.4f}秒、墙钟{models['sealed_input_encoding']['wall_seconds']:.4f}秒。一次预测入口因GSM空子任务形成object元数据而安全拒绝。修复为Unicode，保留原包，向量逐值不变，没有重复编码或拟合。子任务字段仅用于汇总，未进入预测器。

## 优势与风险

优势是B/C目标、模型容量、数据与预处理相同，干预集中于训练全并列行权重，且模型和输入策略先于任何新评价冻结。

风险是B存在大量数值近并列，精确评价均匀策略需要覆盖较宽的动作支持集。输入侧严格换队能支持分数和策略机制诊断，其服务质量后果仍需合法独立评价。
""")
    summary_rows = "\n".join(
        f"| {pool} | {row['new_member_calls']:,} | {row['new_aggregate_calls']:,} | {row['verified_reused_member_calls']:,} | {row['verified_reused_aggregate_calls']:,} |"
        for pool, row in budget["cohorts"].items()
    )
    readiness = dict(
        status="PREPARATION_COMPLETE_FORMAL_EVALUATION_NOT_READY",
        engineering_qualification="PASS", model_frozen=True,
        fresh400_independent_confirmation="FAIL_HISTORICAL_REWARD_EXPOSURE",
        sealed200_answer_seal="PRESERVED",
        sealed200_independence="ONE_NEAR_DUPLICATE_PENDING_DISPOSITION",
        sealed200_power_for_1pp=power_1pp[200],
        independent_pool_proposal="New2400,1200 per task, no admission or selection performed",
        proposed_new_pool_power_for_1pp=power_1pp[2400],
        user_execution_approval="PENDING", execution_started=False,
        new_member_calls_executed=0, new_aggregation_calls_executed=0,
        fresh400_reward_values_decoded=0, sealed200_answer_values_decoded=0,
    )
    write("E20_READINESS_REVIEW.json", readiness)
    write("06_READINESS_REVIEW.md", f"""# E20执行前资格审查

## 结论

**准备工作完成，正式独立评价尚未具备启动条件。** 两个最终模型和策略已冻结，数值及权限资格通过。fresh400已有历史奖励曝光；sealed200仍封存答案，但有1道BBH题需近重复处置，且名义200题对1pp差异的规划功效仅{100 * power_1pp[200]:.2f}%。

| 审查项 | 结果 | 依据 |
|---|---|---|
| 最终600题训练与冻结 | 通过 | 两次实际拟合、模型及策略收据 |
| 优化与输入权限 | 通过 | 端点不等式、KKT、41项只读复核及7项软件守卫 |
| fresh400独立性 | 未通过 | 全部400题属于E18已完成评分范围 |
| sealed200答案封存 | 保持 | 本轮仅输入投影、编码与预测 |
| sealed200题目独立性 | 待处置 | 1题与历史开发及fresh400高相似 |
| 200题检出1pp的80%功效 | 未达到 | 当前规划功效{100 * power_1pp[200]:.2f}% |
| 实际LLM执行 | 等待批准 | 当前新增调用为0 |

## 待审批的独立题池提案

建议另建2400道未曝光独立题，GSM8K与BBH各1200题，sealed200继续保留。旧开发方差下检出1pp的双侧规划功效约{100 * power_1pp[2400]:.2f}%。1908题是80%功效的近似最低总题量，2400保留规划余量，仍需承认最终模型与题源变化造成的方差风险。

既有输入索引盘点剩余GSM8K官方train {remaining['remaining_gsm8k_train']}题、BBH官方test {remaining['remaining_bbh_test']}题，27个BBH子任务均有剩余。原共同BBH子池仅余{remaining['original_bbh_common_subpool_remaining']}题，使用更大题池应明确扩展到完整官方BBH采样框。BBH1200题可采用每子任务44或45题的预定配额，其中12个45题，15个44题，最稀缺的penguins_in_a_table保留44题配额。具体配额表、固定抽样seed和问题身份清单须在选题前批准。该提案未选择或加入任何新题。

新增池的从零执行容量封套为最多43200次成员生成和144000次聚合。实际执行仍只包含B/C正概率支持集并集，获批后在输入侧冻结支持清单才可给出精确调用量。完整20动作是最坏容量边界。当前未执行该封套。

## 两个现有池的精确调用预算

| 题池 | 尚需成员生成 | 尚需聚合 | 可复用成员 | 可复用聚合 |
|---|---:|---:|---:|---:|
{summary_rows}

full200与条件199题方案分别保留。隔离预警题的199题情景需要3486次成员和9336次聚合，100道GSM8K、99道BBH保持任务各50%权重，1pp规划功效约14.64%。具体Token、算子身份、已发生缓存费用和条件上限见执行预算报告。没有将缓存实际Token当成本轮新增Token。

## 后续批准应固定的事项

批准题池规模和来源、近重复隔离处置、具体问题清单生成规则与调用预算后，方可启动实际P2。模型、β/γ、正则、τ及评分规则继续冻结。fresh400只可作开发性评价。任何低功效封存评价均须事先记录能排除的效应范围，保留正负结果。

本轮优势是最终模型和成本清单已具备执行依据，数据曝光与近重复风险得到显式记录。主要限制是独立题池及预算尚未获批，旧开发方差不能保证未来统计功效，也尚无新质量结果。
""")
    write("EXPERIMENT_SUMMARY.md", f"""# E20准备结果

## 两个主要结论

**B/C最终600题模型已训练并冻结，数值资格通过。** 正则沿用E17冻结的0.003472222222222222，306条全并列行保留，仅β权重不同。全部600行及1/600分母、17×20头和原预处理保持。两次主拟合的墙钟合计{fit_b['wall_seconds'] + fit_c['wall_seconds']:.4f}秒。

**当前题池尚不足以开展目标为1pp的充分功效独立确认。** fresh400已在E18全部曝光；sealed200仍封存答案，但有1道BBH输入近重复待处置，名义200题的功效约{100 * power_1pp[200]:.2f}%。旧开发配对方差下，80%检出0.5/1/1.5/2pp分别需7620/1908/850/480道独立题。

## 策略机制与预算

B在fresh400/sealed200平均支持14.88/15.56个组合，C均为唯一动作。τ=1e−8的均匀argmax策略按其正概率支持并集评价。raw-exact诊断未增加所需组合。

| 题池 | 尚需成员生成 | 尚需聚合 | 可复用成员 | 可复用聚合 |
|---|---:|---:|---:|---:|
{summary_rows}

已有输出身份、生成seed、聚合算子和缓存Token按原闭合审计核对。Token上限的成立条件和先验缓存费用单列。

## 实际执行与修正

完成2次真实头拟合、200题冻结MiniLM输入编码及两池输入预测。新增成员生成0、聚合0、fresh奖励读取0、sealed答案读取0。本轮未计算新置信区间或bootstrap。

一次sealed输入包因object元数据被权限接口拒绝，修复Unicode元数据后向量未变，也没有重复编码。正式推理恢复原sklearn/F-order数值顺序，保留代数等价初版；fresh主/raw策略均未改变。全部私有逐题分数、概率、身份、权重与收据保存在独立E20私有目录。

## 优势、风险与下一步决定

优势是干预和模型容量明确、真实执行并集可审计，且历史曝光得到追溯。风险来自τ附近的大量支持动作、宽支持的执行成本、封存池近重复及旧开发方差的可迁移性。

提交待审批的新2400题提案，GSM8K/BBH各1200题；按旧方差检出1pp的规划功效约{100 * power_1pp[2400]:.2f}%。该方案仅盘点来源和容量，尚未选题、扩池或启动生成。sealed200保持答案封存。本轮完成准备后等待用户批准。

E20协议、模型哈希、权限报告、精确调用预算、功效表和复现入口均已保存。原E17/E18/E19结果继续保留。
""")
    complete = dict(schema="e20_complete_preparation_results_v1", source_commit=BASE, branch=BRANCH,
                    status="PREPARATION_COMPLETE_AWAIT_APPROVAL", protocol_sha256=sha(CODE / "E20_PROTOCOL.json"),
                    final_model_numerical_audit=models, data_permissions=access, near_duplicate_audit=near,
                    execution_budget=budget, optional_199_budget=optional199, power_results=power, numerical_review=review,
                    software_guards=guards, readiness=readiness,
                    new_llm_calls=0, new_member_generations=0, new_aggregations=0,
                    new_prediction_head_fits=2, frozen_encoder_questions=200,
                    evaluation_reward_reads=0, sealed_answer_reads=0,
                    new_confidence_intervals=0, new_bootstrap_repetitions=0)
    write("E20_COMPLETE_RESULTS.json", complete)
    print(json.dumps({"status": complete["status"], "model_sha": models["frozen_model_sha256"],
                      "new_llm_calls": 0, "reports": 6}, ensure_ascii=False))


if __name__ == "__main__":
    main()
