# E24 复现入口

父版本为 E23 `3958f9ceaac5b0b1de88bbd47d89ddd940833ff2`。依赖本分支既有的 E18 白名单投影与封存单向签名、E20 冻结 B/C 模型及预测算子、E23 原 train600 组合成功数。私有数据根目录为 `E:\语言模型组合路由`，原题、推理键、模型参数及所有逐题分数均留在该目录的研究工件中。

## 已完成阶段

1. 用 `audit_source_capacity.py --asset-root ASSETS --private-output CANDIDATE_INPUTS --report SOURCE_AUDIT_JSON` 完成候选资格与来源核对。输入包括原 GSM8K 和 BBH 固定快照、登记身份、E18/P2/E23 执行身份、P3 输入及原有封存位置和单向签名。完整公开收据含本次运行耗时，复现可比较其稳定字段与候选 JSONL 摘要。
2. 用 `audit_margins.py --assets ASSETS --output MARGIN_AUDIT_JSON` 从原 P2 输入、预测、原 train600 成绩及冻结模型重放旧题池分差并恢复对偶见证。该步骤读取已授权历史训练成绩。
3. 用 `plan_design.py historical --e23-private E23_PRIVATE --bbh-capacities SOURCE_AUDIT_JSON --output PLANNING_JSON` 在历史四策略效用上复核交互功效和旧调用账本。
4. 用 `prepare_cohort.py --audit SOURCE_AUDIT_JSON --candidates CANDIDATE_INPUTS --assets ASSETS --output NEW_COHORT_DIR` 按冻结协议选题并编码。此入口需要原 P2 `E:\语言模型组合路由\.venv\Scripts\python.exe`，MiniLM 384D、`transformers=4.51.3`、`tokenizers=0.21.4`。本轮候选文件及入组工件位于 `research/e24_training_deployment_replication/private/`。
5. 用 `replication.py freeze --model FINAL600_MODELS_FROZEN_NPZ --inputs INPUTS_ONLY_NPZ --admission INPUT_COHORT_ADMISSION_JSON --output E24_POLICIES_FROZEN_NPZ --receipt E24_POLICY_FREEZE_RECEIPT_JSON` 在原 P2 环境重放 B/C，冻结四策略概率。这里使用原模型参数，预测阶段没有训练。
6. 用 `plan_design.py input-budget --predictions E24_POLICIES_FROZEN_NPZ --prediction-sha256 PREDICTION_SHA256 --output INPUT_ONLY_BUDGET_JSON --private-output FROM_ZERO_EXECUTION_KEYS_JSON` 计算精确调用量与私有任务键。

当前 `E24_PROTOCOL.json` 直接绑定已归档的 `SOURCE_AUDIT.json`，其候选摘要必须与新生成的输入框一致。用该已归档 JSON 作为第 4 步的收据。四种解码在旧 E23 冻结分数上与原四策略逐元素一致。合成数据覆盖 C 近并列时应用先验、训练成功数残余并列的均匀回退、交互符号、完全退化方差和缺失支持拒绝。预算检查单独验证两支 P 的输出全部存在于 U 的执行并集。

## 待批准的正式执行

按私有调用键接入原 P2 推理器和评分器，只接受冻结支持内具有完整原生证据的二值聚合评分。新执行 scope 需绑定 `E24_PROTOCOL.json`、私有题池、预测和调用集合，并记录实际 Token、原生成功、失败及合法缓存复用。原 P2/E23 的启动清单绑定各自历史题池，E24 需要独立清单。

评分阶段生成与预测绑定的 `e24_closed_native_scoring_v1` 收据及 JSONL。每行包含 `question_id`、`generation_seed`、`combination_index`、`score`，每个正概率支持键列一次。收据包含 `prediction_sha256`、`score_sha256`、`aggregation_fingerprint`、`sealed_answer_values_decoded`。入口为 `replication.py evaluate --predictions E24_POLICIES_FROZEN_NPZ --receipt POLICY_RECEIPT --scores CLOSED_SCORES_JSONL --scoring-receipt CLOSED_SCORING_RECEIPT --output E24_RESULTS_JSON`。该入口核对专属收据、支持全集和原题集配额，生成同题池四策略准确率、交互及区域分解。遇到缺失或不可评分，原始问题保留在审计账本，主要投影等待完整评分或预设范围内的单独数据质量处置。
