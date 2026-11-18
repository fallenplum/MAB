# 复现范围、依赖与私有工件

本文件针对论文主线 E17–E23。E17–E22 的审查基准为提交 `8f490a06ebe1153be2fcbeb42ad3944d63075b7d`；E23 补充审查基准为提交 `3958f9ceaac5b0b1de88bbd47d89ddd940833ff2`，具体见第 9 节。主线保留冻结源码、协议、脱敏结果和来源记录；较早实验的完整记录保存在历史研究分支。

**公开结果核验、恢复私有缓存后重算、重新训练以及重新生成语言模型答案，是四种不同的复现工作。当前仓库可以提供公开证据核验入口；单独克隆仓库还不具备完整重新生成全部实验的条件。** 私有逐题数组、模型权重和原生输出未随 Git 提交，审查时还发现原执行链所需的四份冻结源码未出现在 E22 快照中。缺口及已记录的 SHA-256 列在 [private_artifacts.json](../reproducibility/private_artifacts.json)。

本次整理没有重新训练路由器、调用语言模型、访问封存答案、重新评分或计算新的置信区间与 bootstrap。

## 1. 从仓库即可进行的检查

从仓库根目录运行：

```bash
python scripts/verify_repository.py
python scripts/build_paper_tables.py --check
```

第一项检查当前主线交付的文件、来源与完整性；第二项核对论文用表与仓库中已经发布的汇总结果。它们不应加载私有逐题奖励，不应启动历史实验脚本，也不证明缺失的私有运行环境已经恢复。

冻结文件的清单为 [frozen_manifest.json](../reproducibility/frozen_manifest.json)，逐项记录 Git blob SHA、SHA-256、大小和模式。`.gitattributes` 要求 Git 保留原始目录的字节，避免跨平台检出时改变已冻结的行尾。自动核验只解析 Python 语法，不导入会调度生成或写结果的历史脚本。

恢复私有工件后，可以检查清单中已有的文件哈希，例如：

```bash
python scripts/check_private_artifacts.py --asset-root /path/to/restored/assets --dars-root /path/to/frozen/DARS
```

此命令不启动训练或生成，不将原题或答案打印到日志。缺失、哈希不符、需要人工盘点的通配路径，以及未提供的 DARS 根目录都会返回 `INCOMPLETE`（退出码 2）；命令行参数解析错误也返回 2，清单或资产根路径错误返回 1。清单中的 `model_cache` 和 `protected_data` 两节不由该脚本自动验证；模型快照指纹、目录资产、路径映射和运行环境仍须人工或原阶段收据核验。即使单文件哈希匹配，也不把它当作端到端数值复现成功。

公开主结果的入口包括：

| 阶段 | 可直接核对的证据 |
|---|---|
| E17 | `routing_idea2/research/e17_supervision_distortion_audit/E17_COMPLETE_RESULTS.json`、`PERMUTATION_SUMMARIES.json`、`LOSS_ORDER_RESULTS.json` |
| E18 | `routing_idea2/research/e18_confirmatory_execution/final_results/E18_POINT_RESULTS.json` 及同目录表格 |
| E19 | `routing_idea2/research/e19_symmetric_tie_constraint/E19_COMPLETE_RESULTS.json`、`E19_PROTOCOL.json`、`FROZEN_TSPO_REUSE_AUDIT.json` |
| E20 / P2 | `routing_idea2/research/e20_p2_execution/P2_POINT_RESULTS.json`、`P2_FINAL_DIAGNOSTICS.json`、执行账本 |
| E21 / P3 | `routing_idea2/research/e21_p3_external_mechanism/P3_INTERNAL_LISTNET_RESULTS.json`、`P3_EXTERNAL_DARS_RESULTS.json` |
| E22 | `routing_idea2/research/e22_mechanism_specificity/RESULTS.json`、`SUMMARY.json`、`controls.csv`、`geometry.csv` |
| E23 | `routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json`；准备阶段 `RESULTS.json`、`PROTOCOL.json` 与 `FILE_INDEX.json` 分别保留 |

这些文件能够支持数字、结论和来源的交叉核对。只有脱敏汇总时，无法从零验证原始答案的判分，也无法恢复每题的训练或选队过程。

## 2. 环境应按实际用途恢复

不要把某一阶段的 requirements 当作所有阶段通用的环境锁。

| 用途 | 已记录的主要依赖 | 依据 |
|---|---|---|
| E17–E22 数值头及凸优化 | Python 3.11；历史 E17 为 3.11.14；NumPy 1.26.4、SciPy 1.11.4、scikit-learn 1.7.2、Clarabel 0.11.1 | E17 `RUNTIME_VERSIONS.json`，E19/E20 requirements |
| 原冻结 MiniLM 编码 | NumPy 1.26.4、PyTorch 2.4.1+cu121、Transformers 4.51.3、Tokenizers 0.21.4、Safetensors 0.8.0；PCA 使用 scikit-learn 1.7.2 | P2 `requirements_features.txt`、`P2_ENCODER_RECEIPT.json` |
| 历史图的本地渲染 | 原渲染脚本、Pillow 12.0.0，以及脚本指定字体/渲染依赖 | E19 requirements、各图 QA |
| 成员模型和聚合器生成 | 原服务器推理环境、固定模型与 tokenizer、原生成参数、路径映射和算子身份 | `requirements-lock.txt`、E18/P2 执行收据；仅安装 Python 包不能恢复这些资产 |

E20 的 `requirements.txt` 记录过 D 盘通用数值环境中的 Transformers 5.12.1 和 Tokenizers 0.22.2。**这两个版本不能作为恢复原 MiniLM 编码的版本。** P2 曾在编码前检测到版本不符并拒绝启动；正式编码使用 4.51.3 / 0.21.4。原说明保存在 [ENVIRONMENT_CLARIFICATION.md](../routing_idea2/research/e20_p2_execution/ENVIRONMENT_CLARIFICATION.md)。

冻结 MiniLM 为 `sentence-transformers/all-MiniLM-L6-v2`，原 revision 为 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。还需要本地模型目录、下载收据和 `INPUT_ASSETS_RECEIPT.json`。冻结编码器不同于训练路由头，不可用新的编码结果替换后继续宣称复现了原预测。

## 3. Git 归档布局不同于原执行机布局

原私有资产根通常为：

```text
E:\语言模型组合路由
```

原代码在另一工作目录下运行。提交到 Git 后，一些阶段将源码放在 `code/`，一些阶段保持阶段目录平铺；公开 JSON 通常也从原 `public/` 目录平铺到阶段根目录。

| Git 中的位置 | 原程序期望的位置或行为 |
|---|---|
| `e17_supervision_distortion_audit/code/*.py` | E17 用 `MAB_ASSET_ROOT` 和 `MAB_E17_ROOT` 定位资产与输出；还期待 `research/e14_mechanism_diagnostics/public/code/e14_common.py` |
| `e18_mechanism_and_confirmation_preparation/code/*.py` | P2 和 P3 的部分脚本把同级 E18 阶段根加入 `sys.path`，然后导入 `audit_sources`、`candidate_proposal` 或 `e18_reference_adapter`；Git 中这些模块实际位于 `code/` |
| `e18_confirmatory_execution/final_sources/code/.../` | 后续完成版本的冻结运行源码归档；不能无差别覆盖较早版本，需按对应源清单恢复 |
| E19 的 `FROZEN_TSPO_REUSE_AUDIT.json` | `load_packet()` 读取其中记录的原绝对输入包路径；仅设置 `MAB_ASSET_ROOT` 不会自动重写该字段 |
| E20 / P2 阶段根源码 | 数值拟合和部分输入程序支持 `MAB_ASSET_ROOT`；收集、打包和部分调度脚本仍使用原 Windows 或服务器路径 |
| E21 `run_external.py` | 有 `--asset-root`；仍依赖原 E18 输入投影模块，以及冻结 MiniLM 收据 |
| E22 `run_audit.py` | 源码中的 `ASSETS` 直接固定为原 Windows 路径，复用 E19 和 P3 的原私有工件位置；并非任意目录下一条命令即可运行 |

恢复应在隔离的资产副本中进行，显式记录“Git 归档路径 → 原逻辑路径 → 当前物理路径”的映射。**保持冻结源码、协议和来源哈希不变。** 若需要跨平台适配，应把适配器、映射及其验证结果另行记录，不可静默改写协议路径、删除哈希检查或把修改后的源码当作历史原件。

旧协议中的历史分支名、绝对路径和源码哈希是原运行的来源证据。当前 Git 分支改名不要求篡改这些冻结字段；新分支与旧分支的关系由归档映射说明。

## 4. 主线仍依赖的少量早期文件

E1–E16 的完整实验无需留在主线工作目录，但 E17 的直接导入及来源检查要求保留以下五项：

| 保留的 Git 路径 | 用途 |
|---|---|
| `routing_idea2/research/e14_mechanism_diagnostics/code/e14_common.py` | E17 导入的数组哈希、NPZ、JSON 和序列化辅助函数 |
| `routing_idea2/research/e13_partial_combination_feedback/E13_PROTOCOL.json` | 原输入与评分矩阵的身份、五折输入包来源 |
| `routing_idea2/research/e8_m4_mechanism_audit/M4_MECHANISM_PROTOCOL.json` | E17 原 M4 机制来源检查 |
| `routing_idea2/research/e8_m4_mechanism_audit/code/run_distribution_audit.py` | E17 检查该历史源码的哈希；主线没有直接导入执行它 |
| `routing_idea2/research/m4_noise_aware_set_learning/NS_SPO_PROTOCOL.json` | E17 相同后验抽样对照的来源协议 |

这些文件是兼容和来源依赖，不等同于完整保留了 E8/E13/E14 的训练系统。若需要重新建立早期私有缓存，须回到相应历史分支，并恢复其私有输入和运行记录。

主线保留的 `src`、`scripts`、`configs`、`protocol`、`data/cache_audit`、`tests` 与依赖锁是共享基础代码。部分旧脚本仍要求历史阶段的结果或私有目录；保留脚本不意味着它们可以在缺少相应资产时运行。

## 5. 必须恢复的私有工件

完整机器清单位于 [private_artifacts.json](../reproducibility/private_artifacts.json)。其中 SHA-256 均来自已提交的代码或收据；本次整理未打开对应私有文件验证其当前存在性。

### 原 train600 与五折输入

数值主线的核心输入为：

- `artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/selector_train_inputs.npz`：600×384 原问题向量，以及题号、规范化哈希、任务和子任务元数据。
- `artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_outcomes.npz`：形状 3×600×20 的二值组合评分，生成 seed 顺序为 42、137、2024。
- `research/e13_partial_combination_feedback/private/inputs/fold{0..4}/INPUTS_ONLY.npz`：各折 H480/E120、17 维预处理特征及 PCA/scaler 状态。

E19 的 [FROZEN_TSPO_REUSE_AUDIT.json](../routing_idea2/research/e19_symmetric_tie_constraint/FROZEN_TSPO_REUSE_AUDIT.json) 记录五个输入包的确切 SHA、历史 M4 日志和预处理一致性。不能只重新随机划分五折并把所得结果当作原实验复算。

E17 还使用原 M0/M4 权重、lexicographic/reverselex/uniform 参考头、NS 后验抽样与冻结头。对应路径和来源清单在机器清单中列出。

### 最终模型与独立确认

E20 最终 B/C 模型文件为：

```text
research/e20_independent_confirmation_preparation/private/final_models/FINAL600_MODELS_FROZEN.npz
```

其已记录 SHA-256 为：

```text
b8b786eeb0aaac6f869136f5353186c13f78aba0a716ebf0bcad145532d01f92
```

文件包含两支 17×20 权重及 train600 的 PCA/scaler 状态。重新拟合时还会核对原 E17 `FULL600_FIT_LOG.json` 与公开冻结配置，不能仅凭报告中的正则数值跳过来源验证。

P2 的重算或复评分还需要原 2,400 题输入、冻结得分与概率、执行 scope、完整付费输出、答案来源定位和相应收据。仅有 `P2_POINT_RESULTS.json` 时，可以复核已发布汇总，不能重新判定原生回答是否正确。

E18 的最终点结果重算要求原有效题奖励 NPZ 和完成结果 JSON；该阶段有 2 道不可评分题。不得为缺失项补零。准确文件名与已记录哈希见机器清单及原 [final_results/REPRODUCE.md](../routing_idea2/research/e18_confirmatory_execution/final_results/REPRODUCE.md)。

### 外部 DARS 与模型缓存

P3 外部实验使用 [DARS 固定 revision](https://huggingface.co/datasets/AIGNLAI/DARS/tree/d7bdd3638d263986f32ee928a19e0e2868e69bab) 的 GPQA、MATH-500 各自 train/test 评分缓存。四个文件 SHA 在 `run_external.py` 和机器清单中。它复用评分缓存，不需要生成新的模型回答；仍需要冻结 MiniLM 与相匹配的编码环境。

成员模型、聚合器的权重和 tokenizer 不在 Git 中。模型名称、配置或物理目录名相同不足以证明身份相同，应核对原配置哈希、模型指纹、tokenizer、聚合提示、判分器及算子锁。原六成员配置和聚合器身份依据见 E18 [OPERATOR_IDENTITY_QUALIFICATION.json](../routing_idea2/research/e18_mechanism_and_confirmation_preparation/OPERATOR_IDENTITY_QUALIFICATION.json)。

## 6. E22 快照中已发现的源码缺口

下列四个文件在原 E18/P2 执行适配器中被直接导入或校验，但未出现在 E22 全树中。进一步检查了整理前 33 个分支的完整 HEAD 树，以及这些分支可达历史的 53 个提交；四个目标在仓库根目录和 `routing_idea2/` 前缀下的指定路径历史查询均为空。该检查不覆盖无引用的已删除分支、不可达提交或原执行机。完整证据见 [源码恢复审计](../reproducibility/source_recovery_audit.json)。

| 原资产相对路径 | 冻结 SHA-256 |
|---|---|
| `scripts/v2/typed_aggregation_execution.py` | `acbc9cd911635f26da0e23cce4174bf6660a7604d29a5d7224f1df08e083f368` |
| `scripts/v2/typed_operator_contract.py` | `c9c3c7562ea07e9d32074e5ee67c551a458faa6ad34c17d521659c6abf379767` |
| `scripts/v2/stage1_review_common.py` | `2f9ef59c1e93026361343118bce38f25e8646177fad1038986954a0eb4093ab2` |
| `src/budget_router_v2/scoring.py` | `6fc1d2f3cc81b3e954350143abd1071d6556e1cfe4f321d9333f2016baf2150d` |

依据为 E18 `OPERATOR_IDENTITY_QUALIFICATION.json` 的 `dependency_sources`，以及 `p2_execution_adapter.py` 的 `load_original_apis()`。现有 `src/budget_router/scoring.py` 不是该 V2 评分器的替代品。

原聚合器准备还涉及未出现在同一快照中的 `scripts/v2/prepare_typed_aggregator_candidate.py`。原执行链另外要求完整 `OPERATOR_FREEZE_LOCK_v1.json`；使用跨机器模型路径映射时，还需要经过原哈希校验的 `mapped_backend_v1.py` 与映射收据。机器清单区分了这些条件依赖。

恢复应优先寻找原执行工作区或历史快照中的冻结字节，并逐项核对记录的 SHA。若无法恢复，应保持该缺口明确可见；不能编写近似替代实现后把它标记为原实验完整复现。

## 7. 私有资产恢复后的阶段入口

以下列出的是已归档的真实入口及先后关系，**本次整理没有执行这些入口，也没有验证当前机器具备它们的全部输入。** 先按各阶段原 `REPRODUCE.md`、私有收据和路径映射完成恢复，再使用全新的输出目录。

| 目的 | 原入口与顺序 | 额外条件 |
|---|---|---|
| E17 编号与算子对照 | `train_e17.py prepare` → 验证器封存 → `train_e17.py freeze` → 资格检查 → `evaluate_e17.py` | 原早期模型/后验缓存、E13 五折输入与评分；完整顺序见 E17 REPRODUCE |
| E19 B/C 点结果 | `run_e19.py --phase p0` → 通过 P0 后 `--phase p1` | 提供冻结协议及其真实 SHA、隔离 study root、原五折输入 |
| E20 最终头 | `fit_final_models.py` 的 fit 阶段，再按原输入算子预测 | 原 train600、E17 最终拟合来源日志；输入与输出身份保持 |
| P2 缓存投影 | `blind_evaluation.py` 的显式投影阶段 | 已完成、已冻结的稀疏组合评分和策略概率；不重新选队 |
| P2 从原生输出复评分 | `score_paid_outputs.py` | 完整输出先冻结；原 V2 判分源码、批准范围及参考定位已恢复 |
| P3 内部/外部 | `run_internal.py` 或 `run_external.py`：fit → evaluate | 同一协议 SHA；预测先冻结，评价后读取；外部四份 DARS 缓存按 SHA 核对 |
| E22 | `run_audit.py`：train → evaluate；再 `summarize_audit.py` | 复用 E19/P3 私有端点；总结脚本也会读私有逐题评分，并非仅从 SUMMARY.json 重排表格 |

历史 `e18_1_statistics_and_mechanisms/analyze_e18_1.py` 和 `e19_symmetric_tie_constraint/analyze_e19.py` 包含 bootstrap 统计流程。它们作为历史源码保留，**不属于当前公开核验或默认重算命令**。本次没有重新计算或新增此类统计。

封存的 sealed200 答案不用于公开核验、表格重建、旧 train600 机制实验或已经完成的 P2 确认。P2 的数据准入使用既有授权的单向身份指纹。fresh400 已有 E18 曝光历史，恢复文件不能改变其开发数据身份。

## 8. 复现状态的准确表述

可以报告：“论文主线的冻结源代码、协议、脱敏结果及公开核验入口已归档；私有工件的恢复路径、格式、已记录哈希和缺失源码已清单化。”

在私有资产和缺失源码尚未恢复、完整链条尚未实际验证前，不应报告：“单独克隆仓库即可从零重现全部模型调用与论文结果。” 验证公开汇总与重新得到原始科学结果应分别记录。

## 9. E23 部署规则与同题池参照补充

E23 使用原 P2 的 2,400 道题、原 B/C 模型和原评分器。它在 P2 结果已曝光后增加参照策略，属于补充分析。E23 没有增加独立新题，也没有把 P2 改成新的未使用测试集。此次仓库维护只核对公开文件、代码依赖和已记录哈希，没有重新运行 E23。

### 两份结果文件各自保留

[RESULTS.json](../routing_idea2/research/e23_deployment_prior_and_references/RESULTS.json) 是补齐调用前的准备结果：四个控制头已拟合，十二份策略已冻结；原 B/C、训练先验及 raw-exact 六份策略已有完整评分，GlobalFixed、Ridge、Uniform 和三个 Matched 尚有缺失调用。

[RESULTS_FINAL.json](../routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json) 是补齐后的最终结果：复用原 106,536 条聚合评分，新增 1,770 次成员生成及 4,263 次聚合，十二份策略均覆盖完整 2,400 题。两份结果的策略 SHA、原评分 SHA、四项原配对比较和部署差值分解一致。最终文件中的 `new_llm_calls=6033` 是该补齐阶段的总数；`new_llm_calls_during_finalization=0` 表示最后的评分汇总入口没有调用语言模型。两处字段记录的阶段不同。

**最终性能表使用 `RESULTS_FINAL.json`。四个新增控制头的求解日志、原缓存缺口和拟合耗时仍从准备阶段 `RESULTS.json` 读取。** 不用最终文件覆盖准备文件，也不把准备协议中“尚未进行新物理调用”的说明当作完成阶段的状态。

[FILE_INDEX.json](../routing_idea2/research/e23_deployment_prior_and_references/FILE_INDEX.json) 列出 33 份其他文件，加上索引自身共 34 份。此次逐项核对的文件长度、SHA-256 与 Git blob SHA-1 均匹配。索引中的原分支名称和 E22 parent commit 保留为原记录。

### 数值设置与依赖

E23 新拟合的是四个全 train600 TSPO 控制头，不是 E22 五折 OOF 模型。它导入 E22 的 `row_weighted.fit` 与 `run_audit.controls`，使用 `final600` 盐值、三个已定种子、原任务/子任务与特征范数分组。四个头使用 `lambda=0.003472222222222222`、600 行及 `1/600` 分母，包含截距的全部系数受 L2 惩罚，行权重不重新归一化。保存的日志记录 Clarabel 0.11.1、三个主容差均为 `1e-10`、最多 1,000 次迭代和单线程。

B/C 从 E20 原模型恢复，并要求逐头 Fortran-order 矩阵乘所得分数与原冻结分数完全一致。Ridge 从 E17 `FULL600_MODELS.npz` 的第 0 个头恢复，使用该文件自己的 PCA/scaler，原 alpha 为 `333.3333333333333`。E23 没有重新选择这些正则参数，也没有提供各方法相同调参机会后的最优性能比较。

数值运行仍需原 Python 3.11、NumPy、SciPy、scikit-learn 和 Clarabel 环境。E23 没有新增一份完整包版本锁；不能仅从这个阶段的短依赖说明恢复全部数值与推理环境。物理生成沿用原 P2 的模型、tokenizer、NF4 加载、生成参数和服务器运行记录。

### 新增私有文件清单

[private_artifacts.json](../reproducibility/private_artifacts.json) 追加了 20 项 E23 恢复条目；原 25 项资产的路径、身份和哈希保持不变。主要分为以下几组：

| 用途 | 原文件或目录 | 哈希来源 |
|---|---|---|
| 冻结参照策略 | E17 `private/confirmation_input_only/FULL600_MODELS.npz`、P2 `private/P2_PREDICTIONS_FROZEN.npz` | `analyze_references.py` 固定 SHA |
| 复用原 P2 评分 | P2 `private/final_evaluation_v1/PAID_SUPPORT_SCORES.jsonl`，原效用数组和评分/答案账户记录 | `RESULTS.json`、`RESULTS_FINAL.json` 和原 P2 收据 |
| 固定策略与区域 | 原 E23 `--root` 下 `private/POLICIES_FROZEN.npz`、`REGIONS_FROZEN.npz`、`FREEZE.json`、`MISSING_KEYS.json` | 公开策略 SHA 和私有 `FREEZE.json` |
| 补齐执行身份 | 原远端根下 `inputs/EXECUTION_SCOPE.json`、`inputs/AUTHORIZATION.json`、`EXECUTION_PLAN.json` | `RESULTS_FINAL.json` 与 `EXECUTION_AUTHORIZATION.json` |
| 原生输出 | 原远端 `exports/OUTPUTS_FROZEN.json` 及两份 `*_NATIVE_RECEIPTS.jsonl` | 公开冻结收据 SHA；逐个 JSONL 的 SHA 和长度在该私有收据中 |
| 复用新增评分 | 原 `--output-root` 下 `SUPPLEMENT_PAID_SUPPORT_SCORES.jsonl`、`FINAL_POLICY_UTILITIES_PRIVATE.npz` | 公开新增评分 SHA；效用数组需原私有目录收据 |
| 核对实际执行 | 原 P2 registry、成功记录和任务状态，E23 原计划、分片/迁移收据、journal、导出证据 | 原记录中的路径和哈希引用 |

原 E23 分析根与评分输出根由参数指定，公开文件没有给出统一的资产根内位置。清单对此使用 `asset_relative_path: null` 并记录相对于运行根的文件名，不能把它们当作已经恢复的文件。JSON 文件的原始字节 SHA 与规范化 JSON 摘要也分别记录，不能互换。公开 `EXECUTION_AUTHORIZATION.json` 是摘要，不是私有 `AUTHORIZATION.json` 的替代文件。

私有资产检查脚本仍只自动核对已知具体路径的单文件 SHA。调用者路径、目录、模式条目及没有公开单文件 SHA 的项目仍需按原收据核对，不能把出现同名目录当作恢复成功。

### 原执行路径仍需恢复

`analyze_references.py` 与 `package_execution.py` 直接固定资产根为 `E:\语言模型组合路由`，不读取 `MAB_ASSET_ROOT`。打包程序另外要求原代码工作区中的 `.e20_p2_cpu_runtime/OUTPUTS_FROZEN.json`。服务器配置固定读取 `/data2/chendong/e20_p2_execution`，并要求 E23 写根等于 `/data2/chendong/e23_reference_completion`；启动程序固定使用 `/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python`。这些都是原执行路径，不能把参数占位符理解为任意机器可直接运行。

E23 的 `supplement_adapter.py`、`supplement_worker.py` 分别校验原 P2 adapter、member worker 和 aggregation worker 的源码 SHA。这三份主线文件的字节均与所需 SHA 一致。其后仍调用原 P2 执行和评分接口，因此第 6 节列出的四份缺失源码仍是缺口，E23 没有补回它们。

此外，最终入口通过 P2 `score_paid_outputs.load_frozen_apis()` 读取 E18 reference、projection 和 scoring 模块；Git 把这些源码保存在 E18 preparation 的 `code/` 中，原入口却要求阶段根目录平铺。`collect_native_exports.load_auditor()` 要求 `e18_confirmatory_execution/audit_closed_native_archives.py`，原件保存在该阶段 `final_sources/code/e18_confirmatory_execution/` 下。需要按冻结源版本恢复原布局，不能删除哈希检查或用近似代码替代。

### 可核对与可重算的范围

公开结果足以核对最终表格、任务/seed/子任务分解、成本和已记录的来源哈希；无法只凭公开 JSON 检查原生答案的判分。恢复冻结策略及原、新逐动作评分后，可以按原规则重算效用和纠错破坏；只有 `FINAL_POLICY_UTILITIES_PRIVATE.npz` 时，能检查策略准确率，不能恢复全部动作耦合下的纠错破坏计数。

[原 REPRODUCE.md](../routing_idea2/research/e23_deployment_prior_and_references/REPRODUCE.md) 记录了实际阶段顺序。`analyze_references.py --phase freeze` 会拟合四个控制头并新建私有工件；`--phase analyze` 读取原 P2 逐动作评分；`finalize_results.py` 会验证原生输出、再次解码 840 道原 P2 题的已授权参考答案，并调用原评分器评分 4,263 个新增聚合输出。**`finalize_results.py` 不是只读取公开汇总的表格整理程序。** 当前维护没有执行这些入口，也不读取 sealed200 答案。

