# 复现说明与所需文件

本文件针对论文主线 E17–E23。E17–E22 的审查基准为提交 `8f490a06ebe1153be2fcbeb42ad3944d63075b7d`；E23 补充审查基准为提交 `3958f9ceaac5b0b1de88bbd47d89ddd940833ff2`，具体见第 9 节。主线保留冻结源码、协议、脱敏结果和来源记录；较早实验的完整记录保存在历史研究分支。

**克隆仓库后，可以核对公开文件和论文表格。恢复缓存后重算、重新训练、重新生成语言模型回答，还需要相应的私有数据、权重、原生输出和运行环境。** 私有文件的恢复路径按原执行工作区登记；原执行程序另需四份尚待找回的冻结源码。所需文件、缺失项及已记录的 SHA-256 见 [private_artifacts.json](../reproducibility/private_artifacts.json)。

本次整理没有重新训练路由器、调用语言模型、访问封存答案、重新评分或计算新的置信区间与 bootstrap。

## 1. 从仓库即可进行的检查

从仓库根目录运行：

```bash
python scripts/verify_repository.py
python scripts/build_paper_tables.py --check
```

第一项检查主线文件、来源与完整性；第二项核对论文表格与已发布的汇总结果。两项命令只使用公开文件。私有运行环境的恢复和各阶段实验重跑见第 5–7 节。

冻结文件的清单为 [frozen_manifest.json](../reproducibility/frozen_manifest.json)，逐项记录 Git blob SHA、SHA-256、大小和模式。`.gitattributes` 要求 Git 保留原始目录的字节，避免跨平台检出时改变已冻结的行尾。自动核验通过语法解析检查历史 Python 源码；实验入口在资产恢复后单独运行。

恢复私有工件后，可以检查清单中已有的文件哈希，例如：

```bash
python scripts/check_private_artifacts.py --asset-root /path/to/restored/assets --dars-root /path/to/frozen/DARS
```

此命令执行文件校验，日志记录校验结果。缺失、哈希不符、需要人工盘点的通配路径，以及未提供的 DARS 根目录都会返回 `INCOMPLETE`（退出码 2）；命令行参数解析错误也返回 2，清单或资产根路径错误返回 1。清单中的 `model_cache` 和 `protected_data` 两节由人工核对：按原阶段收据检查模型快照指纹、目录资产、路径映射和运行环境。单文件哈希用于确认文件版本；数值复算还需运行对应阶段并核对输出。

公开主结果的入口包括：

| 阶段 | 公开结果文件 |
|---|---|
| E17 | `routing_idea2/research/e17_supervision_distortion_audit/E17_COMPLETE_RESULTS.json`、`PERMUTATION_SUMMARIES.json`、`LOSS_ORDER_RESULTS.json` |
| E18 | `routing_idea2/research/e18_confirmatory_execution/final_results/E18_POINT_RESULTS.json` 及同目录表格 |
| E19 | `routing_idea2/research/e19_symmetric_tie_constraint/E19_COMPLETE_RESULTS.json`、`E19_PROTOCOL.json`、`FROZEN_TSPO_REUSE_AUDIT.json` |
| E20 / P2 | `routing_idea2/research/e20_p2_execution/P2_POINT_RESULTS.json`、`P2_FINAL_DIAGNOSTICS.json`、执行账本 |
| E21 / P3 | `routing_idea2/research/e21_p3_external_mechanism/P3_INTERNAL_LISTNET_RESULTS.json`、`P3_EXTERNAL_DARS_RESULTS.json` |
| E22 | `routing_idea2/research/e22_mechanism_specificity/RESULTS.json`、`SUMMARY.json`、`controls.csv`、`geometry.csv` |
| E23 | `routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json`；准备阶段 `RESULTS.json`、`PROTOCOL.json` 与 `FILE_INDEX.json` 分别保留 |

这些文件用于交叉核对数字、结论和来源。复算答案判分、逐题训练或组合选择，还需对应的原生输出、参考答案、逐题矩阵和权重。

## 2. 环境应按实际用途恢复

各阶段按自己的运行记录恢复环境。下表按数值拟合、编码、绘图和生成分别列出依赖。

| 用途 | 已记录的主要依赖 | 依据 |
|---|---|---|
| E17–E22 数值头及凸优化 | Python 3.11；历史 E17 为 3.11.14；NumPy 1.26.4、SciPy 1.11.4、scikit-learn 1.7.2、Clarabel 0.11.1 | E17 `RUNTIME_VERSIONS.json`，E19/E20 requirements |
| 原冻结 MiniLM 编码 | NumPy 1.26.4、PyTorch 2.4.1+cu121、Transformers 4.51.3、Tokenizers 0.21.4、Safetensors 0.8.0；PCA 使用 scikit-learn 1.7.2 | P2 `requirements_features.txt`、`P2_ENCODER_RECEIPT.json` |
| 历史图的本地渲染 | 原渲染脚本、Pillow 12.0.0，以及脚本指定字体/渲染依赖 | E19 requirements、各图 QA |
| 成员模型和聚合器生成 | 原服务器推理环境、固定模型与 tokenizer、原生成参数、路径映射和算子身份 | `requirements-lock.txt`、E18/P2 执行收据；同时恢复模型文件、路径映射和生成配置 |

E20 的 `requirements.txt` 记录过 D 盘通用数值环境中的 Transformers 5.12.1 和 Tokenizers 0.22.2。这两个版本属于该通用数值环境。P2 曾在编码前检测到版本不符并停止启动；**原 MiniLM 的正式编码版本为 4.51.3 / 0.21.4。**原说明保存在 [ENVIRONMENT_CLARIFICATION.md](../routing_idea2/research/e20_p2_execution/ENVIRONMENT_CLARIFICATION.md)。

冻结 MiniLM 为 `sentence-transformers/all-MiniLM-L6-v2`，原 revision 为 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。还需要本地模型目录、下载收据和 `INPUT_ASSETS_RECEIPT.json`。编码器和路由预测头分别恢复；复算原预测时使用冻结编码器生成的问题向量。

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
| `e18_confirmatory_execution/final_sources/code/.../` | 后续完成版本的运行源码归档；按各阶段对应的源清单恢复版本 |
| E19 的 `FROZEN_TSPO_REUSE_AUDIT.json` | `load_packet()` 读取其中记录的原绝对输入包路径；仅设置 `MAB_ASSET_ROOT` 不会自动重写该字段 |
| E20 / P2 阶段根源码 | 数值拟合和部分输入程序支持 `MAB_ASSET_ROOT`；收集、打包和部分调度脚本仍使用原 Windows 或服务器路径 |
| E21 `run_external.py` | 有 `--asset-root`；仍依赖原 E18 输入投影模块，以及冻结 MiniLM 收据 |
| E22 `run_audit.py` | 源码中的 `ASSETS` 直接固定为原 Windows 路径，复用 E19 和 P3 的原私有工件位置；运行前需要恢复这些路径及文件 |

在独立资产副本中恢复，并记录“Git 归档路径 → 原逻辑路径 → 当前物理路径”的映射。**冻结源码、协议和来源哈希保留原版本。** 跨平台适配代码、路径映射及验证结果单独保存，运行时继续执行原哈希检查。

旧协议中的历史分支名、绝对路径和源码哈希记录了原运行环境，保留原字段。新旧分支名称的对应关系见分支归档映射。

## 4. 主线仍依赖的少量早期文件

E1–E16 的完整源码、报告和结果保存在各历史实验分支。主线为 E17 的直接导入及来源检查保留以下五项：

| 保留的 Git 路径 | 用途 |
|---|---|
| `routing_idea2/research/e14_mechanism_diagnostics/code/e14_common.py` | E17 导入的数组哈希、NPZ、JSON 和序列化辅助函数 |
| `routing_idea2/research/e13_partial_combination_feedback/E13_PROTOCOL.json` | 原输入与评分矩阵的身份、五折输入包来源 |
| `routing_idea2/research/e8_m4_mechanism_audit/M4_MECHANISM_PROTOCOL.json` | E17 原 M4 机制来源检查 |
| `routing_idea2/research/e8_m4_mechanism_audit/code/run_distribution_audit.py` | E17 检查该历史源码的哈希；主线没有直接导入执行它 |
| `routing_idea2/research/m4_noise_aware_set_learning/NS_SPO_PROTOCOL.json` | E17 相同后验抽样对照的来源协议 |

这五项供 E17 导入和检查来源。重建 E8/E13/E14 的私有缓存时，使用相应历史分支的完整训练代码，并恢复该阶段的私有输入和运行记录。

主线保留的 `src`、`scripts`、`configs`、`protocol`、`data/cache_audit`、`tests` 与依赖锁是共享基础代码。运行旧脚本前，先恢复其依赖的历史结果和私有目录。

## 5. 必须恢复的私有工件

完整机器清单位于 [private_artifacts.json](../reproducibility/private_artifacts.json)。其中 SHA-256 均按已提交的代码或运行收据登记；恢复时需要逐项核对实际文件。

### 原 train600 与五折输入

数值主线的核心输入为：

- `artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/selector_train_inputs.npz`：600×384 原问题向量，以及题号、规范化哈希、任务和子任务元数据。
- `artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_outcomes.npz`：形状 3×600×20 的二值组合评分，生成 seed 顺序为 42、137、2024。
- `research/e13_partial_combination_feedback/private/inputs/fold{0..4}/INPUTS_ONLY.npz`：各折 H480/E120、17 维预处理特征及 PCA/scaler 状态。

E19 的 [FROZEN_TSPO_REUSE_AUDIT.json](../routing_idea2/research/e19_symmetric_tie_constraint/FROZEN_TSPO_REUSE_AUDIT.json) 记录五个输入包的确切 SHA、历史 M4 日志和预处理一致性。原实验复算使用所列五个输入包、原划分和预处理状态。

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

文件包含两支 17×20 权重及 train600 的 PCA/scaler 状态。重新拟合时，同时核对原 E17 `FULL600_FIT_LOG.json`、公开冻结配置和输入文件。

P2 的重算或复评分还需要原 2,400 题输入、冻结得分与概率、执行 scope、完整付费输出、答案来源定位和相应收据。`P2_POINT_RESULTS.json` 用于复核已发布汇总；原生回答的判分复算使用上述输出、答案定位和原评分器。

E18 的最终点结果重算要求原有效题奖励 NPZ 和完成结果 JSON；该阶段的 2 道不可评分题沿用原排除处理。准确文件名与已记录哈希见机器清单及原 [final_results/REPRODUCE.md](../routing_idea2/research/e18_confirmatory_execution/final_results/REPRODUCE.md)。

### 外部 DARS 与模型缓存

P3 外部实验使用 [DARS 固定 revision](https://huggingface.co/datasets/AIGNLAI/DARS/tree/d7bdd3638d263986f32ee928a19e0e2868e69bab) 的 GPQA、MATH-500 各自 train/test 评分缓存。四个文件 SHA 在 `run_external.py` 和机器清单中。该实验直接读取现有评分缓存，并使用冻结 MiniLM 及相匹配的编码环境。

成员模型、聚合器的权重和 tokenizer 不在 Git 中。恢复时通过原配置哈希、模型指纹、tokenizer、聚合提示、判分器及算子锁核对模型版本。原六成员配置和聚合器身份依据见 E18 [OPERATOR_IDENTITY_QUALIFICATION.json](../routing_idea2/research/e18_mechanism_and_confirmation_preparation/OPERATOR_IDENTITY_QUALIFICATION.json)。

## 6. 尚待恢复的冻结源码

下列四个文件在原 E18/P2 执行适配器中被直接导入或校验，但未出现在 E22 全树中。进一步检查了整理前 33 个分支的完整 HEAD 树，并查询这些分支 53 个可达提交中的指定路径历史；四个目标在仓库根目录和 `routing_idea2/` 前缀下的查询均为空。无现存引用的其他历史记录及原执行机尚待查找。完整记录见 [源码恢复审计](../reproducibility/source_recovery_audit.json)。

| 原资产相对路径 | 冻结 SHA-256 |
|---|---|
| `scripts/v2/typed_aggregation_execution.py` | `acbc9cd911635f26da0e23cce4174bf6660a7604d29a5d7224f1df08e083f368` |
| `scripts/v2/typed_operator_contract.py` | `c9c3c7562ea07e9d32074e5ee67c551a458faa6ad34c17d521659c6abf379767` |
| `scripts/v2/stage1_review_common.py` | `2f9ef59c1e93026361343118bce38f25e8646177fad1038986954a0eb4093ab2` |
| `src/budget_router_v2/scoring.py` | `6fc1d2f3cc81b3e954350143abd1071d6556e1cfe4f321d9333f2016baf2150d` |

依据为 E18 `OPERATOR_IDENTITY_QUALIFICATION.json` 的 `dependency_sources`，以及 `p2_execution_adapter.py` 的 `load_original_apis()`。现有 `src/budget_router/scoring.py` 属于 V1；原适配器要求上表中的 V2 文件及对应 SHA。

原聚合器准备还涉及未出现在同一快照中的 `scripts/v2/prepare_typed_aggregator_candidate.py`。原执行链另外要求完整 `OPERATOR_FREEZE_LOCK_v1.json`；使用跨机器模型路径映射时，还需要经过原哈希校验的 `mapped_backend_v1.py` 与映射收据。机器清单区分了这些条件依赖。

从原执行工作区或历史快照取回原文件，并逐项核对记录的 SHA。这四项目前标为待恢复；补齐原文件并通过哈希校验后，再更新恢复状态。

## 7. 私有资产恢复后的阶段入口

下表列出各阶段入口及执行顺序。运行前，按各阶段原 `REPRODUCE.md`、私有收据和路径映射恢复所需输入、源码和环境，并使用新的输出目录。

| 目的 | 原入口与顺序 | 额外条件 |
|---|---|---|
| E17 编号与算子对照 | `train_e17.py prepare` → 验证器封存 → `train_e17.py freeze` → 资格检查 → `evaluate_e17.py` | 原早期模型/后验缓存、E13 五折输入与评分；完整顺序见 E17 REPRODUCE |
| E19 B/C 点结果 | `run_e19.py --phase p0` → 通过 P0 后 `--phase p1` | 提供冻结协议及其真实 SHA、隔离 study root、原五折输入 |
| E20 最终头 | `fit_final_models.py` 的 fit 阶段，再按原输入算子预测 | 原 train600、E17 最终拟合日志；核对输入与输出的题号、版本和哈希 |
| P2 缓存投影 | `blind_evaluation.py` 的显式投影阶段 | 使用已完成、已冻结的稀疏组合评分和策略概率 |
| P2 从原生输出复评分 | `score_paid_outputs.py` | 完整输出先冻结；原 V2 判分源码、批准范围及参考定位已恢复 |
| P3 内部/外部 | `run_internal.py` 或 `run_external.py`：fit → evaluate | 同一协议 SHA；预测先冻结，评价后读取；外部四份 DARS 缓存按 SHA 核对 |
| E22 | `run_audit.py`：train → evaluate；再 `summarize_audit.py` | 复用 E19/P3 私有文件；汇总脚本还需要私有逐题评分 |

历史 `e18_1_statistics_and_mechanisms/analyze_e18_1.py` 和 `e19_symmetric_tie_constraint/analyze_e19.py` 包含 bootstrap 统计流程。它们作为历史源码保留。**当前公开核验使用第 1 节的命令；后续汇总按现有约定报告点值，不运行新的置信区间或 bootstrap。**

sealed200 答案保持封存。公开核验、表格重建、旧 train600 机制实验与已完成的 P2 确认使用各自已有输入；P2 的数据准入只使用既有授权的单向身份指纹。fresh400 已在 E18 曝光，沿用开发数据的角色。

## 8. 当前复现状态

论文主线的冻结源码、协议、脱敏结果和公开核验脚本已归档。私有工件清单列出了恢复路径、文件格式、已记录哈希及缺失源码；原始实验重跑仍需补齐这些文件和运行环境。

公开文件核验与实验重跑分别记录结果：前者核对文件和汇总表的一致性，后者记录恢复的输入版本、实际执行入口、输出和与原结果的比较。

## 9. E23 部署规则与同题池参照补充

E23 使用原 P2 的 2,400 道题、原 B/C 模型和原评分器。它在 P2 结果已曝光后，使用同一批题增加参照策略，属于后续补充分析。此次仓库维护只核对公开文件、代码依赖和已记录哈希，没有重新运行 E23。

### 两份结果文件各自保留

[RESULTS.json](../routing_idea2/research/e23_deployment_prior_and_references/RESULTS.json) 是补齐调用前的准备结果：四个控制头已拟合，十二份策略已冻结；原 B/C、训练先验及 raw-exact 六份策略已有完整评分，GlobalFixed、Ridge、Uniform 和三个 Matched 尚有缺失调用。

[RESULTS_FINAL.json](../routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json) 是补齐后的最终结果：复用原 106,536 条聚合评分，新增 1,770 次成员生成及 4,263 次聚合，十二份策略均覆盖完整 2,400 题。两份结果的策略 SHA、原评分 SHA、四项原配对比较和部署差值分解一致。最终文件中的 `new_llm_calls=6033` 是该补齐阶段的总数；`new_llm_calls_during_finalization=0` 表示最后的评分汇总入口没有调用语言模型。两处字段记录的阶段不同。

**最终性能表使用 `RESULTS_FINAL.json`。四个新增控制头的求解日志、原缓存缺口和拟合耗时从准备阶段 `RESULTS.json` 读取。** 两份文件分别保留补齐前后的状态，准备协议中的调用说明按当时阶段阅读。

[FILE_INDEX.json](../routing_idea2/research/e23_deployment_prior_and_references/FILE_INDEX.json) 列出 33 份其他文件，加上索引自身共 34 份。此次逐项核对的文件长度、SHA-256 与 Git blob SHA-1 均匹配。索引中的原分支名称和 E22 parent commit 保留为原记录。

### 数值设置与依赖

E23 新拟合四个全 train600 TSPO 控制头；E22 的控制保留原五折 OOF 模型。它导入 E22 的 `row_weighted.fit` 与 `run_audit.controls`，使用 `final600` 盐值、三个已定种子、原任务/子任务与特征范数分组。四个头使用 `lambda=0.003472222222222222`、600 行及 `1/600` 分母，包含截距的全部系数受 L2 惩罚，行权重不重新归一化。保存的日志记录 Clarabel 0.11.1、三个主容差均为 `1e-10`、最多 1,000 次迭代和单线程。

B/C 从 E20 原模型恢复，并要求逐头 Fortran-order 矩阵乘所得分数与原冻结分数完全一致。Ridge 从 E17 `FULL600_MODELS.npz` 的第 0 个头恢复，使用该文件自己的 PCA/scaler，原 alpha 为 `333.3333333333333`。E23 固定使用这些原正则参数，表中比较的是这组冻结配置下的性能。

数值运行仍需原 Python 3.11、NumPy、SciPy、scikit-learn 和 Clarabel 环境。E23 沿用已有环境记录；完整数值与推理环境按对应阶段的版本锁和运行收据恢复。物理生成沿用原 P2 的模型、tokenizer、NF4 加载、生成参数和服务器运行记录。

### 新增私有文件清单

[private_artifacts.json](../reproducibility/private_artifacts.json) 追加了 20 项 E23 恢复条目；原 25 项资产的路径、版本和哈希保持不变。主要分为以下几组：

| 用途 | 原文件或目录 | 哈希来源 |
|---|---|---|
| 冻结参照策略 | E17 `private/confirmation_input_only/FULL600_MODELS.npz`、P2 `private/P2_PREDICTIONS_FROZEN.npz` | `analyze_references.py` 固定 SHA |
| 复用原 P2 评分 | P2 `private/final_evaluation_v1/PAID_SUPPORT_SCORES.jsonl`，原效用数组和评分/答案账户记录 | `RESULTS.json`、`RESULTS_FINAL.json` 和原 P2 收据 |
| 固定策略与区域 | 原 E23 `--root` 下 `private/POLICIES_FROZEN.npz`、`REGIONS_FROZEN.npz`、`FREEZE.json`、`MISSING_KEYS.json` | 公开策略 SHA 和私有 `FREEZE.json` |
| 补齐调用的执行范围 | 原远端根下 `inputs/EXECUTION_SCOPE.json`、`inputs/AUTHORIZATION.json`、`EXECUTION_PLAN.json` | `RESULTS_FINAL.json` 与 `EXECUTION_AUTHORIZATION.json` |
| 原生输出 | 原远端 `exports/OUTPUTS_FROZEN.json` 及两份 `*_NATIVE_RECEIPTS.jsonl` | 公开冻结收据 SHA；逐个 JSONL 的 SHA 和长度在该私有收据中 |
| 复用新增评分 | 原 `--output-root` 下 `SUPPLEMENT_PAID_SUPPORT_SCORES.jsonl`、`FINAL_POLICY_UTILITIES_PRIVATE.npz` | 公开新增评分 SHA；效用数组需原私有目录收据 |
| 核对实际执行 | 原 P2 registry、成功记录和任务状态，E23 原计划、分片/迁移收据、journal、导出证据 | 原记录中的路径和哈希引用 |

原 E23 分析根与评分输出根由参数指定，公开文件没有给出统一的资产根内位置。清单对此使用 `asset_relative_path: null` 并记录相对于运行根的文件名，恢复时再填入实际位置。JSON 原始字节 SHA 与规范化 JSON 摘要分别用于对应检查。公开 `EXECUTION_AUTHORIZATION.json` 保存批准摘要；执行入口还需要原私有 `AUTHORIZATION.json`。

私有资产检查脚本仍只自动核对已知具体路径的单文件 SHA。由运行者指定的路径、目录、通配条目及未公开单文件 SHA 的项目，需要按原收据逐项检查目录内容、文件长度和哈希。

### 原执行路径仍需恢复

`analyze_references.py` 与 `package_execution.py` 直接固定资产根为 `E:\语言模型组合路由`，不读取 `MAB_ASSET_ROOT`。打包程序另外要求原代码工作区中的 `.e20_p2_cpu_runtime/OUTPUTS_FROZEN.json`。服务器配置固定读取 `/data2/chendong/e20_p2_execution`，并要求 E23 写根等于 `/data2/chendong/e23_reference_completion`；启动程序固定使用 `/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python`。迁移到其他机器时，先按这些原路径建立映射，并恢复指定的推理环境。

E23 的 `supplement_adapter.py`、`supplement_worker.py` 分别校验原 P2 adapter、member worker 和 aggregation worker 的源码 SHA。这三份主线文件的字节均与所需 SHA 一致。其后仍调用原 P2 执行和评分接口，因此第 6 节列出的四份冻结源码仍需恢复。

此外，最终入口通过 P2 `score_paid_outputs.load_frozen_apis()` 读取 E18 reference、projection 和 scoring 模块；Git 把这些源码保存在 E18 preparation 的 `code/` 中，原入口却要求阶段根目录平铺。`collect_native_exports.load_auditor()` 要求 `e18_confirmatory_execution/audit_closed_native_archives.py`，原件保存在该阶段 `final_sources/code/e18_confirmatory_execution/` 下。按所需冻结版本恢复原目录布局，并保留入口中的哈希校验。

### 可核对与可重算的范围

公开结果用于核对最终表格、任务/seed/子任务分解、成本和已记录的来源哈希。策略准确率可以从 `FINAL_POLICY_UTILITIES_PRIVATE.npz` 复算；动作耦合下的纠错破坏计数还需要冻结策略及原、新逐动作评分。原生答案的判分复算另外需要输出、参考答案定位和原评分器。

[原 REPRODUCE.md](../routing_idea2/research/e23_deployment_prior_and_references/REPRODUCE.md) 记录了实际阶段顺序。`analyze_references.py --phase freeze` 会拟合四个控制头并新建私有工件；`--phase analyze` 读取原 P2 逐动作评分；`finalize_results.py` 会验证原生输出、再次解码 840 道原 P2 题的已授权参考答案，并调用原评分器评分 4,263 个新增聚合输出。**该入口用于原生输出核验和逐输出评分；公开表格重建使用 `scripts/build_paper_tables.py`。** 本次维护完成的是公开文件核对，sealed200 答案继续封存。

