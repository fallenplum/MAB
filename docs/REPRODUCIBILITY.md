# 复现范围、依赖与私有工件

本文件针对论文主线 E17–E22，审查基准为提交 `8f490a06ebe1153be2fcbeb42ad3944d63075b7d`。主线保留冻结源码、协议、脱敏结果和来源记录；较早实验的完整记录保存在历史研究分支。

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

