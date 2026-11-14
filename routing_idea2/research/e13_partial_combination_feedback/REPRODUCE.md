# E13 部分组合反馈复现说明

## 实验范围与规范目录

本轮固定六个模型、每次选择三个成员、20个无序组合和原V2聚合器。奖励来自既有train600的真实最终聚合评分。所有方法接收同一已执行部分反馈，缺失动作没有训练标签行。

规范目录为 `E:\语言模型组合路由\research\e13_partial_combination_feedback`。公开报告、协议和聚合结果位于 `public`，模型、逐题动作、已购买观察、掩码及日志位于 `private`。实际worker源码入口位于 `D:\develop\CV\多臂老虎机\experiments\e13_partial_combination_feedback`。交付后相同代码位于规范目录的 `public\code`，可用该位置作为入口。

当前结果属于反复使用train600的开发离线回放。validation200、fresh400和sealed200载荷访问、新LLM/API调用、新gold采购均为0。

## 固定矩阵

| 维度 | 冻结设置 |
|---|---|
| 原始问题 | GSM8K与BBH各300，共600个不同问题 |
| 外折 | 原五折，每折H480与E120互斥 |
| 预算 | 每条生成轨迹累计已揭示60、120、240、300、360、480题 |
| logger种子 | 17、29、43 |
| 生成种子 | 42、137、2024，每次观察使用对应单生成二值奖励 |
| Main | 每道被揭示题执行一个冻结均匀主组合 |
| Pair | 同一主组合加一个冻结均匀单成员替换邻居 |
| 规模 | 5外折×3logger×3生成×2场景，共90条轨迹、540个预算快照 |
| 题目表示 | 原合法H480拟合的MiniLM-PCA16与标准化器 |
| 组映射 | 共同的 \(\hat g(x)=\arg\max_g\phi(x)T_M\)，T_M仅由训练输入和训练组metadata确定 |

三个生成重复与三个logger重复保留在同一原题簇中。5400条“原题×生成×logger”记录对应600个独立原题。预算B表示本轨迹的原题数，单题只购买一个或两个组合的一次生成评分。

## 学习器及权限

`Raw`只拟合已观察的逐题动作奖励。`Group`利用同一部分反馈估计层级组能力，并按共同预测组查表。`GroupResidual`在组能力上加入同一观察集合的逐题残差。`RawCentered`用于检查全局动作先验和缺失动作收缩的影响。

主结果中的 `GlobalFixed`、`TaskFixed`、`SubtaskFixed`也仅使用已观察反馈。任务与子任务基线由共同 \(\hat g(x)\) 获得预测任务身份，保持服务输入权限一致。`TaskFixedOracle`和`SubtaskFixedOracle`额外使用真实评价metadata，单独列为诊断。

`SaveRouterFixedMask`在报告中称为SaveRouter-FixedMask。它动态加载官方固定commit `250b7ef35a741c598aaf1bb264930d1e0bfd1a61` 的原始 `FixedKSparseRouter`，原样执行能力估计和残差拟合。profile固定为作者llmrouterbench，收缩40、先验Ridge10、残差Ridge200、残差缩放2、最少残差观察8。

共同logger替换作者采集器，共同外部分组替换作者分组pipeline，动作映射为20个真实三模型组合，服务采用纯质量argmax。这些是同权限适配。原作者基准数据、原采集算法及完整质量—费用前沿的数值复现属于另一个实验范围。一手核查见 `LITERATURE_AND_BASELINE_MAPPING.md`。

## 依赖与私有输入

使用原项目Python3.11、NumPy1.26.4、SciPy1.11.4、scikit-learn1.7.2、psutil和matplotlib。全部为CPU回放，BLAS每进程单线程，无需下载新模型或数据。

必要私有来源如下。

- 原V2 `selector_train_inputs.npz` 和 `selector_train_outcomes.npz`，含train600真实评分、冻结向量和调用账本。
- 原E3 `public/assets.py`，恢复组合顺序、执行复用、字段及来源验证。
- 原E12每折 `GROUP_MAPPINGS.npz` 和 `MASKS.npz`，含输入可见的组映射及六个嵌套问题掩码。
- 原E8五折M0_tuned的四项PCA/标准化状态。旧奖励拟合系数不进入本轮学习。
- SaveRouter四份固定源码 `data.py`、`grouping.py`、`profiles.py`、`router.py`。默认从E12私有文献缓存动态加载，字节哈希由适配器核对。

源码可通过环境变量 `MAB_SAVEROUTER_SOURCE` 指向另一个相同字节的四文件目录。完整真实来源及哈希见 `E13_PROTOCOL.json`。私有逐题ID、奖励、模型权重和完整数组继续保存在private。

## 真实可执行入口

以下顺序对应本轮流程。规范目录已经生成完整结果时继续保持只读，禁止在原目录重新prepare或覆盖冻结收据。

```powershell
$e13Python = 'E:\语言模型组合路由\.venv\Scripts\python.exe'
$e13Code = 'D:\develop\CV\多臂老虎机\experiments\e13_partial_combination_feedback'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = ''

& $e13Python -B -X utf8 (Join-Path $e13Code 'run_e13.py') prepare
& $e13Python -B -X utf8 (Join-Path $e13Code 'run_e13.py') smoke
& $e13Python -B -X utf8 (Join-Path $e13Code 'run_e13.py') train --workers 3
& $e13Python -B -X utf8 (Join-Path $e13Code 'evaluate_e13.py')
& $e13Python -B -X utf8 (Join-Path $e13Code 'build_costs.py')
& $e13Python -B -X utf8 (Join-Path $e13Code 'finalize_e13.py')
```

`prepare`核对输入与必要软件检查，封存输入包、协议和训练源码。`smoke`只运行一条轨迹的B60完整机制，正式训练复用该快照。`train`完成全部90条轨迹，先封存540个动作快照。`evaluate_e13.py`验证世界、快照与来源锁并封存自身，然后才解码完整奖励作评价。成本与报告入口仅使用这些实际冻结工件。

若需从头独立复现，先指定新目录。

```powershell
$env:MAB_E13_ROOT = 'E:\语言模型组合路由\research\e13_partial_combination_feedback_reproduction_01'
if (Test-Path -LiteralPath $env:MAB_E13_ROOT) { throw '复现目录已存在，请选一个新目录' }
```

保持原 `MAB_ASSET_ROOT`，在新root依次执行上述入口。新协议和收据在新root生成，旧规范输出不改写。已有中途快照仅在当前观察哈希和模型哈希匹配后复用。完整结果存在时，训练和评价入口会拒绝将其覆盖。私有原收据保留实际D/E路径，迁移机器时须恢复相同资产布局，或在新root重建该次来源记录。

## 成本口径

每道Main训练题的单生成完整执行账为3成员、1聚合、1gold和1评分。Pair为4成员、2聚合、1gold和2评分，新增邻居复用已有两个成员及gold。真实prompt/completion token、成员与聚合费用分开记录。

一次canonical question/reference version的答案可复用。每条生成轨迹单列gold数，同时给出跨重复联合采购时的答案键并集。成员和组合执行按question/generation/member或action键去重。共享成员执行无法揭示尚未聚合的组合奖励。

同一快照的所有方法共享同一反馈ledger，采购费只记一次。六预算嵌套、五折历史相交、logger及生成属于重复比较，不能把这些账简单相加当成独立数据采购。Main与Pair的完整采购量不同，比较两场景需要查看实际Token和执行账。

评价期各策略选队不同，服务Token按其实际选择的三成员与聚合器单独汇总。`service_execution_cost.total_across_nine_replays`是五个E120回填600题后，3logger×3生成共5400次服务的总和。`mean_per_600_question_generation_trajectory`为总和除以9，token最小/最大/均值在九条600题轨迹间计算。该账与训练反馈采购分列。

`source_model_wall_s`来自已有缓存调用时间之和，当前新增GPU推理时间保持未测。CPU拟合、规划、更新、评价及资格开销按实际计时记录。

## 统计和保存位置

主比较固定为Pair、Gold120的Group−Raw、Group−RawCentered、GroupResidual−SaveRouter-FixedMask。5000次bootstrap按outerfold×task重采样原题，同题全部方法、生成和logger一起保留。普通95%配对区间与Holm3调整p分别报告。区间条件于已拟合模型，五折训练历史重叠和旧600题开发暴露限制外推。

私有工件中，`inputs/fold*/INPUTS_ONLY.npz`保存固定输入与掩码，`worlds/fold*/logger*/seed*/{Main,Pair}/B*/SNAPSHOT.npz`保存观察、模型、分数和动作。对应FROZEN与WORLD_FROZEN收据绑定哈希，EXECUTION_AND_ACCESS保存合法访问和执行账。`ALL_PREDICTIONS_FROZEN.json`、`EVALUATOR_SOURCE_FROZEN.json`、`PAIRED_OUTCOMES.npz`及评价收据连接训练与统计。

本地工作区缺少Git checkout。本轮的版本证据为实际源码、协议与工件哈希，文献中的作者commit属于外部依赖版本。MAB分支和推送状态由实际交付清单记录，未生成虚构的MAB commit SHA。
