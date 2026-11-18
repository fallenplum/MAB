# E23 部署并列处理与同题池参照复现

## 研究范围

使用 E20 P2 的原 2,400 道题，每个任务 1,200 道，每题保留三个生成 seed。全部策略使用同一题池、同一冻结聚合器和评分器。原 B/C 主结果保留，新增比较属于结果已曝光后的补充分析。

本阶段先完成训练侧先验部署分析与覆盖审计，随后根据用户“只补齐这些缺失调用”的明确授权，执行去重后的 1,770 次成员生成和 4,263 次聚合。`PROTOCOL.json`、`RESULTS.json` 保存准备阶段快照，最终执行与完整参照结果以 `RESULTS_FINAL.json` 为准。

## 固定策略

- B/C 使用 E20 全部 train600 拟合的最终模型，沿用 `tau=1e-8` 的原支持集合。
- 训练先验为 train600 的 1,800 个记录生成事件上各组合的整数成功数。只在原支持集合内选成功数最高的动作。如成功数仍完全相同，按均匀策略处理。
- GlobalFixed 也只由同一训练整数成功数决定，选中组合 5。
- Ridge 恢复 E17 的全量600模型及其专属预处理，沿用原均值标签目标的 alpha=333.3333333333333。
- Uniform 与三个 Matched 控制分别重新拟合全量600最终头。保持原总样本分母、正则和输入，使用 E22 的匹配配方、三个既定种子及 `final600` 盐值。四个头共用 lambda=0.003472222222222222。
- 原 B/C、先验规则、raw-exact、GlobalFixed、Ridge、Uniform、三个 Matched 共十二份策略在补齐调用前冻结。MatchedMean 为三个既定策略的等权混合。

## 依赖与私有资产

本地分析依赖 Python、NumPy、SciPy 和原研究目录中的冻结源码。物理推理复用原 P2 的 Linux 推理环境、Transformers、PyTorch、bitsandbytes、模型权重、tokenizer、NF4配置及生成参数。

以下私有资产需要依原目录归档恢复，原题、答案、权重、逐题矩阵和原生生成文本均保留在私有工件中。

- 原 train600 输入与 `selector_train_outcomes.npz`。
- E17 `FULL600_MODELS.npz`。
- E20 `FINAL600_MODELS_FROZEN.npz`、`P2_PREDICTIONS_FROZEN.npz`、原题输入及预处理工件。
- P2 原生成功记录、闭合执行账本、既有 106,536 条聚合评分及去重 gold 账户。
- E23 `POLICIES_FROZEN.npz`、`REGIONS_FROZEN.npz`、`FREEZE.json`、缺口清单及执行 scope/approval。
- E23 补齐执行的两个原生 receipt JSONL 与 `OUTPUTS_FROZEN.json`。

来源摘要保存在公开汇总和私有冻结收据内。复现入口沿用仓库既有兄弟研究目录布局。默认私有根目录为 `E:\语言模型组合路由`，远端新执行拥有独立写目录。

## 执行顺序

1. 运行 `qualify_prior.py`，完成部署先验合成资格。
2. 先运行 `analyze_references.py --phase freeze --root PRIVATE_RUN_ROOT --output PREPARED_RESULTS_JSON`，再运行 `analyze_references.py --phase analyze --root PRIVATE_RUN_ROOT --output PREPARED_RESULTS_JSON`。前者先冻结所有策略，后者读取既有评分并审计覆盖。冻结入口新建私有目录，禁止覆盖既有工件。
3. 运行 `package_execution.py --root PRIVATE_RUN_ROOT --output NEW_INPUT_BUNDLE_TAR_GZ`，仅打包已批准并集需要的输入和缺口，使用原 P2 成员缓存身份。
4. 运行 `qualify_supplement.py` 与 `qualify_monitor.py`，验证独立 scope、原生缓存、依赖和闭合导出。
5. 在独立远端根目录运行 `configure_execution.py --root REMOTE_ROOT`，建立六份成员清单和八份聚合清单。调用 `launch_supplement.py --root REMOTE_ROOT --worker NAME --gpu INDEX` 启动对应清单。
6. 使用 `execution_monitor.py --root REMOTE_ROOT` 监控实际成功、Token、异常及进程状态。全部已批准输出闭合后，加 `--export-final` 导出原生记录。
7. 下载两份原生记录与冻结收据，然后执行下面的最终评分入口。输出目录必须尚未存在。

```powershell
python -B finalize_results.py `
  --root PRIVATE_RUN_ROOT `
  --scope PRIVATE_SCOPE_JSON `
  --approval PRIVATE_APPROVAL_JSON `
  --outputs-frozen OUTPUTS_FROZEN_JSON `
  --member-records MEMBER_NATIVE_RECEIPTS_JSONL `
  --aggregate-records AGGREGATION_NATIVE_RECEIPTS_JSONL `
  --output-root NEW_PRIVATE_SCORING_DIRECTORY `
  --output RESULTS_FINAL.json
```

最终入口先验证补齐记录，再用原评分器评分新增聚合，合并原 106,536 条与新 4,263 条。只有全部十二份策略的正概率支持都完整，才输出全量 2,400 题结果。原 B/C 效用与已有私有数组逐项一致性检查必须通过。

### 本次长答案成员的调度续跑

成员4原两片在分别写入39和40条成功后安全退出。仅剩余548项改为四个互斥分片，数量139、125、143、141，使用GPU4、7、5、5。原scope、approval、EXECUTION_PLAN、模型及推理参数保持原绑定，八路聚合继续运行。

实际使用的续跑工具为 `stop_member4_v2.py`、`prepare_member_dispatch_v2.py` 和 `launch_member_dispatch_v2.py`。先保存安全退出收据，再生成独立 `DISPATCH_V2.json`，逐片启动。新旧成功 journal 均保留，退休片成功不再次执行。后续监控和最终导出使用 `execution_monitor_v2.py --root REMOTE_ROOT`，完成后加 `--export-final`。导出格式沿用原版，同时附带调度证据收据。十四项分片检查和十九项监控及完整权限路径检查均使用合成工件，正式执行总数仍为6,033。

其他成员完成后，GPU1、3空闲。四片中的 shard2、shard3 在分别完成83、101项后再次安全退出，使用 `move_member_dispatch_v2.py` 将其同一manifest、同一journal分别迁至GPU1、3，恢复全部已有成功，只续跑各自剩余60、40项。启动收据新增为 `resume1.start.json`，原启动收据保留。迁移工具的十六项合成资格通过，任务分配及科学参数均保持原绑定。

## 计数与解释

准确率以 GSM8K、BBH 各50%计算，BBH内部按题平均，另列27子任务及三个 seed。每题三个生成是题内重复。均匀策略及 MatchedMean 的纠错、破坏为耦合下的期望计数。

调用并集的实际费用用于覆盖多份冻结策略的评价。原 P2 费用、补齐费用、缓存复用和确定性重评分分别列示。复用已有 gold 账户不增加购买数，重复解码已有答案仍记录在评分账本中。该费用不能直接充当任一单策略单次服务的费用。

当前实现保留负结果。固定正则的控制服务于机制归因，不能据此声称各方法已经得到相同调参机会后的最佳性能。后续新比较只报告点值、纠错破坏、任务/seed分解及执行成本。
