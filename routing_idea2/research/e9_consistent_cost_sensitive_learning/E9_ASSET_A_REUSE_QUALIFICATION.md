# E8 资产 A 复用资格审计

**结论为 PASS_REUSE_ELIGIBLE，置信度高。**

5 个 outer 均为 480 条历史与 120 条输入，GSM8K 与 BBH 分别占一半。原始题目顺序、canonical、嵌入和组合映射均通过冻结来源核验。

## 内层 API

`e8_assets.inner_splits(history, np.arange(480), 20261113 + outer_fold)`

内部使用 task/sub 联合标签与 `StratifiedKFold(3, shuffle=True, random_state=seed)`。返回 history480 的局部行索引，每折 320 拟合与 160 留出。所有候选的 FIT_LOG 行划分与此 API 完全一致。

## 数值核验

执行 45 个 inner Ridge 与 5 个 outer Ridge 资格 replay。45 个 inner 系数哈希、5 个 tuned outer 系数与预测哈希均与保存资产完全一致。原始 M0 outer 预测也由保存特征与系数精确恢复。

三档 `[0.1, 1, 10]` 确实通过历史 inner OOF 原始 argmax 路由正确率参与选择，任务权重为 GSM8K50% 与 BBH50%。三档动作一致且得分平局，所有 outer 按原规则选最强正则 10。

PCA16 训练投影经过训练 StandardScaler 后满足 `Phi.T @ Phi = n I` 至浮点误差。因此统一 Ridge 正则使分数正比例缩放。原始 M0 与 tuned M0 在全部 600 个 outer 输入上的行动完全相同，正比例因子为 `0.6311475409836067`。

## 优势与风险

- 优势包括保存收据与源码哈希完整，逐候选的训练行、奖励哈希、系数哈希和真实路由选参均可追溯。
- 风险包括复用资格依赖相同划分、特征处理、常数项惩罚、任务权重和评分版本。任何这些契约变化都需要重新核验。

## 访问与成本

本次读取各 fold 合法 history480 奖励与 input-only120，未解码 source aggregate_scores，evaluation 奖励读取次数为 0。完整资格 replay 为 50 次闭式 Ridge。另一次审计尝试在 1 次拟合后因 lambda 浮点运算顺序的精确断言停止，已将断言改为冻结实现的运算顺序，总资格成本为 51 次。E9 正式拟合为 0，物理模型调用为 0。私有局部行追溯记录保存于 `private/E9_ASSET_A_SPLIT_TRACE.json`。

E9 训练代码仍需独立审查，资产资格通过仅支持原样复用 E8 的 A。
