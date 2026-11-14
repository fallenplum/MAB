# E10 实际复现入口

## 环境与私有依赖

主环境为 `E:\语言模型组合路由\.venv\Scripts\python.exe`，Python3.11、NumPy1.26.4、SciPy1.11.4、scikit-learn1.7.2。GPU阶段使用PyTorch2.4.1+cu121、Transformers4.51.3、bitsandbytes0.48.2，原Qwen3-8B checkpoint revision 为 `b968826d9c46dd6066d109eabc6255188de91218`。原权重与 tokenizer 所有文件已按下载收据完整SHA核验。

P0的独立CPU MiniLM重编码使用原冻结权重和 D 盘已记录的环境，版本及逐bit结果见 `CPU_EMBEDDING_REPLAY.json`。其运行时与Qwen主阶段分别记录。

当前代码以原 E 盘绝对资产根为前提。需要保留以下私有来源。

- 原 train600 input/outcome bundle、模型池与 V2 operator/matrix receipts。
- 原 `conditional_gain_20261106/private/FOLDS_PRIVATE.npz`及E8合法折包。
- 原E9完整冻结目录，其1148个文件的初始哈希保存在E10私有快照中。
- 原六模型 checkpoint 中的 Qwen3-8B，以及原 MiniLM tokenizer/model 来源。
- E10 的 `private/input`、`features`、`models`、全部冻结收据、`ALL_E10_PREDICTIONS_FROZEN.npz`与配对结果。
- `private/runtime`保存审计、原失败、文献原文和实际阶段脚本。

## 已执行的来源核验和评分重放

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e10_input_prefill_matching_audit\public\code\e10_reproduce.py' verify
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e10_input_prefill_matching_audit\public\code\e10_reproduce.py' scoring-replay
```

评分重放在新时间戳私有目录复制冻结模型收据、分数及必要来源，重算同一10000次题级bootstrap与完整分解。三个科学结果JSON须逐项精确相同，运行墙钟字段单列。原结果保留。实际重放为 `models_refitted=false`、新增prefill为0。

## 首次正式执行顺序

以下为本轮已实际执行的阶段入口，源码及依赖均保留。冻结目录采用完整性检查和不可覆盖收据。

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e10_input_prefill_matching_audit\public\code\e10_prefill.py' --manifest 'E:\语言模型组合路由\research\e10_input_prefill_matching_audit\private\input\PREFILL_INPUT_MANIFEST.json'
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e10_input_prefill_matching_audit\public\code\e10_train.py'
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e10_input_prefill_matching_audit\public\code\e10_evaluate_adapter.py'
```

prefill 完整缓存命中时先校验来源并返回。首次资格验证完整36层与第24层停止输出、padding及分批一致性，正式只执行24层。三生成评价共用同题一次确定性表征。600个full和6个match输入共606次，594个match向量复用full。

头训练保持480题拟合PCA16/Scaler与固定受罚常数，Ridge有效alpha为1000/3。全部25头和15个内容控制完成后，独立评价器才读取外折奖励。GSM8K子任务规范化由实际薄适配入口执行，其源封存见 `EVALUATION_ADAPTER_SOURCE_FREEZE.json`。

## 审计与统计实现

`code/e10_matching.py`为无文件访问的纯分解函数。P1 v2实际脚本及首次失败位于私有原执行档案。公开数学资格包括排列枚举、退化零值、坐标重命名、逐题公共偏移和组内动作偏移。

P2唯一主比较为R_full−A。按outerfold×task抽取10000次原题簇，seed20261114101。三生成和全部方法共同重采样。内容控制仅作三个固定配置敏感性，输出配对区间具有其条件性解释，未构造正式随机化内容p值。

`code/build_e10_cost.py`生成本轮分账，`code/e10_profile_inference.py`执行已保存模型的纯CPU推理剖析。当前成本版本含完整prefill、载入、资格、25头、额外CPU输入核验及6000次冻结头计时，初版成本保留于私有日志。

## 验收边界

真实结果为反复使用开发集上的探索性证据。新候选的25头已实际拟合一次，独立审计核验系数方程、梯度与恢复。评分重放的通过单独记录。原E9完整重新拟合的执行证据依然为空。

全研究CPU/elapsed、头训练峰值RAM和部分资格独占CPU计时缺少完整记录，COST保留null。组件延迟相加仅作为组件计量，连续在线端到端耗时尚未联合测量。

本候选未通过冻结GO，研究决定限定到此输入链、固定层、PCA16和Ridge配置。后续独立确认、保留集访问及新的候选需要对应的新协议。
