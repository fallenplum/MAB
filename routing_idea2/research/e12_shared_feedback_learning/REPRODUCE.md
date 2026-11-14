# E12 实际运行与资产说明

## 固定范围

本轮沿用用户指定的六个累计预算60、120、240、300、360、480，四条主线加一个Group-revealed控制，并逐预算重算Global_B。原E7-B RBF仅复用240/300/360/480四点。

H480全部训练文本和task/subtask metadata可见。奖励输入只包含当前S_B原题的20组合×3已有生成评分。E120的真实组标签用于评价汇总，实际路由只使用问题表征。

## 环境

实际执行环境为 `E:\语言模型组合路由\.venv\Scripts\python.exe`、Python3.11、NumPy1.26.4及已有合法资产加载器。矩阵计算采用CPU及单线程BLAS，图表采用Python matplotlib。

模型源码为 `code/e12_shared_feedback.py`，图表源码为 `code/plot_e12.py`。原始任务、实际stdout/stderr、CPU/墙钟及模型/掩码保存在新E12私有目录中。

## 原有输入来源

- E7的 `private/E7_SPLITS_FROZEN.npz`，SHA256为 `48e631682448fca1cd6d53dd2c2c29dadda33cdbeb86d136ae3379a74a94f29a`。
- 原 `selector_train_inputs.npz` 的题目ID、task/subtask与MiniLM向量，以及E9合法480/120折包。
- MiniLM每折H480状态位于E8的 `private/training/fold{f}/M0_tuned/model_state.npz`。
- Prefill原向量位于E10的 `private/features/PREFILL_FEATURES_FROZEN.npz`，H480状态位于其 `private/models/fold{f}/R_full/MODEL.npz`。
- 满预算组均值头位于E11的 `private/fold{f}/GROUP_AND_HALF_MODEL.npz`。
- 历史RBF分数位于E7的 `private/b/FROZEN_B_f{f}.npz`，使用 `main_predictions[2]`和该文件的局部E索引。

协议逐项绑定直接来源SHA，私有 `SOURCE_REFERENCES.json`保存完整路径。当前代码以原E盘资产布局为前提，旧文件保持原字节。准备阶段曾定位Random轨迹，正式E12采用本任务指定的确定性六点S_B，来源说明保留在私有执行档案。

## 实际执行命令

首次正式计算已执行85次小矩阵求解。

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e12_shared_feedback_learning\public\code\e12_shared_feedback.py' run
```

程序验证直接来源，恢复原I/U/H/E，按ID哈希和BBH轮转恢复低预算掩码，然后仅把当前揭示的奖励传给拟合操作。每个表征与外折一次T求解，Raw与Group-revealed按各自当前S_B拟合。480点复用并核验保存头，全部分数和动作封存后才统一评分。

本轮实际运行的无新增求解验证命令为

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e12_shared_feedback_learning\public\code\e12_shared_feedback.py' verify
```

验证直接来源、冻结预测SHA、完整六点正确事件数与点准确率。此入口新增求解与prefill均为0。

图表生成入口已执行并做图像目视检查。

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e12_shared_feedback_learning\public\code\plot_e12.py'
```

## 实现与计数

正式85次solve包含10次无奖励T映射、50次低于满预算Raw、25次低于满预算Group-revealed。Group-all的其余60个头由矩阵乘法得到，满预算15个头来自原资产。Global_B只计算当前标签均值与语义argmax。

Raw、Group-revealed的Hessian随当前揭示行变化，Group-all的T固定在H480训练文本和组metadata上。T分数可为负，原实现保留其线性输出。全组能力每预算只从当前S_B计算，未覆盖组的回退也仅使用同S_B全局均值；本次28组均有覆盖，实际回退为0。

RBF采用旧近并列规则 `np.argmax(scores >= scores.max(-1,keepdims=True)-1e-10,axis=-1)`，按 `global_actions[E7_E]=local_actions`恢复全局，模型求解为0。M/P新方法采用原semantic精确并列规则。各预算的Global_B按任务指定的两级均值运算重算，与旧GlobalFixed的近并列浮点敏感性已在总结披露。

## 私有保存

- `private/fold{f}/MASKS.npz`保存六点共享mask。
- `GROUP_MAPPINGS.npz`保存两个T与合法训练组键。
- 各 `B{budget}/MODELS_AND_GROUP_MEANS.npz`保存五类头、当前C_B、支持数和当前已揭示行。
- `SUPPORT_AND_ACCESS.json`保存各预算组支持和反馈访问范围。
- 全局分数/动作、配对正确性、原始进程日志及两张图的绘制收据分别保留。

运行与输出采用不可覆盖检查。需要重新拟合时，应在新空输出目录复制相同协议，导入原模块并仅重定位其 `ROOT`后运行 `train()`和`evaluate()`，原来源继续由 `ASSETS`绑定。此类再计算会新增85次求解，需独立记录。本轮实际执行为一次正式矩阵及一次无新增求解的verify。

## 预算和成本口径

每个题包为1 gold、18历史成员执行、60历史聚合与60评分。预算是累计场景，所有方法共享该份反馈，五折历史交叉继续复用600题原缓存。COST逐fold/budget给出真实历史token、源调用wall以及各方法的服务足迹。

服务账按每生成三个成员与一次聚合计算，服务评分和gold为0，购买反馈单列。实际新增prefill、答案、聚合和gold均为0。旧Qwen表征的准备与部署开销继续保留，当前缓存命中费用与未来新题编码费用采用不同口径。

六点面积、首次达到参考点的预算与所有曲线均为已测点的描述性结果。历史RBF低预算位置保留空白，实际图形连接原点值。未来真实部分动作反馈研究按独立协议实施。
