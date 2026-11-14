# E10 组内匹配分解

## 计算边界

`e10_matching.py` 接收独立评价器提供的冻结动作、题目元数据和奖励立方体。奖励轴固定为题目 × generation × 动作，原 train600 对应 600 × 3 × 20。

E9 的 `E9_PAIRED_OUTCOMES.npz` 只包含各策略已选择动作的 600 × 3 奖励。`V_mix` 还需要所有动作的奖励，完整来源是原 `selector_train_outcomes.npz` 的 `aggregate_scores`。既有 `e8_assets.load_historical` 会验证来源并转换为所需轴顺序。

本模块的预备资格只使用确定性合成矩阵。真实奖励读取与正式分解由协议冻结后的独立评价器执行。

## 权重与分层

- coarse 使用 outerfold × task。
- fine 使用 outerfold × task × subtask。GSM8K 的 subtask 统一为 `gsm8k`，BBH 保留已有子任务。
- 总体中 GSM8K 与 BBH 各占 1/2，任务内题目等权。原数据两个任务各 300 题，等价于每题 1/600。
- 每个 task、outerfold 和 BBH subtask 视图使用该视图中任务等质量、任务内题目等权的同一定义。每个视图重新计算其组内动作频率。原五折各包含两个任务各 60 题。
- generation 分解沿用同一冻结动作和同一组内频率，只替换奖励的 generation 切片。
- `question_weight_in_view` 是各组在该视图中的质量，组数不参与等权平均。
- 退化组的题目覆盖同时报告原始题目比例和任务加权质量。n=1 与单动作类别存在包含关系。

## 精确期望

设某组有 n 题，冻结动作 a 的选择次数为 k_a，p_a=k_a/n。将冻结动作列表的所有 n! 个排列等概率分配给组内题目时，每题收到动作 a 的概率恰为 p_a。因此

\[
\mathbb E[V_{\rm perm,g}]
=\frac1n\sum_{i\in g}\sum_a p_a r_{ia}
=V_{\rm mix,g}.
\]

相同动作重复不会改变此结论。每个不同的动作分配具有相同的重复排列数 `prod_a k_a!`。实现直接计算该有限期望，浮点运算的残差在资格文件中记录。

各组的题目权重在组内恒定，因此可以按组的任务加权质量汇总。最终报告直接使用题目权重，避免引入组等权。

## 协方差与不变量

令 `I_ia=1[a_i=a]`，协方差分母取 n，则

\[
G_{\rm match,g}
=\sum_a\left\{\frac1n\sum_i I_{ia}r_{ia}
-\left(\frac1n\sum_i I_{ia}\right)
 \left(\frac1n\sum_i r_{ia}\right)\right\}
=\sum_a\operatorname{Cov}_g(I_a,r_a).
\]

1. 全排列的平均值由边缘概率 p_a 唯一确定，等于 `V_mix`。
2. 单动作组中，实选动作和混合分配对每题一致。n=1 为其中一个特例，故匹配增益为零。
3. 同时置换奖励动作坐标和动作索引只改变求和次序，全部值保持。
4. 任意逐题公共项 c_i 对实选奖励和混合奖励的增加量均为 c_i，匹配增益保持。
5. 任意组内动作项 d_ga 对实选均值和混合均值的增加量均为 `sum_a p_g(a)d_ga`，匹配增益保持。此结论按所用分组成立，fine 的任意组偏置可以改变 coarse 分解。

两个分解共享 `V_real`，所以逐 generation 以及聚合后均有

\[
G_{\rm coarse}
=(V_{\rm mix,fine}-V_{\rm mix,coarse})+G_{\rm fine}.
\]

## 接口

```python
from e10_matching import decompose_policies

results = decompose_policies(
    rewards,                    # question x generation x action
    frozen_actions_by_policy,   # {name: integer[question]}
    task, subtask, outerfold,
    generation_seeds=[42, 137, 2024],
    action_keys=semantic_action_keys,
)
```

同一接口供 P1 的 A/B/C/D/NS 和 P2 封存后的候选、背景及内容对照调用。输出每个策略的总体、task、outerfold、BBH subtask 视图，每个视图含 generation 指标、coarse/fine 每组支持、动作计数与频率。

## 数学资格

`qualify_e10_matching.py` 检查任务要求的五项性质，并追加不等任务规模权重、coarse/fine 恒等式和协方差恒等式。全排列检查使用 `Fraction` 对 24 个排列精确求均值，再与数值实现比较。

```powershell
& 'D:\Anaconda\envs\tslib\python.exe' qualify_e10_matching.py --output E10_MATCHING_MATHEMATICAL_QUALIFICATION.json
```

## 优势与劣势（风险）

**优势**是动作次数完全固定时可以直接计算期望，无需重新训练或抽取置乱样本。逐题公共难度和同组动作平均优势均由差值消除，协方差表达和数学资格可以独立复核。

**劣势（风险）**包括小 BBH 组的重排支持有限、单动作组的增益天然为零、OOF 历史重叠和旧开发集反复暴露。精确点分解只描述既有有限样本，不能据此作因果命名、正式内容随机化推断或可学习性否定。

数学恒等式的置信度评级为**高**，依据为代数推导和确定性资格。真实 train600 数值需待实际执行后形成结论。
