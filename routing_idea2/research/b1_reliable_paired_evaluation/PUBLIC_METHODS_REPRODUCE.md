# 公开方法参考复现

## 已执行范围

ASI作者AlphaFold原例子20档预算、每档1000次，Active Testing作者标准二次函数例子100次，作者CPS参考例子2000次，以及作者CPS全部17个测试。Robust ASI保留原运算与独立公式一致性核验。原Pew数据的完整复现状态记为部分完成，原因见方法报告。

## 源代码与依赖

在项目Python环境中运行，另外将`numba==0.61.2`和`llvmlite==0.44.0`安装到单独的`private/public_methods/dependencies`。实验使用NumPy1.26.4、SciPy1.11.4及项目已有pandas、scikit-learn、OmegaConf和pytest。

```powershell
$b1Work = 'E:\语言模型组合路由\research\b1_reliable_paired_evaluation'
$b1Python = 'E:\语言模型组合路由\.venv\Scripts\python.exe'
$env:B1_METHODS_WORK = Join-Path $b1Work 'private\public_methods'
& $b1Python -B -X utf8 (Join-Path $b1Work 'public\methods\fetch_public_sources.py') (Join-Path $env:B1_METHODS_WORK 'upstream')
& $b1Python -m pip install --no-deps --target (Join-Path $env:B1_METHODS_WORK 'dependencies') numba==0.61.2 llvmlite==0.44.0
& $b1Python -B -X utf8 (Join-Path $b1Work 'public\methods\download_reference_data.py')
& $b1Python -B -X utf8 (Join-Path $b1Work 'public\methods\run_native_examples.py')
& $b1Python -B -X utf8 (Join-Path $b1Work 'public\methods\qualify_public_methods.py')
```

`fetch_public_sources.py`固定四个作者commit及文件名单。没有许可证的代码仅写入私有源快照目录。AlphaFold数据由作者PPI加载器所使用的公开文件ID下载，下载响应必须具有NPZ格式，保存SHA256后读取。

## B1有限池接口

`public_methods.py`提供主匹配估计器表的ASI、Robust ASI抽样接口。两者的调参输入只来自合法历史反馈。设置`weights`时，预算以人口平均值表示。省略`weights`时，预算以问题池合计表示。两个单位必须保持一致。

`finite_pool_supplements.py`由独立评价器导入，提供以下函数。完整验证成绩由评价器提供给估计函数，每次购买只索引已抽中的问题，采样设计不读取未揭示评分。

```python
fixed_size_asi_supplement(
    d, m, predicted_cost, active, bernoulli_p,
    answer_budget, execution_budget, repeats=2000,
    source_root=private_source_root,
    dependency_root=private_dependency_root,
)
active_testing_supplement(
    d, m, predicted_cost, active,
    answer_budget, execution_budget, repeats=2000,
)
```

补充输出保存样本索引、真实纳入概率或顺序条件概率、逐次估计和区间。样本索引与私有题池顺序的关联保留在私有目录。公开只汇总估计误差、覆盖、费用和来源。

## LURE有限样本区间核验

设活跃题池为M、固定购买数为n，原作者LURE可写成条件无偏PURE步骤的确定加权和

\[
a_t=\frac{\sum_{j<t}\ell_j+\ell_t/q_t}{M},\qquad
w_t=\frac{M(M-n)}{n(M-t)(M-t+1)},\qquad
\widehat R_{LURE}=\sum_t w_t a_t.
\]

当n小于M时，权重和为1，每个步骤的条件期望等于完整有限池均值。作者共享配置的uniform clip为0.5，归一化后有`q_t >= 1/[3(M-t+1)]`。因此各鞅增量条件范围至多

\[
r_t=\frac{3(M-n)}{n(M-t)}.
\]

固定评价时点使用双侧Hoeffding-Azuma半宽`√(log(2/alpha) Σr_t²/2)`，转换到收益再乘`2M/N`。全量购买时直接返回题池真值，零购买时返回有界宽区间。该区间作为本适配的单列推断扩展，原作者主要点估计保持LURE。

N5、n3全部60条有序抽样路径的概率和、期望、加权分解已枚举核验。证明及边界条件经独立子代理检查，证据记录在`PUBLIC_METHODS_QUALIFICATION.json`。
