# E6复现入口

## 环境与输入

已执行环境为Windows、Python3.11、NumPy1.26、SciPy1.11、scikit-learn1.7。CPU计算固定BLAS线程1，新增LLM调用为0。默认资产根为`E:\语言模型组合路由`，也可通过`MAB_ASSET_ROOT`指定拥有相同目录结构的资产副本。

复现需要原train600输入和真实20组合×3种子评分、E3/E4代码、E5的冻结基础预测和门控、E5合成世界，以及E6私有冻结划分。公开库保存代码和脱敏统计，逐题文本、奖励、参数和账本位于本地private。输入指纹见`E6_PROVENANCE_AND_COST.json`，本轮文件指纹见`PUBLIC_DELIVERY_MANIFEST.json`。

原始600道题的生成和聚合已经完成。本轮只读取缓存。历史37274127 Token为已有资产来源费用，新增物理费用中的LLM调用为0。

## 检查现有交付

以下命令只核验已交付结果并更新独立交付审计。

```powershell
$e6Root = 'E:\语言模型组合路由\research\e6_contrast_directed_feedback\public'
$e6Python = 'E:\语言模型组合路由\.venv\Scripts\python.exe'
& $e6Python -B -X utf8 "$e6Root\verify_e6_delivery.py"
```

## 分阶段重放

重新计算会写入该资产根的E6结果。已有运行的私有回放与公开清单保留在原资产根时，可以在其资产副本设置`MAB_ASSET_ROOT`执行。冻结协议和P/Q文件由`prepare_e6.py`在任何E6外折评价前生成，现有协议禁止覆盖。原题划分相同，时间戳和压缩容器字节可能随重放改变，数值结果按独立审计比对。

```powershell
if (-not $env:MAB_ASSET_ROOT) {
    $env:MAB_ASSET_ROOT = 'E:\语言模型组合路由'
}
$e6Root = Join-Path $env:MAB_ASSET_ROOT 'research\e6_contrast_directed_feedback\public'
$e6Python = 'E:\语言模型组合路由\.venv\Scripts\python.exe'
& $e6Python -B -X utf8 "$e6Root\a\run_e6_a.py" qualify
& $e6Python -B -X utf8 "$e6Root\a\run_e6_a.py" train
& $e6Python -B -X utf8 "$e6Root\a\run_e6_a.py" evaluate
& $e6Python -B -X utf8 "$e6Root\a\summarize_e6_a.py"
& $e6Python -B -X utf8 "$e6Root\b\run_contrast_synthetic.py" --replicates 2000
& $e6Python -B -X utf8 "$e6Root\b\audit_and_report_b.py"
foreach ($e6Fold in 0..4) {
    & $e6Python -B -X utf8 "$e6Root\c\run_real_pool.py" --fold $e6Fold
}
& $e6Python -B -X utf8 "$e6Root\c\summarize_real_pool.py"
& $e6Python -B -X utf8 "$e6Root\finalize_e6.py"
```

E6-B的世界奖励和原采样随机数来自E5已保存的六个世界，额外概率使用条件随机数细分。小先导固定5题/活跃层，原先导30题/层，两个规模都计算。主C为50份12题先导×每份40个购买随机数，三预算均计算。支持题、生成种子和购买随机性分别记录。

## 独立审计

```powershell
& $e6Python -B -X utf8 "$e6Root\theory\check_e6_theory.py"
& $e6Python -B -X utf8 "$e6Root\theory\audit_e6_operators.py"
& $e6Python -B -X utf8 "$e6Root\theory\audit_A_bootstrap.py"
& $e6Python -B -X utf8 "$e6Root\theory\audit_design_solver.py"
& $e6Python -B -X utf8 "$e6Root\theory\audit_real_pool.py"
```

审计包括加权分类恒等式、共享反馈协方差、有限池公式、84个完全枚举采样分布、Logistic独立优化、A的完整5000次题级统计、E评分NaN隔离、150000次策略选择、3000个实际条件设计以及Q评分NaN隔离。扰动重放只写`private/audit_replays`。

## 私有文件索引

| 路径 | 内容 |
|---|---|
| `private/PROTOCOL_FROZEN.json` | 首次评价前保存的配置与划分哈希 |
| `private/E6_FGE_PQ_FROZEN.npz` | F/G/E及P/Q原题索引 |
| `private/a/FROZEN_A_f*.json/.npz` | G内CV、参数和E区冻结动作 |
| `private/a/PAIRED_E_RESULTS.npz` | A逐题预测、三生成结果和bootstrap |
| `private/a/PRIMARY_A_CONTRAST.npz` | A主比较的题级贡献和5000次配对统计 |
| `private/b/cases` | 12格完整模拟估计、策略选择和费用 |
| `private/b/pilots` | 合法先导及拟合设计 |
| `private/c/C_f*.npz` | 购买题、真实纳入概率、估计值和策略选择 |
| `private/c/DECISIONS_C_f*.json` | Q评价前保存的决策收据 |
| `private/c/PLANNING_C_f*.json` | 每先导/预算的凸求解、预期费用和条件抽样信息 |
| `private/c/EVALUATED_C_f*.npz` | 仅评价器使用的P/Q配对真值及收益 |
| `private/c/PAIRED_C_RESULTS.npz` | 300道Q题的最终结果与题级bootstrap |

## 统计与解释

主检验为A3减任务内匹配随机、C Contrast减Marginal、C Contrast减Uniform，C主Gold36。三项统一Holm，另给Bonferroni98.3333%区间。原始问题及其三生成、所有策略共同抽样，匹配随机每次重新计算接受数。C的采样Monte Carlo误差以先导为簇，不能和300道Q的题级区间混用。

E6-C的硬预算采样是条件Bernoulli，使用实际纳入概率。独立Bernoulli方差公式仅在单独诊断中应用。预计Token代理约束可能在条件化后偏移，实际Gold、调用和保守Token硬上界另外核验。完整费用与科学失败结果均保存。
