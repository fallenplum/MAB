# B1 复现入口

## 环境与资产

规范实验目录为E:\语言模型组合路由\research\b1_reliable_paired_evaluation，环境为项目原Python3.11及NumPy1.26.4、SciPy1.11.4、scikit-learn1.7.2。下面PowerShell变量使用真实完整路径。全部主计算使用CPU。

```powershell
$b1Work = 'E:\语言模型组合路由\research\b1_reliable_paired_evaluation'
$b1Python = 'E:\语言模型组合路由\.venv\Scripts\python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = ''
& $b1Python -B -X utf8 (Join-Path $b1Work 'public\main\reproduce_guarded.py') verify
& $b1Python -B -X utf8 (Join-Path $b1Work 'public\main\reproduce_guarded.py') replay
```

verify先核验原协议中的全部源码、辅助模型和概率设计哈希。replay只创建带UTC时间的新私有目录，重放完整27格并与原始公开指标逐项比较，原始实验产物保留。

## 隐藏收益与费用边界

private/assets保存冻结模型和输入，不含验证真实评分。private/design保存预先冻结购买概率。评价入口才读取原selector_validation评分，并只在已购买位置构造反馈修正与当次费用。private/evaluator保存独立真值，private/replays保留27格购买矩阵、逐次估计、区间及Token。fresh400和sealed200载荷继续封存。

公开仓库保存聚合结果和重建代码，原始问题、评分矩阵、权重和逐题账本位于本机私有目录。GitHub研究目录routing_idea2/research/b1_reliable_paired_evaluation的内容对应本机规范目录中的public子目录。复制时保持该映射，使用项目已有的私有V2资产及记录哈希运行。

## 数学资格检查

```powershell
& $b1Python -B -X utf8 (Join-Path $b1Work 'public\inference\qualify_inference.py') --output (Join-Path $b1Work 'private\qualification_reproduction')
```

资格产物包含固定题池完整枚举、总体1728状态枚举、LTT合法iid有界例子及无标签回退。公开方法下载与原生例子的命令另见PUBLIC_METHODS_REPRODUCE.md。

## 合成重放

```powershell
$b1Replay = Join-Path $b1Work ('private\synthetic_reproduction_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $b1Python -B -X utf8 (Join-Path $b1Work 'public\inference\run_synthetic.py') --output (Join-Path $b1Replay 'public') --private (Join-Path $b1Replay 'traces') --methods-directory (Join-Path $b1Work 'public\methods') --design-directory (Join-Path $b1Work 'public\main') --repetitions 2000
```

该命令执行全部36个预设设置。预定的三个题量与两个相关性世界分别保留，三个生成重复合并为一道题。

## 补充与原生复现范围

run_supplements.py由独立评价入口调用已冻结方法，在原始私有Oracle已生成后重放CPS与LURE两项补充。默认创建全新的private/supplement_reproductions时间目录，也支持--output指定新的空目录。程序拒绝写回原始规范目录或任何已存在目录，并将重放指标与原始结果比较。正式结果位于SUPPLEMENTARY_FINITE_POOL_RESULTS.json及private/supplements。

Robust ASI原Pew完整实验需用户从Pew官方注册下载ATPW79.sav。当前已完成其原生运算资格和MAB历史反馈适配。Fixed-Size原完整实验仓库不可访问，作者CPS包的实际复现范围见01报告。
