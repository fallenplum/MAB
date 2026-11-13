# E8 复现

## 规范目录与环境

规范实验位于`E:\语言模型组合路由\research\e8_decision_focused_combination_learning`。GitHub的`routing_idea2/research/e8_decision_focused_combination_learning`内容对应本机的`public`目录。私有V2原始缓存、分折包、权重和逐题记录由原项目管理。

使用项目Python3.11、NumPy1.26.4、SciPy1.11.4、scikit-learn1.7.2。精确QP依赖OSQP1.0.4和Clarabel0.11.1，安装在本研究`private/dependencies`。全部训练为CPU，BLAS每进程1线程，正式使用3个外折worker。

## 核验及防覆盖评价重放

```powershell
$e8Work = 'E:\语言模型组合路由\research\e8_decision_focused_combination_learning'
$e8Python = 'E:\语言模型组合路由\.venv\Scripts\python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = ''
& $e8Python -B -X utf8 (Join-Path $e8Work 'public\evaluation\reproduce_guarded.py') verify
& $e8Python -B -X utf8 (Join-Path $e8Work 'public\evaluation\reproduce_guarded.py') evaluate-replay
```

verify检查三套源锁、结果锁及70个模型的全部训练、预测和日志哈希。evaluate-replay创建新的私有时间目录，复制只读输入，恢复全部模型并重算所有14种变体、10000次配对统计与噪声诊断。程序比较完整数值结果和主统计，原始结果保留。实际封存模型的评价重放已通过逐项一致检查。

## 完整训练重建

训练API位于`public/training/e8_training_v1.py`，`train_fold`仅接收当前合法历史区。`run_e8_training_v1.py`读取独立history480与evaluation_inputs_only120包，完成五外折和内层选择。每个variant保存模型、OOF分数、真实拟合目标、配置、优化诊断、外折输入预测及封存收据。

重建时为`--root`指定新目录，复制公共代码和隔离依赖，保留原规范目录的协议、源锁和合法分折包作为只读来源。

```powershell
$e8NewRoot = Join-Path $e8Work ('private\training_reproduction_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
if (Test-Path -LiteralPath $e8NewRoot) { throw 'Reproduction directory already exists' }
New-Item -ItemType Directory -Path $e8NewRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $e8Work 'public') -Destination $e8NewRoot -Recurse
New-Item -ItemType Directory -Path (Join-Path $e8NewRoot 'private') | Out-Null
Copy-Item -LiteralPath (Join-Path $e8Work 'private\dependencies') -Destination (Join-Path $e8NewRoot 'private') -Recurse
& $e8Python -B -X utf8 (Join-Path $e8Work 'public\training\run_e8_training_v1.py') --root $e8NewRoot --protocol (Join-Path $e8Work 'public\E8_PROTOCOL.json') --lock (Join-Path $e8Work 'private\E8_SOURCE_LOCK.json') --packs-manifest (Join-Path $e8Work 'private\qualification\FOLD_PACKS_MANIFEST.json') --workers 3 --base-seed 20261113
```

原运行在均匀真实tie敏感性出现OSQP迭代精度失败。原49个模型保留，执行修订在评价奖励读取前冻结，以同一epigraph凸QP的Clarabel求解补齐21个模型。该修订记录在`E8_EXECUTION_AMENDMENT_01.json`，旧失败日志位于`private/training/RUN_FORMAL_V1.log`。

```powershell
if (-not (Test-Path -LiteralPath (Join-Path $e8NewRoot 'private\training\E8_FROZEN_TRAINING_INDEX.json'))) {
    & $e8Python -B -X utf8 (Join-Path $e8Work 'public\training\run_e8_recovery_v1.py') --root $e8NewRoot --protocol (Join-Path $e8Work 'public\E8_PROTOCOL.json') --amendment (Join-Path $e8Work 'public\E8_EXECUTION_AMENDMENT_01.json') --lock (Join-Path $e8Work 'private\E8_SOURCE_LOCK.json') --recovery-lock (Join-Path $e8Work 'private\E8_RECOVERY_SOURCE_LOCK.json') --packs-manifest (Join-Path $e8Work 'private\qualification\FOLD_PACKS_MANIFEST.json') --workers 3 --base-seed 20261113
}
```

恢复入口复用既有完整收据，对未封存模型执行原训练函数，只替换已审阅的SPO数值后端。已完成全局索引时会拒绝再次恢复。完整重训的近并列敏感变体可能出现数值动作差异，原冻结模型重放与跨后端稳定审阅分别保留。

## 数学与文献复核

`public/qualification/qualify_e8.py`恢复原Ridge并验证190等权平方差恒等式。`review_training_operators.py`独立检查实际梯度、M3正规方程、精确SPO与小型控制。资格脚本输出属于软件检查，正式评分由独立评价入口处理。

`public/literature/fetch_e8_literature.py`下载固定原文字节与作者源码，`check_spo_author_operator.py`重放PyEPO作者SPO目标。原文、源码与元数据清单见`E8_LITERATURE_SOURCE_MANIFEST.json`。论文PDF和下载原始材料保存在`private/literature`。

## 私有产物与结果边界

- `private/qualification/fold_packs`保存每折合法历史、仅输入评价区和统计元数据。
- `private/training/fold*/variant`保存权重、完整合法OOF与拟合日志。
- `private/training/E8_FROZEN_TRAINING_INDEX.json`记录70模型及执行修订哈希。
- `private/evaluation`保存校准权重、逐题指标、分数、实际缓存费用与奖励访问收据。
- `E8_PRIMARY_STATISTICS.json`保存预定两项主检验，`E8_SHRINKAGE_DIAGNOSTIC.json`明确标记事后描述。

所有评价均属于原600题的开发交叉验证。validation200没有用于本轮选择，fresh400和sealed200继续封存。本轮新增LLM/API推理为0。
