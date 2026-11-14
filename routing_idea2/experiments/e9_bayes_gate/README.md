# E9 Bayes gate 离线实验

## 实验身份与入口

本目录对应 `MAB_E9_Codex_Experiment_Prompt.md` 的Proper-Score Mean Router与Hierarchical-Shrinkage Mean Router实验。它与已有经验/后验均值×MSE/CSCE矩阵分开归档。所有输出默认写入工作区 `results/e9_bayes_gate`，私有逐题数组和权重写入其中的 `private`。

原始600题已用于多轮开发，五外折仅作为开发交叉验证。所有输入、奖励和模型来源有哈希清单。新增LLM、API、gold和保留题载荷读取为0。

## 环境

实际环境为项目Python3.11、NumPy1.26.4、SciPy1.11.4、scikit-learn1.7.2、psutil及matplotlib。CPU运行，BLAS单线程。无需安装新包或下载模型。

```powershell
$e9Python = 'E:\语言模型组合路由\.venv\Scripts\python.exe'
$e9Code = 'D:\develop\CV\多臂老虎机\experiments\e9_bayes_gate'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = ''
```

## 实际运行顺序

规范结果禁止覆盖。下列命令已按阶段实际执行，完整结果及日志保留在新目录。

```powershell
& $e9Python -B -X utf8 (Join-Path $e9Code 'audit_assets.py')
& $e9Python -B -X utf8 (Join-Path $e9Code 'math_consistency.py') --output 'D:\develop\CV\多臂老虎机\results\e9_bayes_gate\01_math_qualification.json' --historical-asset-root 'E:\语言模型组合路由'
& $e9Python -B -X utf8 -m unittest discover -s (Join-Path $e9Code 'tests') -p 'test_*.py' -v
& $e9Python -B -X utf8 (Join-Path $e9Code 'run_nested_cv.py') freeze
& $e9Python -B -X utf8 (Join-Path $e9Code 'run_nested_cv.py') smoke
& $e9Python -B -X utf8 (Join-Path $e9Code 'run_nested_cv.py') train --workers 3
& $e9Python -B -X utf8 (Join-Path $e9Code 'qualify_interfaces.py')
& $e9Python -B -X utf8 (Join-Path $e9Code 'diagnostic_signal.py')
& $e9Python -B -X utf8 (Join-Path $e9Code 'evaluate_frozen.py')
& $e9Python -B -X utf8 (Join-Path $e9Code 'finalize_report.py')
```

第一次协议冻结后、正式训练前，只强化了已有资产及数学审计必须为PASS的入口判断。原协议继续保留，`protocol_amendment_01.json`记录原source hash到修订hash的连接。模型、目标、数据、网格和统计规则均沿用原冻结设置，18个合法资源预检头已复用。

主训练入口仅加载每折 `history480.npz` 与 `evaluation_inputs_only120.npz`。后者出现任何奖励、答案或额外字段会拒绝。metadata通过独立无奖励包提供给固定策略和默认回退，两个学习头继续text-only。

## 必需私有输入

- 原V2 `selector_train_inputs.npz`，包含问题ID、canonical、task/subtask和MiniLM384向量。
- 原V2 `selector_train_outcomes.npz`，真实三生成×600题×20组合聚合评分及执行账本。
- 原E8五个history/evaluation-input/metadata合法分折包。
- 原E8 `M0_original`、`M0_tuned`、`M4`的冻结预测、模型状态与内层FIT_LOG。
- 原NS的 `NS_main/PREDICTIONS_FROZEN.npz`。
- 原E8 `public/qualification/e8_assets.py`，恢复原PCA、标准化及内层划分。
- 指定阶段报告、原E5至E8.1协议、源码和manifest，用于来源核查。

完整真实相对路径、大小和SHA见 `00_asset_manifest.json`。逐题奖励、ID、模型和预测继续作为私有工件，公开交付清单排除整个 `private`。

## 防覆盖与重放

`audit_assets.py --output <新的目录>` 和 `diagnostic_signal.py --output <新的目录>` 支持独立核查输出。完整重新拟合应复制本目录的Python源码和tests到新的代码目录，排除旧生成的 `protocol*.json`，将 `MAB_BAYES_RESULTS` 设置为新的结果目录。先向新结果目录运行资产核查与数学资格，再用新代码目录的入口重新freeze、smoke、train、qualify、diagnostic、evaluate和finalize。旧规范结果不改写。数学资格的 `--output` 和两个诊断脚本的 `--output` 都应显式指向新结果目录。

相同已冻结代码与合法输入下，`run_nested_cv.py train` 会逐输入哈希检查已有拟合和预测收据。已有完整总冻结收据时不应再次执行正式训练，而应使用冻结模型进行独立评价重放。评价器拒绝覆盖其结果及奖励访问收据。

## 结果文件与边界

`03_main_results.json`包含全部策略、任务、种子、外折、27子任务、纠错破坏、动作集中度和经验选队损失。`03_question_cluster_ci.json`保存两项预定主比较及区间定义。`04_ablation_and_negative_controls.csv`保留100次完整嵌套控制。`03_cost_ledger.json`区分真实CPU、共同历史反馈和服务执行等价费用。

当前根目录没有本地Git checkout，未建立带MAB历史的本地研究分支。任务书禁止未经许可push，本轮交付均保存在新增本地目录。下一阶段若涉及新LLM或保留题，需要另立协议并获得相应授权。
