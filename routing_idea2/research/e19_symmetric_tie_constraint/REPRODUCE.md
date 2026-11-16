# E19复现与交付清单

## 固定范围

E19以E18.1提交`341a8a3c48ee95b7a38c1a5713ac659fccba8f6c`为基点，新分支为`codex/e19-symmetric-tie-constraints`。原E18提交为`95c6c8a38a71866d6efdb224aec04b9a158adf17`。导师PDF的SHA为`862d0118f58240b7051790846475248b2154ec4198e4bb45afdfc0ed10607fb2`。

本轮实际完成P0→P1，保持原六模型、K=3、20组合、MiniLM/PCA16/标准化及17×20线性头。真实数据拟合为10次主拟合和2次反向编号资格拟合。理论构造和人工软件检查的小维QP另记。新增LLM调用、编码器前向、fresh400奖励读取、sealed200读取和P2/P3/P4执行均为0。

## 数学目标

全部经验最优集合使用均匀标签，γ=0。B取β=1，C取β=0。只有训练折中20组合经验均值完全相同的行改变目标系数。全部480训练行、480个epigraph变量和1/480分母保留，其他行的奖励和集合标签相同。

真实目标为平均数据项加`regularization*||W||_F²`，包括截距行。原冻结系数在fold0–3为0.003472222222222222，在fold4为0.00034722222222222224。真实系数与理论三输入例子的λ=60各自沿用其参数化。两支使用同一系数与惩罚算子，E19没有参数搜索。

`E19_PROTOCOL.json`最终SHA为`d9715da6b15fe5d35d34c4ec876cfd613284373b2a1d792674248c3b9348269c`。它在首个真实拟合之前绑定run_e19.py与weighted_objectives.py源码。预测与逐题概率先写入封存NPZ和收据，然后才解码对应外折评价奖励。

## 数值推理定义

预先声明的主策略将与最大分数相差不超过1e-8的动作作为数值可分辨精度下的argmax集合，在集合内均匀分配概率。通过概率加权缓存奖励精确计算服务期望，执行中的随机动作采样次数为0。

原始浮点数严格相等的argmax集合均匀策略另存为预声明敏感性诊断。它的C−B区间跨零。主策略的正区间与这项边界同时呈现。严格胜者改变仅计B和C的top1−top2均大于1e-8、且真实胜者不同的题。

全题纠错和破坏为最大同动作共享耦合下的期望次数，允许小数。共同概率质量使用相同动作，其余不相交剩余概率独立配对。净期望差等于两支服务期望之差。严格换队150题的38次纠错与19次破坏另按三生成记录给出整数。耦合定义影响纠错/破坏拆分，主效用差由边际概率确定。

## 依赖与私有输入

执行环境为Python3.11.14、NumPy1.26.4、SciPy1.11.4、Clarabel0.11.1。绘图使用Pillow12.0.0。主Bootstrap复用前一提交E18.1的bootstrap_means和interval函数，需保留相邻`e18_1_statistics_and_mechanisms/analyze_e18_1.py`，源码SHA为`34b91f1943d2c40d2bbab0ce236d7a32e89772abe27b84386cb9e80a8bde3204`。

私有输入均已存在。

- train600问题输入`selector_train_inputs.npz`，SHA为`57e39747ec2756bcd79ab00885b3c08ad4f91fdb3ec900cbed15146420cd17c3`。
- 三次生成的完整组合奖励`selector_train_outcomes.npz`，SHA为`a01230f2bed3e7f9d05713bf9e3e4f1cf44987b41c460f91e1d83bc1c29f191e`。
- E13的五个`INPUTS_ONLY.npz`提供冻结H480/E120与纯MiniLM预处理后的phi_h/phi_e。只使用许可字段，旧任务组标签未进入线性头。
- `FROZEN_TSPO_REUSE_AUDIT.json`记录原折身份、PCA/scaler逐数组核对及各输入packet的SHA。

奖励读取器核对NPY头为C-order、float64、形状3×600×20。拟合前仅把H行解码为数值，其他行作为原始字节跳过。该折预测封存后才允许读取E行。五折恰好覆盖原600题一次。

## 实际入口与顺序

下列PowerShell命令用于拥有私有资产的原实验工作区。复现应指定全新的studyRoot以保留旧输出。原P0/P1已经完成，命令记录其真实入口。

```powershell
$pythonPath = 'E:\语言模型组合路由\.venv\Scripts\python.exe'
$codeRoot = 'D:\develop\CV\多臂老虎机\experiments\e19_symmetric_tie_constraint'
$studyRoot = 'E:\语言模型组合路由\research\e19_symmetric_tie_constraint_reproduction'
$protocolPath = "$codeRoot\E19_PROTOCOL.json"
$protocolHash = 'd9715da6b15fe5d35d34c4ec876cfd613284373b2a1d792674248c3b9348269c'

& $pythonPath -B -X utf8 "$codeRoot\run_e19.py" `
  --phase p0 --protocol $protocolPath --protocol-sha256 $protocolHash --study-root $studyRoot

# 仅在P0_QUALIFICATION.json记录PASS_P0_GATE后执行。
& $pythonPath -B -X utf8 "$codeRoot\run_e19.py" `
  --phase p1 --protocol $protocolPath --protocol-sha256 $protocolHash --study-root $studyRoot

& $pythonPath -B -X utf8 "$codeRoot\analyze_e19.py" `
  --study-root $studyRoot --protocol $protocolPath

& $pythonPath -B -X utf8 "$codeRoot\export_training_masks.py" --study-root $studyRoot

& $pythonPath -B -X utf8 "$codeRoot\render_e19_reports.py" `
  --results "$studyRoot\public\E19_COMPLETE_RESULTS.json" `
  --protocol $protocolPath --review "$codeRoot\NUMERICAL_AND_ACCESS_REVIEW.json" `
  --output-root "$studyRoot\public"
```

P0为fold0的B/C及各自一次反向编号拟合，共4次。其余四折各2次，共8次。P1读取并复用P0的fold0封存结果。每折均记录完整目标、F0、JT、非T数据项、正则项、KKT、互补和对偶间隙。P0的中心化分数与均匀动作概率均核验真实动作回映射。

## 统计与机制文件

主统计为C−B的10,000次任务分层原题配对Bootstrap，随机种子20261116019，GSM8K与BBH各占1/2。三生成保持原题簇，BBH子任务宏平均单列。五个训练历史重叠，区间是给定拟合模型的OOF开发不确定性。

E19_COMPLETE_RESULTS.json包含主要点差、区间、strict/tie分解、每折目标资格与成本。逐题完整20维分数、概率、margin及W位于私有`fold*/...PREDICTIONS.npz`。私有OOF_SCORES_AND_PROBABILITY_CHANGES.npz、PER_QUESTION_SCORE_DIAGNOSTICS.csv和BOOTSTRAP_REPLICATES.npz供作者复核。每折TRAINING_LABELS_AND_MASK.npz还保存确切H行、经验均值、均匀最优集合标签、全并列T mask及两支行权重；export_training_masks.py只复读合法H行生成这些派生工件，没有追加拟合。

NUMERICAL_AND_ACCESS_REVIEW.json记录独立的访问顺序、目标和抽样复核。HISTORICAL_NUMERICAL_RECOVERY.json保留新旧求解精度的差异。本轮B采用统一更紧容差重新拟合，历史E17概率未被宣称逐题精确恢复。

## 理论、图与论文稿

01_THEORY_AND_PREDICTIONS.md完整证明两个构造，THEORY_CHECKS.json记录精确算术与独立小维数值验证。该收据先于真实源参数审计产生，其中实际路由正则字段保留当时待核状态；最终真实参数已经由E19_PROTOCOL.json与FROZEN_TSPO_REUSE_AUDIT.json核定，理论验证本身不依赖该字段。`theory_checks.py --output <新文件>`可重新运行构造验证。严格整数重复阈值为`max(1,floor(h)+1)`，理论中的重复数没有用于增加真实调用。

PAPER_THEORY_RESULTS_DISCUSSION.md为待作者审阅的Theory、Results和Discussion草稿。E17/E18/E19联合图的源、SVG、PNG和图注位于figures目录，重现入口为make_joint_mechanism_figure.py。本机draw.io CLI缺失，图由同一可编辑场景本地渲染，原生导出一致性尚待相应环境复核。

## 停止边界

本轮B/C对照只改变已冻结的β端点。预定数值策略下的正开发区间、raw-exact的跨零区间及历史数值精度差异全部保留。严格动作变化提供当前实现中的机制线索，独立服务收益仍需单独确认。P2/P3/P4、参数网格与新的语言模型调用均维持本轮范围之外。
