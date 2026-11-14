# 复现

## 依赖与私有工件

Python 3.11、NumPy、SciPy、scikit-learn、psutil，使用原MAB环境。需要E13完整private/worlds、inputs、冻结收据和public/code，以及原train600输入/输出缓存、原SaveRouter固定源码。上述私有数据不在公开仓库内。

设MAB_ASSET_ROOT指向语言模型组合路由根目录。MAB_E14_ROOT可以指向一个全新输出目录。MAB_E13_CODE可显式指向原E13源码目录。模块从E13复用原预测器与SaveRouter，需同时保留两个阶段源码。

## 执行顺序

```powershell
python -B -X utf8 code/run_e14.py prepare
python -B -X utf8 code/run_e14.py freeze
python -B -X utf8 code/evaluate_e14.py
python -B -X utf8 code/finalize_e14.py
```

prepare执行必要资格并锁定输入与源码；freeze仅使用原已购买H区反馈和实际执行账本，生成A干预动作及C原学习器前缀快照。全部动作封存之后才允许评价器读取原train600真实奖励。B仅在各折E120上计算误差。阶段文件采用不可覆盖写入。

## 成本与输出

private保留逐题诊断、原题索引、原能力表来源、重新拟合权重、反馈前缀、完整账本和所有冻结收据。public只有脱敏汇总及复现源码。C_TOKEN_MATCHES保留完整调用资源、实际Token与离散剩余，历史已执行计量成本与E14新增CPU成本分别记录。

置信区间、bootstrap、假设检验和新增物理LLM/API调用执行数均为0。fresh400和sealed200不读取。原题级相关性按600个原始问题解释，重复日志及生成seed不增加独立题量。
