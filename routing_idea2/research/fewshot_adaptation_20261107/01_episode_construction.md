# 01 少样本适应任务构造

BBH 27个子任务各自成任务环境，GSM8K作为单一任务。每个episode按问题ID分Support/Query；三个语言模型生成seed在episode内保持重复测量。已见任务使用同任务历史半区和新题池；未见任务模式完全排除目标子任务评分。k=0/1/2/4，支持抽样seed为11/29/47/71。

主分析使用同一生成seed的Support反馈更新同一生成seed的Query；充分反馈读取已有真实20组合矩阵，受限反馈仅使用默认组合已执行评分。validation仅开发诊断，fresh400/sealed200未打开。完整episode计数和逐k/任务汇总见EXPERIMENT_RESULTS.json。


## 可行性边界
validation中的BBH子任务只有3—4道题，因此k=4在BBH没有Query，所有k=4汇总行实际只覆盖GSM8K；该行不用于BBH少样本结论。
