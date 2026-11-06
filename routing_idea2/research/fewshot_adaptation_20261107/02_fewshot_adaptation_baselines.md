# 02 少样本适应基线

比较任务/子任务固定、冻结Ridge、在线Ridge、层级收缩、距离加权KNN、2/4维低秩适应和随机Support负对照。所有方法使用同一历史、Support和Query；支持反馈与查询奖励严格分离。

结果按k、任务环境、生成seed报告准确率、Top-1 Regret、更新前后差值、固定策略差值、纠错/破坏和负迁移比例。充分反馈与受限反馈分开，不将完整矩阵标签当作受限反馈。

本轮仅CPU复用已有矩阵，未新增LLM调用；由于BBH每子任务只有11—12个训练题、验证题3—4个，局部episode方差较大，不能替代fresh独立测试。


## 任务分解与配对区间
validation200的逐BBH子任务、GSM8K任务级结果和Support/Query episode聚类95%区间均写入EXPERIMENT_RESULTS.json。区间先在同一episode内平均三个生成seed，再按任务分层bootstrap；不把生成seed当独立任务。validation BBH每个子任务只有3—4道题，k=4对BBH不可行，k=4行只包含GSM8K。


## 关键开发诊断

充分20组合反馈在validation未见子任务k=1/2没有稳定正向；受限单组合反馈中，层级更新k=2的episode聚类均值差约+0.94pp，95%区间约[+0.05,+1.83]pp，但这不是独立确认性证据。随机Support负对照在受限k=1约为负，k=2区间跨0；因此不能把所有收益归因于参数更新。BBH validation k=4不可行，k=4记录只由GSM8K组成。

**决策：暂不进入元学习。** 当前仅有单一权限/k/方法的开发正向信号，full-information主诊断没有稳定收益，需先用冻结普通适应候选做独立fresh400确认。
