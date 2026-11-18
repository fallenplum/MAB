# 原系统源码与冻结论文实验

从仓库根目录的 [README](../README.md) 开始阅读；论文结果见 [paper/RESULTS.md](../paper/RESULTS.md)。

- `research/e17_*` 到 `research/e23_*`：当前论文主线，保存各阶段代码、协议、报告、结果与执行账本。E23 增加部署并列规则和同题池参照比较。
- `src/`、`scripts/`、`configs/`、`protocol/`、`data/cache_audit/` 和 `tests/`：原系统共享实现及验证代码；保留这些文件不代表每个旧实验都是当前论文贡献。
- 其他五项早期来源文件：供后续阶段导入或核对协议/源码哈希，详见 [复现说明](../docs/REPRODUCIBILITY.md)。早期完整实验请查看 [分支索引](../docs/BRANCHES.md)。

本目录原始文件保持冻结字节，包括历史路径和行尾。不要在此批量替换盘符、分支名或重新格式化源码；新适配应单独版本化并记录差异。`code/` 目录是归档布局，不能假定与原执行资产根相同。

共享 `requirements-lock.txt` 记录的是历史 Windows 推理环境，不能代替所有后续阶段的数值与特征环境。请先阅读 [环境及资产要求](../docs/REPRODUCIBILITY.md)，不要直接运行旧调度器或覆盖冻结输出。
