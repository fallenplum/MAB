# 源码与实验

[项目首页](../README.md) · [全部实验](../EXPERIMENTS.md) · [论文结果](../paper/RESULTS.md) · [复现说明](../docs/REPRODUCIBILITY.md)

## 本目录内容

- `research/e17_*` 至 `research/e23_*` 保存当前论文主线的代码、协议、报告、结果和调用记录。
- `src/`、`scripts/`、`configs/`、`protocol/`、`data/cache_audit/` 和 `tests/` 保存原系统的共享实现、配置和检查程序。
- 早期实验的完整记录保存在各自分支，报告、结果和运行说明见 [全部实验目录](../EXPERIMENTS.md)。本目录保留了后续阶段需要导入或核对的早期文件。

## 运行说明

各阶段的依赖、输入路径、参数和执行步骤见对应实验的 `REPRODUCE.md` 或说明文件。原文件内容与实验提交一致，文件校验值见 [论文文件清单](../reproducibility/frozen_manifest.json)。

共享 `requirements-lock.txt` 记录早期 Windows 推理环境，后续阶段使用各自记录的数值计算和表征环境。环境版本、数据和模型文件的位置见 [复现说明](../docs/REPRODUCIBILITY.md)。
