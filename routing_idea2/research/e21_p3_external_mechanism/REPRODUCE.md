# P3 复现

本文目录在仓库中放在 `routing_idea2/research/e21_p3_external_mechanism/`。与 E18 原输入投影、E19 原凸优化及五折缓存同级。使用 `P3_PROTOCOL.json` 的原字节，运行时传入其 SHA-256。两条 runner 均分为 `fit` 和 `evaluate`，第二步要求第一步的预测和配置已冻结。

## 原域损失对照

`run_internal.py --phase fit --protocol P3_PROTOCOL.json --protocol-sha256 <SHA> --output-root <空运行根>`。输入为 E19 的 train600 五折私有输入包及评分矩阵，不使用原 P2 的新题缓存。对同一目录运行 `--phase evaluate` 写出 `public/P3_INTERNAL_LISTNET_RESULTS.json`。沿用 E19 原折预处理及逐折正则，来源摘要在产物中。需要 Python 3.11、NumPy 1.26 和 SciPy 1.11。

## 外部单模型池

从 [DARS 固定提交](https://huggingface.co/datasets/AIGNLAI/DARS/tree/d7bdd3638d263986f32ee928a19e0e2868e69bab) 获取 `gpqa`、`math-500` 各自的 `train_scored_generations.jsonl` 和 `test_scored_generations.jsonl`，在私有目录按任务子目录排列。下载不调用评分模型或生成 API；任何镜像所得文件须与 `run_external.py` 所列 SHA-256 一致。

`run_external.py --phase fit --protocol P3_PROTOCOL.json --protocol-sha256 <SHA> --data <DARS目录> --run <空私有运行根> --output <新汇总JSON>`。使用 P2 的 MiniLM 本地环境及 E19 Clarabel 依赖，必要时显式给出 `--asset-root`。对相同参数运行 `--phase evaluate` 才解码正式测试分数。训练只用固定改写的重复生成，主评估用原题温度 0.7，其他温度为独立配置诊断。原题、模型输出、逐题数组和权重不进入公开包。

两条路径均使用原题为单位，主解码为 `tau=1e-8` 的均匀 argmax，raw-exact 预先固定为敏感性诊断。论文定位与结果见 `P3_RESULTS.md`。
