# E18 完成阶段增量源码交付

本目录保存已完成的371尾分片控制、只读来源审计及CPU scope摘要复用源码。最终实验结果由正式评分完成后的脱敏汇总另行补入。

## 发布合并

以现有提交`f5e7ac42f297e7957c52f11f3bc659842d8b76bc`为已发布基础，保留原95份实验源码及40份操作层归档。将`PUBLICATION_MANIFEST.json`各行的`package_path`映射到其`publication_path`，加入同一E18研究分支。

本次合并15份已有小文件，包括两个来源overlay的11份文件，以及CPU scope memo的4份文件。两个原overlay索引保留原字节，分别保存为`INDEX_TAIL371_EXTENSION.json`和`INDEX_TAIL371_READONLY_AUDIT.json`。

发布前由主执行进程追加正式结果和最终快速评分入口，并记录实际commit及远端核验结果。此目录当前状态为`SOURCE_STAGED_AWAITING_ACTUAL_RESULTS_AND_ROOT_PUBLICATION`。

## 已有软件记录

| 模块 | 已记录测试数 | 记录状态 | 检查范围 |
|---|---:|---|---|
| 371尾分片扩展 | 19 | PASS | 明确parent白名单、既有控制层继承与分区身份 |
| 371只读审计 | 13 | PASS | 元数据来源与cache标记双向字节转换 |
| CPU scope memo | 22 | PASS | 人工四题fixture下原路径与包装路径的语义结果、只读快照与完整性 |

本次仅整理已有记录，测试重跑数为0。上述软件记录支持实现资格，实际3000题服务质量由正式评价结果报告。

## 源码入口和依赖

- `member4_tail371_broker.py`继承原`member4_tail_partition.py`，对应已批准的371任务尾分片。
- `incremental_aggregation_planner_v6.py`继承已冻结v5，明确接纳原512与371两个parent来源。
- `audit_tail371_static_sources.py`和`audit_tail371_export_lineage.py`复用已发布`audit_tail_export_lineage.py`。
- `cpu_scope_memo.py`在原CPU运行目录调用原`e18_final_scoring.py`，保留原适配器、评分及reference组件接口。
- `qualify_cpu_scope_memo.py`依赖原`qualify_final_scoring.py`的人工fixture。完整依赖由前述已发布95份源码及40份操作层归档提供。

现有人工软件资格命令保存在各模块README中。正式评价的执行入口以主执行进程追加的最终快速context入口为准。

## 费用与结果边界

本次源码整理打开的文件限于清单列明的源码、使用说明和已脱敏软件记录。整理过程LLM调用、Gold读取、真实奖励读取、私有native归档审计及GitHub操作均为0。

原题、答案、逐题矩阵、权重、私有运行收据、任务和物理尝试标识保存在私有资产目录。实际Token、成员调用、聚合调用、准确率、纠错与破坏由最终脱敏结果提供。后续结果遵循项目约定，保留点值和分组分解。

## 机制、优势与风险

本次增量源码保留原科学worker和评分器的冻结版本，操作层扩展及性能包装具有单独的源码身份和已有软件记录。优势是来源可追溯且原实现可继续复用。完整复现仍依赖原环境和授权私有资产，软件资格与源文件身份不能代替真实结果的科学判断。
