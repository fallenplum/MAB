# E18 v2复现补充

## 依赖与输入

使用原冻结Python3.11环境及原budget_router源码、六模型配置、typed协议和来源收据。原私有依赖清单见根目录REPRODUCE。实际运行时可通过MAB_ASSET_ROOT指向包含原src、scripts、configs、artifacts及protocol的私有资产根。测试使用临时合成题目与FakeBackend，reference测试只用临时合成JSONL/JSON。

## 纯软件资格

在本交付目录的code中运行以下入口，输出文件需使用新的路径。

```powershell
$env:MAB_ASSET_ROOT = 'E:\语言模型组合路由'
python -B -X utf8 .\e18_execution_adapter.py --describe-only
python -B -X utf8 .\qualify_execution_adapter.py --output .\execution_recheck_new.json
python -B -X utf8 .\qualify_e18_reference_adapter.py --output .\reference_recheck_new.json
python -B -X utf8 .\qualify_e18_scoring_adapter.py --asset-root $env:MAB_ASSET_ROOT --output .\scoring_recheck_new.json
```

当前最终独立复核分别为33、22、15项，通过结果及实际被测源码SHA存于本目录ROOT_FINAL2记录。初期toy来源角色、模型alias、CLI输出参数和审批拥有者路径衔接失败已保留。软件资格中的问题数不参与真实科学样本量计算。

## 批准后的执行接口

`build_scope_contract`只接受冻结输入和四策略动作并集。`E18ExecutionAdapter`构造原生job，`run_approved_task`逐key调用原Store并保存实际费用。原始本地后端工厂为`original_backend_factory`，原服务器映射后端工厂为`mapped_backend_factory`。详细参数及审批schema见EXECUTION_ADAPTER_USAGE_v2.md。

`build_reference_scope`冻结源SHA、原行定位、版本和排除身份。`E18ReferenceProvider.get_reference`提供批准范围内的单题reference。`build_scoring_scope`与`SparseScoringAdapter.score_one`连接完整执行收据和原确定性评分器。费用与失败记录读取`.ledger()`及`resource_ledger_from_execution_receipts`。稀疏评分再交给`project_frozen_policy_scores`和`power_and_statistics.paired_question_test`。

生产收据由执行者根据用户的明确批准生成，绑定固定输出拥有者根和各scope。实际主机上的模型、tokenizer映射及资源分配在执行前核对。完整多worker cohort调度尚未实现，当前交付为可审查的单任务接口及其软件资格。

## 封存核验

v1清单及其文件保持原字节。v2/PUBLIC_DELIVERY_MANIFEST_v2.json覆盖完整追加交付，逐文件检查SHA256即可回读完整性。E17公共清单及5份冻结私有工件的核验方法保留在原verify_and_seal_e18.py与e18_common.preserve_e17中。准备构建脚本采用exclusive-create模式，在同一封存目录重复执行会拒绝覆盖。
