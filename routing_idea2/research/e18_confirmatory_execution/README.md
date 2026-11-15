# E18 通用执行代码与软件资格

**当前研究状态为RUNNING。** 本包提供通用worker、dispatcher、增量聚合planner、queue daemon及对应的软件资格记录。

所有源码逐字节复制自当前冻结源码，文件SHA见INVENTORY.json。源码中的参数化引用接口与完整性哈希守卫保持原值。软件PASS只说明记录的合成软件场景通过，正式研究结论依赖完整实验与冻结评价。

## 运行依赖

运行期的scope、授权、任务分片、加载映射、数据身份、输入、答案、权重及native缓存另行受控保存。调用者通过原接口提供这些依赖，本包未携带真实运行配置。

通用GPU与聚合资格脚本依赖既有E18适配器、qualify_execution_adapter测试fixture及原V2资产，需要按原源码的相对模块关系提供。增量planner的toy资格使用标准库与合成临时文件，可单独检查。

```bash
python -B code/qualify_incremental_aggregation_planner.py --output synthetic_qualification.json
```

## 文件范围

code包含12份原样源码，software_pass包含5份纯软件PASS记录，共85项已记录测试。聚合v2源码供v3资格中的函数一致性检查使用。INVENTORY.json逐文件记录大小与SHA，PRIVACY_SELF_REVIEW.json只保存脱敏核查计数。

本包的优势是保留源码可追溯性并缩小公开信息范围。完整真实重跑仍依赖受控资产，软件资格结果不能替代实际GPU、费用或质量评价。
