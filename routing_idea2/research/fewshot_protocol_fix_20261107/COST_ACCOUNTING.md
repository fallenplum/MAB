# 完整反馈与服务成本

## 研究问题

反馈权限是否通过隐藏的成员或聚合调用获得额外优势。

## 实验设置

本轮只执行CPU训练和真实缓存重放，新增LLM、成员生成、聚合、评价调用均为0。

下表是单个部署episode的反事实成本。每环境Support提供k道不同题的gold，full20为6k次成员与20k次聚合，single为3k次成员与k次聚合。gold均为k。历史训练矩阵视作共同已付成本。

CPU统计为适应更新和预测的实测合计除Query数，历史PCA与基础拟合不包含在该项。GPU时间为原缓存调用耗时之和，不能用来预测新硬件延迟或并发墙钟。

## 对照与数值结果

|配置|方法|每Query prompt tokens|每Query completion tokens|每Query缓存GPU秒|每Query更新预测CPU毫秒|
|---|---|---:|---:|---:|---:|
|seen/full20/k1|no_update|1361.45|789.65|39.7857|0.0013|
|seen/full20/k1|fixed|1350.85|767.55|40.2797|0.0031|
|seen/full20/k1|online_ridge|1360.53|788.39|39.7193|0.1200|
|seen/full20/k1|hierarchical|1355.74|763.37|38.4786|0.0097|
|seen/full20/k1|knn|1360.43|780.79|39.3568|0.6177|
|seen/full20/k1|adapter2|1360.01|763.70|38.6285|0.0227|
|seen/full20/k1|adapter4|1359.64|758.27|38.2164|0.0176|
|seen/full20/k2|no_update|1361.45|789.65|39.7857|0.0006|
|seen/full20/k2|fixed|1350.85|767.55|40.2797|0.0016|
|seen/full20/k2|online_ridge|1360.34|787.92|39.6844|0.0580|
|seen/full20/k2|hierarchical|1351.32|754.39|37.9722|0.0049|
|seen/full20/k2|knn|1359.92|773.10|38.9932|0.2130|
|seen/full20/k2|adapter2|1352.02|753.28|38.0730|0.0112|
|seen/full20/k2|adapter4|1357.52|748.51|37.8995|0.0067|
|seen/single/k1|no_update|1361.45|789.65|39.7857|0.0004|
|seen/single/k1|fixed|1350.85|767.55|40.2797|0.0009|
|seen/single/k1|online_ridge|1359.93|785.42|39.5887|0.0234|
|seen/single/k1|hierarchical|1353.17|774.29|39.0627|0.0027|
|seen/single/k1|knn|1357.78|781.22|39.3030|0.0172|
|seen/single/k1|adapter2|1351.38|763.91|38.6459|0.0055|
|seen/single/k1|adapter4|1348.16|732.43|37.1711|0.0039|
|seen/single/k2|no_update|1361.45|789.65|39.7857|0.0006|
|seen/single/k2|fixed|1350.85|767.55|40.2797|0.0015|
|seen/single/k2|online_ridge|1361.29|787.66|39.6610|0.0505|
|seen/single/k2|hierarchical|1349.43|767.27|38.7702|0.0047|
|seen/single/k2|knn|1355.43|774.23|38.9291|0.0324|
|seen/single/k2|adapter2|1337.27|730.22|37.0412|0.0096|
|seen/single/k2|adapter4|1344.87|726.23|36.9719|0.0068|
|unseen/full20/k1|no_update|1352.12|767.13|39.1217|0.0004|
|unseen/full20/k1|fixed|1370.25|835.06|43.0661|0.0010|
|unseen/full20/k1|online_ridge|1352.46|768.38|39.1625|0.0332|
|unseen/full20/k1|hierarchical|1357.15|769.42|39.0078|0.0031|
|unseen/full20/k1|knn|1366.67|760.85|38.7036|0.1347|
|unseen/full20/k1|adapter2|1354.45|750.71|38.0840|0.0069|
|unseen/full20/k1|adapter4|1346.53|734.33|37.3198|0.0040|
|unseen/full20/k2|no_update|1352.12|767.13|39.1217|0.0006|
|unseen/full20/k2|fixed|1370.25|835.06|43.0661|0.0016|
|unseen/full20/k2|online_ridge|1352.22|769.69|39.2082|0.0613|
|unseen/full20/k2|hierarchical|1356.34|765.18|38.7193|0.0052|
|unseen/full20/k2|knn|1364.00|753.82|38.4572|0.2142|
|unseen/full20/k2|adapter2|1352.44|749.66|37.9885|0.0113|
|unseen/full20/k2|adapter4|1350.30|738.50|37.5293|0.0072|
|unseen/single/k1|no_update|1352.12|767.13|39.1217|0.0003|
|unseen/single/k1|fixed|1370.25|835.06|43.0661|0.0009|
|unseen/single/k1|online_ridge|1352.55|767.49|39.1425|0.0234|
|unseen/single/k1|hierarchical|1351.69|754.01|38.6922|0.0028|
|unseen/single/k1|knn|1357.35|763.16|38.9944|0.0172|
|unseen/single/k1|adapter2|1337.88|739.11|37.6147|0.0056|
|unseen/single/k1|adapter4|1346.86|735.49|37.5282|0.0036|
|unseen/single/k2|no_update|1352.12|767.13|39.1217|0.0006|
|unseen/single/k2|fixed|1370.25|835.06|43.0661|0.0014|
|unseen/single/k2|online_ridge|1352.60|767.62|39.1449|0.0513|
|unseen/single/k2|hierarchical|1354.30|757.61|38.9491|0.0041|
|unseen/single/k2|knn|1359.09|767.16|39.1285|0.0331|
|unseen/single/k2|adapter2|1339.95|736.97|37.4431|0.0090|
|unseen/single/k2|adapter4|1339.44|721.42|36.8474|0.0060|

## 统计结论

每条服务决策固定3成员加1聚合。Support成本与Query服务成本分列。无更新和固定策略可以不购买Support，表中的共同Support预算只是实验对齐额度。

当前成本文件仅报告三个同种子主对角的服务成本。跨种子六方向用于随机性诊断，其服务成本没有混入主对角账单。

## 失败原因与风险

历史矩阵的48,000条组合评分可共享其成员生成与gold。把每个组合评分重复计为一次新gold，或者把穷举的互斥Support部署episode相加，都会夸大费用。

## 下一步

fresh先购买冻结Support动作，再冻结Query选择并计算合法缓存并集，之后才允许读取Query评分。
