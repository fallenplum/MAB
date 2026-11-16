# P2复现与运行入口

## 环境

输入编码使用`E:\语言模型组合路由\.venv\Scripts\python.exe`。版本以requirements_features.txt、P2_ENCODER_ENVIRONMENT.json及P2_ENCODER_RECEIPT.json为准。Transformers4.51.3和Tokenizers0.21.4为原冻结MiniLM依赖。一次D盘通用环境启动在版本核验时拒绝，编码前向0次，完整情况见ENVIRONMENT_CLARIFICATION.md。

原本机代码目录为`D:\develop\CV\多臂老虎机\experiments\e20_p2_execution`。GitHub研究分支中对应目录为`routing_idea2/research/e20_p2_execution`。源码依赖同级E18准备目录和E20最终模型准备目录，保留父提交中的原操作符与白名单解析代码。

私有资产根目录为`E:\语言模型组合路由`。输入程序支持MAB_ASSET_ROOT环境变量。正式GPU工作器使用服务器已核验Python环境`/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python`，服务模型、生成参数和聚合算子沿用原E18冻结逻辑身份。

## 实际准备顺序

先固定P2_PRESELECTION_PROTOCOL.json。数据脚本依次执行select、encode和predict，随后冻结答案定位元数据。下列命令从本机代码目录运行。

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 prepare_independent_pool.py --phase select
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 prepare_independent_pool.py --phase encode
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 prepare_independent_pool.py --phase predict
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 prepare_reference_locators.py
```

这些入口已实际执行。输出采用不可覆盖写入，原目录重复运行会拒绝。完整从零复现应使用隔离私有资产副本及对应运行目录。模型继续加载E20已有最终600题权重，脚本不提供路由器重训练阶段。

## 输入与执行资格

科学scope位于私有EXECUTION_SCOPE.json。资格命令只读取输入侧预测和作用域，不打开答案。

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 qualify_blind_evaluation.py
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 blind_evaluation.py --scope 'E:\语言模型组合路由\research\e20_p2_execution\private\EXECUTION_SCOPE.json' --predictions 'E:\语言模型组合路由\research\e20_p2_execution\private\P2_PREDICTIONS_FROZEN.npz' --output 'E:\语言模型组合路由\research\e20_p2_execution\private\P2_SCOPE_RECHECK.json'
```

prepare_runtime_sources.py根据既有E18源码构建P2概率scope适配器，prepare_launch.py将冻结输入、概率、执行键和实际算子映射绑定为运行清单。生产打包入口无额外CLI选项，其输入来自私有冻结路径。

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 prepare_runtime_sources.py
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 prepare_launch.py
```

运行scope位于`private/runtime_launch_v2/inputs/EXECUTION_SCOPE_RUNTIME.json`。启动包及清单位于私有P2_PRODUCTION_INPUTS_AND_MANIFESTS_v2.zip，输入阶段产生的科学scope与运行scope经过并集一致性核验。重建已有启动包时，其不可覆盖保护仍生效。

## 服务器工作器CLI

两个生产入口均要求`--manifest`、`--manifest-sha256`和`--gpu`。manifest必须显式标记生产执行并匹配已批准的P2作用域。实际manifest哈希由P2_PRODUCTION_LAUNCH_RECEIPT_v2.json或后续运行收据提供，GPU编号依据实时进程、显存和利用率选定。

```bash
/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python -B /data2/chendong/e20_p2_execution/code/p2_member_worker.py --manifest /data2/chendong/e20_p2_execution/manifests/member0_v2.json --manifest-sha256 "$P2_MEMBER_MANIFEST_SHA" --gpu "$P2_GPU"
/data1/data2/chendong/routing/envs/inference_311_matched_v2/bin/python -B /data2/chendong/e20_p2_execution/code/p2_aggregation_worker.py --manifest /data2/chendong/e20_p2_execution/manifests/aggregation_v2.json --manifest-sha256 "$P2_AGG_MANIFEST_SHA" --gpu "$P2_GPU"
```

上述变量来自该次已核验收据及资源安排，示例不指定固定GPU。成员输出复用按问题、真实模型及seed核对，聚合输出复用按问题、真实组合、seed及算子身份核对。失败和重试保留物理成本。

## 完整输出后的盲评入口

score_paid_outputs.py的必需参数为`--scope`、`--predictions`、`--outputs-frozen`、`--aggregate-records`和`--output-root`。默认只做资格检查。真正评分还需`--execute-scoring`、`--reference-plan`和`--approval`，且必须先通过全部输出冻结及来源资格。P2_REFERENCE_LOCATORS_FROZEN.json提供批准2400题的官方源位置。sealed200答案继续排除。

blind_evaluation.py在投影阶段使用`--execute-projection`、`--score-rows`和`--outputs-frozen`，将已合法付费执行的稀疏组合评分按冻结概率精确求期望效用。评分与投影均不更换策略、参数、题池或评价权重。

## 私有依赖与审查限制

需要原官方数据文件、完整历史排除索引、sealed单向指纹、E20最终权重与预处理状态、冻结MiniLM、E18逻辑模型映射和既有推理源码。逐题身份、文本、向量、概率、原生输出及答案均为私有依赖。

2400题包含BBH共同子池39题和官方源扩展1161题，来源层需分别报告。固定近重复规则不能保证完全语义独立，输入侧编号及严格换队诊断也不直接提供质量结论。结果不计算新置信区间或仅用于区间的bootstrap。
