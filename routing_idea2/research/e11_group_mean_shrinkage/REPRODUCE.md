# E11 运行与资产说明

## 环境与输入

实际使用 `E:\语言模型组合路由\.venv\Scripts\python.exe`、Python3.11、NumPy1.26.4及已有合法折包加载器。全部计算为CPU，BLAS单线程。

资产根为 `E:\语言模型组合路由`。E11的私有 `SOURCE_REFERENCES.json`保存直接来源路径与SHA，公开协议保存对应整体SHA。需要保留以下原资产。

- E10的 `private/features/PREFILL_FEATURES_FROZEN.npz`及原冻结收据。
- E10五个 `private/models/fold{outer}/R_full/MODEL.npz`，含系数与PCA/Scaler。
- E10的 `private/ALL_E10_PREDICTIONS_FROZEN.npz`和预测冻结收据。
- E9的 `private/ALL_E9_PREDICTIONS_FROZEN.npz`，用于A、NS、GlobalFixed背景。
- E9现有 `fold_packs`、E8原480/120折包、原train600聚合奖励及其合法来源。
- E10纯函数 `public/code/e10_matching.py`，仅用于紧凑描述性分解。

代码只调用原预测读取、语义argmax和奖励聚合接口。完整E10评分入口由其原研究目录维护。

## 本轮实际命令

首次正式执行已完成。

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e11_group_mean_shrinkage\public\code\e11_group_mean.py' run
```

该命令求解5个组均值头、派生5个半收缩头、复用5个原头，随后统一输出两个新版本的真实点结果。已存在的预测包受不可覆盖检查保护。

保存结果的最小校验已实际通过。

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e11_group_mean_shrinkage\public\code\e11_group_mean.py' verify
```

verify核对直接复用来源、保存模型/均值/分数哈希及十方法真实事件计数，额外求解和prefill均为0。正式源码包含一次4行合成收缩线性核对，使用一组多右端求解，计数与五次正式求解分别记录。

## 另建目录重算的方法

核心模块的 `ROOT`只控制E11输出，原E9/E10来源继续通过固定 `ASSETS`寻址。需要完整再次求解时，可在Python中创建新的空目录，把公开 `E11_PROTOCOL.json`复制到新目录的 `public/`，导入原 `code/e11_group_mean.py`模块，将模块 `ROOT`设为该新目录，然后执行 `train()`和`evaluate()`。源码SHA、原资产SHA与同一协议仍须匹配。

这种重算会额外执行五次正式求解，产生新的模型及计时。本轮实际执行记录为首次正式五次求解和一次无新增求解的verify。

## 私有结果

每折保存 `GROUP_AND_HALF_MODEL.npz`、`TRAIN_GROUP_MEANS.npz`、`PREDICTIONS.npz`和简短 `FIT_LOG.json`。全局预测包、首次外折奖励访问记录、配对结果及原始进程stdout/stderr保存在E11的 `private/`。

保存的均值表只来自对应history480。真实评价task/subtask在评分后用于描述；选队服务只计算文本prefill经过固定PCA/Scaler的线性分数。

## 计数与解释

E11新增prefill、答案、聚合、gold均为0。已有train600仍代表600 gold、10800成员调用及36000组合聚合的共同历史投入。E10原606次prefill成本为缓存投资，E11增量单列。

CPU短调用可能记为0，solver墙钟保留原精度。训练阶段墙钟包括来源检查、读取和写出，完整命令进程时间另见 `private/logs/RUN_001.json`。本轮保留全部固定版本点结果，推进依据按新的用户研究决定解释。
