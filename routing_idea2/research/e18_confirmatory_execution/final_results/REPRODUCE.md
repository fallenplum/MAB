# E18 最终部分可评分报告复现

最终评分状态为`COMPLETE_WITH_UNSCORABLE_INPUTS`。原计划3000道独立问题，可评分2998道，GSM8K1500道、BBH1498道。BBH中ruin_names与movie_recommendation各排除1道，公开缺失原因码为`CHOICE_REFERENCE_HAS_NO_UNIQUE_VISIBLE_OPTION`。排除题奖励保持未知，全部原计划范围实际调用及Token费用保留。

报告主表在可评分问题内按GSM8K与BBH各50%计量。另以原计划每任务1500题为分母，保存缺失结果的确定性最坏和最好范围。每题三个生成重复保留在同一原题簇内。

## 完成资产与源码

| 资产 | SHA256 |
|---|---|
| 完成partial结果JSON | `86a91632c460085db016bf7718b56cd58fc40808f39a487d898c0dce080db41f` |
| 私有有效题数值数组 | `5c8c8702ba1dd5b74474ec69475806a7325213e6b3ac7664e16f7a9de397e70a` |
| generate_point_only_report_v3.py | `22f6f9491d140724752ed0618afe7de7e290e516949e7b99b19f105240cbc461` |
| 最终partial评分入口 | `12d73edda4bfee0c184eb595bf4e714b32b023e5497c07462f011503db13b2a1` |
| 继承bulk评分入口 | `812e72bed58ae6ef2aa50fd4464318965d3a6223f67da46afb45b75d89e3afd0` |

## 实际报告命令

使用原Python与NumPy环境。完成结果和数值数组位于各自授权资产目录。

```powershell
$e18Code = '<E18完整源码目录>'
$e18Result = '<已完成partial E18_FINAL_RESULTS.json完整路径>'
$e18Rewards = '<已完成FINAL_POLICY_REWARDS.npz完整路径>'

python -B -X utf8 "$e18Code/generate_point_only_report_v3.py" `
  --results "$e18Result" `
  --results-sha256 86a91632c460085db016bf7718b56cd58fc40808f39a487d898c0dce080db41f `
  --rewards "$e18Rewards" `
  --rewards-sha256 5c8c8702ba1dd5b74474ec69475806a7325213e6b3ac7664e16f7a9de397e70a `
  --output-root '<全新报告目录>'
```

本次命令已实际生成8份报告、JSON和CSV。报告从有效题奖励复算四策略质量、配对点差和纠错计数，并与完成结果核对。来源数组含有效mask，排除题没有补零。

v2报告源按预先约定接口编写，首次调用在schema检查处停止，实际partial输出采用独立schema及12d入口身份。v3精确适配上述实际完成格式。评分结果、冻结动作和旧报告源码保留各自身份。

## 交付范围

`INDEX.json`列明8份实际脱敏报告及最终v3源码的发布路径和SHA。结果映射至研究目录的`final_results/`，新源码映射至`final_sources/code/e18_confirmatory_execution/`。主执行进程另行提供研究结论、错误记录与实际运行费用说明，并统一完成公开范围核对和推送。

逐题数组、题目标识、原始输出、答案、评分收据及权重保存在私有边界。报告生成新增模型与评分调用、Gold与sealed读取以及大归档审计均为0。
