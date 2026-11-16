# E20复现入口

## 环境与目录

使用原已安装Python环境及requirements.txt版本。原本机入口为`D:\Anaconda\envs\tslib\python.exe`。在GitHub分支checkout内进入`routing_idea2/research/e20_independent_confirmation_preparation/`。原D盘运行目录为`D:\develop\CV\多臂老虎机\experiments\e20_independent_confirmation_preparation/`。

以下命令均从E20源码目录执行，`python`指上述冻结环境。模型拟合与预测接口具有不可覆盖保护，已有私有冻结工件在同一路径重复执行会拒绝。

从GitHub checkout复核本机E18账本时，先将`MAB_WORKSPACE_ROOT`设为原D盘工作区，确保元数据账本路径可定位。该变量只用于审计文件定位。

```powershell
$env:MAB_WORKSPACE_ROOT = 'D:\develop\CV\多臂老虎机'
```

## 已完成工件的只读资格复核

```powershell
python -B -X utf8 test_preparation_guards.py
python -B -X utf8 review_prediction_operator.py
python -B -X utf8 audit_final_models.py
```

第一条仅使用小型合成输入，后两条读取已完成模型、历史训练标签和输入侧预测。均不训练新模型、不读取fresh奖励或sealed答案。复核脚本更新独立审查输出，正式协议与原模型收据应先保留以作比较。

## 原实际执行顺序

1. 写入E20_PREPARATION_CONTRACT.json，核对原600题正则来源。
2. 执行`fit_final_models.py --phase fit`，生成两支最终模型与训练前后收据。
3. 对fresh400已有输入包执行原输入预测，并以正式数值算子回放。
4. 执行隔离输入投影、冻结编码、object元数据修复，再对sealed200作正式输入预测。
5. 执行权限、近重复、预算与可选199题情景审计。
6. 执行功效及独立数值复核，汇总报告并冻结协议。

关键入口如下。其输出路径已有工件时会拒绝覆盖，完整从零复现应使用隔离资产副本和新的运行根目录。

```powershell
python -B -X utf8 fit_final_models.py --phase fit
python -B -X utf8 audit_data_and_budget.py --phase sealed-inputs
python -B -X utf8 encode_sealed_inputs.py
python -B -X utf8 repair_input_storage.py
python -B -X utf8 predict_final_inputs.py --label fresh400 --input "E:\语言模型组合路由\artifacts\private\routing_redesign_20261104\full_feedback_inputs\v1\fresh_dev_test_inputs.npz"
python -B -X utf8 predict_final_inputs.py --label sealed200 --input "E:\语言模型组合路由\research\e20_independent_confirmation_preparation\private\inputs\SEALED200_INPUTS_ONLY_UNICODE.npz"
python -B -X utf8 audit_data_and_budget.py --phase audit
python -B -X utf8 audit_data_and_budget.py --phase near
python -B -X utf8 audit_data_and_budget.py --phase budget
python -B -X utf8 audit_data_and_budget.py --phase optional-scenario
python -B -X utf8 power_analysis.py --help
```

`fit_final_models.py`的初始predict子命令仅作为已保留的代数等价实现。正式预测入口使用`predict_final_inputs.py`，其恢复原sklearn和逐头F-order运算。fresh主策略和raw-exact均没有由此变化。

功效脚本的`--source`仅指向E19 OOF效用，`--output-dir`指定汇总JSON和报告的输出目录。详细参数以`--help`为准。`assemble_e20.py`从已生成的脱敏审计JSON重建报告。封存收据生成后不直接覆盖原协议。

## 依赖与已知限制

原MiniLM编码源码和模型、E8私有Clarabel依赖目录、E17 full600冻结配置、E18输入投影与缓存身份审计为必要依赖。输入包含Y、reward或答案字段时，预测入口在解码成员前拒绝。

sealed200的原编码包具有object型空子任务元数据，已保留为历史。正式预测使用修复后的Unicode包，嵌入逐值不变。规范化元数据不进入预测特征。

当前审查结论为准备完成、实际评价待批准。fresh400历史曝光和sealed近重复处置属于独立确认资格限制。新题池未选定，2400题的调用数只提供事前容量封套。完整缓存期望效用的功效不覆盖另行随机抽单动作的评价设计。
