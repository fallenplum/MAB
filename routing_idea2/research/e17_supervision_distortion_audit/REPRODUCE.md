# E17复现入口

## 环境与私有资产

Python3.11、NumPy、SciPy、scikit-learn、Clarabel0.11.1。使用既有E8私有dependencies中的Clarabel。默认MAB_ASSET_ROOT为E:\语言模型组合路由，可通过同名环境变量切换。MAB_E17_ROOT可指定新的独立结果目录。

私有资产包括train600输入与20×3评分、E13各折输入包、原M0/M4及lex/reverse/uniform权重、NS各折8+8draw与512均值摘要、fresh400输入编码及8个缓存导入收据。原题、答案、私有逐题矩阵与权重均不在公开发布中。

## 顺序

在code目录执行以下命令。

```powershell
python -B -X utf8 train_e17.py prepare
python -B -X utf8 validate_frozen.py freeze-source
python -B -X utf8 train_e17.py freeze --workers 2
python -B -X utf8 validate_frozen.py qualify
python -B -X utf8 freeze_confirmation.py freeze-inputs-only
python -B -X utf8 fresh_numbering_inputs.py audit-inputs-only
python -B -X utf8 evaluate_e17.py
python -B -X utf8 synthetic_and_theory.py
python -B -X utf8 finalize_e17.py
```

原运行使用分阶段代码冻结。训练源码先冻结，验证器及各评价/输入审计源码在各自读取前单独冻结。发布版包含随后新增的只读验证器和汇总代码，完整新运行的协议source_hash集合会因此包含更多文件。来源数据、算法、排列和配置必须保持一致，逐项结果资格不依赖运行时长或压缩文件字节一致。

prepare和封存/评价写入均采用不可覆盖记录，重新复现应使用新的MAB_E17_ROOT。训练中断可在同一协议下继续freeze，完成后需通过独立资格。输入独立确认脚本一旦封存，重新运行采用新输出目录。既有公开报告不覆盖。

## 评价边界

evaluate_e17仅在全部预测及资格完成后读取旧train600外折评分。独立fresh准确率未实现自动执行入口。后续必须先核实缓存与预算、确认一次性评价协议。原sealed200不由本轮程序访问。

统计为题级点差、纠错破坏、排列分布及任务/seed分解。没有置信区间和bootstrap。预测头重复、生成重复、重编号重复均不扩大独立题量。
