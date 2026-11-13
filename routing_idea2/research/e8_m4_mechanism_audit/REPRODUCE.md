# M4机制验证复现

## 运行环境与数据边界

原资产根目录为 `E:\语言模型组合路由`，本轮归档为其 `research\e8_m4_mechanism_audit`。使用原 `.venv\Scripts\python.exe` 的Python3.11环境、NumPy1.26.4、SciPy1.11.4、scikit-learn1.7.2及E8私有隔离目录中的Clarabel0.11.1。CPU运行，BLAS单线程，Python加 `-B`。

源码位于公开目录 `public/code` 与 `public/orchestration`。私有题目、真实奖励、模型权重、冻结逐题预测和保留题访问收据位于 `private`。完整私有原资产用于执行复现。

## 已完成的核验

`M4_AUDIT_VERIFICATION.json`记录原E8的1219个文件逐字节哈希一致，25个新机制对照恢复预测误差为0，两个独立评价器的95次纠错和83次破坏完全一致。全部25个精确QP均为Solved。原E8的70个冻结模型也通过其原验证程序。

新模块的来源锁、预测先冻结收据、评分访问记录保存在对应 `private/distribution`、`private/stable`、`private/content_power`。内容模块后加角色元数据与日志语义的版本说明保存在 `CONTENT_ANALYSIS_AMENDMENT.json`，数值结果保持一致。首版屏蔽收据的时间戳未留存。

## 验证与评价复演

在PowerShell执行以下命令。

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e8_m4_mechanism_audit\public\orchestration\reproduce_m4_audit.py' verify
```

独立复演三个模块的科学指标使用以下命令。

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e8_m4_mechanism_audit\public\orchestration\reproduce_m4_audit.py' analysis-replay
```

复演程序创建时间戳命名的私有新目录。稳定差模块仅排除本次冻结时间引起的协议哈希与执行耗时后逐项比较，其余模块完整比较JSON。评价复演使用已冻结权重。原E8和本轮原始结果继续保留。

## 25个受控拟合的完整重算

把 `public/code/run_distribution_audit.py` 与 `write_distribution_report.py`原样复制到一个全新的私有运行目录。依次用Python执行前者的 `freeze`、`train`、`evaluate`模式，再执行后者。每次运行须使用新的目录，旧收据与原始结果予以保留。

程序沿用原E8每折已选正则、480题训练区和冻结特征。完整头为340个参数，常数先验为20个参数，全部预测封存后才打开外折旧开发评分。不同并列规则是机制干预，完整重算无需语言模型推理。

## 输出解释

所有bootstrap均以原始题目为单位并在外折×任务内分层，同题三个生成与方法结果一起保留。各模块的预固定bootstrap种子分别为2026111401、2026111402、2026111403，区间端点的轻微差异保留在对应结果中。

全部结果属于已曝光开发数据上的条件诊断。均匀真实并列变体具有大量测试近并列，严格argmax重算可能受数值微差影响。独立确认草案仍为待执行状态。
