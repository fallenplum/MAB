# R1细分来源与闭合输出的只读审计

## 证据链

`audit_tail_export_lineage.py`将三类既有工件绑定到同一份私有收据。

1. fine静态来源ZIP包含原512任务、原上游租约、195个checkpoint成功记录、317个未执行键、fine plan/manifest、控制器、planner sidecar及pilot接纳元数据。
2. 原`a205`导出器的`EXPORT_MANIFEST.json`登记fine plan、fine manifests和键表。租约、checkpoint审计及控制器的字节由前述独立静态ZIP补充承载。
3. 已完成的`c506`独立原生审计提供archive SHA、manifest摘要、原生ModelSpec/job/owner/费用身份完整性。新增工具核对每个闭合archive与该结果一致，再将fine键逐个对应到其已登记生产来源。

原`e18_final_scoring.py`的`9b164c40`版本另行承载完整3000题执行门槛。来源收据保留该全局接纳的外部证明状态，质量评价由root的正式门槛与评分收据承载。

## 入口

```powershell
$e18Private = 'E:\语言模型组合路由\research\e18_confirmatory_execution\private'
$e18Code = 'D:\develop\CV\多臂老虎机\experiments\e18_confirmatory_execution'
$bundle = Join-Path $e18Private 'resource_scheduling\MEMBER4_TAIL_STATIC_RUNTIME_STATE_v1.zip'
$nativeAudit = Join-Path $e18Code 'CLOSED_NATIVE_INDEPENDENT_AUDIT_v13.json'
$scope = Join-Path $e18Private 'EXECUTION_SCOPE.json'
$gate = Join-Path $e18Code 'e18_final_scoring.py'

python (Join-Path $e18Code 'audit_tail_export_lineage.py') `
  --private-root $e18Private `
  --bundle-ref $bundle (Get-FileHash $bundle -Algorithm SHA256).Hash.ToLower() `
  --native-audit-ref $nativeAudit (Get-FileHash $nativeAudit -Algorithm SHA256).Hash.ToLower() `
  --scope-ref $scope (Get-FileHash $scope -Algorithm SHA256).Hash.ToLower() `
  --gate-source-ref $gate (Get-FileHash $gate -Algorithm SHA256).Hash.ToLower() `
  --output NEW_PRIVATE_RECEIPT.json
```

末批审计增加`--require-complete-fine`。全部317个fine结果与195个checkpoint成功记录齐备时，该标志才通过。每次输出使用新文件名。

## 覆盖与访问

收据列明实际已导出的fine数量、剩余数量、旧成功记录一致数、逐archive覆盖与pilot原生文件绑定。早期覆盖记录具有PARTIAL状态。历史source接纳沿不可变来源SHA完成，broker正常完成退出后的来源仍可核验。

JSON值解码范围为执行元数据。工具在值解码前拒绝题目文本、标准答案、奖励和原生答案等保护字段。gzip/TAR遍历会跳过原生文件内容，原生答案字符串的JSON解码次数为0。模型、评分、Gold和进程信号调用均为0。

合法缓存复用可能使`LAST_TASK_RECEIPT.json`的顶层`from_cache`从False变为True。发生这种差异时，工具仅对原始字节中的该布尔值作逆向替换，要求所得SHA精确等于旧快照。此路径跳过答案字符串。原native记录与费用身份仍按原始SHA核对。

## 软件资格

`TAIL_EXPORT_LINEAGE_QUALIFICATION_v2.json`记录19项纯软件资格。包括完整/部分覆盖、强制完整门槛、源登记缺失、backend/ModelSpec差异、重复物理尝试、checkpoint原生变化、固定门禁、保护字段和缓存标记证明。合成原生文件故意使用不可解析内容，检查元数据读取器确实跳过这些值。完整原生审计结果在软件例子中由fixture给定。

初始源码与15项资格保存于`software_archive/tail_lineage_fe94/`，最初真实来源收据保持原版本。新增工具与全部资格沿独立新文件保存，冻结科学worker、adapter和最终门禁保留原字节。

## 优势与风险

来源键、执行并集、旧成功快照、fine登记及独立原生完整性形成可核对的链条，实际覆盖量随收据保留。风险包括上游静态来源或独立原生审计不完整，以及合法缓存元数据变化。操作层实时活性由既有逐调用保护承担，完整全局接纳和科学质量仍分别需要正式收据。

真实来源收据包含逐任务摘要和物理尝试身份，按私有工件管理。公开追加部分使用源码、软件资格及本说明。
