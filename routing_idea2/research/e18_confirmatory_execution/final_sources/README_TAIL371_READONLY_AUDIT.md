# E18 371任务尾分片只读来源核验

本工具仅核验已冻结的控制来源与闭合归档绑定。实际准确率、评分和全局完成条件由原E18评价流程处理。

## 来源和边界

- 原始R1 plan SHA为`a627f3094c742ec0d352582a9abe8b6338af6eaf0967333ebf30f45727c97166`。
- 原始part15 manifest SHA为`5fc23696b12239804d4e22eb7fa4a7c15c0c3882e8b7ebc2ec2b0c56ee762995`，共371个原有科学任务。
- 安全检查点保留348份SUCCESS和23份UNEXECUTED，后者由6个子分片精确覆盖，大小为4、4、4、4、4、3。
- fine plan SHA为`af3fddaa5ec89e4724d3dbcc840c2b20abc239b8bb0192c4a60628fe075d66b3`。
- 只读取元数据和源码字节。原题、答案、奖励和评分器均不参与本工具。
- 私有收据含任务哈希及尝试来源，按原私有资产规则保存。

## 静态核验

从原代码目录运行，Python标准库即可。所有输出采用新文件名，已有收据无法覆盖。

```powershell
python -B audit_tail371_static_sources.py --bundle '<私有路径>/resource_scheduling/MEMBER4_TAIL371_STATIC_RUNTIME_STATE_v1.zip' --receipt '<私有路径>/cpu_qualification/TAIL371_STATIC_SOURCE_AUDIT_v1.json'
```

静态工具核验ZIP CRC、202个payload的全部SHA与大小、原16个R1子分片对上游8051键的精确覆盖、371=348+23分区、fine请求字段、原5ac/45c/81/3d源字节，以及v6继承v5的两parent注册关系。配置文件字节位于ZIP之外，静态工具核对的是其冻结引用。

静态通过只表示来源分区成立。实际FLOCK活性和结果完成需要运行日志及闭合输出进一步绑定。

## 闭合输出绑定

先由原`audit_closed_native_archives.py`完成全部物理输出的独立native审计。该审计必须绑定canonical catalogue与所有闭合tar文件。

```powershell
python -B audit_tail371_export_lineage.py --private-root '<私有路径>' --bundle-ref '<私有路径>/resource_scheduling/MEMBER4_TAIL371_STATIC_RUNTIME_STATE_v1.zip' 99dcd3dc4f7e2c701eb29e68c184405cf7bd3d55cf6734a3359eb402f13c446d --native-audit-ref '<累计native审计.json>' '<SHA256>' --scope-ref '<私有路径>/EXECUTION_SCOPE.json' cae646111612ceb26196d73ed4469165e77b163a332f4db81deeb6f66551f73e --gate-source-ref '<代码目录>/e18_final_scoring.py' 9b164c40ad6b3a1b21f84c7dc446438a11ce6421d56991d82d224ae2c4b0daa8 --require-complete-fine --output '<私有路径>/cpu_qualification/TAIL371_COMPLETE_EXPORT_LINEAGE_v1.json'
```

工具复用冻结98b元数据解析器及90b静态核验器。仅解码`EXPORT_MANIFEST.json`。已有348份输出必须与检查点的native SHA相同。23份fine输出必须带原parent及实际fine producer注册，全部371份输出具有同一ModelSpec指纹。

旧worker续跑会把receipt顶层`from_cache`从false改成true。归档和检查点若处于这次合法变化两侧，工具只允许该单一字面量的双向转换。反向候选还必须通过原98b解析器的前驱SHA复核。答案字符串保留为字节，嵌套字段及其他任何变化均拒绝。

原`e18_final_scoring.py`的9b全局门禁继续负责全范围native接纳。本收据保存操作层来源证据，完整实验完成状态仍需该门禁。

## 软件资格

```powershell
python -B qualify_tail371_readonly_audits.py --output TAIL371_READONLY_AUDIT_QUALIFICATION_v1.json
```

13项软件测试覆盖精确SHA、两个cache转换方向、嵌套字段与答案变化拒绝、重复顶层flag拒绝、敏感元数据字段拒绝，以及冻结来源绑定。测试使用人工字节字符串，真实模型、原题、答案和奖励读取均为0。

## 冻结源码

| 文件 | SHA256 |
|---|---|
| audit_tail371_static_sources.py | `90b5551718845b27632b341c24a131fa664fffbdac82758cb4801377ef63e60f` |
| audit_tail371_export_lineage.py | `54eb2094b75eb56e37e695d3bdacfa6f959ef23ef17a888e7dad4b13e40fcc3d` |
| qualify_tail371_readonly_audits.py | `2c627941f90f43cadfdd2124190f840cc5664a239d5ffec4fc57131ba1836f04` |

原98b、45c、5ac、21e、9b及v5/v6源文件均保留其既有字节。
