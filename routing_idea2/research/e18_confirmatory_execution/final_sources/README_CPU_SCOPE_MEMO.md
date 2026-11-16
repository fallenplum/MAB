# E18 CPU scope摘要复用包装层

`cpu_scope_memo.py`以独立运行入口调用原`e18_final_scoring.py`。原9b、de5、21e、7eb、8848文件保持冻结字节。包装层复用显式登记的只读scope摘要，原生记录、job、ModelSpec、物理费用、失败栅栏、审批读取及reference源扫描继续走原函数。

## 运行

在原E18 CPU运行目录执行，提供已核验的final manifest及原SHA。两个模式与原入口一致。

```powershell
python -B cpu_scope_memo.py --manifest '<FINAL_SCORING_MANIFEST.json>' --manifest-sha256 '<SHA256>' --native-read-root '<D盘已还原native根>' --runtime-root '<D盘CPU输出根>' --preflight-only --performance-receipt '<新PERFORMANCE_PREFLIGHT.json>'
python -B cpu_scope_memo.py --manifest '<FINAL_SCORING_MANIFEST.json>' --manifest-sha256 '<SHA256>' --native-read-root '<D盘已还原native根>' --runtime-root '<D盘CPU输出根>' --execute-scoring --performance-receipt '<新PERFORMANCE_SCORING.json>'
```

`--disable-scope-memo`关闭摘要复用，用于软件等价和计时检查。完整运行仍执行原门禁，实际评分需要原有明确授权。性能收据采用新文件名，包含命中次数、原digest调用数、完整性核验次数及CPU和墙钟时间。

## 身份与完整性

- execution、reference、scoring三个scope分别绑定原文件SHA及首次原canonical摘要。
- 内部scope快照使用递归`FrozenDict`和`FrozenList`。普通赋值、删除、增删改、排序、原地运算和重新初始化被拒绝，deepcopy保留只读快照。
- 仅精确对象identity可命中摘要缓存。相同内容的未注册对象、native记录、job和其他值继续执行原JSON序列化与SHA。
- 原loader动态构造的适配器、评分器及reference组件的scope拷贝单独登记。源码校验和每次审批role、operation、mode、bindings检查保留。
- 每次原finalizer发布完整输出之前以及退出时，绕过缓存重新核验scope完整摘要、绑定源文件SHA。完整性失败阻止该次发布并生成失败性能状态。私有评分journal与已写的单条中间收据继续按原规则保留，以支持故障追溯。

Python内建基类函数可绕过子类的普通只读接口。该包装层运行在同一受控进程中，完整性核验负责发现这类显式绕过。它不提供隔离执行或恶意代码沙箱。

## 资格边界

软件资格使用原final scorer的四题人工fixture、FakeBackend和人工reference，比较原路径与包装路径的门禁、最终策略奖励和语义结果。时间、UUID及运行目录不参与语义等价判断。真实GPU、原题、标准答案及正式评分均不用于软件资格。正式3000题的性能和结果仍需实际运行后核验。
