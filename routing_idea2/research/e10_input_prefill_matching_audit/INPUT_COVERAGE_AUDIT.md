# E10 P0-B 输入覆盖与映射审计

## 实际结果

**train600 中 6 题发生真实右截断，占 1.00%。** GSM8K 为 0/300，BBH 为 6/300。BBH 的 causal_judgement 有 2 题，salient_translation_error_detection 有 4 题。其余 25 个 BBH 子任务本次均为 0。全部 27 个子任务的计数、比例和长度分布保存在 `INPUT_COVERAGE_AUDIT.json`。

冻结 MiniLM 的实际调用上限为 256 token，包含 `[CLS]` 和 `[SEP]` 共 2 个特殊 token，因此最多保留 254 个内容 token。编码前内容 token 的中位数为 55，P95 为 190.05，最大值为 568。逐题内容 token 保留比例均值为 0.998208，最小值为 0.447183。分位数采用 NumPy 默认线性插值。

私有逐题文件记录编码前长度、真实截断标记、保留比例、最后保留 token 和第一个遗漏 token 的原文 offset。600 个 offset 前缀重新使用冻结 tokenizer 编码，均精确恢复旧输入保留的内容 token ID。未截断题使用完整原文。R_match 因而增加 6 次前缀编码，其他 594 题可复用 R_full。

## 冻结来源

模型和 tokenizer 均来自 `sentence-transformers/all-MiniLM-L6-v2`，revision 为 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。本地模型、tokenizer、词表、配置和 pooling 文件共 9 项，均通过原始下载收据的字节数和 SHA-256 核验。冻结 tokenizer.json 的 SHA-256 为 `be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037`。本次全部使用已有本地资产。

输入字段为原始 `question`，模板为恒等原文。NFKC 和空白折叠用于 canonical 身份哈希。编码本身继承原始文本。实际 pooling 为 attention-mask mean 后 L2 归一化，模型运算和缓存均为 float32。tokenizer 配置的 model_max_length 为 512，原 encoder 显式传入 max_length=256，审计复现该实际调用。

原题从两个原始 JSONL 流式定位。仅在 ID 属于 train600 时调用既有 input-only 投影函数，跳过 reference 字段的 JSON 解码。共享源文件的物理字节需要完整流读以核验原 SHA，外部 split 的题目 payload 和全部 reference 值均未解码。

## 资产链与独立复算

已逐项核验 `features_v1` 到 V2 `selector_train_inputs.npz` 的 600 个原始 embedding 完全逐 bit 一致。共享 800 行缓存的题目元数据用于查找 train600 位置，向量 NPY 只解码选中的 600 行。旧 `phi` 和旧 PCA 均未读取。V2、E3、E8 和 E9 绑定的 train 输入 SHA-256 一致，为 `57e39747ec2756bcd79ab00885b3c08ad4f91fdb3ec900cbed15146420cd17c3`。

5 折 E8 history480、evaluation 输入和 metadata 中的 embedding、ID、canonical、task/subtask、global_indices 均与 train600 对应行精确一致。history NPZ 中的 Y 键未访问。E9 资格文件逐折绑定相同输入包 SHA。600 个题目 ID 和 canonical 均唯一。原向量非有限元素、零向量和精确重复向量均为 0，L2 范数最大误差为 1.1920929e-7。

另外使用原 `FrozenQuestionEncoder` 类、同一冻结权重、CPU、2 线程、batch_size=8 对 train600 全部原文重新编码。**600×384 个 float32 元素与旧缓存逐 bit 完全相等，最大绝对误差为 0。** 600 个复算向量各自最接近自己的缓存行。该复算使用 D 盘 Python 环境，运行时版本和耗时保存在 `CPU_EMBEDDING_REPLAY.json`，没有 GPU 运算和答案生成。

## 末尾内容规则检查

规则检查为可重放的语法检查。末尾问号子句由最后一个问号和前面的标点或换行定位。选项规则需要至少两个行首字母选项。约束规则使用固定英文关键词表。逐题命中区间保存在私有文件。

| 规则 | 完整保留 | 全部或部分丢失 | unknown |
|---|---:|---:|---:|
| 末尾问号子句 | 415 | 2 | 183 |
| 行首字母选项 | 185 | 4 | 411 |
| 显式约束关键词区间 | 246 | 2 | 352 |

表中规则只判断命中区间的可见性。约束关键词的保留不能确认完整约束语义。缺少规则命中时保留 unknown。人工语义复核没有执行，600 题的该字段均为 `unknown_not_performed`。

## 优势与劣势（风险）

优势包括真实冻结 tokenizer 计算、原文 offset 及 token ID 双重验证、完整输入资产链复核和全部 600 题独立 CPU embedding 复算。输入映射与截断计数的置信度为高，依据为逐项确定性核验和精确向量重放。

劣势和风险包括语法规则的语义覆盖有限、截断题仅有 6 题，以及本次仍使用反复开发过的 train600。截断对子任务或 E1–E9 结果的因果贡献仍待配对实验检验。本次审计本身不能估计该贡献。

## 私有接口与复现

供后续 prefill 固定使用的私有接口为 `private/PREFILL_INPUT_MANIFEST_NATIVE_FIRSTPASS.json`，SHA-256 为 `829406a0893657097fdd9ede1e614fb9db9446328c9363fe207e8ea42ca77be3`。该 JSON 数组包含 600 项，顺序与 selector_train_inputs.npz 相同。`private/PREFILL_INPUT_MANIFEST.json` 为 AutoTokenizer 完整审计生成的等价副本，两份内容已逐项核验相等。原题、个体 ID、前缀和逐题 token 信息均在 private 目录中保留。

按顺序执行以下源码即可从私有原资产重放。

1. `quick_native_coverage.py` 使用冻结 tokenizer.json 的原始 Rust 后端生成快速私有接口。
2. `audit_input_coverage.py` 使用 AutoTokenizer 验证同一冻结资产，并生成完整分布与逐题 offset 记录。
3. `audit_asset_chain.py` 读取 train600 输入键，核验 features_v1、V2、E3、E8 和 E9 的身份与向量映射。
4. `verify_embeddings_cpu.py` 使用原 encoder 类执行 600 题 CPU embedding 重放。
5. `finalize_input_audit.py` 汇总结果并生成冻结收据。

本次完整审计和 CPU 复算使用 `D:\Anaconda\envs\tslib\python.exe -B -X utf8`。资产链和快速后端审计也已在 E 盘 venv 执行。早期 E 盘 transformers 导入需要扫描大量模型源码，曾中止一次慢导入。第一次完整审计在 GSM8K 原始 subtask=None 与 V2 空字符串等价性检查处停止，随后按照 V2 的 `row.get("subtask") or ""` 规范修正并重跑。这些执行记录单列在 `INPUT_AUDIT_IMPLEMENTATION_NOTES.json`。

## 拓展思考与后续核验

R_full 与 A 的主比较涉及输入覆盖和表征两项变化。已验证的 6 个原文前缀支持 R_match 配对诊断。该诊断的效应方向与大小由后续冻结评分决定。
