# 成员Token计量来源补充

## 逐phase分布

8个phase输入Token分别随各题真实tokenizer输入长度变化，合计560000。每个phase并未分配固定70000 Token。

| 成员 | seed | 执行数 | 输入Token合计 | 最小 | 最大 | 不同长度数 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 42 | 400 | 63652 | 87 | 644 | 151 |
| 0 | 137 | 400 | 63652 | 87 | 644 | 151 |
| 1 | 42 | 400 | 70452 | 104 | 661 | 151 |
| 2 | 42 | 400 | 88950 | 137 | 763 | 160 |
| 2 | 137 | 400 | 88950 | 137 | 763 | 160 |
| 3 | 42 | 400 | 61746 | 85 | 620 | 142 |
| 4 | 137 | 400 | 60852 | 80 | 637 | 151 |
| 5 | 42 | 400 | 61746 | 85 | 620 | 142 |

## 测量来源

物理Transformers执行器在tokenization后取input_ids实际张量长度作为prompt_tokens，实际model.generate后取生成后缀张量的numel作为completion_tokens。它们是运行时序列长度计数。当前核验没有重新执行tokenizer或模型。

源码SHA与8份原执行计划、8份物理context逐项一致，8份import receipt及8份context均明确software_test_only=false。Qwen3和Granite在两个已有seed的400道同题上，输入长度均逐题完全相同。

## 身份口径

3200条cache key的canonical_id直接取原E17同一input-only NPZ中的canonical_hashes。再与安全原题输入文件的canonical字段及question文本SHA交叉核对，三项分别3200/3200匹配。该审核未使用标准答案或真实正确性。

## 优势与限制

优势是逐题长度分布、原物理生产context、源码绑定和导入账本相互核对。限制是历史调用没有在本阶段重新运行，验证依赖已保存且哈希一致的执行证据。这里的actual指模型输入与生成后缀序列计数，源wall合计仍无法替代真实GPU kernel时间。已核验来源的置信度为高。
