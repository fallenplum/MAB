# 本机模型协议说明（E0 前候选修正）

本说明记录工程选择，不把模型官方基准成绩作为本实验结果。正式冻结仍以 `E0_LOCK.json` 为准。

六个成员均使用精确仓库提交、NF4 4bit、单次固定 seed 42、temperature 0.7、top_p 0.95、最大输出2048 token、最大输入4096 token。提示统一放在一个 user 消息中；`enable_thinking=False` 显式进入模型配置和指纹，模板是否使用该开关由已锁定的原始 tokenizer 模板决定。Qwen3 使用其关闭 thinking 的模式。模型池、题量和20个组合不变。

修正原因：DeepSeek 官方建议 R1 系列使用采样、避免 system prompt；Qwen3 官方说明默认启用 thinking，并警告 thinking 模式使用 greedy 可能退化。因此，在没有任何正式 E1 输出之前弃用原候选的 greedy/256-token 成员设置。2048是本机工程上限，不等于这些模型的最优性能，也不保证所有题都不会截断。资格题上的格式与截断规则必须通过；正式实验逐模型、逐任务报告截断和无法解析计数，失败输出不能删除。

强单模型对照从100道独立E1选择题上确定，随后用同一模型、同样生成参数和上限进行3次新seed独立采样，再交给相同真实聚合器。禁止复制同一个回答充当三次采样。

聚合器保持 Qwen2.5-7B-Instruct 固定版本、NF4、temperature 0、最大输出256、最大输入6144。每条候选保留聚合 tokenizer 下最多512个原始source token（head128+tail384），省略标记额外计入实际输入费用；原完整成员生成成本照记。

官方依据：[DeepSeek-R1 模型说明](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)、[Qwen3 模型说明](https://huggingface.co/Qwen/Qwen3-8B)。这些链接用于解释设置，不代表本实验已通过资格或得到了性能结论。
