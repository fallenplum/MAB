# V2 Step 2：独立Top-3已完成的验证结果

开发验证200道不同题，GSM8K/BBH各100，3个生成seed，共600条配对实例；原fresh400未评价。该基线在已看过原选择器验证结果后补充，性质为事后探索性，不能作为原预注册gate通过。

|范围|准确率|
|---|---:|
|两任务宏平均|86.3333%|
|gsm8k|92.0000%|
|bbh|80.6667%|
|BBH子任务宏平均|81.5844%|

|生成seed|准确率|Top1 Regret|
|---:|---:|---:|
|42|86.50%|0.095000|
|137|86.50%|0.100000|
|2024|86.00%|0.100000|

差值方向统一为IndependentTop3减对照：

|对照|准确率差pp|95%配对题簇CI pp|正向seed数|
|---|---:|---|---:|
|selector|0.6667|[-1.6667,3.0000]|2/3|
|global_fixed|0.3333|[-1.5000,2.1667]|2/3|
|task_fixed|0.1667|[-2.5000,2.8333]|2/3|
|subtask_fixed|-0.6667|[-3.0000,1.5000]|1/3|

全部区间跨0，尚无稳定增益证据。95%题簇CI按题保留三个重复，并对task/BBH子任务分层；不是把600实例当独立样本。总体和逐seed配对CI完整保存在STEP2_INDEPENDENT_TOP3_PAIRED_CI_v1.json。

|BBH子任务|不同题数|独立Top3准确率|
|---|---:|---:|
|boolean_expressions|3|100.00%|
|causal_judgement|4|33.33%|
|date_understanding|4|66.67%|
|disambiguation_qa|4|58.33%|
|dyck_languages|4|25.00%|
|formal_fallacies|4|83.33%|
|geometric_shapes|4|41.67%|
|hyperbaton|3|88.89%|
|logical_deduction_five_objects|4|100.00%|
|logical_deduction_seven_objects|4|83.33%|
|logical_deduction_three_objects|4|100.00%|
|movie_recommendation|4|75.00%|
|multistep_arithmetic_two|3|100.00%|
|navigate|4|91.67%|
|object_counting|4|100.00%|
|penguins_in_a_table|4|100.00%|
|reasoning_about_colored_objects|3|100.00%|
|ruin_names|4|66.67%|
|salient_translation_error_detection|3|77.78%|
|snarks|3|100.00%|
|sports_understanding|4|41.67%|
|temporal_sequences|4|100.00%|
|tracking_shuffled_objects_five_objects|3|100.00%|
|tracking_shuffled_objects_seven_objects|4|100.00%|
|tracking_shuffled_objects_three_objects|4|100.00%|
|web_of_lies|4|91.67%|
|word_sorting|3|77.78%|

STEP2_INDEPENDENT_TOP3_VALIDATION_v1.json 是原完整验证报告的机器路径脱敏导出；STEP2_INDEPENDENT_TOP3_ACCURACY_BY_TASK_v1.json 保留五方法的总体/任务/27子任务/逐seed结果，便于复核。冻结与谱系摘要另外单列。

优势：复用既有10800训练成员标签和已执行的验证组合矩阵，不重复推理。局限：事后开发对照、小子任务支持仅3–4题；涉及42的部分上游回答混合平台，137↔2024为同A6000。原报告没有的逐子任务配对CI未填造；未新增训练、评分或推理。
