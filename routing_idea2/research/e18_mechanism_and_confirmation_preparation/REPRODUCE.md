# E18真实入口与复核顺序

## 依赖与私有资产

Python3.11、NumPy1.26.4、SciPy1.11.4、scikit-learn1.7.2，原冻结MiniLM使用Torch2.4.1、Transformers4.51.3及原tokenizer/safetensors。远端核验依赖Paramiko，机制图使用Pillow。资产根为原E盘MAB目录，原E17来源和完整模型权重保持不变。

私有清单包括原1708身份注册、源官方输入文件、一次性保护签名/HMAC私钥、proposed_new2600_inputs_only.jsonl、PROPOSED2600_INPUTS_ONLY_v2.npz、FROZEN3000_INPUT_PREDICTIONS.npz、E17 full600权重与预处理、原8缓存、8份新增镜像和缓存资源账本。原题、标准答案、逐题矩阵和权重没有上传。

## 本轮实际执行顺序

1. 核对E17远端父提交和本地封存哈希。
2. audit_sources.py盘点来源；candidate_proposal.py执行一次已明确授权的保护输入指纹及候选构建；complete_registered_reference_audit.py补审仅去重参考；verify_source_assets.py核对候选身份。
3. qualify_theory.py完成数学资格，make_mechanism_diagram.py生成共享几何的Drawio/SVG/PNG。
4. power_and_statistics.py产生双侧规划；qualify_operator_identity.py独立核验typed算子和统计公式。
5. cache_audit.py与token_provenance_audit.py核对原8文件；remote_cache_audit_v2.py只读远端成员；remote_cache_postcheck.py复核身份和账本。
6. frozen_input_budget.py qualify-frozen加载原权重；propose-budget仅做新输入编码和调用计数。当前source含metadata修复，原实际执行source v1快照单独留私有。
7. repair_input_metadata.py追加Unicode v2并验证四数值字段及预测不变。
8. remote_aggregation_audit_v3.py、aggregation_scope_v3_report.py核对聚合范围；materialize_remote_members_v4.py及materialization_postcheck_v4.py同步8份已验证远端成员文件。
9. final_reconciliation.py扣除已验证缓存；prepare_proposed_execution_dag.py准备审核DAG；执行作用域审计和sparse_policy_projection.py只做合成软件资格。
10. assemble_e18.py生成六主报告与协议；verify_and_seal_e18.py复核和封存交付。

带argparse的脚本提供--help，其他模块的真实入口是文件末尾的main或纯函数。写入采用不可覆盖记录，当前资产布局的普通复核命令为python -B -X utf8 verify_and_seal_e18.py verify。完整重建需要隔离输出空间及原私有输入清单，不能在现有封存目录上直接重复运行写入阶段。一次性保护指纹已处理200题，普通复核只使用已保存签名，不能再次自动解码原保护题。SSH通过交互getpass，密码不入命令行、源码或报告。原地址超时与新地址核验记录分别保留。

## 实际评价入口状态

当前不存在已经通过E18准入的付费运行命令，旧600/200/400和全20组合入口不能直接使用。EXECUTION_ADAPTER_READINESS.md列所需作用域适配，稀疏结果投影与审核DAG有独立合成/拓扑资格。任何模型执行和gold评分都等待明确批准以及对应软件闸口通过。

题级配对推断纯函数为power_and_statistics.paired_question_test，输入为[3,原题数]两策略合法评分和任务标签。当前只在合成数据上验证。没有以旧600题补算真实置信区间。
