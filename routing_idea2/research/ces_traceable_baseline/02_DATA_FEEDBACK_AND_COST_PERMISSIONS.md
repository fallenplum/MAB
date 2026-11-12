# CES公开数据、反馈与成本权限

## 研究问题

公开CES软件、题目和论文成绩能否建立可追溯的原协议复现，再迁移到六模型、K=3、真实聚合评分。当前先完成证据审计与零调用数值复核。

## 审计设置

官方master固定为`9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7`，树中926个blob。Zenodo标签`v1.0.0-kdd2026`对应`778ffd9e24ac7e2ee0a3790bc2a52711296dd59f`，925个blob。对完整Git树和完整归档逐文件核验，未使用抽样目录推断整个仓库。[官方固定树](https://github.com/JinkunXu/CES/tree/9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7)

Zenodo原记录20363903指向新记录20364184，两者列出同一归档与MD5。实际下载39,524,383字节，MD5为`ba087d464f99dbefcebeb618b6b0b413`，SHA256为`c3297e7e0e85594933239a1e89b486cee5290459b380eb950f20ad9540dd4583`。归档925个文件全部与Git标签的blob字节匹配。master相对标签仅README变化和新增根LICENSE。当前master根代码MIT，归档标签没有根LICENSE，各捆绑子树保留自身授权，完整ZIP保存在私有来源归档。[Zenodo原记录](https://zenodo.org/records/20363903)，[Zenodo当前记录](https://zenodo.org/records/20364184)

## 可获得的数据与原结果对照

| 资产 | 实际发现 | 可直接重算的范围 |
|---|---|---|
| CES生成日志、逐题选择、奖励、最终裁判、usage | 官方master与完整ZIP中没有CES的runs/outputs/cache结果 | 当前无法重算论文主表CES成绩 |
| Alpaca参考输出 | 805条，generator为gpt4_1106_preview | 输入与参考资产计数，未提供CES作答 |
| MT-Bench题 | 80题，每题2轮，8类各10题 | 题目规模与轮数 |
| MT-Bench gpt-4参考 | 30题，reasoning/math/coding各10 | 参考辅助资产，与CES输出角色不同 |
| FLASK-Hard | bundled文件89题 | README limit200实际最多处理89题 |
| FLASK完整评价/原始输入 | 分别1740/1700条 | 两份文件计数，不混为相同评测集 |
| 捆绑CSV榜单 | 扫描21份，未见CES行 | 其他项目的榜单不能当CES原实验结果 |
| MMLU能力profile | 15模型、10能力维、split test、每subject10、seed0 | 聚合字段算术复核，缺少原题回答与判分 |

`runs/`、`outputs/`是README所描述的运行后产物。本轮完整目录中未发现这些CES结果。原题和参考答案可用只证明输入可获得，原CES性能仍缺少逐题路由、作答、评分、模型/数据版本与费用日志。当前不将算法算子一致性或迁移成绩填入原论文复现栏。[README](https://github.com/JinkunXu/CES/blob/9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7/README.md)，[数据加载器](https://github.com/JinkunXu/CES/blob/9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7/src/data/dataset_loader.py)

## 实际完成的零API数值复核

对15个模型的`cap_correct/cap_total`和`overall_acc`重算。14个模型的比值与公布准确率一致。`qwen-2.5-32b-instruct`的公布能力与总体准确率均为0，但其`cap_correct`包含0.25、0.1916667等非整数，无法作为答对题数解释。机械相除与公布字段不一致。该比值只用于揭示内部不一致，不能作为更正后的真实准确率。原文件保持原样。

| 模型 | 分母题数 | 公布overall | 核验结论 |
|---|---:|---:|---|
| qwen-3 | 570 | 87.3684% | 汇总算术一致 |
| deepseek-r1 | 570 | 85.2632% | 汇总算术一致 |
| llama-3.1 | 570 | 85.0877% | 汇总算术一致 |
| deepseek-3 | 570 | 85.7895% | 汇总算术一致 |
| claude-4.5 | 570 | 79.1228% | 汇总算术一致 |
| gpt-4o | 570 | 85.0877% | 汇总算术一致 |
| Mistral-8x22B | 570 | 78.5965% | 汇总算术一致 |
| mistral-small-creative | 570 | 77.8947% | 汇总算术一致 |
| grok-4.1-fast | 570 | 92.1053% | 汇总算术一致 |
| qwen-2.5-72b | 570 | 83.6842% | 汇总算术一致 |
| gemma-2-27b-it | 570 | 75.6140% | 汇总算术一致 |
| qwen-2.5-32b-instruct | 570 | 0.0000% | 计数字段非整数且汇总不一致 |
| glm-4-32b | 570 | 81.0526% | 汇总算术一致 |
| llama-3.1-70b-instruct | 570 | 81.7544% | 汇总算术一致 |
| gpt-3.5-turbo | 570 | 65.9649% | 汇总算术一致 |

profile分母对应15×570=8550个预期模型—问题评价位置。公开文件没有8550条实际预测记录，无法核验原题ID、跨模型题目对齐或API失败情况。代码的计数是整数累加，上述分数形式不由该正常计数路径直接产生，原因尚不清楚。本轮重算状态为`REAGGREGATED_WITH_DISCREPANCIES`，审计工作已完成。[公开profile](https://github.com/JinkunXu/CES/blob/9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7/artifacts/mmlu_meta_10d.json)，[能力评测代码](https://github.com/JinkunXu/CES/blob/9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7/scripts/eval_mmlu_meta.py)

## 反馈和特征权限

原CES能力向量来自MMLU test上的额外成员级监督，包含10能力分数与成本维度。原在线MCQ使用当前服务组合的聚合正确性，每题反馈更新；开放问答runner另外调用参考模型及在线LLM裁判。原代码的逐题反馈权限与有限Gold揭示预算需要分别对齐。

`no_offline_warm_start`只将`--warm_start`设为random，成员特征仍从profile构造。它没有自动移除能力向量。`no_meta_vectors`是另一个消融。迁移至六个新冻结模型时，成员能力信息必须说明来源、监督和费用，不能把原15个云模型的profile无说明地重绑定。[消融入口](https://github.com/JinkunXu/CES/blob/9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7/scripts/run_ces_ablation_suite.py)，[MCQ构造](https://github.com/JinkunXu/CES/blob/9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7/scripts/run_objective_mcq.py)

原embedding默认是`paraphrase-multilingual-MiniLM-L12-v2`，公开归档没有原实验的逐题编码缓存。模型标识、数据加载调用和依赖没有完整锁定发布时的服务快照，重新调用当前云服务不能自动保证原始数值一致。[编码器](https://github.com/JinkunXu/CES/blob/9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7/src/embeddings/encoders.py)

## 调用预算

下表按README的K=2、冷缓存、成功逻辑调用计算，保持默认参考与在线裁判。额外重试、路由失败、提供商切换和离线能力评测单列。最终评判列为逻辑评价实例，其实际请求数还依赖batch和重复评判配置。

| README运行 | 实际题数 | 成员 | 聚合 | 在线参考 | 在线裁判 | 在线逻辑调用 | 最终评判实例另计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARC-Challenge README example | 200 | 400 | 200 | 0 | 0 | 600 | 0 |
| AlpacaEval2 README example | 805 | 1610 | 805 | 805 | 805 | 4025 | 805 |
| MT-Bench README example | 80 | 320 | 160 | 160 | 80 | 720 | 160 |
| FLASK-Hard bundled input at README limit200 | 89 | 178 | 89 | 89 | 89 | 445 | 89 |

若只选一个原任务的最小真实在线复现，ARC-Challenge 200题当前默认需要600次成功逻辑生成调用，另需现有profile或重新建立先验。原profile每subject10的同规模重建需要名义8550次成员调用；README给出的20题/subject会变成17100次，并改变公开先验。能力脚本的每次max_tokens16给出已发布规模的136800 completion-token上界，完整输入Token和实际费用缺少日志，保持未确定。

模型静态cost特征与API真实usage是不同账本。缺失usage时源码有回退0，无法由此推断原实际开销为0。本轮实际物理模型API调用为0，39.5MB仅为公开软件归档下载。[README](https://github.com/JinkunXu/CES/blob/9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7/README.md)，[provider实现](https://github.com/JinkunXu/CES/blob/9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7/src/provider/openrouter_client.py)

## 统计结论与失败原因

可以复核代码来源、完整数据目录、基准输入和部分公开聚合算术。原论文CES主性能在当前发布材料上仍未原数复现。主要缺项为逐题CES输出/评分/usage、运行到表格的配置映射及发布时模型与数据版本。profile还有一项内部不一致。

源码和缓存适配能够用零新增模型调用继续检验，但应明确标注来源核验、算子一致性、受控迁移这几项各自完成到哪一步。迁移结果能回答同一六模型协议下的比较，当前不足以直接归因原论文与本项目差距。

## 优势与风险

优势是完整软件归档已下载、925个blob逐字节匹配、模型先验15项真实重算、预算按89题FLASK和两轮MT修正。风险包括发布结果缺失、模型服务漂移、能力先验不完整及不同反馈权限。公开数据可获得性的判断置信度为高，范围为上述固定Git与Zenodo资产。

## 下一步

先以固定原作者核心代码建立不调用模型的数值与选择/更新一致性证据，再执行明确权限和费用的六模型K=3缓存迁移。若要声称原任务的原数性能复现，应取得对应逐题输出/评分或另行批准有预算的单任务复跑，并保存完整模型、题序、奖励和usage来源。
