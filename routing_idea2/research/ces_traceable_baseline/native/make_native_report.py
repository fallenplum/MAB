"""Traceable CES source/protocol audit, with separate reported and reproduced results."""
from pathlib import Path
import hashlib
import json

ROOT=Path(r'E:\语言模型组合路由')
WORK=ROOT/'research/ces_traceable_baseline'
PUBLIC=WORK/'public'
OLD=ROOT/'third_party/ces_source_audit/upstream'
COMMIT='9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7'
BASE=f'https://github.com/JinkunXu/CES/blob/{COMMIT}/'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    PUBLIC.mkdir(parents=True,exist_ok=True)
    operator=json.loads((PUBLIC/'NATIVE_OPERATOR_QUALIFICATION.json').read_text())
    paper=json.loads((WORK/'private/native/PAPER_METADATA.json').read_text())
    snapshot=json.loads((ROOT/'third_party/ces_source_audit/SOURCE_MANIFEST.json').read_text())
    source_files=[]
    for record in snapshot['files']:
        path=OLD/record['path']
        raw=path.read_bytes()
        digest=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert sha(path)==record['sha256'] and digest==record['git_blob_sha']
        source_files.append(dict(path=record['path'],sha256=record['sha256'],git_blob=record['git_blob_sha'],bytes=len(raw),url=BASE+record['path']))
    official_results=dict(
        CES=dict(alpaca_LC_percent=63.12,alpaca_LC_printed_plusminus=.4,alpaca_WR_percent=61.47,alpaca_WR_printed_plusminus=.3,
                 MT_avg=8.94,MT_turn1=9.13,MT_turn2=8.75,ARC_accuracy_percent=97.5),
        KABB=dict(alpaca_LC_percent=59.38,alpaca_WR_percent=52.43,MT_avg=8.89,ARC_accuracy_percent=95.5),
        MoA=dict(alpaca_LC_percent=57.93,alpaca_WR_percent=73.46,MT_avg=9.26,ARC_accuracy_percent=96.0),
        DeepSeek_R1=dict(alpaca_LC_percent=76.17,alpaca_WR_percent=80.95,MT_avg=9.48,ARC_accuracy_percent=92.5),
        source='Local author paper pp6-7, printed5805-5806, Tables3-5, visually verified',
        role='Paper-reported numbers only, not current independently replayed native benchmark results',
        printed_uncertainty='The extracted experimental description does not determine replicate count or whether plusminus is SD/SE/CI; no95%CI claim')
    budgets=[]
    for k,label in [(2,'README examples'),(3,'paper Table5 default')]:
        for task,N,turns,ref,judge in [('ARC-Challenge',200,1,0,0),('AlpacaEval2',805,1,1,1),('MT-Bench',80,2,1,1),('FLASK-Hard bundled',89,1,1,1)]:
            members=N*turns*k
            aggregates=N*turns
            references=N*turns*ref
            judges=N*judge
            budgets.append(dict(setting=label,k=k,task=task,questions=N,turns=turns,
                member_calls=members,aggregate_calls=aggregates,reference_calls=references,online_judge_calls=judges,
                logical_calls_before_external_evaluation=members+aggregates+references+judges,
                costs_available='Budget only, no native generation executed',
                caveat='FLASK cannot launch unchanged because9 profile keys missing' if task.startswith('FLASK') else None))
    differences=[
        dict(item='selection',paper='expert-wise mean plus alpha*sqrt(phiVinvphi), independent TopK in Alg1 and complexity section',
             code='Greedy marginal of set-UCB using accumulated feature z',effect='Different policies; 140/200 artificial configurations selected different sets'),
        dict(item='feature scaling',paper='[normalized x, normalized v, normalized x hadamard normalized v]',
             code='[normalized x,3 normalized v,3 separately normalized(x hadamard v)]',effect='Changes action geometry and prior regularization'),
        dict(item='initial covariance',paper='Vinv=I and initial w=0',code='Runner ridge.1, so Vinv=10I and w=0',effect='Changes exploration scale'),
        dict(item='K',paper='Table5 default3',code='README and runner defaults2',effect='Paper-default replication needs explicitk3'),
        dict(item='Alpaca final reference',paper='Table3 caption GPT4o reference and GPT4Turbo judge',
             code='Published reference.json generator gpt4_1106_preview; online reference Qwen2.5-72B',
             effect='Offline final reference file does not establish the paper evaluation reference'),
        dict(item='MT aggregation model',paper='Qwen2.5-72B default unless specified',
             code='run_mt_bench aggregator qwen/qwen2.5-vl-72b-instruct',effect='Model identity differs'),
        dict(item='FLASK expert pool',paper='Table2 lists15',code='run_flask21 experts,9 missing profile entries',effect='Direct public runner hits missing metadata before generation'),
        dict(item='published profile',paper='offline MMLU capability estimation',code='15 profiles, one has fractional cap_correct with zero reported accuracies',
             effect='14 profiles satisfy count-based reconstruction, one schema inconsistent; never silently repaired'),
    ]
    public_paper={key:value for key,value in paper.items() if key!='path'}
    public_paper['source_role']='Local author-paper copy, full text and renders remain private'
    differences.append(dict(item='cost accounting',paper='Section6.4 states reference and judge included',
        code='Runner reference usage counted, online judge helper returns only scalar and its usage is not added to actual_cost_usd',
        effect='Public runner dollar sums do not establish the published all-in cost'))
    result=dict(status='AUDIT_COMPLETE_NATIVE_BENCHMARK_NOT_REPRODUCED',
        method='CES',paper=public_paper,reported_results=official_results,
        upstream=dict(repository='JinkunXu/CES',default_branch='master',head=COMMIT,head_date='2026-05-28T07:24:30Z',
            tree='1451ea798bd66b75b9b451ab56a322a5576a58e4',tree_entries=1280,blobs=926,
            current_head_equals_existing_pin=True,release_tag='v1.0.0-kdd2026',
            release_commit='778ffd9e24ac7e2ee0a3790bc2a52711296dd59f',
            release_root_license=False,current_master_root_license='MIT',
            tag_to_head='README update plus added root MIT license, mathematical core unchanged',
            old_zenodo='https://zenodo.org/records/20363903',latest_zenodo='https://zenodo.org/records/20364184',
            archive_md5='ba087d464f99dbefcebeb618b6b0b413',
            native_output_paths_in_tree=[],root_data_files=['artifacts/mmlu_meta_10d.json','data/alpaca_eval/reference.json']),
        source_scope='21 already audited exact-byte files, not entire third-party repository or all licenses',
        source_files=source_files,operator_qualification=operator,
        paper_code_data_differences=differences,native_call_budget=budgets,
        budget_scope='Cold cache logical client.chat invocations, excluding final external evaluation, retries, reference regeneration and offline profiling. Online no calls executed.',
        profiling=dict(published_n_per_subject=10,published_questions_per_model=570,models=15,
            published_model_question_observation_slots=8550,README_regenerate_n_per_subject=20,
            nominal_regenerate_questions_per_model=1140,nominal_regenerate_member_calls=17100,
            raw_question_ids_and_member_answers_published=False,
            aggregate_recheck='CES_CAPABILITY_AGGREGATE_RECHECK.json, reaggregation with discrepancies'),
        native_benchmark_performance_reproduced=False,independent_current_native_score=None,
        native_results_gap=['No CES per-question answers/cache/selected sets/rewards/judgments/usage logs in current tree or camera-ready archive',
            'Paper and public artifact protocol differences listed',
            'One capability profile is internally inconsistent',
            'Online model snapshots are not pinned to immutable provider revisions'],
        reproducible_without_LLM_calls=['Exact official core selection and Sherman-Morrison update qualification',
            'Published profile aggregate arithmetic with14 valid count-based rows and1 discrepancy',
            'Benchmark input/reference inventory and manifest hashes'],
        chosen_migration_source='Pinned official code. Preserve code behavior and label benchmark/pool/feature/feedback changes as adaptation.',
        physical_new_LLM_calls=0)
    (PUBLIC/'NATIVE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    text=f'''# CES原协议的可追溯复现范围

## 研究问题

先确定公开CES的算法、输入、监督、执行和评价权限，再区分原论文报告值、可重算公开材料与当前受控适配结果。本轮使用作者源码和已有公开数据进行零LLM调用核验。

## 论文与代码来源

本机保存的原论文共10页，DOI为10.1145/3770855.3817902，SHA256为`{paper['sha256']}`。表3、表4和表5同时经过文本提取与页面渲染核对。全文、页面提取和渲染仅保存于private。[论文正式出处](https://doi.org/10.1145/3770855.3817902)

当前官方master仍为`{COMMIT}`，提交时间2026-05-28，与先前审阅的版本完全相同。完整Git树有1280项、926个文件。本地21份精简快照逐个重新核对Git blob和SHA256，核心代码来源可追溯。[官方固定提交](https://github.com/JinkunXu/CES/tree/{COMMIT})

相机就绪tag为`778ffd9e24ac7e2ee0a3790bc2a52711296dd59f`。现master增加root MIT License并更新README，核心选择和更新实现相同。Zenodo原记录20363903指向更新记录20364184，两条记录的39.5MB ZIP具有相同MD5。归档目录、字节哈希及下载结果另见数据审计。[正式归档](https://zenodo.org/records/20364184)

## 原论文报告成绩

以下保留原表口径，全部是作者报告值。本轮没有独立复算这些主基准成绩。

| 方法 | Alpaca LC胜率 | Alpaca原胜率 | MT平均 | ARC准确率 |
|---|---|---|---|---|
| CES | 63.12±0.4% | 61.47±0.3% | 8.94 | 97.5% |
| KABB | 59.38±0.6% | 52.43±0.7% | 8.89 | 95.5% |
| MoA | 57.93±0.2% | 73.46±0.6% | 9.26 | 96.0% |
| DeepSeek-R1 | 76.17% | 80.95% | 9.48 | 92.5% |

CES的MT第一轮为9.13、第二轮8.75。原表并未据此证明CES在每个任务超过最强单模型。提取的实验说明没有明确±的统计定义及重复次数，此处保留排印值，不把它们转换为95%区间。FLASK以雷达图提供12能力表现，本轮没有从图片像素猜测精确值。[原论文表3—5](https://doi.org/10.1145/3770855.3817902)

## 公开材料能复算什么

当前树及正式ZIP未发现CES的运行输出、缓存、逐题反馈、所选组合、最终裁判或真实usage日志。根数据包含MMLU能力聚合与Alpaca参考输出；Bundled FastChat、MoA子树的其他模型数据与排行榜不能作为CES原始实验轨迹。[官方目录](https://github.com/JinkunXu/CES/tree/{COMMIT})

MMLU公开profile有15个模型、每模型570个题目计数槽位，记录的n_per_subject为10。14个模型的计数可重算已发布准确率。`qwen-2.5-32b-instruct`的cap_correct含小数且对应cap_acc和overall为0，违反原eval程序整数计数约定。完整差异见`CES_CAPABILITY_AGGREGATE_RECHECK.json`。这是聚合算术核验，缺少逐题输出与gold映射时不能声称重新评估过模型答案。[公开profile](https://github.com/JinkunXu/CES/blob/{COMMIT}/artifacts/mmlu_meta_10d.json)

**当前原协议主基准复现状态为未完成。** 已有材料支持源码算子核验、输入清单和部分聚合数据重算，尚不能无API重放原论文主表。

## 原版数据、特征与反馈权限

| 项目 | 固定作者实现 |
|---|---|
| 模型池 | Alpaca、MT、MCQ各15名专家，FLASK脚本21名，包含多个闭源或大参数API模型 |
| 问题特征 | multilingual MiniLM-L12，query前缀及L2归一化，加8维手工向量，实际前三维为长度、数字比例和代码标记 |
| 专家能力 | MMLU的10维能力准确率，加池内归一化元成本，形成11维meta |
| 投影 | 默认d_x32，query为随机正交投影，meta以能力矩阵标准化后的SVD方向初始化 |
| 特征与组合 | build_phi分别归一化x、v、x*v，后两块乘3，组合特征为所选成员之和 |
| 在线反馈 | MCQ每题执行选中集合后取得最终正确性，单次update；开放题每题生成reference并用LLM评分，再update |
| 暖启动 | meta-vector带有成员级离线监督，默认bandit w初始0；没有免费加载额外真实V0/b0状态 |
| 成本 | 路由reward减静态成员元成本，另累计实际API usage美元，二者不是同一计量 |

原代码直接用真实聚合最终评分更新集合特征，没有给每名成员分配虚构正确标签。开放任务在线judge为GPT-4o；MCQ通过字符解析与公开gold比较。论文6.4声明计入reference和judge，但在线judge费用未进入部分公开runner的actual_cost_usd累计，usage缺失还可能回退0。因此公开累计美元不足以复算原论文全成本，总调用预算需独立列reference、judge、最终评估、失败与预评估费用。[MCQ入口](https://github.com/JinkunXu/CES/blob/{COMMIT}/scripts/run_objective_mcq.py)，[Router](https://github.com/JinkunXu/CES/blob/{COMMIT}/src/orchestration/router.py)

## 纸面协议与公开实现的具体差异

论文Assumption4.1把组合期望奖励建模为成员线性贡献之和。第5节给出计算复杂度分析，该分析没有证明当前六模型真实聚合评分满足此加性假设。迁移后的服务质量需独立评价，单个CES适配结果不足以判定整个任务不存在可学习收益。

1. 论文Algorithm1及复杂度段落使用独立expert UCB的Top-K。公开核心采用逐步贪心的集合UCB边际增益。二者在本轮200个人工状态中有140次选队集合不同。迁移时以固定作者代码为主，不能静默改成纸面Top-K后称逐字执行。
2. 论文Eq5没有代码中的两块乘3，也没有代码中Hadamard块的单独归一化。论文初始V逆为I，原runner ridge0.1对应10I。两处均会改变探索几何。
3. 论文表5默认K=3，README和CLI默认K=2。两口径的调用预算分开列。
4. 论文Alpaca最终评估为GPT-4o reference与GPT-4-Turbo judge，公开参考文件generator为gpt4_1106_preview。在线reference和最终评价reference属于不同角色，缺少论文实际最终参考集时无法确认原数。
5. MT公开实现两轮分别选队、生成和reference，循环结束后只对第二轮做一次在线judge，用两轮特征之和更新一次。其聚合器模型为Qwen2.5-VL-72B，纸面默认描述为Qwen2.5-72B。
6. FLASK脚本21个专家中9个在公开profile缺失，原样构建会触发KeyError。Bundled FLASK-Hard只有89题，limit200不会产生200题。原pool表、脚本和数据需在复现时分别核对。
7. `Router.step_real`传入select_topk不支持的mode参数。官方MCQ runner直接调用受支持签名，可以作为最小可运行数学/控制入口，不能把该接口问题扩大成所有runner均不可用。

这些差异及对应blob哈希全部保留于`NATIVE_AUDIT.json`。原码和profile未作修复。[核心源码](https://github.com/JinkunXu/CES/blob/{COMMIT}/src/bandits/linucb_combinatorial.py)，[投影与特征](https://github.com/JinkunXu/CES/blob/{COMMIT}/src/embeddings/projectors.py)

## 原协议调用预算

下面是冷缓存下逻辑client.chat调用数，成功一次/项，尚未加最终外部评价、重试和离线能力估计。FLASK行假定先解决缺失profile。它不是当前实际支出。

| 任务 | 有效题数 | K | 成员 | 聚合 | reference | 在线judge | 小计 |
|---|---|---|---|---|---|---|---|
| ARC README示例 | 200 | 2 | 400 | 200 | 0 | 0 | 600 |
| ARC paper默认 | 200 | 3 | 600 | 200 | 0 | 0 | 800 |
| Alpaca README示例 | 805 | 2 | 1610 | 805 | 805 | 805 | 4025 |
| Alpaca paper默认 | 805 | 3 | 2415 | 805 | 805 | 805 | 4830 |
| MT README示例 | 80×2轮 | 2 | 320 | 160 | 160 | 80 | 720 |
| MT paper默认 | 80×2轮 | 3 | 480 | 160 | 160 | 80 | 880 |
| FLASK bundled | 89 | 2 | 178 | 89 | 89 | 89 | 445 |
| FLASK bundled | 89 | 3 | 267 | 89 | 89 | 89 | 534 |

若按README重新生成MMLU profile，20题/subject、15模型、每模型约1140题，名义上需17100次成员推理。已发布profile对应8550个模型—题目槽位，但原始回答与实际失败调用数未发布，不能由槽位数反推出精确API账单。

最终AlpacaEval比较、FastChat双轮judge、FLASK多能力review另行计费，其请求批处理和重试配置会影响HTTP请求数。公开client外层最多3次重试，SDK自身重试配置和服务端行为也需固定。原提示无统一输入Token硬截断，缺少原usage时无法给出可靠美元或全Token精确复现预算。[官方运行说明](https://github.com/JinkunXu/CES/blob/{COMMIT}/README.md)

## 零API执行结果

直接导入精确作者核心并执行200组独立人工矩阵配置，8601项核验通过。Sherman-Morrison均值/协方差与密集逆的最大误差为{operator['sequential_dense_inverse_max_error']:.2e}，贪心集合选择与逐步独立计算完全相同。另在原默认96维特征、15专家、K=3下执行8条各10次更新路径，均值重拟合最大误差为{operator['original_default_dimension_check']['max_dense_mean_error']:.2e}。该资格检查只导入NumPy数学模块，API客户端创建数为0，LLM调用数为0。

执行入口为`native/audit_native_ces.py`，结果为`NATIVE_OPERATOR_QUALIFICATION.json`。它证明迁移所用核心可追溯、实现内部一致，原论文主表的可复现性仍受上述数据与协议缺口限制。

## 优势与风险及下一步

优势是作者原始核心可以直接执行，原论文和公开快照均有版本与哈希证据，K=3与最终聚合评分在机制上接近当前项目。风险包括公开profile异常、代码/纸面差异、缺少原运行轨迹，以及provider模型修订未固定。

下一步可在本项目的已付费历史题上构造六模型能力描述，保留官方build_phi、集合UCB和共享update，逐项记录更换的编码、模型池、聚合器与稀疏反馈权限。所得结果应命名为受控迁移，原协议成绩仍单独保留为作者报告值。

源码与公开产物审计的置信度为高。作者表格数字的当期独立可复现性尚未建立，本轮不以自行补造的输出、裁判或真值填补。
'''
    (PUBLIC/'01_NATIVE_REPRODUCIBILITY_AUDIT.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(status=result['status'],source_files=len(source_files),physical_calls=0),ensure_ascii=False))


if __name__=='__main__':
    main()
