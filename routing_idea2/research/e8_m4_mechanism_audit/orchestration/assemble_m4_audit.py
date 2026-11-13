"""Assemble separate M4 mechanism modules without modifying E8."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

STAGE = Path(__file__).resolve().parent
ROOT = Path(r'E:\语言模型组合路由\research\e8_m4_mechanism_audit')
PUBLIC = ROOT / 'public'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def copy_once(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha(source) != sha(target):
        raise RuntimeError('Refusing to overwrite a different archived file ' + str(target))
    if not target.exists():
        shutil.copy2(source, target)


def main():
    for module in ('distribution', 'stable', 'content_power'):
        folder = STAGE / module
        for path in (folder / 'public').rglob('*'):
            if path.is_file():
                target = PUBLIC / ('code/' + path.name if path.suffix == '.py' else path.name)
                copy_once(path, target)
        for path in (folder / 'private').rglob('*'):
            if path.is_file():
                copy_once(path, ROOT / 'private' / module / path.relative_to(folder / 'private'))
        for path in folder.glob('*.py'):
            if path.name.startswith(('run_', 'write_')):
                copy_once(path, PUBLIC / 'code' / path.name)
    for filename in ('verify_m4_audit.py', 'assemble_m4_audit.py'):
        copy_once(STAGE / filename, PUBLIC / 'orchestration' / filename)
    distribution = read(PUBLIC / 'M4_DISTRIBUTION_AND_TIE_RESULTS.json')
    stable = read(PUBLIC / 'M4_STABLE_GAP_RESULTS.json')
    content = read(PUBLIC / 'M4_CONTENT_AND_POWER_RESULTS.json')
    protocol = {
        'status': 'EXPLORATORY_MECHANISM_AUDIT_COMPLETE',
        'research_branch': 'codex/e8-m4-mechanism-audit',
        'parent_commit': '45767047d556e6cd9cac0e8905cf6dd81d5e3d4f',
        'data': 'original train600, five frozen480/120 splits, three cached generation repeats',
        'scope': 'selection concentration, true-optimum tie rules, cross-generation nonzero gaps, content controls and future sample-size planning',
        'new_parameter_searches': 0, 'new_intervention_fits': 25,
        'new_fits_role': 'matched exact-SPO+ causal mechanism controls, never replacements selected by outer score',
        'new_physical_LLM_calls': 0,
        'preserved_outcome_accesses': [],
        'prior_E8_results': 'all parent files checked against pre-study byte hashes',
        'distribution_design': distribution['design'],
        'stable_design_sha256': stable['protocol_sha256'],
        'content_statistics': content['inference'],
        'main_result_claims': 'exploratory conditional development evidence, no confirmatory new-method claim',
        'independent_confirmation': 'proposed only, current lexicographic M4 kept as the candidate definition',
        'advantages': ['cached real aggregation outcomes', 'same features, capacity and selected regularization in full-head tie interventions', 'question-level paired uncertainty'],
        'limitations': ['reused development outcomes', 'overlapping fold training histories', 'three empirical repeats', 'near-tie instability of uniform-target controls', 'power based on historical plug-in variance'],
    }
    put(PUBLIC / 'M4_MECHANISM_PROTOCOL.json', protocol)
    result = {
        'status': 'COMPLETE_EXPLORATORY_MECHANISM_STUDY',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'distribution_and_tie_audit': distribution,
        'stable_nonzero_gap_audit': stable,
        'content_and_power_audit': content,
        'scientific_decision': 'NO_CONFIRMED_ROUTING_GAIN; STRONG_TRUE_TIE_LABEL_DIRECTION_EFFECT; NO_AUTOMATIC_FEEDBACK_OR_HOLDOUT_EXECUTION',
        'new_physical_LLM_calls': 0,
        'source_files': {name: sha(PUBLIC / name) for name in (
            'M4_DISTRIBUTION_AND_TIE_RESULTS.json', 'M4_STABLE_GAP_RESULTS.json',
            'M4_CONTENT_AND_POWER_RESULTS.json')},
    }
    put(PUBLIC / 'M4_MECHANISM_COMPLETE_RESULTS.json', result)
    write_summary(distribution, stable, content)
    print(json.dumps({'status': 'ASSEMBLED', 'root': str(ROOT)}, ensure_ascii=False))


def write_summary(distribution, stable, content):
    summary = '''# M4小规模机制验证总结

## 研究问题与设置

本轮检查E8的M4点值提升来自怎样的组合选择，并诊断真实非零差值、题目内容和保留题功效。复用原600题五折外层预测、20个三成员组合、三个冻结生成重复及V2聚合评分。新增25个精确SPO+轻量拟合，只改变预声明的真实最优并列规则或移除题目特征。全部超参数沿用历史配置，新增LLM调用为0。

## 一、M4是否偏向少数组合

**是。原M4有467/600题选择组合0，占77.83%。** 该组合包含Qwen3-8B、Qwen2.5-Coder-7B-Instruct与Granite-3.3-8b-instruct。GSM8K有283/300题选择该组合。原Ridge最高频组合只有55/600题。

同Clarabel求解器按原最小编号真实并列规则，完整恢复原M4的600个动作。只改为最大编号真实并列规则，594/600题改变动作，505/600题转向组合19。两种模型都有明确预测分差。**训练真实标签的并列选择规则对集中方向有直接影响。** 均匀真实并列变体出现大量测试得分近并列，其严格argmax结果随微小数值差异变化，保留为退化诊断。

## 二、稳定非零差值上是否更好

原M4相对Ridge有95次纠错、83次破坏，净增12条题目×生成记录。600题三生成平均差只有59题为正、50题为负、491题为0。

用两个非留出生成定义“存在同向非零组合对”，再评价第三个生成，支持164道原题、344条留出记录。M4有50次纠错、53次破坏，净−3条，收益−0.872个百分点，题簇95%CI为[−7.837,+6.232]。**该诊断没有支持M4在较稳定非零组合差问题上优于Ridge。** 宽区间也未建立其确定劣势。

这些分层用到同题反馈，尚不能仅凭新题输入识别。三生成经验稳定性不能替代真实条件期望。

## 三、题目内容与无内容先验

| 方法 | 开发准确率 |
|---|---:|
| 原Ridge | 85.278% |
| 原M4 | 85.944% |
| 任务/子任务内奖励块置乱M4 | 85.667% |
| 仅保留M4常数项 | 84.889% |
| 训练区全局均值固定组合 | 84.944% |

M4相对匹配置乱为+0.278个百分点，95%CI为[−1.056,+1.500]。可靠的逐题内容收益尚未建立。相对始终组合0，M4偏离的133题贡献45次纠错、26次破坏，说明仍有部分内容利用的可能性。相对Ridge，这133题却贡献19次纠错、31次破坏。两种参考共同保留。

M4相对训练区全局先验为+1.00个百分点，95%CI跨0。相近点值没有证明两种策略等效。

## 四、保留题能否确认亚百分点收益

根据600题开发OOF配对差的题间方差，单侧5%检验、独立问题采样的规划近似如下。

| 独立题数 | 若真实收益0.667pp的功效 | 若真实收益1pp的功效 | 80%可检测收益 |
|---:|---:|---:|---:|
| 200 | 11.7% | 16.8% | 3.63pp |
| 400 | 15.9% | 24.9% | 2.57pp |
| 600 | 19.6% | 32.3% | 2.10pp |

检出1pp收益且达到80%功效约需2640道独立问题，检出当前0.667pp点值约需5939题。该规划依赖开发方差、未来任务组成与最终冻结模型，不能保证实际功效。200道最终保留题可提供效果区间，对当前小点值的强确认能力不足。

## 优势与局限

本轮优势是复用真实缓存、在固定线性容量下控制训练标签方向、保留所有负结果，并按原始问题处理生成相关性。

主要局限是开发资产多次曝光、外折训练历史重叠、只有三次生成、均匀变体数值退化以及功效的外推不确定性。新增统计均为探索性诊断，E8原数字与Holm结论保持原样。不同模块预固定的bootstrap种子产生略有不同的区间端点，各自完整保留。

## 研究决定

**选队分布的字典序方向效应证据强，稳定的新题质量增益尚未建立。** 原M4及所有机制对照继续冻结。当前不推进反馈采集，不用逆字典序或均匀变体的开发点值挑选替代候选。

如开展独立确认，需先固定原lex M4和原Ridge的全600题训练规则、权重、动作、生成与统计协议，再一次性评价。fresh400和sealed200的实际历史曝光须在执行前完成来源审计。本轮未读取其真实奖励，也未启动新增调用。较小保留题适合报告不确定性，需要更大独立题量才能有力检出亚百分点到1pp收益。

## 文件入口

- `01_SELECTION_DISTRIBUTION_AND_TIE_AUDIT.md`给出20组合频次、并列数学、受控干预和无内容先验。
- `02_STABLE_NONZERO_GAPS.md`给出两生成筛选、第三生成评价及全部任务支持量。
- `03_CONTENT_AND_INDEPENDENT_CONFIRMATION_PLAN.md`给出置乱、输入屏蔽、功效和先行注册草案。
- `M4_MECHANISM_COMPLETE_RESULTS.json`汇总全部原始聚合指标。
- `M4_AUDIT_VERIFICATION.json`记录原E8保全、模型恢复和独立评价器一致性。
'''
    (PUBLIC / 'EXPERIMENT_SUMMARY.md').write_text(summary, encoding='utf-8')


if __name__ == '__main__':
    main()
