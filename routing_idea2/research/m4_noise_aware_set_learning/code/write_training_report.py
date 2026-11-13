"""Render training and decision diagnostics from immutable evaluation output."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / 'public/NS_TRAINING_RESULTS.json').read_text(encoding='utf-8'))


def percentage(value):
    return f'{100 * value:.3f}%'


def pp(value):
    return f'{100 * value:+.3f}'


def ci(interval):
    return f'[{pp(interval[0])}, {pp(interval[1])}]'


names = {
    'M0_Ridge': '原直接20输出Ridge',
    'M4_lex': '原M4，最小编号真实并列',
    'TSPO_empirical': '经验集合TSPO+，缓存均匀并列对照',
    'NS_main': 'NS-SPO+，δ=0.05',
    'NS_delta0': 'NS-SPO+，δ=0敏感性',
    'Davidson_main': 'Davidson，相同8样本三状态目标',
}
lines = ['# 集合与噪声感知损失的固定容量训练验证', '', '## 研究问题', '',
    '在原M4的并列标签偏好已经明确后，验证集合与噪声感知训练能否在相同题目表示和线性容量下改善外折选队。全部结果来自已多次使用的600题开发资产。', '',
    '## 实验设置与对照', '',
    '沿用五个480/120问题级外折。每个预测器使用冻结MiniLM384维输入、各折历史拟合PCA16与StandardScaler、一个带正则的常数项和17×20线性参数，共340个系数。每题20组合、三个生成重复始终属于同一问题簇。', '',
    '每折正则直接沿用原M4训练区内已选值，公式为100×multiplier/(3×480×20)。本轮不新增超参数搜索。这是固定训练配置的机制对照，不能解释为各种损失已充分调参后的算法排名。', '',
    '联合工作后验由历史480题内部三折交叉拟合构造。每题选择相同8个联合20维样本。NS主方法使用δ=0.05，δ=0为预声明敏感性。每个样本的近最优集合权重独立构造，精确优化全部样本max损失的平均值，正则只加一次。', '',
    'Davidson主目标也由完全相同的8个联合样本形成190个组合对的胜、平、负概率。每道题的190对取平均，再对问题平均。ν由训练目标总平局率p0设为p0/(1−p0)，边界为[1e−6,1e6]。这一设定在预测margin为0时匹配总平局率，属于事先固定的统计对照适配，ν未进行联合最大似然估计。512样本概率仅用于诊断MC误差。', '',
    '经验三重复均值只有0、1/3、2/3、1四档，因此本轮δ=0与δ=0.05给出相同经验最优集合。经验TSPO+直接复用上一轮相同容量、相同正则的均匀并列缓存。', '',
    '全部16个新模型及逐题分数在读取外折旧评分前保存，独立评价器逐一恢复模型并核对动作。物理LLM调用为0。', '',
    '## 最终服务准确率', '',
    '| 方法 | 总体 | GSM8K | BBH | seed42 / 137 / 2024 |',
    '|---|---:|---:|---:|---|']
for key, label in names.items():
    item = result['methods'][key]
    seeds = ' / '.join(percentage(value) for value in item['seed_accuracy'])
    lines.append(f"| {label} | {percentage(item['accuracy'])} | {percentage(item['task_accuracy']['gsm8k'])} | {percentage(item['task_accuracy']['bbh'])} | {seeds} |")
lines += ['', '## 两项预声明主要比较', '',
    '| 比较 | 收益，百分点 | 题簇配对95%CI，百分点 | 纠错/破坏，题×生成 | Holm p |',
    '|---|---:|---|---:|---:|']
for key, label in [('NS_main_minus_M4', 'NS主方法减原M4'), ('NS_main_minus_Ridge', 'NS主方法减原Ridge')]:
    item = result['primary_statistics'][key]
    lines.append(f"| {label} | {pp(item['gain'])} | {ci(item['ci95'])} | {item['corrected_events']}/{item['destroyed_events']} | {item['Holm_p']:.6f} |")
lines += ['',
    'Bootstrap按原始问题配对，在外折×任务内分层抽样10000次，种子20261114091。每道题的三个生成和全部方法一起重采样。两项主要比较共享Holm处理。外折训练历史重叠且开发资产已曝光，表内检验只提供本轮条件性开发证据。', '',
    '## 选队分布与经验排序', '',
    '| 方法 | 最高频组合 | 占比 | 熵有效组合数 | 预测近并列题数，小于1e−8 | 严格非平局组合对排序准确率 |',
    '|---|---:|---:|---:|---:|---:|']
for key, label in names.items():
    item = result['methods'][key]
    dist = item['distribution']
    lines.append(f"| {label} | {dist['top_action']} | {percentage(dist['top_fraction'])} | {dist['entropy_effective_actions']:.3f} | {dist['score_gap_below_1e8']} | {percentage(item['strict_pair_rank_accuracy'])} |")
lines += ['',
    '组合编号从0开始，沿用六选三字典序。预测得分近并列的动作对严格argmax的数值残差敏感。排序指标对每题真实均值差非零的组合对计算，再对有支持的问题平均，预测差绝对值小于1e−12记半分。三重复经验最优损失保存在JSON，它不是总体Regret。不同损失的原始得分没有共同概率校准，本轮没有把原始平方误差用作公平的概率预测比较。', '',
    '## 预声明MC资格对照', '']
mc = result['MC_qualification']
lines += [f"仅第0折使用另一组不重叠的8个后验样本重新拟合NS主方法。120道外折题中有{mc['action_changes']}题改变选队，原8样本准确率{percentage(mc['primary_accuracy'])}，另一组为{percentage(mc['alternative_accuracy'])}。两组分数最大绝对差为{mc['max_raw_score_difference']:.6g}。这项对照未用于选择保留哪组模型。", '',
    '不重叠后验样本仍来自同一MCMC工作后验，独立性未由不重叠本身保证。该诊断衡量有限积分的敏感性。MC8主训练不能被称作精确后验积分。', '',
    '## 数学与软件资格', '',
    'NS完整MC梯度与有限差分最大误差1.47e−9，Davidson为1.95e−10。独立数学算子与训练实现的损失完全一致，NS梯度误差1.11e−16、Davidson为6.94e−18。精确MC epigraph与小型独立SLSQP目标差约1.11e−16。', '',
    '这些检查验证实现忠实于给定目标。数学审查同时发现精确后验下的两动作Bayes不一致反例，凸性、非负性和去编号偏好不足以保证最终动作最优。完整条件与反例见数学报告。', '',
    '## CPU成本与求解状态', '']
cost = result['computational_cost']
lines += [f"16个新拟合的成功日志CPU合计{cost['successful_fit_cpu_seconds']:.3f}秒，单拟合wall合计{cost['successful_fit_wall_seconds']:.3f}秒。该合计不包含后验构建、资格检查、评价bootstrap和共同历史LLM成本，部分进程并行，因此不能作为端到端墙钟时间。", '',
    'NS模型使用精确有限MC目标的Clarabel epigraph QP。每个480题拟合有4180个变量、76800个不等式、1,382,400个稀疏非零项。内存日志中的Windows peak working set是进程自创建以来峰值，包含导入及前序拟合。Davidson用稳定logsumexp和解析梯度的L-BFGS-B。全部逐模型求解状态、目标误差、对偶间隙、CPU时间和内存在结果JSON中保留。', '',
    '## 优势、风险与统计结论', '',
    '本轮优势是保持相同问题输入与340个参数，主损失使用相同后验样本，固定全部训练配置，分离训练和外折评价，并保留数值退化、负结果和积分敏感性。', '',
    '风险包括联合Gaussian工作模型的统计失配、只有三次真实生成、8样本有限积分、ν的边际适配、固定正则可能不适合所有目标，以及重复开发评价。MCMC收敛与后验预测校准分别报告，数值fallback为0不代表统计校准已成立。', '']
passed = [key for key, item in result['primary_statistics'].items() if item['exploratory_positive_support']]
if passed:
    lines.append('主要比较中出现经过本轮Holm处理的正向开发信号，仍需结合后验校准、MC资格、任务与生成方向判断其适用范围，再冻结独立确认方案。')
else:
    lines.append('两项主要比较均未建立经过本轮Holm处理的正向收益。当前结果保留为集合损失与工作后验的机制诊断，未支持启动新的有限反馈实验。')
lines += ['', '## 下一步', '',
    '先联合数学一致性、噪声校准与本报告的服务质量判断瓶颈。fresh400和sealed200真实评分本轮读取次数为0。已有E8与M4机制审计保持原始数字，下一阶段不由本报告自动启动。', '']
(ROOT / 'public/03_SET_LOSS_TRAINING_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
with (ROOT / 'public/TRAINING_SUMMARY.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['method', 'accuracy', 'gsm8k', 'bbh', 'top_action', 'top_fraction', 'near_tie_questions'])
    writer.writeheader()
    for key, item in result['methods'].items():
        writer.writerow(dict(method=key, accuracy=item['accuracy'], gsm8k=item['task_accuracy']['gsm8k'],
            bbh=item['task_accuracy']['bbh'], top_action=item['distribution']['top_action'],
            top_fraction=item['distribution']['top_fraction'], near_tie_questions=item['distribution']['score_gap_below_1e8']))
print(json.dumps({'status': 'REPORT_WRITTEN', 'file': str(ROOT / 'public/03_SET_LOSS_TRAINING_RESULTS.md')}, ensure_ascii=False))
