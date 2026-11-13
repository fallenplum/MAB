"""Archive the three frozen NS-SPO+ modules and build a bounded summary."""
from pathlib import Path
import hashlib
import json
import shutil

STAGE = Path(__file__).resolve().parent
ROOT = Path(r'E:\语言模型组合路由\research\m4_noise_aware_set_learning')
PUBLIC = ROOT / 'public'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def copy_once(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(source) != sha(destination):
            raise AssertionError('Archived bytes differ ' + str(destination))
    else:
        shutil.copy2(source, destination)


def main():
    for module in ('math_literature', 'noise', 'training'):
        source = STAGE / module
        runtime = ROOT / 'private/runtime' / module
        if runtime.exists():
            raise FileExistsError('Runtime archive must be new')
        shutil.copytree(source, runtime)
        for path in (source / 'public').rglob('*'):
            if path.is_file():
                copy_once(path, PUBLIC / ('code/' + path.name if path.suffix == '.py' else path.name))
        for path in source.glob('*.py'):
            copy_once(path, PUBLIC / 'code' / path.name)
    lock = read(ROOT / 'private/CORE_SOURCE_LOCK.json')
    for item in lock['files']:
        if sha(ROOT / item['public_path']) != item['sha256']:
            raise AssertionError('Frozen core source changed')
    training = read(PUBLIC / 'NS_TRAINING_RESULTS.json')
    noise = read(PUBLIC / 'JOINT_NOISE_COMPLETE_RESULTS.json')
    mathematics = read(PUBLIC / 'MATHEMATICAL_SOFTWARE_CHECKS.json')
    combined = {
        'status': 'COMPLETE_EXPLORATORY_MECHANISM_VALIDATION',
        'delta': .05, 'delta_authorization': 'explicit user reply',
        'theory_checks': mathematics, 'joint_noise_diagnostics': noise,
        'fixed_capacity_training': training,
        'source_hashes': {name: sha(PUBLIC / name) for name in (
            'NS_TRAINING_RESULTS.json', 'NS_PRIMARY_STATISTICS.json',
            'JOINT_NOISE_COMPLETE_RESULTS.json', 'MATHEMATICAL_SOFTWARE_CHECKS.json')},
        'decision': 'NO_RELIABLE_GAIN_ESTABLISHED; GENERAL_CONSISTENCY_COUNTEREXAMPLE; POSTERIOR_AND_MC_LIMITS_REMAIN',
        'new_physical_LLM_calls': 0, 'retained_reward_reads': 0,
    }
    put(PUBLIC / 'NS_SPO_COMPLETE_RESULTS.json', combined)
    replay = ROOT / 'private/runtime/training/private/evaluation_replay/REPRODUCTION_VERIFIED.json'
    copy_once(replay, PUBLIC / 'NS_SCORING_REPRODUCTION_VERIFIED.json')
    for filename in ('initialize_study.py', 'freeze_core.py', 'assemble_study.py'):
        copy_once(STAGE / filename, PUBLIC / 'orchestration' / filename)
    summary = '''# NS-SPO+与集合训练机制验证总结

## 研究问题与固定设置

本轮按用户确认的δ=0.05验证集合损失、联合奖励工作后验与Davidson三状态对照。固定六个语言模型、K=3、20组合、V2聚合器及原train600。沿用五个480/120问题外折，问题特征和17×20线性预测头共340系数保持一致。正则沿用每折原M4已选值，本轮新增参数搜索为0。

完成16个新CPU拟合，包括两个NS目标的五折模型、Davidson五折模型和首折另一组8后验样本资格对照。全部预测先封存再读取旧开发评分。物理LLM调用及保留题真实评分读取均为0。

## 数学核查

集合损失非负、在固定目标权重下对得分凸，并在近最优集合只有一个元素时恢复原SPO+。真实最大值唯一但次优差小于δ时，集合仍可包含多个动作。

对于δ近最优权重，指定奖励向量下的决策损失满足

\\[
\\operatorname{Reg}_r(\\arg\\max q)\\le L_{\\mathrm{TSPO+}}(q,r)+\\delta.
\\]

δ=0时保留原逐奖励向量的上界。δ>0时需要该偏差项。噪声目标必须保留逐联合样本max及同样本权重与奖励的乘积。

**这些性质没有给出普遍的Bayes决策一致性。** 联合后验以0.9概率给出(0.6,0.5)、以0.1概率给出(0,1)时，后验平均奖励为(0.54,0.55)，期望NS损失却可偏向第一个动作。该反例在δ=0和0.05均成立，并已嵌入20动作及小正则场景独立验证。

三次二值生成均值只有四档。δ=0和0.05的经验最优集合相同，经验TSPO+沿用上一轮均匀并列缓存。全奖励并列时该目标鼓励全部预测得分相等，测试argmax仍需要并列规则。

## 联合奖励模型

每个合法480题历史区内，先验与噪声超参数再做三折交叉拟合。协方差采用固定20%向对角收缩。联合Gaussian工作后验在[0,1]²⁰截断，使用Gibbs采样，保留512个诊断样本。NS与Davidson的主训练使用相同8个联合样本。

该工作近似保留同题组合相关性，二值奖励的精确联合似然尚未建立。成员生成seed为42、137、2024，聚合seed固定42。不同seed标签尚不能证明生成块条件独立。若跨重复协方差为Γ，当前噪声/L公式还需要Γ=0假设。

采样数值资格通过，数值回退触发0次。LOSO边际Brier为0.084545，两生成经验均值为0.082625，差值+0.001920，题簇95%CI为[−0.000965,+0.004670]。Gaussian矩的标称95%配对预测区间实际覆盖90.426%。这项结果检查工作预测矩，潜在胜负及等效概率的校准仍未建立。

经验集合TSPO+作为均匀集合回退对照保留。数值失败回退与统计解释资格分别记录。名义NS结果用于失配机制诊断，不能据数值回退为0将软权重宣称为可靠的潜在概率。

## 真实缓存服务结果

| 方法 | 准确率 |
|---|---:|
| 原直接20输出Ridge | 85.278% |
| 原M4 | 85.944% |
| 经验集合TSPO+ | 85.500% |
| NS-SPO+，δ=0.05 | 86.056% |
| NS-SPO+，δ=0 | 86.000% |
| Davidson固定ν对照 | 85.944% |

| 主比较 | 收益，百分点 | 题簇95%CI，百分点 | 纠错/破坏 | Holm p |
|---|---:|---|---:|---:|
| NS主方法减原M4 | +0.111 | [−1.278,+1.500] | 70/68 | 0.892 |
| NS主方法减原Ridge | +0.778 | [−0.667,+2.222] | 73/59 | 0.621 |

两项主比较均未建立正向质量增益。区间按600个独立原始问题配对重采样，同题三个生成与各方法结果共同保留。旧开发数据多次曝光，五折历史重叠，全部结果保持探索性身份。

## 去编号偏好与积分敏感性

NS主方法最高频组合占28.33%，原M4为77.83%，有效组合数由3.04增为10.84。NS没有低于10⁻⁸的最高分近并列，经验均匀集合TSPO+有397题近并列。

首折更换预先固定、互不重叠的另一组8个后验样本后，85/120题改变选队。原组和替代组准确率分别为88.333%与87.778%，研究保留原组作为主结果。该变化显示有限积分有明显敏感性，数值求解收敛不能消除它。

## 优势与局限

优势是保留真实缓存、同题相关性、严格历史交叉拟合、相同线性容量及同批后验样本，数学检查和实际代码相互核验，完整保留负结果。

局限包括Gaussian工作似然、Γ=0及同方差假设、仅三次真实生成、8样本MC、固定正则和Davidson固定ν。ν按历史平局率在零margin处匹配，并非联合最大似然估计。Davidson1970全文受访问限制，公式以NeurIPS2025原文核查，作者仓库只发布README与LICENSE。本轮对照是独立公式实现。

成功拟合CPU合计123秒，后验构建约100秒。日志合计与并行端到端墙钟时间分开，资格、统计与文献处理另行记录。

## 研究决定

**集合化降低了原M4的单一组合集中，但当前版本的稳定服务收益尚未建立。** 理论一般一致性、后验可检验性及有限积分稳定性均有明确限制。完整负结果继续保留，本轮没有启动新的反馈采集、保留题评价或语言模型生成。

后续若继续该候选，应先写清适用的奖励分布和决策一致性条件，明确能够检验的后验目标，再冻结积分精度与独立确认方案。当前结果没有证明NS-SPO+已构成新的有效组合路由算法。
'''
    (PUBLIC / 'EXPERIMENT_SUMMARY.md').write_text(summary, encoding='utf-8')
    print(json.dumps({'status': 'ASSEMBLED', 'root': str(ROOT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
