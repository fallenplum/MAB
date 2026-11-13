"""Summarize frozen fold-local noise diagnostics at the original-question unit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item.item()) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    out = args.out
    received, fields, task, source_hashes, posteriors = [], {}, np.empty(600, dtype='<U5'), {}, []
    multiplicity = np.zeros(600, dtype=int)
    for fold in range(5):
        folder = out / f'private/fold{fold}'
        receipt_path = folder / 'POSTERIOR_FREEZE_RECEIPT.json'
        receipt = read(receipt_path)
        if receipt['fold'] != fold or receipt['outer_evaluation_rewards_opened']:
            raise AssertionError('Fold scope changed')
        posterior_path = folder / 'POSTERIOR_TRAINING.npz'
        if sha(posterior_path) != receipt['posterior_sha256']:
            raise AssertionError('Frozen posterior target hash changed')
        posterior = arrays(posterior_path)
        records_path = folder / 'LOSO_PREDICTIVE_RECORDS.npz'
        records = arrays(records_path)
        indices = records['global_indices']
        if not np.array_equal(indices, posterior['history_indices']):
            raise AssertionError('Noise target and diagnostic identities differ')
        if len(set(indices.tolist())) != 480:
            raise AssertionError('Unexpected historical question count')
        task[indices] = records['task']
        multiplicity[indices] += 1
        for key, value in records.items():
            if key not in ('global_indices', 'task', 'subtask'):
                if key not in fields:
                    fields[key] = np.zeros((600, 3), dtype=float)
                fields[key][indices] += value
        received.append(receipt)
        posteriors.append(posterior)
        source_hashes[str(receipt_path)] = sha(receipt_path)
        source_hashes[str(records_path)] = sha(records_path)
    if not np.all(multiplicity == 4):
        raise AssertionError('Each of600 original questions must occur in4 histories')
    fields = {key: value / multiplicity[:, None] for key, value in fields.items()}
    rng = np.random.default_rng(20261114091)
    samples = np.concatenate([rng.choice(np.flatnonzero(task == label), size=(10000, 300), replace=True)
                              for label in ('gsm8k', 'bbh')], axis=1)
    mean_brier = fields['marginal_brier'].mean(axis=1)
    comparisons = {}
    for name in ('empirical_two_seed_brier', 'historical_prior_brier', 'weak_Jeffreys_brier'):
        difference = mean_brier - fields[name].mean(axis=1)
        boot = difference[samples].mean(axis=1)
        comparisons[name] = dict(posterior_brier=float(mean_brier.mean()),
            reference_brier=float(fields[name].mean()), posterior_minus_reference=float(difference.mean()),
            paired95_question_cluster=np.quantile(boot, [.025, .975]),
            question_clusters=600, bootstrap_repetitions=10000,
            task={label: dict(posterior_brier=float(mean_brier[task == label].mean()),
                             reference_brier=float(fields[name][task == label].mean()),
                             difference=float(difference[task == label].mean())) for label in ('gsm8k', 'bbh')})
    summaries = {}
    for suffix in ('delta005', 'delta000'):
        summaries[suffix] = {
            key: float(np.concatenate([p[key + '_' + suffix] for p in posteriors]).mean())
            for key in ('effective_combinations', 'entropy', 'mean_set_size')}
        summaries[suffix]['largest_average_weight'] = float(np.concatenate(
            [p['weights_' + suffix].max(axis=1) for p in posteriors]).mean())
    all_rhat = np.concatenate([p['rhat_max'] for p in posteriors])
    all_ess = np.concatenate([p['ess_min'] for p in posteriors])
    fallback = int(sum(p['fallback_mask'].sum() for p in posteriors))
    result = dict(status='COMPLETE_EXPLORATORY_JOINT_WORKING_POSTERIOR_DIAGNOSTIC',
        generated_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        original_questions=600, repeated_training_targets=2400, target_history_multiplicity=4,
        diagnostic_question_heldseed_context_records=7200,
        latent_samples_per_context=512, training_MC_samples=8,
        hyperparameters_fit_to_other320=True, reused_histories_overlap=True,
        posterior_sampling=dict(fallback_contexts=fallback, Rhat_limit=1.1, ESS_limit=32,
            maximum_observed_Rhat=float(all_rhat.max()), minimum_observed_ESS=float(all_ess.min())),
        soft_target_summaries=summaries, brier_comparisons=comparisons,
        predictive_moment_summary={key: float(value.mean()) for key, value in fields.items()
            if key not in ('empirical_two_seed_brier', 'historical_prior_brier', 'weak_Jeffreys_brier')},
        posterior_qualification='COMPUTATIONAL_SAMPLING_PASSED; STATISTICAL_CALIBRATION_NOT_ESTABLISHED',
        latent_equivalence_calibration='unidentifiable from the single held-generation realization',
        sources=source_hashes, physical_LLM_calls=0,
        preserved_reward_splits_read=[], source_e8_modified=False,
        limitations=['Binary observations are approximated by a constant Gaussian noise moment model',
            'Latent Gaussian is jointly truncated to[0,1], raw posterior draws are not clipped',
            'Individual generation seeds are member-generation blocks with shared members and fixed aggregation seed42',
            'Rhat and approximate ESS measure numerical mixing, not true probability calibration',
            'Pair moment intervals can under-cover and binary Frechet moments can be infeasible',
            'Bootstrap resamples600 original questions while conditioning on overlapping developmental training histories'])
    put(out / 'public/JOINT_NOISE_COMPLETE_RESULTS.json', result)
    np.savez_compressed(out / 'private/QUESTION_CLUSTER_NOISE_DIAGNOSTICS.npz',
        task=task, history_multiplicity=multiplicity, **fields)
    write_report(out, result)
    print(json.dumps(dict(status=result['status'], brier_comparisons=comparisons,
        posterior_sampling=result['posterior_sampling'], predictive=result['predictive_moment_summary']),
        default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item.item()), flush=True)


def write_report(out, result):
    s = result['posterior_sampling']
    lines = ['# 联合生成噪声与有界工作后验诊断', '', '## 研究问题', '',
        '利用历史题的三次成员生成，估计20个组合的相关生成误差，为集合损失提供带不确定性的联合软标签，并检查该统计近似的预测表现。', '',
        '## 实验设置', '',
        '沿用E8五个外折，每折480道历史题。历史区再做固定三折，每个目标问题的先验与噪声超参数仅由其余320题估计。外折120题的真实评分没有进入本模块。', '',
        '同题三重复的20维中心化残差协方差采用固定20%向对角收缩，并加入10⁻⁶数值ridge。组合先验均值来自合法历史区，题间先验协方差由题均值协方差减去噪声协方差的三分之一、作PSD投影后再收缩得到。以上均为预声明估计假设。', '',
        '以三重复均值的Gaussian矩近似构造共轭工作参数，再将20维联合分布截断到[0,1]。使用两个过分散初值的Gibbs链，每链预热128轮、每两轮取一次、保留256个样本。软权重及配对三状态概率使用512个样本，训练MC损失预先取其中8个分散样本。另保留不重叠的8样本作数值稳定性对照。', '',
        '该模型是经验Bayes Gaussian工作近似，二值奖励的精确联合似然尚未建立。联合截断是采样目标，程序没有把无界样本直接裁剪后充作目标后验。', '',
        '## 生成块的来源', '',
        '缓存接口将同一道题、同一成员别名与成员生成seed绑定到唯一来源记录。每个题目×生成块中的六份成员输出被20个组合合法复用。聚合器seed固定42，变化的是成员生成seed42、137、2024。来源收据与缓存键支持这样的联合块定义，seed标签不能证明独立同分布。', '',
        '## 软件与采样资格', '',
        '单变量截断正态的解析矩、参数排列等变、概率单纯形、两个MC批次分离和显式回退通过软件检查。不同位置卡死的两链产生无穷Rhat和0 ESS，相同位置卡死也因0 ESS被拒绝。', '',
        f'2400个训练上下文来自600道原始问题，每题出现在四个外折历史中。最大Rhat为{s["maximum_observed_Rhat"]:.4f}，最小近似ESS为{s["minimum_observed_ESS"]:.1f}。触发数值/混合资格回退的上下文为{s["fallback_contexts"]}。回退规则是在失败上下文使用经验集合TSPO+，该身份保存在私有mask中。', '',
        '## 留一生成预测结果', '',
        '每次用两个生成的观测构造工作后验，预测第三个生成的边际正确性。三次轮换均使用未含该目标问题的历史超参数。一个原始问题的三轮和四种历史上下文一起保留，配对区间按600个原始问题、任务分层抽样10000次。', '',
        '| 参照 | 工作后验Brier | 参照Brier | 差值 | 题簇配对95%CI |',
        '|---|---:|---:|---:|---|']
    names = {'empirical_two_seed_brier': '两个生成经验均值', 'historical_prior_brier': '历史组合均值先验',
             'weak_Jeffreys_brier': '弱Jeffreys边际平滑'}
    for key, row in result['brier_comparisons'].items():
        ci = row['paired95_question_cluster']
        lines.append(f'| {names[key]} | {row["posterior_brier"]:.6f} | {row["reference_brier"]:.6f} | {row["posterior_minus_reference"]:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |')
    p = result['predictive_moment_summary']
    lines += ['', 'Brier越小越好。以上比较检查新生成的边际预测，不直接检验潜在组合期望差事件的后验概率。', '',
        f'基于Gaussian矩的标称95%配对差区间实际覆盖率为{100*p["gaussian_moment_interval_coverage"]:.2f}%。配对二值Fréchet约束违例比例为{p["binary_Frechet_violation_fraction"]:.6g}，边际方差上界违例比例为{p["binary_marginal_variance_excess_fraction"]:.6g}。这些区间保留为工作矩诊断，实际覆盖率和二值可行性需单独核查。', '',
        '## 软权重', '',
        '| δ | 平均有效组合数 exp(entropy) | 平均最大权重 | 每样本平均最优集合大小 |', '|---|---:|---:|---:|']
    for key, label in (('delta005', '.05'), ('delta000', '0')):
        row = result['soft_target_summaries'][key]
        lines.append(f'| {label} | {row["effective_combinations"]:.3f} | {row["largest_average_weight"]:.4f} | {row["mean_set_size"]:.3f} |')
    lines += ['', '权重分散程度来自所选工作模型和观测。较大的权重或较小的方差尚不构成真实组合优势的证明。', '',
        '## 统计结论、优势与风险', '',
        '数值采样资格通过，潜在胜负与实际等效事件的概率校准仍未建立。只有一个留出生成时，其一次正确性不能给出该题真实条件期望差。Gaussian近似的二值可行性、跨题同方差、三个生成的交换性和先验收缩均需要保留为假设。', '',
        '优势是完整保留同题组合相关性、严格历史交叉拟合，以及采样和失败回退可追溯。风险是工作后验失配、只有三个成员生成重复、复用开发题和训练历史重叠。', '',
        '## 下一步', '',
        '将冻结软标签交给相同线性容量的NS-SPO+和Davidson作探索性对照，同时保留经验TSPO+。最终服务准确率与MC稳定性必须独立评价。后验混合通过和预测器优化收敛不能替代科学假设成立。', '',
        '## 产物', '',
        'JOINT_NOISE_COMPLETE_RESULTS.json汇总数值，JOINT_NOISE_PROTOCOL.json保存源锁，JOINT_REPLICATE_PROVENANCE_AUDIT.json记录缓存来源。每折POSTERIOR_TRAINING.npz、联合先验参数、LOSO逐题记录及访问日志保存在private。新增LLM调用为0，保留题奖励读取为0。', '']
    (out / 'public/JOINT_NOISE_DIAGNOSTIC.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
