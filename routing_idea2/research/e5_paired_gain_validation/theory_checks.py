"""Numerical implementation checks for manuscript Appendix B and Proposition 13."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

from itertools import combinations
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.linalg import solve

TEAMS = tuple(combinations(range(6), 3))
M = len(TEAMS)
C = np.eye(M) - np.ones((M, M)) / M
MANUSCRIPT_SHA256 = 'b2c9c7f9bf09a3183f4907eb7daedd3a7537dc9f6460f62ccfc246e525bb9e99'
SEED = 20261110


def action_design():
    membership = np.asarray([[float(member in team) for member in range(6)] for team in TEAMS])
    interactions = [membership[:, i] * membership[:, j] for i, j in combinations(range(6), 2)]
    return np.column_stack((np.ones(M), membership, *interactions))


def strict_construction(wrong_action, gamma, epsilon):
    if not (0 < gamma < .5 and 0 < epsilon < min(6 * gamma, 1 - 2 * gamma)):
        raise ValueError('Appendix B requires strict valid gamma/epsilon bounds')
    chosen = TEAMS[wrong_action]
    complement = tuple(member for member in range(6) if member not in chosen)
    pairs = tuple(zip(chosen, complement))
    overlap = np.asarray([len(set(team) & set(chosen)) for team in TEAMS])
    transversal = np.asarray([all(len(set(team) & set(pair)) == 1 for pair in pairs) for team in TEAMS])
    contrast = np.where(transversal, (-1.) ** overlap, 0.)
    linear = overlap / 3. - .5
    reward = .5 + gamma * contrast + epsilon * linear
    projected = .5 + epsilon * linear
    return reward, projected, contrast, linear, transversal, overlap


def maximizing_set(scores, atol=1e-10):
    scores = np.asarray(scores)
    return tuple(np.flatnonzero(scores >= scores.max() - atol).tolist())


def wilson(successes, total, z=1.959963984540054):
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [float(center-half), float(center+half)]


def population_checks():
    A = action_design()
    parameter_pairs = [(.2,.1), (.1,.01), (.45,.05), (.05,.25), (.001,.001), (.49,.01), (.01,.05994)]
    expected_rank = (15, 14, 19)
    actual_rank = tuple(int(np.linalg.matrix_rank(value)) for value in (A, C @ A, C))
    assert actual_rank == expected_rank
    projection = A @ np.linalg.pinv(A)
    output = []
    ridge_output = []
    for wrong in range(M):
        for gamma, epsilon in parameter_pairs:
            r, q, w, h, transversal, overlap = strict_construction(wrong,gamma,epsilon)
            empirical_projection = projection @ r
            error = float(np.max(abs(empirical_projection-q)))
            theoretical_regret = 2*gamma-epsilon/3
            actual_regret = float(r.max()-r[wrong])
            true_maxima = np.flatnonzero(transversal & (overlap == 2))
            assert np.sum(transversal) == 8
            assert np.max(abs(A.T@w)) < 1e-12
            assert np.sum(w) == 0 and np.sum(abs(w)) == np.sum(w*w) == 8
            assert (r > 0).all() and (r < 1).all()
            assert error < 1e-12
            assert maximizing_set(q) == maximizing_set(empirical_projection) == (wrong,)
            assert maximizing_set(r) == tuple(true_maxima)
            assert len(true_maxima) == 3
            assert abs(actual_regret-theoretical_regret) < 1e-12 and actual_regret > 0
            assert abs(np.mean((r-q)**2)-2*gamma*gamma/5) < 1e-12
            assert abs(np.max(abs(r-q))-gamma) < 1e-12
            good_action = int(true_maxima[0])
            good_linear_score = np.asarray([len(set(team)&set(TEAMS[good_action]))/3 for team in TEAMS])
            assert maximizing_set(good_linear_score) == (good_action,)
            assert abs(r.max()-r[good_action]) < 1e-12
            noise_constant = float(np.mean(r*(1-r)))
            output.append(dict(wrong_action=wrong,wrong_members=list(TEAMS[wrong]),gamma=gamma,epsilon=epsilon,
                population_projection_max_error=error,predicted_gap=float(q[wrong]-np.max(np.delete(q,wrong))),
                realized_regret=actual_regret,theoretical_regret=theoretical_regret,
                minimum_mean_value_mse=float(np.mean((r-q)**2)),
                binary_population_loss=float(np.mean((r-q)**2)+noise_constant),
                binary_noise_constant=noise_constant,true_maximizers=true_maxima.tolist(),
                all_reward_means=r.tolist(),all_population_optimal_scores=q.tolist()))
            # A distinct regularized objective. It is not the theorem's population LS objective.
            for penalty in (.0001,.01,1.):
                coef = solve(A.T@A/M+penalty*np.eye(22), A.T@r/M, assume_a='pos')
                regularized = A@coef
                selected = min(maximizing_set(regularized))
                ridge_output.append(dict(wrong_action=wrong,gamma=gamma,epsilon=epsilon,penalty=penalty,
                    objective='mean_action squared error plus penalty times coefficient norm squared',
                    chosen_action=selected,regret=float(r.max()-r[selected]),
                    score_projection_difference=float(np.max(abs(regularized-q))),
                    mean_value_mse=float(np.mean((regularized-r)**2))))
    invalid = [(0,.1),(.5,.1),(.2,0),(.01,6*.01),(.49,1-2*.49)]
    invalid_rejected = 0
    for gamma,epsilon in invalid:
        try:
            strict_construction(0,gamma,epsilon)
        except ValueError:
            invalid_rejected += 1
    assert invalid_rejected == len(invalid)
    return dict(ranks=dict(A=actual_rank[0],CA=actual_rank[1],C=actual_rank[2]),
                singular_values_A=np.linalg.svd(A,compute_uv=False).tolist(),
                singular_values_CA=np.linalg.svd(C@A,compute_uv=False).tolist(),
                coefficient_nullity=22-actual_rank[0],cases=output,population_ridge=ridge_output,
                invalid_parameter_cases_rejected=invalid_rejected)


def finite_sample_checks(repetitions=30):
    A=action_design()
    pinv=np.linalg.pinv(A)
    results=[]
    for repeats_per_action in (10,100,1000):
        records=[]
        for wrong in range(M):
            r,q,*_=strict_construction(wrong,.2,.1)
            for repeat in range(repetitions):
                rng=np.random.default_rng(np.random.SeedSequence([SEED,wrong,repeats_per_action,repeat]))
                means=rng.binomial(repeats_per_action,r)/repeats_per_action
                ls= A@pinv@means
                ridge_coef=solve(repeats_per_action*A.T@A+100*np.eye(22),repeats_per_action*A.T@means,assume_a='pos')
                ridge=A@ridge_coef
                selected_ls=min(maximizing_set(ls))
                selected_ridge=min(maximizing_set(ridge))
                records.append(dict(wrong_action=wrong,replicate=repeat,ls_action=selected_ls,
                    ridge_action=selected_ridge,ls_chose_population_bad_action=selected_ls==wrong,
                    ridge_chose_population_bad_action=selected_ridge==wrong,
                    ls_regret=float(r.max()-r[selected_ls]),ridge_regret=float(r.max()-r[selected_ridge]),
                    ls_squared_distance_to_population_projection=float(np.mean((ls-q)**2)),
                    ridge_squared_distance_to_population_projection=float(np.mean((ridge-q)**2))))
        summary={}
        for method in ('ls','ridge'):
            selected_bad=sum(row[method+'_chose_population_bad_action'] for row in records)
            summary[method]=dict(population_bad_action_rate=selected_bad/len(records),
                monte_carlo_wilson_ci95=wilson(selected_bad,len(records)),
                mean_regret=float(np.mean([row[method+'_regret'] for row in records])),
                positive_regret_rate=float(np.mean([row[method+'_regret']>1e-12 for row in records])))
        results.append(dict(repeats_per_action=repeats_per_action,binary_observations=20*repeats_per_action,
            simulation_fits=len(records),summary=summary,records=records))
    return dict(parameters=dict(gamma=.2,epsilon=.1,ridge_penalty=100,repetitions_per_action_choice=repetitions),
        sampling='Balanced finite design with independent Binomial rewards for each action; synthetic MC, not real question evidence',
        ridge_objective='sum of all individual binary squared losses plus100 coefficient norm squared',
        tie_rule='lexicographically first maximizing action within absolute numerical tolerance1e-10',cases=results)


def centering_check(X,Y,Xeval,penalty,question_weights=None,output_weight=1.,eval_rewards=None,eval_weights=None,case_id=''):
    """Independent-head Proposition13 check, also accepting common row weights.

    Y contains one target vector per row. To reproduce three-repetition summed
    losses, pass mean rewards as Y and question_weights=3*ones(n).
    """
    X,Y,Xeval=np.asarray(X,float),np.asarray(Y,float),np.asarray(Xeval,float)
    if Y.shape!=(len(X),20) or Xeval.shape[1]!=X.shape[1] or penalty<=0:
        raise ValueError('Matched positive-penalty independent-head design required')
    if np.ndim(output_weight)!=0 or output_weight<=0:
        raise ValueError('The equivalence requires a common action weight, not arbitrary action weights')
    weights=np.ones(len(X)) if question_weights is None else np.asarray(question_weights,float)
    if weights.shape!=(len(X),) or (weights<=0).any():raise ValueError('Positive common question weights required')
    gram=(X.T*weights)@X*output_weight
    rhs=(X.T*weights)@Y*output_weight
    Babs=solve(gram+penalty*np.eye(X.shape[1]),rhs,assume_a='pos')
    # Independently solve the full centered normal equation in column-major vectorization.
    centered_precision=np.kron(C,gram)+penalty*np.eye(X.shape[1]*20)
    Bctr=solve(centered_precision,(rhs@C).ravel(order='F'),assume_a='pos').reshape((X.shape[1],20),order='F')
    absolute=Xeval@Babs
    centered=Xeval@Bctr
    maxsets_abs=[maximizing_set(row) for row in absolute]
    maxsets_ctr=[maximizing_set(row) for row in centered]
    selected_abs=np.asarray([min(row) for row in maxsets_abs])
    selected_ctr=np.asarray([min(row) for row in maxsets_ctr])
    gradient=gram@Bctr@C-rhs@C+penalty*Bctr
    result=dict(case_id=case_id,fit_questions=len(X),query_questions=len(Xeval),features=X.shape[1],
        design_rank=int(np.linalg.matrix_rank(X)),penalty=float(penalty),common_action_weight=float(output_weight),
        coefficient_identity_max_error=float(np.max(abs(Bctr-Babs@C))),
        prediction_centering_max_error=float(np.max(abs(centered-absolute@C))),
        centered_normal_equation_gradient_max=float(np.max(abs(gradient))),
        maximizing_set_mismatches=sum(a!=b for a,b in zip(maxsets_abs,maxsets_ctr)),
        declared_lexicographic_selection_mismatches=int(np.sum(selected_abs!=selected_ctr)),
        raw_argmax_mismatches=int(np.sum(np.argmax(absolute,axis=1)!=np.argmax(centered,axis=1))),
        tie_tolerance=1e-10,
        scope='same design, common action weight, common question weights, isotropic positive coefficient penalty')
    assert result['coefficient_identity_max_error']<1e-9
    assert result['prediction_centering_max_error']<1e-9
    assert result['centered_normal_equation_gradient_max']<1e-9
    assert result['maximizing_set_mismatches']==result['declared_lexicographic_selection_mismatches']==0
    if eval_rewards is not None:
        rewards=np.asarray(eval_rewards,float)
        if rewards.ndim==2:rewards=rewards[:,None,:]
        if rewards.shape[0]!=len(Xeval) or rewards.shape[2]!=20:raise ValueError('Query rewards have wrong shape')
        query_weights=np.ones(len(Xeval))/len(Xeval) if eval_weights is None else np.asarray(eval_weights,float)
        query_weights=query_weights/query_weights.sum()
        rabs=rewards[np.arange(len(Xeval)),:,selected_abs].mean(1)
        rctr=rewards[np.arange(len(Xeval)),:,selected_ctr].mean(1)
        result['outer_accuracy_absolute']=float(query_weights@rabs)
        result['outer_accuracy_centered']=float(query_weights@rctr)
        result['outer_accuracy_difference']=float(query_weights@(rctr-rabs))
        assert result['outer_accuracy_difference']==0
    return result


def centering_qualification():
    rng=np.random.default_rng(SEED)
    rows=[]
    for n,d,rank_deficient in [(30,5,False),(30,17,False),(30,17,True),(120,17,False),(360,17,False)]:
        X=np.column_stack((np.ones(n),rng.normal(size=(n,d-1))))
        if rank_deficient:X[:,-1]=X[:,1]
        Xeval=np.column_stack((np.ones(40),rng.normal(size=(40,d-1))))
        if rank_deficient:Xeval[:,-1]=Xeval[:,1]
        Y=rng.integers(0,2,(n,3,20)).mean(1)
        eval_rewards=rng.integers(0,2,(40,3,20))
        for row_mode in ('uniform','heterogeneous'):
            weights=np.ones(n) if row_mode=='uniform' else rng.uniform(.2,2,n)
            for penalty in (.1,100.):
                row=centering_check(X,Y,Xeval,penalty,weights,eval_rewards=eval_rewards,
                                    case_id=f'n{n}_d{d}_rankdef{rank_deficient}_{row_mode}_lambda{penalty}')
                rows.append(row)
    # Deliberate exact tie: both losses must preserve the full twenty-action maximizer set.
    X=np.column_stack((np.ones(60),rng.normal(size=(60,4))))
    Y=np.repeat(rng.uniform(0,1,(60,1)),20,axis=1)
    rows.append(centering_check(X,Y,X[:20],100.,case_id='all_action_scores_tied'))
    # Out-of-scope negative example. Identical nonuniform action weights in both losses are insufficient.
    X=np.column_stack((np.ones(60),rng.normal(size=(60,5))))
    Y=rng.uniform(0,1,(60,20))
    gram=X.T@X
    rhs=X.T@Y
    action_weights=np.linspace(.2,3.,20)
    Babs=np.column_stack([solve(action_weights[j]*gram+2*np.eye(6),action_weights[j]*rhs[:,j],assume_a='pos') for j in range(20)])
    precision=C@np.diag(action_weights)@C
    Bctr=solve(np.kron(precision,gram)+2*np.eye(120),(rhs@precision).ravel(order='F'),assume_a='pos').reshape((6,20),order='F')
    counterexample=dict(nonuniform_action_weights=action_weights.tolist(),
        coefficient_identity_max_error=float(np.max(abs(Bctr-Babs@C))),
        prediction_centering_max_error=float(np.max(abs(X@Bctr-X@Babs@C))),
        expected='Generally unequal, outside Proposition13')
    assert counterexample['coefficient_identity_max_error']>1e-5
    return dict(cases=rows,nonuniform_action_weight_counterexample=counterexample,
        real_outer_checks='The parent training pipeline should call centering_check with each actual F/E design; synthetic checks are reported separately')


def report(result):
    p=result['population']
    rows=['# 01 理论构造与实现一致性','',
          '## 研究问题','',
          '验证导师初稿附录B的无并列二阶投影反例，以及附录C命题13的匹配Ridge中心化等价性。数值资格与真实路由效果分别记录。','',
          '## 输入与设置','',
          f"稿件SHA256为`{MANUSCRIPT_SHA256}`，全文29页。已核对PDF第12、17页公式图像和对应双栏文字。六选三组合按Python itertools字典序枚举，基础随机种子为{SEED}。",'',
          '## 总体投影反例','',
          '|检验|实测|', '|---|---:|',
          f"|rank(A)|{p['ranks']['A']}|",f"|rank(CA)|{p['ranks']['CA']}|",f"|rank(C)|{p['ranks']['C']}|",
          f"|二阶系数零空间维数|{p['coefficient_nullity']}|",f"|指定错误动作与合法参数交叉案例|{len(p['cases'])}|",
          f"|总体投影最大误差|{max(row['population_projection_max_error'] for row in p['cases']):.3g}|",'',
          '每个指定错误组合S0，将其成员与补集成员一一配对。三个配对均选一个成员的八个transversal上取w(S)=(-1)^|S∩S0|，其余取0。线性扰动h(S)=|S∩S0|/3−1/2。程序直接投影真实均值向量，独立核对A转置乘w为0、预测最大组合唯一、真实最大组合恰有三个，以及MSE和Regret恒等式。','',
          '在gamma=0.2、epsilon=0.1时，指定错误组合的真实奖励为0.35，最优真实奖励为0.7166667，Regret为0.3666667。总体最优预测MSE为0.016，预测最大分数0.55，次大分数0.5166667。因此错误选队具有严格分数间隔。','',
          '总体投影解是唯一的评分向量，22维表示系数因7维零空间仍可不唯一。另用有界线性分数唯一选中一个真实最优组合，得到零Regret，区分评分逼近限制与策略表达能力。','',
          '## 有限样本与正则化对照','',
          '有限样本采用每个动作相同观测次数的二值奖励抽样，每个指定S0及题量分别使用独立Monte Carlo种子。Ridge使用所有个体观测的平方损失和系数惩罚100。这个目标与无正则总体投影分开。','',
          '|每动作观测数|MC拟合数|LS选中总体错误动作比例|Ridge选中总体错误动作比例|LS平均Regret|Ridge平均Regret|',
          '|---|---:|---:|---:|---:|---:|']
    for case in result['finite']['cases']:
        a,b=case['summary']['ls'],case['summary']['ridge']
        rows.append(f"|{case['repeats_per_action']}|{case['simulation_fits']}|{a['population_bad_action_rate']:.4f}|{b['population_bad_action_rate']:.4f}|{a['mean_regret']:.6f}|{b['mean_regret']:.6f}|")
    cases=result['centering']['cases']
    rows.extend(['','全部二值MC结果、Wilson区间及140×3个总体Ridge对照均保存在原始JSON。有限样本与Ridge的结果只用于刻画同一构造的实际拟合行为，均不被直接套作定理的总体最优结论。','',
        '## 中心化Ridge','',
        '程序独立求解绝对损失的逐头正规方程和中心化损失的完整Kronecker正规方程。匹配共同题权重、各动作同一标量输出权重及各系数同一正惩罚。验证Bctr=Babs C、预测中心化、最大分数组合集合和带1e−10数值并列容差的字典序动作。','',
        f"合成设置共{len(cases)}例，包含秩亏问题设计及完全并列。系数最大误差{max(row['coefficient_identity_max_error'] for row in cases):.3g}，中心化预测最大误差{max(row['prediction_centering_max_error'] for row in cases):.3g}。最大组合集合与固定并列动作差异均为0。合成评价奖励下准确率差均为0。",'',
        f"直接使用无数值容差的np.argmax时，全部{sum(row['query_questions'] for row in cases)}个合成评价行中有{sum(row['raw_argmax_mismatches'] for row in cases)}行产生差异，均来自20行完全并列的压力测试。完整Kronecker求解残留的浮点微差会被无容差argmax放大成动作变化。该差异单独保留，预声明的1e−10并列容差下差异为0。",'',
        '共同题权重可以不同，因为它从左侧作用于问题设计。不同动作权重一般不与C交换，无法仅凭两次拟合使用相同动作权重就主张等价。负资格示例保留两次拟合相同但非统一的动作权重，得到非零的Bctr−Babs C，明确命题条件。','',
        f"该负资格示例的系数差最大为{result['centering']['nonuniform_action_weight_counterexample']['coefficient_identity_max_error']:.6g}。",'',
        '## 统计结论','',
        '本阶段是确定性数学实现核验与有限样本合成诊断。定理反例与匹配Ridge等价性的数值证据置信度高，真实数据中二阶结构是否造成主要误差由E5-B另行评价。中心化等价本身不构成新方法提升。','',
        '## 优势与风险','',
        '优势是覆盖全部20个指定错误组合，严格区分总体投影、有限观测与系数惩罚。风险在于数值测试覆盖有限参数网格，真实矩阵的拟合设计、权限和外折准确率必须另行核对。原始直接argmax在浮点完全并列时可能受舍入影响，资格使用预声明的并列容差。','',
        '## 下一步','',
        'E5-B训练入口可调用centering_check，对实际F区同一P2设计、相同三生成损失权重和外折E奖励核验等价。真实外折检查与本合成结果分开保存。',''])
    return '\n'.join(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    start=time.perf_counter()
    result=dict(schema='e5_theory_implementation_v1',status='PASS',manuscript_sha256=MANUSCRIPT_SHA256,
        seed=SEED,population=population_checks(),finite=finite_sample_checks(),centering=centering_qualification())
    result['wall_s']=time.perf_counter()-start
    result['new_llm_calls']=0
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'THEORY_IMPLEMENTATION_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (args.output/'01_THEORY_IMPLEMENTATION_CHECKS.md').write_text(report(result),encoding='utf8')
    print(json.dumps(dict(status='PASS',population_cases=len(result['population']['cases']),finite_fits=sum(c['simulation_fits'] for c in result['finite']['cases']),
        centering_cases=len(result['centering']['cases']),wall_s=result['wall_s'])),flush=True)


if __name__=='__main__':main()
