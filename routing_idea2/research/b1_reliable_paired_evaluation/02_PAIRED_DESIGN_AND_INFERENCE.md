# B1 配对估计、分配与有效推断

## 研究问题与对象

冻结默认与候选动作，原始问题为统计单位。令d_i为同题三个生成重复的配对收益平均，m_i为冻结辅助预测。动作相同的题已知d_i=m_i=0，购买概率为0，保留在N中。其余p_i至少0.05。

## 估计器与证明

独立Bernoulli购买A_i时，

\[
\widehat G_N=N^{-1}\sum_i[m_i+A_i(d_i-m_i)/p_i].
\]

给定固定题池及冻结m、p，有E[A_i/p_i]=1，故E_A[估计]=G_N。辅助预测有偏不会破坏这一设计无偏性。独立购买使协方差项为零，因此

\[
V_N=N^{-2}\sum_i(1-p_i)(d_i-m_i)^2/p_i.
\]

对每个被购入项再除一次p_i，得到无偏Vhat。三个生成重复已包含在d_i内。这个方差未包含未来问题及新生成重复的抽样变化。

## 双期望预算分配

最小化sum_i vhat_i/p_i，约束sum p≤B_answer、sum chat_i p_i≤B_execution，p∈[.05,1]。Lagrangian对内部坐标求导得−vhat_i/p_i²+lambda_answer+lambda_execution chat_i=0。由严格凸性及可行域凸性，满足KKT的截断平方根解为全局最优。

\[
p_i=\operatorname{clip}_{[.05,1]}
\sqrt{\widehat v_i/(\lambda_a+\lambda_e\widehat c_i)}.
\]

数值程序通过两个对偶乘子的单调求根求解，报告原始与对偶目标、互补松弛和约束绑定状态。NUMERICAL_QUALIFICATION.json含独立约束优化器对照。工作二阶矩的预测准确性不由KKT保证。

## 固定池保守区间

记e_i=d_i−m_i，b_i=1+|m_i|。Vhat未除N²的非负随机项A_i(1−p_i)e_i²/p_i²的共同上界为c。独立有界非负和的lower-tail Chernoff不等式可反解得到

\[
V_{\rm sum}\le\widehat V_{\rm sum}+ct+
\sqrt{2ct\widehat V_{\rm sum}+c^2t^2}
\]

以至少1−delta_v成立，其中t=log(1/delta_v)。还可与确定性的sum(1−p_i)b_i²/p_i取较小值。对中心化的A_i e_i/p_i−e_i使用Bernstein界，界限M=max_i b_i max(1,(1−p_i)/p_i)。双侧尾概率与方差上界失效概率采用union bound。单侧决定采用各自5%界，阈值0和0.01分别报告。

完整枚举验证估计器与Vhat无偏。保守界证明及进一步随机小池枚举由独立模块审阅。没有标签时返回由结构性零和有界收益给出的宽区间。工作后验与普通Wald区间单列为渐近诊断。

## 未来总体与LTT

总体推断使用独立同分布原始问题簇，p(x)、m(x)由独立历史固定，且有全输入支持上的IPW变量界。采用总体二阶矩上界及固定时点Bernstein推断。当前输入池归一化概率一般会让变量相互依赖，不能直接调用该iid公式。

LTT是检验组织框架。单候选对应一个检验。原生有界损失检验只在合法iid有界观测下使用，IPW变量超出[0,1]时不调用原生Bernoulli公式。当前研究不进行反复查看后提前停止。

## 优势、风险与后续

设计无偏性允许辅助预测存在偏差，这是可直接证明的性质。逆概率权重在失准时会放大方差，保守界也可能很宽。固定池和总体目标的覆盖范围各自限定，后续是否值得独立评价取决于完整精度与费用证据。
