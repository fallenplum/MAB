"""Exact arithmetic and tiny convex numerical checks for E19 workflow A.

No experiment matrix, question, answer, trained router or model backend is read.
The optimizations below are the stated one- and three-dimensional constructions.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize, minimize_scalar


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def as_fraction(value):
    return value if isinstance(value, Fraction) else Fraction(str(value))


def positive(value):
    return max(value, 0)


def binomial_weights(a, repeats):
    a = as_fraction(a)
    require(0 <= a <= 1 and type(repeats) is int and repeats >= 1,
            "Require a in[0,1] and positive integer repeats")
    return [Fraction(math.comb(repeats, k)) * a**k * (1-a)**(repeats-k)
            for k in range(repeats+1)]


def a1_risk(gap, weights, p, regularization):
    gap, p, regularization = map(as_fraction, (gap, p, regularization))
    repeats = len(weights)-1
    tied = weights[0] * (2*p*positive(-gap) + 2*(1-p)*positive(gap))
    other = sum(weights[k] * positive(2*gap + Fraction(k, repeats))
                for k in range(1, repeats+1))
    return tied + other + regularization*gap*gap/2


def a1_subgradient(gap, weights, p, regularization):
    """Exact left/right derivatives at hinge breakpoints."""
    gap, p, regularization = map(as_fraction, (gap, p, regularization))
    repeats, tied = len(weights)-1, weights[0]
    active = sum(weights[k] for k in range(1,repeats+1)
                 if 2*gap + Fraction(k,repeats) > 0)
    equal = sum(weights[k] for k in range(1,repeats+1)
                if 2*gap + Fraction(k,repeats) == 0)
    if gap == 0:
        return 2*active-2*tied*p, 2*active+2*tied*(1-p)
    tie_slope = -2*tied*p if gap < 0 else 2*tied*(1-p)
    left = regularization*gap + tie_slope + 2*active
    return left, left+2*equal


def a1_exact_minimizer(a, repeats, p, regularization):
    """Enumerate every binomial hinge and every stationary interval point."""
    p, regularization = map(as_fraction, (p, regularization))
    require(0 <= p <= 1 and regularization > 0, "Require p in[0,1], lambda>0")
    weights = binomial_weights(a, repeats)
    knots = sorted({Fraction(-k, 2*repeats) for k in range(repeats+1)})
    candidates = list(knots)
    for left, right in zip(knots, knots[1:]):
        midpoint = (left+right)/2
        active = sum(weights[k] for k in range(1,repeats+1)
                     if 2*midpoint + Fraction(k,repeats) > 0)
        stationary = (2*weights[0]*p-2*active)/regularization
        if left < stationary < right:
            candidates.append(stationary)
    gap = min(candidates, key=lambda value: a1_risk(value, weights, p, regularization))
    left, right = a1_subgradient(gap, weights, p, regularization)
    require(left <= 0 <= right, "Candidate lacks exact optimal subgradient")
    return gap, a1_risk(gap, weights, p, regularization)


def first_strict_repeat(a, p):
    """Strict threshold with exact correction at integer logarithmic boundaries."""
    a, p = map(as_fraction, (a, p))
    require(0 < a < 1 and 0 <= p <= 1, "Log threshold requires interior a")
    ratio = math.log1p(float(p)) / -math.log1p(-float(a))
    candidate = max(1, math.floor(ratio)+1)

    def strict(repeats):
        t = (1-a)**repeats
        return t*p < 1-t

    while not strict(candidate):
        candidate += 1
    while candidate > 1 and strict(candidate-1):
        candidate -= 1
    return candidate, ratio


def check_a1():
    checked = 0
    maximum_numeric_gap, maximum_distance = 0.0, 0.0
    for a in (Fraction(1,100), Fraction(1,10), Fraction(1,2), Fraction(9,10)):
        for repeats in (1,2,3,5,10):
            weights = binomial_weights(a, repeats)
            require(sum(weights) == 1, "Binomial probabilities do not sum to one")
            for p in (Fraction(0), Fraction(1,2), Fraction(1)):
                for regularization in (Fraction(1,10), Fraction(1), Fraction(10)):
                    gap, value = a1_exact_minimizer(a, repeats, p, regularization)
                    t = (1-a)**repeats
                    require((gap == 0) == (t*p >= 1-t), "Collapse criterion failed")
                    require(gap <= 0, "Construction spuriously prefers A strictly")
                    numeric = minimize_scalar(
                        lambda d: float(a1_risk(Fraction(float(d)), weights, p, regularization)),
                        bounds=(-0.6,0.1), method="bounded", options={"xatol":1e-12})
                    require(numeric.success, "Independent scalar optimizer failed")
                    error = abs(float(numeric.fun)-float(value))
                    distance = abs(float(numeric.x)-float(gap))
                    require(error < 1e-7 and distance < 1e-5,
                            "Exact and independent scalar optima disagree")
                    maximum_numeric_gap = max(maximum_numeric_gap,error)
                    maximum_distance = max(maximum_distance,distance)
                    checked += 1
    boundaries = []
    for a,p,at_boundary,first in (
        (Fraction(1,2),Fraction(1),1,2),
        (Fraction(1,3),Fraction(1,2),1,2),
        (Fraction(1,10),Fraction(19,81),2,3),
    ):
        gap,_ = a1_exact_minimizer(a,at_boundary,p,1)
        subsequent,_ = a1_exact_minimizer(a,first,p,1)
        result,ratio = first_strict_repeat(a,p)
        require(gap == 0 and subsequent < 0 and result == first,
                "Strict integer threshold incorrectly treated as a ceiling")
        boundaries.append(dict(a=str(a),p=str(p),equality_repeats=at_boundary,
                               first_strict_repeat=first,logarithmic_threshold=ratio))
    examples = []
    for p,expected in ((Fraction(1),69),(Fraction(1,2),41),(Fraction(0),1)):
        minimum,ratio = first_strict_repeat(Fraction(1,100),p)
        require(minimum == expected, "Illustrative repeat threshold differs")
        examples.append(dict(a="1/100",p=str(p),first_strict_repeat=minimum,
                             logarithmic_threshold=ratio,role="Mathematical illustration only"))
    for p in (Fraction(0),Fraction(1,2),Fraction(1)):
        for regularization in (Fraction(1),Fraction(10)):
            zero,_ = a1_exact_minimizer(0,3,p,regularization)
            one,_ = a1_exact_minimizer(1,3,p,regularization)
            require(zero == 0 and one == max(Fraction(-1,2),-2/regularization),
                    "Endpoint a boundary failed")
    return dict(status="PASS",exact_grid_cases=checked,
                independent_scalar_optimizations=checked,
                maximum_objective_error=maximum_numeric_gap,
                maximum_minimizer_distance=maximum_distance,
                strict_integer_boundaries=boundaries,illustrative_thresholds=examples,
                finite_n_claim="At fixed L and fixed positive regularization, the finite-alphabet empirical risk converges to the same population risk as independent question count grows.")


FEATURES = np.asarray([[1,4,4],[1,4,0],[1,0,4]],dtype=float)


def a2_theta(beta):
    beta = as_fraction(beta)
    return [Fraction(-7,300)+beta/200,
            Fraction(-2,25)+beta/50,
            Fraction(-1,75)+beta/50]


def a2_qp(beta):
    """Independent six-variable epigraph QP with three genuine coefficients."""
    phi_t,phi_1,phi_2 = FEATURES
    matrix = np.zeros((6,6))
    matrix[0,:3],matrix[0,3] = -phi_t,1
    matrix[1,:3],matrix[1,3] = phi_t,1
    matrix[2,:3],matrix[2,4] = -2*phi_1,1
    matrix[3,4] = 1
    matrix[4,:3],matrix[4,5] = -2*phi_2,1
    matrix[5,5] = 1
    lower = np.asarray([0,0,1,0,1,0],dtype=float)
    linear = np.asarray([0,0,0,0.3*float(beta),0.6,0.1])

    def objective(value):
        return 30*float(value[:3]@value[:3])+float(linear@value)

    def gradient(value):
        answer = linear.copy()
        answer[:3] += 60*value[:3]
        return answer

    result = minimize(objective,np.asarray([0,0,0,0,1,1],dtype=float),jac=gradient,
                      method="SLSQP",constraints=[LinearConstraint(matrix,lower,np.inf)],
                      options={"ftol":1e-12,"maxiter":1000})
    require(result.success, "Independent synthetic convex QP failed: "+str(result.message))
    require(np.min(matrix@result.x-lower) > -1e-8,"Synthetic QP is infeasible")
    return result


def check_a2():
    rows = []
    critical = Fraction(46,51)
    for beta in (Fraction(0),Fraction(1,2),critical,Fraction(1)):
        theta = a2_theta(beta)
        exact_gaps = [theta[0]+4*theta[1]+4*theta[2],
                      theta[0]+4*theta[1],theta[0]+4*theta[2]]
        formulas = [Fraction(-238,600)+99*beta/600,
                    Fraction(-206,600)+51*beta/600,
                    Fraction(-46,600)+51*beta/600]
        require(exact_gaps == formulas,"Analytic coefficient/gap formula failed")
        require(exact_gaps[0] < 0 and 2*exact_gaps[1]+1 > 0 and 2*exact_gaps[2]+1 > 0,
                "Claimed common active region does not hold")
        numeric_theta = np.asarray([float(value) for value in theta])
        stationary = 60*numeric_theta-0.3*float(beta)*FEATURES[0]+1.2*FEATURES[1]+0.2*FEATURES[2]
        require(np.linalg.norm(stationary,np.inf)<1e-12,"Analytic stationary gradient failed")
        qp = a2_qp(beta)
        error = float(np.linalg.norm(qp.x[:3]-numeric_theta,np.inf))
        require(error<1e-7,"Independent QP and exact theta disagree")
        utility = Fraction(1) if exact_gaps[2]<0 else Fraction(9,10) if exact_gaps[2]>0 else Fraction(19,20)
        rows.append(dict(beta=str(beta),theta_exact=list(map(str,theta)),
                         score_gaps_exact=list(map(str,exact_gaps)),
                         uniform_argmax_utility=float(utility),qp_solver="SciPy SLSQP epigraph convex QP",
                         qp_success=bool(qp.success),qp_iterations=int(qp.nit),
                         coefficient_max_error=error,stationarity_inf=float(np.linalg.norm(stationary,np.inf))))
    require(rows[0]["uniform_argmax_utility"]==1 and rows[-1]["uniform_argmax_utility"]==0.9,
            "Strict endpoint utilities failed")
    require(rows[2]["uniform_argmax_utility"]==0.95,"Critical uniform-tie utility failed")
    no_penalty = [Fraction(-1),Fraction(1,8),Fraction(1,8)]
    require(no_penalty[0]+4*no_penalty[1]+4*no_penalty[2]==0
            and no_penalty[0]+4*no_penalty[1]==Fraction(-1,2)
            and no_penalty[0]+4*no_penalty[2]==Fraction(-1,2),
            "Unregularized zero-loss witness failed")
    return dict(status="PASS",critical_beta=str(critical),strong_convexity=60,
                construction_regularization="30||theta||^2=lambda||theta||^2/2 with lambda=60",
                actual_router_regularization="PENDING_ROOT_FROZEN_IMPLEMENTATION_AUDIT",
                independent_tiny_convex_qps=len(rows),rows=rows,
                local_x2_slope_exact=str(Fraction(3,10)*17/60),
                original_x2_gap_exact="-23/300",critical_x2_uniform_tie_utility=0.95)


def set_loss(scores,rewards):
    alpha = (rewards == rewards.max()).astype(float)
    alpha /= alpha.sum()
    reflected = 2*scores-rewards
    return float(reflected.max()-alpha@reflected)


def check_a3():
    rng = np.random.default_rng(20261116019)
    scores = rng.normal(size=(4,20))
    rewards = np.vstack([np.zeros(20),np.ones(20),
                         np.r_[np.ones(3),np.zeros(17)],np.arange(20)/19])
    tied = np.ptp(rewards,axis=1)==0
    losses = np.asarray([set_loss(q,r) for q,r in zip(scores,rewards)])
    original_n = len(scores)
    jt = 2/original_n*sum(q.max()-q.mean() for q in scores[tied])
    errors = []
    for beta in (0,0.25,1):
        weighted = losses.copy()
        weighted[tied] *= beta
        f_beta = weighted.sum()/original_n
        f_zero = losses[~tied].sum()/original_n
        errors.append(abs(f_beta-f_zero-beta*jt))
    require(max(errors)<1e-12,"F_beta=F_zero+beta*J_T identity failed")
    gamma_errors = []
    for gamma in (0,0.5,1):
        q=scores[0]
        alpha=np.full(20,(1-gamma)/20)
        alpha[7]+=gamma
        direct=2*q.max()-2*alpha@q
        expected=2*(q.max()-q.mean())-2*gamma*(q[7]-q.mean())
        gamma_errors.append(abs(direct-expected))
    require(max(gamma_errors)<1e-12,"Direction/intensity decomposition failed")
    return dict(status="PASS",synthetic_rows=original_n,actions=20,
                maximum_endpoint_identity_error=max(errors),
                maximum_direction_decomposition_error=max(gamma_errors),
                original_denominator_retained=True,real_model_training_runs=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("THEORY_CHECKS.json"))
    args=parser.parse_args()
    require(not args.output.exists(),"Theory qualification outputs are append-only")
    started=time.perf_counter()
    result=dict(schema="e19_symmetric_tie_theory_checks_v1",status="PASS",
                a1=check_a1(),a2=check_a2(),a3=check_a3(),
                new_real_model_training_runs=0,new_llm_calls=0,new_experiment_reward_reads=0,
                sealed_or_fresh_reward_reads=0,bootstrap_runs=0,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result["wall_seconds"]=time.perf_counter()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(result,stream,ensure_ascii=False,sort_keys=True,indent=2)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"],a1_cases=result["a1"]["exact_grid_cases"],
                          a2_qps=result["a2"]["independent_tiny_convex_qps"],
                          source_sha256=result["source_sha256"],wall_seconds=result["wall_seconds"])))


if __name__=="__main__":
    main()
