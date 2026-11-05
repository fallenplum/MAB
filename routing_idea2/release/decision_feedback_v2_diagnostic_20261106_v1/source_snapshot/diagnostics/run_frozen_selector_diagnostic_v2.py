"""Read-only frozen-model replay and descriptive train/validation diagnostics.

No fit routine, LLM backend, raw-answer scorer, fresh/test outcome or search is used.
The combination projection below is a descriptive per-question algebraic residual;
it is never an input-based learned predictor, candidate router, or deployment rule.
"""
from pathlib import Path
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = Path(r"E:\语言模型组合路由")
sys.path.insert(0, str(ROOT / "src"))
from budget_router_v2 import full_feedback as ff
from budget_router.features import SubsetFeatureMap
from budget_router.ranking_evaluation import ranking_metrics

FF = ROOT / "artifacts/private/routing_redesign_20261104/ff_v1"
OUT = FF / "diagnostics_safe_pause_20261106_v1/selector_generalization"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def ref(p):
    return {"path": str(p), "sha256": sha(p)}


def check(record):
    p = Path(record["path"])
    assert sha(p) == record["sha256"], p
    return p


def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def put(p, v):
    with p.open("x", encoding="utf-8") as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def context(inp, model):
    x = (inp["embeddings"] - model["pca_mean"]) @ model["pca_components"].T
    return np.column_stack((np.ones(len(x)), (x-model["scale_mean"])/model["scale_scale"]))


def it3_choices(inp, model):
    scores = context(inp, model) @ model["theta"].T
    members = np.sort(np.argsort(-scores, axis=1, kind="stable")[:, :3], axis=1)
    teams = list(itertools.combinations(range(6), 3))
    return np.asarray([teams.index(tuple(row)) for row in members]), scores


def counts(delta):
    d = np.asarray(delta)
    return {"wins": int((d>0).sum()), "losses": int((d<0).sum()), "ties": int((d==0).sum()),
            "support": int(d.size), "mean_difference": float(d.mean())}


def support_pair(a, b, ca, cb, task, subtask, seeds):
    d = a-b
    return {"seed_question_outcomes": counts(d), "unique_questions_averaging_three_generations": counts(d.mean(axis=0)),
            "same_selected_team_questions": int((ca == cb).sum()),
            "different_selected_team_questions": int((ca != cb).sum()),
            "per_generation": [{"generation_seed": int(seed), **counts(d[i])} for i, seed in enumerate(seeds)],
            "per_task": {t: {"question_count": int((task==t).sum()),
                                "seed_question_outcomes": counts(d[:, task==t]),
                                "unique_questions_averaging_three_generations": counts(d[:, task==t].mean(axis=0)),
                                "same_selected_team_questions": int((ca[task==t] == cb[task==t]).sum())}
                         for t in ff.TASKS},
            "per_bbh_subtask": {s: {"question_count": int((subtask==s).sum()),
                                      "seed_question_outcomes": counts(d[:, subtask==s]),
                                      "unique_questions_averaging_three_generations": counts(d[:, subtask==s].mean(axis=0))}
                                  for s in sorted(set(subtask[task=="bbh"]))},
            "paired_cluster_interval": ff._cluster_difference(d, task, subtask, seeds, 2000, 20261104)}


def domain_stats(y, p, phi, model, inp):
    centered = inp["embeddings"] - model["pca_mean"]
    reduced = centered @ model["pca_components"].T
    mu = y.mean(axis=0)
    bm = SubsetFeatureMap(6, 3)._basis
    u, s, vh = np.linalg.svd(bm, full_matrices=False)
    rank = int((s > max(bm.shape)*np.finfo(float).eps*s[0]).sum())
    projector = u[:, :rank] @ u[:, :rank].T
    unrepresented = mu - mu @ projector
    centered_target = mu - mu.mean(axis=1, keepdims=True)
    centered_pred = p - p.mean(axis=1, keepdims=True)
    current_gap_error = float(np.mean((centered_target-centered_pred)**2))
    null_mse = float(np.mean(unrepresented**2))
    # Identity: squared error across repeats = repeat variability + mean-target error.
    original_mse = float(np.mean((y-p[None])**2))
    stochastic_component = float(np.mean((y-mu[None])**2))
    mean_target_mse = float(np.mean((mu-p)**2))
    assert abs(original_mse-stochastic_component-mean_target_mse) < 1e-12
    return {"questions": len(inp["question_ids"]),
            "original_embedding_dimensions": 384, "question_context_dimensions": int(phi.shape[1]),
            "context_matrix_rank": int(np.linalg.matrix_rank(phi)),
            "PCA16_fraction_of_centered_input_squared_norm": float(np.sum(reduced**2)/np.sum(centered**2)),
            "combination_basis_shape": list(bm.shape), "combination_basis_rank": rank,
            "combination_vector_dimension_not_spanned": 20-rank,
            "nominal_theta_parameters": 374,
            "design_rank_product": int(np.linalg.matrix_rank(phi))*rank,
            "current_prediction_per_repeat_binary_score_MSE": original_mse,
            "observed_repeat_variability_component_MSE": stochastic_component,
            "current_prediction_three_seed_mean_score_MSE": mean_target_mse,
            "observed_repeat_variability_fraction_of_total_MSE": stochastic_component/original_mse,
            "current_within_question_centered_mean_score_MSE": current_gap_error,
            "mean_target_component_outside_second_order_combination_span_MSE": null_mse,
            "outside_combo_span_fraction_of_current_centered_MSE": null_mse/current_gap_error if current_gap_error else None,
            "projection_interpretation": "Descriptive orthogonal projection across 20 observed combination values separately per question; not a fitted input predictor, not a deployable oracle, and not a causal diagnosis. Three-seed means retain sampling noise.",
            "predicted_cell_min": float(p.min()), "predicted_cell_max": float(p.max()),
            "predicted_cells_below_zero": int((p<0).sum()), "predicted_cells_above_one": int((p>1).sum()),
            "predicted_top_margin_quantiles": [float(v) for v in np.quantile(np.sort(p,axis=1)[:,-1]-np.sort(p,axis=1)[:,-2], [0,.25,.5,.75,1])]}


def summarize(inp, outcomes, pred, itchoice, model):
    y, task, subtask, seeds = [outcomes[k] if k in outcomes else inp[k] for k in
                              ("aggregate_scores", "task_labels", "subtask_labels", "generation_seeds")]
    n = len(task)
    assert y.shape == (3,n,20) and set(np.unique(y)) <= {0,1}
    choices = {k: np.argmax(v,axis=1) for k,v in pred.items()}
    choices["independent_top3"] = itchoice
    selected = {k: y[np.arange(3)[:,None], np.arange(n)[None], v[None]] for k,v in choices.items()}
    oracle = y.max(axis=2)
    methods = {}
    for name, a in selected.items():
        methods[name] = {"accuracy": ff._macro(a,task), "top1_regret": ff._macro(oracle-a,task),
                         "per_task": {t: {"questions": int((task==t).sum()), "accuracy": float(a[:,task==t].mean()),
                                          "top1_regret": float((oracle-a)[:,task==t].mean())} for t in ff.TASKS},
                         "per_generation_accuracy": [{"generation_seed": int(seed), "accuracy": ff._macro(a[i],task)} for i,seed in enumerate(seeds)],
                         "selected_team_counts": {str(int(team)): int(num) for team,num in zip(*np.unique(choices[name],return_counts=True))},
                         "per_bbh_subtask": {s: {"questions": int((subtask==s).sum()), "accuracy": float(a[:,subtask==s].mean())}
                                             for s in sorted(set(subtask[task=="bbh"]))}}
    comparison = {k: support_pair(selected["selector"], selected[k], choices["selector"], choices[k],task,subtask,seeds)
                  for k in selected if k != "selector"}
    ranks = []
    for gi,seed in enumerate(seeds):
        rows = [ranking_metrics(pred["selector"][r],y[gi,r]) for r in range(n)]
        valid = [r for r in rows if r["pairwise_accuracy"] is not None]
        ranks.append({"generation_seed": int(seed), "rankable_questions": len(valid),
                      "question_mean_pairwise_accuracy": float(np.mean([r["pairwise_accuracy"] for r in valid])),
                      "ordered_pair_support": sum(r["ordered_pair_support"] for r in valid)})
    return {"questions": n,"generation_seeds": seeds.tolist(), "oracle_accuracy": ff._macro(oracle,task),
            "methods": methods,"selector_minus_other": comparison,"ranking": ranks,
            "representation_and_error": domain_stats(y,pred["selector"],context(inp,model),model,inp)}, choices, selected


def main():
    cpu, wall = time.process_time(), time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    target = OUT / "SELECTOR_GENERALIZATION_DIAGNOSTIC_v1.json"
    assert not target.exists(), "Do not overwrite a diagnostic run"
    receipt = read(FF/"model/MODEL_FROZEN.json")
    assert sha(FF/"model/MODEL_FROZEN.json") == "a0a87c669a5b9ed91945a5ab4d5ebf046231a9a60a629516250caf3e37cc3535"
    assert sha(ROOT/"src/budget_router_v2/full_feedback.py") == "58b20505182149eef2537a42867c04ba98dfc6959f527fe62c1b8c5ff3f39dc2"
    for r in receipt["algorithm_sources"]:
        check(r)
    model_path = check(receipt["model"])
    train_path, train_y_path = check(receipt["training_inputs"]), check(receipt["training_outcomes"])
    val_path = check(receipt["validation_inputs"])
    eval_path = FF/"model/validation_evaluation/evaluation.json"
    original_eval = read(eval_path)
    val_y_path = check(original_eval["outcomes"])
    vp_receipt_path = check(original_eval["frozen_predictions"])
    vp_receipt = read(vp_receipt_path)
    assert check(vp_receipt["model_receipt"]) == FF/"model/MODEL_FROZEN.json"
    vp_path = check(vp_receipt["predictions"])
    it_receipt_path = FF/"it3_v1/MODEL_FROZEN.json"
    it_receipt = read(it_receipt_path)
    it_model_path = check(it_receipt["model"])
    it_val_receipt_path = FF/"it3_v1/validation/PREDICTIONS_FROZEN.json"
    it_val_receipt = read(it_val_receipt_path)
    assert check(it_val_receipt["model_receipt"]) == it_receipt_path
    it_val_path = check(it_val_receipt["predictions"])
    protected = [FF/"model/MODEL_FROZEN.json", model_path, FF/"pred/PREDICTIONS_FROZEN.json",FF/"pred/predictions.npz",
                 train_path, train_y_path,val_path,val_y_path,vp_receipt_path,vp_path,it_receipt_path,it_model_path,it_val_receipt_path,it_val_path,
                 ROOT/"src/budget_router_v2/full_feedback.py",ROOT/"src/budget_router/features.py",ROOT/"src/budget_router/learner.py"]
    before = {str(p): sha(p) for p in protected}
    model, itmodel = arrays(model_path), arrays(it_model_path)
    train,val,ty,vy,vp,itvp = [arrays(p) for p in [train_path,val_path,train_y_path,val_y_path,vp_path,it_val_path]]
    for inp,ys in [(train,ty),(val,vy)]:
        assert set(inp) == ff.INPUT_KEYS
        for k in ["question_ids","canonical_hashes"]:assert np.array_equal(inp[k],ys[k])
        assert np.array_equal(ys["generation_seeds"], [42,137,2024])
    assert not (set(train["question_ids"]) & set(val["question_ids"]))
    assert not (set(train["canonical_hashes"]) & set(val["canonical_hashes"]))
    predtrain,predval = ff.predictions(model,train),ff.predictions(model,val)
    replay = {name: {"max_abs_score_difference": float(np.max(np.abs(predval[name]-vp[name]))),
                     "selected_team_differences": int(np.sum(np.argmax(predval[name],axis=1)!=np.argmax(vp[name],axis=1))),
                     "scores_bitwise_identical": bool(np.array_equal(predval[name],vp[name]))}
              for name in ff.METHODS}
    for name in ff.METHODS:
        assert replay[name]["max_abs_score_difference"] < 1e-6 and replay[name]["selected_team_differences"] == 0,name
    # Metrics on validation always use original frozen arrays, not reconstructed scores.
    predval = {name: vp[name] for name in ff.METHODS}
    ict, _ = it3_choices(train,itmodel)
    icv, isv = it3_choices(val,itmodel)
    assert np.array_equal(icv,itvp["selected_team_index"])
    itreplay = {"max_abs_score_difference": float(np.max(np.abs(isv-itvp["predicted_member_scores"]))),
                "selected_team_differences": int(np.sum(icv!=itvp["selected_team_index"]))}
    assert itreplay["max_abs_score_difference"] < 1e-6
    trainreport, tc, ts = summarize(train,ty,predtrain,ict,model)
    valreport,vc,vs = summarize(val,vy,predval,icv,model)
    assert abs(valreport["methods"]["selector"]["accuracy"] - original_eval["methods"]["selector"]["task_macro_quality"]) < 1e-12
    assert before == {str(p):sha(p) for p in protected}, "Protected input was mutated"
    report = {"schema":"routing_v2_frozen_selector_generalization_diagnostic_v1", "status":"EXPLORATORY_TRAIN_VALIDATION_DIAGNOSTIC_ONLY",
              "scientific_learnability_established":False,"formal_stage2_complete":False,
              "new_model_calls":0,"new_answer_scoring_operations":0,"new_model_fits":0,
              "fresh_or_sealed_outcome_files_opened":0,"hyperparameter_search":False,
              "read_scope":"Frozen model/source bytes; existing train/validation embeddings and scored outcome matrices; no raw answers or gold answers parsed.",
              "training_is_in_sample":True,"independent_top3_is_post_original_validation_exploratory":True,
              "source_lineage": [ref(p) for p in protected],"frozen_validation_choices_replayed_exactly":True,
              "validation_metrics_use_original_frozen_score_arrays":True,
              "floating_point_replay":{"original_methods":replay,"independent_top3":itreplay,
                                       "asserted_max_score_difference_bound":1e-6,
                                       "no_claim_of_bitwise_score_identity":True},
              "train":trainreport,"validation":valreport,
              "train_minus_validation_accuracy":{k:trainreport["methods"][k]["accuracy"]-valreport["methods"][k]["accuracy"] for k in trainreport["methods"]},
              "model_structure":{"question_features":"Frozen MiniLM384 -> training PCA16 -> training scaling -> prepend constant; no task/subtask one-hot",
                                  "combination_features":"Shared second-order structured ridge: kron(phi17, [1, six memberships, fifteen member-pair products]); not twenty independent heads",
                                  "theta_shape":list(model["theta"].shape),"rank_deficiency_is_expected_for_fixed_K":True,
                                  "per_BBH_training_subtask_questions":"11 or 12","per_BBH_validation_subtask_questions":"3 or 4"},
              "limitations":["Validation has already informed this diagnostic. No claim is independent test confirmation.",
                             "In-sample performance does not estimate generalization; simple baselines also use training labels.",
                             "Three generations share the same questions; confidence intervals keep them in one question cluster.",
                             "Intervals across multiple contrasts/subtasks are descriptive and lack familywise correction.",
                             "An interval crossing zero does not prove absence of useful routing signal.",
                             "Second-order projection residual is a descriptive representational lower bound on the observed mean matrix, not causal evidence that architecture is the principal failure.",
                             "PCA retained variance is not retained task-relevant information; no alternative representation or selector was fitted.",
                             "Original code labels strata support >=2 adequate; BBH cells with 3-4 validation questions remain weak for scientific claims."],
              "runtime":{"cpu_s":time.process_time()-cpu,"wall_s":time.perf_counter()-wall},"source_script":ref(Path(__file__))}
    put(target,report)
    np.savez_compressed(OUT/"EXISTING_MATRIX_SELECTED_OUTCOMES_DIAGNOSTIC_v1.npz",
        train_task_labels=train["task_labels"],train_subtask_labels=train["subtask_labels"],
        validation_task_labels=val["task_labels"],validation_subtask_labels=val["subtask_labels"],
        **{"train_choice_"+k:v for k,v in tc.items()},**{"validation_choice_"+k:v for k,v in vc.items()},
        **{"train_selected_"+k:v for k,v in ts.items()},**{"validation_selected_"+k:v for k,v in vs.items()})
    lines=["# 冻结选择器的已有数据诊断", "", "本报告只回放原冻结模型与已有训练/验证评分矩阵；未新增模型调用、答案评分、训练或超参数搜索，未读取 fresh/封存答案。", "", "| 方法 | 训练准确率 | 验证准确率 | 训练 Top-1 Regret | 验证 Top-1 Regret |", "|---|---:|---:|---:|---:|"]
    for name in trainreport["methods"]:
        a,b=trainreport["methods"][name],valreport["methods"][name]
        lines.append(f"| {name} | {a['accuracy']:.4%} | {b['accuracy']:.4%} | {a['top1_regret']:.6f} | {b['top1_regret']:.6f} |")
    lines += ["", "## 验证集逐题比较", "", "赢/输按三种子平均后的题目质量比较；种子×题目次数另列。置信区间保持同题的三种子一起重采样，task/BBH子任务分层。所有新增分析均为探索性。", "", "| Selector 对照 | 种子×题目 赢/输/平 | 题目 赢/输/平 | 同组合题数 | 差值百分点 | 95% CI 百分点 |", "|---|---:|---:|---:|---:|---:|"]
    for name,v in valreport["selector_minus_other"].items():
        a,b,c=v["seed_question_outcomes"],v["unique_questions_averaging_three_generations"],v["paired_cluster_interval"]
        lines.append(f"| {name} | {a['wins']}/{a['losses']}/{a['ties']} | {b['wins']}/{b['losses']}/{b['ties']} | {v['same_selected_team_questions']} | {c['task_macro_difference']*100:.4f} | [{c['question_cluster_ci'][0]*100:.4f}, {c['question_cluster_ci'][1]*100:.4f}] |")
    lines += ["", "## 结构与解释限制", "", "原模型是共享二阶组合结构的岭回归，不是20个独立组合预测头。17维题目特征与22维组合基底形成374个名义参数；组合基底在固定K=3时秩为15，训练设计秩为255，不能表示20维组合评分空间中的全部方向。JSON给出了观察矩阵的描述性投影残差。", "", "PCA16没有显式任务或子任务one-hot。训练BBH各子任务只有11–12题，验证每子任务只有3–4题。现有结果不能证明PCA压缩或二阶结构是主要原因，也不能用样本内得分替代未见题泛化。", "", "IndependentTop3是在看过原验证结果后补充的探索性对照。任何置信区间含0都不能据此认定方向无效。下一步若改特征或选择器，需要另立开发方案；本诊断没有改原冻结文件，也没有进入新测试。", "", "详细按任务/27个BBH子任务、每个种子、赢输支持数、MSE及全部SHA见同目录 JSON。"]
    (OUT/"DIAGNOSTIC_README_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    put(OUT/"DELIVERY_v1.json",{"schema":"selector_diagnostic_delivery_v1","report":ref(target),
        "rows":ref(OUT/"EXISTING_MATRIX_SELECTED_OUTCOMES_DIAGNOSTIC_v1.npz"),"readme":ref(OUT/"DIAGNOSTIC_README_zh.md"),
        "protected_inputs_unchanged":True,"new_llm_calls":0,"new_answer_scoring":0,"new_model_fits":0})
    print(json.dumps({"report":ref(target),"train":{k:v["accuracy"] for k,v in trainreport["methods"].items()},
                      "validation":{k:v["accuracy"] for k,v in valreport["methods"].items()},
                      "representation":valreport["representation_and_error"],"runtime":report["runtime"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
