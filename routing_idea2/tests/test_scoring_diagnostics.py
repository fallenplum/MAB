"""Synthetic unit fixtures only; these tests are not research results."""
from itertools import combinations
import numpy as np
import pytest

from budget_router.scoring import BBH_TASKS, score_answer
from budget_router.diagnostics import evaluate_e1


@pytest.mark.parametrize("prediction,gold,expected", [
    ("Work: 10+32=42.\n#### 42", "#### 42", 1),
    ("Final answer: $1,200.", "1200", 1),
    (r"Therefore \boxed{\frac{1}{2}}", "0.5", 1),
    ("<answer>-0.00</answer>", "0", 1),
    ("The answer is 6.02e2", "602", 1),
    ("We compute 20 and then 42.", "42", 0),
    ("Final answer: 41", "42", 0),
    ("####", "42", 0),
    ("Final answer: 2+2", "4", 0),
    ("Final answer: __import__('os').system('echo unsafe')", "1", 0),
    ("Final answer: 1/0", "1", 0),
])
def test_gsm8k_safe_final_extraction(prediction,gold,expected):
    result=score_answer("gsm8k",prediction,gold)
    assert result.score == expected


def test_bad_reference_stops_instead_of_becoming_zero():
    with pytest.raises(ValueError,match="reference"):
        score_answer("gsm8k","42","missing gold")


@pytest.mark.parametrize("subtask,prediction,gold", [
    ("date_understanding","Reasoning first. The answer is (C).","(C)"),
    ("logical_deduction_three_objects","Final answer: A","(A)"),
    ("boolean_expressions","The answer is TRUE.","True"),
    ("formal_fallacies","Final answer: invalid","invalid"),
    ("navigate","No.","No"),
    ("object_counting","The answer is 12.","12"),
    ("multistep_arithmetic_two",r"\boxed{-9}","-9"),
    ("dyck_languages","Final answer: ] > )"," ]  >   )"),
    ("word_sorting","Final answer: apple banana peach.","apple banana peach"),
])
def test_bbh_task_specific_exact_scoring(subtask,prediction,gold):
    assert score_answer("bbh",prediction,gold,subtask=subtask).score == 1


def test_bbh_does_not_take_an_earlier_cot_letter():
    result=score_answer("bbh","Option C is tempting but I cannot decide.","(C)",subtask="date_understanding")
    assert result.score == 0
    assert not result.parsed
    assert len(BBH_TASKS) == 27
    with pytest.raises(ValueError,match="requires"):
        score_answer("bbh","A","A")
    with pytest.raises(ValueError,match="Unknown"):
        score_answer("bbh","yes","yes",subtask="invented_task")


def test_choice_text_requires_pinned_choice_mapping():
    assert score_answer("bbh","Final answer: green","(B)",subtask="reasoning_about_colored_objects",
                        choices={"A":"blue","B":"green"}).score == 1


def fixture_data():
    n,nc=200,100
    combos=list(combinations(range(6),3))
    agg=np.zeros((n,20)); agg[::2,0]=1; agg[1::2,1]=1
    cal=np.zeros((nc,20)); cal[:,0]=1
    member=np.zeros((n,6)); member[::4,:]=1
    cal_member=np.zeros((nc,6)); cal_member[:,:3]=1
    evidence={k:"a"*64 for k in ["e0_lock_sha256","cache_revision","aggregator_revision",
        "raw_aggregate_artifact_sha256","raw_baseline_artifact_sha256","split_manifest_sha256"]}
    evidence.update({"scorer_version":"safe_exact_v1","cost_unit":"synthetic_fixture_unit", "real_frozen_llm_aggregation":True,
        "all_combinations_executed":True,"budget_matched_controls":True,"e1_isolated_from_online_and_pca":True,
        "unit_fixture_not_scientific_evidence":True})
    return dict(aggregate_scores=agg,member_scores=member,combinations=combos,
        calibration_aggregate_scores=cal,calibration_member_scores=cal_member,
        question_ids=[f"eval_{i}" for i in range(n)],calibration_question_ids=[f"selection_{i}" for i in range(nc)],
        task_labels=["gsm8k"]*100+["bbh"]*100,calibration_task_labels=["gsm8k"]*50+["bbh"]*50,
        experiment_seeds=[42],aggregator_alone_scores=(np.arange(n)%5==0).astype(float),
        strong_model_resample_scores=(np.arange(n)%5<2).astype(float),execution_costs=np.ones((n,20)),
        diagnostic_total_execution_cost=4200.0,
        baseline_costs={"aggregator_alone":np.ones(n),"strong_model_resample":np.ones(n),"strongest_single_calibration":np.ones(n)},
        evidence=evidence,bootstrap_replicates=100)


def test_e1_uses_calibration_only_and_labels_oracle():
    result=evaluate_e1(**fixture_data())
    assert result["status"] == "PASS"
    assert result["overall_research_status"].startswith("PARTIAL")
    assert result["selected_on_calibration"]["best_fixed_indices"] == [0]
    assert result["metrics"]["best_fixed_calibration"]["mean"] == .5
    assert result["metrics"]["diagnostic_oracle_upper_reference"]["mean"] == 1
    assert result["paired_differences"]["fixed_minus_independent_topk"]["low"] == 0
    assert result["paired_differences"]["fixed_minus_independent_topk"]["high"] == 0
    assert result["experimental_seed_count"] == 1
    assert result["robustness_not_yet_checked"] is True
    assert "never deployable" in result["gate"]["limitation"]
    assert result["costs"]["full_actual_E1_diagnostic_cost"] == 4200
    assert result["costs"]["sum_standalone_combo_service_costs_not_diagnostic_actual"] == 4000


def test_missing_real_controls_is_partial_even_with_good_oracle():
    data=fixture_data();data["aggregator_alone_scores"]=None
    data["strong_model_resample_scores"]=None;data["evidence"]={}
    result=evaluate_e1(**data)
    assert result["status"] == "PARTIAL"
    assert "strong_model_resample_real_outputs_and_scores" in result["missing_requirements"]
    assert result["gate"]["passed_statistical_opportunity_check"] is True


def test_missing_actual_diagnostic_cost_is_not_replaced_by_combo_sum():
    data=fixture_data();data.pop("diagnostic_total_execution_cost")
    result=evaluate_e1(**data)
    assert result["status"] == "PARTIAL"
    assert "full_actual_E1_diagnostic_cost_including_selection_and_controls" in result["missing_requirements"]


def test_weak_true_aggregate_quality_stops_e1():
    data=fixture_data()
    data["aggregate_scores"][:]=0
    data["member_scores"][:]=0
    data["aggregator_alone_scores"][:]=0
    data["strong_model_resample_scores"][:]=0
    assert evaluate_e1(**data)["status"] == "STOP_E1"


def test_split_leakage_and_fake_scores_fail_closed():
    data=fixture_data();data["question_ids"][0]=data["calibration_question_ids"][0]
    with pytest.raises(ValueError,match="leakage"):
        evaluate_e1(**data)
    data=fixture_data();data["aggregate_scores"][1,2]=np.nan
    with pytest.raises(ValueError,match="never impute"):
        evaluate_e1(**data)
    data=fixture_data();data["combinations"].pop()
    with pytest.raises(ValueError,match="complete"):
        evaluate_e1(**data)


def test_duplicate_question_content_with_different_ids_is_rejected():
    data=fixture_data();data["question_fingerprints"]=[str(i) for i in range(200)]
    data["calibration_question_fingerprints"]=[str(i+100) for i in range(100)]
    with pytest.raises(ValueError,match="normalized"):
        evaluate_e1(**data)
