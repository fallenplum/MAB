"""Recompute the published CES capability-profile aggregate arithmetic."""
from pathlib import Path
import hashlib
import json
import numpy as np

ASSET=Path(r'E:\语言模型组合路由')
WORK=ASSET/'research/ces_traceable_baseline'
SOURCE=ASSET/'third_party/ces_source_audit/upstream/artifacts/mmlu_meta_10d.json'


def main():
    raw=SOURCE.read_bytes()
    profile=json.loads(raw)
    rows=[]
    maximum=0.
    for name,p in profile['models'].items():
        numerator=np.asarray(p['cap_correct'],float)
        denominator=np.asarray(p['cap_total'],float)
        expected=numerator/denominator
        observed=np.asarray(p['cap_acc_10d'],float)
        overall=float(numerator.sum()/denominator.sum())
        cap_error=float(np.max(abs(expected-observed)))
        overall_error=abs(overall-float(p['overall_acc']))
        maximum=max(maximum,cap_error,overall_error)
        assert (numerator>=0).all() and (numerator<=denominator).all()
        rows.append(dict(name=name,openrouter_model=p['openrouter_model'],correct=float(numerator.sum()),
                         total=int(denominator.sum()),reported_overall=float(p['overall_acc']),recomputed_overall=overall,
                         cap_max_error=cap_error,overall_error=overall_error,
                         reported_cap_accuracy=observed.tolist(),recomputed_cap_accuracy=expected.tolist(),
                         correct_fields_are_integer_counts=bool(np.all(numerator==np.floor(numerator))),
                         mechanically_recomputed_ratio_is_valid_accuracy=bool(np.all(numerator==np.floor(numerator))),
                         cap_correct=numerator.tolist(),cap_total=denominator.astype(int).tolist()))
    totals=set(r['total'] for r in rows)
    result=dict(status='REAGGREGATED_WITH_DISCREPANCIES' if maximum>=1e-14 else 'PASS',scope='Numerical re-aggregation of published MMLU capability counts; not a rerun of original model answers or CES benchmark routing scores',
        source_url='https://github.com/JinkunXu/CES/blob/9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7/artifacts/mmlu_meta_10d.json',
        source_sha256=hashlib.sha256(raw).hexdigest(),source_bytes=len(raw),license='MIT at current pinned master',
        split=profile['split'],n_per_subject=profile['n_per_subject'],seed=profile['seed'],cap_names=profile['cap_names'],
        rows=rows,models=len(rows),aggregate_model_question_records=sum(r['total'] for r in rows),
        intended_model_question_count_from_denominators=sum(r['total'] for r in rows),
        underlying_question_prediction_records_public=False,
        unique_MMLU_questions_claim='570 intended shared questions from57 subjects x10; raw IDs absent so identity overlap cannot independently be verified',
        max_arithmetic_error=maximum,consistent_models=sum(r['cap_max_error']<1e-14 and r['overall_error']<1e-14 and r['correct_fields_are_integer_counts'] for r in rows),
        invalid_model_interpretation='Noninteger correct-count fields invalidate their interpretation as empirical counts. The mechanical ratio is only an internal consistency check, not an estimated true accuracy.',
        physical_model_API_calls=0,
        absent_evidence=['per-question model answers','per-question grades','question ID mapping','HF dataset revision','actual Token/usage logs'],
        reconstruction_budget=dict(published_profile_lower_bound_generation_calls=8550,
            README_n20_generation_calls_if57_subjects=17100,published_unique_gold_question_upper_assumption=570,
            published_max_completion_tokens_per_call=16,published_completion_token_cap=8550*16,
            total_token_cap=None,total_token_note='Prompt lengths and source usage records unavailable, cannot give complete actual Token cost'),
        original_CES_score_recomputed=False)
    out=WORK/'public/CES_CAPABILITY_AGGREGATE_RECHECK.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],models=len(rows),records=result['aggregate_model_question_records'],max_error=maximum,
        mismatches=[r for r in rows if r['cap_max_error']>=1e-14 or r['overall_error']>=1e-14]),ensure_ascii=False))


if __name__=='__main__':
    main()
