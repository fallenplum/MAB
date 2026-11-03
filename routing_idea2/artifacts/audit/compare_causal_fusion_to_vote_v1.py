"""Compare completed strong fusion against the existing no-inference majority control."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,sys,time
import numpy as np
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'src'))
from budget_router.scoring import score_answer
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]
def main():
 A=R/'artifacts/audit/e1_failure_diagnosis_20261103_v2/causal_diagnostic_600_v1';out=A/'report_summary.json';assert not out.exists()
 s=read(A/'summary.json');ids=[r['question_id'] for r in read(R/'data/derived/split_manifest.json')['records'] if r['split']=='e1_diagnostic_evaluation']
 qs={q['question_id']:q for f in ['available_records.jsonl','gsm8k_questions.jsonl'] for q in rows(R/'data/derived'/f) if q['question_id'] in set(ids)}
 samples=defaultdict(list)
 for r in rows(R/'results/raw/e1/strong_samples.jsonl'):samples[r['question_id']].append(r)
 ys=[]
 for qid in ids:
  q=qs[qid];rr=sorted(samples[qid],key=lambda r:r['sample_index']);assert [r['sample_index'] for r in rr]==[0,1,2]
  parsed=[score_answer(q['task'],r['result']['text'],q['reference'],subtask=q.get('subtask')) for r in rr]
  preds=[v.prediction for v in parsed];counts=Counter(x for x in preds if x is not None);mx=max(counts.values(),default=0)
  vote=next((x for x in preds if x is not None and counts[x]==mx),None);ys.append(float(vote==parsed[0].reference))
 assert sum(ys)==165
 private=read(R/'artifacts/private/server_migration_20261102/isolated_e1_causal_returned_v1/private_per_question_scores.json')
 new={r['question_id']:r['score'] for r in private['qwen3_long_fuse']};d=np.array([new[q]-ys[i] for i,q in enumerate(ids)])
 tasks=np.array([qs[q]['task'] for q in ids]);rng=np.random.default_rng(20261103);boot=np.zeros(5000)
 for t in ['gsm8k','bbh']:
  v=d[tasks==t];boot+=v[rng.integers(0,100,size=(5000,100))].mean(axis=1)/2
 s['contrasts']['qwen3_fuse_minus_existing_majority_vote']={'difference':float(d.mean()),'paired_task_stratified_95_interval':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'per_task':{t:float(d[tasks==t].mean()) for t in ['gsm8k','bbh']}}
 s['existing_majority_vote']={'correct':165,'n':200,'accuracy':.825,'per_task':{'gsm8k':.93,'bbh':.72},'logical_total_tokens_per_question':931.695,'new_model_calls':0,'tie_rule':'lowest original sample_index among equally frequent parsed answers; no gold used in vote'}
 s['report_interpretation']={'observed_output_cap_contribution':'Original 193 non-cap outputs exactly reproduced; all seven cap-limited texts are prefixes of longer generations; four extra BBH answers scored correct.','strong_model_vs_weak_model':'Same message strings at max2048 yield 83.0% versus 78.5%; exploratory development evidence.','learned_heterogeneous_routing_benefit_established':False,'candidate_benefit_above_same_model_alone_established_by_current_paired_interval':False,'strong_fusion_improvement_over_vote_established_by_current_paired_interval':False,'original_e1_results_modified':False}
 s['summary_before_vote_comparison_sha256']=hashlib.sha256((A/'summary.json').read_bytes()).hexdigest()
 out.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'comparison':s['contrasts']['qwen3_fuse_minus_existing_majority_vote'],'output':str(out),'new_model_calls':0},ensure_ascii=False))
if __name__=='__main__':main()
