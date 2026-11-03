"""Posthoc exact option-text diagnostic. Original scores are never changed."""
from pathlib import Path
from collections import Counter
import json,re,runpy,sys,time,unicodedata
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'src'))
from budget_router.scoring import score_answer,_candidate,CHOICE_TASKS
def rows(p):
 with Path(p).open(encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]
def main():
 out=R/'artifacts/audit/e1_failure_diagnosis_20261103_v2/option_text_diagnostic.json';assert not out.exists()
 manifest=json.loads((R/'data/derived/split_manifest.json').read_text());ev={r['question_id'] for r in manifest['records'] if r['split']=='e1_diagnostic_evaluation'}
 qs={q['question_id']:q for f in ['available_records.jsonl','gsm8k_questions.jsonl'] for q in rows(R/'data/derived'/f) if q['question_id'] in ev}
 m=json.loads((out.parent/'audit_metrics.json').read_text());fixed=set(m['fixed_team'])
 norm=lambda x:' '.join(unicodedata.normalize('NFKC',x).strip().removesuffix('.').split()).casefold()
 results={}
 for kind in ['fixed','strong_aggregations','aggregator_alone']:
  counts=Counter();examples=[]
  rp=R/'results/raw/e1'/('aggregations.jsonl' if kind=='fixed' else kind+'.jsonl')
  for r in rows(rp):
   if r['status']!='success' or r['question_id'] not in ev or (kind=='fixed' and set(r['members'])!=fixed):continue
   q=qs[r['question_id']];g=score_answer(q['task'],r['result']['text'],q['reference'],subtask=q.get('subtask'));counts['n']+=1;counts['original_correct']+=g.score
   if not g.parsed:
    counts['unparsed']+=1
    if q.get('subtask') in CHOICE_TASKS:
     opts=dict(re.findall(r'(?m)^\(([A-Z])\)\s*(.+?)\s*$',q['question']));literal,_=_candidate(r['result']['text']);hits=[k for k,v in opts.items() if norm(v)==norm(literal)]
     if len(hits)==1:
      pred=hits[0];counts['unique_exact_option_text_mapping']+=1;correct=pred==g.reference;counts['exact_mapped_correct']+=correct
      if len(examples)<12:examples.append({'question_id':q['question_id'],'subtask':q['subtask'],'literal_final':literal,'label_from_options_only':pred,'reference_label':g.reference,'mapped_correct':correct})
  results[kind]={'counts':dict(counts),'examples':examples}
 report={'schema':'posthoc_option_text_diagnostic_v1','created_at_unix':time.time(),'new_model_calls':0,'official_scores_modified':False,'mapping_rule':'Unique exact normalized final text to original question option text; option mapping uses no gold. Diagnostic only, not a replacement for frozen letter-format score.','results':results}
 out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v['counts'] for k,v in results.items()},ensure_ascii=False))
if __name__=='__main__':main()
