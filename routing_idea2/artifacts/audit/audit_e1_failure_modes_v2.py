"""Read-only E1 failure audit. No inference, protocol changes, or final-test scoring."""
from pathlib import Path
from collections import Counter,defaultdict
from itertools import combinations
from decimal import Decimal
from fractions import Fraction
import csv,hashlib,json,re,sys,time,unicodedata
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src')]
from budget_router.scoring import score_answer
from budget_router.member_execution import member_messages
OUT=ROOT/'artifacts/audit/e1_failure_diagnosis_20261103_v2'
def rows(p):
 with Path(p).open(encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(t):return ' '.join(unicodedata.normalize('NFKC',t).split())
def grade(row,q):return score_answer(q['task'],row['result']['text'],q['reference'],subtask=q.get('subtask'),choices=q.get('choices'))
def cost(row):return row['result']['prompt_tokens']+row['result']['completion_tokens']
def strict_number(text):
 text=unicodedata.normalize('NFKC',text).replace('−','-').strip()
 lines=re.findall(r'(?im)^\s*Final\s+answer\s*:\s*([^\r\n]*)',text)
 if lines:v=lines[-1]
 elif '####' in text:v=text.rsplit('####',1)[1].strip().split('\n')[0]
 else:
  b=re.findall(r'\\boxed\{([^{}]+)\}',text);v=b[-1] if b else text
 v=v.strip().strip(chr(96)+'*$ ').removesuffix('.').strip().removesuffix('%').strip().replace(',','')
 box=re.fullmatch(r'\\boxed\{([^{}]+)\}',v)
 if box:v=box[1]
 if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*/\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+))?',v):return None
 try:
  a,*b=v.split('/');return Fraction(Decimal(a.strip()))/(Fraction(Decimal(b[0].strip())) if b else 1)
 except (ValueError,ZeroDivisionError):return None
def main():
 assert not OUT.exists() or not any(OUT.iterdir()), 'Refusing existing audit outputs'
 OUT.mkdir(exist_ok=True)
 manifest=json.loads((ROOT/'data/derived/split_manifest.json').read_text());e1=[r for r in manifest['records'] if r['split'].startswith('e1_')];rec={r['question_id']:r for r in e1}
 questions={}
 for name in ['available_records.jsonl','gsm8k_questions.jsonl']:
  for q in rows(ROOT/'data/derived'/name):
   if q['question_id'] in rec:questions[q['question_id']]=q
 pool=json.loads((ROOT/'configs/pool_v1.json').read_text())['models'];aliases=[m['alias'] for m in pool];combos=list(combinations(aliases,3))
 official={sp:rows(ROOT/'data/raw/gsm8k_official'/(sp+'.jsonl')) for sp in ['train','test']}
 source_check=Counter()
 for q in questions.values():
  if q['task']=='gsm8k':
   raw=official[q['source_split']][q['source_index']]
   source_check['gsm_exact_question_match']+=q['question']==raw['question'];source_check['gsm_exact_reference_match']+=q['reference']==raw['answer']
   source_check['gsm_official_final_answer_numeric_match']+=strict_number(q['reference'])==strict_number(raw['answer'].rsplit('####',1)[1])
  else:
   data=json.loads((ROOT/q['subtask_source_file']).read_text());match=[x for x in data['examples'] if canon(x['input'])==canon(q['question'])]
   source_check['bbh_official_exact_question_and_target_match']+=len(match)==1 and str(match[0]['target']).strip()==q['reference'].strip()
   source_check['bbh_official_question_and_semantic_target_match']+=len(match)==1 and score_answer('bbh',str(match[0]['target']),q['reference'],subtask=q['subtask']).score==1
 assert source_check['gsm_exact_question_match']==source_check['gsm_official_final_answer_numeric_match']==150 and source_check['bbh_official_question_and_semantic_target_match']==150
 members={}
 for a in aliases:
  for r in rows(ROOT/'results/raw/e1'/('members_'+a.lower()+'.jsonl')):
   if r['status']=='success':members[(r['question_id'],a)]=r
 aggregates={r['job_id']:r for r in rows(ROOT/'results/raw/e1/aggregations.jsonl') if r['status']=='success'}
 aggidx={(r['question_id'],tuple(sorted(r['members']))):r for r in aggregates.values()}
 jobs={j['job_id']:j for j in rows(ROOT/'data/jobs/e1/aggregations.jsonl')}
 prompts=Counter()
 for f in (ROOT/'data/jobs/e1').glob('members_*.jsonl'):
  for j in rows(f):
   q=questions[j['question_id']];prompts['ordinary_member_jobs']+=1;prompts['ordinary_exact_question_only_messages']+=j['messages']==member_messages(q['question'])
 strong=defaultdict(list)
 for r in rows(ROOT/'results/raw/e1/strong_samples.jsonl'):
  if r['status']=='success':strong[r['question_id']].append(r)
 for v in strong.values():v.sort(key=lambda r:r['sample_index'])
 for j in rows(ROOT/'data/jobs/e1/strong_samples.jsonl'):
  prompts['resample_jobs']+=1;prompts['resample_exact_question_only_messages']+=j['messages']==member_messages(questions[j['question_id']]['question'])
 assert prompts['ordinary_exact_question_only_messages']==prompts['ordinary_member_jobs']==1800 and prompts['resample_exact_question_only_messages']==prompts['resample_jobs']==600
 alone={r['question_id']:r for r in rows(ROOT/'results/raw/e1/aggregator_alone.jsonl') if r['status']=='success'}
 fused={r['question_id']:r for r in rows(ROOT/'results/raw/e1/strong_aggregations.jsonl') if r['status']=='success'}
 selected=json.loads((ROOT/'results/e1/e1_diagnostics.json').read_text())['selected_on_calibration'];fixed=tuple(sorted(combos[selected['best_fixed_indices'][0]]));topk=tuple(sorted(combos[selected['independent_topk_indices'][0]]));single=aliases[selected['strongest_single_indices'][0]]
 ev=[r['question_id'] for r in e1 if r['split']=='e1_diagnostic_evaluation'];cal=[r['question_id'] for r in e1 if r['split']=='e1_selection_calibration'];task=np.array([questions[q]['task'] for q in ev])
 def stats(rr):
  return {'n':len(rr),'correct':sum(grade(r,questions[r['question_id']]).score for r in rr),'parse_failures':sum(not grade(r,questions[r['question_id']]).parsed for r in rr),'output_cap_hits':sum(r['result'].get('completion_reached_cap') is True for r in rr),'total_tokens':sum(cost(r) for r in rr),'mean_tokens':sum(cost(r) for r in rr)/len(rr)}
 def transition(team,qids):
  c=Counter();examples=[]
  for qid in qids:
   q=questions[qid];rs=[members[qid,a] for a in team];r=aggidx[qid,tuple(sorted(team))];ss=[grade(x,q) for x in rs];g=grade(r,q);n=sum(x.score for x in ss);c['questions']+=1;c['candidate_coverage']+=n>0;c['final_correct']+=g.score;c['correct_candidate_to_wrong']+=n>0 and not g.score;c['no_correct_candidate_to_correct']+=n==0 and bool(g.score);c['all_three_correct']+=n==3;c['all_three_correct_to_wrong']+=n==3 and not g.score;c['parse_failures']+=not g.parsed;c['cap_hits']+=r['result'].get('completion_reached_cap') is True
   if n>0 and not g.score:
    c['lost_without_output_cap']+=not r['result'].get('completion_reached_cap');c['lost_with_output_cap']+=r['result'].get('completion_reached_cap') is True
    j=jobs[r['job_id']];c['lost_with_any_candidate_truncation']+=any(x['truncated'] for x in j['candidate_statistics'].values())
    if len(examples)<20:examples.append({'question_id':qid,'task':q['task'],'subtask':q.get('subtask'),'reference':g.reference,'member_predictions':{a:s.prediction for a,s in zip(team,ss)},'aggregator_prediction':g.prediction,'aggregator_cap':r['result'].get('completion_reached_cap'),'aggregator_text_tail':r['result']['text'][-700:]})
  return dict(c),examples
 table=[]
 for combo in combos:
  t=tuple(sorted(combo));cr=[aggidx[q,t] for q in cal];er=[aggidx[q,t] for q in ev];table.append({'members':list(t),'calibration_accuracy':sum(grade(r,questions[r['question_id']]).score for r in cr)/100,'evaluation_accuracy':sum(grade(r,questions[r['question_id']]).score for r in er)/200})
 fixed_metrics,fixed_examples=transition(fixed,ev)
 strong_counts=Counter();votes=[];single_y=[];fused_y=[];strict_mismatch=[];sample_rates=[[],[],[]]
 for qid in ev:
  q=questions[qid];samps=strong[qid];ss=[grade(r,q) for r in samps];g=grade(fused[qid],q);sg=grade(members[qid,single],q);vals=[s.prediction for s in ss];vc=Counter(x for x in vals if x is not None);best=max(vc.values()) if vc else 0;vp=next((x for x in vals if x is not None and vc[x]==best),None);vy=float(vp==g.reference)
  strong_counts['questions']+=1;strong_counts['coverage']+=any(s.score for s in ss);strong_counts['vote_correct']+=vy;strong_counts['fused_correct']+=g.score;strong_counts['vote_correct_fuser_wrong']+=bool(vy) and not g.score;strong_counts['vote_wrong_fuser_correct']+=not vy and bool(g.score);strong_counts['unanimous_correct']+=all(s.score for s in ss);strong_counts['unanimous_correct_fuser_wrong']+=all(s.score for s in ss) and not g.score;strong_counts['fuser_cap_hits']+=fused[qid]['result'].get('completion_reached_cap') is True;strong_counts['fuser_parse_failures']+=not g.parsed;strong_counts['vote_ties']+=sum(n==best for n in vc.values())>1
  votes.append(vy);single_y.append(sg.score);fused_y.append(g.score)
  for i,s in enumerate(ss):sample_rates[i].append(s.score)
  if q['task']=='gsm8k':
   raw=official[q['source_split']][q['source_index']];g0=strict_number(raw['answer'].rsplit('####',1)[1]);pred=strict_number(members[qid,single]['result']['text']);strict_y=float(pred is not None and pred==g0)
   if strict_y!=sg.score:strict_mismatch.append({'question_id':qid,'official_score':sg.score,'strict_score':strict_y,'output':members[qid,single]['result']['text']})
 def summary(y):
  a=np.array(y);return {'overall':float(a.mean()),'by_task':{t:float(a[task==t].mean()) for t in ['gsm8k','bbh']}}
 def paired(y,x):
  d=np.array(y)-np.array(x);rng=np.random.default_rng(20261103);bs=[]
  for _ in range(5000):bs.append(np.mean([d[rng.choice(np.flatnonzero(task==t),100,replace=True)].mean() for t in ['gsm8k','bbh']]))
  return {'difference':float(d.mean()),'paired_question_stratified_95_interval':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],'conditioning':'fixed observed generations; not independent new model replications'}
 allnormal=[r for r in aggregates.values() if r['question_id'] in set(ev)]
 out={'schema':'read_only_posthoc_e1_failure_audit_v2','created_at_unix':time.time(),'new_model_calls':0,'official_protocol_or_scores_modified':False,'source_matches':dict(source_check),'source_split_counts':dict(Counter(questions[q]['task']+'/'+questions[q]['source_split'] for q in ev)),'prompt_audit':dict(prompts),'strongest_model':single,'fixed_team':fixed,'independent_topk_team':topk,'single_by_task':{t:stats([members[q,single] for q in ev if questions[q]['task']==t]) for t in ['gsm8k','bbh']},'gsm_independent_strict_score_mismatches':strict_mismatch,'fixed_transitions':fixed_metrics,'fixed_transitions_by_task':{t:transition(fixed,[q for q in ev if questions[q]['task']==t])[0] for t in ['gsm8k','bbh']},'strong_three_sample':dict(strong_counts),'strong_sample_individual_scores':[summary(x) for x in sample_rates],'three_sample_vote':summary(votes),'three_sample_fuser':summary(fused_y),'single':summary(single_y),'three_sample_fuser_minus_single':paired(fused_y,single_y),'three_sample_vote_minus_single':paired(votes,single_y),'three_sample_vote_minus_fuser':paired(votes,fused_y),'all_eval_aggregate_outputs':stats(allnormal),'aggregator_alone_by_task':{t:stats([alone[q] for q in ev if questions[q]['task']==t]) for t in ['gsm8k','bbh']},'all_20_fixed_teams':table,'posthoc_eval_best_team':max(table,key=lambda x:x['evaluation_accuracy']),'fixed_error_examples':fixed_examples,'vote_rule':'Majority of parsed answers; tie breaks by smallest original sample_index, never gold; posthoc diagnostic only.'}
 (OUT/'audit_metrics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
 with (OUT/'all_fixed_teams.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=['members','calibration_accuracy','evaluation_accuracy']);w.writeheader();w.writerows(table)
 print(json.dumps({k:v for k,v in out.items() if k not in ['fixed_error_examples','all_20_fixed_teams']},ensure_ascii=False))
if __name__=='__main__':main()
