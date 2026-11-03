"""Independently reconcile and score all 600 isolated posthoc E1 causal calls.
No new inference; no changes to original E0/E1/final-test artifacts."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,sqlite3,tarfile,time
import numpy as np
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'src'))
from budget_router.scoring import score_answer
P=R/'artifacts/private/server_migration_20261102';B=P/'isolated_e1_causal_bundle_v1'
EXPECTED={'qwen25_long':'47aec8b2e07e80e46b1c71dfbd6ad084618773e89c2babf6d044027091a9810a','qwen3_long_fuse':'d4dcbc6e3ff5cad32d2e9544ea8528e2ace88b066648f05af5ca11a2b4a606e8','qwen3_long_alone':'dddc3abc24fa5e3690ad68722c8a0d4b4e3b63ed1c5127a84965784e6d0d8d87'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def main():
 assert sha(B/'DIAGNOSTIC_PLAN.json')=='6a94b1ec663997a959655478e6fc160094c48d808c5888ff519a2825007409f3'
 plan=read(B/'DIAGNOSTIC_PLAN.json')
 for rel,h in plan['source_artifacts_sha256'].items():assert sha(R/rel)==h
 ev=[x for x in read(R/'data/derived/split_manifest.json')['records'] if x['split']=='e1_diagnostic_evaluation'];ids=[x['question_id'] for x in ev]
 assert len(ids)==200 and Counter(x['task'] for x in ev)=={'gsm8k':100,'bbh':100}
 qs={q['question_id']:q for f in ['available_records.jsonl','gsm8k_questions.jsonl'] for q in rows(R/'data/derived'/f) if q['question_id'] in set(ids)}
 tasks=np.array([qs[q]['task'] for q in ids])
 source_jobs={kind:{j['question_id']:j for j in rows(R/'data/jobs/e1'/(kind+'.jsonl'))} for kind in ['strong_aggregations','aggregator_alone']}
 out=R/'artifacts/audit/e1_failure_diagnosis_20261103_v2/causal_diagnostic_600_v1';out.mkdir(exist_ok=False)
 extracted=P/'isolated_e1_causal_returned_v1';extracted.mkdir(exist_ok=False)
 values={};outputs={};summary={};archiveproof={};private={}
 originals={r['question_id']:r for r in rows(R/'results/raw/e1/strong_aggregations.jsonl')}
 def grade(qid,text):return score_answer(qs[qid]['task'],text,qs[qid]['reference'],subtask=qs[qid].get('subtask'))
 values['original_qwen25_256']=np.array([grade(q,originals[q]['result']['text']).score for q in ids])
 for c in plan['conditions']:
  name=c['condition'];archive=P/('isolated_e1_'+name+'.outputs_v1.tar.gz');assert sha(archive)==EXPECTED[name]
  dest=extracted/name;dest.mkdir()
  names={'EXPORT_MANIFEST.json','outputs.jsonl','status.json','jobs.jsonl','journal_snapshot.sqlite3'}
  with tarfile.open(archive,'r:gz') as tf:
   members=tf.getmembers();assert len(members)==5 and {m.name for m in members}==names and all(m.isfile() and m.size<100_000_000 for m in members)
   for m in members:
    p=dest/m.name;assert p.resolve().is_relative_to(dest.resolve());p.write_bytes(tf.extractfile(m).read())
  export=read(dest/'EXPORT_MANIFEST.json');st=read(dest/'status.json')
  assert export['condition']==name and export['successful_rows']==200 and export['unknown_cost_rows']==0 and export['source_process_ended'] is True and export['plan_sha256']==sha(B/'DIAGNOSTIC_PLAN.json')
  for f,v in export['files'].items():assert f in names and sha(dest/f)==v['sha256'] and (dest/f).stat().st_size==v['bytes']
  assert sha(dest/'jobs.jsonl')==c['jobs_sha256']==sha(B/c['jobs_relative'])
  assert st['context']['actual_model_spec']==c['model_config'] and st['context']['logical_fingerprint']==c['model_fingerprint']
  assert st['runtime_versions']=={'torch':'2.4.1+cu121','transformers':'4.51.3','bitsandbytes':'0.48.2'}
  js=rows(dest/'jobs.jsonl');jm={j['job_id']:j for j in js};rs=rows(dest/'outputs.jsonl');assert len(jm)==len(rs)==200 and {r['question_id'] for r in rs}==set(ids)
  conn=sqlite3.connect('file:'+str(dest/'journal_snapshot.sqlite3')+'?mode=ro',uri=True)
  context=conn.execute("SELECT value FROM metadata WHERE key='context_sha256'").fetchone()[0];assert context==digest(st['context'])==export['context_sha256']
  calls={r[0]:r[1:] for r in conn.execute('SELECT attempt_id,job_id,request_sha,status,response_json,response_sha,local_record_sha FROM calls')};assert len(calls)==200
  usage={'prompt_tokens':0,'completion_tokens':0};wall=0.;seen=set();byid={}
  for r in rs:
   assert r['record_sha256']==digest({k:v for k,v in r.items() if k!='record_sha256'}) and r['status']=='success' and r['cost_status']=='recorded_actual_usage'
   j=jm[r['job_id']];source=source_jobs['aggregator_alone' if name=='qwen3_long_alone' else 'strong_aggregations'][r['question_id']]
   assert j['messages']==source['messages'] and j['source_job_id']==source['job_id'] and j['source_job_sha256']==digest(source)
   assert j['job_id']==digest({k:v for k,v in j.items() if k!='job_id'})
   attempt=r['attempt_id'];assert attempt not in seen;seen.add(attempt)
   payload={'attempt_id':attempt,'job_id':j['job_id'],'model_fingerprint':c['model_fingerprint'],'split':j['split'],'seed':j['seed'],'messages':j['messages'],'job_sha256':digest(j)}
   body=r['response'];call=calls[attempt];assert call[0]==j['job_id'] and call[1]==digest(payload)==r['request_sha256']==body['request_sha256'] and call[2]=='SUCCESS' and json.loads(call[3])==body and call[4]==digest(body) and call[5]==r['record_sha256']
   assert context==r['context_sha256']==body['context_sha256']
   v=body['result'];assert v['model_id']==c['model_config']['model_id'] and v['model_revision']==c['model_config']['revision'] and v['model_identity_verified'] is True and v['seed']==42 and v['backend']=='transformers'
   assert isinstance(v['prompt_tokens'],int) and 0<v['prompt_tokens']<=6144 and isinstance(v['completion_tokens'],int) and 0<v['completion_tokens']<=2048 and v['wall_time_s']>0
   for k in usage:usage[k]+=v[k]
   wall+=v['wall_time_s'];byid[r['question_id']]=v
  conn.close();assert usage==export['actual_usage']==st['actual_usage']
  ys=np.array([grade(q,byid[q]['text']).score for q in ids]);values[name]=ys;outputs[name]=byid
  summary[name]={'n':200,'correct':int(ys.sum()),'accuracy':float(ys.mean()),'per_task':{t:{'n':100,'correct':int(ys[tasks==t].sum()),'accuracy':float(ys[tasks==t].mean())} for t in ['gsm8k','bbh']},'parse_failures':sum(not grade(q,byid[q]['text']).parsed for q in ids),'output_cap_hits':sum(byid[q]['completion_reached_cap'] is True for q in ids),'actual_usage':usage,'model_call_wall_seconds_sum':wall,'mean_call_tokens':sum(usage.values())/200,'model_id':c['model_config']['model_id'],'model_revision':c['model_config']['revision']}
  archiveproof[name]={'archive_sha256':sha(archive),'raw_output_sha256':sha(dest/'outputs.jsonl'),'jobs_sha256':c['jobs_sha256'],'journal_snapshot_sha256':sha(dest/'journal_snapshot.sqlite3'),'matched_requests_responses_and_commits':200}
  private[name]=[{'question_id':q,'score':float(ys[i]),'prediction':grade(q,byid[q]['text']).prediction} for i,q in enumerate(ids)]
 def contrast(a,b):
  d=values[a]-values[b];rng=np.random.default_rng(20261103);boot=np.zeros(5000)
  for t in ['gsm8k','bbh']:
   v=d[tasks==t];boot+=v[rng.integers(0,100,size=(5000,100))].mean(axis=1)/2
  return {'difference':float(d.mean()),'paired_task_stratified_95_interval':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'per_task':{t:float(d[tasks==t].mean()) for t in ['gsm8k','bbh']}}
 contrasts={'qwen25_2048_minus_original_256':contrast('qwen25_long','original_qwen25_256'),'qwen3_fuse_minus_qwen25_long':contrast('qwen3_long_fuse','qwen25_long'),'qwen3_fuse_minus_qwen3_alone':contrast('qwen3_long_fuse','qwen3_long_alone')}
 old_noncap=[q for q in ids if not originals[q]['result']['completion_reached_cap']];old_cap=[q for q in ids if originals[q]['result']['completion_reached_cap']]
 prefix={'original_noncap_n':len(old_noncap),'noncap_exact_output_matches':sum(originals[q]['result']['text']==outputs['qwen25_long'][q]['text'] for q in old_noncap),'original_cap_n':len(old_cap),'cap_original_text_prefix_of_new_text':sum(outputs['qwen25_long'][q]['text'].startswith(originals[q]['result']['text']) for q in old_cap)}
 member_tokens=sum(r['result']['prompt_tokens']+r['result']['completion_tokens'] for r in rows(R/'results/raw/e1/strong_samples.jsonl'))/200
 for name in summary:summary[name]['logical_total_tokens_per_question']=summary[name]['mean_call_tokens']+(0 if name=='qwen3_long_alone' else member_tokens)
 totals={k:sum(v['actual_usage'][k] for v in summary.values()) for k in ['prompt_tokens','completion_tokens']}
 result={'schema':'complete_isolated_E1_causal_600_diagnostic_v1','status':'ALL_600_REQUEST_RESPONSE_COMMIT_PAIRS_RECONCILED_AND_SCORED','created_at_unix':time.time(),'new_model_calls_by_this_auditor':0,'additional_actual_diagnostic_model_calls':600,'original_E1_model_calls':8800,'combined_E1_and_posthoc_calls_excluding_E0':9400,'unknown_cost_rows':0,'plan_sha256':sha(B/'DIAGNOSTIC_PLAN.json'),'conditions':summary,'contrasts':contrasts,'same_weak_model_greedy_prefix_check':prefix,'actual_incremental_usage':totals,'actual_incremental_total_tokens':sum(totals.values()),'archive_proof':archiveproof,'scorer_version':'safe_exact_v1_1','analysis_status':'EXPLORATORY_POSTHOC_DEVELOPMENT_DIAGNOSIS_NOT_NEW_TEST_OR_NEW_GENERATION_REPLICATION','ci_conditioning':'5000 task-stratified paired question bootstraps conditional on existing outputs; no multiplicity-adjusted confirmatory claim','scope_limits':['Same 200 E1 development questions, including official GSM8K train. Does not settle contamination.','Fused candidates are three new samples of Qwen3, not heterogeneous three-model combinations; cannot establish heterogeneous routing benefit.','Model-specific chat templates differ even when message strings match.','Logical token cost is a proxy, not USD or normalized GPU time.','Original E0/E1 results, sealed evaluation and online 12000 aggregation hold unchanged.']}
 (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 (extracted/'private_per_question_scores.json').write_text(json.dumps(private,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ['archive_proof','scope_limits']},ensure_ascii=False))
if __name__=='__main__':main()
