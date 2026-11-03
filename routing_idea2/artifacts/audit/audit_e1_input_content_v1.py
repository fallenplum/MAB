"""Reconstruct E1 generation inputs from official questions and frozen raw candidates.
No model generation. No changes to frozen scores, configs, splits or sealed evaluation."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src')]
from budget_router.execution import aggregation_messages,head_tail_candidate,SYSTEM_PROMPT,ALONE_CANDIDATE_TEXT,_digest
from budget_router.member_execution import member_messages
from budget_router.scoring import score_answer,_candidate
from transformers import AutoTokenizer
def rows(p):
 with Path(p).open(encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=ROOT/'artifacts/audit/e1_failure_diagnosis_20261103_v2'
 assert not (out/'input_reconstruction.json').exists()
 rec={r['question_id']:r for r in json.loads((ROOT/'data/derived/split_manifest.json').read_text())['records'] if r['split'].startswith('e1_')}
 questions={q['question_id']:q for f in ['available_records.jsonl','gsm8k_questions.jsonl'] for q in rows(ROOT/'data/derived'/f) if q['question_id'] in rec}
 cfg=json.loads((ROOT/'configs/aggregator_v1.json').read_text())
 tk=AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True,trust_remote_code=False)
 class CachedTokenizer:
  def __init__(self,t):self.t=t;self.enc={};self.dec={}
  def encode(self,text,add_special_tokens=False):
   if text not in self.enc:self.enc[text]=self.t.encode(text,add_special_tokens=add_special_tokens)
   return self.enc[text]
  def decode(self,ids,skip_special_tokens=True):
   key=tuple(ids)
   if key not in self.dec:self.dec[key]=self.t.decode(ids,skip_special_tokens=skip_special_tokens)
   return self.dec[key]
 tokenizer=CachedTokenizer(tk)
 pool=json.loads((ROOT/'configs/pool_v1.json').read_text())['models']
 members={(r['question_id'],m['alias']):r for m in pool for r in rows(ROOT/'results/raw/e1'/('members_'+m['alias'].lower()+'.jsonl')) if r['status']=='success'}
 samples={(r['question_id'],r['model_alias'].split('::sample')[0]+'::sample'+str(r['sample_index'])):r for r in rows(ROOT/'results/raw/e1/strong_samples.jsonl') if r['status']=='success'}
 counts=Counter();losses=[];artifacts={}
 for kind in ['aggregations','strong_aggregations','aggregator_alone']:
  jp=ROOT/'data/jobs/e1'/(kind+'.jsonl');rp=ROOT/'results/raw/e1'/(kind+'.jsonl')
  jobs=rows(jp);outputs={r['job_id']:r for r in rows(rp) if r['status']=='success'}
  assert len(jobs)==len(outputs) and {j['job_id'] for j in jobs}==set(outputs)
  artifacts[str(jp.relative_to(ROOT))]=digest(jp);artifacts[str(rp.relative_to(ROOT))]=digest(rp)
  for j in jobs:
   q=questions[j['question_id']];qid=q['question_id'];counts[kind+'_jobs']+=1
   assert j['job_id']==_digest({k:v for k,v in j.items() if k!='job_id'})
   if kind=='aggregator_alone':
    expected=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':'Original problem:\n'+q['question']+'\n\nAnonymous candidate responses:\n'+ALONE_CANDIDATE_TEXT}]
    order=[]
   else:
    source=members if kind=='aggregations' else samples
    ans={a:source[qid,a]['result']['text'] for a in j['members']}
    expected,order=aggregation_messages(q['question'],ans,question_id=qid,seed=j['seed'],tokenizer=tokenizer)
    assert j['source_hashes']['member_answers_sha256']=={a:hashlib.sha256(t.encode()).hexdigest() for a,t in ans.items()}
    assert j['source_hashes']['question_sha256']==hashlib.sha256(q['question'].encode()).hexdigest()
    if rec[qid]['split']=='e1_diagnostic_evaluation':
     g=score_answer(q['task'],outputs[j['job_id']]['result']['text'],q['reference'],subtask=q.get('subtask'))
     for a,t in ans.items():
      render,stat=head_tail_candidate(t,tokenizer)
      assert stat==j['candidate_statistics'][a]
      c=score_answer(q['task'],t,q['reference'],subtask=q.get('subtask'))
      if c.score:
       counts[kind+'_correct_candidate_exposures']+=1
       retained=score_answer(q['task'],render,q['reference'],subtask=q.get('subtask')).score==1
       counts[kind+'_correct_candidate_final_retained']+=retained
       if not retained:losses.append({'kind':kind,'job_id':j['job_id'],'question_id':qid,'alias':a,'truncated':stat['truncated'],'aggregate_correct':g.score})
   assert j['messages']==expected and j['candidate_order']==order, (kind,j['job_id'])
   counts[kind+'_exact_reconstructed_messages']+=1
 for kind,raw in [('ordinary',members),('strong_samples',samples)]:
  for (qid,a),r in raw.items():
   assert r['source_question_sha256']==hashlib.sha256(questions[qid]['question'].encode()).hexdigest()
   counts[kind+'_raw_output_question_hash_bound']+=1
 # Source presence and type validity can be checked without reusing historical correctness for selection.
 train=rows(ROOT/'data/raw/gsm8k_official/train.jsonl');test=rows(ROOT/'data/raw/gsm8k_official/test.jsonl')
 norm=lambda s:' '.join(__import__('unicodedata').normalize('NFKC',s).split())
 trainq={norm(q['question']) for q in train};testq={norm(q['question']) for q in test}
 report={'schema':'e1_input_content_reconstruction_v1','created_at_unix':time.time(),'new_model_calls':0,'counts':dict(counts),'input_leakage_audit_scope':'Stored member/aggregation job messages reconstructed exactly from official question and real generated candidate text, with pinned formatting and tokenizer compression. Does not establish absence of pretraining/posttraining contamination.','raw_gsm_train_test_exact_question_overlap':len(trainq&testq),'correct_candidate_final_not_retained':losses,'bound_file_sha256':artifacts,'audit_script_sha256':digest(__file__)}
 (out/'input_reconstruction.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k not in ['bound_file_sha256','correct_candidate_final_not_retained']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
