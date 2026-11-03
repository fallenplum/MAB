"""Disjoint four-question runtime qualification jobs for the pinned Mistral sensitivity."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from transformers import AutoTokenizer
from budget_router.preparation import select_records,immutable_jsonl
from budget_router.execution import make_job
from budget_router.inference import ModelSpec
from budget_router.stage_validation import read_json


def main():
 gate=read_json(ROOT/"results/e1/e1_diagnostics.json")
 if gate.get("status")!="PASS":raise ValueError("Second runtime qualification follows complete E1 PASS")
 _,records,questions,pool=select_records(ROOT,"qualification")
 spec=ModelSpec(**read_json(ROOT/"configs/aggregator_mistral_v03.json"));spec.validate()
 tokenizer=AutoTokenizer.from_pretrained(spec.model_path,local_files_only=True,trust_remote_code=False)
 jobs=[]
 for record in records:
  if record["task"]!="bbh":continue
  q=questions[record["question_id"]];answers={};provenance={}
  for entry in pool[:3]:
   old=q["member_outputs"][entry["historical_cache_alias"]]
   answers[entry["alias"]]=old["raw_output"]
   provenance[entry["alias"]]={"source_file":old["source_file"],"historical_checkpoint_revision":"UNKNOWN_NOT_CURRENT_MEMBER_REVISION", "usage":"runtime_qualification_only_not_formal_research"}
  jobs.append(make_job(question_id=q["question_id"],task=q["task"],subtask=q.get("subtask"),question=q["question"],
   answers=answers,seed=42,model_spec=spec,split="qualification",tokenizer=tokenizer,member_provenance=provenance))
 if len(jobs)!=4:raise ValueError("Exact disjoint four BBH runtime qualification questions required")
 result=immutable_jsonl(ROOT/"data/jobs/qualification/aggregator_mistral_v03.jsonl",jobs)
 print(json.dumps({"status":"SECOND_AGGREGATOR_RUNTIME_JOBS_NOT_RESEARCH","artifact":result}))
 return result


if __name__=="__main__":main()
