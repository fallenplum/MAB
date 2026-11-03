"""Released sealed-test jobs, actual executions and independent matrix assembly.

No original E0/E1 generator guard is weakened. Only this separately hash-bound
release path can consume sealed question text. Policies never enter this module;
the immutable full decision bundle must exist before any final model call.
"""
from pathlib import Path
from itertools import combinations
import hashlib,json,sys,time,uuid,os,math
from contextlib import contextmanager
import numpy as np
from .inference import ModelSpec,configuration_fingerprint,make_backend,validate_local_model_source
from .runtime_diagnostics import make_backend_with_diagnostics
from .member_execution import make_member_job,MemberStore,validate_member_job
from .execution import (make_job,ExecutionStore,validate_job,SYSTEM_PROMPT,
 AGGREGATION_PROMPT_VERSION,TOKEN_POLICY_VERSION)
from .preparation import load_inputs,read_spec,immutable_jsonl
from .stage_validation import (artifact_reference,verified_reference,read_json,immutable_json,
 canonical_digest,sha_file,read_jsonl)
from .frozen_evaluation import validate_final_release,_read_frozen_decisions
from .scoring import score_answer,SCORER_VERSION

ROOT=Path(__file__).resolve().parents[2]
SOURCES=("src/budget_router/final_generation.py","scripts/prepare_final_jobs.py",
 "scripts/run_final_generation.py","scripts/prepare_final_matrices.py","src/budget_router/runtime_memory.py","src/budget_router/runtime_diagnostics.py")


def aggregator_descriptor(spec):
 return {"model_id":spec.model_id,"revision":spec.revision,"fingerprint":configuration_fingerprint(spec),
  "prompt_sha256":canonical_digest({"system":SYSTEM_PROMPT,"prompt_version":AGGREGATION_PROMPT_VERSION,"token_policy":TOKEN_POLICY_VERSION})}


def _inside(path):
 path=Path(path).resolve()
 if not path.is_relative_to(ROOT.resolve()):raise ValueError("Final artifact escaped experiment root")
 return path


def context(release_path,decisions_directory):
 validated=validate_final_release(release_path)
 _,receipt=_read_frozen_decisions(decisions_directory,validated)
 release=validated["release"]
 for name in SOURCES:
  if name not in release["final_sources"] or verified_reference(release["final_sources"][name])!=(ROOT/name).resolve():
   raise ValueError("Final execution source is not bound to this release")
 manifest,records,questions,pool=load_inputs(ROOT)
 selected=[r for r in records if r["split"]=="sealed_test"]
 ids=[r["question_id"] for r in release["final_questions"]]
 if len(ids)!=200 or set(ids)!={r["question_id"] for r in selected}:
  raise ValueError("Final execution must preserve the whole sealed 200-question split")
 selected_map={r["question_id"]:r for r in selected}
 selected=[selected_map[qid] for qid in ids]
 return validated,receipt,selected,questions,pool


def _spec_for_condition(condition):
 for name in ("aggregator_v1.json","aggregator_mistral_v03.json"):
  path=ROOT/"configs"/name
  if path.is_file():
   spec=ModelSpec(**read_json(path));spec.validate()
   if aggregator_descriptor(spec)==condition["aggregator"]:return spec,path
 raise ValueError("No exact local aggregator configuration matches the released condition")


def _under(path,subtree):
 path=Path(path).resolve();base=(ROOT/subtree).resolve()
 if path==base or not path.is_relative_to(base):raise ValueError("Final artifact escaped its dedicated subtree")
 return path


def _batch_paths(kind,seed,*,alias=None,condition_id=None):
 if isinstance(seed,bool) or not isinstance(seed,int) or seed not in (42,137,2024):raise ValueError("Undeclared final member generation seed")
 if kind=="member":
  if not isinstance(alias,str) or not alias or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_." for c in alias):raise ValueError("Invalid frozen model alias path")
  jobs=ROOT/f"data/jobs/final/generation{seed}/members_{alias}.jsonl"
  output=ROOT/f"results/raw/final/generation{seed}/members_{alias}.jsonl"
 elif kind=="aggregation":
  if not isinstance(condition_id,str) or not condition_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in condition_id):raise ValueError("Invalid final condition path")
  jobs=ROOT/f"data/jobs/final/{condition_id}/aggregations.jsonl";output=ROOT/f"results/raw/final/{condition_id}/aggregations.jsonl"
 else:raise ValueError("Unknown released final job kind")
 return jobs.resolve(),jobs.with_suffix(".manifest.json").resolve(),output.resolve()


def _qualification(path,spec,*,role,config,alias=None):
 from .protocol_lock import verify_qualification
 config=Path(config).resolve()
 if role=="member":
  name=alias.lower();expected=ROOT/f"artifacts/qualification/member_{name}.json"
  jobs=ROOT/f"data/jobs/qualification/members_{name}.jsonl";outputs=ROOT/f"results/raw/qualification/members_{name}.jsonl"
 elif config==(ROOT/"configs/aggregator_v1.json").resolve():
  expected=ROOT/"artifacts/qualification/aggregator.json";jobs=ROOT/"data/jobs/qualification/historical_runtime_qualification.jsonl";outputs=ROOT/"results/raw/qualification/aggregator.jsonl"
 elif config==(ROOT/"configs/aggregator_mistral_v03.json").resolve():
  expected=ROOT/"artifacts/qualification/aggregator_mistral_v03.json";jobs=ROOT/"data/jobs/qualification/aggregator_mistral_v03.jsonl";outputs=ROOT/"results/raw/qualification/aggregator_mistral_v03.jsonl"
 else:raise ValueError("Unreleased aggregator qualification role")
 if Path(path).resolve()!=expected.resolve():raise ValueError("Qualification receipt is outside its canonical role path")
 verify_qualification(ROOT,expected,spec,role=role,expected_jobs_path=jobs,expected_outputs_path=outputs,
  expected_config_path=config,expected_alias=alias if role=="member" else None)
 return artifact_reference(expected)


def _authority(release_path,decisions_directory,release):
 return {"release":artifact_reference(release_path),"decisions":artifact_reference(Path(decisions_directory)/"decision_receipt.json"),"value":release}


def _load_tokenizer(spec):
 from transformers import AutoTokenizer
 validate_local_model_source(spec)
 return AutoTokenizer.from_pretrained(spec.model_path,local_files_only=True,trust_remote_code=False)


def _validate_manifest_header(m,path,*,authority,pool):
 if (m.get("schema")!="released_final_jobs_v1" or m.get("split")!="sealed_test"
  or m.get("gold_excluded_from_model_inputs") is not True or m.get("research_complete") is not False):raise ValueError("Explicit released label-free final job manifest required")
 for key in ("release","decisions"):
  if m.get(key)!=authority[key]:raise ValueError("Final batch binds a different release or frozen decision bundle")
  verified_reference(m[key])
 expected_sources={name:artifact_reference(ROOT/name) for name in SOURCES}
 if m.get("sources")!=expected_sources:raise ValueError("Final execution source closure changed")
 config=verified_reference(m["model_config"]);spec=ModelSpec(**read_json(config));spec.validate()
 if configuration_fingerprint(spec)!=m.get("model_fingerprint"):raise ValueError("Final model fingerprint changed")
 kind=m.get("kind");seed=m.get("generation_seed");entry=None;condition=None
 if kind=="member":
  entry=next((e for e in pool if configuration_fingerprint(read_spec(ROOT,e))==configuration_fingerprint(spec)),None)
  if entry is None:raise ValueError("Final member is outside the frozen pool")
  canonical_config=(ROOT/entry["config"]).resolve()
  paths=_batch_paths(kind,seed,alias=entry["alias"].lower());count=200
  qualification=ROOT/f"artifacts/qualification/member_{entry['alias'].lower()}.json"
  if m.get("condition_id") is not None:raise ValueError("Member generation caches must be shared across aggregator conditions")
 elif kind=="aggregation":
  condition=next((c for c in authority["value"]["evaluation_conditions"] if c["condition_id"]==m.get("condition_id")),None)
  if condition is None or condition["generation_seed"]!=seed or condition["aggregator"]!=aggregator_descriptor(spec):raise ValueError("Unreleased final aggregation condition")
  _,canonical_config=_spec_for_condition(condition);canonical_config=canonical_config.resolve()
  paths=_batch_paths(kind,seed,condition_id=condition["condition_id"]);count=4000
  qualification=ROOT/("artifacts/qualification/aggregator.json" if canonical_config.name=="aggregator_v1.json" else "artifacts/qualification/aggregator_mistral_v03.json")
 else:raise ValueError("Unknown final job role")
 if seed not in {c["generation_seed"] for c in authority["value"]["evaluation_conditions"]}:raise ValueError("Final seed was not released")
 if config!=canonical_config or m["model_config"]!=artifact_reference(canonical_config):raise ValueError("Final batch config differs from its frozen role")
 if Path(path).resolve()!=paths[1] or verified_reference(m["jobs"])!=paths[0] or Path(m["output"]).resolve()!=paths[2]:raise ValueError("Final jobs/manifest/output must use exact dedicated canonical paths")
 _under(paths[0],"data/jobs/final");_under(paths[2],"results/raw/final")
 if m.get("job_count")!=count:raise ValueError("Final batch scale changed")
 qref=_qualification(verified_reference(m["qualification"]),spec,role="member" if kind=="member" else "aggregator",config=config,alias=None if entry is None else entry["alias"])
 if m["qualification"]!=qref:raise ValueError("Final qualification artifact changed")
 return spec,entry,condition


def _manifest(jobs,*,kind,spec,config,output,qualification,release_path,decisions_directory,generation_seed,condition_id=None):
 alias=jobs[1][0].get("model_alias") if kind=="member" else None
 canonical,manifest_path,canonical_output=_batch_paths(kind,generation_seed,alias=None if alias is None else alias.lower(),condition_id=condition_id)
 if Path(jobs[0]).resolve()!=canonical or Path(output).resolve()!=canonical_output:raise ValueError("Final preparation cannot write outside dedicated canonical paths")
 qref=_qualification(qualification,spec,role="member" if kind=="member" else "aggregator",config=config,alias=alias)
 ref=immutable_jsonl(canonical,jobs[1])
 value={"schema":"released_final_jobs_v1","kind":kind,"generation_seed":generation_seed,"condition_id":condition_id,
  "release":artifact_reference(release_path),"decisions":artifact_reference(Path(decisions_directory)/"decision_receipt.json"),
  "model_config":artifact_reference(config),"model_fingerprint":configuration_fingerprint(spec),"qualification":qref,
  "jobs":artifact_reference(ref["path"]),"job_count":ref["jobs"],"output":str(canonical_output),"split":"sealed_test",
  "gold_excluded_from_model_inputs":True,"sources":{name:artifact_reference(ROOT/name) for name in SOURCES},"research_complete":False}
 immutable_json(manifest_path,value);return artifact_reference(manifest_path)


def prepare_member_jobs(release_path,decisions_directory):
 v,receipt,records,questions,pool=context(release_path,decisions_directory)
 seeds=sorted({c["generation_seed"] for c in v["release"]["evaluation_conditions"]});refs=[]
 for seed in seeds:
  for entry in pool:
   alias=entry["alias"].lower();spec=read_spec(ROOT,entry);jobs=[]
   for record in records:
    q=questions[record["question_id"]]
    jobs.append(make_member_job(question_id=q["question_id"],task=q["task"],subtask=q.get("subtask"),
     question=q["question"],model_alias=entry["alias"],seed=seed,model_spec=spec,split="sealed_test"))
   refs.append(_manifest((ROOT/f"data/jobs/final/generation{seed}/members_{alias}.jsonl",jobs),kind="member",spec=spec,
    config=ROOT/entry["config"],output=ROOT/f"results/raw/final/generation{seed}/members_{alias}.jsonl",
    qualification=ROOT/f"artifacts/qualification/member_{alias}.json",release_path=release_path,
    decisions_directory=decisions_directory,generation_seed=seed))
 return refs


def _read_jobs(manifest):
 path=verified_reference(manifest["jobs"]);jobs=read_jsonl(path)
 if len(jobs)!=manifest["job_count"] or len({j["job_id"] for j in jobs})!=len(jobs):raise ValueError("Final job batch incomplete or duplicated")
 return jobs


def _evaluation_helpers():
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
 from scripts.evaluate_real_e1 import _read_stores,_validate_result,_validate_aggregate_sources,_sum_cost
 return _read_stores,_validate_result,_validate_aggregate_sources,_sum_cost


def _audit_store(path,jobs,records,spec,kind,*,call_cap):
 if not Path(path).exists():return {},{},{}
 read_stores,validate_result,_,_=_evaluation_helpers();missing=set()
 successes,costs,calls,_,ignored=read_stores([path],{r["question_id"]:r for r in records},missing)
 if missing or ignored:raise ValueError("Unknown, missing or unrelated actual final call cost")
 expected={j["job_id"]:j for j in jobs}
 fields=("question_id","task","subtask","split","seed","model_fingerprint","prompt_version")
 fields+=(("model_alias","sample_index","member_mode","sampling_control","source_question_sha256") if kind=="member" else
  ("members","source_hashes","candidate_token_policy","candidate_statistics","member_provenance"))
 for item in calls.values():
  row=item["record"];job=expected.get(row["job_id"])
  if job is None or any(row.get(key)!=job.get(key) for key in fields):raise ValueError("Final attempt identity differs from its exact released job")
  if row["status"]=="success":validate_result(row,spec)
 for job_id,cost in costs.items():
  if cost["calls"]>call_cap or cost["prompt_tokens"]>cost["calls"]*spec.max_input_tokens or cost["completion_tokens"]>cost["calls"]*spec.max_new_tokens:
   raise ValueError("Actual final calls/tokens exceed the locked per-cell caps")
 return successes,costs,calls


def _call_cap(release,kind):
 config=read_json(verified_reference(release["artifacts"]["config"]))
 value=config.get("budget",{}).get("member_call_cap" if kind=="member" else "aggregator_call_cap",1)
 if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError("Explicit positive final call cap required")
 return value


def _expected_members(records,questions,entry,seed,spec):
 return [make_member_job(question_id=r["question_id"],task=questions[r["question_id"]]["task"],
  subtask=questions[r["question_id"]].get("subtask"),question=questions[r["question_id"]]["question"],
  model_alias=entry["alias"],seed=seed,model_spec=spec,split="sealed_test") for r in records]


def _stored_members(records,pool,seed,questions,*,authority):
 members={r["question_id"]:{} for r in records};provenance={qid:{} for qid in members};stores=[]
 for entry in pool:
  _,path,output=_batch_paths("member",seed,alias=entry["alias"].lower());m=read_json(path)
  spec,observed,_=_validate_manifest_header(m,path,authority=authority,pool=pool)
  if observed!=entry:raise ValueError("Stored member role/order differs from frozen pool")
  jobs=_read_jobs(m)
  if jobs!=_expected_members(records,questions,entry,seed,spec):raise ValueError("Final member jobs differ from exact released questions/protocol")
  _audit_store(output,jobs,records,spec,"member",call_cap=_call_cap(authority["value"],"member"))
  store=MemberStore(output)
  if store.unreconciled or set(store.completed)!={j["job_id"] for j in jobs}:raise ValueError("Final member coverage incomplete or has unknown costs")
  for job in jobs:
   row=store._cached(job,spec)
   if row is None or row["question_id"] not in members:raise ValueError("Final member coverage incomplete")
   qid=row["question_id"];members[qid][entry["alias"]]=row
   provenance[qid][entry["alias"]]={"job_id":row["job_id"],"record_sha256":row["record_sha256"],
    "model_id":spec.model_id,"revision":spec.revision,"model_fingerprint":configuration_fingerprint(spec)}
  stores.append(output)
 if any(len(row)!=6 for row in members.values()):raise ValueError("All six actual final member outputs are required")
 return members,provenance,stores


def _expected_aggregates(records,questions,pool,spec,members,provenance,tokenizer):
 aliases=[entry["alias"] for entry in pool];jobs=[]
 for r in records:
  qid=r["question_id"];q=questions[qid]
  for team in combinations(aliases,3):
   jobs.append(make_job(question_id=qid,task=q["task"],subtask=q.get("subtask"),question=q["question"],
    answers={key:members[qid][key]["result"]["text"] for key in team},seed=42,model_spec=spec,
    split="sealed_test",tokenizer=tokenizer,member_provenance={key:provenance[qid][key] for key in team}))
 return jobs


def prepare_aggregate_jobs(release_path,decisions_directory,condition_id):
 v,receipt,records,questions,pool=context(release_path,decisions_directory)
 condition=next((c for c in v["release"]["evaluation_conditions"] if c["condition_id"]==condition_id),None)
 if condition is None:raise ValueError("Undeclared final condition")
 seed=condition["generation_seed"];authority=_authority(release_path,decisions_directory,v["release"])
 members,provenance,_=_stored_members(records,pool,seed,questions,authority=authority)
 spec,config=_spec_for_condition(condition);tokenizer=_load_tokenizer(spec)
 jobs=_expected_aggregates(records,questions,pool,spec,members,provenance,tokenizer)
 qualification=(ROOT/"artifacts/qualification/aggregator.json" if config.name=="aggregator_v1.json" else ROOT/"artifacts/qualification/aggregator_mistral_v03.json")
 return _manifest((ROOT/f"data/jobs/final/{condition_id}/aggregations.jsonl",jobs),kind="aggregation",spec=spec,config=config,
  output=ROOT/f"results/raw/final/{condition_id}/aggregations.jsonl",qualification=qualification,
  release_path=release_path,decisions_directory=decisions_directory,generation_seed=seed,condition_id=condition_id)


def _validated_batch(manifest_path,release_path,decisions_directory,*,validated,records,questions,pool):
 path=_under(manifest_path,"data/jobs/final");m=read_json(path)
 authority=_authority(release_path,decisions_directory,validated["release"])
 spec,entry,condition=_validate_manifest_header(m,path,authority=authority,pool=pool);jobs=_read_jobs(m)
 members=None;provenance=None;member_paths=[]
 if m["kind"]=="member":expected=_expected_members(records,questions,entry,m["generation_seed"],spec)
 else:
  members,provenance,member_paths=_stored_members(records,pool,m["generation_seed"],questions,authority=authority)
  expected=_expected_aggregates(records,questions,pool,spec,members,provenance,_load_tokenizer(spec))
 if jobs!=expected:raise ValueError("Final jobs differ from exact released questions, actual member sources, tokenizer or fixed seed")
 for job in jobs:
  (validate_member_job if m["kind"]=="member" else validate_job)(job,spec)
 cap=_call_cap(validated["release"],m["kind"])
 successes,costs,calls=_audit_store(m["output"],jobs,records,spec,m["kind"],call_cap=cap)
 return {"manifest":m,"manifest_path":path,"spec":spec,"jobs":jobs,"members":members,"provenance":provenance,
  "member_paths":member_paths,"successes":successes,"costs":costs,"calls":calls,"call_cap":cap}


def _gpu_admission():
 """Reuse the inspected local scheduling observations without mutating E0 code."""
 import importlib.util
 module_path=ROOT/"scripts/run_local_e1.py"
 spec=importlib.util.spec_from_file_location("_read_only_final_gpu_probe",module_path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 consecutive=0
 while consecutive<2:
  sample=module.gpu_snapshot()
  allowed=(sample["free_mib"]>=6144 and sample["utilization_percent"]<=20
   and not sample["other_python_gpu_pids"] and sample["host_available_bytes"]>=5*1024**3)
  consecutive=consecutive+1 if allowed else 0
  print(json.dumps({"event":"final_gpu_admission","consecutive_ready_samples":consecutive,"sample":sample}),flush=True)
  if consecutive<2:time.sleep(30)


@contextmanager
def _final_execution_lock(manifest_path,output):
 import psutil
 lock=ROOT/"logs/final_gpu_execution.lock";lock.parent.mkdir(parents=True,exist_ok=True)
 if lock.exists():
  old=read_json(lock);alive=False
  try:
   process=psutil.Process(old["pid"]);alive=abs(process.create_time()-old["create_time"])<.01 and process.is_running()
  except (psutil.NoSuchProcess,KeyError):pass
  if alive:raise RuntimeError("Another final GPU batch is active")
  old_manifest=Path(old.get("manifest",""))
  if not old_manifest.is_file():raise RuntimeError("Stale final GPU lock has no reconcilable source manifest")
  old_output=_under(read_json(old_manifest)["output"],"results/raw/final")
  if old_output.with_name(old_output.name+".lock").exists():raise RuntimeError("Interrupted model call lock needs cost reconciliation before recovery")
  audit=ROOT/"artifacts/audit/final_stale_lock_recovery";audit.mkdir(parents=True,exist_ok=True)
  immutable_json(audit/(uuid.uuid4().hex+".json"),{"schema":"dead_final_supervisor_lock_recovery_v1","old_lock":old,"source_manifest":artifact_reference(old_manifest),"no_execution_store_lock":True})
  lock.unlink()
 token=uuid.uuid4().hex;fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 try:
  os.write(fd,json.dumps({"pid":os.getpid(),"create_time":psutil.Process().create_time(),"manifest":str(manifest_path),"token":token}).encode());os.fsync(fd)
 finally:os.close(fd)
 try:yield
 finally:
  if lock.exists() and read_json(lock).get("token")==token:lock.unlink()


def _completed_batch_receipt(batch,*,runtime_memory=None):
 m=batch["manifest"];path=Path(m["output"]).with_suffix(".batch.json")
 expected={"schema":"final_generation_batch_receipt_v1","status":"COMPLETE_REAL_FINAL_BATCH",
  "manifest":artifact_reference(batch["manifest_path"]),"output":artifact_reference(m["output"]),
  "processed":len(batch["jobs"]),"actual_failures":sum(x["record"]["status"]=="failure" for x in batch["calls"].values()),
  "physical_attempts":len(batch["calls"]),"research_complete":False}
 if path.exists():
  old=read_json(path)
  if any(old.get(k)!=value for k,value in expected.items()):raise ValueError("Existing final batch receipt differs from actual jobs, outputs or attempt costs")
  return old
 value={**expected,"memory_housekeeping_this_worker":runtime_memory or {"new_model_calls":0},"receipt_created_at_unix":time.time(),"completion_basis":"All exact real outputs and known attempt costs verified; timestamp is receipt creation, not a new model execution"}
 immutable_json(path,value);return value


def run_released_batch(manifest_path):
 m=read_json(_under(manifest_path,"data/jobs/final"))
 release_path=verified_reference(m["release"]);decision_path=verified_reference(m["decisions"])
 v,receipt,records,questions,pool=context(release_path,decision_path.parent)
 batch=_validated_batch(manifest_path,release_path,decision_path.parent,validated=v,records=records,questions=questions,pool=pool)
 m=batch["manifest"];spec=batch["spec"];jobs=batch["jobs"]
 store=(MemberStore if m["kind"]=="member" else ExecutionStore)(m["output"])
 if store.unreconciled:raise ValueError("Unknown prior final call cost; never auto-retry")
 pending=[job for job in jobs if store._cached(job,spec) is None]
 if not pending:return _completed_batch_receipt(batch)
 if store.path.with_name(store.path.name+".lock").exists():raise RuntimeError("Interrupted/in-flight call lock requires cost reconciliation")
 for job in pending:
  if batch["costs"].get(job["job_id"],{}).get("calls",0)>=batch["call_cap"]:
   raise RuntimeError("Locked per-cell call cap exhausted; cannot silently retry a paid failure")
 with _final_execution_lock(manifest_path,m["output"]):
  _gpu_admission();backend=make_backend_with_diagnostics(spec,project_root=ROOT,factory=make_backend)
  from .runtime_memory import cleanup_after_committed_success,cleanup_after_backend_initialization,RuntimeMemorySummary
  memory_summary=RuntimeMemorySummary();runner_source_sha256=sha_file(ROOT/"scripts/run_final_generation.py")
  memory_summary.observe(cleanup_after_backend_initialization(backend,project_root=ROOT,output_path=m["output"],
   runner_name="run_final_generation.py",runner_source_sha256=runner_source_sha256))
  for i,job in enumerate(jobs,1):
   row,cached=store.run(backend,job)
   print(json.dumps({"event":"final_real_model_call","processed":i,"requested":len(jobs),"job_id":job["job_id"],"status":row["status"],"cache_hit":cached,"cost_status":row.get("cost_status")}),flush=True)
   if row["status"]!="success":raise RuntimeError("Actual final model failure preserved; stop for independent cost/protocol review")
   if not cached:
    memory_summary.observe(cleanup_after_committed_success(backend,row,cached=False,project_root=ROOT,output_path=m["output"],
     runner_name="run_final_generation.py",runner_source_sha256=runner_source_sha256))
  successes,costs,calls=_audit_store(m["output"],jobs,records,spec,m["kind"],call_cap=batch["call_cap"])
  if set(successes)!={j["job_id"] for j in jobs}:raise RuntimeError("Final real batch did not complete all released jobs")
  batch.update(successes=successes,costs=costs,calls=calls)
  return _completed_batch_receipt(batch,runtime_memory=memory_summary.as_dict())


def _append_grading_progress(path,row,previous):
 value={**row,"previous_record_sha256":previous};value["record_sha256"]=canonical_digest(value)
 with Path(path).open("a",encoding="utf-8",newline="\n") as stream:
  stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n");stream.flush();os.fsync(stream.fileno())
 return value["record_sha256"]


class _GradingAttempt:
 """Durable per-attempt CPU work; interrupted work is never replaced with zero."""
 def __init__(self,directory,condition_id,input_identity,scorer_source):
  self.directory=Path(directory)/".grading_attempts"/condition_id/uuid.uuid4().hex
  self.directory.mkdir(parents=True);self.attempt_id=self.directory.name;self.input_identity=input_identity
  self.counts={"member":0,"aggregate":0};self.failures=0;self.process_s=0.;self.wall_s=0.;self.references=set();self.previous=None
  self.started_wall=time.perf_counter();self.started_cpu=time.process_time()
  immutable_json(self.directory/"attempt.json",{"schema":"final_grading_attempt_v1","attempt_id":self.attempt_id,
   "condition_id":condition_id,"input_identity_sha256":input_identity,"scorer_source":scorer_source,
   "scorer_version":SCORER_VERSION,"started_at_unix":time.time(),"pid":os.getpid(),
   "planned_member_labels":1200,"planned_aggregate_labels":4000,"planned_unique_questions":200,
   "time_scope":"Measured actual scorer-call CPU and wall intervals; source validation and progress I/O excluded; attempt elapsed reported separately",
   "role":"private_evaluator_only_no_policy_feedback_grants"})
 def checkpoint(self,event,qid):
  self.previous=_append_grading_progress(self.directory/"progress.jsonl",{"event":event,"question_id":qid,
   "member_label_evaluations":self.counts["member"],"aggregate_label_evaluations":self.counts["aggregate"],
   "process_time_s":self.process_s,"wall_time_s":self.wall_s,"score_failures":self.failures},self.previous)
 def score(self,stem,record,question):
  self.counts[stem]+=1;self.references.add(record["question_id"]);cpu=time.process_time();wall=time.perf_counter()
  try:
   value=score_answer(question["task"],record["result"]["text"],question["reference"],subtask=question.get("subtask"),choices=question.get("choices")).score
   if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=1:raise ValueError("Invalid actual verifier score")
   return value
  except BaseException:
   self.failures+=1;raise
  finally:self.process_s+=time.process_time()-cpu;self.wall_s+=time.perf_counter()-wall
 def finish(self,status):
  if (self.directory/"completion.json").exists():
   old=read_json(self.directory/"completion.json")
   if old.get("attempt_id")!=self.attempt_id or old.get("input_identity_sha256")!=self.input_identity:raise ValueError("Grading attempt completion changed")
   return old
  value={"schema":"actual_final_grading_attempt_completion_v1","attempt_id":self.attempt_id,"status":status,
   "input_identity_sha256":self.input_identity,"attempt":artifact_reference(self.directory/"attempt.json"),
   "progress":artifact_reference(self.directory/"progress.jsonl") if (self.directory/"progress.jsonl").is_file() else None,
   "member_label_evaluations":self.counts["member"],"aggregate_label_evaluations":self.counts["aggregate"],
   "unique_reference_questions":len(self.references),"process_time_s":self.process_s,"wall_time_s":self.wall_s,
   "score_failures":self.failures,"attempt_elapsed_process_time_s":time.process_time()-self.started_cpu,
   "attempt_elapsed_wall_time_s":time.perf_counter()-self.started_wall,"policy_feedback_grants":0}
  immutable_json(self.directory/"completion.json",value);return value


def _grading_inventory(directory,condition_id,input_identity,selected_attempt,score_digest):
 attempts=[];totals={"member":0,"aggregate":0,"cpu":0.,"wall":0.};upper={"member":0,"aggregate":0};unknown=0
 for folder in sorted((Path(directory)/".grading_attempts"/condition_id).iterdir()):
  if not (folder/"attempt.json").is_file():continue
  header=read_json(folder/"attempt.json")
  if header.get("input_identity_sha256")!=input_identity or header.get("attempt_id")!=folder.name or header.get("scorer_version")!=SCORER_VERSION:
   raise ValueError("Prior grading work binds different actual source outputs/attempt identity")
  verified_reference(header["scorer_source"])
  entry={"attempt":artifact_reference(folder/"attempt.json")}
  if (folder/"completion.json").is_file():
   done=read_json(folder/"completion.json")
   if done.get("input_identity_sha256")!=input_identity or done.get("status") not in ("COMPLETED","INTERRUPTED_OR_FAILED"):
    raise ValueError("Grading attempt completion identity/status mismatch")
   if done.get("attempt_id")!=folder.name or verified_reference(done["attempt"])!=(folder/"attempt.json").resolve():raise ValueError("Grading completion reused another physical attempt")
   if done.get("progress") is not None and verified_reference(done["progress"])!=(folder/"progress.jsonl").resolve():raise ValueError("Grading completion references another attempt progress")
   for key in ("member_label_evaluations","aggregate_label_evaluations","unique_reference_questions","score_failures"):
    if isinstance(done.get(key),bool) or not isinstance(done.get(key),int) or done[key]<0:raise ValueError("Physical grading counts must be nonnegative integers")
   if done["status"]=="COMPLETED" and (done["member_label_evaluations"]!=1200 or done["aggregate_label_evaluations"]!=4000 or done["unique_reference_questions"]!=200 or done["score_failures"]!=0):
    raise ValueError("Completed grading pass has incomplete or failed score evaluations")
   entry["completion"]=artifact_reference(folder/"completion.json");entry["status"]=done["status"]
   for key,target in (("member_label_evaluations","member"),("aggregate_label_evaluations","aggregate"),("process_time_s","cpu"),("wall_time_s","wall")):
    value=done[key]
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError("Invalid measured grading work")
    totals[target]+=value
   upper["member"]+=done["member_label_evaluations"];upper["aggregate"]+=done["aggregate_label_evaluations"]
  else:
   unknown+=1;last={"member_label_evaluations":0,"aggregate_label_evaluations":0,"process_time_s":0.,"wall_time_s":0.};previous=None;partial=False
   progress=folder/"progress.jsonl"
   if progress.is_file():
    payload=progress.read_bytes();lines=payload.split(b"\n");partial=bool(lines[-1]);entry["progress"]=artifact_reference(progress)
    for line in lines[:-1]:
     if not line:continue
     row=json.loads(line);checksum=row.pop("record_sha256",None)
     if checksum!=canonical_digest(row) or row.get("previous_record_sha256")!=previous:raise ValueError("Corrupt interrupted grading progress chain")
     previous=checksum;last=row
   entry.update(status="ABANDONED_PROCESS_WORK_NOT_FULLY_OBSERVED",counts_are_lower_bounds=True,unrecorded_block_max_calls=26,
    process_time_s=None,wall_time_s=None,incomplete_progress_tail=partial)
   totals["member"]+=last["member_label_evaluations"];totals["aggregate"]+=last["aggregate_label_evaluations"]
   totals["cpu"]+=last["process_time_s"];totals["wall"]+=last["wall_time_s"]
   uncertain=last.get("event")!="QUESTION_COMPLETED" or partial
   upper["member"]+=last["member_label_evaluations"]+(6 if uncertain else 0)
   upper["aggregate"]+=last["aggregate_label_evaluations"]+(20 if uncertain else 0)
  attempts.append(entry)
 selected=read_json(Path(selected_attempt)/"completion.json")
 if selected["status"]!="COMPLETED" or selected["member_label_evaluations"]!=1200 or selected["aggregate_label_evaluations"]!=4000 or selected["unique_reference_questions"]!=200:
  raise ValueError("A complete actual 1200+4000 final grading pass is required")
 return {"schema":"actual_final_grading_work_v2","scorer_version":SCORER_VERSION,"input_identity_sha256":input_identity,
  "selected_complete_attempt":artifact_reference(Path(selected_attempt)/"completion.json"),"score_arrays_sha256":score_digest,
  "member_label_evaluations":1200,"aggregate_label_evaluations":4000,"unique_reference_questions":200,
  "process_time_s":None if unknown else totals["cpu"],"wall_time_s":None if unknown else totals["wall"],
  "selected_attempt_process_time_s":selected["process_time_s"],"selected_attempt_wall_time_s":selected["wall_time_s"],
  "physical_member_label_evaluations":None if unknown else totals["member"],
  "physical_aggregate_label_evaluations":None if unknown else totals["aggregate"],
  "physical_member_label_evaluations_bounds":[totals["member"],upper["member"]],
  "physical_aggregate_label_evaluations_bounds":[totals["aggregate"],upper["aggregate"]],
  "known_process_time_lower_bound_s":totals["cpu"],"known_wall_time_lower_bound_s":totals["wall"],
  "unfully_observed_attempts":unknown,"attempts":attempts,"policy_feedback_grants":0,"model_judge_calls":0,
  "role":"Actual private final scoring CPU work only; distinct from LLM generation, tokens and online policy feedback. Include all attempts; never impute abandoned work as zero."}


def _future_reference(staged,final):
 return {**artifact_reference(staged),"path":str(Path(final).resolve())}


def _score_digest(arrays):
 return canonical_digest({name:arrays[name].tolist() for name in ("member_scores","aggregate_scores")})


def _validate_matrix_arrays(arrays,records):
 expected={"question_ids","task_labels","member_scores","aggregate_scores"}
 for stem in ("member","aggregate"):
  expected.update(stem+"_"+field for field in ("prompt_tokens","completion_tokens","execution_counts","wall_time_s"))
 if set(arrays)!=expected:raise ValueError("Unexpected final matrix arrays; no references/answers may be serialized")
 n=len(records)
 if arrays["question_ids"].tolist()!=[r["question_id"] for r in records] or arrays["task_labels"].tolist()!=[r["task"] for r in records]:raise ValueError("Final matrix row identity/order changed")
 for stem,width in (("member",6),("aggregate",20)):
  scores=arrays[stem+"_scores"]
  if scores.shape!=(n,width) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):raise ValueError("Invalid real final score matrix")
  for field in ("prompt_tokens","completion_tokens","execution_counts"):
   value=arrays[stem+"_"+field]
   if value.shape!=(n,width) or value.dtype.kind not in "iu" or np.any(value<(1 if field=="execution_counts" else 0)):raise ValueError("Invalid actual final token/call matrix")
  wall=arrays[stem+"_wall_time_s"]
  if wall.shape!=(n,width) or not np.isfinite(wall).all() or (wall<0).any():raise ValueError("Invalid actual source execution wall times")


def _verify_existing_bundle(bundle,expected,accounting,records):
 manifest_path=bundle/"manifest.json"
 if not manifest_path.is_file():raise ValueError("Partial final bundle has no immutable commit manifest")
 manifest=read_json(manifest_path)
 if any(manifest.get(k)!=value for k,value in expected.items()):raise ValueError("Existing final bundle differs from actual release, jobs or raw outputs")
 for key,name in (("matrix","matrix.npz"),("actual_grading_work","grading.json"),("physical_call_accounting","physical_calls.json")):
  if Path(manifest[key]["path"]).resolve()!=(bundle/name).resolve():raise ValueError("Final bundle reference escaped immutable bundle")
  verified_reference(manifest[key])
 if manifest["matrix_sha256"]!=manifest["matrix"]["sha256"]:raise ValueError("Final matrix SHA differs from its manifest")
 for source in manifest["sources"]:verified_reference(source)
 if read_json(bundle/"physical_calls.json")!=accounting:raise ValueError("Final physical generation cost ledger changed")
 grading=read_json(bundle/"grading.json")
 if grading.get("input_identity_sha256")!=manifest["grading_input_identity_sha256"]:raise ValueError("Grading source identity differs")
 for entry in grading["attempts"]:
  for key in ("attempt","progress","completion"):
   if key in entry:verified_reference(entry[key])
 verified_reference(grading["selected_complete_attempt"])
 with np.load(bundle/"matrix.npz",allow_pickle=False) as data:
  if data["question_ids"].tolist()!=[r["question_id"] for r in records] or data["task_labels"].tolist()!=[r["task"] for r in records]:raise ValueError("Final matrix question order changed")
  arrays={key:data[key] for key in data.files}
 _validate_matrix_arrays(arrays,records)
 if _score_digest(arrays)!=grading.get("score_arrays_sha256"):raise ValueError("Final score arrays differ from completed actual grading")
 return artifact_reference(manifest_path)


@contextmanager
def _assembly_lock(directory,condition_id):
 import psutil
 path=Path(directory)/("."+condition_id+".assembly.lock")
 if path.exists():
  old=read_json(path)
  try:
   process=psutil.Process(old["pid"]);alive=process.is_running() and abs(process.create_time()-old["create_time"])<.01
  except (psutil.NoSuchProcess,KeyError):alive=False
  if alive:raise RuntimeError("Another final scoring assembly is active")
  path.unlink()  # Prior grading attempts remain durable and enter the new inventory.
 token=uuid.uuid4().hex;fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 try:os.write(fd,json.dumps({"pid":os.getpid(),"create_time":psutil.Process().create_time(),"token":token}).encode());os.fsync(fd)
 finally:os.close(fd)
 try:yield
 finally:
  if path.exists() and read_json(path).get("token")==token:path.unlink()


def assemble_final_matrix(release_path,decisions_directory,condition_id,output_dir):
 """Create-only atomic bundle; cached complete bundles are verified without regrading."""
 output=_under(output_dir,"results/raw/final");output.mkdir(parents=True,exist_ok=True)
 v,decision_receipt,records,questions,pool=context(release_path,decisions_directory)
 release=v["release"];condition=next((c for c in release["evaluation_conditions"] if c["condition_id"]==condition_id),None)
 if condition is None:raise ValueError("Undeclared final matrix condition")
 _batch_paths("aggregation",condition["generation_seed"],condition_id=condition_id)
 _,jobs_manifest_path,_=_batch_paths("aggregation",condition["generation_seed"],condition_id=condition_id)
 batch=_validated_batch(jobs_manifest_path,release_path,decisions_directory,validated=v,records=records,questions=questions,pool=pool)
 jobs=batch["jobs"];spec=batch["spec"];members=batch["members"];member_paths=batch["member_paths"]
 read_stores,validate_result,validate_sources,sum_cost=_evaluation_helpers()
 store=ExecutionStore(batch["manifest"]["output"])
 if store.unreconciled or set(store.completed)!={j["job_id"] for j in jobs}:raise ValueError("Final matrix needs all 4000 actual aggregate outputs")
 aliases=[entry["alias"] for entry in pool];teams=list(combinations(aliases,3));indices={tuple(sorted(team)):i for i,team in enumerate(teams)};by_cell={}
 for job in jobs:
  row=store._cached(job,spec)
  if row is None:raise ValueError("Incomplete final aggregator output")
  validate_result(row,spec);qid=row["question_id"];validate_sources(row,questions[qid],{key:members[qid][key] for key in row["members"]})
  cell=(qid,indices[tuple(row["members"])])
  if cell in by_cell:raise ValueError("Duplicate final question-combination")
  by_cell[cell]=row
 if set(by_cell)!={(r["question_id"],i) for r in records for i in range(20)}:raise ValueError("Final question/combination coverage differs")
 missing=set();aggregate_path=Path(batch["manifest"]["output"])
 successes,costs,calls,_,ignored=read_stores([*member_paths,aggregate_path],{r["question_id"]:r for r in records},missing)
 expected_ids={r["job_id"] for r in by_cell.values()}|{r["job_id"] for values in members.values() for r in values.values()}
 if missing or ignored or set(successes)!=expected_ids or set(costs)!=expected_ids:raise ValueError("Final source costs missing, unknown or out of protocol")
 import budget_router.scoring as scorer
 source_refs=[{"kind":"member_outputs",**artifact_reference(p)} for p in member_paths]+[{"kind":"aggregate_outputs",**artifact_reference(aggregate_path)},
  {"kind":"scorer_source",**artifact_reference(scorer.__file__)}]
 for path in [*(ROOT/f"data/jobs/final/generation{condition['generation_seed']}/members_{e['alias'].lower()}.manifest.json" for e in pool),jobs_manifest_path]:
  source_refs.append({"kind":"generation_job_manifest",**artifact_reference(path)})
 accounting={"schema":"physical_final_unique_call_accounting_v1","condition_id":condition_id,"unique_calls":len(calls),
  "totals":sum_cost(x["cost"] for x in calls.values()),
  "call_records":[{"attempt_id":attempt,"job_id":item["record"]["job_id"],"record_sha256":item["record"]["record_sha256"],
   "status":item["record"]["status"],"cost":item["cost"]} for attempt,item in sorted(calls.items())],
  "cross_condition_rule":"Deduplicate actual attempt_id with matching record_sha256, including shared generation42 members; conflicting SHA for one attempt is an error."}
 reference=v["policies"][0][2]._metadata["online_provenance"]
 input_identity=canonical_digest({"release":artifact_reference(release_path),"decisions":artifact_reference(Path(decisions_directory)/"decision_receipt.json"),"condition":condition,"sources":source_refs})
 expected={"schema":"sealed_real_matrix_v1","split":"sealed_test","software_test_only":False,"release_sha256":release["release_sha256"],
  "frozen_decisions_sha256":decision_receipt["decisions_sha256"],"condition_id":condition_id,"generation_seed":condition["generation_seed"],
  "all_combinations_executed":True,"real_frozen_llm_aggregation":True,"aggregate_scores_from_executed_aggregator":True,
  "failure_costs_included":True,"no_unknown_costs":True,"gold_excluded_from_model_inputs":True,
  "combinations":[list(team) for team in v["policies"][0][2].mapper.subsets],"model_pool":reference["model_pool"],
  "aggregator":condition["aggregator"],"scorer":reference["scorer"],"question_fingerprints":[r["canonical_fingerprint"] for r in release["final_questions"]],
  "sources":source_refs,"grading_input_identity_sha256":input_identity,"research_complete":False}
 for name in ("split_manifest","pca_artifact","pool_config"):expected[name]=release["artifacts"][name]
 bundle=output/condition_id
 with _assembly_lock(output,condition_id):
  if bundle.exists():return _verify_existing_bundle(bundle,expected,accounting,records)
  n=len(records);arrays={"question_ids":np.array([r["question_id"] for r in records]),"task_labels":np.array([r["task"] for r in records]),
   "member_scores":np.empty((n,6)),"aggregate_scores":np.empty((n,20))}
  for stem,width in (("member",6),("aggregate",20)):
   for field in ("prompt_tokens","completion_tokens","execution_counts"):arrays[stem+"_"+field]=np.zeros((n,width),dtype=np.int64)
   arrays[stem+"_wall_time_s"]=np.zeros((n,width))
  attempt=_GradingAttempt(output,condition_id,input_identity,artifact_reference(scorer.__file__))
  def put(stem,row,col,record):
   bill=costs[record["job_id"]]
   for field,source in (("prompt_tokens","prompt_tokens"),("completion_tokens","completion_tokens"),("execution_counts","calls"),("wall_time_s","wall_time_s")):
    arrays[stem+"_"+field][row,col]=bill[source]
   arrays[stem+"_scores"][row,col]=attempt.score(stem,record,questions[record["question_id"]])
  try:
   for i,r in enumerate(records):
    qid=r["question_id"];attempt.checkpoint("QUESTION_STARTED",qid)
    for j,alias in enumerate(aliases):put("member",i,j,members[qid][alias])
    for j in range(20):put("aggregate",i,j,by_cell[(qid,j)])
    attempt.checkpoint("QUESTION_COMPLETED",qid)
   attempt.finish("COMPLETED")
  except BaseException:
   attempt.finish("INTERRUPTED_OR_FAILED");raise
  _validate_matrix_arrays(arrays,records)
  grading=_grading_inventory(output,condition_id,input_identity,attempt.directory,_score_digest(arrays))
  staging=output/("."+condition_id+"."+uuid.uuid4().hex+".pending");staging.mkdir()
  matrix=staging/"matrix.npz";np.savez_compressed(matrix,**arrays)
  with matrix.open("r+b") as stream:os.fsync(stream.fileno())
  immutable_json(staging/"physical_calls.json",accounting);immutable_json(staging/"grading.json",grading)
  manifest={**expected,"matrix_sha256":sha_file(matrix),"matrix":_future_reference(matrix,bundle/"matrix.npz"),
   "actual_grading_work":_future_reference(staging/"grading.json",bundle/"grading.json"),
   "physical_call_accounting":_future_reference(staging/"physical_calls.json",bundle/"physical_calls.json")}
  immutable_json(staging/"manifest.json",manifest)
  if bundle.exists():raise FileExistsError("Another worker published the immutable final bundle")
  os.rename(staging,bundle)
  return _verify_existing_bundle(bundle,expected,accounting,records)
