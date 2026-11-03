"""Independent E4 receipts; full legal-info adaptation, never native scores."""
from pathlib import Path
from dataclasses import asdict, dataclass
import json, os, uuid
from .stage_validation import (read_json,atomic_json,immutable_json,artifact_reference,
 verified_reference,canonical_digest,input_identity,verify_input_identity,settings_from_config,
 episode_specs,episode_name,read_jsonl,validate_e3_gate)

METHODS=("ces","c2mabv_awc","c2mabv_suc","c2mabv_aic")
ARTIFACTS=("summary.json","traces.jsonl","warmup_anonymous.json","policy_checkpoint.json","cost_curve.csv")
SOURCES=("src/budget_router/neighbor_policies.py","src/budget_router/neighbor_simulation.py",
 "src/budget_router/neighbor_validation.py","scripts/run_neighbors.py","scripts/evaluate_e4.py",
 "protocol/neighbor_implementation_manifest.json",
 "third_party/ces_source_audit/SOURCE_MANIFEST.json",
 "third_party/ces_source_audit/upstream/LICENSE",
 "third_party/ces_source_audit/upstream/src/bandits/linucb_combinatorial.py",
 "third_party/ces_source_audit/upstream/src/embeddings/projectors.py")


def locked_grid(config):
 return {"stage":"e4","strategies":list(METHODS),
  "seeds":config["statistics"]["algorithm_stream_seeds"],
  "rates":config["feedback"]["final_score_budget_per_100_questions"],
  "delays":config["feedback"]["delays"]}


def make_identity(root,inputs,e3_gate_path):
 validate_e3_gate(e3_gate_path,inputs)
 config=read_json(inputs["assets"]["config"]["path"])
 value={"schema":"neighbor_grid_identity_v1","inputs":inputs,
  "grid":locked_grid(config),"settings":json.loads(json.dumps(asdict(settings_from_config(config)))),
  "e3_gate":artifact_reference(e3_gate_path),
  "neighbor_sources":{name:artifact_reference(Path(root)/name) for name in SOURCES},
  "native_system":"NOT_RUN","one_pinned_generation_cache":True}
 value["identity_sha256"]=canonical_digest(value)
 return value


def verify_identity(identity,*,parent=True):
 core={k:v for k,v in identity.items() if k!="identity_sha256"}
 if identity.get("schema")!="neighbor_grid_identity_v1" or identity.get("identity_sha256")!=canonical_digest(core):
  raise ValueError("E4 identity changed")
 verify_input_identity(identity["inputs"])
 if set(identity.get("neighbor_sources",{}))!=set(SOURCES):raise ValueError("E4 source closure incomplete")
 for ref in identity["neighbor_sources"].values():verified_reference(ref)
 gate=verified_reference(identity["e3_gate"])
 if parent:validate_e3_gate(gate,identity["inputs"])
 config=read_json(identity["inputs"]["assets"]["config"]["path"])
 if identity["grid"]!=locked_grid(config) or identity["settings"]!=json.loads(json.dumps(asdict(settings_from_config(config)))):
  raise ValueError("E4 requires the full default locked grid and paid common warmup")
 return identity


def _cost_add(total,value):
 for key in ("executions","labels","tokens"):
  x=value[key]
  if isinstance(x,bool) or not isinstance(x,int) or x<0:raise ValueError("Invalid paid resource count")
  total[key]+=x


def validate_result(result,spec,identity):
 s=result["summary"];c=result["policy_checkpoint"];tr=result["traces"]
 if s.get("status")!="COMPLETED_RESTRICTED_FEEDBACK_SIMULATION_ONLY" or s.get("native_system")!="NOT_RUN":
  raise ValueError("E4 receipt rejects software-only or native-result claims")
 if any(s.get(k)!=v or c.get(k)!=v for k,v in spec.items()):raise ValueError("E4 episode parameters changed")
 if s.get("settings")!=identity["settings"] or c.get("settings")!=identity["settings"]:raise ValueError("E4 settings changed")
 n=read_json(identity["inputs"]["assets"]["config"]["path"])["split"]["online_total"]
 if n!=600 or s.get("n_questions")!=n or s.get("all_questions_served") is not True or len(tr)!=n:
  raise ValueError("E4 must serve all original 600 questions")
 if c.get("schema")!="neighbor_restricted_feedback_policy_checkpoint_v1" or c.get("software_test_only") is not False:
  raise ValueError("Missing real neighbor policy checkpoint")
 if c.get("next_tick")!=n or c.get("frozen_as_of_tick")!=n-1 or c.get("pending_feedback_flushed") is not False:
  raise ValueError("E4 checkpoint time/feedback boundary changed")
 if len({v["question_id"] for v in tr})!=n or [v["step"] for v in tr]!=list(range(n)):
  raise ValueError("E4 trace incomplete, duplicated or reordered")
 total={"executions":0,"labels":0,"tokens":0};_cost_add(total,s["warmup_cost"])
 for row in tr:
  if row.get("service_frozen_before_audit") is not True or row.get("prediction_snapshot_precedes_current_execution_and_feedback") is not True:
   raise ValueError("E4 service timing boundary missing")
  if row.get("alternate_index") is not None or row.get("rounding_rejection_resampling") is not False:
   raise ValueError("E4 base comparison must not add unpriced audits or rejection-rounding")
  _cost_add(total,row["service_cost"]);_cost_add(total,row["purchase_cost"])
  if total!=row["cumulative_cost"]:raise ValueError("E4 cumulative actual resources mismatch")
 budget=s["budget"]
 if total!=budget["spent"] or c.get("budget")!=budget or any(total[k]>budget["limits"][k] for k in total):
  raise ValueError("E4 paid budget does not reconcile")
 if budget["future_service"]!={"executions":0,"labels":0,"tokens":0} or budget["pending"]:
  raise ValueError("E4 financial reservations not reconciled")
 if s.get("warmup_mode")!="random_combo" or s.get("feedback_mode")!="verifiable":raise ValueError("E4 shared-information regime changed")
 for key in ("controller_process_time_s","controller_wall_time_s","offline_evaluation_process_time_s","offline_evaluation_wall_time_s"):
  v=s.get(key)
  if isinstance(v,bool) or not isinstance(v,(int,float)) or v<0:raise ValueError("E4 measured CPU/wall clock missing")
 for broker in (c.get("feedback",{}),c.get("warmup_feedback",{})):
  if broker.get("pending_values_serialized") is not False:raise ValueError("E4 pending values serialized")
  for pending in broker.get("pending_paid_requests_without_values",[]):
   if {"value","values","score","member_scores","features"}&set(pending):raise ValueError("E4 future labels leaked")
 a=identity["inputs"]["assets"];p=identity["inputs"]["locked_provenance"]
 for split in ("online","calibration"):
  for container in (s,c):
   observed=container.get(split+"_provenance",{})
   for k,v in {"matrix_sha256":a[split+"_matrix"]["sha256"],"e0_lock_sha256":p["e0_lock"]["sha256"],
    "e1_gate_sha256":p["e1_gate"]["sha256"],"pca_sha256":p["pca_artifact"]["sha256"]}.items():
    if observed.get(k)!=v:raise ValueError("E4 provenance differs from exact input matrices/locks")
 for container in (s,c):
  auth=container.get("e3_authorization",{})
  if auth.get("gate_sha256")!=identity["e3_gate"]["sha256"]:raise ValueError("E4 missing exact E3 authorization")
 json.dumps(result,allow_nan=False)
 return result


class NeighborEpisodeStore:
 def __init__(self,directory,identity,*,resume=False,parent=True):
  self.directory=Path(directory).resolve();self.identity=verify_identity(identity,parent=parent)
  if self.directory.exists():
   if not resume or read_json(self.directory/"grid_plan.json")!=identity:raise ValueError("E4 resume identity differs")
  else:self.directory.mkdir(parents=True);atomic_json(self.directory/"grid_plan.json",identity)
  self.allowed={episode_name(s):s for s in episode_specs(identity["grid"])}

 def load(self,spec):
  name=episode_name(spec)
  if self.allowed.get(name)!=spec:raise ValueError("E4 episode outside locked grid")
  folder=self.directory/name
  if not folder.exists():return None
  r=read_json(folder/"episode_receipt.json")
  if r.get("schema")!="completed_neighbor_episode_v1" or r.get("spec")!=spec or r.get("run_identity_sha256")!=self.identity["identity_sha256"]:
   raise ValueError("E4 episode receipt identity changed")
  if set(r.get("artifacts",{}))!=set(ARTIFACTS):raise ValueError("E4 artifact inventory incomplete")
  for name in ARTIFACTS:
   expected=artifact_reference(folder/name);ref=r["artifacts"][name]
   if ref!={**expected,"path":name}:raise ValueError("E4 completed artifact changed")
  result={"summary":read_json(folder/"summary.json"),"traces":read_jsonl(folder/"traces.jsonl"),
   "warmup":read_json(folder/"warmup_anonymous.json"),"policy_checkpoint":read_json(folder/"policy_checkpoint.json")}
  return validate_result(result,spec,self.identity)

 def commit(self,result,spec,writer):
  validate_result(result,spec,self.identity)
  if self.load(spec) is not None:raise FileExistsError("E4 completed episode exists")
  name=episode_name(spec);temporary=self.directory/("."+name+".pending."+uuid.uuid4().hex)
  writer(result,temporary)
  for artifact in ARTIFACTS:
   with (temporary/artifact).open("r+b") as stream:os.fsync(stream.fileno())
  artifacts={name:{**artifact_reference(temporary/name),"path":name} for name in ARTIFACTS}
  atomic_json(temporary/"episode_receipt.json",{"schema":"completed_neighbor_episode_v1","spec":spec,
   "run_identity_sha256":self.identity["identity_sha256"],"artifacts":artifacts,"research_complete":False})
  os.replace(temporary,self.directory/name)
  return self.load(spec)


def verify_complete_neighbor_grid(directory,*,parent=True):
 directory=Path(directory).resolve();identity=read_json(directory/"grid_plan.json")
 store=NeighborEpisodeStore(directory,identity,resume=True,parent=parent);results=[];receipts=[]
 for spec in episode_specs(identity["grid"]):
  result=store.load(spec)
  if result is None:raise ValueError("Incomplete E4 grid: "+episode_name(spec))
  results.append(result);receipts.append(artifact_reference(directory/episode_name(spec)/"episode_receipt.json"))
 return identity,results,receipts


_FINAL_BATCH_TOKEN=object()

@dataclass(frozen=True)
class NeighborFinalBatchAuthorization:
 grid_plan_path:str
 grid_plan_sha256:str
 e3_gate_sha256:str
 _token:object=None


def authorize_neighbor_final_batch(grid_plan_ref):
 """Verify upstream source/technical evidence once for a frozen final batch."""
 plan=verified_reference(grid_plan_ref);identity=read_json(plan)
 verify_identity(identity,parent=True)
 return NeighborFinalBatchAuthorization(str(plan),grid_plan_ref["sha256"],identity["e3_gate"]["sha256"],_FINAL_BATCH_TOKEN)


def validate_neighbor_episode(checkpoint_ref,episode_receipt_ref,grid_plan_ref,*,batch_authorization=None):
 plan=verified_reference(grid_plan_ref);identity=read_json(plan)
 if batch_authorization is None:parent=True
 else:
  auth=batch_authorization
  if (not isinstance(auth,NeighborFinalBatchAuthorization) or auth._token is not _FINAL_BATCH_TOKEN
   or str(plan)!=auth.grid_plan_path or grid_plan_ref["sha256"]!=auth.grid_plan_sha256
   or identity["e3_gate"]["sha256"]!=auth.e3_gate_sha256):
   raise ValueError("Invalid or changed final neighbor batch authorization")
  verified_reference(identity["e3_gate"]);parent=False
 store=NeighborEpisodeStore(plan.parent,identity,resume=True,parent=parent)
 receipt_path=verified_reference(episode_receipt_ref);receipt=read_json(receipt_path)
 expected=plan.parent/episode_name(receipt["spec"])/"episode_receipt.json"
 if receipt_path!=expected.resolve():raise ValueError("E4 episode outside its exact grid")
 result=store.load(receipt["spec"])
 if verified_reference(checkpoint_ref)!=(receipt_path.parent/"policy_checkpoint.json").resolve():raise ValueError("Neighbor checkpoint differs from paid episode")
 return result


def validate_e4_gate(path,current_inputs):
 """Mandatory four-neighbour acceptance; a separate Flex receipt completes E4."""
 gate=read_json(path)
 if (gate.get("schema")!="e4_mandatory_neighbors_gate_v1" or gate.get("status")!="E4_MANDATORY_NEIGHBORS_PASS"
  or gate.get("scope")!="technical_completeness_not_scientific_positivity" or gate.get("software_test_only") is not False):
  raise ValueError("A real independently accepted mandatory-neighbor gate is required")
 directory=verified_reference(gate["grid_plan"]).parent
 identity,results,receipts=verify_complete_neighbor_grid(directory,parent=True)
 if identity["inputs"]!=current_inputs or gate.get("input_identity_sha256")!=canonical_digest(current_inputs):
  raise ValueError("E4 gate belongs to different data/config/core algorithm sources")
 if gate.get("run_identity_sha256")!=identity["identity_sha256"] or gate.get("episode_receipts")!=receipts or gate.get("e3_gate")!=identity["e3_gate"]:
  raise ValueError("E4 full-grid/parent binding changed")
 summary=read_json(verified_reference(gate["grid_summary"]))
 if summary.get("run_identity_sha256")!=identity["identity_sha256"] or summary.get("episode_receipts")!=receipts:
  raise ValueError("E4 grid summary incomplete")
 report=read_json(verified_reference(gate["evaluation"]))
 comparisons=report.get("paired_comparisons",[])
 expected={(r,d,"E4/"+method) for r in identity["grid"]["rates"] for d in identity["grid"]["delays"] for method in METHODS}
 observed={(x.get("label_rate"),x.get("feedback_delay"),x.get("reference")) for x in comparisons}
 if (report.get("schema")!="e4_mandatory_neighbors_evaluation_v1" or report.get("run_identity_sha256")!=identity["identity_sha256"]
  or report.get("episodes_evaluated")!=len(receipts) or report.get("comparison_count")!=48 or len(comparisons)!=48
  or observed!=expected or report.get("actual_costs_independently_recomputed") is not True):
  raise ValueError("E4 complete neighbor/control/cost evidence missing")
 for comparison in comparisons:
  if comparison.get("strategy")!="E3/decision" or comparison.get("stream_seeds")!=sorted(identity["grid"]["seeds"]) or comparison.get("question_count")!=600:
   raise ValueError("E4 comparison missing five full paired streams")
  for metric in ("quality","weighted_model_feedback_cost","executions","labels","tokens","controller_cpu_s","controller_wall_s"):
   interval=comparison.get("metrics",{}).get(metric+"_difference",{})
   if interval.get("question_support")!=600 or interval.get("per_task_support")!={"bbh":300,"gsm8k":300} or interval.get("adequate_support") is not True:
    raise ValueError("E4 paired interval lacks full question/task support")
 diagnostics=report.get("selected_service_diagnostics",[])
 if len(diagnostics)!=len(receipts):raise ValueError("E4 selected-service diagnostics missing")
 for ref in diagnostics:
  d=read_json(verified_reference(ref))
  if d.get("schema")!="neighbor_selected_service_diagnostics_v1" or len(d.get("rows",[]))!=600 or d.get("evaluation_only") is not True or d.get("source_policy_update") is not False:
   raise ValueError("E4 diagnostic missing or non-isolated")
 for ref in report.get("e3_decision_episode_receipts",[]):verified_reference(ref)
 if len(report.get("e3_decision_episode_receipts",[]))!=60:raise ValueError("E4 reused E3 decision grid incomplete")
 bindings={"input_identity_sha256":canonical_digest(identity["inputs"]),"neighbor_sources_sha256":canonical_digest(identity["neighbor_sources"]),
  "required_grid_sha256":canonical_digest(identity["grid"]),"e3_gate_sha256":identity["e3_gate"]["sha256"],
  "episode_receipts_sha256":canonical_digest(receipts)}
 if gate.get("bindings")!=bindings:raise ValueError("E4 binding digests differ")
 return gate,bindings
