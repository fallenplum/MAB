"""Accept mandatory shared-permission E4 neighbours; Flex is a separate reference."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from budget_router.simulation import MatrixEnvironment
from budget_router.neighbor_validation import verify_complete_neighbor_grid,METHODS
from budget_router.stage_validation import (read_json,immutable_json,artifact_reference,canonical_digest,
 verified_reference,EpisodeStore,episode_specs)
from budget_router.ranking_evaluation import stratified_question_interval
from evaluate_e3 import project_episode,METRICS


def compare_paired(left,right,*,label_rate,delay,reference,seeds):
 def indexed(results):
  return {r["summary"]["seed"]:r for r in results if r["summary"]["label_rate"]==label_rate and r["summary"]["feedback_delay"]==delay}
 a=indexed(left);b=indexed(right)
 if set(a)!=set(seeds) or set(b)!=set(seeds):raise ValueError("E4 paired five-seed grid incomplete")
 out={}
 for metric in METRICS:
  rows=[];missing=0
  for seed in seeds:
   x={r["question_id"]:r for r in a[seed]["traces"]};y={r["question_id"]:r for r in b[seed]["traces"]}
   if set(x)!=set(y) or len(x)!=600:raise ValueError("E4 pair has different or incomplete questions")
   for qid in x:
    if x[qid]["task"]!=y[qid]["task"]:raise ValueError("E4 task identity changed")
    v,w=x[qid].get(metric),y[qid].get(metric)
    if v is None or w is None:missing+=1;continue
    rows.append({"question_id":qid,"task":x[qid]["task"],"difference":float(v)-float(w)})
  out[metric+"_difference"]=(stratified_question_interval(rows,value_key="difference") if not missing else
   {"status":"MISSING_SOURCE_MEASUREMENTS_NOT_ZERO","missing_seed_question_pairs":missing,"interval":None,"question_support":0,"adequate_support":False})
 return {"strategy":"E3/decision","reference":reference,"label_rate":label_rate,"feedback_delay":delay,
  "stream_seeds":sorted(seeds),"question_count":600,"metrics":out,
  "scope":"Question-stratified paired intervals conditional on one LM generation cache; 5 algorithm seeds averaged within question; descriptive multiple comparisons"}


def evaluate(directory,gate_path=None):
 directory=Path(directory).resolve();identity,results,receipts=verify_complete_neighbor_grid(directory)
 a=identity["inputs"]["assets"]
 online=MatrixEnvironment.from_npz(a["online_matrix"]["path"],a["online_manifest"]["path"],expected_split="online_stream")
 cal=MatrixEnvironment.from_npz(a["calibration_matrix"]["path"],a["calibration_manifest"]["path"],expected_split="online_calibration")
 projected={method:[] for method in METHODS};support=[]
 for r in results:
  method=r["summary"]["strategy"];projected[method].append(project_episode(r,online,cal))
  summary=r["summary"];rows=[]
  by_id={online.context(i).question_id:i for i in range(len(online))}
  for tr in r["traces"]:
   row=by_id[tr["question_id"]];team=online.mapper.subsets[tr["main_index"]]
   coverage=float(np.any(online._arrays["member_scores"][row,list(team)]==1.))
   quality=float(online._arrays["aggregate_scores"][row,tr["main_index"]])
   if coverage!=tr["candidate_coverage"] or coverage-quality!=tr["aggregation_loss"]:
    raise ValueError("E4 coverage/aggregate diagnostic differs from executed source matrix")
   rows.append({"question_id":tr["question_id"],"task":tr["task"],"quality":quality,
    "candidate_coverage":coverage,"aggregation_loss":coverage-quality,
    "oracle_selection_regret":float(np.max(online._arrays["aggregate_scores"][row]))-quality})
  path=directory/"e4_diagnostics"/(f"{method}_seed{summary['seed']}_rate{summary['label_rate']}_delay{summary['feedback_delay']}.json")
  immutable_json(path,{"schema":"neighbor_selected_service_diagnostics_v1","rows":rows,
   "evaluation_only":True,"source_policy_update":False})
  support.append(artifact_reference(path))
 e3=read_json(verified_reference(identity["e3_gate"]));e3plan=read_json(verified_reference(e3["grid_plan"]))
 e3store=EpisodeStore(Path(e3["grid_plan"]["path"]).parent,e3plan,resume=True)
 decision=[];control_refs=[]
 for spec in episode_specs(e3plan["grid"]):
  if spec["strategy"]!="decision":continue
  r=e3store.load(spec);decision.append(project_episode(r,online,cal))
  from budget_router.stage_validation import episode_name
  control_refs.append(artifact_reference(e3store.directory/episode_name(spec)/"episode_receipt.json"))
 comparisons=[]
 for rate in identity["grid"]["rates"]:
  for delay in identity["grid"]["delays"]:
   for method in METHODS:comparisons.append(compare_paired(decision,projected[method],label_rate=rate,
    delay=delay,reference="E4/"+method,seeds=identity["grid"]["seeds"]))
 grid_summary=read_json(directory/"grid_summary.json")
 if grid_summary.get("run_identity_sha256")!=identity["identity_sha256"] or grid_summary.get("episode_receipts")!=receipts:
  raise ValueError("E4 complete-grid summary changed")
 report={"schema":"e4_mandatory_neighbors_evaluation_v1","status":"COMPLETE_MANDATORY_NEIGHBOR_EVIDENCE",
  "run_identity_sha256":identity["identity_sha256"],"episodes_evaluated":len(receipts),
  "paired_comparisons":comparisons,"comparison_count":len(comparisons),"e3_decision_episode_receipts":control_refs,
  "selected_service_diagnostics":support,"actual_costs_independently_recomputed":True,
  "native_systems":"NOT_RUN","adaptations":"CES paid two-task profile; C2 fixed parallel K3/shared paid feedback/hard cap; AWC local SLSQP, no approximation or regret theorem inherited",
  "information_regime":"Same generated-answer/reference permissions and same paid random combo warmup as primary methods; actual resource vectors and costs reported, equal caps do not imply equal spending",
  "FlexRouter_reference":"NOT_ACCEPTED_BY_THIS_MANDATORY_GATE; separate trained, explicitly priced coverage-reference evidence is required before E4 full completion",
  "academic_conclusion":"DESCRIPTIVE_MULTIPLE_COMPARISONS_NO_NEW_POSITIVE_GATE","research_complete":False}
 path=directory/"e4_diagnostics"/"mandatory_evaluation.json";immutable_json(path,report)
 gate={"schema":"e4_mandatory_neighbors_gate_v1","status":"E4_MANDATORY_NEIGHBORS_PASS",
  "scope":"technical_completeness_not_scientific_positivity","software_test_only":False,
  "grid_plan":artifact_reference(directory/"grid_plan.json"),"grid_summary":artifact_reference(directory/"grid_summary.json"),
  "evaluation":artifact_reference(path),"episode_receipts":receipts,"e3_gate":identity["e3_gate"],
  "input_identity_sha256":canonical_digest(identity["inputs"]),"run_identity_sha256":identity["identity_sha256"],
  "bindings":{"input_identity_sha256":canonical_digest(identity["inputs"]),"neighbor_sources_sha256":canonical_digest(identity["neighbor_sources"]),
   "required_grid_sha256":canonical_digest(identity["grid"]),"e3_gate_sha256":identity["e3_gate"]["sha256"],
   "episode_receipts_sha256":canonical_digest(receipts)},"research_complete":False}
 destination=Path(gate_path) if gate_path else directory/"e4_mandatory_gate.json";immutable_json(destination,gate)
 from budget_router.neighbor_validation import validate_e4_gate
 validate_e4_gate(destination,identity["inputs"])
 return gate


if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--directory",required=True);p.add_argument("--gate")
 a=p.parse_args();print(json.dumps(evaluate(a.directory,a.gate),ensure_ascii=False))
