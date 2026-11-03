"""Run the complete four-adapter E4 grid after independent E3 acceptance."""
from pathlib import Path
import argparse,json,sys,time,uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from budget_router.simulation import MatrixEnvironment,save_simulation
from budget_router.neighbor_simulation import authorize_neighbor_pair,run_neighbor_simulation
from budget_router.ranking_evaluation import write_cost_curve
from budget_router.stage_validation import (read_json,input_identity,settings_from_config,
 episode_specs,episode_name,atomic_json,immutable_json,artifact_reference)
from budget_router.neighbor_validation import make_identity,NeighborEpisodeStore


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 for key in ("online-matrix","online-manifest","calibration-matrix","calibration-manifest","e3-gate","output-dir"):
  p.add_argument("--"+key,required=True)
 p.add_argument("--config",default=str(ROOT/"configs/experiment_v1.json"));p.add_argument("--resume",action="store_true")
 a=p.parse_args(argv);config=read_json(a.config)
 inputs=input_identity(ROOT,online_matrix=a.online_matrix,online_manifest=a.online_manifest,
  calibration_matrix=a.calibration_matrix,calibration_manifest=a.calibration_manifest,config_path=a.config)
 online=MatrixEnvironment.from_npz(a.online_matrix,a.online_manifest,expected_split="online_stream")
 cal=MatrixEnvironment.from_npz(a.calibration_matrix,a.calibration_manifest,expected_split="online_calibration")
 if online.software_test_only or cal.software_test_only or len(online)!=600 or len(cal)!=200:
  raise ValueError("Original real E4 matrix scale is required")
 auth=authorize_neighbor_pair(online,cal,gate_path=a.e3_gate,current_inputs=inputs)
 identity=make_identity(ROOT,inputs,a.e3_gate);settings=settings_from_config(config)
 output=Path(a.output_dir);store=NeighborEpisodeStore(output,identity,resume=a.resume,parent=False)
 refs=[];resource_rows=[];specs=episode_specs(identity["grid"])
 for spec in specs:
  result=store.load(spec)
  if result is None:
   attempt=output/"attempts"/(episode_name(spec)+"."+uuid.uuid4().hex+".json")
   record={"schema":"neighbor_cpu_execution_attempt_v1","spec":spec,"status":"STARTED", "started_at_unix":time.time(),
    "run_identity_sha256":identity["identity_sha256"],"scope":"Actual CPU simulation; no new model generation"}
   atomic_json(attempt,record);cpu=time.process_time();wall=time.perf_counter()
   try:
    result=run_neighbor_simulation(online,cal,strategy=spec["strategy"],seed=spec["seed"],
     label_rate=spec["label_rate"],delay=spec["feedback_delay"],settings=settings,e3_authorization=auth)
    def writer(value,folder):
     save_simulation(value,folder);write_cost_curve(value,folder/"cost_curve.csv")
    store.commit(result,spec,writer);record["status"]="COMMITTED"
   except BaseException as error:
    record.update(status="INTERRUPTED_OR_FAILED",error_type=type(error).__name__);raise
   finally:
    record.update(process_time_s=time.process_time()-cpu,wall_time_s=time.perf_counter()-wall)
    atomic_json(attempt,record)
  refs.append(artifact_reference(output/episode_name(spec)/"episode_receipt.json"))
  s=result["summary"]
  resource_rows.append({"spec":spec,"quality_task_macro":s["quality_task_macro"],
   "actual_resources":s["budget"]["spent"],"limits":s["budget"]["limits"],
   "controller_process_time_s":s["controller_process_time_s"],"controller_wall_time_s":s["controller_wall_time_s"],
   "source_service_execution_wall_time_s":s["source_service_execution_wall_time_s"]})
  atomic_json(output/"progress.json",{"status":"RUNNING_NEIGHBOR_GRID","validated_complete_episodes":len(refs),
   "required_episodes":len(specs),"last_episode":episode_name(spec),"research_complete":False})
  print(json.dumps({"validated_episode":episode_name(spec),"complete":len(refs),"required":len(specs),"research_complete":False}),flush=True)
 report={"schema":"neighbor_grid_summary_v1","status":"COMPLETE_MANDATORY_NEIGHBOR_GRID_AWAITING_INDEPENDENT_ACCEPTANCE",
  "run_identity_sha256":identity["identity_sha256"],"grid":identity["grid"],"episode_receipts":refs,
  "resource_measurements_by_episode":resource_rows,"native_systems":"NOT_RUN",
  "FlexRouter_reference":"Separate paid-information coverage-reference evidence required; pure mathematical components are not a trained experiment",
  "physical_generation_replicates":1,"algorithm_seeds_are_not_independent_generations":True,"research_complete":False}
 immutable_json(output/"grid_summary.json",report)
 return report


if __name__=="__main__":main()
