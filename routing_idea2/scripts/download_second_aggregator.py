"""Pinned second-aggregator assets, permitted only after complete real E1 PASS."""
from pathlib import Path
import json,sys,time
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from budget_router.stage_validation import read_json,atomic_json
from download_models_resumable import locked_download_model


def main():
 gate=read_json(ROOT/"results/e1/e1_diagnostics.json")
 if gate.get("status")!="PASS" or gate.get("evidence",{}).get("software_test_only") is not False:
  raise ValueError("Actual full E1 PASS required before second-aggregator asset work")
 spec=read_json(ROOT/"configs/aggregator_mistral_v03.json")
 plan=read_json(ROOT/"protocol/e5_robustness_plan_v1.json")["second_aggregator"]
 if (spec["model_id"],spec["revision"])!=(plan["model_id"],plan["revision"]):raise ValueError("Second model differs from predeclared identity")
 status=ROOT/"artifacts/audit/download_second_aggregator_status.json"
 attempt=0
 while True:
  attempt+=1
  atomic_json(status,{"status":"RUNNING","pid":__import__("os").getpid(),"attempt":attempt,"updated_at_unix":time.time(),"model_id":spec["model_id"],"revision":spec["revision"]})
  try:
   locked_download_model("aggregator_second",spec["model_id"],spec["revision"])
   atomic_json(status,{"status":"COMPLETED","attempt":attempt,"updated_at_unix":time.time(),"research_complete":False});return
  except (requests.RequestException,TimeoutError,ConnectionError) as e:
   atomic_json(status,{"status":"TRANSIENT_NETWORK_RETRY","attempt":attempt,"error_type":type(e).__name__,"updated_at_unix":time.time()})
   time.sleep(30)


if __name__=="__main__":main()
