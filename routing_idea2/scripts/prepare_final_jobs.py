"""Prepare released sealed-test jobs only after every final choice is frozen."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from budget_router.final_generation import prepare_member_jobs,prepare_aggregate_jobs


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--kind",choices=("members","aggregations"),required=True)
 p.add_argument("--release",required=True);p.add_argument("--decisions-dir",required=True);p.add_argument("--condition")
 a=p.parse_args(argv)
 if a.kind=="members":
  if a.condition:raise ValueError("Member caches are shared across declared conditions")
  result=prepare_member_jobs(a.release,a.decisions_dir)
 else:
  if not a.condition:raise ValueError("An explicitly released condition is required")
  result=prepare_aggregate_jobs(a.release,a.decisions_dir,a.condition)
 print(json.dumps({"status":"RELEASED_FINAL_JOBS_PREPARED_NOT_EXECUTED","artifacts":result,"research_complete":False}))
 return result


if __name__=="__main__":main()
