"""Execute an exact released final job batch on the exclusive local GPU."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from budget_router.final_generation import run_released_batch


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--jobs-manifest",required=True)
 a=p.parse_args(argv);result=run_released_batch(a.jobs_manifest)
 print(json.dumps(result));return result


if __name__=="__main__":main()
