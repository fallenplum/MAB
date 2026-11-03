"""Score actual released final outputs; this evaluator never updates policies."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from budget_router.final_generation import assemble_final_matrix


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("release","decisions-dir","condition","output-dir"):p.add_argument("--"+name,required=True)
 a=p.parse_args(argv);result=assemble_final_matrix(a.release,a.decisions_dir,a.condition,a.output_dir)
 print(json.dumps({"status":"REAL_RELEASED_FINAL_MATRIX_READY_NOT_RESEARCH_COMPLETE","artifact":result,"research_complete":False}))
 return result


if __name__=="__main__":main()
