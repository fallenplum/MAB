"""Freeze sealed-test decisions, or evaluate a separately bound real matrix."""
from pathlib import Path
import argparse
import json
import sys
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from budget_router.frozen_evaluation import freeze_final_decisions,evaluate_final_matrix,paired_final_comparisons


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    freeze=commands.add_parser('freeze')
    freeze.add_argument('--release',required=True,type=Path)
    freeze.add_argument('--output',required=True,type=Path)
    evaluate=commands.add_parser('evaluate')
    for name in ('release','decisions','matrix','manifest','output'):
        evaluate.add_argument('--'+name,required=True,type=Path)
    evaluate.add_argument('--reference',default=None)
    evaluate.add_argument('--bootstrap-replicates',type=int,default=2000)
    args=parser.parse_args(argv)
    if args.command=='freeze':result=freeze_final_decisions(args.release,args.output)
    else:
        if args.output.exists():raise FileExistsError('Final evaluation output already exists')
        result=evaluate_final_matrix(args.release,args.decisions,args.matrix,args.manifest)
        if args.reference is not None:
            result['paired_comparisons']=paired_final_comparisons(result,reference=args.reference,replicates=args.bootstrap_replicates)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,ensure_ascii=False,allow_nan=False,indent=2)
    print(json.dumps({'status':result['status'],'research_complete':False},ensure_ascii=False))


if __name__=='__main__':main()
