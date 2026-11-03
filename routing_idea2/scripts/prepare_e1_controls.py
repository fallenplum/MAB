from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from budget_router.e1_controls import select_strong_model,prepare_strong_samples,prepare_strong_aggregation
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['select','samples','aggregate'],required=True);p.add_argument('--qualification',action='store_true');a=p.parse_args()
if a.kind=='select':r=select_strong_model(ROOT)[1]
elif a.kind=='samples':r=prepare_strong_samples(ROOT,qualification=a.qualification)
else:r=prepare_strong_aggregation(ROOT)
print(json.dumps(r,ensure_ascii=False))
