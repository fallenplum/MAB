from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from budget_router.preparation import prepare_members,prepare_aggregations,prepare_alone

def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['members','aggregations','alone'],required=True)
 p.add_argument('--stage',choices=['qualification','e1','online_calibration','online_stream'],default='qualification')
 p.add_argument('--alias');p.add_argument('--historical-qualification',action='store_true');a=p.parse_args()
 if a.kind=='alone' and a.stage!='e1':p.error('--kind alone requires --stage e1')
 if a.historical_qualification and (a.kind!='aggregations' or a.stage!='qualification'):
  p.error('--historical-qualification requires --kind aggregations --stage qualification')
 if a.alias is not None and a.kind!='members':p.error('--alias is valid only for member jobs')
 if a.kind=='members':receipt=prepare_members(ROOT,a.stage,only_alias=a.alias)
 elif a.kind=='alone':receipt=prepare_alone(ROOT)
 else:receipt=prepare_aggregations(ROOT,a.stage,qualification_from_historical=a.historical_qualification)
 print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
