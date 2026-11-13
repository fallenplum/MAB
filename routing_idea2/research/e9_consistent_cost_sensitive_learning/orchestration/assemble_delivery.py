"""Assemble the completed E9 evidence, preserving frozen sources and results."""
from pathlib import Path
import hashlib
import json
import shutil

STAGE=Path(__file__).resolve().parent
ROOT=Path(r'E:\语言模型组合路由\research\e9_consistent_cost_sensitive_learning')
PUBLIC=ROOT/'public'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def copy_once(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and sha(source)!=sha(target):raise AssertionError('Frozen public bytes differ '+str(target))
    if not target.exists():shutil.copy2(source,target)


def main():
    for folder in (STAGE/'code',STAGE/'math/public'):
        for path in folder.glob('*.py'):copy_once(path,PUBLIC/'code'/path.name)
    for folder in (STAGE/'math/public',STAGE/'audit',STAGE/'public_qualification',STAGE/'reports'):
        for path in folder.glob('*'):
            if path.is_file() and path.suffix in ('.json','.md','.csv'):
                copy_once(path,PUBLIC/path.name)
    copy_once(STAGE/'reports/build_cost_and_tables.py',PUBLIC/'code/build_cost_and_tables.py')
    for name in ('README.md','REPRODUCE.md'):copy_once(STAGE/name,PUBLIC/name)
    for path in STAGE.glob('*.py'):copy_once(path,PUBLIC/'orchestration'/path.name)
    replay=sorted((ROOT/'private/reproductions').glob('scoring-replay_*/REPRODUCTION_VERIFIED.json'))
    if len(replay)!=1:raise AssertionError('One completed scoring replay expected')
    copy_once(replay[0],PUBLIC/'E9_SCORING_REPRODUCTION_VERIFIED.json')
    main_result=read(PUBLIC/'E9_MAIN_RESULTS.json')
    combined=dict(status='COMPLETE_EXPLORATORY_E9',main_matrix=main_result,
        cost=read(PUBLIC/'COST.json'),primary=read(PUBLIC/'E9_PRIMARY_STATISTICS.json'),
        decision='HOLD_CURRENT_CANDIDATES_AND_STOP_SAME_MINILM_LOSS_ACQUISITION_GRID',
        decision_scope='finite declared grid and repeatedly used development data, not all model classes',
        new_LLM_calls=0,retained_outcome_reads=0,
        artifact_sha256={name:sha(PUBLIC/name) for name in (
            'E9_PROTOCOL.json','E9_MAIN_RESULTS.json','E9_PRIMARY_STATISTICS.json','COST.json',
            'LITERATURE_AND_TARGET.md','EXPERIMENT_SUMMARY.md')})
    (PUBLIC/'E9_COMPLETE_RESULTS.json').write_text(json.dumps(combined,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='ASSEMBLED',public_files=len([p for p in PUBLIC.rglob('*') if p.is_file()]),
        bytes=sum(p.stat().st_size for p in PUBLIC.rglob('*') if p.is_file())),ensure_ascii=False))


if __name__=='__main__':main()
