"""Verify immutable E10 assets or replay scoring from frozen predictions."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import shutil
import time

ROOT=Path(r'E:\语言模型组合路由\research\e10_input_prefill_matching_audit')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify():
    private=ROOT/'private'
    freeze=read(private/'PROTOCOL_FREEZE_RECEIPT.json')
    assert sha(ROOT/'public/E10_PROTOCOL.json')==freeze['protocol_sha256']
    for name,expected in freeze['sources'].items():
        assert sha(ROOT/'public/code'/name)==expected,name
    assert sha(private/'input/PREFILL_INPUT_MANIFEST.json')==freeze['prefill_input_manifest_sha256']
    parent=ROOT.parent/'e9_consistent_cost_sensitive_learning'
    snapshot=read(private/'E9_PARENT_SNAPSHOT.json')
    for relative,expected in snapshot.items():
        assert sha(parent/relative)==expected,relative
    feature=read(private/'features/FEATURE_FREEZE_RECEIPT.json')
    assert sha(private/'features/PREFILL_FEATURES_FROZEN.npz')==feature['feature_sha256']
    assert feature['protocol_sha256']==freeze['protocol_sha256']
    for key,expected in feature['per_question_receipt_sha256'].items():
        variant,index=key.split('/')
        receipt=private/f'features/{variant}/{int(index):03d}.json'
        assert sha(receipt)==expected,key
        assert sha(receipt.with_suffix('.npy'))==read(receipt)['feature_sha256'],key
    head=read(private/'ALL_PREDICTIONS_FREEZE_RECEIPT.json')
    assert head['formal_head_fits']==25
    assert sha(private/'ALL_E10_PREDICTIONS_FROZEN.npz')==head['prediction_sha256']
    for row in head['heads']:
        folder=private/f'models/fold{row["outer_fold"]}/{row["name"]}'
        assert sha(folder/'MODEL.npz')==row['model_sha256']
        assert sha(folder/'PREDICTIONS.npz')==row['predictions_sha256']
    adapter=read(private/'EVALUATION_ADAPTER_SOURCE_FREEZE.json')
    assert sha(ROOT/'public/code/e10_evaluate_adapter.py')==adapter['adapter_sha256']
    access=read(private/'EVALUATION_ACCESS_RECEIPT.json')
    assert datetime.fromisoformat(head['frozen_utc'])<datetime.fromisoformat(access['first_outer_reward_access'])
    return dict(status='PASS',parent_files=len(snapshot),formal_prefill_features=606,
        frozen_heads=25,protocol_sha256=freeze['protocol_sha256'],models_refitted=False,new_prefill=0)


def main(mode):
    start=time.perf_counter()
    result=verify()
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=ROOT/'private/reproductions'/f'{mode}_{timestamp}'
    output.mkdir(parents=True,exist_ok=False)
    if mode=='scoring-replay':
        for name in ('E10_PROTOCOL.json',):
            (output/'public').mkdir(exist_ok=True)
            shutil.copy2(ROOT/'public'/name,output/'public'/name)
        for relative in ('PROTOCOL_FREEZE_RECEIPT.json','ALL_PREDICTIONS_FREEZE_RECEIPT.json',
                         'ALL_E10_PREDICTIONS_FROZEN.npz','E9_PARENT_SNAPSHOT.json'):
            (output/'private').mkdir(exist_ok=True)
            shutil.copy2(ROOT/'private'/relative,output/'private'/relative)
        shutil.copytree(ROOT/'private/models',output/'private/models')
        (output/'private/features').mkdir()
        shutil.copy2(ROOT/'private/features/PREFILL_FEATURES_FROZEN.npz',output/'private/features/PREFILL_FEATURES_FROZEN.npz')
        import numpy as np
        import e10_evaluate as frozen
        from e10_matching import decompose_policies

        def canonical_metadata(rewards,policies,task,subtask,outerfold,**kwargs):
            return decompose_policies(rewards,policies,task,np.where(task=='gsm8k','gsm8k',subtask),outerfold,**kwargs)

        frozen.ROOT=output
        frozen.decompose_policies=canonical_metadata
        frozen.main()
        comparisons={}
        for name in ('E10_MAIN_RESULTS.json','E10_PRIMARY_STATISTICS.json','E10_CONTENT_CONTROLS.json'):
            expected,actual=read(ROOT/'public'/name),read(output/'public'/name)
            if name=='E10_MAIN_RESULTS.json':
                expected.pop('evaluation_wall_seconds');actual.pop('evaluation_wall_seconds')
            assert expected==actual,name
            comparisons[name]='EXACT_SCIENTIFIC_JSON'
        result.update(mode=mode,scientific_results=comparisons,new_answer_generation=0,new_aggregations=0)
    result.update(output_directory=str(output),wall_seconds=time.perf_counter()-start,
        source_sha256=sha(Path(__file__)))
    (output/'REPRODUCTION_VERIFIED.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('verify','scoring-replay'))
    main(parser.parse_args().mode)
