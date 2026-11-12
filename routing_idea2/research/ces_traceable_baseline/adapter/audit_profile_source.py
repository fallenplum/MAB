"""Read-only schema/lineage check for an eligible historical CES profile."""
import os
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/ces_traceable_baseline'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cache=ROOT/'artifacts/private/routing_redesign_20261104/stage2_generation/member_score_cache_actual_20261105_v3'
    receipt_path=cache/'MEMBER_SCORE_CACHE_RECEIPT_v1.json'
    receipt=json.loads(receipt_path.read_text())
    npz=cache/'TRAIN_MEMBER_SCORES_v1.npz'
    assert receipt['status']=='VERIFIED_COMPLETE_TRAIN_MEMBER_SCORE_CACHE'
    assert receipt['software_test_only'] is False
    assert sha(npz)==receipt['score_cache']['sha256']
    bundle_path=ROOT/'artifacts/private/routing_redesign_20261104/ff_v1/BUNDLE_TRAINVAL.json'
    bundle=json.loads(bundle_path.read_text())
    p=Path(bundle['inputs']['selector_train']['path'])
    inputs=p if p.is_absolute() else ROOT/p
    assert sha(inputs)==bundle['inputs']['selector_train']['sha256']
    with np.load(inputs,allow_pickle=False) as a, np.load(npz,allow_pickle=False) as b:
        assert np.array_equal(a['question_ids'],b['question_ids'])
        assert np.array_equal(a['canonical_hashes'],b['canonical_hashes'])
        assert np.array_equal(b['generation_seeds'],[42,137,2024])
        assert b['scores'].shape==b['parsed'].shape==(3,600,6)
        assert np.isin(b['scores'],[0,1]).all()
        assert len(set(b['execution_ids'].ravel().tolist()))==10800
        assert not np.any((~b['parsed'])&(b['scores']!=0))
        parser_failures=int((~b['parsed']).sum())
    result=dict(status='PASS',claim_scope='Eligibility and exact order of existing historical member outcomes, not a free online profile',
        source_cache_sha256=sha(npz),receipt_sha256=sha(receipt_path),input_sha256=sha(inputs),
        question_count=600,generation_seeds=[42,137,2024],members=6,
        successful_execution_cells=10800,unique_execution_ids=10800,
        reward_shape=[3,600,6],binary=True,unparsed_correctness_zero=True,unparsed_cells=parser_failures,
        source_status=receipt['status'],member_scorer=receipt['scorer_version'],reference_version=receipt['reference_version'],
        original_cache_resources=receipt['actual_physical_resources'],
        learner_permission='Only the predeclared I240 rows are eligible for historical profile fitting. Remaining cache outcomes stay in the environment.',
        no_new_inference_calls=0,no_new_gold_acquisition_calls=0)
    out=WORK/'public/adapter'
    out.mkdir(parents=True,exist_ok=True)
    (out/'CES_PROFILE_SOURCE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
