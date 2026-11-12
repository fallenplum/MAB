"""Read-only CES author-code imports and six-model cache provenance."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import ast
import importlib.util
import sys
import json
import re
import hashlib
from typing import List
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/ces_traceable_baseline'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
UPSTREAM=ROOT/'third_party/ces_source_audit/upstream'
E7=ROOT/'research/e7_opportunity_and_feedback_learning'
sys.path.insert(0,str(E7/'public'))
from e7_common import load_train,resources,put,sha,splits,paired_interval,holm,TEAMS

SOURCE_PINS={
    'src/bandits/linucb_combinatorial.py':'ac066dfe08517d98180b7cf8792358493b3bfaf95bd38a26a22dd334ec95bc8e',
    'src/embeddings/projectors.py':'397b78f6544239b88baed2e528f599dfcf21b97482ce35e54871efe0f833392f',
    'scripts/run_objective_mcq.py':'255a12c33dc8c965cdb85beca41b9a89fb9090d1901b729d0e2e03e027a5b27e',
    'LICENSE':'1f7167285b17af5f1f4dbc60d25a4c1a2603149f963bf7a530b169256385b321'}


def _module(relative):
    path=UPSTREAM/relative
    assert sha(path)==SOURCE_PINS[relative]
    name='_ces_traceable_'+SOURCE_PINS[relative]
    if name in sys.modules:
        return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def author_operators():
    for name,digest in SOURCE_PINS.items():
        assert sha(UPSTREAM/name)==digest
    core=_module('src/bandits/linucb_combinatorial.py')
    projectors=_module('src/embeddings/projectors.py')
    runner=ast.parse((UPSTREAM/'scripts/run_objective_mcq.py').read_text('utf-8'))
    function=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name=='init_mats_pca_warmstart')
    namespace={'np':np,'List':List}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(UPSTREAM/'scripts/run_objective_mcq.py'),'exec'),namespace)
    encoder_path=UPSTREAM/'src/embeddings/encoders.py'
    source=ast.parse(encoder_path.read_text('utf-8'))
    manual=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='FeatEncoder')
    exec(compile(ast.Module(body=[manual],type_ignores=[]),str(encoder_path),'exec'),namespace)
    return core.LinUCBSharedCombinatorial,projectors.build_phi,namespace['init_mats_pca_warmstart'],namespace['FeatEncoder']


def load_member_scores(data):
    folder=ROOT/'artifacts/private/routing_redesign_20261104/stage2_generation/member_score_cache_actual_20261105_v3'
    receipt=json.loads((folder/'MEMBER_SCORE_CACHE_RECEIPT_v1.json').read_text('utf-8'))
    path=folder/'TRAIN_MEMBER_SCORES_v1.npz'
    assert receipt['status']=='VERIFIED_COMPLETE_TRAIN_MEMBER_SCORE_CACHE'
    assert sha(path)==receipt['score_cache']['sha256']=='5c10eee69598e22d9f5f561c57ea6c2ff5a08c1c18278543a682df4da12a3c0c'
    with np.load(path,allow_pickle=False) as a:
        assert np.array_equal(a['question_ids'],data['ids'])
        assert np.array_equal(a['canonical_hashes'],data['canonical'])
        assert np.array_equal(a['generation_seeds'],[42,137,2024])
        scores=a['scores'].transpose(1,0,2).astype(float)
        parsed=a['parsed'].transpose(1,0,2)
    assert scores.shape==(600,3,6) and np.isin(scores,[0,1]).all()
    assert np.all(scores[~parsed]==0)
    scores.flags.writeable=False
    return scores,dict(cache_sha256=sha(path),receipt_sha256=sha(folder/'MEMBER_SCORE_CACHE_RECEIPT_v1.json'),
        failed_parse_retained=int((~parsed).sum()),score_unit='original executed member correctness with legal shared Gold')


def manual_inputs(data):
    """Extract only target IDs and JSON question strings, never reference fields."""
    idpattern=re.compile(r'(?<!\\)"question_id"\s*:\s*("(?:[^"\\]|\\.)*")')
    qpattern=re.compile(r'(?<!\\)"question"\s*:\s*("(?:[^"\\]|\\.)*")')
    required=set(map(str,data['ids']))
    found={}
    sources=[]
    for relative in ['data/derived/available_records.jsonl','data/derived/gsm8k_questions.jsonl']:
        path=ROOT/relative
        sources.append(dict(path=relative,sha256=sha(path)))
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                match=idpattern.search(line)
                if match is None:
                    continue
                identity=json.loads(match.group(1))
                if identity not in required:
                    continue
                match=qpattern.search(line)
                if match is None:
                    raise ValueError('Missing selected question text')
                question=json.loads(match.group(1))
                if identity in found and found[identity]!=question:
                    raise ValueError('Conflicting question text')
                found[identity]=question
    if set(found)!=required:
        raise ValueError('Selected question text coverage differs: '+str((len(found),len(required))))
    _,_,_,Encoder=author_operators()
    encoder=Encoder(d_c=8)
    values=np.asarray([encoder(found[str(q)]) for q in data['ids']])
    assert values.shape==(600,8) and np.isfinite(values).all()
    return values,dict(source_files=sources,target_question_strings=600,reference_fields_decoded_by_extractor=0,
        fields=['question_id','question'],encoder_class_sha256=sha(UPSTREAM/'src/embeddings/encoders.py'))


def make_features(data,member,I,seed,manual):
    _,build_phi,initialize,_=author_operators()
    I=np.asarray(I,int)
    ability=np.column_stack([member[I[data['task'][I]==task]].mean(axis=(0,1)) for task in ['gsm8k','bbh']])
    historical_cost=(data['cost']['member_prompt_tokens'][:,I]+data['cost']['member_completion_tokens'][:,I]).mean(axis=(0,1))
    spread=float(historical_cost.max()-historical_cost.min())
    cost=np.zeros(6) if spread<=0 else (historical_cost-historical_cost.min())/spread
    profile=np.column_stack((ability,cost))
    W_x,W_m=initialize(d_x=32,d_u=392,s_list=list(profile),seed=int(seed))
    x=np.column_stack((data['E'],manual))@W_x.T
    v=profile@W_m.T
    phi=np.asarray([[build_phi(q,vi,beta=3.,gamma=3.) for vi in v] for q in x])
    assert phi.shape==(600,6,96)
    return phi,dict(profile=profile,W_x=W_x,W_m=W_m,x=x,v=v,historical_member_tokens=historical_cost,
        alpha=np.array(3.5),ridge=np.array(.1),profile_questions=I)
