"""Release-gated CPU encoding of final questions; reuse the frozen PCA only."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata
import uuid
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from budget_router.frozen_evaluation import validate_preparation_release,project_sealed_embeddings
from budget_router.stage_validation import artifact_reference,sha_file,verified_reference,read_json,immutable_json


def load_question_texts(validated,selected):
    """Read public dictionaries but retain only requested IDs/tasks/question text.

    Reference/gold/output fields are neither returned, passed to the encoder,
    nor written into any artifact. Canonical checks use question text only.
    """
    final=selected;wanted={r['question_id']:r for r in final}
    canonical=validated['split']['canonical_question_fingerprints'];texts={}
    for source in validated['question_paths']:
        with Path(source).open(encoding='utf-8-sig') as stream:
            for line in stream:
                if not line.strip():continue
                row=json.loads(line);qid=row.get('question_id')
                if qid not in wanted:continue
                if qid in texts:raise ValueError('Duplicate sealed question in public source dictionaries')
                question=row.get('question')
                if not isinstance(question,str) or not question.strip() or row.get('task')!=wanted[qid]['task']:
                    raise ValueError('Sealed public question metadata/content mismatch')
                normalized=' '.join(unicodedata.normalize('NFKC',question).split())
                if hashlib.sha256(normalized.encode()).hexdigest()!=canonical[qid]:
                    raise ValueError('Sealed question text differs from its locked canonical fingerprint')
                texts[qid]=question
    if set(texts)!=set(wanted):raise ValueError('Complete final public question text coverage missing')
    return [texts[r['question_id']] for r in final]



def load_final_question_texts(validated):
    return load_question_texts(validated,validated['final_records'])


ENCODER_REPLAY_MAX_ABS_TOL=5e-6


class CalibrationEncoderDrift(ValueError):
    def __init__(self,report):
        self.report=report;super().__init__('Current CPU encoder exceeds the fixed calibration replay tolerance')


def compare_calibration_encoder_replay(current,old_embeddings,old_phi,pca_path,calibration_ids):
    current=np.asarray(current,dtype=float);old=np.asarray(old_embeddings,dtype=float);phi=np.asarray(old_phi,dtype=float)
    if (current.ndim!=2 or current.shape!=old.shape or current.shape[0]!=200 or current.shape[1]<16 or phi.shape!=(200,17)
            or len(calibration_ids)!=200 or len(set(calibration_ids))!=200
            or not all(np.isfinite(x).all() for x in (current,old,phi))
            or not np.allclose(phi[:,0],1.)
            or not np.allclose(np.linalg.norm(current,axis=1),1.,atol=1e-5,rtol=1e-5)
            or not np.allclose(np.linalg.norm(old,axis=1),1.,atol=1e-5,rtol=1e-5)):
        raise ValueError('Calibration replay requires exact 200 aligned finite frozen embeddings and phi')
    projected=project_sealed_embeddings(current,pca_path,calibration_question_ids=calibration_ids)
    original_projection=project_sealed_embeddings(old,pca_path,calibration_question_ids=calibration_ids)
    if not np.allclose(original_projection,phi,atol=1e-6,rtol=1e-6):
        raise ValueError('Original calibration phi differs from original saved PCA/embeddings')
    embedding_delta=float(np.max(np.abs(current-old)));phi_delta=float(np.max(np.abs(projected-phi)))
    report={'schema':'calibration_encoder_replay_v1','status':'CALIBRATION_ENCODER_REPLAY_PASS',
        'questions':200,'feature_dimensions':17,'embedding_dimensions':current.shape[1],
        'max_abs_tolerance':ENCODER_REPLAY_MAX_ABS_TOL,'max_abs_embedding_difference':embedding_delta,
        'max_abs_phi_difference':phi_delta,'fit_or_refit_performed':False,'labels_used':False,
        'sealed_final_encoded_during_replay':False,'only_original_calibration_question_text_encoded':True}
    if max(embedding_delta,phi_delta)>ENCODER_REPLAY_MAX_ABS_TOL:
        report['status']='CALIBRATION_ENCODER_REPLAY_FAILED';raise CalibrationEncoderDrift(report)
    return report


def calibration_encoder_replay(validated,encoder,runtime):
    refs=validated['release']['calibration_encoder_replay_sources']
    source=verified_reference(refs['frozen_features']);audit_path=verified_reference(refs['feature_audit']);audit=read_json(audit_path)
    cal_ids=validated['calibration_question_ids'];config=validated['config']['features']
    if (audit.get('fit_split')!='online_calibration' or set(audit.get('fit_question_ids',[]))!=set(cal_ids)
            or len(audit.get('fit_question_ids',[]))!=200
            or audit.get('encoder_id')!=config['encoder_id'] or audit.get('encoder_revision')!=config['encoder_revision']
            or audit.get('encoder_source_receipt_sha256')!=validated['release']['artifacts']['encoder_source_receipt']['sha256']
            or audit.get('feature_artifact',{}).get('sha256')!=refs['frozen_features']['sha256']
            or audit.get('pca_artifact',{}).get('sha256')!=validated['release']['artifacts']['pca_artifact']['sha256']):
        raise ValueError('Original calibration encoder/PCA feature audit identity changed')
    with np.load(source,allow_pickle=False) as data:
        if set(data.files)!={'question_ids','task_labels','splits','phi','embeddings'}:
            raise ValueError('Original frozen feature schema changed')
        ids=data['question_ids'].tolist();splits=data['splits'].tolist();features=np.asarray(data['phi']);embeddings=np.asarray(data['embeddings'])
        if (len(ids)!=800 or len(set(ids))!=800 or features.shape!=(800,17) or embeddings.shape!=(800,384)
                or len(splits)!=800 or not np.isfinite(features).all() or not np.isfinite(embeddings).all()):
            raise ValueError('Original full-scale 800-row frozen feature archive changed')
        locations={qid:i for i,qid in enumerate(ids)}
        if any(qid not in locations or splits[locations[qid]]!='online_calibration' for qid in cal_ids):
            raise ValueError('Calibration replay may use only exact original 200 calibration rows')
        indices=[locations[qid] for qid in cal_ids];old_embeddings=embeddings[indices].copy();old_phi=features[indices].copy()
    records={r['question_id']:r for r in validated['split']['records']}
    selected=[records[qid] for qid in cal_ids]
    if any(r['split']!='online_calibration' for r in selected):raise ValueError('Encoder consistency guard cannot encode online/final questions')
    texts=load_question_texts(validated,selected);cpu=time.process_time();wall=time.perf_counter()
    details={'source_features':refs['frozen_features'],'source_feature_audit':refs['feature_audit'],
        'pca_artifact':validated['release']['artifacts']['pca_artifact'],
        'encoder_source_receipt':validated['release']['artifacts']['encoder_source_receipt'],
        'source_code':{name:validated['release']['final_sources'][name] for name in ('scripts/prepare_final_features.py','src/budget_router/encoding.py')},
        'calibration_question_ids':cal_ids,'runtime':runtime,'device':'cpu','encoder_max_tokens':256,'batch_size':8}
    try:
        current=encoder.encode(texts,batch_size=8)
        report=compare_calibration_encoder_replay(current,old_embeddings,old_phi,validated['paths']['pca_artifact'],cal_ids)
    except BaseException as error:
        report=getattr(error,'report',{'schema':'calibration_encoder_replay_v1','status':'CALIBRATION_ENCODER_REPLAY_FAILED',
            'failure_type':type(error).__name__,'max_abs_tolerance':ENCODER_REPLAY_MAX_ABS_TOL,
            'max_abs_embedding_difference':None,'max_abs_phi_difference':None})
        report={**report,**details,'process_time_s':time.process_time()-cpu,'wall_time_s':time.perf_counter()-wall}
        path=ROOT/'artifacts/audit/calibration_encoder_replay'/(uuid.uuid4().hex+'.json');immutable_json(path,report)
        raise RuntimeError('Frozen calibration encoder replay failed; original features remain unchanged; report '+str(path)) from error
    report={**report,**details,'process_time_s':time.process_time()-cpu,'wall_time_s':time.perf_counter()-wall}
    path=ROOT/'artifacts/audit/calibration_encoder_replay'/(uuid.uuid4().hex+'.json');immutable_json(path,report)
    return artifact_reference(path)


def prepare_final_features(preparation_release_path,output_directory):
    # No raw question or encoder load occurs until actual pre-final gates pass.
    validated=validate_preparation_release(preparation_release_path)
    output=Path(output_directory).resolve()
    if output.exists():raise FileExistsError('Final features are immutable; output directory already exists')
    import torch
    import transformers
    from budget_router.encoding import FrozenQuestionEncoder
    torch.set_num_threads(2)
    config=validated['config'];spec=config['features']
    encoder_receipt=validated['paths']['encoder_source_receipt'];encoder_root=Path(validated['encoder']['path'])
    if not encoder_root.is_absolute():encoder_root=ROOT/encoder_root
    started=time.perf_counter();cpu_started=time.process_time()
    encoder=FrozenQuestionEncoder(encoder_root,revision=spec['encoder_revision'],receipt=encoder_receipt,
        device='cpu',max_length=256)
    runtime={'python':sys.version.split()[0],'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__}
    replay=calibration_encoder_replay(validated,encoder,runtime)
    texts=load_final_question_texts(validated)
    embeddings=encoder.encode(texts,batch_size=8)
    phi=project_sealed_embeddings(embeddings,validated['paths']['pca_artifact'],
        calibration_question_ids=validated['calibration_question_ids'])
    if embeddings.shape[0]!=200 or phi.shape!=(200,17):raise ValueError('Final CPU encoding changed original scale')
    if sha_file(validated['paths']['pca_artifact'])!=validated['release']['artifacts']['pca_artifact']['sha256']:
        raise ValueError('Frozen PCA changed during final CPU encoding')
    temporary=output.with_name('.'+output.name+'.'+uuid.uuid4().hex+'.pending');temporary.mkdir(parents=True)
    feature_path=temporary/'final_features_v1.npz'
    np.savez_compressed(feature_path,question_ids=np.array([r['question_id'] for r in validated['final_records']]),
        task_labels=np.array([r['task'] for r in validated['final_records']]),embeddings=embeddings,phi=phi)
    with feature_path.open('r+b') as stream:os.fsync(stream.fileno())
    manifest={'schema':'sealed_features_only_v1','status':'FROZEN_FINAL_FEATURES_READY_NOT_FINAL_RESULTS',
        'split':'sealed_test','no_labels':True,'question_count':200,'feature_dimensions':17,'pca_dimensions':16,
        'features_sha256':sha_file(feature_path),'pca_sha256':validated['release']['artifacts']['pca_artifact']['sha256'],
        'encoder_id':spec['encoder_id'],'encoder_revision':spec['encoder_revision'],'pooling':spec['pooling'],
        'encoder_max_tokens':256,'device':'cpu','batch_size':8,'cpu_threads':2,
        'encoder_source_receipt':artifact_reference(encoder_receipt),'calibration_encoder_replay':replay,
        'feature_generation_source':artifact_reference(Path(__file__).resolve()),
        'feature_dependency_sources':{'src/budget_router/encoding.py':artifact_reference(ROOT/'src/budget_router/encoding.py')},
        'preparation_release':artifact_reference(preparation_release_path),
        'preparation_release_sha256':validated['release']['preparation_release_sha256'],
        'frozen_pca_fit_split':'online_calibration','pca_refitted':False,'gold_or_member_outputs_serialized':False,
        'features_artifact':{'path':str(output/'final_features_v1.npz'),'sha256':sha_file(feature_path),'bytes':feature_path.stat().st_size},
        'elapsed_wall_time_s':time.perf_counter()-started,'cpu_process_time_s':time.process_time()-cpu_started,
        'runtime':runtime,
        'scientific_completion':False}
    with (temporary/'sealed_features_manifest.json').open('x',encoding='utf-8') as stream:
        json.dump(manifest,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.flush();os.fsync(stream.fileno())
    os.rename(temporary,output)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-release',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    result=prepare_final_features(args.preparation_release,args.output)
    print(json.dumps({'status':result['status'],'question_count':result['question_count'],
        'features_sha256':result['features_sha256'],'scientific_completion':False},ensure_ascii=False))


if __name__=='__main__':main()
