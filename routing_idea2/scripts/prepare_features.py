"""Real frozen CPU encoding, train-only PCA; excludes all E1 and sealed tests."""
from pathlib import Path
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from budget_router.preparation import load_inputs,file_sha
from budget_router.encoding import FrozenQuestionEncoder
from budget_router.features import FrozenPCAFeatures


def main():
    torch.set_num_threads(2)
    manifest,records,questions,_=load_inputs(ROOT)
    selected=[r for r in records if r['split'] in {'online_calibration','online_stream'}]
    fit=[r['question_id'] for r in selected if r['split']=='online_calibration']
    if len(selected)!=800 or len(fit)!=200:raise ValueError('Full-scale feature split required')
    excluded={r['question_id'] for r in records if r['split'] not in {'online_calibration','online_stream'}}
    ids=[r['question_id'] for r in selected]
    if set(ids)&excluded:raise ValueError('E1/qualification/sealed embedding leakage')
    revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
    encoder=FrozenQuestionEncoder(ROOT/'models/sentence-transformers--all-MiniLM-L6-v2',revision=revision,
                                 receipt=ROOT/'artifacts/audit/encoder_model_download.json',device='cpu',max_length=256)
    started=time.time();embeddings=encoder.encode([questions[qid]['question'] for qid in ids],batch_size=8)
    fitmask=np.asarray([r['split']=='online_calibration' for r in selected])
    projection=FrozenPCAFeatures(16,encoder_id='sentence-transformers/all-MiniLM-L6-v2').fit(
        embeddings[fitmask],split='train',sample_ids=fit)
    phi=projection.transform(embeddings)
    if not np.isfinite(phi).all() or not np.allclose(phi[:,0],1.) or not np.allclose(phi[fitmask,1:].mean(0),0.,atol=1e-6):
        raise ValueError('Invalid train-only frozen features')
    output=ROOT/'data/derived/features';output.mkdir(parents=True,exist_ok=True)
    pca_path=output/'pca_v1.npz';feature_path=output/'features_v1.npz'
    if pca_path.exists() or feature_path.exists():raise ValueError('Existing real features are immutable; verify/reuse instead of overwrite')
    np.savez_compressed(pca_path,components=projection._pca.components_,mean=projection.pca_mean,
      fit_question_ids=np.asarray(fit),explained_variance=projection._pca.explained_variance_)
    np.savez_compressed(feature_path,question_ids=np.asarray(ids),task_labels=np.asarray([r['task'] for r in selected]),
      splits=np.asarray([r['split'] for r in selected]),phi=phi,embeddings=embeddings)
    audit={'status':'REAL_FROZEN_FEATURES_READY_NOT_MODEL_COMBINATION_RESULTS','fit_split':'online_calibration',
      'fit_question_ids':fit,'questions':len(ids),'fit_questions':len(fit),'feature_dimensions':17,
      'encoder_id':'sentence-transformers/all-MiniLM-L6-v2','encoder_revision':revision,
      'encoder_source_receipt_sha256':file_sha(ROOT/'artifacts/audit/encoder_model_download.json'),
      'pooling':'attention_mask_mean_then_L2','encoder_max_tokens':256,'pca_dimensions':16,
      'pca_artifact':{'path':pca_path.relative_to(ROOT).as_posix(),'sha256':file_sha(pca_path)},
      'feature_artifact':{'path':feature_path.relative_to(ROOT).as_posix(),'sha256':file_sha(feature_path)},
      'split_manifest_sha256':file_sha(ROOT/'data/derived/split_manifest.json'),
      'e1_questions_labels_embeddings_used':False,'sealed_test_questions_encoded':False,
      'elapsed_seconds':time.time()-started,'scientific_completion':False}
    destination=ROOT/'artifacts/audit/feature_audit.json';destination.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('fit_question_ids',)},ensure_ascii=False))
if __name__=='__main__':main()
