"""Independent dense references for the actual E7 learner implementation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.linalg import cho_factor,cho_solve

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e7_opportunity_and_feedback_learning'
sys.path.insert(0,str(WORK/'public/b'))
sys.path.insert(0,str(ROOT/'research/e5_paired_gain_validation/public'))
from e7_models import build_spec,ModelSpec
from e5_common import load_train
from e4_gaussian import GaussianTemplate


def dense(spec,rows,y):
    H=np.asarray(rows,dtype=int)
    factor=cho_factor(spec.K[np.ix_(H,H)]+spec.noise*np.eye(len(H)),lower=True)
    weights=cho_solve(factor,y-spec.offset)
    mean=spec.offset+spec.K[:,H]@weights
    covariance=spec.K-spec.K[:,H]@cho_solve(factor,spec.K[H])
    return mean,(covariance+covariance.T)/2


def main():
    data,_=load_train()
    source=WORK/'public/b/e7_models.py'
    with np.load(WORK/'private/E7_SPLITS_FROZEN.npz') as s:
        print(json.dumps(dict(split_keys=s.files)),flush=True)
        I,U,E=(s['f0_'+k] for k in ['I','U','E'])
    cases=[]
    saved=WORK/'private/audit_replays/models'
    saved.mkdir(parents=True,exist_ok=True)
    for kind in ['P1','P2','P3']:
        spec=build_spec(data['E'],I,data['Y'][I],kind,tasks=data['task'],cv_seed=20261112)
        posterior=spec.posterior(I,data['Y'][I])
        mean,cov=dense(spec,I,data['Y'][I].mean(axis=1))
        initial_mean_error=float(np.max(abs(posterior.mean-mean)))
        initial_cov_error=float(np.max(abs(posterior.cov-cov)))
        rows=I.copy()
        labels=data['Y'][I].mean(axis=1)
        maxmean,maxcov=initial_mean_error,initial_cov_error
        for q in U[:10]:
            posterior.update(q,data['Y'][q])
            rows=np.r_[rows,q]
            labels=np.vstack((labels,data['Y'][q].mean(axis=0)))
            mean,cov=dense(spec,rows,labels)
            maxmean=max(maxmean,float(np.max(abs(posterior.mean-mean))))
            maxcov=max(maxcov,float(np.max(abs(posterior.cov-cov))))
        assert maxmean<1e-10 and maxcov<1e-10
        path=saved/f'{kind}_spec.npz'
        spec.save(path)
        restored=ModelSpec.load(path)
        restored_posterior=restored.posterior(rows,labels)
        assert np.max(abs(restored_posterior.mean-mean))<1e-12
        assert np.max(abs(restored_posterior.cov-cov))<1e-12
        assert spec.initial.shape==(240,)
        assert np.isfinite(posterior.mean).all()
        assert np.linalg.eigvalsh(posterior.cov).min()>-1e-9
        cases.append(dict(kind=kind,steps=10,initial_mean_error=initial_mean_error,
            initial_covariance_error=initial_cov_error,sequential_reference_mean_max_error=maxmean,
            sequential_reference_covariance_max_error=maxcov,serialization_replay_exact=True,
            working_noise=spec.noise,latent_covariance_min_eigenvalue=float(np.linalg.eigvalsh(posterior.cov).min())))
        if kind=='P1':
            # Compare to the original Gaussian P2 actor's historical objective.
            legacy=GaussianTemplate(data,I,'P2',I[:54],variant='N1',alpha=100.)
            legacy_pred=legacy.X@legacy.mean.reshape(legacy.question_dimension,legacy.action_dimension)@legacy.B.T
            initial=spec.posterior(I,data['Y'][I])
            error=float(np.max(abs(initial.mean-legacy_pred)))
            design_error=float(np.max(abs(spec.parameters['design']-legacy.X)))
            assert error<1e-10 and design_error<1e-10
            cases[-1]['legacy_P2_initial_prediction_max_error']=error
            cases[-1]['legacy_P2_question_design_max_error']=design_error
    result=dict(status='PASS',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        cases=cases,units='one question three-repeat mean, shared20 output package',
        covariance_scope='Working-model internal covariance, no empirical coverage assertion',
        historical_actor_preserved='P1 is original P2 PCA16 Ridge, rawalpha100 preserved at meanalpha100/3',
        score_access='Test receives I labels and the explicitly purchased first10 U packages. E scores unused in fitting and checks.')
    (WORK/'public/theory/ACTUAL_MODEL_OPERATOR_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
