"""Independent direct normal-equation reconstruction of E7-B artifacts."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import numpy as np
from scipy.spatial.distance import pdist,cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from e7_common import ROOT,WORK,PRIVATE,PUBLIC,load_train,splits,sha,put
from e7_models import ModelSpec,build_spec


def main():
    data,_=load_train()
    output=PUBLIC/'b'
    folder=PRIVATE/'b'
    checks=0
    errors=[]
    def equal(a,b,label,tol=1e-10):
        nonlocal checks
        a=np.asarray(a);b=np.asarray(b)
        assert a.shape==b.shape,(label,a.shape,b.shape)
        error=float(np.max(abs(a-b))) if a.size else 0.
        checks+=a.size
        assert error<tol,(label,error)
        errors.append(error)
    def selected(q):
        return np.argmax(q>=q.max(axis=-1,keepdims=True)-1e-10,axis=-1)
    def kernel(x,y,sigma):
        return np.exp(-cdist(x,y,'sqeuclidean')/(2*sigma**2))
    def cv_for(I,y,seed):
        scores=np.zeros((3,len(I)))
        for fit,held in StratifiedKFold(3,shuffle=True,random_state=seed).split(I,data['task'][I]):
            xx=data['E'][I[fit]]
            sigma=float(np.median(pdist(xx)[pdist(xx)>0]))
            mean=y[fit].mean(axis=0)
            gram=kernel(xx,xx,sigma)
            cross=kernel(data['E'][I[held]],xx,sigma)
            for j,lam in enumerate([.1,1,10]):
                prediction=mean+cross@np.linalg.solve(gram+lam*np.eye(len(fit)),y[fit]-mean)
                scores[j,held]=y[held,selected(prediction)]
        best=np.flatnonzero(scores.mean(axis=1)>=scores.mean(axis=1).max()-1e-12)[-1]
        return float([.1,1,10][best]),scores
    main_scores=[];perm_scores=[];alltasks=[];allfold=[]
    main_reconstructed=0
    permutation_reconstructed=0
    for f in range(5):
        role=splits(f);I,U,E=role['I'],role['U'],role['E']
        assert not set(E)&(set(I)|set(U))
        receipt=json.loads((folder/f'FROZEN_B_f{f}.json').read_text())
        assert receipt['predictions_sha256']==sha(folder/f'FROZEN_B_f{f}.npz')
        with np.load(folder/f'EVALUATED_B_f{f}.npz') as a:
            actual=a['Y'];main_pred=a['main_predictions'];perm_pred=a['permutation_predictions']
        d=int(receipt['default'])
        expected_default=int(selected(data['Y'][I].mean(axis=(0,1))))
        assert d==expected_default
        scalar_main=np.zeros((3,4,120))
        scalar_perm=np.zeros((20,4,120))
        for j,kind in enumerate(['P1','P2','P3']):
            spec=ModelSpec.load(folder/f'SPEC_f{f}_{kind}.npz')
            if kind=='P1':
                pca=PCA(n_components=16,svd_solver='full').fit(data['E'][I])
                scaler=StandardScaler().fit(pca.transform(data['E'][I]))
                design=np.column_stack((np.ones(600),scaler.transform(pca.transform(data['E']))))
                equal(design,spec.parameters['design'],f'PCA/scaler{f}')
            elif kind=='P2':
                design=np.column_stack((np.ones(600),data['E']))
                equal(design,spec.parameters['design'],f'fullvector{f}')
            else:
                sigma=float(np.median(pdist(data['E'][I])[pdist(data['E'][I])>0]))
                equal([sigma],[spec.parameters['bandwidth']],f'bandwidth{f}')
                lam,scores=cv_for(I,data['Y'][I].mean(axis=1),20261112+f)
                assert lam==spec.noise
                equal(scores,np.asarray(spec.metadata['lambda_selection']['question_heldout_scores']),f'CV{f}')
            for k,n in enumerate([240,300,360,480]):
                H=np.r_[I,U[:n-240]]
                y=data['Y'][H].mean(axis=1)
                if kind in ['P1','P2']:
                    coef=np.linalg.solve(design[H].T@design[H]+(100/3)*np.eye(design.shape[1]),design[H].T@y)
                    predict=design[E]@coef
                else:
                    offset=data['Y'][I].mean(axis=(0,1))
                    gram=kernel(data['E'][H],data['E'][H],sigma)
                    predict=offset+kernel(data['E'][E],data['E'][H],sigma)@np.linalg.solve(gram+spec.noise*np.eye(n),y-offset)
                equal(predict,main_pred[j,k],f'main{f,j,n}')
                assert np.array_equal(selected(predict),selected(main_pred[j,k]))
                scalar_main[j,k]=actual[np.arange(120),:,selected(predict)].mean(axis=1)
                main_reconstructed+=1
        sigma=float(np.median(pdist(data['E'][I])[pdist(data['E'][I])>0]))
        sem=lambda ids:np.where(data['task'][ids]=='gsm8k','gsm8k',data['sub'][ids])
        for p,metadata in enumerate(receipt['permutation_specs']):
            sourceI=np.asarray(metadata['initial_reward_source'])
            assert set(sourceI)==set(I) and np.array_equal(sem(sourceI),sem(I))
            yi=data['Y'][sourceI].mean(axis=1)
            lam,scores=cv_for(I,yi,20261112+f)
            assert lam==metadata['lambda_value']
            equal(scores,np.asarray(metadata['CV']['question_heldout_scores']),f'permCV{f,p}')
            mu=yi.mean(axis=0)
            for k,n in enumerate([240,300,360,480]):
                addon=U[:n-240]
                sourceU=np.asarray(receipt['permutation_U_sources'][p][k],dtype=int)
                assert set(sourceU)==set(addon) and np.array_equal(sem(sourceU),sem(addon))
                H=np.r_[I,addon]
                sources=np.r_[sourceI,sourceU]
                y=data['Y'][sources].mean(axis=1)
                gram=kernel(data['E'][H],data['E'][H],sigma)
                prediction=mu+kernel(data['E'][E],data['E'][H],sigma)@np.linalg.solve(gram+lam*np.eye(n),y-mu)
                equal(prediction,perm_pred[p,k],f'perm{f,p,n}')
                assert np.array_equal(selected(prediction),selected(perm_pred[p,k]))
                scalar_perm[p,k]=actual[np.arange(120),:,selected(prediction)].mean(axis=1)
                permutation_reconstructed+=1
        if f==0:
            poisoned=data['Y'].copy()
            poisoned[E]=np.nan
            for kind in ['P1','P2','P3']:
                altered=build_spec(data['E'],I,poisoned[I],kind,tasks=data['task'],cv_seed=20261112)
                normal=ModelSpec.load(folder/f'SPEC_f0_{kind}.npz')
                equal(altered.K,normal.K,'E-NaN-kernel'+kind)
                equal(altered.offset,normal.offset,'E-NaN-offset'+kind)
                assert altered.noise==normal.noise
                observed=np.r_[I,U]
                prediction=altered.posterior(observed,poisoned[observed]).predict(E)
                equal(prediction,main_pred[['P1','P2','P3'].index(kind),3],'E-NaN-fullfit'+kind)
        main_scores.append(scalar_main);perm_scores.append(scalar_perm);alltasks.append(data['task'][E]);allfold.extend([f]*120)
    scores=np.concatenate(main_scores,axis=2)
    shuf=np.concatenate(perm_scores,axis=2)
    task=np.concatenate(alltasks);fold=np.asarray(allfold)
    learning=scores[2,3]-scores[2,0]
    content=scores[2,3]-shuf[:,3].mean(axis=0)
    rng=np.random.default_rng(20261112)
    draws=[]
    for _ in range(5000):
        ix=np.concatenate([rng.choice(ids,len(ids),replace=True)
            for f in range(5) for t in ['gsm8k','bbh'] for ids in [np.flatnonzero((fold==f)&(task==t))]])
        draws.append([learning[ix].mean(),content[ix].mean()])
    with np.load(folder/'PRIMARY_B_COMPARISONS.npz') as a:
        equal(learning,a['learning_contribution'],'primarylearning')
        equal(content,a['content_contribution'],'primarycontent')
        equal(np.asarray(draws)[:,0],a['learning_bootstrap'],'learningbootstrap')
        equal(np.asarray(draws)[:,1],a['content_bootstrap'],'contentbootstrap')
    report=json.loads((output/'E7_B_RESULTS.json').read_text())
    for j,kind in enumerate(['P1','P2','P3']):
        for k,n in enumerate([240,300,360,480]):
            equal([scores[j,k].mean()],[report['configurations'][kind][str(n)]['metrics']['accuracy']],'publicaccuracy')
    result=dict(status='PASS',scalar_checks=int(checks),maximum_reconstruction_error=max(errors),
        independently_reconstructed_main_fits=main_reconstructed,
        independently_reconstructed_permutation_fits=permutation_reconstructed,
        main_CV_fits=5,permutation_CV_fits=100,semantic_permutation_permission=True,
        original_question_bootstrap_repetitions=5000,E_NaN_counterfactual_fold=0,
        E_NaN_all3_model_geometries_offsets_and480_predictions_pass=True,
        raw_or_mean_regularization='Original raw3 alpha100 equals mean alpha100/3',
        errors_are_original_question_conditional_development=True,
        results_sha256=sha(output/'E7_B_RESULTS.json'),source_sha256=sha(__file__),
        software_control_status=json.loads((output/'E7_B_SOFTWARE_CONTROLS.json').read_text())['status'])
    put(output/'E7_B_INDEPENDENT_AUDIT.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
