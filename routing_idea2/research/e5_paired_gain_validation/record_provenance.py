from e5_common import *
import platform
import scipy
import sklearn


def main():
    data,refs=load_train()
    pdf=PRIVATE/'REFERENCE_MANUSCRIPT.pdf'
    task=PRIVATE/'E5_USER_TASK.txt'
    assert sha(pdf)=='b2c9c7f9bf09a3183f4907eb7daedd3a7537dc9f6460f62ccfc246e525bb9e99'
    old_manifest=json.loads((E4/'PUBLIC_DELIVERY_MANIFEST.json').read_text())
    for name,record in old_manifest['files'].items():
        assert sha(E4/name)==record['sha256'],name
    # Preserve exact original-question roles locally. Public records carry identities only.
    role_arrays={}
    counts=[]
    for fold in split_roles(data):
        f=fold['fold']
        for name in ('F','G','E','feedback_G20'):role_arrays[f'f{f}_{name}']=fold[name]
        counts.append(dict(fold=f,F=len(fold['F']),G=len(fold['G']),E=len(fold['E']),
            original_question_overlap=0,source_indices_sha256=hashlib.sha256(fold['F'].tobytes()+fold['G'].tobytes()+fold['E'].tobytes()).hexdigest()))
    np.savez_compressed(PRIVATE/'FGE_SPLITS_PRIVATE.npz',**role_arrays)
    sanitized={name:dict(path=str(Path(record['path']).relative_to(ROOT)),sha256=record['sha256'])
               for name,record in refs.items() if isinstance(record,dict) and 'path' in record}
    dependencies={}
    for directory,names in [(E4,['e4_common.py','e4_gaussian.py','e4_binary.py','e4_metadata.py','run_binary.py','e4_metrics.py']),
                            (ROOT/'research/decision_acquisition_e3/public',['assets.py','predictors.py'])]:
        for name in names:dependencies[str((directory/name).relative_to(ROOT))]=sha(directory/name)
    put(PUBLIC/'REFERENCE_AND_INPUTS.json',dict(status='VERIFIED',
        manuscript=dict(title='From Hindsight Oracles to Reliable Ensemble Routing',pages=29,
            sha256=sha(pdf),bytes=pdf.stat().st_size,local_private_file=pdf.name,
            sections={'A':'Section4, AppendixB/C','B':'Section4, AppendixC',
                      'C':'Section5, AppendixG','D':'Section6, AppendixD/F.1','E':'Section6.1, AppendixE/F'}),
        task=dict(sha256=sha(task),local_private_file=task.name),
        assets=sanitized,identity=refs['identity'],prior_dependencies=dependencies,
        E4_public_manifest_sha256=sha(E4/'PUBLIC_DELIVERY_MANIFEST.json'),
        E4_manifest_hashed_files_verified=len(old_manifest['files']),
        folds=counts,private_role_file_sha256=sha(PRIVATE/'FGE_SPLITS_PRIVATE.npz'),
        base_commit='b5fa3317c77b1256037d734c93a001ecb19bfa28',
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                     BLAS_threads=1,compute='CPU'),
        data_boundaries=dict(decoded_reward_splits=['selector_train'],validation200=False,fresh400=False,sealed200=False,
                             evidence_scope='source, explicit perturbation qualification and saved artifacts; no OS-wide nonaccess claim'),
        physical_new_llm_calls=0))
    print(json.dumps(dict(status='VERIFIED',E4_hashed_files=len(old_manifest['files']),question_count=600)))


if __name__=='__main__':main()
