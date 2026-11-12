"""Independent reconstruction of frozen E7-A collections and paired statistics."""
from run_opportunity import *
from math import comb


def main():
    data,_=load_train()
    checks=0
    max_error=0.
    def equal(actual,expected,tolerance=1e-12):
        nonlocal checks,max_error
        actual=np.asarray(actual)
        expected=np.asarray(expected)
        assert actual.shape==expected.shape
        error=float(np.max(abs(actual.astype(float)-expected.astype(float)))) if actual.size else 0.
        max_error=max(max_error,error)
        assert error<=tolerance,error
        checks+=actual.size
    with np.load(PRIVATE/'OPPORTUNITY_PAIRED.npz') as a:
        saved={key:a[key] for key in a.files}
    with np.load(PRIVATE/'OPPORTUNITY_BOOTSTRAP.npz') as a:
        original_bootstrap=a['bootstrap']
    protocol_sha=sha(WORK/'public/E7_PROTOCOL.json')
    for f in range(5):
        receipt=json.loads((PRIVATE/f'FROZEN_CANDIDATES_f{f}.json').read_text())
        assert receipt['protocol_sha256']==protocol_sha
        assert receipt['candidate_sha256']==sha(PRIVATE/f'CANDIDATES_f{f}.npz')
        with np.load(PRIVATE/f'CANDIDATES_f{f}.npz') as a:
            records={key:a[key] for key in a.files}
        rows=records['E']
        b=int(records['default'])
        reward=data['Y'][rows]
        equal(rows,saved['rows'][f*120:(f+1)*120])
        equal(reward,saved['Y'][f*120:(f+1)*120])
        rng=np.random.default_rng(np.random.SeedSequence([SEED,f,751]))
        choices=np.array([s for s in range(20) if s!=b])
        reproduced=np.asarray([[rng.permutation(choices) for _ in range(200)] for q in rows])
        equal(reproduced,records['random_rank'])
        for group,key in [('P2','p2_masks'),('Fixed','fixed_masks'),('Random','random_masks')]:
            collections=records[key]
            for size_index,mask in enumerate(collections):
                if group!='Random':
                    mask=mask[:,None,:]
                nr=mask.shape[1]
                for held in range(3):
                    train=reward[:,[j for j in range(3) if j!=held]].sum(axis=1)/2
                    train=np.repeat(train,nr,axis=0)
                    candidate_mask=mask.reshape(-1,20)
                    score=np.where(candidate_mask,train,-100.)
                    tied=candidate_mask&(score>=score.max(axis=1,keepdims=True)-1e-10)
                    expected=np.min(np.where(tied,np.arange(20),20),axis=1)
                    test=np.repeat(reward[:,held],nr,axis=0)
                    expected_reward=test[np.arange(len(test)),expected].reshape(len(rows),nr)
                    expected_ties=tied.sum(axis=1).reshape(len(rows),nr)
                    actual=saved[group+'_choices'][size_index,f*120:(f+1)*120]
                    actual_ties=saved[group+'_tie_counts'][size_index,f*120:(f+1)*120]
                    if group=='Random':
                        equal(actual[:,:,held],expected.reshape(len(rows),nr))
                        equal(actual_ties[:,:,held],expected_ties)
                        equal(saved['random_draw_scores'][size_index,f*120:(f+1)*120,:,held],expected_reward)
                    else:
                        equal(actual[:,held],expected)
                        equal(actual_ties[:,held],expected_ties[:,0])
                        expected_reward=expected_reward[:,0]
                        li=(0 if group=='P2' else 4)+size_index
                        equal(saved['outcomes'][li,f*120:(f+1)*120,held],expected_reward)
    n=len(saved['rows'])
    rng=np.random.default_rng(np.random.SeedSequence([SEED,752]))
    count=np.zeros((5000,n))
    for rep in range(5000):
        ix=np.concatenate([rng.choice(cell,len(cell),replace=True) for f in range(5) for t in ('gsm8k','bbh')
            for cell in [np.flatnonzero((saved['fold']==f)&(saved['task']==t))]])
        count[rep]=np.bincount(ix,minlength=n)
    acc=count@saved['outcomes'].mean(axis=2).T/n
    base=count@saved['baseline'].mean(axis=1)/n
    full=acc[:,3]
    retention=(count@saved['retained'].sum(axis=2).T)/(count@saved['opportunity'].sum(axis=1))[:,None]
    expected_bootstrap=np.stack([acc,acc-base[:,None],acc-full[:,None],retention],axis=2)
    equal(original_bootstrap,expected_bootstrap)
    opportunity=saved['opportunity']
    counts=saved['Y'].sum(axis=2)
    analytic=[]
    for index,m in enumerate(MS):
        probability=np.asarray([[1-(comb(19-int(k),m)/comb(19,m) if 19-int(k)>=m else 0) for k in row] for row in counts])
        probability=np.where(opportunity,probability,0)
        point=float(probability.sum()/opportunity.sum())
        observed=float(saved['retained'][8+index].sum()/opportunity.sum())
        analytic.append(dict(m=m,analytic_random_opportunity_retention=point,
            archived_200_draw_retention=observed,absolute_difference=abs(point-observed)))
    result=dict(status='PASS',scalar_checks=int(checks),maximum_reconstruction_error=max_error,
        five_frozen_collections_sha_verified=True,random_collections_regenerated_without_rewards=True,
        all_random_and_deterministic_leaveout_choices_reconstructed=True,
        bootstrap_reconstructions=5000,original_questions=600,generation_repeats=3,
        analytic_random_set_coverage=analytic,
        scope='Frozen development candidate diagnosis with original-question paired bootstrap')
    put(PUBLIC/'E7_A_INDEPENDENT_AUDIT.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
