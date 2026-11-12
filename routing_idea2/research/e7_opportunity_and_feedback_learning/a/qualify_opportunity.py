"""Outcome-free software qualification for the E7-A candidate diagnostics."""
from run_opportunity import *


def main():
    rng=np.random.default_rng(20261112751)
    checks=0
    maximum_error=0.
    for trial in range(50):
        n=8
        reward=rng.integers(0,2,size=(n,3,20)).astype(float)
        score=rng.normal(size=(n,20))
        base=int(rng.integers(20))
        order=ranking(score,base)
        assert np.array_equal(order[:,0],argmax(score,base))
        assert np.all(np.sort(np.column_stack((np.full(n,base),order)),axis=1)==np.arange(20))
        previous=np.zeros((n,20),dtype=bool)
        for m in MS:
            mask=make_mask(order,base,m)
            assert np.all(mask.sum(axis=1)==m+1) and np.all(mask[:,base])
            assert np.all(previous<=mask)
            previous=mask
            got,choice,uniform,counts=cross_repeat(reward,mask)
            for q in range(n):
                for held in range(3):
                    train=reward[q,[g for g in range(3) if g!=held]].mean(axis=0)
                    ids=np.flatnonzero(mask[q])
                    best=ids[train[ids]>=train[ids].max()-TOLERANCE]
                    assert choice[q,held]==best[0]
                    assert got[q,held]==reward[q,held,best[0]]
                    err=abs(uniform[q,held]-reward[q,held,best].mean())
                    maximum_error=max(maximum_error,float(err))
                    assert err<1e-14 and counts[q,held]==len(best)
                    checks+=4
            for held in range(3):
                modified=reward.copy()
                modified[:,held]=1-modified[:,held]
                alternate=cross_repeat(modified,mask)
                assert np.array_equal(alternate[1][:,held],choice[:,held])
                assert np.array_equal(alternate[3][:,held],counts[:,held])
                checks+=2
            repeated=np.repeat(mask[:,None],5,axis=1)
            got_random=cross_repeat(reward,repeated)
            assert np.array_equal(got_random[0],np.repeat(got[:,None],5,axis=1))
            assert np.array_equal(got_random[1],np.repeat(choice[:,None],5,axis=1))
            checks+=4
        checks+=2
    PUBLIC.mkdir(parents=True,exist_ok=True)
    payload=dict(status='PASS',scalar_or_array_assertions=checks,
        synthetic_trials=50,questions_per_trial=8,generation_repeats=3,
        maximum_uniform_tie_expectation_error=maximum_error,
        checks=['original P2 argmax equivalence','default inclusion and nested candidate sizes',
                'independent exhaustive leave-one-repeat evaluation','uniform expectation across tied training optima',
                'held-out reward perturbation cannot change its selection or tie count',
                'random-set axis and deterministic-set axis agree'],uses_real_rewards=False)
    put(PUBLIC/'E7_A_QUALIFICATION.json',payload)
    print(json.dumps(payload),flush=True)


if __name__=='__main__':
    main()
