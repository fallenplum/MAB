"""Read-only sensitivity of synthetic plug-in selection to numerical zero/ties."""
from e5_common import *
from time import perf_counter


def selected_values(estimate,truth,tolerance=0.,positive_ties=False):
    maximum=estimate.max(axis=1)
    index=np.argmax(estimate>=maximum[:,None]-tolerance,axis=1) if positive_ties else np.argmax(estimate,axis=1)
    selected=np.where(maximum>tolerance,index,-1)
    gain=np.where(selected>=0,truth[np.maximum(selected,0)],0.)
    return selected,gain


def main():
    start=perf_counter()
    path=PUBLIC/'ALLOCATION_SYNTHETIC_RESULTS.json'
    original_report_hash=sha(path)
    report=json.loads(path.read_text())
    records=[]
    pairs=[]
    source_hashes={}
    changes=0
    for row in report['cases']:
        name=f"{row['family']}_ann{int(row['annotation_fraction']*100)}_exec{int(row['execution_fraction']*100)}"
        npz=PRIVATE/'synthetic_allocation/cases'/f'{name}.npz'
        source_hashes[npz.name]=sha(npz)
        with np.load(npz) as a:
            estimate=a['estimates'];truth=a['true_gains'];methods=a['method_names'].astype(str).tolist()
        best=max(0.,float(truth.max()))
        old=[];zero=[];full=[]
        private={}
        for m,method in enumerate(methods):
            original,oldgain=selected_values(estimate[:,m],truth)
            zerotol,zerogain=selected_values(estimate[:,m],truth,1e-12)
            fulltol,fullgain=selected_values(estimate[:,m],truth,1e-12,True)
            maxscore=estimate[:,m].max(1)
            count=int(np.sum((maxscore>0)&(maxscore<=1e-12)))
            changes+=count
            mse=np.mean((estimate[:,m]-truth)**2,axis=0)
            assert np.max(abs(mse-np.asarray(row['methods'][method]['MSE'])))<1e-12
            assert abs(np.mean(best-oldgain)-row['methods'][method]['plugin_policy_selection']['mean_value_loss'])<1e-12
            old.append(best-oldgain);zero.append(best-zerogain);full.append(best-fullgain)
            private[method+'_original_selected']=original
            private[method+'_zero_tolerance_selected']=zerotol
            private[method+'_full_tolerance_selected']=fulltol
            records.append(dict(scenario=name,family=row['family'],method=method,
                near_zero_positive_original_estimates=count,
                zero_threshold_decision_changes=int(np.sum(original!=zerotol)),
                full_threshold_and_positive_tie_changes=int(np.sum(original!=fulltol)),
                original_mean_policy_value_loss=float(np.mean(best-oldgain)),
                zero_tolerance_mean_policy_value_loss=float(np.mean(best-zerogain)),
                full_tolerance_mean_policy_value_loss=float(np.mean(best-fullgain)),
                zero_tolerance_minus_original_mean_true_gain=float(np.mean(zerogain-oldgain)),
                full_tolerance_minus_original_mean_true_gain=float(np.mean(fullgain-oldgain)),
                worst_gate_estimator_MSE=float(mse.max()),estimator_MSE_changed=False,
                maximum_positive_near_zero_estimate=float(maxscore[(maxscore>0)&(maxscore<=1e-12)].max()) if count else None))
        old,zero,full=np.asarray(old),np.asarray(zero),np.asarray(full)
        est=methods.index('Estimated-two-resource');uniform=methods.index('Uniform')
        contrast={}
        for label,loss in [('original',old),('zero_tolerance',zero),('full_tolerance',full)]:
            delta=loss[est]-loss[uniform]
            se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
            contrast[label]=dict(estimated_minus_uniform_mean_value_loss=float(delta.mean()),
                paired_MC_standard_error=se,paired_normal_MC_ci95=[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)])
        pairs.append(dict(scenario=name,family=row['family'],comparisons=contrast,
            zero_tolerance_change_in_contrast=contrast['zero_tolerance']['estimated_minus_uniform_mean_value_loss']-contrast['original']['estimated_minus_uniform_mean_value_loss'],
            full_tolerance_change_in_contrast=contrast['full_tolerance']['estimated_minus_uniform_mean_value_loss']-contrast['original']['estimated_minus_uniform_mean_value_loss'],
            estimated_worst_MSE=row['methods']['Estimated-two-resource']['worst_gate_MSE'],
            uniform_worst_MSE=row['methods']['Uniform']['worst_gate_MSE'],
            worst_MSE_comparison_unchanged=True))
        target=PRIVATE/'selection_numerical_sensitivity'/f'{name}.npz'
        target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,**private)
        assert sha(npz)==source_hashes[npz.name]
    assert sha(path)==original_report_hash
    sub=[row for row in pairs if row['family']=='nonnested']
    summary=dict(
        positive_near_zero_estimates=changes,total_scenario_method_MC_draws=len(report['cases'])*7*2000,
        nonnested_scenarios=9,
        nonnested_estimated_selection_loss_worse_than_uniform={label:sum(row['comparisons'][label]['estimated_minus_uniform_mean_value_loss']>1e-12 for row in sub) for label in ('original','zero_tolerance','full_tolerance')},
        estimated_worst_MSE_better_than_uniform=sum(row['estimated_worst_MSE']<row['uniform_worst_MSE']-1e-12 for row in pairs),
        total_MSE_comparisons=len(pairs),MSE_comparison_changed_count=0,
        maximum_absolute_zero_tolerance_change_in_estimated_vs_uniform_contrast=max(abs(row['zero_tolerance_change_in_contrast']) for row in pairs),
        maximum_absolute_full_tolerance_change_in_estimated_vs_uniform_contrast=max(abs(row['full_tolerance_change_in_contrast']) for row in pairs))
    result=dict(schema='e5_allocation_selection_numerical_sensitivity_v1',status='COMPLETE',
        tolerance=1e-12,summary=summary,method_rows=records,paired_policy_comparisons=pairs,
        original_ALLOCATION_SYNTHETIC_RESULTS_sha256=original_report_hash,source_NPZ_sha256=source_hashes,
        code_sha256=sha(Path(__file__)),
        original_results_modified=False,LLM_calls=0,wall_s=perf_counter()-start,
        scope='Frozen synthetic estimates only. Original strict-positive rule retained. Numerical zero/tie sensitivity is separate from the fixed-horizon certificate and Text-only main gate.')
    put(PUBLIC/'ALLOCATION_SELECTION_NUMERICAL_SENSITIVITY.json',result)
    print(json.dumps(dict(status='COMPLETE',**summary)),flush=True)


if __name__=='__main__':main()
