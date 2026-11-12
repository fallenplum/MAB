"""Independent truth-process invocation of the predeclared public supplements."""
from __future__ import annotations

import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from run_b1 import BASE, frozen_inputs, load_npz, put, sha, distribution, wilson
from finite_pool_supplements import fixed_size_asi_supplement, active_testing_supplement


def main(output_directory=None):
    if output_directory is None:
        output_directory=BASE/'private/supplement_reproductions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination=Path(output_directory)
    if destination.resolve()==BASE.resolve() or destination.exists():
        raise FileExistsError('Supplement replay requires a new output directory')
    destination.mkdir(parents=True)
    inputs,_=frozen_inputs()
    oracle=load_npz(BASE/'private/evaluator/ORACLE_PAIRED.npz')
    designs=load_npz(BASE/'private/design/FROZEN_DESIGNS.npz')
    summary=json.loads((BASE/'public/DESIGN_SUMMARY.json').read_text(encoding='utf-8'))
    common=summary['primary__A1_ASI']
    native=BASE/'private/public_methods/upstream'
    dependencies=BASE/'private/public_methods/dependencies'
    repeats=2000
    outputs={}
    specifications={
        'FixedSize_ASI':dict(function=fixed_size_asi_supplement,
            arguments=dict(bernoulli_p=designs['primary__A1_ASI__p'],
                source_root=native/'cps-sampling',dependency_root=dependencies)),
        'ActiveTesting_LURE':dict(function=active_testing_supplement,arguments={})}
    for name,spec in specifications.items():
        started=time.perf_counter()
        result=spec['function'](d=oracle['d'],m=inputs['m'],predicted_cost=inputs['cost_prediction'],
            active=inputs['active'],answer_budget=common['answer_cap'],
            execution_budget=common['predicted_token_cap'],repeats=repeats,**spec['arguments'])
        samples=result['samples']; estimates=result['estimates']; intervals=result['intervals']
        truth=float(oracle['d'].mean()); error=estimates-truth
        coverage=(intervals[:,0]<=truth)&(intervals[:,1]>=truth)
        resources={}
        for key in ['gold','member_calls','aggregator_calls','tokens','input_tokens','output_tokens','source_model_wall_s']:
            resources[key]=distribution(oracle[key][samples].sum(axis=1))
        resources['predicted_tokens']=distribution(inputs['cost_prediction'][samples].sum(axis=1))
        serial={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
        target=destination/'private/supplements'/ (name+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,**serial)
        outputs[name]=dict(status='COMPLETE_TASK_ADAPTER',method=result['method'],trials=repeats,
            original_question_clusters=200,fixed_active_sample_size=result['n'],truth=truth,
            bias=float(error.mean()),RMSE=float(np.sqrt(np.mean(error**2))),
            interval_coverage=float(coverage.mean()),coverage_MC95=wilson(int(coverage.sum()),repeats),
            mean_interval_width=float(np.mean(intervals[:,1]-intervals[:,0])),
            inference=result['diagnostics']['interval'],resources=resources,
            diagnostics=result['diagnostics'],private_artifact_sha256=sha(target),
            wall_seconds=time.perf_counter()-started)
        print(json.dumps(dict(method=name,RMSE=outputs[name]['RMSE'],coverage=outputs[name]['interval_coverage'],
            n=result['n'],seconds=outputs[name]['wall_seconds'])),flush=True)
    put(destination/'public/SUPPLEMENTARY_FINITE_POOL_RESULTS.json',outputs)
    original=json.loads((BASE/'public/SUPPLEMENTARY_FINITE_POOL_RESULTS.json').read_text(encoding='utf-8'))
    keys=['truth','bias','RMSE','interval_coverage','mean_interval_width','fixed_active_sample_size','resources']
    for method,result in outputs.items():
        if any(result[key]!=original[method][key] for key in keys):
            raise AssertionError('Supplement replay differs from the frozen metrics')
    put(destination/'REPRODUCTION_AUDIT.json',dict(status='PASS',
        original_outputs_overwritten=False,both_supplements_reproduced=True,
        numerical_metrics_and_resource_summaries_identical=True))
    print(json.dumps(dict(status='PASS_SUPPLEMENTS_REPRODUCED',directory=str(destination))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output')
    main(parser.parse_args().output)
