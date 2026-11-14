"""Measure frozen PCA16 plus linear routing CPU latency, without fitting."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import psutil

ROOT=Path(r'E:\语言模型组合路由\research\e10_input_prefill_matching_audit')
sys.path.insert(0,str(ROOT.parent/'e9_consistent_cost_sensitive_learning/public/code'))
from csce_operators import semantic_argmax
from e9_common import ACTION_KEYS


def main():
    features=np.load(ROOT/'private/features/PREFILL_FEATURES_FROZEN.npz',allow_pickle=False)['full']
    frozen=np.load(ROOT/'private/ALL_E10_PREDICTIONS_FROZEN.npz',allow_pickle=False)
    durations=[];maximum_error=0.
    cpu_started=time.process_time();started=time.perf_counter()
    for fold in range(5):
        model=dict(np.load(ROOT/f'private/models/fold{fold}/R_full/MODEL.npz',allow_pickle=False))
        rows=np.flatnonzero(frozen['fold']==fold)
        for repeat in range(10):
            for row in rows:
                before=time.perf_counter()
                value=features[row].astype(np.float64)
                projected=(value-model['pca_mean'])@model['pca_components'].T
                standardized=(projected-model['scaler_mean'])/model['scaler_scale']
                score=np.r_[1.,standardized]@model['coefficient']
                action=semantic_argmax(score[None,:],ACTION_KEYS)[0]
                durations.append(time.perf_counter()-before)
                maximum_error=max(maximum_error,float(abs(score-frozen['scores__R_full'][row]).max()))
                assert action==frozen['actions__R_full'][row]
    assert maximum_error<1e-12 and len(durations)==6000
    memory=psutil.Process().memory_info()
    record=dict(status='PASS_FROZEN_INFERENCE_ONLY',created_utc=datetime.now(timezone.utc).isoformat(),
        independent_questions=600,repeats=10,timed_inference_events=6000,new_fits=0,new_prefill=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='float32-to-float64 plus frozen PCA16/standardizer/penalized-linear-head score and semantic argmax, excludes feature extraction and asset reads',
        cpu_seconds=time.process_time()-cpu_started,whole_profile_wall_seconds=time.perf_counter()-started,
        latency_seconds=dict(zip(('min','p50','p95','max'),np.quantile(durations,[0,.5,.95,1]).tolist())),
        mean_latency_seconds=float(np.mean(durations)),maximum_restoration_error=maximum_error,
        process_peak_ram_bytes=getattr(memory,'peak_wset',None),peak_scope='this standalone profiling process, includes imports and feature/model arrays')
    output=ROOT/'public/INFERENCE_CPU_PROFILE.json'
    assert not output.exists()
    output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':
    main()
