"""Privileged/predicted metadata diagnostics. P2 only, never D candidate selection."""
from e4_common import *
from run_binary import worker
from concurrent.futures import ProcessPoolExecutor


if __name__=='__main__':
    jobs=[(f,'P2',mode) for f in range(5) for mode in ('subtask-oracle','subtask-predicted')]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(worker,jobs):
            print(json.dumps(result),flush=True)
    put(PRIVATE/'METADATA_SENSITIVITY_DONE.json',dict(groups=10,predictor='P2',
        role='mechanism analysis only, privileged oracle excluded from deployment candidates',new_llm_calls=0))
