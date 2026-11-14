"""Metadata-only adapter for the frozen evaluator's GSM8K subtask interface."""
import hashlib
import json
from pathlib import Path

import numpy as np
import e10_evaluate as frozen

ROOT=frozen.ROOT


def main():
    receipt=json.loads((ROOT/'private/EVALUATION_ADAPTER_SOURCE_FREEZE.json').read_text())
    assert receipt['adapter_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert receipt['core_sha256']==hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest()
    original=frozen.decompose_policies

    def canonical_metadata(rewards,policies,task,subtask,outerfold,**kwargs):
        subtask=np.where(np.asarray(task)=='gsm8k','gsm8k',np.asarray(subtask))
        return original(rewards,policies,task,subtask,outerfold,**kwargs)

    frozen.decompose_policies=canonical_metadata
    frozen.main()


if __name__=='__main__':
    main()
