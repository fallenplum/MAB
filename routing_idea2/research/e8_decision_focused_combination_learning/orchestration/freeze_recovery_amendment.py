"""Seal a pre-evaluation numerical backend amendment, preserving all fits."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

BASE=Path(r'E:\语言模型组合路由\research\e8_decision_focused_combination_learning')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output=BASE/'public/E8_EXECUTION_AMENDMENT_01.json'
    if output.exists():raise FileExistsError('A numerical amendment already exists')
    qualifications=[BASE/'public/qualification/CLARABEL_RECOVERY_QUALIFICATION.json',
                    BASE/'public/qualification/CLARABEL_INDEPENDENT_REVIEW.json']
    for path in qualifications:
        if not str(json.loads(path.read_text(encoding='utf-8')).get('status','')).startswith('PASS'):
            raise AssertionError('Equivalent backend qualification is incomplete')
    original=json.loads((BASE/'private/E8_SOURCE_LOCK.json').read_text(encoding='utf-8'))
    sources=[BASE/'public/training/e8_recovery_v1.py',BASE/'public/training/run_e8_recovery_v1.py']
    lock=dict(schema='E8_equivalent_numerical_backend_source_lock_v1',
        files={**original['files'],**{str(path):sha(path) for path in sources}})
    lock_path=BASE/'private/E8_RECOVERY_SOURCE_LOCK.json'
    lock_path.write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    receipts=list((BASE/'private/training').glob('fold*/*/FROZEN_MODEL_RECEIPT.json'))
    if len(receipts)!=49:raise AssertionError('Expected exactly49 prior frozen models')
    value=dict(schema='E8_pre_evaluation_numerical_execution_amendment_v1',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        original_protocol_sha256=sha(BASE/'public/E8_PROTOCOL.json'),
        reason='OSQP returned solved inaccurate at fixed200000 iterations for a uniform-true-tie inner fit',
        failure=dict(inner_questions=320,outer_fold=0,true_tie_rule='uniform',lambda_multiplier=1,
            original_primal_violation=2.16308e-8,original_declared_eps=1e-8,
            original_failure_log_preserved=True),
        numerical_recovery=dict(solver='Clarabel0.11.1',exact_problem='same P,q,A,b epigraph convex SPO+ QP',
            tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=1000,max_threads=1,
            binding='explicit single solver-function proxy in a new recovery entry point',
            scope='SPO+ fits in the21 previously unsealed variants only',
            all_other_training_function_objects_unchanged=True),
        preserved_models=49,remaining_models=21,
        preserved_artifact_policy='reuse and verify every prior receipt and its model/prediction hashes',
        scientific_parameters='same historical inputs, reward labels, linear capacity, losses, lambda grid and selected parameters, folds and seeds',
        prior_source_files_unchanged=True,prior_protocol_retained=True,
        outer_evaluation_rewards_read=False,no_new_physical_LLM_calls=True,
        limitations='uniform true-tie sensitivity shows near-zero score margins and solver-dependent exact argmax; interpreted as numerical sensitivity',
        source_lock_sha256=sha(lock_path),source_hashes={str(p.relative_to(BASE)):sha(p) for p in sources},
        qualification_sha256={str(p.relative_to(BASE)):sha(p) for p in qualifications})
    output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='NUMERICAL_RECOVERY_FROZEN',amendment_sha256=sha(output),
        recovery_lock_sha256=sha(lock_path),preserved=49,remaining=21)))


if __name__=='__main__':main()
