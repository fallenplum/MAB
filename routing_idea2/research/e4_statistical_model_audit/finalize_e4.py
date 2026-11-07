"""Bind completed E4 artifacts to frozen inputs and public byte identities."""
from e4_common import *
import hashlib


def main():
    data,_=load_train()
    splits=e4_splits(data)
    arrays={}
    split_rows=[]
    for fold in splits['folds']:
        f=fold['fold']
        for name in ('history','stream','audit_support','audit_query'):
            arrays[f'f{f}_{name}']=fold[name]
        for n,rows in fold['nested'].items():arrays[f'f{f}_history{n}']=rows
        for order,rows in fold['orders'].items():arrays[f'f{f}_order{order}']=rows
        for order,rows in fold['audit_orders'].items():arrays[f'f{f}_support_order{order}']=rows
        split_rows.append(dict(fold=f,history=480,outer_query=120,controlled_support=20,controlled_query=100,
            history_query_overlap=int(len(np.intersect1d(fold['history'],fold['stream']))),
            support_query_overlap=int(len(np.intersect1d(fold['audit_support'],fold['audit_query']))),
            nested_sizes={str(n):len(v) for n,v in fold['nested'].items()},
            source_indices_sha256=hashlib.sha256(fold['history'].tobytes()+fold['stream'].tobytes()).hexdigest()))
    np.savez_compressed(PRIVATE/'E4_SPLITS_PRIVATE.npz',**arrays)
    put(PUBLIC/'SPLIT_PERMISSION_MANIFEST.json',dict(folds=split_rows,independent_original_questions=600,
        controlled_distinct_query=500,generation_repeats=3,
        private_split_sha256=sha(PRIVATE/'E4_SPLITS_PRIVATE.npz'),
        train_identifiers_sha256=hashlib.sha256('\n'.join(map(str,data['ids'])).encode()).hexdigest(),
        primary='original-question OOF, fitted preprocessing historical only',
        metadata_permission='true BBH IDs privileged except split stratification and declared oracle diagnostic',
        validation_search=False,fresh400_scores=False,sealed200_scores=False,
        limitation='Loader and explicit invariance/manifest tests provide scope evidence. OS-level access logs are unavailable.'))
    required=['01_DATA_QUALITY_AND_STABILITY.md','02_EFFECTIVE_SAMPLE_SIZE.md','03_RIDGE_POSTERIOR_AUDIT.md',
              '04_NOISE_MODEL_ABLATION.md','05_BINARY_REWARD_MODELS.md','06_SUBTASK_METADATA_AUDIT.md',
              '07_FEEDBACK_DECISION_CALIBRATION.md','08_E4_RESEARCH_CONCLUSION.md',
              'E4_EXPERIMENT_CONFIG.json','E4_PROTOCOL_AMENDMENTS.json']
    results=['RAW_AUDIT_PUBLIC.json','LEARNING_CURVE_RESULTS.json','STABILITY_AND_ORACLE.json',
             'POSTERIOR_AUDIT_RESULTS.json','CALIBRATION_AND_INTERVALS.json','CONTROLLED_COST_LEDGER.json',
             'FULL_SERVICE_COST_ACCOUNTS.json','BINARY_COMPARISON_RESULTS.json','BINARY_ONLINE_RESULTS.json',
             'METADATA_SENSITIVITY_RESULTS.json','DECISION_LIVE_RESULTS.json','DECISION_QUALITY_COST_CURVES.json',
             'DECISION_PROXY_CALIBRATION.json','D_PROXY_AUDIT_COSTS.json','BINARY_OPTIMIZATION_ACCURACY.json']
    for name in required+results+['DELIVERY_INDEPENDENT_AUDIT.json']:
        if not (PUBLIC/name).is_file():raise FileNotFoundError(name)
    put(PUBLIC/'E4_COMPLETE_RESULTS.json',dict(schema='e4_completed_research_v1',status='COMPLETE',
        research_branch='codex/e4-statistical-model-audit-20261107',base_commit='45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6',
        completed=dict(A_learning_fits=650,A_raw_aggregate_score_checks=36000,A_member_executions=10800,
                       B_paths=900,C_fit_groups=30,C_models_per_group=4,C_inner_fits_per_model=9,
                       C_online_paths=720,D_live_paths=165,D_snapshot_candidates=1200),
        new_physical_llm_calls=0,independent_original_questions=600,
        major_findings=dict(
            audited_score_mismatches=0,
            cross_seed_oracle_accuracy=.8838888888888887,
            cross_seed_oracle_minus_oof_subtask_fixed=.025555555555555567,
            P2_probit_minus_tuned_gaussian_brier=-.000005658188830181383,
            P2_probit_minus_tuned_gaussian_accuracy=.006666666666666667,
            E3_DA_minus_Random=-.005555555555555556,E4_DA_minus_Random=-.004999999999999999,
            evidence_scope='paired, conditional development audit; no independent fresh400 confirmation'),
        hypotheses=dict(H1='INSUFFICIENT_CAUSAL_EVIDENCE',H2='NOT_SUPPORTED_IN_THIS_AUDIT',
                        H3='INSUFFICIENT_EVIDENCE',H4='NOT_SUPPORTED_IN_THIS_AUDIT'),
        next_algorithm_gate='NOT_PASSED',fresh400_scores_used=False,sealed200_scores_used=False,
        results={name:dict(sha256=sha(PUBLIC/name),bytes=(PUBLIC/name).stat().st_size) for name in results},
        independent_verification_report='DELIVERY_INDEPENDENT_AUDIT.json',
        reports=required[:8],configuration_files=required[8:],
        limitations=['fixed-overlapping-fold development inference','three member-generation repeats with aggregator seed fixed42',
                     'metadata sensitivity P2 only','D order17 only','candidate chosen on development Brier',
                     'true latent reward coverage unavailable','source wall time is not GPU kernel time',
                     'official gold reconstruction and pretraining contamination not identified']))
    # All files in this public directory are prepared explicitly by this experiment.
    # Raw rows/arrays are excluded at creation, and an independent scanner checks this set.
    files={}
    for path in sorted(PUBLIC.iterdir()):
        if not path.is_file() or path.name in ('PUBLIC_DELIVERY_MANIFEST.json','DELIVERY_INDEPENDENT_AUDIT.json'):continue
        value=path.read_bytes()
        files[path.name]=dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest(),
            git_blob=hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest())
    put(PUBLIC/'PUBLIC_DELIVERY_MANIFEST.json',dict(schema='e4_public_delivery_v1',files=files,
        separately_versioned_verification_report='DELIVERY_INDEPENDENT_AUDIT.json',
        self_reference_rule='Manifest and verification receipt are versioned as Git blobs; receipt checks the manifest without a circular content hash.',
        exclusion='raw questions, answers, gold, embeddings, private arrays and line-level records remain private',
        public_prefix='routing_idea2/research/e4_statistical_model_audit/'))
    print(json.dumps(dict(status='COMPLETE',public_files=len(files)+1)))


if __name__=='__main__':main()
