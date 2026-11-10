"""Bind the completed, gated E5 protocol and explicit public artifact identities."""
from e5_common import *


def read(name):return json.loads((PUBLIC/name).read_text(encoding='utf-8-sig'))


def main():
    reports=['01_THEORY_IMPLEMENTATION_CHECKS.md','02_CENTERED_ERROR_DIAGNOSTICS.md',
        '03_SWITCH_RANKING_AND_MATCHED_CONTROLS.md','04_CERTIFICATION_VALIDITY_AND_POWER.md',
        '05_TWO_RESOURCE_ALLOCATION.md','EXPERIMENT_SUMMARY.md','REPRODUCE.md']
    results=['THEORY_IMPLEMENTATION_RESULTS.json','ACTUAL_CENTERING_RESULTS.json',
        'CENTERED_DIAGNOSTICS_RESULTS.json','SWITCH_RANKING_RESULTS.json','SWITCH_ABLATION_COMPARISONS.json',
        'METADATA_PERMISSION_CONTROLS.json','CERTIFICATION_SYNTHETIC_RESULTS.json',
        'CERTIFICATION_DEVELOPMENT_POWER.json','ALLOCATION_SYNTHETIC_RESULTS.json',
        'ALLOCATION_SELECTION_NUMERICAL_SENSITIVITY.json','COST_AND_REUSE_RESULTS.json',
        'REAL_ALLOCATION_GATE.json','REFERENCE_AND_INPUTS.json','E5_EXPERIMENT_CONFIG.json','E5_PROTOCOL_AMENDMENTS.json']
    for name in reports+results+['E5_INDEPENDENT_AUDIT.json']:
        assert (PUBLIC/name).is_file(),name
    a,cert,allocation,gate=map(read,['THEORY_IMPLEMENTATION_RESULTS.json',
        'CERTIFICATION_SYNTHETIC_RESULTS.json','ALLOCATION_SYNTHETIC_RESULTS.json','REAL_ALLOCATION_GATE.json'])
    assert a['status']=='PASS' and cert['status']=='COMPLETE' and allocation['status']=='COMPLETE'
    assert gate['status']=='NOT_PASSED' and not gate['passed']
    assert allocation['real_train600_allocation']=='NOT_RUN_GATE_NOT_PASSED'
    finite=sum(row['simulation_fits'] for row in a['finite']['cases'])
    assert (len(a['population']['cases']),len(a['population']['population_ridge']),finite)==(140,420,1800)
    assert len(cert['rows'])==640 and len(allocation['cases'])==18
    assert all(len(case['methods'])==7 for case in allocation['cases'])
    scientific=dict(
        error_component_result='P2_to_Probit raw absolute loss falls, common offset falls more, centered loss rises',
        primary_gate_evidence='INSUFFICIENT_FOR_WITHIN_TASK_RANKING',
        certification='mathematical/numerical validity under declared model assumptions, limited small-gain power',
        allocation='second-moment and estimation improvements separate from strategy-selection quality',
        real_allocation='NOT_RUN_GATE_NOT_PASSED')
    put(PUBLIC/'E5_COMPLETE_RESULTS.json',dict(schema='e5_completed_gated_protocol_v1',
        status='COMPLETED_AS_GATED_PROTOCOL',branch='codex/e5-paired-gain-validation',
        base_commit='b5fa3317c77b1256037d734c93a001ecb19bfa28',
        manuscript_sha256=a['manuscript_sha256'],independent_original_E_questions=600,generation_repeats=3,
        physical_new_llm_calls=0,
        stages=dict(A=dict(status='COMPLETE',population_cases=140,population_ridge_cases=420,
            finite_paired_trials=1800,finite_LS_and_Ridge_fits=3600,synthetic_centering_cases=21,actual_centering_folds=5),
            B=dict(status='COMPLETE',F_fit_models=15,feedback_paths=45),
            C=dict(status='COMPLETE',folds=5,rules_per_fold=20,main_score_procedures=4,metadata_permission_fits=10),
            D=dict(status='COMPLETE',synthetic_configurations=640,MC_per_configuration=10000,
                   real_development_fold_certificates=5,development_resampling_configurations=25,MC_resampling_each=2000),
            E_synthetic=dict(status='COMPLETE',budget_family_cases=18,allocation_methods=7,MC_each=2000,
                             estimated_weight_cost_sensitivity_plans=900),
            E_real=dict(status='NOT_RUN_GATE_NOT_PASSED',required_if_development_gate_passes=False)),
        primary_comparisons=gate['primary_comparisons'],scientific=scientific,
        reports=reports,results={name:dict(sha256=sha(PUBLIC/name),bytes=(PUBLIC/name).stat().st_size) for name in results},
        independent_verification_report='E5_INDEPENDENT_AUDIT.json',
        preserved_history='Original E1-E4 artifacts and pre-existing Git blobs preserved',
        data_scope='Controlled development re-use. No validation/fresh400/sealed200 score payloads used.',
        limitations=['historical model selection and overlapping training histories',
            'three seed IDs do not establish repetition independence or common conditional law',
            'matched controls condition on frozen fold policies and cached generation blocks',
            'u conditional likelihood retains original1/L observation weights',
            'G OOF versus full-G head distribution shift',
            'upper sample planning is certificate-specific and plug-in, lower bound applies to its stated distribution class',
            'synthetic allocation budgets are expected resources, not hard realized caps',
            'GPU kernel time and original encoder preparation timing unavailable']))
    files={}
    for path in sorted(PUBLIC.iterdir()):
        if not path.is_file() or path.name in ('PUBLIC_DELIVERY_MANIFEST.json','E5_INDEPENDENT_AUDIT.json'):continue
        value=path.read_bytes()
        files[path.name]=dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest(),
            git_blob=hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest())
    put(PUBLIC/'PUBLIC_DELIVERY_MANIFEST.json',dict(schema='e5_public_delivery_v1',files=files,
        separately_versioned_verification_report='E5_INDEPENDENT_AUDIT.json',
        self_reference_rule='Manifest and verification receipt are separately Git-versioned; receipt checks manifest without cyclic content hashes.',
        private_artifacts='raw questions, gold, responses, embeddings, paired arrays, small model parameters, raw pilot and MC worlds',
        prefix='routing_idea2/research/e5_paired_gain_validation/'))
    print(json.dumps(dict(status='COMPLETED_AS_GATED_PROTOCOL',hashed_files=len(files),public_files=len(files)+2)))


if __name__=='__main__':main()
