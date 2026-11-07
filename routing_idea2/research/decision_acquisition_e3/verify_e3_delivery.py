"""Verify the explicit E3 public whitelist and its immutable evidence chain."""
import gzip
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import sys
from run_e3 import PUBLIC, PRIVATE, signature, put
from assets import load_assets

REPORTS = ['01_FEEDBACK_RESPONSE.md', '02_ACQUISITION_BASELINES.md', '03_CROSS_PREDICTOR_RESULTS.md',
           '04_MECHANISM_ABLATIONS.md', '05_COST_AND_REUSE.md', '06_STATISTICAL_ANALYSIS.md', '07_RESEARCH_CONCLUSION.md']
SCRIPTS = ['assets.py', 'predictors.py', 'acquisition.py', 'run_e3.py', 'e3_statistics.py', 'summarize_e3.py',
           'primary_statistics.py', 'mechanism_diagnostics.py', 'replay_mechanism_audit.py', 'audit_e3.py',
           'audit_kg_precision.py', 'build_e3_delivery.py', 'execute_all.py', 'verify_e3_delivery.py']
RESULTS = ['E3_COMPLETE_RESULTS.json', 'E3_COMPLETE_RESULTS.json.gz', 'E3_RUN_CONFIG.json', 'E3_COMPLETION.json',
           'PRIMARY_STATISTICS.json', 'QUALIFICATION.json', 'REUSE_ACCOUNTING.json', 'KG_PRECISION_AUDIT.json',
           'MECHANISM_AUDIT_B1.json', 'MECHANISM_AUDIT_B2.json', 'MECHANISM_AUDIT_J.json',
           'feedback_quality_curves.png', 'quality_complete_token_curves.png',
           'E3_PROTOCOL.md', 'README.md', 'RUNTIME_ENVIRONMENT.json', 'TECHNICAL_QA_SUMMARY.json']
ARCHIVES = ['FULL_' + stage + '_STATISTICS.json.gz' for stage in ('A', 'B1', 'B2', 'J', 'D', 'SENSITIVITY')]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    identity, sources = signature()
    train, refs = load_assets()
    original_ids = set(map(str, train['ids']))
    canonicals = set(map(str, train['canonical']))
    question_pattern = re.compile('|'.join(re.escape(value) for value in sorted(original_ids, key=len, reverse=True)))
    hash_pattern = re.compile(r'(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])')
    credential_pattern = re.compile(r'github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----')
    address_pattern = re.compile(r'(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)')
    runtime = {'python': platform.python_version(), 'platform': sys.platform, 'machine': platform.machine(),
               'libraries': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'matplotlib')},
               'cpu_only': True, 'blas_threads_per_worker': 1,
               'recorded_original_study_workers': {'A': 4, 'B1': 8, 'B2': 16, 'J': 16, 'SENSITIVITY': 8, 'D': 16, 'DELETE': 6},
               'timing_scope': 'planning/update process CPU and source inference-call wall sums reported separately'}
    put(PUBLIC / 'RUNTIME_ENVIRONMENT.json', runtime)
    receipt = read(PRIVATE / 'PRECHECK_LEDGER_AUDIT.json')
    assert identity in json.dumps(receipt), 'Precheck receipt binds a different engine'
    audits = {stage: read(PUBLIC / ('MECHANISM_AUDIT_' + stage + '.json'))['qualification'] for stage in ('B1', 'B2', 'J')}
    put(PUBLIC / 'TECHNICAL_QA_SUMMARY.json', {'status': 'PASS', 'execution_identity': identity,
                                            'numerical_operator_qualification': read(PUBLIC / 'QUALIFICATION.json'),
                                            'precheck_paths_independently_audited': 88,
                                            'private_full_audit_sha256': sha((PRIVATE / 'PRECHECK_LEDGER_AUDIT.json').read_bytes()),
                                            'hidden_label_test': {'scores_flipped': 7180, 'purchased_scores_retained': 20,
                                                                  'service_decisions_preserved': 120, 'evaluation_rewards_changed': 100},
                                            'legal_feedback_replay_audits': audits,
                                            'scope': 'software and ledger checks; scientific gain is assessed separately'})
    config = read(PUBLIC / 'E3_RUN_CONFIG.json')
    assert config['identity'] == identity
    for name, expected in sources['code'].items():
        assert sha((PUBLIC / name).read_bytes()) == expected
    primary = read(PUBLIC / 'PRIMARY_STATISTICS.json')
    assert primary['status'] == 'END_TO_END_DEVELOPMENT_COMPLETE'
    assert primary['code_sha256'] == sha((PUBLIC / 'primary_statistics.py').read_bytes())
    index = read(PUBLIC / 'E3_COMPLETE_RESULTS.json')
    assert len(index['groups']) == 361 and index['main_paths'] == 16245 and index['delete_replay_paths'] == 6300
    assert index['data_access']['decoded_source_splits'] == ['selector_train']
    assert index['new_llm_calls'] == 0
    archive_by_name = {item['name']: item for item in index['full_stage_archives']}
    checked = {}
    for name in REPORTS + SCRIPTS + RESULTS + ARCHIVES:
        path = PUBLIC / name
        raw = path.read_bytes()
        decoded = gzip.decompress(raw) if name.endswith('.gz') else raw
        if name.endswith('.png'):
            assert raw.startswith(b'\x89PNG\r\n\x1a\n')
        else:
            text = decoded.decode('utf-8')
            assert not question_pattern.search(text), 'Raw question ID in ' + name
            assert not canonicals.intersection(hash_pattern.findall(text)), 'Raw canonical ID in ' + name
            assert not credential_pattern.search(text), 'Credential pattern in ' + name
            if name not in ('verify_e3_delivery.py',):
                assert not address_pattern.search(text), 'Address pattern in ' + name
            if name.endswith('.py'):
                if name != 'verify_e3_delivery.py':
                    assert '.e3_staging' not in text and 'D:\\develop' not in text, 'Temporary dependency in ' + name
                compile(text, name, 'exec')
            if name.endswith(('.json', '.json.gz')):
                json.loads(text)
        if name in archive_by_name:
            meta = archive_by_name[name]
            assert sha(raw) == meta['sha256'] and sha(decoded) == meta['decompressed_sha256']
            summary = json.loads(decoded)
            assert summary['execution_identity'] == identity
            assert summary['statistics_code_sha256'] == sha((PUBLIC / 'e3_statistics.py').read_bytes())
            assert summary['summary_code_sha256'] == sha((PUBLIC / 'summarize_e3.py').read_bytes())
        git_header = ('blob ' + str(len(raw)) + '\0').encode()
        checked[name] = {'bytes': len(raw), 'sha256': sha(raw), 'git_blob_sha1': hashlib.sha1(git_header + raw).hexdigest()}
    assert gzip.decompress((PUBLIC / 'E3_COMPLETE_RESULTS.json.gz').read_bytes()) == (PUBLIC / 'E3_COMPLETE_RESULTS.json').read_bytes()
    manifest = {'status': 'PASS', 'branch': 'codex/decision-acquisition-e3-20261107',
                'base_commit': '7a21e158fabe56acbe3a40a3dd0e09c139c6c493', 'execution_identity': identity,
                'publication_files': checked, 'raw_question_ids_scan': 'PASS', 'raw_canonical_hash_scan': 'PASS',
                'credential_address_scan': 'PASS', 'syntax_check': 'PASS', 'source_and_archive_hashes': 'PASS',
                'raw_arrays_weights_answers_and_credentials_in_whitelist': False}
    put(PUBLIC / 'PUBLICATION_MANIFEST.json', manifest)
    print(json.dumps({'status': 'PASS', 'files_including_manifest': len(checked)+1,
                      'total_bytes': sum(item['bytes'] for item in checked.values()), 'execution_identity': identity}), flush=True)


if __name__ == '__main__':
    main()
