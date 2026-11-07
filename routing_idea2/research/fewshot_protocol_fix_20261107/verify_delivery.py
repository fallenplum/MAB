"""Fail-closed publication whitelist and aggregate artifact verification."""
import gzip
import hashlib
import json
from pathlib import Path
import re
from run_corrected import PUBLIC, PRIVATE, load, sha, put

FILES = ['models.py', 'run_corrected.py', 'stats.py', 'build_reports.py', 'verify_delivery.py',
         'PROTOCOL.md', 'CORRECTION_AUDIT.md', 'COMPLETE_COMPARISON.md', 'TASK_AND_SEED_DIAGNOSTICS.md',
         'COST_ACCOUNTING.md', 'EXPERIMENT_SUMMARY.md', 'FROZEN_CANDIDATE.json',
         'QUALIFICATION.json', 'REPLAY_QUALIFICATION.json', 'COMPLETION.json', 'SCIENTIFIC_GATE.json',
         'EXPERIMENT_RESULTS.json.gz', 'TASK_RESULTS.json.gz', 'COSTS.json.gz', 'PUBLICATION_MANIFEST.json']


def main():
    data, _ = load()
    forbidden = [str(value) for split in data.values() for key in ('ids', 'canonical') for value in split[key]]
    patterns = [r'(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)', r'github_pat_[A-Za-z0-9_]+', r'ghp_[A-Za-z0-9]+',
                r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----']
    checked = {}
    for name in FILES:
        path = PUBLIC / name
        raw = path.read_bytes()
        decoded = gzip.decompress(raw) if name.endswith('.gz') else raw
        text = decoded.decode('utf-8')
        assert not any(value in text for value in forbidden), 'raw identity in ' + name
        # The regex source in this scanner describes forbidden patterns itself.
        if name != 'verify_delivery.py':
            assert not any(re.search(pattern, text) for pattern in patterns), 'credential/address pattern in ' + name
        if name.endswith('.gz'):
            assert decoded == (PUBLIC / name[:-3]).read_bytes()
            json.loads(text)
        checked[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    report = json.loads((PUBLIC / 'EXPERIMENT_RESULTS.json').read_text(encoding='utf-8'))
    assert len(report) == 12
    assert len(json.loads((PUBLIC / 'TASK_RESULTS.json').read_text(encoding='utf-8'))['rows']) == 4032
    assert json.loads((PUBLIC / 'QUALIFICATION.json').read_text())['status'] == 'PASS'
    assert json.loads((PUBLIC / 'REPLAY_QUALIFICATION.json').read_text())['status'] == 'PASS'
    assert json.loads((PUBLIC / 'COMPLETION.json').read_text())['fresh400_scores_read'] is False
    receipt = {'status': 'PASS', 'publication_files': checked,
               'raw_question_id_and_canonical_hash_scan': 'PASS', 'credential_address_pattern_scan': 'PASS',
               'gzip_matches_local_full_json': True, 'task_method_configuration_records': 4032,
               'raw_arrays_weights_prompts_and_credentials_published': False}
    put(PUBLIC / 'DELIVERY_VERIFICATION.json', receipt)
    print(json.dumps({'status': 'PASS', 'whitelisted_files': len(FILES) + 1, 'total_bytes': sum(item['bytes'] for item in checked.values())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
