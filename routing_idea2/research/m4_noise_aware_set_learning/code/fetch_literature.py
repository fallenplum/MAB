"""Bounded primary-source lookup, provenance and full-PDF extraction."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import urllib.request
import urllib.error
import urllib.parse
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent.parent
PRIVATE = HERE / 'private/literature'
PUBLIC = HERE / 'public'
PDFS = {
    'dpo_ties_2025': 'https://proceedings.neurips.cc/paper_files/paper/2025/file/4dbf3488a8da908a8177d525ede1a43a-Paper-Conference.pdf',
    'noisy_partial_ranking_2025': 'https://publications.rwth-aachen.de/record/1004566/files/1004566.pdf',
    'spo_plus': 'https://optimization-online.org/wp-content/uploads/2018/12/6398.pdf',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url):
    last = None
    for attempt in range(3):
        try:
            # The Python environment lacks this host's issuer chain. Windows
            # validates the same TLS request using its installed trust store.
            handle = tempfile.NamedTemporaryFile(dir=PRIVATE, delete=False, suffix='.download')
            target = Path(handle.name)
            handle.close()
            command = ("$ErrorActionPreference='Stop'; Invoke-WebRequest -UseBasicParsing -Uri '"
                       + url.replace("'", "''") + "' -OutFile '"
                       + str(target).replace("'", "''") + "' -TimeoutSec 40")
            completed = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                                       capture_output=True, text=True, timeout=50)
            if completed.returncode:
                raise RuntimeError(completed.stderr[-1500:])
            body = target.read_bytes()
            target.unlink()
            return body, {'url': url, 'http_status': 200,
                          'retrieval': 'Windows Invoke-WebRequest, TLS certificate validation enabled',
                          'redirect_final_url': 'not independently recorded',
                          'sha256': sha(body), 'bytes': len(body)}
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            last = error
            time.sleep(1)
    raise RuntimeError(str(last))


def main():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    records = {}
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    for key, url in PDFS.items():
        payload, record = fetch(url)
        if not payload.startswith(b'%PDF'):
            if key != 'noisy_partial_ranking_2025':
                raise AssertionError('Expected actual PDF ' + url)
            rejected = dict(record, reason='HTTP200 body is not a PDF')
            payload, record = fetch('https://arxiv.org/pdf/2405.18259')
            record['rejected_primary_download'] = rejected
            record['version_scope'] = 'author arXiv full-text version, publisher DOI identity checked separately'
            if not payload.startswith(b'%PDF'):
                raise AssertionError('Author full text also not PDF')
        path = PRIVATE / (key + '.pdf')
        path.write_bytes(payload)
        reader = PdfReader(path)
        pages = [page.extract_text() or '' for page in reader.pages]
        (PRIVATE / (key + '.txt')).write_text('\n\n'.join(
            f'PAGE {i+1}\n{text}' for i, text in enumerate(pages)), encoding='utf-8')
        record.update(pages=len(pages), extracted_characters=sum(map(len, pages)), full_text_saved_privately=True)
        records[key] = record
    for doi in ('10.1080/01621459.1970.10481082', '10.1007/s41060-025-00722-1',
                '10.1287/mnsc.2020.3922', '10.52202/085713-1801'):
        endpoint = 'https://api.crossref.org/works/' + urllib.parse.quote(doi, safe='')
        try:
            payload, record = fetch(endpoint)
            value = json.loads(payload)
            if value.get('status') != 'ok' or value['message']['DOI'].lower() != doi.lower():
                raise AssertionError('Crossref identity mismatch')
            record.update(doi=value['message']['DOI'], title=value['message']['title'],
                          issued=value['message'].get('issued'))
            (PRIVATE / ('crossref_' + doi.replace('/', '_') + '.json')).write_bytes(payload)
            records[doi] = record
        except Exception as error:
            records[doi] = {'url': endpoint, 'status': 'FAILED', 'reason': str(error)}
    repo = 'https://api.github.com/repos/EriChen0615/DPO-RKD'
    payload, record = fetch(repo)
    metadata = json.loads(payload)
    branch = metadata['default_branch']
    payload, _ = fetch(repo + '/commits/' + branch)
    commit = json.loads(payload)['sha']
    payload, tree_record = fetch(repo + '/git/trees/' + commit + '?recursive=1')
    tree = json.loads(payload)
    if tree.get('truncated'):
        raise AssertionError('Incomplete author tree')
    items = [item['path'] for item in tree['tree'] if item['type'] == 'blob']
    (PRIVATE / 'author_repo_tree.json').write_bytes(payload)
    records['author_repo'] = dict(**record, commit=commit, files=items,
        training_code_available=any(path.endswith(('.py', '.ipynb', '.jl', '.R')) for path in items),
        tree_sha256=tree_record['sha256'])
    davidson_url = 'https://www.tandfonline.com/doi/pdf/10.1080/01621459.1970.10481082'
    try:
        payload, record = fetch(davidson_url)
        record['pdf_verified'] = payload.startswith(b'%PDF')
        if record['pdf_verified']:
            (PRIVATE / 'davidson_1970.pdf').write_bytes(payload)
        records['davidson_original_fulltext_access'] = record
    except Exception as error:
        records['davidson_original_fulltext_access'] = {
            'url': davidson_url, 'status': 'UNAVAILABLE', 'reason': str(error),
            'formula_source': 'NeurIPS2025 primary full paper Eq8,9,12,13 and AppendixB.1'}
    result = {'status': 'PRIMARY_LITERATURE_LOOKUP_COMPLETED_WITH_ACCESS_LIMITATION',
              'accessed_utc': datetime.now(timezone.utc).isoformat(), 'sources': records,
              'code_scope': 'paper-formula clean implementation, author training pipeline not reproduced'}
    (PUBLIC / 'LITERATURE_SOURCE_MANIFEST.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pdf_pages': {key: records[key]['pages'] for key in PDFS},
                      'author_commit': commit, 'author_files': items}, ensure_ascii=False))


if __name__ == '__main__':
    main()
