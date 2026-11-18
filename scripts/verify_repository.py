#!/usr/bin/env python3
"""Offline verification of the curated archive; never executes experiment code."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'reproducibility/frozen_manifest.json'


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def checked_path(relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Unsafe manifest path: {relative!r}')
    target = ROOT.joinpath(*path.parts)
    if not target.resolve().is_relative_to(ROOT):
        raise ValueError(f'Path leaves repository: {relative!r}')
    return target


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['schema_version'] != 1 or manifest['repository'] != 'fallenplum/MAB':
        raise ValueError('Unexpected frozen manifest schema/repository')
    entries = manifest['files']
    if len({x['path'] for x in entries}) != len(entries):
        raise ValueError('Duplicate frozen manifest paths')
    for entry in entries:
        file = checked_path(entry['path'])
        data = file.read_bytes()
        if len(data) != entry['size'] or git_blob_sha(data) != entry['git_blob_sha'] or \
                hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'Frozen bytes differ: {entry["path"]}')

    archive = json.loads((ROOT / 'reproducibility/archive_inventory.json').read_text(encoding='utf-8'))
    archived_paths = {x['path'] for x in archive['files']}
    if len(archived_paths) != len(archive['files']) or archived_paths & {x['path'] for x in entries}:
        raise ValueError('Archived/retained source inventories overlap or contain duplicates')
    if archive['source_commit'] != manifest['source_commit']:
        raise ValueError('Source inventories disagree about the immutable commit')

    # Parse archived source rather than import it: many historical runners have
    # machine-specific paths and some launch paid generation or mutate outputs.
    python_paths = [checked_path(x['path']) for x in entries if x['path'].endswith('.py')]
    python_paths += list((ROOT / 'scripts').rglob('*.py'))
    python_paths += list((ROOT / 'tests').rglob('*.py'))
    for file in python_paths:
        ast.parse(file.read_bytes(), filename=file.relative_to(ROOT).as_posix())
    json_paths = [checked_path(x['path']) for x in entries if x['path'].endswith('.json')]
    json_paths += list((ROOT / 'reproducibility').glob('*.json'))
    for file in json_paths:
        json.loads(file.read_text(encoding='utf-8-sig'))

    # Validate relative file links in new entry-point documentation only.
    # Frozen reports retain original external paths and historical identities.
    docs = [ROOT / 'README.md', ROOT / 'routing_idea2/README.md']
    docs += list((ROOT / 'docs').glob('*.md')) + list((ROOT / 'paper').glob('*.md'))
    relative_links = 0
    for file in docs:
        text = file.read_text(encoding='utf-8')
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            target = target.strip().split(' "', 1)[0].strip('<>')
            if not target or target.startswith('#') or urlsplit(target).scheme:
                continue
            linked = (file.parent / unquote(target.split('#', 1)[0])).resolve()
            if not linked.is_relative_to(ROOT) or not linked.exists():
                raise ValueError(f'Broken relative link in {file.relative_to(ROOT)}: {target}')
            relative_links += 1

    from migrate_branch_names import load_manifest, MigrationError
    migration_entries = []
    try:
        for path in sorted((ROOT / 'reproducibility').glob('branch_migration*.json')):
            migration_entries.extend(load_manifest(path))
    except MigrationError as error:
        raise ValueError(f'Invalid branch migration manifest: {error.code}') from None
    old_names, new_names = set(), set()
    for item in migration_entries:
        if item.old in old_names or item.new in new_names:
            raise ValueError('Duplicate branch migration record')
        old_names.add(item.old)
        new_names.add(item.new)

    from build_paper_tables import main as check_tables
    if check_tables(['--check']) != 0:
        raise ValueError('Paper tables differ from archived inputs')

    private = json.loads((ROOT / 'reproducibility/private_artifacts.json').read_text(encoding='utf-8'))
    missing_sources = [x['git_path'] for x in private['missing_direct_runtime_sources']
                       if not checked_path(x['git_path']).is_file()]
    return {
        'status': 'PASS',
        'scope': 'offline archive integrity, source syntax, entry links, and archived-summary reconstruction',
        'source_commit': manifest['source_commit'],
        'frozen_files_verified': len(entries),
        'historical_file_versions_indexed': len(archive['files']),
        'python_sources_parsed_without_execution': len(python_paths),
        'json_files_parsed': len(json_paths),
        'relative_entry_links_verified': relative_links,
        'branch_name_mappings_validated_offline': len(migration_entries),
        'missing_direct_generation_sources': missing_sources,
        'full_training_or_generation_reproduced': False,
        'private_asset_status': 'Not supplied or verified by this offline check; see docs/REPRODUCIBILITY.md',
    }


def main() -> int:
    try:
        result = verify()
    except (OSError, ValueError, KeyError, SyntaxError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
