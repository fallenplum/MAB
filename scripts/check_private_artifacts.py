#!/usr/bin/env python3
"""Check restored asset bytes against recorded receipts; do not run experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_entry(entry: dict, asset_root: Path, kind: str) -> dict:
    relative = entry.get('asset_relative_path')
    result = {'id': entry.get('id', relative), 'kind': kind,
              'asset_relative_path': relative, 'expected_sha256': entry.get('expected_sha256')}
    if relative is None or any(c in relative for c in '{}<>*?'):
        result['status'] = 'manual_inventory_required'
        return result
    parts = PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts or '\\' in relative or ':' in relative:
        raise ValueError('Inventory entry is not a safe asset-relative path')
    path = asset_root.joinpath(*parts.parts)
    if not path.exists():
        result['status'] = 'missing'
    elif not path.is_file() or result['expected_sha256'] is None:
        result['status'] = 'present_but_no_single_frozen_file_hash'
    else:
        actual = sha256_file(path)
        result['actual_sha256'] = actual
        result['status'] = 'hash_verified' if actual == result['expected_sha256'] else 'hash_mismatch'
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, required=True,
                        help='Restored private asset root outside the Git checkout')
    parser.add_argument('--dars-root', type=Path,
                        help='Optional root with gpqa/ and math-500/ from the frozen DARS revision')
    args = parser.parse_args(argv)
    try:
        assets = args.asset_root.resolve()
        if not assets.is_dir():
            raise ValueError('The supplied asset root does not exist or is not a directory')
        inventory = json.loads((ROOT / 'reproducibility/private_artifacts.json').read_text(encoding='utf-8'))
        entries = [(x, 'private_asset') for x in inventory['artifacts']]
        entries += [(x, 'direct_generation_source') for x in inventory['missing_direct_runtime_sources']]
        entries += [(x, 'preparation_source') for x in inventory['missing_preparation_source']]
        results = [inspect_entry(entry, assets, kind) for entry, kind in entries]
        for entry in inventory['external_dars']['files']:
            external = {'id': 'dars/' + entry['relative_to_data_argument'],
                        'asset_relative_path': entry['relative_to_data_argument'],
                        'expected_sha256': entry['expected_sha256']}
            if args.dars_root is None:
                results.append({**external, 'kind': 'external_dars', 'status': 'data_root_not_supplied'})
            else:
                results.append(inspect_entry(external, args.dars_root.resolve(), 'external_dars'))
        incomplete = any(x['status'] != 'hash_verified' for x in results)
        print(json.dumps({'status': 'INCOMPLETE' if incomplete else 'RECORDED_FILE_HASHES_VERIFIED',
                          'scope': 'Recorded file hashes only; no model training, inference, scoring or execution-path validation',
                          'full_reproduction_verified': False,
                          'excluded_inventory_sections': ['model_cache', 'protected_data'],
                          'manual_requirements': ['Model snapshots/fingerprints, pattern inventories, path mapping and runtime environment remain separate checks.'],
                          'entries': results}, ensure_ascii=False, indent=2))
        return 2 if incomplete else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
