"""Create an append-only M4 audit and snapshot its read-only E8 parent."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

STAGING = Path(__file__).resolve().parent
ASSETS = Path(r'E:\语言模型组合路由')
PARENT = ASSETS / 'research/e8_decision_focused_combination_learning'
OUT = ASSETS / 'research/e8_m4_mechanism_audit'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise RuntimeError('New output directory already exists')
    (OUT / 'private/lineage').mkdir(parents=True)
    (OUT / 'public/orchestration').mkdir(parents=True)
    files = {}
    for path in sorted(PARENT.rglob('*')):
        if path.is_file():
            files[path.relative_to(PARENT).as_posix()] = {
                'sha256': sha(path), 'bytes': path.stat().st_size}
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'parent_root': str(PARENT),
        'parent_commit': '45767047d556e6cd9cac0e8905cf6dd81d5e3d4f',
        'files': files,
    }
    (OUT / 'private/lineage/PARENT_E8_SNAPSHOT.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'public/M4_AUDIT_SCOPE.md').write_bytes((STAGING / 'M4_AUDIT_SCOPE.md').read_bytes())
    (OUT / 'public/orchestration/initialize_audit.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'root': str(OUT), 'parent_files': len(files)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
