"""Reproducible project-venv-only backport of bitsandbytes PR #1910.

This changes optional Gaudi package discovery, not CUDA kernels or generation.
"""
from pathlib import Path
import argparse,hashlib,json,os,time
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'.venv/Lib/site-packages/bitsandbytes/backends/utils.py'
ORIGINAL_SHA256='88f7a4fb362463367466a1f9fc1456217d879be395eca66a2ee2ade29db51656'
PATCHED_SHA256='c885968713bd7d24b828ab7ba2de10e3ed9ddc7d9da1dec8a54970ffa6e55223'
SOURCE_URL='https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1910'
SOURCE_MERGE_COMMIT='4986b43b39d219fd717b33e337a82e76a10ee0c7'


def sha(data):return hashlib.sha256(data).hexdigest()


def build_patched_source(raw):
    if sha(raw)!=ORIGINAL_SHA256:raise ValueError('Unknown upstream bytes; no patch applied')
    source=raw.decode('utf-8');nl='\r\n' if b'\r\n' in raw else '\n'
    if not source.startswith('import subprocess'+nl):raise ValueError('Unexpected upstream import')
    source=source.replace('import subprocess'+nl,'from importlib.metadata import metadata'+nl,1)
    start=source.index('def get_gaudi_sw_version():')
    end=source.index('GAUDI_SW_VER = get_gaudi_sw_version()',start)
    function=[
      'def get_gaudi_sw_version():',
      '    """Return optional Gaudi plugin version without a platform shell command."""',
      '    try:',
      '        plugin_version = metadata("habana-torch-plugin").get("Version")',
      '        # PR #1910 intent, including the missing-Version boundary case.',
      '        return version.parse(plugin_version) if plugin_version else None',
      '    except Exception:',
      '        return None',
    ]
    return (source[:start]+nl.join(function)+nl*3+source[end:]).encode('utf-8')


def planned_bytes(raw):
    digest=sha(raw)
    if digest==PATCHED_SHA256:return raw
    candidate=build_patched_source(raw)
    if sha(candidate)!=PATCHED_SHA256:raise ValueError('Patch output is not the reviewed deterministic bytes')
    return candidate


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');a=p.parse_args(argv)
    target=TARGET.resolve(strict=True)
    if not target.is_relative_to((ROOT/'.venv').resolve()) or target!=TARGET.absolute():
        raise ValueError('Only this project virtual environment may be patched')
    package_version=version('bitsandbytes')
    if package_version!='0.48.2':raise ValueError('This backport is reviewed only for bitsandbytes 0.48.2')
    before=target.read_bytes();after=planned_bytes(before)
    report={'package':'bitsandbytes','package_version':package_version,'source_url':SOURCE_URL,
      'source_merge_commit':SOURCE_MERGE_COMMIT,'target':target.relative_to(ROOT).as_posix(),
      'original_sha256':ORIGINAL_SHA256,'patched_sha256':PATCHED_SHA256,
      'change':'Optional Gaudi plugin metadata discovery only; no shell, CUDA kernel, model weight or generation setting change',
      'scientific_completion':False}
    if not a.apply:
        print(json.dumps({'status':'ALREADY_PATCHED' if before==after else 'PATCH_AVAILABLE',**report},ensure_ascii=False));return 0
    receipt=ROOT/'artifacts/audit/runtime_patch_bitsandbytes_gaudi.json'
    if before==after:
        if not receipt.exists():raise ValueError('Patched bytes exist without their audit receipt; reconcile instead of inventing history')
        existing=json.loads(receipt.read_text('utf-8'))
        if existing.get('patched_sha256')!=PATCHED_SHA256:raise ValueError('Patch receipt differs')
        print(json.dumps({'status':'ALREADY_PATCHED','receipt':str(receipt)},ensure_ascii=False));return 0
    backup=ROOT/'artifacts/runtime_patches/bitsandbytes_gaudi_0.48.2.original.py'
    backup.parent.mkdir(parents=True,exist_ok=True)
    if backup.exists() and sha(backup.read_bytes())!=ORIGINAL_SHA256:raise ValueError('Original backup differs; preserve it')
    if not backup.exists():backup.write_bytes(before)
    pending=target.with_name(target.name+'.project_patch.pending')
    with pending.open('wb') as stream:stream.write(after);stream.flush();os.fsync(stream.fileno())
    os.replace(pending,target)
    report.update(status='APPLIED',applied_at_unix=time.time(),backup=backup.relative_to(ROOT).as_posix())
    receipt.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
