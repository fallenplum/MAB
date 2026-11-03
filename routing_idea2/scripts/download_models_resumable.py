"""Pinned HF assets with verified offline cache reuse and persistent Range resume.

Completed receipts are never trusted without rereading complete local hashes.
A pinned metadata cache needs its recorded digest or a completed source receipt.
No TLS downgrade, checkpoint substitution or inference is performed here.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
import argparse, fnmatch, hashlib, json, os, re, shutil, sys, time, uuid
import requests
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from budget_router.downloads import (resumable_download, DownloadIntegrityError,
    DownloadLockedError, RangeProtocolError)
from download_inference_models import PATTERNS

class AssetValidationError(ValueError):
    pass

class AssetLockError(RuntimeError):
    pass


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _json(path):return json.loads(Path(path).read_text('utf-8-sig'))


def _atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        with temp.open('w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def _inside(path,parent):
    path=Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(Path(parent).resolve()):
        raise AssetValidationError('Asset path escaped its project/model root')
    return path


def _transport_policy():
    """Optional transport only: official HF metadata remains authoritative."""
    path=ROOT/'configs/download_transport_v1.json'
    if not path.exists():return None
    _inside(path,ROOT)
    raw=path.read_bytes()
    try:policy=json.loads(raw.decode('utf-8-sig'))
    except (ValueError,UnicodeError) as error:
        raise AssetValidationError('Invalid download transport policy JSON') from error
    if not isinstance(policy,dict) or policy.get('schema')!='verified_modelscope_transport_v1' or not isinstance(policy.get('sources'),list):
        raise AssetValidationError('Unsupported download transport policy schema')
    sources={}
    for source in policy['sources']:
        if not isinstance(source,dict):raise AssetValidationError('Transport source must be an object')
        repo=source.get('repo_id');revision=source.get('hf_revision');msrepo=source.get('modelscope_repo_id');msrevision=source.get('modelscope_revision')
        if not isinstance(repo,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo) or any(x in {'.','..'} for x in repo.split('/')):
            raise AssetValidationError('Unsafe transport repository identifier')
        if msrepo!=repo:raise AssetValidationError('ModelScope repository must exactly match the HF repository')
        if any(not isinstance(x,str) or not re.fullmatch(r'[0-9a-fA-F]{40}',x) for x in (revision,msrevision)):
            raise AssetValidationError('Both transport repository revisions must be fixed full commits')
        key=(repo,revision)
        if key in sources:raise AssetValidationError('Duplicate transport source identity')
        files=source.get('files')
        if not isinstance(files,list) or not files:raise AssetValidationError('Transport source file inventory required')
        inventory={}
        for record in files:
            if not isinstance(record,dict):raise AssetValidationError('Transport file must be an object')
            name=record.get('path');size=record.get('size');digest=record.get('sha256')
            if not isinstance(name,str) or not name.endswith('.safetensors') or '\\' in name:
                raise AssetValidationError('ModelScope transport permits safetensors files only')
            relative=Path(name)
            if relative.is_absolute() or relative.drive or '..' in relative.parts or name in inventory:
                raise AssetValidationError('Unsafe or duplicate transport file path')
            if isinstance(size,bool) or not isinstance(size,int) or size<=0 or not isinstance(digest,str) or not re.fullmatch(r'[0-9a-fA-F]{64}',digest):
                raise AssetValidationError('Invalid transport file size or SHA256')
            inventory[name]={'size':size,'sha256':digest.lower()}
        sources[key]={'repo_id':repo,'revision':revision,'modelscope_revision':msrevision,'files':inventory}
    return {'sources':sources,'config_sha256':hashlib.sha256(raw).hexdigest(),
        'config_path':path.relative_to(ROOT).as_posix()}


def _transport_sources(repo,revision,descriptors):
    """Bind each opt-in MS artifact to the exact official HF file identity."""
    policy=_transport_policy()
    if policy is None:return {}
    source=policy['sources'].get((repo,revision))
    if source is None:return {}
    selected={}
    for name,record in source['files'].items():
        descriptor=descriptors.get(name)
        if descriptor is None or descriptor['size']!=record['size'] or not descriptor.get('sha256') or descriptor['sha256'].lower()!=record['sha256']:
            raise AssetValidationError('ModelScope transport file differs from authoritative HF size/SHA256 inventory')
        selected[name]={'transport_origin':'modelscope','modelscope_repo_id':repo,
            'modelscope_revision':source['modelscope_revision'],
            'transport_config_sha256':policy['config_sha256'],'transport_config_path':policy['config_path'],
            'url':f"https://modelscope.cn/api/v1/models/{repo}/repo?Revision={source['modelscope_revision']}&FilePath={quote(name,safe='')}"}
    return selected


def _metadata_paths(role):
    base=ROOT/'artifacts/audit'
    return base/f'{role}_official_metadata.json',base/f'{role}_official_metadata.provenance.json'


def _source_url(repo,revision):return f'https://huggingface.co/api/models/{repo}/revision/{revision}'


def _descriptors(data,repo,revision):
    if not re.fullmatch(r'[0-9a-fA-F]{40}',revision) or data.get('sha')!=revision:
        raise AssetValidationError('Official pinned metadata revision mismatch')
    if data.get('id',data.get('modelId'))!=repo:
        raise AssetValidationError('Official metadata model identity mismatch')
    siblings=data.get('siblings')
    if not isinstance(siblings,list):raise AssetValidationError('Official file inventory missing')
    selected={}
    for item in siblings:
        name=item.get('rfilename','')
        if not isinstance(name,str) or not any(fnmatch.fnmatchcase(name,p) for p in PATTERNS):continue
        relative=Path(name)
        if not name or relative.is_absolute() or relative.drive or '..' in relative.parts or name in selected:
            raise AssetValidationError('Unsafe/duplicate upstream asset path')
        size=item.get('size');lfs=item.get('lfs') or {};digest=lfs.get('sha256');blob=item.get('blobId')
        if isinstance(size,bool) or not isinstance(size,int) or size<0:
            raise AssetValidationError('Official asset size missing')
        if digest is not None:
            if not re.fullmatch(r'[0-9a-fA-F]{64}',digest) or lfs.get('size',size)!=size:
                raise AssetValidationError('Invalid official LFS identity')
        elif not isinstance(blob,str) or not re.fullmatch(r'[0-9a-fA-F]{40}',blob):
            raise AssetValidationError('Official Git blob identity missing')
        selected[name]={'rfilename':name,'size':size,'sha256':digest,'git_blob_sha1':None if digest else blob}
    if not selected:raise AssetValidationError('No locked inference assets')
    return selected


def _cached_metadata(repo,revision,role):
    path,anchor=_metadata_paths(role)
    if not path.exists():
        if anchor.exists():raise AssetValidationError('Metadata anchor exists but content is missing')
        return None
    _inside(path,ROOT);data=_json(path);_descriptors(data,repo,revision);digest=_sha(path)
    if anchor.exists():
        a=_json(anchor)
        if (a.get('repo_id'),a.get('revision'),a.get('sha256'),a.get('source_url'))!=(repo,revision,digest,_source_url(repo,revision)):
            raise AssetValidationError('Cached official metadata digest/identity changed')
        return data
    receipt=ROOT/'artifacts/audit'/f'{role}_model_download.json'
    if receipt.exists():
        r=_json(receipt)
        if (r.get('repo_id'),r.get('revision'))!=(repo,revision):
            raise AssetValidationError('Existing source receipt belongs to another model revision')
        relative=path.relative_to(ROOT).as_posix()
        if r.get('official_metadata_path')!=relative or r.get('official_metadata_sha256')!=digest:
            raise AssetValidationError('Completed source receipt does not bind cached official metadata')
        return data
    # Legacy incomplete caches have no durable digest anchor. One official small
    # request is needed to authenticate their selected-file descriptors.
    return None


def metadata(repo,revision,role,*,network_attempts=10):
    if isinstance(network_attempts,bool) or not isinstance(network_attempts,int) or network_attempts<=0:raise ValueError('Positive metadata retry count required')
    cached=_cached_metadata(repo,revision,role)
    if cached is not None:return cached
    target,anchor=_metadata_paths(role)
    legacy=_json(target) if target.exists() else None
    for attempt in range(network_attempts):
        response=None
        try:
            response=requests.get(_source_url(repo,revision),params={'blobs':'true'},timeout=(20,60))
            response.raise_for_status();result=response.json();desc=_descriptors(result,repo,revision)
            if legacy is not None:
                if _descriptors(legacy,repo,revision)!=desc:
                    raise AssetValidationError('Legacy cached file identities differ from official pinned metadata')
            else:_atomic_json(target,result)
            _atomic_json(anchor,{'schema':'pinned-official-metadata-v1','repo_id':repo,'revision':revision,
                'sha256':_sha(target),'source_url':_source_url(repo,revision),'verified_at_unix':time.time()})
            return legacy if legacy is not None else result
        except requests.RequestException:
            if attempt==network_attempts-1:raise
            time.sleep(min((attempt+1)*5,30))
        finally:
            if response is not None:response.close()


def _verify_file(path,descriptor,record):
    if path.stat().st_size!=descriptor['size'] or record.get('bytes')!=descriptor['size']:
        raise AssetValidationError('Completed asset size changed')
    sha=hashlib.sha256();git=hashlib.sha1();git.update(f"blob {descriptor['size']}\0".encode('ascii'))
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):sha.update(block);git.update(block)
    actual=sha.hexdigest()
    if actual!=record.get('sha256') or (descriptor['sha256'] and actual!=descriptor['sha256']):
        raise AssetValidationError('Completed model asset SHA256 changed')
    if descriptor['git_blob_sha1'] and git.hexdigest()!=descriptor['git_blob_sha1']:
        raise AssetValidationError('Completed model Git blob identity changed')
    if record.get('lfs_sha256')!=descriptor['sha256']:
        raise AssetValidationError('Completed receipt LFS identity differs from official metadata')


def _has_matching_completed_receipt(role,repo,revision,alias=None):
    """Cheap scheduling hint only; never substitutes for complete hash validation."""
    path=ROOT/'artifacts/audit'/f'{role}_model_download.json'
    try:
        r=_json(path)
        dest=(ROOT/'models'/repo.replace('/','--')).resolve()
        return (r.get('repo_id'),r.get('revision'),Path(r.get('path','')).resolve())==(repo,revision,dest) and bool(
            r.get('completed_at_unix') and r.get('files')) and (alias is None or r.get('model_alias')==alias)
    except (OSError,ValueError,TypeError,AttributeError):
        return False


def verified_completed_receipt(role,repo,revision,alias=None):
    path=ROOT/'artifacts/audit'/f'{role}_model_download.json'
    if not path.exists():return None
    r=_json(path);dest=(ROOT/'models'/repo.replace('/','--')).resolve()
    if (r.get('repo_id'),r.get('revision'),Path(r.get('path','')).resolve())!=(repo,revision,dest):
        raise AssetValidationError('Existing completed model receipt identity/path mismatch')
    if not r.get('completed_at_unix') or not r.get('files'):
        raise AssetValidationError('Existing model receipt is not completed')
    if alias is not None and r.get('model_alias')!=alias:raise AssetValidationError('Completed model alias changed')
    started=time.perf_counter()
    print(json.dumps({'event':'verify_completed_model_started','role':role,'repo':repo,
        'revision':revision,'files':len(r['files']),'full_hash_validation_required':True}),flush=True)
    source=_cached_metadata(repo,revision,role)
    if source is None:raise AssetValidationError('Completed receipt has no verifiable official metadata')
    descriptors=_descriptors(source,repo,revision);inventory={}
    for row in r['files']:
        relative=Path(row['path'])
        if relative.is_absolute() or relative.drive or '..' in relative.parts:
            raise AssetValidationError('Completed inventory path is not project-relative')
        target=_inside(ROOT/relative,dest);name=target.relative_to(dest).as_posix()
        if name in inventory:raise AssetValidationError('Duplicate completed inventory entries')
        inventory[name]=(target,row)
    if set(inventory)!=set(descriptors):raise AssetValidationError('Completed model inventory is incomplete or differs from pinned metadata')
    checked_bytes=0
    for index,(name,(target,row)) in enumerate(inventory.items(),1):
        _verify_file(target,descriptors[name],row)
        checked_bytes+=row['bytes']
        print(json.dumps({'event':'verify_completed_model_file_completed','role':role,'repo':repo,
            'file':name,'files_checked':index,'files_total':len(inventory),'checked_bytes':checked_bytes}),flush=True)
    print(json.dumps({'event':'verify_completed_model_completed','role':role,'repo':repo,'revision':revision,
        'files':len(inventory),'checked_bytes':checked_bytes,'elapsed_s':time.perf_counter()-started}),flush=True)
    return r


def _reclaim_stale_lock(lock,identity):
    lock=Path(lock)
    if not lock.exists():return False
    _inside(lock,ROOT);original=lock.read_bytes()
    try:owner=json.loads(original.decode('utf-8-sig'))
    except (ValueError,UnicodeDecodeError):raise AssetLockError('Unparseable lock requires explicit reconciliation') from None
    if any(owner.get(k)!=v for k,v in identity.items()):raise AssetLockError('Lock identity differs from pinned model/file')
    pid=owner.get('pid')
    if isinstance(pid,bool) or not isinstance(pid,int) or pid<=0:raise AssetLockError('Lock owner PID is invalid')
    if psutil.pid_exists(pid):raise AssetLockError('Lock owner still exists; never steal an active lock')
    if lock.read_bytes()!=original:raise AssetLockError('Lock changed during reconciliation')
    lock.unlink()
    print(json.dumps({'event':'reconciled_dead_download_lock','path':str(lock),'owner_pid':pid}),flush=True)
    return True


def migrate_partial(dest,expected,expected_size):
    stable=dest.with_name(dest.name+'.part')
    if stable.exists() or not expected:return None
    cache=dest.parent/'.cache/huggingface/download'
    if not cache.exists():return None
    candidates=[p for p in cache.glob(f'*.{expected}.*.incomplete') if p.is_file() and not p.is_symlink()
        and p.resolve().is_relative_to(ROOT.resolve()) and 0<p.stat().st_size<=expected_size]
    if not candidates:return None
    source=max(candidates,key=lambda p:p.stat().st_size);shutil.copyfile(source,stable)
    return {'source':source.relative_to(ROOT).as_posix(),'bytes':stable.stat().st_size,'full_file_hash_still_required':True}


def _target_has_complete_size(dest,descriptor):
    """Stat-only transport hint; the worker still hashes every target in full."""
    target=_inside(Path(dest)/descriptor['rfilename'],dest)
    try:return target.is_file() and target.stat().st_size==descriptor['size']
    except OSError:return False


def download_model(role,repo,revision,alias=None):
    complete=verified_completed_receipt(role,repo,revision,alias)
    if complete is not None:
        print(json.dumps({'event':'verified_model_reused_offline','role':role,'repo':repo,'revision':revision,
            'files':len(complete['files']),'bytes':sum(x['bytes'] for x in complete['files'])}),flush=True)
        return complete
    source=metadata(repo,revision,role);descriptors=_descriptors(source,repo,revision)
    transport_sources=_transport_sources(repo,revision,descriptors)
    dest=_inside(ROOT/'models'/repo.replace('/','--'),ROOT);dest.mkdir(parents=True,exist_ok=True)
    def one(descriptor):
        relative=Path(descriptor['rfilename']);target=_inside(dest/relative,dest)
        target.parent.mkdir(parents=True,exist_ok=True)
        expected=descriptor['sha256'];size=descriptor['size'];blob=descriptor['git_blob_sha1']
        _reclaim_stale_lock(Path(str(target)+'.lock'),{'expected_size':size,'sha256':expected,'git_blob_sha1':blob})
        migrated=migrate_partial(target,expected,size);last_reported=-64*1024*1024
        def progress(event):
            nonlocal last_reported
            if event['event'] in {'start','verified','interrupted'} or event['bytes']-last_reported>=64*1024*1024:
                print(json.dumps({'role':role,'file':relative.as_posix(),**event}),flush=True);last_reported=event['bytes']
        transport=transport_sources.get(relative.as_posix(),{'transport_origin':'huggingface',
            'url':f'https://huggingface.co/{repo}/resolve/{revision}/{quote(relative.as_posix(),safe="/")}'})
        origin={key:value for key,value in transport.items() if key!='url'}
        print(json.dumps({'event':'download_source_selected','role':role,'repo':repo,'hf_revision':revision,
            'file':relative.as_posix(),'source_url':transport['url'],**origin}),flush=True)
        session=None
        try:
            options={}
            if transport['transport_origin']=='modelscope':
                session=requests.Session();session.trust_env=False
                options['session']=session
            result=resumable_download(transport['url'],target,size,sha256=expected,git_blob_sha1=blob,
                retries=10,progress_callback=progress,**options)
        finally:
            if session is not None:session.close()
        selected_origin=origin['transport_origin']
        if result.get('cache_hit') is True:origin['transport_origin']='existing_local_file'
        actual_transport={'selected_transport_origin':selected_origin,
            **{key:result.get(key) for key in ('cache_hit','resumed_bytes','downloaded_bytes')},
            'preexisting_prefix_bytes':result.get('resumed_bytes')}
        return {'path':target.relative_to(ROOT).as_posix(),'bytes':target.stat().st_size,'sha256':result['sha256'],
            'lfs_sha256':expected,'git_blob_sha1':blob,'download_transport':'persistent_http_range_v1','migration':migrated,
            **origin,**actual_transport}
    ordered_descriptors=sorted(descriptors.values(),key=lambda descriptor:_target_has_complete_size(dest,descriptor))
    print(json.dumps({'event':'download_file_schedule','role':role,'incomplete_targets_first':True,
        'all_targets_full_hash_validation_required':True,
        'files':[descriptor['rfilename'] for descriptor in ordered_descriptors]}),flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:files=list(executor.map(one,ordered_descriptors))
    metadata_path,_=_metadata_paths(role)
    receipt={'role':'member' if alias else role,'model_alias':alias,'repo_id':repo,'revision':revision,'path':str(dest),
        'files':sorted(files,key=lambda x:x['path']),'completed_at_unix':time.time(),
        'authoritative_metadata_endpoint':'https://huggingface.co/api/models/',
        'official_metadata_path':metadata_path.relative_to(ROOT).as_posix(),'official_metadata_sha256':_sha(metadata_path)}
    _atomic_json(ROOT/'artifacts/audit'/f'{role}_model_download.json',receipt)
    print(json.dumps({'event':'pinned_model_complete','role':role,'repo':repo,'files':len(files),'bytes':sum(x['bytes'] for x in files)}),flush=True)
    return receipt


def locked_download_model(role,repo,revision,alias=None):
    dest=_inside(ROOT/'models'/repo.replace('/','--'),ROOT);dest.mkdir(parents=True,exist_ok=True)
    lock=dest/'.model_download.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        _reclaim_stale_lock(lock,{'repo':repo,'revision':revision})
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    token=uuid.uuid4().hex
    try:
        os.write(fd,json.dumps({'pid':os.getpid(),'repo':repo,'revision':revision,'owner_token':token}).encode());os.fsync(fd)
        return download_model(role,repo,revision,alias)
    finally:
        os.close(fd)
        try:
            if lock.exists() and _json(lock).get('owner_token')==token:lock.unlink()
        except (OSError,ValueError):pass


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['aggregator','members','all'],required=True);a=p.parse_args(argv)
    plan=[]
    if a.role in ('aggregator','all'):
        plan.append(('aggregator','Qwen/Qwen2.5-7B-Instruct','a09a35458c702b33eeacc393d103063234e8bc28',None))
    if a.role in ('members','all'):
        for item in _json(ROOT/'artifacts/audit/candidate_model_identities.json'):
            plan.append(('member_'+item['alias'].lower(),item['repo_id'],item['revision'],item['alias']))
    # This orders transport only. Keep every requested model, and do not mutate
    # the pool or qualification order. All completed assets still get full hashes
    # before this process can return success to download_with_retry.py.
    plan=sorted(plan,key=lambda entry:_has_matching_completed_receipt(*entry))
    print(json.dumps({'event':'download_schedule','incomplete_models_first':True,
        'completed_models_full_revalidation_required':True,'roles':[entry[0] for entry in plan]}),flush=True)
    for entry in plan:locked_download_model(*entry)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (AssetValidationError,AssetLockError,DownloadIntegrityError,DownloadLockedError,RangeProtocolError) as error:
        print(json.dumps({'status':'STOPPED_ASSET_VALIDATION','error_type':type(error).__name__,'reason':str(error)}),file=sys.stderr)
        raise SystemExit(3) from None
