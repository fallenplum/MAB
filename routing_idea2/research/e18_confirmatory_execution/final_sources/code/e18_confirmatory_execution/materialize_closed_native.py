"""Restore SHA-bound native archives byte-for-byte into a private D: NTFS view.

Canonical archives and remote paths are read-only. Only the three original
owner files are published. The EXPORT_MANIFEST remains inside its archive.
No reference provider, scorer, model, subprocess or remote client is imported.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time
import uuid


OWNER_FILES={"ATTEMPT_STATE.json","LAST_TASK_RECEIPT.json","records.jsonl"}
STAGES={"member_generation","aggregation"}
MAX_FILE_BYTES=64*1024*1024
MAX_ARCHIVE_CONTENT_BYTES=256*1024*1024
FORBIDDEN={"gold","gold_answer","reference","references","reference_answer","reference_answers",
    "ground_truth","target","targets","reward","rewards","score","scores","label","labels","is_correct"}


def io_path(path):
    """Use Win32 extended-length I/O without changing logical receipt paths."""
    path=Path(os.path.abspath(path))
    if os.name!="nt" or str(path).startswith("\\\\?\\"): return path
    text=str(path)
    return Path("\\\\?\\UNC\\"+text[2:] if text.startswith("\\\\") else "\\\\?\\"+text)


def file_sha(path):
    result=hashlib.sha256()
    with io_path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None


def reject_protected_properties(raw):
    """Inspect JSON keys without decoding generated text or other values."""
    position=0
    while position<len(raw):
        if raw[position]!=34:
            position+=1
            continue
        start=position
        position+=1
        while position<len(raw):
            if raw[position]==92: position+=2
            elif raw[position]==34:
                position+=1
                break
            else: position+=1
        else: raise ValueError("Unterminated JSON string in native evidence")
        end=position
        while position<len(raw) and raw[position] in b" \r\n\t": position+=1
        if position<len(raw) and raw[position]==58 and json.loads(raw[start:end]).casefold() in FORBIDDEN:
            raise ValueError("Gold/reward payload refused before value decoding or materialization")


def strict_json(raw):
    reject_protected_properties(raw)
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError("Duplicate metadata JSON key")
            result[key]=value
        return result
    def invalid(value): raise ValueError("Non-finite metadata JSON number")
    return json.loads(raw,object_pairs_hook=unique,parse_constant=invalid)


def reject_links(path):
    """Reject every existing ancestor symlink/junction before any write."""
    path=Path(os.path.abspath(path))
    for current in reversed([path,*path.parents]):
        try: info=io_path(current).lstat()
        except FileNotFoundError: continue
        if stat.S_ISLNK(info.st_mode) or getattr(info,"st_file_attributes",0)&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",1024):
            raise ValueError("Symlink or reparse path refused")
    return path


def regular_file(path):
    path=reject_links(path)
    info=io_path(path).lstat()
    if not stat.S_ISREG(info.st_mode): raise ValueError("Regular source/target file required")
    return info


def stat_signature(info):
    return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns


def read_reference(reference):
    if not isinstance(reference,dict) or set(reference)!={"path","sha256"} or not is_sha(reference["sha256"]):
        raise ValueError("An explicit catalog path/SHA reference is required")
    path=Path(reference["path"])
    before=regular_file(path)
    raw=io_path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=reference["sha256"] or stat_signature(before)!=stat_signature(regular_file(path)):
        raise ValueError("Catalog bytes differ or changed during read")
    return strict_json(raw)


def native_path(name,scope_sha):
    pure=PurePosixPath(name)
    parts=pure.parts
    if pure.is_absolute() or "\\" in name or name!=pure.as_posix() or len(parts)!=4:
        raise ValueError("Noncanonical or escaping native archive path")
    if parts[0]!=scope_sha or parts[1] not in STAGES or not is_sha(parts[2]) or parts[3] not in OWNER_FILES:
        raise ValueError("Archive file is outside its exact four-component owner namespace")
    return parts


def catalogs_to_entries(catalog_refs,software_test_only):
    entries={}
    repeated=0
    scope=None
    for reference in catalog_refs:
        value=read_reference(reference)
        schema=value.get("schema") if isinstance(value,dict) else None
        if schema=="e18_completed_native_export_result_v1":
            if value.get("software_test_only") is not software_test_only:
                raise ValueError("Catalog production/software-test mode differs")
            if any(value.get(name)!=0 for name in ("new_model_calls","scorer_calls","gold_reads")):
                raise ValueError("Catalog is not a generation-output-only export result")
            if scope is not None and value["scope_sha256"]!=scope:
                raise ValueError("Export result scopes differ")
            scope=value["scope_sha256"]
            source=list(value.get("new_archives",[]))+list(value.get("recovered_archives",[]))
        elif schema=="e18_independent_closed_native_archive_audit_v1":
            if software_test_only or value.get("status")!="PASS_INDEPENDENT_STREAMING_NATIVE_AUDIT":
                raise ValueError("A completed production archive audit is required")
            if scope is not None and value["execution_scope_sha256"]!=scope:
                raise ValueError("Independent-audit and export-result scopes differ")
            scope=value["execution_scope_sha256"]
            source=[dict(path=row["filename"],sha256=row["sha256"],bytes=row["compressed_bytes"],
                tasks=row["native_tasks"],manifest_sha256=row["manifest_semantic_sha256"]) for row in value["archives"]]
        else:
            raise ValueError("Unsupported export-result/archive-audit catalog schema")
        for row in source:
            if not isinstance(row,dict) or not {"path","sha256"}<=set(row) or not is_sha(row["sha256"]):
                raise ValueError("A bound external archive SHA is required")
            name=PurePosixPath(row["path"]).name
            if re.fullmatch(r"native-[0-9a-f]{32}\.tar\.gz",name) is None:
                raise ValueError("Unexpected canonical archive filename")
            if name in entries:
                old=entries[name]
                for key in ("sha256","bytes","tasks","manifest_sha256"):
                    if key in old and key in row and old[key]!=row[key]:
                        raise ValueError("One archive has conflicting catalog metadata")
                old.update({key:value for key,value in row.items() if key not in old})
                repeated+=1
            else: entries[name]=dict(row)
    if scope is not None and not is_sha(scope): raise ValueError("Malformed export scope SHA")
    return scope,entries,repeated


def inspect_archive(path,expected,scope_sha,software_test_only):
    """Validate all bytes and metadata before allowing any restoration writes."""
    before=regular_file(path)
    if ("bytes" in expected and before.st_size!=expected["bytes"]) or file_sha(path)!=expected["sha256"]:
        raise ValueError("Canonical archive external SHA/size differs")
    files={}
    manifest=None
    total=0
    with tarfile.open(io_path(path),"r|gz") as archive:
        for member in archive:
            if not member.isfile() or not 0<=member.size<=MAX_FILE_BYTES:
                raise ValueError("Archive link, special file or excessive native member refused")
            if manifest is not None: raise ValueError("Archive has data after its closing manifest")
            raw=archive.extractfile(member).read(MAX_FILE_BYTES+1)
            if len(raw)!=member.size: raise ValueError("Archive member byte count differs")
            total+=len(raw)
            if total>MAX_ARCHIVE_CONTENT_BYTES: raise ValueError("Archive content exceeds the established audit bound")
            if member.name=="EXPORT_MANIFEST.json":
                manifest=strict_json(raw)
                continue
            native_path(member.name,scope_sha)
            if member.name in files: raise ValueError("Duplicate path inside one archive")
            reject_protected_properties(raw)
            files[member.name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    if manifest is None: raise ValueError("Archive closing manifest is absent")
    if not (manifest.get("schema")=="e18_closed_native_export_manifest_v1" and manifest.get("scope_sha256")==scope_sha
        and manifest.get("software_test_only") is software_test_only and manifest.get("original_files_retained") is True
        and all(manifest.get(key)==0 for key in ("new_model_calls","scorer_calls","gold_reads"))):
        raise ValueError("Archive manifest schema/scope/mode/side effects differ")
    if path.name!="native-"+str(manifest.get("archive_id"))+".tar.gz":
        raise ValueError("Archive filename and manifest identity differ")
    if "manifest_sha256" in expected and digest(manifest)!=expected["manifest_sha256"]:
        raise ValueError("External semantic manifest SHA differs")
    original_root=manifest.get("owned_root")
    if not isinstance(original_root,str) or not PurePosixPath(original_root).is_absolute() or ".." in PurePosixPath(original_root).parts:
        raise ValueError("Original owned-root metadata is invalid")
    declared={row["path"]:row for row in manifest["files"]}
    if len(declared)!=len(manifest["files"]) or set(declared)!=set(files):
        raise ValueError("Manifest file inventory is duplicated or incomplete")
    for name,actual in files.items():
        if any(declared[name].get(key)!=value for key,value in actual.items()):
            raise ValueError("Manifest file SHA/size differs")
    tasks={}
    task_paths=set()
    for task in manifest["tasks"]:
        key=task.get("task_key")
        stage=task.get("stage")
        if not isinstance(key,list) or len(key)!=3 or stage not in STAGES:
            raise ValueError("Malformed manifest native task")
        task_id=digest([scope_sha,stage,key])
        if task.get("task_id")!=task_id or task_id in tasks:
            raise ValueError("Duplicate or inconsistent manifest task identity")
        prefix=scope_sha+"/"+stage+"/"+digest(key)+"/"
        paths={prefix+name for name in OWNER_FILES}
        if not paths<=set(files): raise ValueError("Native owner does not contain the three exact files")
        task_paths.update(paths)
        tasks[task_id]={key:value for key,value in task.items() if key!="manifest_sha256s"}
    if set(files)!=task_paths or not 1<=len(tasks)<=1000 or ("tasks" in expected and len(tasks)!=expected["tasks"]):
        raise ValueError("Native task/file count differs from the bound catalog")
    if stat_signature(before)!=stat_signature(regular_file(path)) or file_sha(path)!=expected["sha256"]:
        raise ValueError("Canonical archive changed during preflight")
    return dict(files=files,tasks=tasks,manifest_sha256=digest(manifest),original_owned_root=original_root,
        archive_sha256=expected["sha256"],archive_bytes=before.st_size,stat_signature=stat_signature(before))


def destination_metadata(path,software_test_only):
    path=reject_links(path)
    if not software_test_only:
        if os.name!="nt" or path.drive.upper()!="D:":
            raise ValueError("Production restoration requires the explicitly chosen D: private NTFS root")
        from ctypes import wintypes
        filesystem=ctypes.create_unicode_buffer(64)
        information=ctypes.windll.kernel32.GetVolumeInformationW
        information.argtypes=(wintypes.LPCWSTR,wintypes.LPWSTR,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),ctypes.POINTER(wintypes.DWORD),wintypes.LPWSTR,wintypes.DWORD)
        information.restype=wintypes.BOOL
        if not information(path.anchor,None,0,None,None,None,filesystem,len(filesystem)):
            raise OSError("Unable to verify the destination filesystem")
        if filesystem.value.upper()!="NTFS": raise ValueError("Destination filesystem must be NTFS")
        return path,filesystem.value
    return path,"SOFTWARE_FIXTURE_FILESYSTEM"


def publish_file(root,parts,source,expected,crash_hook=None):
    target=root.joinpath(*parts)
    reject_links(target)
    if io_path(target).exists():
        before=regular_file(target)
        if before.st_size!=expected["bytes"] or file_sha(target)!=expected["sha256"] or stat_signature(before)!=stat_signature(regular_file(target)):
            raise ValueError("Existing destination conflicts with canonical bytes; no overwrite performed")
        return "reused"
    io_path(target.parent).mkdir(parents=True,exist_ok=True)
    reject_links(target)
    temporary=target.with_name(".m-"+uuid.uuid4().hex+".tmp")
    checksum=hashlib.sha256()
    size=0
    with io_path(temporary).open("xb") as stream:
        for block in iter(lambda:source.read(1024*1024),b""):
            checksum.update(block)
            size+=len(block)
            if size>expected["bytes"]: raise ValueError("Native extraction exceeded its bound size")
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    if size!=expected["bytes"] or checksum.hexdigest()!=expected["sha256"]:
        raise ValueError("Staged native bytes differ; final path was not published")
    if crash_hook: crash_hook("before_file_publish",temporary,target)
    if regular_file(temporary).st_size!=expected["bytes"] or file_sha(temporary)!=expected["sha256"]:
        raise ValueError("Staged native file changed before atomic publication")
    reject_links(target)
    try:
        # Atomic create-if-absent. Existing files are never replaced or truncated.
        os.link(io_path(temporary),io_path(target),follow_symlinks=False)
        outcome="created"
    except FileExistsError:
        if regular_file(target).st_size!=expected["bytes"] or file_sha(target)!=expected["sha256"]:
            raise ValueError("Concurrent destination publication conflicts")
        outcome="reused"
    if regular_file(target).st_size!=expected["bytes"] or file_sha(target)!=expected["sha256"]:
        raise ValueError("Published native file verification failed")
    # Remove only this call's explicitly created staging link in the D: root.
    if not temporary.is_relative_to(root): raise ValueError("Staging path escaped the destination")
    io_path(temporary).unlink()
    return outcome


def materialize(catalog_refs,archive_directory,destination_root,*,software_test_only=False,crash_hook=None):
    started=time.perf_counter()
    archive_directory=reject_links(archive_directory)
    destination,filesystem=destination_metadata(destination_root,software_test_only)
    if destination.is_relative_to(archive_directory) or archive_directory.is_relative_to(destination):
        raise ValueError("Canonical archive storage and native runtime view must be separate")
    scope,entries,repeated_catalog_entries=catalogs_to_entries(catalog_refs,software_test_only)
    inspections={}
    planned_files={}
    planned_tasks={}
    attempts={}
    duplicate_files=duplicate_tasks=0
    original_root=None
    for name,expected in sorted(entries.items()):
        info=inspect_archive(archive_directory/name,expected,scope,software_test_only)
        if original_root is not None and info["original_owned_root"]!=original_root:
            raise ValueError("Archives have different original owned roots")
        original_root=info["original_owned_root"]
        inspections[name]=info
        for path,metadata in info["files"].items():
            if path in planned_files:
                if planned_files[path]!=metadata: raise ValueError("Cross-archive path has conflicting native bytes")
                duplicate_files+=1
            else: planned_files[path]=metadata
        for task_id,metadata in info["tasks"].items():
            if task_id in planned_tasks:
                if planned_tasks[task_id]!=metadata: raise ValueError("Cross-archive task has conflicting native identity")
                duplicate_tasks+=1
            else: planned_tasks[task_id]=metadata
            attempt=metadata["physical_attempt_id"]
            if attempt in attempts and attempts[attempt]!=task_id:
                raise ValueError("One physical execution cannot identify different native tasks")
            attempts[attempt]=task_id
    # Preflight all target conflicts before creating any final native file.
    for name,expected in planned_files.items():
        target=reject_links(destination.joinpath(*native_path(name,scope)))
        if io_path(target).exists() and (regular_file(target).st_size!=expected["bytes"] or file_sha(target)!=expected["sha256"]):
            raise ValueError("Existing destination conflict found during preflight")
    created=reused=0
    processed=set()
    io_path(destination).mkdir(parents=True,exist_ok=True)
    for name,inspection in inspections.items():
        path=archive_directory/name
        before=regular_file(path)
        if file_sha(path)!=inspection["archive_sha256"]: raise ValueError("Archive changed before restoration")
        with tarfile.open(io_path(path),"r|gz") as archive:
            for member in archive:
                if member.name=="EXPORT_MANIFEST.json": continue
                if member.name in processed: continue
                outcome=publish_file(destination,native_path(member.name,scope),archive.extractfile(member),
                    planned_files[member.name],crash_hook)
                created+=outcome=="created"
                reused+=outcome=="reused"
                processed.add(member.name)
        if stat_signature(before)!=stat_signature(regular_file(path)) or file_sha(path)!=inspection["archive_sha256"]:
            raise ValueError("Canonical archive changed during restoration")
    if processed!=set(planned_files): raise ValueError("Restoration did not cover every planned native file")
    return dict(schema="e18_closed_native_materialization_receipt_v1",
        status="PASS_BYTE_EXACT_PRIVATE_RUNTIME_VIEW" if planned_files else "NO_BOUND_ARCHIVES_TO_MATERIALIZE",
        catalog_refs=catalog_refs,canonical_archive_directory=str(archive_directory),native_read_root=str(destination),
        destination_filesystem=filesystem,scope_sha256=scope,original_owned_root_preserved_in_payloads=original_root,
        archives=len(inspections),archive_receipts=[dict(filename=name,**{key:value for key,value in info.items()
            if key in ("archive_sha256","archive_bytes","manifest_sha256")}) for name,info in inspections.items()],
        unique_native_tasks=len(planned_tasks),unique_physical_attempts=len(attempts),unique_native_files=len(planned_files),
        created_files=created,existing_identical_files_reused=reused,repeated_catalog_archive_entries_reused=repeated_catalog_entries,
        identical_cross_archive_tasks_reused=duplicate_tasks,identical_cross_archive_paths_reused=duplicate_files,
        native_bytes=sum(row["bytes"] for row in planned_files.values()),
        canonical_archives_modified=0,remote_operations=0,scope_or_receipt_or_native_json_rewrites=0,
        gold_sources_read=0,scoring_calls=0,model_calls=0,software_test_only=software_test_only,
        source_sha256=file_sha(__file__),wall_s=time.perf_counter()-started)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--catalog-ref",nargs=2,action="append",required=True,metavar=("PATH","SHA256"))
    parser.add_argument("--archive-directory",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    reject_links(args.receipt)
    if io_path(args.receipt).exists(): raise FileExistsError("Engineering receipts refuse overwrite")
    io_path(args.receipt.parent).mkdir(parents=True,exist_ok=True)
    reject_links(args.receipt)
    try:
        report=materialize([dict(path=path,sha256=checksum) for path,checksum in args.catalog_ref],args.archive_directory,args.destination)
    except Exception as error:
        report=dict(schema="e18_closed_native_materialization_receipt_v1",status="FAILED_REQUIRES_REVIEW",
            error_type=type(error).__name__,reason=str(error),canonical_archives_modified=0,remote_operations=0,
            gold_sources_read=0,scoring_calls=0,model_calls=0,source_sha256=file_sha(__file__),
            catalog_refs=[dict(path=path,sha256=checksum) for path,checksum in args.catalog_ref],
            canonical_archive_directory=str(args.archive_directory),native_read_root=str(args.destination),
            native_runtime_may_be_partial=True)
        with io_path(args.receipt).open("x",encoding="utf-8") as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write("\n")
        raise
    with io_path(args.receipt).open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:report[key] for key in ("status","archives","unique_native_tasks","unique_native_files","created_files","existing_identical_files_reused","native_bytes")},ensure_ascii=False))
