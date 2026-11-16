"""Append-only P2 manifest/journal evidence registry, independent of live ownership.

Historical full manifests remain evidence sources after disjoint successors are
registered. Dispatch plans and the launcher separately enforce live ownership.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "p2_manifest_journal_registry_v1"
SCOPE_SHA = "b355581aaaab1a581e76dab35107a897948da7b4086a1d9e2a33be4a3e1cd1b9"
MODEL_SHA = "b8b786eeb0aaac6f869136f5353186c13f78aba0a716ebf0bcad145532d01f92"
COMMON_FIELDS = ("scope_ref", "approval_ref", "owned_output_root", "question_records_ref", "adapter_ref", "asset_root")


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def reference(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=file_sha(path))


def checked(reference_value,cache):
    if set(reference_value)!={"path","sha256"}:
        raise ValueError("An exact path and SHA reference is required")
    path=Path(reference_value["path"]).resolve(strict=True)
    identity=(str(path),reference_value["sha256"])
    if identity not in cache:
        if file_sha(path)!=reference_value["sha256"]:
            raise ValueError("Referenced file changed "+str(path))
        cache[identity]=path
    return path


def iter_complete_events(path):
    """Ignore only a final incomplete line, retaining complete-line corruption."""
    path=Path(path)
    if not path.is_file():
        return
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                break
            if raw.strip():
                yield json.loads(raw)


def context(common,software_test_only,cache):
    scope=read_json(checked(common["scope_ref"],cache))
    approval=read_json(checked(common["approval_ref"],cache))
    if (scope.get("schema"),scope.get("role"),scope.get("cohort"))!=("p2_scope_contract_v1","execution","e20_p2_confirm"):
        raise ValueError("Original P2 execution scope required")
    scope_sha=digest(scope)
    if not software_test_only and (scope_sha!=SCOPE_SHA or scope["bindings"].get("final_model_sha256")!=MODEL_SHA):
        raise ValueError("Production scope or frozen final models differ")
    if (approval.get("schema"),approval.get("status"),approval.get("cohort"))!=("p2_scope_approval_v1","APPROVED","e20_p2_confirm"):
        raise PermissionError("Bound P2 approval required")
    if approval.get("software_test_only") is not software_test_only or approval.get("scope_sha256_by_role",{}).get("execution")!=scope_sha:
        raise PermissionError("Approval scope or mode differs")
    if approval.get("bindings")!=scope.get("bindings") or approval.get("approved_output_root")!=common["owned_output_root"]:
        raise PermissionError("Approval bindings or output root differ")
    if any(approval.get("execution_limits",{}).get(stage,-1)<scope["execution_limits"][stage] for stage in ("member_generation","aggregation")):
        raise PermissionError("Approval does not cover the original stage limits")
    for name in ("question_records_ref","adapter_ref"):
        checked(common[name],cache)
    required={"member_generation":set(map(tuple,scope["required_member_keys"])),
              "aggregation":set(map(tuple,scope["required_aggregate_keys"]))}
    return scope_sha,required


def validate_entry(spec,common,software_test_only,required,backends,cache):
    if set(spec)!={"worker_id","manifest_ref","worker_source_ref"} or not isinstance(spec["worker_id"],str) or not spec["worker_id"]:
        raise ValueError("An explicit worker ID, manifest and worker-source reference required")
    manifest=read_json(checked(spec["manifest_ref"],cache))
    checked(spec["worker_source_ref"],cache)
    if {name:manifest.get(name) for name in COMMON_FIELDS}!=common or manifest.get("software_test_only") is not software_test_only:
        raise ValueError("Manifest differs from common scope, approval, input or execution identity")
    stage=manifest.get("stage")
    if stage not in required:
        raise ValueError("Only original member and aggregation stages are supported")
    profile=manifest.get("member_index") if stage=="member_generation" else None
    if stage=="member_generation" and (type(profile) is not int or profile not in range(6)):
        raise ValueError("A frozen six-model member index required")
    profile_key=f"member{profile}" if profile is not None else "aggregation"
    if profile_key in backends and manifest.get("backend")!=backends[profile_key]:
        raise ValueError("A new entry changed the frozen model/loading profile")
    keys=list(map(tuple,read_json(checked(manifest["task_keys_ref"],cache))))
    if any(len(key)!=3 or not isinstance(key[0],str) or type(key[1]) is not int
           or key[1] not in (42,137,2024) or type(key[2]) is not int for key in keys):
        raise ValueError("Exact original question, generation seed and integer action keys required")
    if len(keys)!=len(set(keys)):
        raise ValueError("Duplicate semantic keys inside one manifest")
    if not set(keys)<=required[stage] or (profile is not None and any(key[2]!=profile for key in keys)):
        raise ValueError("Manifest keys exceed its frozen stage/model support")
    return dict(spec,manifest=manifest,stage=stage,member_index=profile,task_keys=keys,
                journal_path=str(Path(common["owned_output_root"])/"_dispatcher"/spec["manifest_ref"]["sha256"] /"EVENTS.jsonl"))


def event_row(event,seq,previous):
    row=dict(event,seq=seq,previous_event_sha256=previous)
    return dict(row,event_sha256=digest(row))


def load_registry(path):
    raw=Path(path).read_bytes()
    rows=list(iter_complete_events(path))
    if not rows or rows[0].get("event")!="INIT" or rows[0].get("schema")!=SCHEMA:
        raise ValueError("An initialized P2 registry is required")
    previous="0"*64
    for seq,row in enumerate(rows):
        unsigned={key:value for key,value in row.items() if key!="event_sha256"}
        if row.get("seq")!=seq or row.get("previous_event_sha256")!=previous or row.get("event_sha256")!=digest(unsigned):
            raise ValueError("Registry append chain changed")
        previous=row["event_sha256"]
    header=rows[0]
    common=header["common"]
    software=header["software_test_only"]
    cache={}
    scope_sha,required=context(common,software,cache)
    entries=[]
    ids=set()
    manifests=set()
    for row in rows[1:]:
        if row.get("event")!="ADD":
            raise ValueError("Registry accepts only ADD references after initialization")
        spec=row["entry"]
        if spec["worker_id"] in ids or spec["manifest_ref"]["sha256"] in manifests:
            raise ValueError("Registry worker ID redefinition or manifest alias")
        entry=validate_entry(spec,common,software,required,header["profile_backends"],cache)
        entries.append(entry)
        ids.add(spec["worker_id"])
        manifests.add(spec["manifest_ref"]["sha256"])
    if not set(header["baseline_worker_ids"])<=ids:
        raise ValueError("The original six members and aggregation sources are incomplete")
    return dict(schema=SCHEMA,common=common,software_test_only=software,entries=entries,
                scope_canonical_sha256=scope_sha,trailing_partial_line=bool(raw and not raw.endswith(b"\n")),
                last_event_sha256=previous,event_count=len(rows),profile_backends=header["profile_backends"],
                event_hashes=[row["event_sha256"] for row in rows])


def semantic_key(task):
    stage=task.get("stage")
    if stage not in ("member_generation","aggregation"):
        raise ValueError("Unknown native task stage")
    index=task["member_index" if stage=="member_generation" else "combination_index"]
    return stage,(task["question_id"],task["generation_seed"],index)


def iter_successes(registry,stage=None):
    seen=set()
    for entry in registry["entries"]:
        manifest_sha=entry["manifest_ref"]["sha256"]
        if manifest_sha in seen or (stage is not None and entry["stage"]!=stage):
            continue
        seen.add(manifest_sha)
        allowed=set(entry["task_keys"])
        for event in iter_complete_events(entry["journal_path"]):
            if event.get("event")!="TASK_RESULT" or event.get("status")!="SUCCESS":
                continue
            event_stage,key=semantic_key(event["task"])
            if event_stage!=entry["stage"] or key not in allowed:
                raise ValueError("A success journal escaped its registered manifest keys")
            yield entry,event,key


@contextmanager
def append_lock(path):
    with Path(str(path)+".lock").open("a+b") as stream:
        if os.name=="nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name=="nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def create_registry(path,specs):
    cache={}
    manifests=[read_json(checked(spec["manifest_ref"],cache)) for spec in specs]
    if len(specs)!=7:
        raise ValueError("Initialize with exactly the six original member manifests and original aggregation manifest")
    common={name:manifests[0].get(name) for name in COMMON_FIELDS}
    software=manifests[0].get("software_test_only")
    if type(software) is not bool:
        raise ValueError("Explicit execution mode required")
    _,required=context(common,software,cache)
    backends={}
    for spec,manifest in zip(specs,manifests):
        entry=validate_entry(spec,common,software,required,{},cache)
        key=f"member{entry['member_index']}" if entry["stage"]=="member_generation" else "aggregation"
        if key in backends:
            raise ValueError("Duplicate original profile")
        expected_keys={key for key in required[entry["stage"]] if entry["member_index"] is None or key[2]==entry["member_index"]}
        if set(entry["task_keys"])!=expected_keys:
            raise ValueError("An original baseline manifest must retain all of its original execution keys")
        backends[key]=manifest["backend"]
    if set(backends)!={"member"+str(index) for index in range(6)}|{"aggregation"}:
        raise ValueError("Complete original model profiles required")
    if len({spec["worker_id"] for spec in specs})!=7 or len({spec["manifest_ref"]["sha256"] for spec in specs})!=7:
        raise ValueError("Duplicate original worker identity or manifest")
    header=dict(event="INIT",schema=SCHEMA,common=common,software_test_only=software,profile_backends=backends,
                baseline_worker_ids=[spec["worker_id"] for spec in specs],created_utc=datetime.now(timezone.utc).isoformat())
    rows=[event_row(header,0,"0"*64)]
    for seq,spec in enumerate(specs,1):
        rows.append(event_row(dict(event="ADD",entry=spec),seq,rows[-1]["event_sha256"]))
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False).encode()+b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return load_registry(path)


def original_specs(root):
    """Reference the seven existing actors once, without selecting latest aliases."""
    root=Path(root).resolve(strict=True)
    expected_manifests=(
        "601a22fe2808d5a7c122d22f200de55903a1530851a149b2c82532b0af3b3e95",
        "53ebebeb1ce784ff8ef93701937da09962823a7753fb22060bc514b4b4f462f1",
        "37c6fc83a12cf4c443054f8818a500f1049441dc97232f520788f342ad2ff282",
        "253ab5d7e4453788218aa3be8e0aba4710f59a089db9eff406d4ebcbb00fffde",
        "83d942d7bf1e6900158a5f12748f5bb1d1106a25ecd389b1e85a71c669737401",
        "e3cf523e4207d89e0246b30ddb8913910e09c675791bfbc48c287a346f963249",
        "2b1f634e7c8775a0222bb65284182d7db8fcc372bdcdb8d2b22231e21c3dffc2",
    )
    profiles=["member"+str(index) for index in range(6)]+["aggregation"]
    specs=[]
    for profile,expected in zip(profiles,expected_manifests):
        manifest_ref=reference(root/"manifests"/(profile+"_v2.json"))
        source="p2_member_worker.py" if profile!="aggregation" else "p2_aggregation_follower.py"
        source_ref=reference(root/"code"/source)
        expected_source=("58839460471e33536cd2b00274ede029b98355d355df2919b8bf3790f16b6c81"
                         if profile!="aggregation" else "95f6eb7c1502c98ececfd8f13f69a05220b7f03fabfa93ac24b79f534b374c67")
        if manifest_ref["sha256"]!=expected or source_ref["sha256"]!=expected_source:
            raise ValueError("An original production manifest or worker source changed")
        specs.append(dict(worker_id=profile+"-v2",manifest_ref=manifest_ref,worker_source_ref=source_ref))
    return specs


def add_manifest(path,spec):
    with append_lock(path):
        registry=load_registry(path)
        if registry["trailing_partial_line"]:
            raise RuntimeError("A partial registry append requires explicit recovery before another ADD")
        for old in registry["entries"]:
            if old["worker_id"]==spec["worker_id"]:
                if all(old[name]==spec[name] for name in ("worker_id","manifest_ref","worker_source_ref")):
                    return registry
                raise ValueError("Worker ID cannot be redefined")
            if old["manifest_ref"]["sha256"]==spec["manifest_ref"]["sha256"]:
                raise ValueError("An existing manifest cannot be added under another worker ID")
        _,required=context(registry["common"],registry["software_test_only"],{})
        validate_entry(spec,registry["common"],registry["software_test_only"],required,registry["profile_backends"],{})
        row=event_row(dict(event="ADD",entry=spec),registry["event_count"],registry["last_event_sha256"])
        with Path(path).open("ab") as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False).encode()+b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    return load_registry(path)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--registry",type=Path,required=True)
    parser.add_argument("--initialize",type=Path)
    parser.add_argument("--initialize-original-root",type=Path)
    parser.add_argument("--add",type=Path)
    args=parser.parse_args()
    if sum(value is not None for value in (args.initialize,args.initialize_original_root,args.add))>1:
        parser.error("Choose initialization or one ADD")
    if args.initialize:
        state=create_registry(args.registry,read_json(args.initialize))
    elif args.initialize_original_root:
        state=create_registry(args.registry,original_specs(args.initialize_original_root))
    elif args.add:
        state=add_manifest(args.registry,read_json(args.add))
    else:
        state=load_registry(args.registry)
    print(json.dumps(dict(schema=SCHEMA,entries=len(state["entries"]),software_test_only=state["software_test_only"],last_event_sha256=state["last_event_sha256"])))
