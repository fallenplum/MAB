from pathlib import Path
import collections, json, os, unicodedata
from .audit import canonical_task, canonical_model, sha_text, json_stable, valid_record, REQUIRED
from .download import write_json, digest

def nfkc_query(text):
    return " ".join(unicodedata.normalize("NFKC",text).split())

def official_bbh_index(root):
    root=Path(root)
    receipt_path=root/"data/raw/bbh_official/source_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("Official BBH fixed-revision source receipt missing; cannot infer subtask")
    receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("repo")!="suzgunmirac/BIG-Bench-Hard" or len(receipt.get("files",[]))!=27:
        raise ValueError("Unexpected official BBH provenance")
    index=collections.defaultdict(dict)
    examples=0
    for info in receipt["files"]:
        p=root/info["path"]
        if digest(p)!=info["sha256"]:
            raise ValueError("Official BBH task source hash mismatch")
        data=json.loads(p.read_bytes())
        subtask=Path(info["source_path"]).stem
        for ex in data["examples"]:
            query,target=ex["input"],ex["target"]
            if not isinstance(query,str) or not isinstance(target,str):
                raise ValueError("Malformed official BBH source example")
            identity=(subtask,nfkc_query(target))
            index[nfkc_query(query)][identity]=dict(subtask=subtask,reference=target,source_file=info["path"])
            examples+=1
    return index,dict(repo=receipt["repo"],revision=receipt["revision"],examples=examples,
                      normalization="Unicode NFKC followed by whitespace collapse; exact equality only",
                      official_source_files=27)

def normalize_available(root, extracted, file_reports, requested_models, require_bbh_subtasks=True):
    root,extracted=Path(root).resolve(),Path(extracted).resolve()
    wanted={canonical_model(m):m for m in requested_models}
    groups=collections.defaultdict(dict)
    model_identity=collections.defaultdict(set)
    conflicts=set()
    target_sources=[x for x in file_reports if x.get("status")=="PARSED" and x["dataset_key"] in ("gsm8k","bbh") and x["model_key"] in wanted]
    index,subtask_source={},None
    subtask_source_error=None
    if require_bbh_subtasks and any(x["dataset_key"]=="bbh" for x in target_sources):
        try:
            index,subtask_source=official_bbh_index(root)
        except (ValueError,OSError,json.JSONDecodeError) as exc:
            subtask_source_error=type(exc).__name__+": "+str(exc)
    for info in target_sources:
        p=extracted/info["path"]
        if digest(p)!=info["sha256"]:
            raise ValueError("Target source changed after full cache audit")
        obj=json.loads(p.read_bytes())
        tk,split,mk=info["dataset_key"],info["split"],info["model_key"]
        source_file=p.relative_to(root).as_posix()
        model_identity[mk].add(info["model"])
        for row in obj["records"]:
            if valid_record(row):
                continue
            key=sha_text(tk+"\n"+split+"\n"+" ".join(row["origin_query"].split()))
            old=groups[(tk,split,key)].get(mk)
            sig=sha_text(json_stable({k:row.get(k) for k in REQUIRED}))
            if old and old["signature"]!=sig:
                conflicts.add((tk,split,key))
                continue
            subtask=next((row[k] for k in ("subtask","task_name","task","category") if isinstance(row.get(k),str) and row[k]),None)
            groups[(tk,split,key)][mk]=dict(signature=sig,question=row["origin_query"],reference=row["ground_truth"],
                 subtask=subtask,source_index=row.get("question_id",row.get("id",row.get("index"))),
                 member=dict(raw_output=row["raw_output"],score=row["score"],prompt_tokens=row["prompt_tokens"],
                             completion_tokens=row["completion_tokens"],cost=row["cost"],
                             source_file=source_file,source_model_name=info["model"]))
    output=root/"data/derived/available_records.jsonl"
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_name(output.name+".tmp")
    counts,missing_subtasks,excluded=collections.Counter(),collections.Counter(),collections.Counter()
    match=collections.Counter()
    match_samples=collections.defaultdict(list)
    subtask_counts=collections.Counter()
    eligible_common=collections.Counter()
    records=[]
    for (tk,split,key),members in sorted(groups.items()):
        if (tk,split,key) in conflicts:
            excluded["ambiguous_repeated_output"]+=1
            continue
        if not all(mk in members for mk in wanted):
            excluded["not_six_model_common"]+=1
            continue
        refs={sha_text(json_stable(members[mk]["reference"])) for mk in wanted}
        if len(refs)!=1:
            excluded["cross_model_reference_conflict"]+=1
            continue
        eligible_common[tk]+=1
        first=members[next(iter(wanted))]
        subtask=first["subtask"]
        extra={}
        if tk=="bbh" and require_bbh_subtasks:
            candidates=index.get(nfkc_query(first["question"]),{})
            if subtask_source_error:
                reason="official_source_unavailable"
            elif not candidates:
                reason="unmatched"
            elif len(candidates)!=1:
                reason="official_duplicate_input_reference_conflict"
            else:
                found=next(iter(candidates.values()))
                if not isinstance(first["reference"],str) or nfkc_query(first["reference"])!=nfkc_query(found["reference"]):
                    reason="reference_mismatch"
                else:
                    reason="matched"
                    subtask=found["subtask"]
                    extra=dict(subtask_source_file=found["source_file"],subtask_source_revision=subtask_source["revision"])
                    subtask_counts[subtask]+=1
            match[reason]+=1
            if reason!="matched":
                if len(match_samples[reason])<12:
                    match_samples[reason].append("bbh:"+key)
                continue
        record=dict(question_id=tk+":"+key,task=tk,subtask=subtask,question=first["question"],
                    reference=first["reference"],source_split=split,
                    member_outputs={wanted[mk]:members[mk]["member"] for mk in wanted},
                    source_question_ids={wanted[mk]:members[mk]["source_index"] for mk in wanted},**extra)
        records.append(record)
    mapping_failed=bool(require_bbh_subtasks and eligible_common["bbh"] and match["matched"]!=eligible_common["bbh"])
    # Fail closed per questionable record; keep independently verified questions.
    # Root approved excluding exactly the two official conflicting-input cases.
    with temp.open("w",encoding="utf-8",newline="\n") as f:
        for record in records:
            f.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
            counts[record["task"]]+=1
            if record["subtask"] is None:
                missing_subtasks[record["task"]]+=1
    os.replace(temp,output)
    receipt=dict(status="AVAILABLE_WITH_DATA_QUALITY_EXCLUSIONS" if mapping_failed and counts["bbh"] else "BLOCKED_BBH_SUBTASK_MAPPING" if mapping_failed else "NORMALIZED_AVAILABLE_COMMON_RECORDS",
                 path=output.relative_to(root).as_posix(),bytes=output.stat().st_size,sha256=digest(output),
                 task_counts=dict(counts),eligible_six_model_common_questions=dict(eligible_common),
                 target_tasks=["gsm8k","bbh"],missing_tasks=[x for x in ("gsm8k","bbh") if counts[x]==0],
                 requested_model_aliases=requested_models,
                 model_identity=[dict(alias=wanted[mk],cache_model_names=sorted(model_identity[mk]),weight_repository_id=None,
                                      weight_revision=None,identity_status="CACHE_NAME_ONLY; full weight ID/revision must be independently verified") for mk in wanted],
                 excluded_question_counts=dict(excluded),missing_subtask_metadata_counts=dict(missing_subtasks),
                 bbh_subtask_mapping=dict(required=require_bbh_subtasks,status="PASS_WITH_EXCLUSIONS" if mapping_failed and counts["bbh"] else "BLOCKED" if mapping_failed else "PASS" if require_bbh_subtasks else "SYNTHETIC_TEST_BYPASS",
                     source=subtask_source,source_error=subtask_source_error,counts=dict(match),
                     exclusion_policy="Exclude every ambiguous/unmatched/reference-conflicting question before split selection; keep only exact unique matches. Two duplicate-input Yes/No conflicts excluded with root authorization; E1 total remains 300.",
                     match_rate=(match["matched"]/eligible_common["bbh"] if eligible_common["bbh"] else None),
                     subtask_counts=dict(subtask_counts),failure_question_id_samples=dict(match_samples)),
                 split_warning="Source split retained verbatim; normalized availability is not a frozen calibration/online/final-test allocation.",
                 quality_warning="Scores copied as source provenance only; no quality-based subset or model-pool selection.",
                 precision_warning="Published member generation precision/weight commit are unknown; future local NF4 generations may differ and require task-specific execution-condition disclosure.",
                 privacy="Private data contains questions, references and raw member outputs. Never upload it to GitHub.",
                 no_math500_substitution=True,combination_aggregations_present=False)
    write_json(root/"artifacts/audit/llmrouterbench/available_records_receipt.json",receipt)
    print(json.dumps({k:receipt[k] for k in ("status","path","bytes","task_counts","missing_tasks","missing_subtask_metadata_counts","bbh_subtask_mapping")},ensure_ascii=False),flush=True)
    return receipt
