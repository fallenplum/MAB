from pathlib import Path
import collections, hashlib, json, math, re
from .download import digest, write_json, EXPECTED_SHA256, EXPECTED_BYTES

DEFAULT_MODELS = ["Qwen3-8B", "Qwen2.5-Coder-7B-Instruct", "granite-3.3-8b-instruct",
                  "DeepHermes-3-Llama-3-8B-Preview", "DeepSeek-R1-Distill-Qwen-7B", "cogito-v1-preview-llama-8B"]
REQUIRED = ["origin_query", "raw_output", "ground_truth", "score",
            "prompt_tokens", "completion_tokens", "cost"]
def canonical_model(name):
    return re.sub("[^a-z0-9]", "", str(name).split("/")[-1].lower())
def canonical_task(name):
    return re.sub("[^a-z0-9]", "", str(name).lower())
def sha_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
def json_stable(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
def valid_record(row):
    issues=[]
    if not isinstance(row, dict):
        return ["not_object"]
    for k in REQUIRED:
        if k not in row:
            issues.append("missing_" + k)
    for k in ("origin_query","raw_output"):
        if not isinstance(row.get(k),str) or not row.get(k,"").strip():
            issues.append("empty_or_nonstring_" + k)
    if row.get("ground_truth") is None:
        issues.append("null_ground_truth")
    score=row.get("score")
    if not number(score) or not 0 <= score <= 1:
        issues.append("invalid_score")
    for k in ("prompt_tokens","completion_tokens"):
        v=row.get(k)
        if not number(v) or v<0 or int(v)!=v:
            issues.append("invalid_" + k)
    if not number(row.get("cost")) or row.get("cost",-1)<0:
        issues.append("invalid_cost")
    return sorted(set(issues))

def audit_cache(root, extracted, models=None, output_dir=None, verify_distribution=True):
    root, extracted = Path(root).resolve(), Path(extracted).resolve()
    output_dir=Path(output_dir or root/"artifacts/audit/llmrouterbench").resolve()
    output_dir.relative_to(root)
    models=models or DEFAULT_MODELS
    wanted={canonical_model(m):m for m in models}
    if len(wanted)!=6:
        raise ValueError("E0 protocol requires six distinct candidate models")
    files=sorted(p for p in extracted.rglob("*.json") if p.name!=".cache_extraction_receipt.json")
    source_errors=[]
    expected_files={}
    marker=extracted/".cache_extraction_receipt.json"
    if verify_distribution:
        if not marker.is_file():
            source_errors.append("missing_verified_extraction_receipt")
        else:
            receipt=json.loads(marker.read_text(encoding="utf-8"))
            if receipt.get("archive_sha256")!=EXPECTED_SHA256:
                source_errors.append("receipt_archive_hash_not_pinned_source")
            expected_files={x["path"]:x for x in receipt.get("files",[])}
            actual_files={p.relative_to(extracted).as_posix() for p in extracted.rglob("*") if p.is_file() and p!=marker}
            if actual_files!=set(expected_files):
                source_errors.append("extraction_members_differ_from_receipt")
        source_archive=root/"data/raw/llmrouterbench/bench-release.tar.gz"
        if not source_archive.is_file() or source_archive.stat().st_size!=EXPECTED_BYTES or digest(source_archive)!=EXPECTED_SHA256:
            source_errors.append("source_archive_missing_or_sha256_mismatch")
    reports, parse_errors, datasets, truth_groups = [], [], {}, collections.defaultdict(set)
    seen_ids, collision_counts = collections.defaultdict(dict), collections.Counter()
    all_rows=0
    for num,p in enumerate(files,1):
        rel=p.relative_to(extracted).as_posix()
        h=digest(p)
        if verify_distribution and expected_files.get(rel,{}).get("sha256")!=h:
            source_errors.append("extracted_member_sha256_mismatch:"+rel)
        try:
            obj=json.loads(p.read_bytes())
        except (json.JSONDecodeError,UnicodeDecodeError) as exc:
            parse_errors.append(dict(path=rel,kind=type(exc).__name__))
            continue
        if not isinstance(obj,dict) or not isinstance(obj.get("records"),list):
            reports.append(dict(path=rel,sha256=h,status="NOT_RECORD_CACHE",top_type=type(obj).__name__))
            continue
        rows=obj["records"]
        parts=Path(rel).parts
        task=str(obj.get("dataset_name") or (parts[1] if len(parts)>1 else "UNKNOWN"))
        split=str(obj.get("split") if obj.get("split") is not None else "<missing>")
        model=str(obj.get("model_name") or (parts[-2] if len(parts)>1 else "UNKNOWN"))
        tk,mk=canonical_task(task),canonical_model(model)
        group=datasets.setdefault(tk,dict(names=set(),splits={}))
        group["names"].add(task)
        sg=group["splits"].setdefault(split,{})
        mg=sg.setdefault(mk,dict(names=set(),keys={},duplicates=0,conflicting_keys=set(),rows=0))
        mg["names"].add(model)
        fields=collections.Counter()
        types=collections.defaultdict(collections.Counter)
        issues=collections.Counter()
        field_null=collections.Counter()
        source_ids=collections.Counter()
        valid_count=0
        token_sums={"prompt_tokens":0,"completion_tokens":0,"cost":0.0}
        zero_counts=collections.Counter()
        subtask_fields=collections.Counter()
        for row in rows:
            all_rows+=1
            mg["rows"]+=1
            bad=valid_record(row)
            issues.update(bad)
            if not isinstance(row,dict):
                continue
            fields.update(row.keys())
            for k,v in row.items():
                types[k][type(v).__name__]+=1
                if v is None:
                    field_null[k]+=1
            for k in ("subtask","task","task_name","category"):
                if row.get(k) is not None:
                    subtask_fields[k]+=1
            sid=row.get("question_id",row.get("id",row.get("index")))
            source_ids[type(sid).__name__]+=1
            query=row.get("origin_query")
            if not isinstance(query,str) or not query.strip():
                continue
            query_norm=" ".join(query.split())
            key=sha_text(tk+"\n"+split+"\n"+query_norm)
            truth=sha_text(json_stable(row.get("ground_truth")))
            truth_groups[(tk,split,key)].add(truth)
            sig=sha_text(json_stable({k:row.get(k) for k in REQUIRED}))
            state=mg["keys"].get(key)
            if state is None:
                mg["keys"][key]=dict(signature=sig,valid=not bad,truth=truth)
            else:
                mg["duplicates"]+=1
                if state["signature"]!=sig:
                    mg["conflicting_keys"].add(key)
                    state["valid"]=False
                else:
                    state["valid"]=state["valid"] and not bad
            if sid is not None:
                sk=(tk,split,mk)
                old=seen_ids[sk].get(str(sid))
                if old and old!=key:
                    collision_counts[sk]+=1
                seen_ids[sk][str(sid)]=key
            if not bad:
                valid_count+=1
            for k in token_sums:
                if number(row.get(k)):
                    token_sums[k]+=row[k]
                    zero_counts[k]+=row[k]==0
        claimed=obj.get("counts")
        totals_check={}
        for k,total in token_sums.items():
            aggregate=obj.get(k)
            totals_check[k]=dict(record_sum=total,declared_total=aggregate,
                                  matches=bool(number(aggregate) and math.isclose(total,aggregate,rel_tol=1e-7,abs_tol=1e-5)))
        reports.append(dict(path=rel,sha256=h,bytes=p.stat().st_size,status="PARSED",
                            dataset=task,dataset_key=tk,split=split,model=model,model_key=mk,
                            records=len(rows),declared_counts=claimed,counts_match=claimed==len(rows),
                            valid_required_records=valid_count,field_presence=dict(fields),
                            field_nulls=dict(field_null),field_types={k:dict(v) for k,v in types.items()},
                            row_issue_counts=dict(issues),source_id_types=dict(source_ids),
                            metadata_keys=list(obj.keys()),data_fingerprint=obj.get("data_fingerprint"),
                            subtask_field_presence=dict(subtask_fields),zero_cost_or_token_counts=dict(zero_counts),
                            aggregate_checks=totals_check))
        if num%40==0:
            print("AUDITED_CACHE_FILES",num,"records",all_rows,flush=True)
    inconsistent={k for k,v in truth_groups.items() if len(v)>1}
    target_results={}
    coverage=[]
    for tk,group in sorted(datasets.items()):
        for split,sg in sorted(group["splits"].items()):
            model_counts={}
            common_all,common_valid=None,None
            for mk,mg in sorted(sg.items()):
                validkeys={k for k,v in mg["keys"].items() if v["valid"] and (tk,split,k) not in inconsistent}
                model_counts[mk]=dict(names=sorted(mg["names"]),rows=mg["rows"],unique_questions=len(mg["keys"]),
                                       valid_questions=len(validkeys),duplicate_rows=mg["duplicates"],
                                       conflicting_duplicate_questions=len(mg["conflicting_keys"]),
                                       source_id_collisions=collision_counts[(tk,split,mk)])
            for mk in wanted:
                mg=sg.get(mk,dict(keys={}))
                keys=set(mg["keys"])
                good={k for k,v in mg["keys"].items() if v["valid"] and (tk,split,k) not in inconsistent}
                common_all=keys if common_all is None else common_all & keys
                common_valid=good if common_valid is None else common_valid & good
            row=dict(dataset_key=tk,dataset_names=sorted(group["names"]),split=split,
                     model_counts=model_counts,requested_models=models,
                     missing_requested_models=[v for k,v in wanted.items() if k not in sg],
                     six_model_common_questions=len(common_all or set()),
                     six_model_valid_common_questions=len(common_valid or set()),
                     cross_model_ground_truth_conflicts=sum(k[:2]==(tk,split) for k in inconsistent))
            coverage.append(row)
            if tk in ("gsm8k","bbh"):
                target_results.setdefault(tk,[]).append(row)
    missing=[x for x in ("gsm8k","bbh") if x not in target_results]
    positive={x:any(y["six_model_valid_common_questions"]>0 for y in target_results.get(x,[])) for x in ("gsm8k","bbh")}
    status="E0_CACHE_STRUCTURE_READY" if all(positive.values()) and not parse_errors and not source_errors else "E0_BLOCKED_CACHE_REQUIREMENTS"
    summary=dict(status=status,scope="Published member-response cache audit; no combination aggregation or routing run",
                 source_archive_revision="0e5af1b84bf73437a01a1849c0f1d2468baa93fc",
                 source_integrity=dict(pinned_archive_sha256=EXPECTED_SHA256,distribution_verification_required=verify_distribution,errors=source_errors),
                 json_files=len(files),record_cache_files=sum(x["status"]=="PARSED" for x in reports),
                 total_records_scanned=all_rows,parse_errors=parse_errors,
                 datasets=sorted(datasets),target_tasks=["gsm8k","bbh"],requested_models=models,
                 missing_required_tasks=missing,target_common_coverage_ready=positive,
                 target_task_details=target_results,
                 limitations=["Raw model output is required; parsed prediction alone is not a substitute.",
                              "Query keys hash dataset+source split+whitespace-normalized origin_query; source index is not assumed globally unique.",
                              "No examples were selected using scores. Different repeated outputs for the same model/question are marked ambiguous, not best-of chosen.",
                              "A split named test in the source cache is provenance, not permission to use final held-out labels for model selection.",
                              "A zero recorded cost is not evidence of free inference. Nonzero token usage and an explicit price/GPU ledger are required separately.",
                              "Reported model-name strings do not establish a frozen weight commit, tokenizer revision, sampling protocol, or training contamination status.",
                              "Member responses/scores cannot substitute for real frozen-aggregator combination responses and scores.",
                              "GSM8K absence blocks the GSM8K+BBH first-version protocol; MATH500 is not substituted.",
                              "This audit does not select the final model pool or claim the E0 end-to-end aggregator gate is complete."])
    write_json(output_dir/"cache_audit.json",summary)
    write_json(output_dir/"file_audit.json",reports)
    write_json(output_dir/"common_coverage.json",coverage)
    write_json(output_dir/"provenance.json",dict(extracted_root=str(extracted),
               source_metadata_path=str(root/"data/raw/llmrouterbench/source_metadata.json"),
               identity_rule="sha256(dataset_key + newline + source_split + newline + normalized_origin_query)",
               content_not_copied=True,files=sorted(x["path"] for x in reports)))
    print(json.dumps({k:summary[k] for k in ["status","json_files","total_records_scanned","datasets",
                     "missing_required_tasks","target_common_coverage_ready"]},ensure_ascii=False),flush=True)
    return summary
