"""Serial, real NF4 member generation. No scoring/reference input is accepted."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from budget_router.inference import ModelSpec,make_backend,configuration_fingerprint,GenerationFailure
from budget_router.runtime_diagnostics import make_backend_with_diagnostics,BackendInitializationError
from budget_router.runtime_memory import cleanup_after_committed_success,cleanup_after_backend_initialization,RuntimeMemorySummary
from budget_router.member_execution import MemberStore,validate_member_job,validate_resample_groups


def _qualification_receipts(paths):
    receipts={}
    for path in paths or []:
        receipt=json.loads(path.read_text(encoding="utf-8-sig"))
        if receipt.get("status")!="PASS" or not receipt.get("model_fingerprint"):
            raise ValueError("Member qualification must be a reviewed PASS receipt")
        key=receipt["model_fingerprint"]
        if key in receipts:raise ValueError("Duplicate model qualification receipts")
        receipts[key]=receipt
    return receipts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--jobs",type=Path,required=True)
    parser.add_argument("--model-config",type=Path,action="append",required=True,
                        help="Repeat for each model/sampling spec; groups execute serially")
    parser.add_argument("--qualification-receipt",type=Path,action="append")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--qualification",action="store_true")
    parser.add_argument("--max-jobs",type=int)
    args=parser.parse_args()
    if args.max_jobs is not None and args.max_jobs<=0:raise ValueError("max-jobs must be positive")
    specs={}
    config_hashes={}
    for path in args.model_config:
        raw=path.read_bytes()
        spec=ModelSpec(**json.loads(raw.decode("utf-8-sig")))
        fingerprint=configuration_fingerprint(spec)
        if fingerprint in specs:raise ValueError("Duplicate member model configuration")
        specs[fingerprint]=spec
        config_hashes[fingerprint]=hashlib.sha256(raw).hexdigest()
    source=args.jobs.read_bytes()
    # Only LF delimits JSONL. U+2028/U+0085 are valid answer/question characters.
    jobs=[json.loads(line) for line in source.decode("utf-8-sig").split("\n") if line.strip()]
    if not jobs:raise ValueError("Empty member batch is not qualification success")
    seen=set()
    for job in jobs:
        fingerprint=job.get("model_fingerprint")
        if fingerprint not in specs:raise ValueError("Member job has no matching model config")
        jid=validate_member_job(job,specs[fingerprint])
        if jid in seen:raise ValueError("Duplicate member job in batch")
        seen.add(jid)
        if job["split"]=="sealed_test":raise ValueError("Sealed test has its own release/checkpoint gate")
    validate_resample_groups(jobs)
    if args.qualification:
        if any(job["split"]!="qualification" for job in jobs):
            raise ValueError("Qualification cannot consume E1/online/test questions")
        jobs=jobs[:args.max_jobs if args.max_jobs is not None else 2]
    else:
        if any(job["split"]=="qualification" for job in jobs):
            raise ValueError("Qualification jobs require explicit qualification mode")
        receipts=_qualification_receipts(args.qualification_receipt)
        if any(job["model_fingerprint"] not in receipts for job in jobs):
            raise ValueError("Formal member execution requires PASS receipt for each exact model/sampling spec")
        if args.max_jobs is not None:jobs=jobs[:args.max_jobs]
    store=MemberStore(args.output)
    processed=failures=cache_hits=0
    housekeeping=RuntimeMemorySummary()
    runner_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for fingerprint,spec in specs.items():
        group=[job for job in jobs if job["model_fingerprint"]==fingerprint]
        if not group:continue
        pending=[job for job in group if store._cached(job,spec) is None]
        backend=make_backend_with_diagnostics(spec,project_root=ROOT,factory=make_backend) if pending else None
        try:
            if backend is not None:
                housekeeping.observe(cleanup_after_backend_initialization(backend,project_root=ROOT,output_path=args.output,
                    runner_name=Path(__file__).name,runner_source_sha256=runner_source_sha256))
            for job in group:
                if backend is None:
                    row,cached=store._cached(job,spec),True
                else:
                    row,cached=store.run(backend,job)
                if not cached and row["status"]=="success":
                    housekeeping.observe(cleanup_after_committed_success(backend,row,cached=cached,project_root=ROOT,
                        output_path=args.output,runner_name=Path(__file__).name,runner_source_sha256=runner_source_sha256))
                processed+=1
                failures+=int(row["status"]!="success")
                cache_hits+=int(cached)
                print(json.dumps({"processed":processed,"requested":len(jobs),"job_id":job["job_id"],
                    "model_alias":job["model_alias"],"sample_index":job["sample_index"],
                    "status":row["status"],"cache_hit":cached,"cost_status":row.get("cost_status")},ensure_ascii=False),flush=True)
                if row.get("cost_status")=="unknown_must_reconcile":
                    raise RuntimeError("Unknown member failure cost; stop before further calls")
                if failures>=3:raise RuntimeError("Three member failures; reconcile before continuing")
        finally:
            if backend is not None:
                # Drop the caller's reference before releasing CUDA allocations.
                torch=getattr(backend,"torch",None)
                del backend
                gc.collect()
                if torch is not None and torch.cuda.is_available():torch.cuda.empty_cache()
    summary={"status":"member_qualification_outputs_recorded" if args.qualification else "member_generation_batch_complete",
        "qualification_status":"PENDING_REVIEW" if args.qualification else "PREVIOUS_REVIEWED_PASS",
        "processed":processed,"requested":len(jobs),"failures":failures,"cache_hits":cache_hits,
        "models_in_batch":len({job["model_fingerprint"] for job in jobs}),"serial_models":True,
        "scientific_completion":False,"scoring_performed":False,
        "job_source_sha256":hashlib.sha256(source).hexdigest(),"config_sha256_by_model":config_hashes,
        "completed_at_unix":time.time(),"runner_source_sha256":runner_source_sha256,
        "runtime_memory_housekeeping":housekeeping.as_dict()}
    args.output.with_name(args.output.name+".batch.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))
    if failures:raise SystemExit(2)


if __name__=="__main__":
    try:main()
    except GenerationFailure as error:
        print(json.dumps({"status":"blocked","reason_code":error.reason_code,"scientific_completion":False}),file=sys.stderr)
        raise SystemExit(2) from None
    except BackendInitializationError as error:
        print(json.dumps(error.public_record(),ensure_ascii=False))
        raise SystemExit(2) from None
    except Exception as error:
        print(json.dumps({"status":"blocked","error_type":type(error).__name__,
                          "reason_code":"member_protocol_or_execution_validation_failed","scientific_completion":False}),file=sys.stderr)
        raise SystemExit(2) from None
