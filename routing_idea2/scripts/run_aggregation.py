"""Run real frozen jobs. Software smoke/qualification is never E0/E1 completion."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from budget_router.inference import ModelSpec,make_backend,configuration_fingerprint,GenerationFailure
from budget_router.runtime_diagnostics import make_backend_with_diagnostics,BackendInitializationError
from budget_router.runtime_memory import cleanup_after_committed_success,cleanup_after_backend_initialization,RuntimeMemorySummary
from budget_router.execution import ExecutionStore,validate_job


def _read_qualification(path,spec):
    if path is None:raise ValueError("Formal batches require a reviewed qualification PASS receipt")
    receipt=json.loads(path.read_text(encoding="utf-8-sig"))
    if receipt.get("status")!="PASS" or receipt.get("model_fingerprint")!=configuration_fingerprint(spec):
        raise ValueError("Qualification receipt does not pass this exact model configuration")
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--jobs",type=Path,required=True)
    parser.add_argument("--model-config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--qualification",action="store_true")
    parser.add_argument("--qualification-receipt",type=Path)
    parser.add_argument("--max-jobs",type=int,default=None)
    args=parser.parse_args()
    if args.max_jobs is not None and args.max_jobs<=0:raise ValueError("max-jobs must be positive")
    config_bytes=args.model_config.read_bytes()
    spec=ModelSpec(**json.loads(config_bytes.decode("utf-8-sig")))
    spec.validate()
    fingerprint=configuration_fingerprint(spec)
    source=args.jobs.read_bytes()
    jobs=[json.loads(line) for line in source.decode("utf-8-sig").split("\n") if line.strip()]
    if not jobs:raise ValueError("Empty job set is not a qualification/generation success")
    ids=[validate_job(job,spec,require_token_policy=not args.qualification) for job in jobs]
    if len(ids)!=len(set(ids)):raise ValueError("Duplicate jobs in batch; preserve unique successful accounting")
    if any(job["split"]=="sealed_test" for job in jobs):
        raise ValueError("Sealed test requires its separate release/checkpoint gate")
    if args.qualification:
        if any(job["split"]!="qualification" for job in jobs):
            raise ValueError("Qualification cannot silently consume E1/online/test questions")
        jobs=jobs[:args.max_jobs if args.max_jobs is not None else 2]
    else:
        if any(job["split"]=="qualification" for job in jobs):
            raise ValueError("Qualification jobs must be explicitly marked as qualification")
        _read_qualification(args.qualification_receipt,spec)
        if args.max_jobs is not None:jobs=jobs[:args.max_jobs]
    store=ExecutionStore(args.output)
    housekeeping=RuntimeMemorySummary()
    runner_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Validate cached identities/protocol even when every job has already finished.
    pending=[job for job in jobs if store._cached(job,spec) is None]
    if not pending:
        print(json.dumps({"status":"already_recorded","jobs":len(jobs),"scientific_completion":False,
            "runner_source_sha256":runner_source_sha256,"runtime_memory_housekeeping":housekeeping.as_dict()}))
        return
    backend=make_backend_with_diagnostics(spec,project_root=ROOT,factory=make_backend)
    housekeeping.observe(cleanup_after_backend_initialization(backend,project_root=ROOT,output_path=args.output,
        runner_name=Path(__file__).name,runner_source_sha256=runner_source_sha256))
    failures=0
    processed=0
    cache_hits=0
    for index,job in enumerate(jobs,1):
        row,cached=store.run(backend,job)
        if not cached and row["status"]=="success":
            housekeeping.observe(cleanup_after_committed_success(backend,row,cached=cached,project_root=ROOT,
                output_path=args.output,runner_name=Path(__file__).name,runner_source_sha256=runner_source_sha256))
        processed+=1
        cache_hits+=int(cached)
        failures+=int(row["status"]!="success")
        print(json.dumps({"processed":index,"requested":len(jobs),"job_id":job["job_id"],
            "status":row["status"],"cache_hit":cached,"cost_status":row.get("cost_status")},ensure_ascii=False),flush=True)
        if row.get("cost_status")=="unknown_must_reconcile":
            raise RuntimeError("Unknown failed-call cost; stop before additional execution")
        if failures>=3:raise RuntimeError("Three failures; reconcile the unqualified configuration")
    summary={"status":"qualification_outputs_recorded" if args.qualification else "generation_batch_complete",
        "qualification_status":"PENDING_REVIEW" if args.qualification else "PREVIOUS_REVIEWED_PASS",
        "model_fingerprint":fingerprint,"requested":len(jobs),"processed":processed,"failures":failures,
        "cache_hits":cache_hits,"scientific_completion":False,
        "job_source_sha256":hashlib.sha256(source).hexdigest(),
        "model_config_file_sha256":hashlib.sha256(config_bytes).hexdigest(),"completed_at_unix":time.time(),
        "runner_source_sha256":runner_source_sha256,"runtime_memory_housekeeping":housekeeping.as_dict()}
    audit=args.output.with_name(args.output.name+".batch.json")
    audit.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
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
        # Exceptions may contain HTTP credentials or response text. Never echo them.
        print(json.dumps({"status":"blocked","error_type":type(error).__name__,
                          "reason_code":"protocol_or_execution_validation_failed","scientific_completion":False}),file=sys.stderr)
        raise SystemExit(2) from None
