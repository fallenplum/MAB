from pathlib import Path
import argparse, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"data"))
from cache_audit.audit import audit_cache
def main():
    p=argparse.ArgumentParser(description="Full raw-cache field, alignment and six-model common-coverage E0 audit.")
    p.add_argument("--extracted",type=Path,default=ROOT/"data/raw/llmrouterbench/extracted")
    p.add_argument("--output-dir",type=Path,default=ROOT/"artifacts/audit/llmrouterbench")
    p.add_argument("--models",nargs=6)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        from cache_audit.tests import run_tests
        run_tests(ROOT)
        return
    if not args.extracted.is_dir():
        p.error("Extracted cache directory is not available; download and safely extract first")
    result=audit_cache(ROOT,args.extracted,args.models,args.output_dir)
    if not result["source_integrity"]["errors"]:
        import json
        from cache_audit.normalize import normalize_available
        reports=json.loads((args.output_dir/"file_audit.json").read_text(encoding="utf-8"))
        normalize_available(ROOT,args.extracted,reports,result["requested_models"])
    if result["status"]!="E0_CACHE_STRUCTURE_READY":
        raise SystemExit(2)
if __name__=="__main__":
    main()
