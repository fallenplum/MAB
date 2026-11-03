from pathlib import Path
import argparse, sys, json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
from cache_audit.download import metadata, download, safe_extract
def main():
    p = argparse.ArgumentParser(description="Pinned public-cache acquisition; no API keys or model calls.")
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--max-attempts", type=int, default=3)
    args = p.parse_args()
    if args.metadata_only:
        info=metadata(ROOT)
        print(json.dumps({k:info[k] for k in ["repo","revision","bytes","sha256","head"]},ensure_ascii=False))
        return
    path = download(ROOT, args.max_attempts)
    if args.extract:
        safe_extract(path, ROOT / "data/raw/llmrouterbench/extracted")
if __name__ == "__main__":
    main()
