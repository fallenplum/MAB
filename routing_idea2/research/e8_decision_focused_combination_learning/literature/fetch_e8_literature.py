"""Fetch the five fixed E8 primary papers and pinned author source files.

Original papers and source snapshots belong in the private research workspace.
This downloader performs no model inference and downloads no experiment data.
"""
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PAPERS = [
    {"key": "ranknet2005", "title": "Learning to Rank using Gradient Descent", "authors": ["Christopher J. C. Burges", "Tal Shaked", "Erin Renshaw", "Ari Lazier", "Matt Deeds", "Nicole Hamilton", "Greg Hullender"], "year": 2005, "doi": "10.1145/1102351.1102363", "url": "https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf", "role": "RankNet original"},
    {"key": "listnet2007", "title": "Learning to Rank: From Pairwise Approach to Listwise Approach", "authors": ["Zhe Cao", "Tao Qin", "Tie-Yan Liu", "Ming-Feng Tsai", "Hang Li"], "year": 2007, "doi": "10.1145/1273496.1273513", "url": "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf", "role": "author full Microsoft technical report MSR-TR-2007-40"},
    {"key": "spo_plus", "title": "Smart Predict, then Optimize", "authors": ["Adam N. Elmachtoub", "Paul Grigas"], "year": 2022, "doi": "10.1287/mnsc.2020.3922", "arxiv": "1710.08005v5", "url": "https://arxiv.org/pdf/1710.08005v5", "role": "original SPO/SPO+ full author preprint"},
    {"key": "liu_grigas2021", "title": "Risk Bounds and Calibration for a Smart Predict-then-Optimize Method", "authors": ["Heyuan Liu", "Paul Grigas"], "year": 2021, "arxiv": "2108.08887v2", "url": "https://papers.nips.cc/paper_files/paper/2021/file/b943325cc7b7422d2871b345bf9b067f-Paper.pdf", "role": "NeurIPS2021 final main paper"},
    {"key": "liu_grigas2021_supplement", "title": "Risk Bounds and Calibration for a Smart Predict-then-Optimize Method - Supplemental", "authors": ["Heyuan Liu", "Paul Grigas"], "year": 2021, "url": "https://papers.nips.cc/paper_files/paper/2021/file/b943325cc7b7422d2871b345bf9b067f-Supplemental.pdf", "role": "same study appendix and full proofs"},
    {"key": "pyepo", "title": "PyEPO: A PyTorch-based End-to-End Predict-then-Optimize Library for Linear and Integer Programming", "authors": ["Bo Tang", "Elias B. Khalil"], "year": 2024, "doi": "10.1007/s12532-024-00255-x", "arxiv": "2206.14234v3", "url": "https://arxiv.org/pdf/2206.14234v3", "role": "full author preprint of library paper, revision 2026-04-17"},
]

REPOSITORIES = [
    {"repo": "paulgrigas/SmartPredictThenOptimize", "commit": "4002e9fe94a539c95d1e7520afd4f84d3063adf8", "license": "MIT", "files": ["README.md", "LICENSE", "solver/util.jl", "solver/sgd.jl", "solver/reformulation.jl", "solver/validation_set.jl", "oracles/basic_optimization_oracles.jl"]},
    {"repo": "khalil-research/PyEPO", "commit": "e5d768e2749bccdeba7494b4ec848c3eb3ffbd25", "license": "MIT", "files": ["README.md", "LICENSE", "pkg/pyepo/func/surrogate.py", "pkg/pyepo/func/abcmodule.py", "pkg/pyepo/func/_common.py", "pkg/pyepo/EPO.py"]},
    {"repo": "HeyuanLiu/SPO_plus_public", "commit": "60f5ea2896ae315e541e220fb8267a7dd485e80c", "license": "MIT", "files": ["README.md", "LICENSE", "tools/loss_func_tools.py", "tools/optimization_oracle_tools.py", "tools/spo_framework.py", "tools/data_generation_tools.py"]},
]


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "MAB-E8-literature-audit/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read(8_000_000)
        return raw, response.headers.get("Content-Type", ""), response.url


def fetch(root, sources_only=False):
    root = Path(root)
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "physical_LLM_calls": 0, "papers": [], "repositories": []}
    if sources_only:
        previous = json.loads((root / "E8_LITERATURE_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
        manifest["papers"] = previous["papers"]
    for source in ([] if sources_only else PAPERS):
        target = root / "papers" / (source["key"] + ".pdf")
        target.parent.mkdir(parents=True, exist_ok=True)
        raw, mime, final = get(source["url"])
        if not raw.startswith(b"%PDF"):
            raise ValueError("Primary source response is not PDF " + source["key"])
        target.write_bytes(raw)
        manifest["papers"].append({**source, "relative_path": str(target.relative_to(root)).replace("\\", "/"), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "response_content_type": mime, "final_url": final, "public_redistribution": False})
        if "arxiv.org" in source["url"]:
            time.sleep(3)
    for repository in REPOSITORIES:
        item = {key: repository[key] for key in ["repo", "commit", "license"]}
        item["files"] = []
        for filename in repository["files"]:
            url = f"https://raw.githubusercontent.com/{repository['repo']}/{repository['commit']}/{filename}"
            try:
                raw, mime, final = get(url)
            except Exception as exc:
                item["files"].append({"path": filename, "url": url, "error": str(exc)})
                continue
            target = root / "upstream" / repository["repo"].split("/")[-1] / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            item["files"].append({"path": filename, "url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "git_blob_sha1": hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()})
        manifest["repositories"].append(item)
    (root / "E8_LITERATURE_SOURCE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import sys
    result = fetch(sys.argv[1], sources_only="--sources-only" in sys.argv)
    print(json.dumps({"papers": len(result["papers"]), "repositories": len(result["repositories"]), "errors": [f for r in result["repositories"] for f in r["files"] if "error" in f]}, ensure_ascii=False, indent=2))
