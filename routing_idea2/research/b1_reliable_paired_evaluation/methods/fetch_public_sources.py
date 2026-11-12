"""Fetch pinned public reference sources into the private B1 methods workspace.

ASI and Robust-ASI have no license file in their pinned tree. Their snapshots
are retained privately for traceability. This script does not republish them.
"""
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    "tijana-zrnic/active-inference": {
        "commit": "0f99f7ee6055b6f86c0ce88ece3cb2bf3c9c19cd", "license": None,
        "files": ["README.md", "utils.py", "alphafold-ptm.ipynb", "pew79-post-election-research.ipynb"],
    },
    "lphLeo/Robust-Active-Statistical-Inference": {
        "commit": "e8037bc113c62f420929f9a7c680a93f036674ec", "license": None,
        "files": ["README.md", "utils.py", "utils_optimization.py", "training.py", "pew_post_election.ipynb"],
    },
    "jlko/active-testing": {
        "commit": "2807fac9ad91fb12e5814e71b9a26c55df9d50fb", "license": "MIT",
        "files": ["README.md", "LICENSE", "requirements.txt", "conf/config.yaml", "conf/paper/SyntheticQuadraticLinear.yaml", "activetesting/loss.py", "activetesting/risk_estimators.py", "activetesting/acquisition.py", "activetesting/datasets.py", "activetesting/models/skmodels.py"],
    },
    "oberst-lab/cps-sampling": {
        "commit": "a5f3fdbed1500137c54d39544991d9d1850f52e7", "license": "MIT",
        "files": ["README.md", "LICENSE", "pyproject.toml", "src/cps_sampling/__init__.py", "src/cps_sampling/_kernels.py", "src/cps_sampling/calibration.py", "src/cps_sampling/design.py", "src/cps_sampling/joint.py", "src/cps_sampling/variance.py", "tests/test_design.py", "tests/test_joint.py", "tests/test_variance.py"],
    },
}


def fetch(destination, download=True):
    destination = Path(destination)
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "sources": [],
                "missing_source": {"repo": "oberst-lab/fixed-size-asi", "status": "HTTP404", "fallback": "paper-linked official author cps-sampling package"}}
    for repo, configuration in SOURCES.items():
        record = dict(repo=repo, commit=configuration["commit"], license=configuration["license"], files=[])
        for filename in configuration["files"]:
            target = destination / repo.split("/")[-1] / filename
            url = f"https://raw.githubusercontent.com/{repo}/{configuration['commit']}/{filename}"
            if download:
                raw = urllib.request.urlopen(url, timeout=30).read(2_000_000)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            else:
                raw = target.read_bytes()
            record["files"].append(dict(path=filename, url=url, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        manifest["sources"].append(record)
    output = destination.parent / "UPSTREAM_SOURCE_MANIFEST.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    manifest = fetch(destination, download="--inventory-only" not in sys.argv)
    print(json.dumps({"sources": len(manifest["sources"]), "files": sum(len(x["files"]) for x in manifest["sources"])}, indent=2))
