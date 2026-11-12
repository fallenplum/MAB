"""Retrieve the exact small AlphaFold reference file used by the ASI notebook."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("B1_METHODS_WORK", str(Path(__file__).resolve().parent)))
target = ROOT / "reference_data" / "alphafold.npz"
target.parent.mkdir(parents=True, exist_ok=True)
urls = [
    "https://drive.google.com/uc?export=download&id=1lOhdSJEcFbZmcIoqmlLxo3LgLG1KqPho",
    "https://drive.usercontent.google.com/download?id=1lOhdSJEcFbZmcIoqmlLxo3LgLG1KqPho&export=download&authuser=0",
]
record = {"source_loader": "https://github.com/aangelopoulos/ppi_py/blob/main/ppi_py/datasets/datasets.py", "attempts": []}
if not target.exists():
    for url in urls:
        try:
            data = urllib.request.urlopen(url, timeout=30).read(20_000_000)
            if not data.startswith(b"PK"):
                raise ValueError("Response is not an NPZ zip file")
            target.write_bytes(data)
            record["attempts"].append({"url": url, "ok": True})
            break
        except Exception as exc:
            record["attempts"].append({"url": url, "error": str(exc)})
if target.exists():
    raw = target.read_bytes()
    record.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest(), path=str(target))
(target.parent / "DOWNLOAD.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
