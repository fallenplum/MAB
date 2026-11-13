"""Local text and page-image extraction for the fixed primary PDFs."""
import hashlib
import json
import os
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
root = Path(os.environ.get("E8_LITERATURE_WORK", str(HERE / "private")))
summaries = []
for path in sorted((root / "papers").glob("*.pdf")):
    document = pymupdf.open(path)
    output = root / "text" / (path.stem + ".txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"\n===== PDF PAGE {i+1} =====\n"+page.get_text() for i,page in enumerate(document))
    output.write_text(text, encoding="utf-8")
    summaries.append({"paper": path.stem, "pages": len(document), "pdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "characters": len(text)})
    pages = {"ranknet2005": [2], "listnet2007": [3,4], "spo_plus": [15,16,22], "liu_grigas2021": [2,4], "liu_grigas2021_supplement": [2], "pyepo": [11]}.get(path.stem, [])
    image_dir = root / "page_images"
    image_dir.mkdir(exist_ok=True)
    for page in pages:
        document[page].get_pixmap(matrix=pymupdf.Matrix(1.4,1.4)).save(image_dir / f"{path.stem}_p{page+1}.png")
(root / "PDF_EXTRACTION_MANIFEST.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
print(json.dumps(summaries, indent=2))
