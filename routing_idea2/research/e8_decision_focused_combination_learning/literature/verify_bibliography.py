"""Resolve the four supplied DOI identities through Crossref's exact endpoint."""
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("E8_LITERATURE_WORK", str(Path(__file__).resolve().parent / "private")))
records = []
for key, doi in [("ranknet2005","10.1145/1102351.1102363"),("listnet2007","10.1145/1273496.1273513"),("spo_plus","10.1287/mnsc.2020.3922"),("pyepo","10.1007/s12532-024-00255-x")]:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi,safe="")
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"MAB-E8-literature-audit/1.0"}),timeout=30) as response:
            raw = response.read(500_000)
        content = json.loads(raw)
        if content.get("status") != "ok" or content.get("message-type") != "work":
            raise ValueError("Unexpected Crossref response shape")
        message = content["message"]
        record = {"key":key,"query_doi":doi,"url":url,"response_sha256":hashlib.sha256(raw).hexdigest(),
                  "DOI":message["DOI"],"title":message.get("title"),"authors":message.get("author"),
                  "container-title":message.get("container-title"),"issued":message.get("issued"),
                  "published-print":message.get("published-print"),"published-online":message.get("published-online"),
                  "volume":message.get("volume"),"issue":message.get("issue"),"page":message.get("page"),"status":"RESOLVED"}
        assert record["DOI"].lower() == doi.lower()
        metadata = ROOT / "crossref"
        metadata.mkdir(exist_ok=True)
        (metadata / (key+".json")).write_bytes(raw)
    except Exception as exc:
        record = {"key":key,"query_doi":doi,"url":url,"status":"UNRESOLVED","error":str(exc)}
    records.append(record)
    time.sleep(.3)
(ROOT / "E8_BIBLIOGRAPHY_IDENTITY_CHECK.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps([{k:v for k,v in r.items() if k in ["key","DOI","title","issued","status","error"]} for r in records],ensure_ascii=False,indent=2))
