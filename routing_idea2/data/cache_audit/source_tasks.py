from pathlib import Path
import json, hashlib, requests, time, re

def fetch_official_task_sources(root, tasks=("gsm8k", "bbh")):
    root=Path(root).resolve()
    report={}
    specs={
      "gsm8k":dict(repo="openai/grade-school-math",branch="master",revision="3101c7d5072418e28b9008a6636bde82a006892c",prefix="grade_school_math/data/",files=["grade_school_math/data/train.jsonl","grade_school_math/data/test.jsonl"]),
      "bbh":dict(repo="suzgunmirac/BIG-Bench-Hard",branch="main",revision="9ee07bd481feebf959a6b59d61ea57bdcf30964d",prefix="bbh/",files=None)
    }
    for task in tasks:
        spec=specs[task]
        branch_url="https://github.com/"+spec["repo"]+"/tree/"+spec["branch"]
        # These commits were read from official currentOid pages on 2026-10-30.
        # Never silently advance a frozen task source on a future rerun.
        rev=spec["revision"]
        files=spec["files"]
        if files is None:
            directory_url="https://github.com/"+spec["repo"]+"/tree/"+rev+"/bbh"
            r=requests.get(directory_url,timeout=(15,45)); r.raise_for_status()
            files=sorted(set(re.findall(r'"path":"(bbh/[^"]+\.json)"',r.text)))
            if len(files)!=27:
                raise ValueError("Unexpected official BBH task file count: "+str(len(files)))
        base=root/"data/raw"/(task+"_official")
        base.mkdir(parents=True,exist_ok=True)
        old_receipt=base/"source_receipt.json"
        old=json.loads(old_receipt.read_text(encoding="utf-8")) if old_receipt.exists() else None
        if old and old["revision"]!=rev:
            raise ValueError("Official source HEAD changed; retained prior pin rather than overwrite")
        manifest=[]
        for name in files:
            destination=base/Path(name).name
            url="https://raw.githubusercontent.com/"+spec["repo"]+"/"+rev+"/"+name
            if destination.exists():
                expected=next((x for x in (old or {}).get("files",[]) if x["source_path"]==name),None)
                if expected and hashlib.sha256(destination.read_bytes()).hexdigest()==expected["sha256"]:
                    payload=destination.read_bytes()
                else:
                    raise ValueError("Existing official task file lacks matching verified provenance")
            else:
                r=requests.get(url,timeout=(15,50)); r.raise_for_status()
                payload=r.content
                destination.write_bytes(payload)
            manifest.append(dict(source_path=name,path=destination.relative_to(root).as_posix(),bytes=len(payload),
                                 sha256=hashlib.sha256(payload).hexdigest(),url=url))
        receipt=dict(repo=spec["repo"],revision=rev,files=manifest,
                     retrieved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                     revision_resolution=dict(method="Official currentOid verified 2026-10-30; immutable commit pinned in downloader",url=branch_url),
                     role="Official problem/reference provenance only; not model outputs or paid feedback.")
        from .download import write_json
        write_json(old_receipt,receipt)
        report[task]=receipt
        print("OFFICIAL_TASK_SOURCE",task,"files",len(files),"revision",rev,flush=True)
    if "gsm8k" in report:
        from .audit import sha_text
        output=root/"data/derived/gsm8k_questions.jsonl"
        output.parent.mkdir(parents=True,exist_ok=True)
        counts={}
        with output.open("w",encoding="utf-8",newline="\n") as out:
            for info in report["gsm8k"]["files"]:
                split=Path(info["source_path"]).stem
                count=0
                for n,line in enumerate((root/info["path"]).open(encoding="utf-8")):
                    if not line.strip():continue
                    row=json.loads(line)
                    question,answer=row["question"],row["answer"]
                    if "####" not in answer:raise ValueError("Official GSM8K answer missing final-answer separator")
                    reference=answer.rsplit("####",1)[1].strip()
                    key=sha_text("gsm8k\n"+split+"\n"+" ".join(question.split()))
                    record=dict(question_id="gsm8k:"+key,task="gsm8k",subtask=None,question=question,
                                reference=reference,source_split=split,source_file=info["path"],source_index=n,
                                source_revision=report["gsm8k"]["revision"],member_outputs={},
                                status="QUESTION_REFERENCE_ONLY_MODEL_GENERATIONS_MISSING")
                    out.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
                    count+=1
                counts[split]=count
        write_json(root/"artifacts/audit/gsm8k_source_audit.json",
                   dict(task="gsm8k",counts=counts,path=output.relative_to(root).as_posix(),
                        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),model_outputs_present=False,
                        reference_rule="Exact official answer suffix after final ####; full worked answers retained in raw.",
                        allocation_warning="Source train/test retained; no calibration/online/final allocation made.",
                        source=report["gsm8k"]))
        print("GSM8K_SOURCE_COUNTS",json.dumps(counts),flush=True)
    return report
