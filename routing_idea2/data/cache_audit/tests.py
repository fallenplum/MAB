from pathlib import Path
import io, json, tarfile, tempfile
from .download import validate_member, safe_extract
from .audit import valid_record, audit_cache, DEFAULT_MODELS

def run_tests(root):
    checks=[]
    good=tarfile.TarInfo("bench-release/bbh/test/model/file.json")
    good.size=1
    validate_member(good)
    checks.append("normal archive path accepted")
    for name in ("../escape","/absolute","C:/drive","a/../../escape","a\\escape","a/CON","a/trailing. "):
        member=tarfile.TarInfo(name)
        try:
            validate_member(member)
        except ValueError:
            pass
        else:
            raise AssertionError("Traversal accepted")
    checks.append("traversal and Windows path escape rejected")
    for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.CHRTYPE,tarfile.FIFOTYPE):
        member=tarfile.TarInfo("link")
        member.type=kind
        try:
            validate_member(member)
        except ValueError:
            pass
        else:
            raise AssertionError("Nonregular entry accepted")
    checks.append("links/devices/fifo rejected")
    row=dict(index=1,origin_query="2 + 2?",raw_output="answer is 4",prediction="4",
             ground_truth="4",score=1.0,prompt_tokens=10,completion_tokens=4,cost=0.0)
    assert not valid_record(row)
    no_raw=dict(row)
    no_raw.pop("raw_output")
    assert "missing_raw_output" in valid_record(no_raw)
    assert "invalid_score" in valid_record(dict(row,score=float("nan")))
    assert "invalid_prompt_tokens" in valid_record(dict(row,prompt_tokens=-1))
    checks.append("parsed prediction cannot replace raw output; malformed numeric fields rejected")
    audit_root=Path(root)/"artifacts/audit"
    audit_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cache_audit_test_",dir=audit_root) as folder:
        folder=Path(folder)
        archive=folder/"safe.tar.gz"
        with tarfile.open(archive,"w:gz") as tar:
            item=tarfile.TarInfo("safe/file.txt")
            data=b"fixture"
            item.size=len(data)
            tar.addfile(item,io.BytesIO(data))
        dest=folder/"extracted"
        safe_extract(archive,dest)
        assert (dest/"safe/file.txt").read_bytes()==b"fixture"
        safe_extract(archive,dest)
        checks.append("safe extraction receipt verifies reuse")
        (dest/"safe/file.txt").write_bytes(b"tampered")
        try:
            safe_extract(archive,dest)
        except ValueError:
            pass
        else:
            raise AssertionError("Tampering accepted")
        checks.append("tampered existing extraction rejected")
        cache=folder/"cache"
        for task in ("gsm8k","bbh"):
            for model in DEFAULT_MODELS:
                p=cache/task/model/"run.json"
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(json.dumps(dict(model_name=model,dataset_name=task,split="fixture",
                                             counts=1,records=[row])),encoding="utf-8")
        result=audit_cache(root,cache,output_dir=folder/"audit1",verify_distribution=False)
        assert result["status"]=="E0_CACHE_STRUCTURE_READY"
        assert result["target_task_details"]["bbh"][0]["six_model_valid_common_questions"]==1
        checks.append("six-model question alignment works without treating local indices as global IDs")
        from .normalize import normalize_available
        reports=json.loads((folder/"audit1/file_audit.json").read_text(encoding="utf-8"))
        normalized=normalize_available(folder,cache,reports,DEFAULT_MODELS,require_bbh_subtasks=False)
        assert normalized["task_counts"]=={"bbh":1,"gsm8k":1}
        rows=[json.loads(line) for line in (folder/"data/derived/available_records.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(len(x["member_outputs"])==6 and x["question"]==row["origin_query"] for x in rows)
        assert all(v["weight_revision"] is None for v in normalized["model_identity"])
        checks.append("normalization retains real raw outputs, six-model source identity and unguessed revisions")
        from . import normalize as normalizer
        prior_index=normalizer.official_bbh_index
        try:
            norm=normalizer.nfkc_query(row["origin_query"])
            fake_meta=dict(repo="SYNTHETIC_FIXTURE",revision="fixture",examples=1)
            normalizer.official_bbh_index=lambda unused: ({norm:{("fixture_task","4"):dict(subtask="fixture_task",reference="4",source_file="synthetic.json")}},fake_meta)
            normalized=normalize_available(folder,cache,reports,DEFAULT_MODELS)
            assert normalized["task_counts"]=={"bbh":1,"gsm8k":1}
            assert normalized["bbh_subtask_mapping"]["counts"]["matched"]==1
            normalizer.official_bbh_index=lambda unused: ({norm:{("fixture_task","4"):dict(subtask="fixture_task",reference="4",source_file="synthetic.json"),("fixture_task","5"):dict(subtask="fixture_task",reference="5",source_file="synthetic.json")}},fake_meta)
            normalized=normalize_available(folder,cache,reports,DEFAULT_MODELS)
            assert normalized["task_counts"].get("bbh",0)==0
            assert normalized["bbh_subtask_mapping"]["counts"]["official_duplicate_input_reference_conflict"]==1
        finally:
            normalizer.official_bbh_index=prior_index
        checks.append("unique official-subtask match accepted; conflicting references fail closed per question")
        p=cache/"gsm8k"/DEFAULT_MODELS[0]/"run.json"
        obj=json.loads(p.read_text())
        obj["records"][0]["ground_truth"]="different"
        p.write_text(json.dumps(obj),encoding="utf-8")
        result=audit_cache(root,cache,output_dir=folder/"audit2",verify_distribution=False)
        assert result["status"]=="E0_BLOCKED_CACHE_REQUIREMENTS"
        assert result["target_task_details"]["gsm8k"][0]["six_model_valid_common_questions"]==0
        checks.append("cross-model truth mismatch blocks usable common coverage")
        for p in (cache/"gsm8k").rglob("*.json"):
            p.unlink()
        result=audit_cache(root,cache,output_dir=folder/"audit3",verify_distribution=False)
        assert "gsm8k" in result["missing_required_tasks"]
        checks.append("missing GSM8K blocks first protocol; BBH cannot substitute")
    receipt=dict(status="PASS",checks=checks,real_cache_used=False,paid_api_calls=0)
    from .download import write_json
    write_json(audit_root/"cache_audit_self_test.json",receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
