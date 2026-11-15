"""Complete duplicate screening with registered duplicate-source references.

Official BBH repeated-input positions are excluded as new candidates. Any such
input already present in the historical registry must still be included exactly
once as a near-duplicate reference. This append-only check preserves the initial
audit, verifies the full original reference set, and checks the proposed cohort.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from audit_sources import (
    DEFAULT_ROOT, HEADER, INPUT_FIELDS, STUDY_NAME, _rank, _screen_near_duplicates,
    canonical_hash, project, read_json, sha, write_json,
)
from candidate_proposal import protected_screen, safe_source_rows


def main():
    cpu,wall=time.process_time(),time.perf_counter()
    root=DEFAULT_ROOT
    study=root/"research"/STUDY_NAME
    private,public=study/"private",study/"public"
    previous=read_json(public/"DATA_AUDIT.json")
    exclusions=read_json(private/"REGISTERED_EXCLUSION_METADATA.json")["records"]
    expected={row["question_id"]:row for row in exclusions if row["split"]!="sealed_test"}
    protected_ids={row["question_id"] for row in exclusions if row["split"]=="sealed_test"}
    signature_root=private/"protected_input_fingerprints"
    access=read_json(signature_root/"AUTHORIZED_ACCESS_RECEIPT.json")
    if sha(signature_root/"OFFICIAL_BBH_INPUT_IDENTITY_INDEX.json")!=access["bbh_position_index_sha256"]:
        raise ValueError("External position index changed")
    source=read_json(private/"SOURCE_MANIFEST.json")
    for record in source["files"]:
        if sha(root/record["path"])!=record["sha256"]:
            raise ValueError("Frozen source changed")
    index=read_json(signature_root/"OFFICIAL_BBH_INPUT_IDENTITY_INDEX.json")["records"]
    rows,scans,duplicate_groups=safe_source_rows(root,source,exclusions,index)
    present={row["question_id"] for row in rows if row["registered_excluded"]}
    missing=set(expected)-present
    if missing&protected_ids:
        raise ValueError("Protected identity entered nonprotected reference restore")
    restored=[]
    with (root/"data/derived/available_records.jsonl").open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            header=HEADER.match(raw.removeprefix(b"\xef\xbb\xbf"))
            if header is None:
                raise ValueError("Metadata header absent")
            qid=header.group(1).decode("ascii")
            if qid not in missing:
                continue
            row=project(raw,INPUT_FIELDS|{"subtask_source_file"})
            row["canonical_question_sha256"]=canonical_hash(row["question"])
            if row["canonical_question_sha256"]!=expected[qid]["canonical_question_sha256"]:
                raise ValueError("Restored reference canonical identity differs")
            row["source_frame"]="bbh_old_llmrouterbench_common_subpool"
            row["registered_excluded"]=True
            row["reference_only_duplicate_source_position"]=True
            rows.append(row)
            restored.append(qid)
    if set(restored)!=missing or len({row["question_id"] for row in rows if row["registered_excluded"]})!=1508:
        raise ValueError("Nonprotected registered reference coverage remains incomplete")
    groups,pending,edges=_screen_near_duplicates(rows)
    removed_format=set()
    for group in groups:
        registered=[i for i in group if rows[i]["registered_excluded"]]
        keep=None if registered else min(group,key=lambda i:_rank("format-representative",rows[i]["canonical_question_sha256"]))
        removed_format.update(i for i in group if i!=keep)
    if sha(signature_root/"PROTECTED_INPUT_SIGNATURES.json")!=access["signature_sha256"]:
        raise ValueError("Private protected signatures changed")
    signatures=read_json(signature_root/"PROTECTED_INPUT_SIGNATURES.json")["records"]
    removed_protected,protected_edges,signature_summary=protected_screen(rows,(signature_root/"FINGERPRINT_KEY.bin").read_bytes(),signatures)
    eligible=[row for i,row in enumerate(rows) if not row["registered_excluded"] and i not in pending and i not in removed_format and i not in removed_protected]
    gsm=sorted((row for row in eligible if row["task"]=="gsm8k"),key=lambda row:_rank("gsm1300",row["canonical_question_sha256"]))[:1300]
    bbh=defaultdict(list)
    for row in eligible:
        if row["task"]=="bbh":
            bbh[row["subtask"]].append(row)
    selected=list(gsm)
    for subtask,quota in sorted(previous["predeclared_bbh_quotas"].items()):
        candidates=sorted(bbh[subtask],key=lambda row:_rank("bbh1300",row["canonical_question_sha256"]))
        if len(candidates)<quota:
            raise ValueError("A fixed quota cannot be supplied")
        selected.extend(candidates[:quota])
    proposal_path=private/"proposed_new2600_inputs_only.jsonl"
    if sha(proposal_path)!=previous["candidate_input_sha256"]:
        raise ValueError("The original input proposal changed")
    with proposal_path.open("rb") as stream:
        old=[project(raw,{"question_id"})["question_id"] for raw in stream if raw.strip()]
    new=[row["question_id"] for row in selected]
    unchanged=new==old
    write_json(private/"REGISTERED_REFERENCE_COMPLETION_DETAILS.json",dict(
        restored_nonprotected_registered_question_ids=restored,review_edges=edges,
        protected_signature_edges=protected_edges,all_registered_references_present=True,
        exact_reselected_candidate_ids=new,selected_ids_unchanged=unchanged,
    ))
    result=dict(
        schema="e18_registered_reference_completion_v1",
        status="COMPLETE_PROPOSAL_UNCHANGED" if unchanged else "CANDIDATE_REFREEZE_REQUIRED",
        correction="The initial official duplicate-position exclusion also removed one registered input from the near-duplicate reference pool. This check restores registered duplicate inputs exactly once as reference-only rows.",
        prior_data_audit_sha256=sha(public/"DATA_AUDIT.json"),
        registered_exclusion_metadata_questions=1708,nonprotected_registered_reference_questions=1508,
        protected_reference_signatures=200,restored_nonprotected_reference_questions=len(restored),
        full_near_screen_input_questions=len(rows),official_duplicate_groups=len(duplicate_groups),
        format_equivalence_clusters=len(groups),high_similarity_review_edges=len(edges),
        high_similarity_review_endpoint_questions=len(pending),protected_signature_screen=signature_summary,
        eligible_after_complete_reference_screen=dict(Counter(row["task"] for row in eligible)),
        candidate_selection_repeated_with_same_frozen_rank_and_quotas=True,
        candidate_original_order_unchanged=unchanged,candidate_input_sha256=sha(proposal_path),
        protected_input_values_decoded_in_this_supplement=0,answer_values_decoded=0,fresh_reward_reads=0,
        protected_one_time_audit_total_input_values_decoded=200,new_llm_calls=0,new_encoder_forwards=0,
        confidence_intervals=0,bootstrap_runs=0,source_code_sha256=sha(Path(__file__)),
        cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall,
    )
    write_json(public/"DATA_AUDIT_REGISTERED_REFERENCE_SUPPLEMENT.json",result)
    print(json.dumps(result,ensure_ascii=False))
    if not unchanged:
        raise RuntimeError("Do not admit the initial proposal; source-only refreeze is required")


if __name__=="__main__":
    main()
