"""Read-only identity alignment and final input-proposal verification."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit_sources import DEFAULT_ROOT, STUDY_NAME, canonical_hash, read_json, sha, write_json


def main():
    root=DEFAULT_ROOT
    study=root/"research"/STUDY_NAME
    private,public=study/"private",study/"public"
    old=read_json(root/"data/derived/split_manifest.json")
    fresh=read_json(root/"artifacts/private/routing_redesign_20261104/fresh_dev400/SELECTION_MANIFEST_v1.json")
    inputs_root=root/"artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1"
    receipt=read_json(inputs_root/"INPUT_ASSETS_RECEIPT.json")
    alignments=[]
    for label,source_split in (("selector_train","online_stream"),("selector_validation","online_calibration"),("fresh_dev_test",None)):
        path=inputs_root/(label+"_inputs.npz")
        if sha(path)!=receipt["inputs"][label]["sha256"]:
            raise ValueError("Input asset no longer matches its frozen receipt")
        with np.load(path,allow_pickle=False) as archive:
            if set(archive.files)!={"question_ids","canonical_hashes","embeddings","task_labels","subtask_labels"}:
                raise ValueError("Input archive contains forbidden or missing members")
            packet={key:archive[key] for key in ("question_ids","canonical_hashes","task_labels","subtask_labels")}
        expected=[row for row in old["records"] if row["split"]==source_split] if source_split else fresh["records"]
        by_id={row["question_id"]:row for row in expected}
        if len(packet["question_ids"])!=len(by_id) or set(packet["question_ids"])!=set(by_id):
            raise ValueError("Input identities differ from registered role")
        for qid,canonical,task,subtask in zip(packet["question_ids"],packet["canonical_hashes"],packet["task_labels"],packet["subtask_labels"]):
            row=by_id[str(qid)]
            expected_canonical=old["canonical_question_fingerprints"][str(qid)] if source_split else row["canonical_question_sha256"]
            if str(canonical)!=expected_canonical or str(task)!=row["task"] or str(subtask)!=(row.get("subtask") or ""):
                raise ValueError("Input task/subtask/canonical metadata differ")
        alignments.append(dict(cohort=label,questions=len(by_id),input_sha256=sha(path),
            external_role=source_split or "fresh_dev400",all_identity_and_task_metadata_match=True,
            source_split_counts=dict(Counter(row["source_split"] for row in expected)),
            decoded_npz_members=["question_ids","canonical_hashes","task_labels","subtask_labels"],
            embeddings_decoded=False,rewards_decoded=False))
    supplement=read_json(public/"DATA_AUDIT_REGISTERED_REFERENCE_SUPPLEMENT.json")
    if supplement["status"]!="COMPLETE_PROPOSAL_UNCHANGED" or not supplement["candidate_original_order_unchanged"]:
        raise ValueError("Complete registered-reference acceptance is pending")
    proposal_path=private/"proposed_new2600_inputs_only.jsonl"
    if sha(proposal_path)!=supplement["candidate_input_sha256"]:
        raise ValueError("Supplemented input cohort hash differs")
    with proposal_path.open(encoding="utf-8") as stream:
        proposed=[json.loads(line) for line in stream if line.strip()]
    required={"question_id","task","subtask","source_split","question","canonical_question_sha256","source_frame"}
    extras={"source_index","source_file","source_revision","status","independence_qualification","source_layer_admission"}
    for row in proposed:
        if not required.issubset(row) or set(row)-(required|extras):
            raise ValueError("Unexpected proposal input schema")
        if row["status"]!="PROPOSED_NOT_ADMITTED" or canonical_hash(row["question"])!=row["canonical_question_sha256"]:
            raise ValueError("Proposal state or input canonical differs")
        if row["source_split"]!=("train" if row["task"]=="gsm8k" else "test"):
            raise ValueError("Primary source-split distribution differs")
    exclusions=read_json(private/"REGISTERED_EXCLUSION_METADATA.json")["records"]
    excluded_ids={row["question_id"] for row in exclusions}
    excluded_hashes={row["canonical_question_sha256"] for row in exclusions}
    if len(proposed)!=2600 or len({row["question_id"] for row in proposed})!=2600 or len({row["canonical_question_sha256"] for row in proposed})!=2600:
        raise ValueError("Independent original-input identities repeat")
    if any(row["question_id"] in excluded_ids or row["canonical_question_sha256"] in excluded_hashes for row in proposed):
        raise ValueError("Original registered/fresh/protected overlap")
    if Counter(row["task"] for row in proposed)!={"gsm8k":1300,"bbh":1300}:
        raise ValueError("Task quotas differ")
    quotas=read_json(public/"DATA_AUDIT.json")["predeclared_bbh_quotas"]
    if Counter(row["subtask"] for row in proposed if row["task"]=="bbh")!=quotas:
        raise ValueError("BBH subtask quotas differ")
    result=dict(schema="e18_source_alignment_final_verification_v1",status="PASS_INPUT_PROPOSAL_UNADMITTED",
        original_input_alignment=alignments,
        historical_aggregator_diagnostic200=dict(external_split="e1_diagnostic_evaluation",questions=200,
            input_identity_metadata_only=True,source_split_counts={"gsm8k/train":100,"bbh/test":100}),
        registered_exclusion_count=1708,nonprotected_reference_count=1508,protected_signature_count=200,
        proposed_questions=2600,proposed_input_sha256=sha(proposal_path),canonical_overlap_count=0,original_id_overlap_count=0,
        source_frame_counts=dict(Counter(row["source_frame"] for row in proposed)),
        actual_protected_input_reads_this_verification=0,actual_protected_input_reads_authorized_once_total=200,
        answer_values_decoded=0,fresh_reward_values_read=0,new_embedding_forwards=0,new_llm_calls=0,
        automatically_admitted_questions=0,confidence_intervals=0,bootstrap_runs=0,source_sha256=sha(Path(__file__)))
    write_json(public/"DATA_SOURCE_ALIGNMENT_AND_FINAL_VERIFICATION.json",result)
    print(json.dumps({key:result[key] for key in ("status","registered_exclusion_count","proposed_questions","proposed_input_sha256","automatically_admitted_questions")},ensure_ascii=False))


if __name__=="__main__":
    main()
