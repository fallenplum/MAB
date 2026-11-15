"""Append-only repair of E18 input-only subtask dtype from locked metadata.

The v1 object-array payload is never decoded. Existing numeric/Unicode arrays
are copied with allow_pickle=False and checked byte-for-byte after writing v2.
No encoder, head fitting, prediction, answer generation or reward access runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import sys
import time
import zipfile

from e18_common import read_json, sha256, write_json
import numpy as np


SOURCE=Path(__file__).resolve()
EXPECTED_CANDIDATE_SHA="bfef0550d9ff18de6ff625ed72767284224ce0d789ef8df1c38ce5e05ea08e12"
EXPECTED_V1_SHA="312420e0d714d027fd5f944ba52f8335fd90dfc13283881752d3f956bc221d27"
SAFE_FIELDS=("embeddings","question_ids","canonical_hashes","task_labels")


def array_hash(array):
    array=np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--asset-root",type=Path,default=Path(os.environ.get("MAB_ASSET_ROOT",r"E:\语言模型组合路由")))
    args=parser.parse_args()
    cpu,wall=time.process_time(),time.perf_counter()
    study=args.asset_root/"research/e18_mechanism_and_confirmation_preparation"
    private,public=study/"private",study/"public"
    original=private/"PROPOSED2600_INPUTS_ONLY.npz"
    repaired=private/"PROPOSED2600_INPUTS_ONLY_v2.npz"
    candidate=private/"proposed_new2600_inputs_only.jsonl"
    receipt_name="INPUT_METADATA_REPAIR_v2.json"
    if repaired.exists() or any((folder/receipt_name).exists() for folder in (private,public)):
        raise FileExistsError("Append-only v2 repair artifacts already exist")
    checks=[]
    def require(condition,label):
        if not bool(condition):raise AssertionError(label)
        checks.append(label)
    require(sha256(original)==EXPECTED_V1_SHA,"original v1 NPZ matches frozen input receipt")
    require(sha256(candidate)==EXPECTED_CANDIDATE_SHA,"candidate JSONL matches frozen candidate SHA")
    frozen_receipt=read_json(private/"FROZEN_BUDGET_RECEIPT.json")
    require(frozen_receipt["input_vectors_sha256"]==EXPECTED_V1_SHA,"v1 budget receipt identity matches")
    untouched=[original,private/"FROZEN3000_INPUT_PREDICTIONS.npz",private/"FROZEN_CALL_UNION_3000.json",
               public/"E18_EXACT_CALL_BUDGET.json",private/"FROZEN_BUDGET_RECEIPT.json"]
    before={str(path):sha256(path) for path in untouched}

    # Read only the .npy header of the unsafe field. Its pickle bytes stay closed.
    with zipfile.ZipFile(original) as archive:
        with archive.open("subtask_labels.npy") as stream:
            version=np.lib.format.read_magic(stream)
            old_shape,old_fortran,old_dtype=np.lib.format._read_array_header(stream,version)
    require(old_shape==(2600,) and old_dtype.hasobject,"v1 defect is confined to object subtask header")
    with np.load(original,allow_pickle=False) as archive:
        require(set(archive.files)==set(SAFE_FIELDS)|{"subtask_labels"},"v1 contains the five declared input-only fields")
        arrays={field:archive[field].copy() for field in SAFE_FIELDS}
    require(all(not array.dtype.hasobject for array in arrays.values()),"all copied source fields are pickle-free")
    rows=[]
    import json
    with candidate.open(encoding="utf-8-sig") as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    require(len(rows)==2600,"locked candidate has2600 rows")
    require([row["question_id"] for row in rows]==arrays["question_ids"].tolist(),"question order matches locked candidate")
    require([row["canonical_question_sha256"] for row in rows]==arrays["canonical_hashes"].tolist(),"canonical order matches locked candidate")
    require([row["task"] for row in rows]==arrays["task_labels"].tolist(),"task order matches locked candidate")
    require(len(set(arrays["question_ids"]))==len(set(arrays["canonical_hashes"]))==2600,"candidate identities remain unique")
    require(Counter(row["task"] for row in rows)=={"gsm8k":1300,"bbh":1300},"candidate task counts remain1300/1300")
    require(all(row.get("subtask") is None or isinstance(row.get("subtask"),str) for row in rows),"subtasks use only declared null/string metadata")
    subtask_labels=np.asarray([row.get("subtask") or "" for row in rows],dtype=str)
    require(subtask_labels.dtype.kind=="U" and not subtask_labels.dtype.hasobject,"normalized subtask labels have Unicode dtype")
    require(all(label=="" for row,label in zip(rows,subtask_labels) if row["task"]=="gsm8k"),"GSM8K subtasks normalize to empty string")
    sys.path.insert(0,str(args.asset_root/"src"))
    from budget_router_v2.scoring import BBH_TASKS
    bbh_labels=[label for row,label in zip(rows,subtask_labels) if row["task"]=="bbh"]
    require(set(bbh_labels)==set(BBH_TASKS),"BBH retains exactly27 declared subtasks")
    safe_hashes={field:array_hash(array) for field,array in arrays.items()}
    payload={**arrays,"subtask_labels":subtask_labels}
    with repaired.open("xb") as stream:
        np.savez_compressed(stream,**payload)
    with np.load(repaired,allow_pickle=False) as archive:
        require(set(archive.files)==set(payload),"v2 field set matches the input-only schema")
        restored={field:archive[field].copy() for field in archive.files}
    require(all(not array.dtype.hasobject for array in restored.values()),"all five v2 arrays load with allow_pickle=False")
    for field in SAFE_FIELDS:
        require(restored[field].dtype==arrays[field].dtype and restored[field].shape==arrays[field].shape,
                field+" dtype and shape unchanged")
        require(array_hash(restored[field])==safe_hashes[field] and np.array_equal(restored[field],arrays[field]),
                field+" values and bytes unchanged")
    require(np.array_equal(restored["subtask_labels"],subtask_labels),"v2 subtasks exactly match locked normalized metadata")
    after={str(path):sha256(path) for path in untouched}
    require(before==after,"v1 inputs, predictions, call union and budget remain byte-identical")
    record=dict(status="PASS",schema="E18_input_metadata_dtype_repair_v2",checks=len(checks),check_names=checks,
        source_sha256=sha256(SOURCE),candidate_sha256=sha256(candidate),
        original_input=dict(path=str(original),sha256=sha256(original),role="preserved v1 metadata-only dtype defect",
                            unsafe_field="subtask_labels",unsafe_field_dtype=str(old_dtype),unsafe_field_shape=list(old_shape)),
        repaired_input=dict(path=str(repaired),sha256=sha256(repaired),role="input-only Unicode metadata repair",
                            subtask_dtype=str(subtask_labels.dtype),fields=sorted(payload)),
        unchanged_array_hashes=safe_hashes,normalized_null_subtasks=sum(row.get("subtask") is None for row in rows),
        task_counts=dict(Counter(row["task"] for row in rows)),
        source_frame_counts=dict(Counter(row["source_frame"] for row in rows)),
        source_frame_boundary="BBH full-source expansion remains separately identified; this dtype repair makes no same-source-distribution admission",
        unsafe_object_payload_decoded=False,allow_pickle=False,
        unchanged_owner_artifacts=[dict(path=path,sha256=digest) for path,digest in before.items()],
        new_encoder_forwards=0,pca_refits=0,scaler_refits=0,head_fits=0,predictions_recomputed=0,
        changed_service_choices=0,changed_logical_call_counts=0,physical_llm_calls=0,
        fresh_reward_reads=0,sealed_payload_reads=0,e17_writes=0,
        cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall)
    write_json(private/receipt_name,record)
    write_json(public/receipt_name,record)
    print(json.dumps(dict(status=record["status"],checks=record["checks"],repaired_input_sha256=record["repaired_input"]["sha256"],
                         new_encoder_forwards=0,predictions_recomputed=0),ensure_ascii=False))


if __name__=="__main__":
    main()
