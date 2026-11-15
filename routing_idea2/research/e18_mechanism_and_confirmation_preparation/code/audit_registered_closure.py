"""Bounded metadata-only closure audit of E1-E17, CES and B1 input lineage.

No reward/gold array, encoder vector, model coefficient or trajectory is decoded.
Mixed NPZ packets are opened only for explicitly named identity/index members.
This audit concerns the listed local canonical metadata, not all possible manual
inspection or every historical/remote process.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import time

import numpy as np

from audit_sources import DEFAULT_ROOT, STUDY_NAME, object_fields, project, read_json, sha, skip_value, write_json


PROTOCOLS={
    "E0_E1_original": ["configs/experiment_v1.json"],
    "E2_original": ["data/derived/online_matrices_v1/matrix_preparation_receipt.json",
        "data/derived/online_matrices_v1/online_calibration.manifest.json", "data/derived/online_matrices_v1/online_stream.manifest.json"],
    "V2_stage1_and_partition": ["protocol/decision_feedback_v2/EXPERIMENT_PROTOCOL_v2.json",
        "artifacts/private/routing_redesign_20261104/PARTITION_PROPOSAL_v2.json",
        "artifacts/private/routing_redesign_20261104/ff_v1/BUNDLE_TRAINVAL.json"],
    "conditional_gain": ["research/conditional_gain_20261106/private/INPUT_LINEAGE.json",
        "research/conditional_gain_20261106/repo/routing_idea2/research/conditional_gain_20261106/CPU_EXPERIMENT_PROTOCOL.json"],
    "fewshot_protocol_fix": ["research/fewshot_protocol_fix_20261107/public/PROTOCOL.md"],
    "E3": ["research/decision_acquisition_e3/public/E3_RUN_CONFIG.json"],
    "E4": ["research/e4_statistical_model_audit/public/E4_EXPERIMENT_CONFIG.json",
        "research/e4_statistical_model_audit/public/SPLIT_PERMISSION_MANIFEST.json"],
    "E5": ["research/e5_paired_gain_validation/public/E5_EXPERIMENT_CONFIG.json",
        "research/e5_paired_gain_validation/public/REFERENCE_AND_INPUTS.json"],
    "E6": ["research/e6_contrast_directed_feedback/public/E6_RUN_CONFIG.json"],
    "E7": ["research/e7_opportunity_and_feedback_learning/public/E7_PROTOCOL.json"],
    "E8": ["research/e8_decision_focused_combination_learning/public/E8_PROTOCOL.json",
        "research/e8_decision_focused_combination_learning/private/qualification/FOLD_PACKS_MANIFEST.json"],
    "E8_M4_mechanism": ["research/e8_m4_mechanism_audit/public/M4_MECHANISM_PROTOCOL.json"],
    "NS_TSPO_Davidson": ["research/m4_noise_aware_set_learning/public/NS_SPO_PROTOCOL.json",
        "research/m4_noise_aware_set_learning/public/JOINT_NOISE_PROTOCOL.json"],
    "E9_CSCE": ["research/e9_consistent_cost_sensitive_learning/public/E9_PROTOCOL.json",
        "research/e9_consistent_cost_sensitive_learning/private/HEAD_SOURCE_LOCK.json"],
    "E9_Bayes_distinct": ["experiments/e9_bayes_gate/protocol_frozen.json", "results/e9_bayes_gate/00_asset_manifest.json"],
    "E10": ["research/e10_input_prefill_matching_audit/public/E10_PROTOCOL.json",
        "research/e10_input_prefill_matching_audit/public/INPUT_ASSET_CHAIN_AUDIT.json"],
    "E11": ["research/e11_group_mean_shrinkage/public/E11_PROTOCOL.json",
        "research/e11_group_mean_shrinkage/private/SOURCE_REFERENCES.json"],
    "E12": ["research/e12_shared_feedback_learning/public/E12_PROTOCOL.json"],
    "E13": ["research/e13_partial_combination_feedback/public/E13_PROTOCOL.json"],
    "E14": ["research/e14_mechanism_diagnostics/public/E14_PROTOCOL.json",
        "research/e14_mechanism_diagnostics/private/INPUT_MANIFEST.json"],
    "E15": ["research/e15_feedback_structure_interaction/public/E15_PROTOCOL.json",
        "research/e15_feedback_structure_interaction/private/INPUT_MANIFEST.json"],
    "E15_preflight_archive": ["research/e15_feedback_structure_interaction_preflight_v1_reference_fix/public/E15_PROTOCOL.json",
        "research/e15_feedback_structure_interaction_preflight_v1_reference_fix/private/INPUT_MANIFEST.json"],
    "E16": ["research/e16_signal_replication_representation/public/E16_PROTOCOL.json",
        "research/e16_signal_replication_representation/private/INPUT_MANIFEST.json"],
    "E17": ["research/e17_supervision_distortion_audit/public/E17_PROTOCOL.json",
        "research/e17_supervision_distortion_audit/private/INPUT_MANIFEST.json",
        "research/e17_supervision_distortion_audit/public/CONFIRMATION_INPUT_ONLY_CONFIG_FROZEN.json",
        "research/e17_supervision_distortion_audit/private/confirmation_input_only/INPUT_AND_CACHE_MANIFEST.json"],
    "CES": ["research/ces_traceable_baseline/public/CES_PROTOCOL.json"],
    "B1": ["research/b1_reliable_paired_evaluation/public/B1_PROTOCOL.json",
        "research/b1_reliable_paired_evaluation/private/assets/FREEZE_RECEIPT.json"],
}


def strings(value):
    if isinstance(value,str):
        yield value
    elif isinstance(value,dict):
        for key,item in value.items():
            yield str(key)
            yield from strings(item)
    elif isinstance(value,list):
        for item in value:
            yield from strings(item)


def input_array_members(path,keys):
    with np.load(path,allow_pickle=False) as archive:
        missing=set(keys)-set(archive.files)
        if missing:
            raise ValueError("Missing explicitly permitted metadata member: "+str(missing))
        return {key:archive[key].copy() for key in keys}


def input_manifest_rows(path,allowed):
    """Project list-of-object input metadata without decoding prompt text."""
    text=Path(path).read_bytes().decode("utf-8-sig")
    position=0
    while text[position].isspace():
        position+=1
    if text[position]!="[":
        raise ValueError("Expected input-only metadata list")
    position+=1
    result=[]
    while True:
        while text[position].isspace():
            position+=1
        if text[position]=="]":
            return result
        end=skip_value(text,position)
        result.append(project(text[position:end],allowed))
        position=end
        while text[position].isspace():
            position+=1
        if text[position]==",":
            position+=1
        elif text[position]!="]":
            raise ValueError("Malformed input metadata list")


def main():
    cpu,wall=time.process_time(),time.perf_counter()
    root=DEFAULT_ROOT
    study=root/"research"/STUDY_NAME
    private,public=study/"private",study/"public"
    registered=read_json(private/"REGISTERED_EXCLUSION_METADATA.json")["records"]
    registry={row["question_id"]:row for row in registered}
    canonical_registry={row["canonical_question_sha256"]:row for row in registered}
    train_ids={row["question_id"] for row in registered if row["split"]=="online_stream"}
    val_ids={row["question_id"] for row in registered if row["split"]=="online_calibration"}
    fresh_ids={row["question_id"] for row in registered if row["split"]=="fresh_dev400"}
    protocol_records=[]
    packets=[]
    extra_ids=set()
    extra_canonical=set()
    metadata_by_module=defaultdict(list)
    packet_modules=defaultdict(list)
    anchors={}

    def identities(module,path,id_key,canonical_key=None,expected=None):
        keys=[id_key]+([canonical_key] if canonical_key else [])
        values=input_array_members(path,keys)
        ids=np.asarray(values[id_key])
        if ids.ndim!=1:
            raise ValueError("Question identities must be one dimensional")
        ids=list(map(str,ids))
        if len(ids)!=len(set(ids)):
            raise ValueError("Repeated question identities in input metadata packet")
        canonicals=list(map(str,values[canonical_key])) if canonical_key else None
        if canonicals is not None and len(canonicals)!=len(ids):
            raise ValueError("ID/canonical metadata length differs")
        extras=set(ids)-set(registry)
        extra_ids.update(extras)
        if canonicals is not None:
            extra_canonical.update(set(canonicals)-set(canonical_registry))
            for qid,fingerprint in zip(ids,canonicals):
                if qid in registry and registry[qid]["canonical_question_sha256"]!=fingerprint:
                    raise ValueError("Registered ID/canonical pairing changed")
        mismatch=len(set(ids)-expected) if expected is not None else 0
        record=dict(module=module,path=str(path.relative_to(root)),metadata_keys_decoded=keys,
                    questions=len(ids),outside_registry_ids=len(extras),outside_expected_role_ids=mismatch,
                    selected_member_only=True,reward_or_embedding_members_decoded=False)
        packets.append(record)
        packet_modules[module].append(record["path"])
        if mismatch:
            raise ValueError("Input packet used identities outside its declared registered cohort")
        return ids

    # Authoritative feature identities, with embeddings left encoded.
    inputs=root/"artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1"
    input_receipt=read_json(inputs/"INPUT_ASSETS_RECEIPT.json")
    for name,expected in (("selector_train",train_ids),("selector_validation",val_ids),("fresh_dev_test",fresh_ids)):
        identities("shared_registered_inputs",inputs/(name+"_inputs.npz"),"question_ids","canonical_hashes",expected)
        anchors[input_receipt["inputs"][name]["sha256"]]=name+"_input_identity_asset"
    old_manifest=root/"data/derived/split_manifest.json"
    anchors[sha(old_manifest)]="original1308_external_split_registry"
    bundle_path=root/"artifacts/private/routing_redesign_20261104/ff_v1/BUNDLE_TRAINVAL.json"
    bundle=read_json(bundle_path)
    anchors[sha(bundle_path)]="registered_train600_val200_bundle_metadata"
    for split,reference in bundle["outcomes"].items():
        if split in ("selector_train","selector_validation"):
            anchors[reference["sha256"]]=split+"_outcome_reference_only_NOT_OPENED"

    # Legacy E2 manifests contain identity fingerprints; their matrices are not opened.
    for split,expected in (("online_calibration",val_ids),("online_stream",train_ids)):
        path=root/f"data/derived/online_matrices_v1/{split}.manifest.json"
        selected=project(path.read_bytes(),{"question_fingerprints","split"})
        fingerprints=selected["question_fingerprints"]
        expected_canonical={registry[qid]["canonical_question_sha256"] for qid in expected}
        if isinstance(fingerprints,dict):
            if set(fingerprints)!=expected or any(registry[qid]["canonical_question_sha256"]!=fingerprint for qid,fingerprint in fingerprints.items()):
                raise ValueError("Original E2 ID/canonical metadata differ")
        elif isinstance(fingerprints,list):
            if len(fingerprints)!=len(expected) or set(fingerprints)!=expected_canonical:
                raise ValueError("Original E2 canonical fingerprint set differs from registered role")
        else:
            raise ValueError("Unsupported original E2 fingerprint schema")
        packets.append(dict(module="E2_original",path=str(path.relative_to(root)),metadata_keys_decoded=["question_fingerprints","split"],questions=len(fingerprints),outside_registry_ids=0,reward_or_embedding_members_decoded=False))
        packet_modules["E2_original"].append(str(path.relative_to(root)))
    identities("E2_original",root/"data/derived/features/features_v1.npz","question_ids",expected=train_ids|val_ids)

    # Validate E8's explicitly scoped history/evaluation identity packets.
    pack_manifest_path=root/"research/e8_decision_focused_combination_learning/private/qualification/FOLD_PACKS_MANIFEST.json"
    pack_manifest=read_json(pack_manifest_path)
    anchors[sha(pack_manifest_path)]="E8_original_train600_fold_identity_manifest"
    for fold in pack_manifest["folds"]:
        for role in ("history","evaluation_inputs"):
            identities("E8",Path(fold[role]["path"]),"ids","canonical",train_ids)

    # Explicit original integer-index metadata belongs to the registered train600.
    index_specs=[
        ("conditional_gain","research/conditional_gain_20261106/private/FOLDS_PRIVATE.npz",r"fold\d+_(train|holdout)"),
        ("E4","research/e4_statistical_model_audit/private/E4_SPLITS_PRIVATE.npz",r"f\d+_(history.*|stream|audit_support|audit_query|order\d+|support_order\d+)"),
        ("E5","research/e5_paired_gain_validation/private/FGE_SPLITS_PRIVATE.npz",r"f\d+_(F|G|E|feedback_G20)"),
        ("E7","research/e7_opportunity_and_feedback_learning/private/E7_SPLITS_FROZEN.npz",r"f\d+_(H|I|U|E|refs)"),
    ]
    for module,relative,pattern in index_specs:
        path=root/relative
        with np.load(path,allow_pickle=False) as archive:
            keys=[name for name in archive.files if re.fullmatch(pattern,name)]
            if not keys:
                raise ValueError("No registered question-index metadata fields found")
            for key in keys:
                values=archive[key]
                if values.ndim!=1 or values.dtype.kind not in "iu" or np.any(values<0) or np.any(values>=600):
                    raise ValueError("Historical split index escaped original train600")
        anchors[sha(path)]=module+"_verified_train600_index_metadata"
        record=dict(module=module,path=relative,metadata_keys_decoded=keys,
                    index_domain="original registered train600 order",outside_registry_ids=0,
                    source_pool_identity_verified=True,reward_or_embedding_members_decoded=False)
        packets.append(record)
        packet_modules[module].append(relative)

    for fold in range(5):
        # E13 source packets feed the subsequent E14-E17 controlled studies.
        path=root/f"research/e13_partial_combination_feedback/private/inputs/fold{fold}/INPUTS_ONLY.npz"
        values=input_array_members(path,["H","E"])
        if set(values["H"])&set(values["E"]) or set(values["H"])|set(values["E"])!=set(range(600)):
            raise ValueError("E13 original index partition differs")
        anchors[sha(path)]="E13_verified_train600_input_packet_fold"+str(fold)
        record=dict(module="E13",path=str(path.relative_to(root)),metadata_keys_decoded=["H","E"],questions=600,
                    outside_registry_ids=0,index_domain="registered_train600",reward_or_embedding_members_decoded=False)
        packets.append(record);packet_modules["E13"].append(record["path"])
        path=root/f"research/m4_noise_aware_set_learning/private/runtime/noise/private/fold{fold}/POSTERIOR_TRAINING.npz"
        identities("NS_TSPO_Davidson",path,"history_ids",expected=train_ids)
        for seed in (17,29,43):
            path=root/f"research/ces_traceable_baseline/private/prepared/FEATURES_f{fold}_s{seed}.npz"
            values=input_array_members(path,["I","U","E"])
            groups=[set(map(int,values[name])) for name in ("I","U","E")]
            if [len(group) for group in groups]!=[240,240,120] or set.union(*groups)!=set(range(600)) or any(groups[i]&groups[j] for i,j in ((0,1),(0,2),(1,2))):
                raise ValueError("CES prepared input indices escaped registered train600")
            record=dict(module="CES",path=str(path.relative_to(root)),metadata_keys_decoded=["I","U","E"],questions=600,
                        outside_registry_ids=0,index_domain="registered_train600",reward_or_embedding_members_decoded=False)
            packets.append(record);packet_modules["CES"].append(record["path"])

    identities("B1",root/"research/b1_reliable_paired_evaluation/private/assets/FROZEN_INPUTS.npz","val_ids","canonical_hashes",val_ids)
    identities("fewshot_protocol_fix",root/"research/fewshot_protocol_fix_20261107/private/FRESH400_INPUT_ONLY_PLAN.npz","question_ids",expected=fresh_ids)
    for relative in ("research/e17_supervision_distortion_audit/private/confirmation_input_only/FRESH_POLICY_ACTIONS_FROZEN.npz",
                     "research/e17_supervision_distortion_audit/private/fresh_numbering_input_only/FRESH_NUMBERING_ACTIONS_FROZEN.npz"):
        identities("E17",root/relative,"question_ids","canonical_hashes",fresh_ids)

    prefill_path=root/"research/e10_input_prefill_matching_audit/private/input/PREFILL_INPUT_MANIFEST.json"
    prefill=input_manifest_rows(prefill_path,{"global_index","question_id","canonical_id"})
    if len(prefill)!=600 or {row["question_id"] for row in prefill}!=train_ids:
        raise ValueError("E10 encoder input provenance escaped registered train600")
    for row in prefill:
        if row["canonical_id"]!=registry[row["question_id"]]["canonical_question_sha256"]:
            raise ValueError("E10 encoder input canonical identity differs")
    packets.append(dict(module="E10",path=str(prefill_path.relative_to(root)),metadata_keys_decoded=["global_index","question_id","canonical_id"],questions=600,outside_registry_ids=0,reward_or_embedding_members_decoded=False))
    packet_modules["E10"].append(str(prefill_path.relative_to(root)))
    feature_receipt_path=root/"research/e10_input_prefill_matching_audit/private/features/FEATURE_FREEZE_RECEIPT.json"
    feature_receipt=read_json(feature_receipt_path)
    # E11/E12 identify this cache by hash. Only its provenance receipt is read.
    feature_strings=list(strings(feature_receipt))
    prefill_feature_sha="4235fe7a68fd95bd128c6c34543d0c142986e53dcc886182bf5687f058bc3d19"
    if prefill_feature_sha not in feature_strings:
        raise ValueError("Prefill cache identity missing from receipt")
    anchors[prefill_feature_sha]="E10_encoder_cache_input_manifest_verified600_VECTOR_NOT_DECODED"

    for module,relatives in PROTOCOLS.items():
        for relative in relatives:
            path=root/relative
            if not path.is_file():
                protocol_records.append(dict(module=module,path=relative,status="MISSING_METADATA_CONTAINER"))
                continue
            if path.suffix.lower()==".json":
                obj=read_json(path)
                values=list(strings(obj))
            else:
                values=[path.read_text(encoding="utf-8-sig")]
            matched=sorted({anchors[value] for value in values if value in anchors})
            record=dict(module=module,path=relative,sha256=sha(path),bytes=path.stat().st_size,
                        status="READ_METADATA_ONLY",matched_registered_source_anchors=matched)
            protocol_records.append(record)
            metadata_by_module[module].append(record)

    modules=[]
    # Some older mechanisms have no separate identity export, so their evidence
    # remains an explicit frozen-scope declaration plus the verified E8 parent.
    parent_links={"E3":"conditional_gain", "E8_M4_mechanism":"E8", "E9_CSCE":"E8",
                  "E9_Bayes_distinct":"E8", "E11":"E10", "E12":"E7",
                  "E14":"E13", "E15":"E13", "E15_preflight_archive":"E13", "E16":"E13"}
    declared_only=[]
    for module in PROTOCOLS:
        records=metadata_by_module[module]
        matched=sorted({anchor for record in records for anchor in record["matched_registered_source_anchors"]})
        direct=packet_modules[module]
        if direct:
            level="IDENTITY_OR_INDEX_CONTAINER_VERIFIED"
        elif matched:
            level="REGISTERED_SOURCE_HASH_REFERENCE_VERIFIED"
        else:
            level="FROZEN_SCOPE_DECLARATION_AND_PARENT_LINEAGE_ONLY"
            declared_only.append(module)
        modules.append(dict(module=module,reviewed_metadata_containers=len(records),evidence_level=level,
            registered_source_hash_anchors=matched,identity_or_index_packets_checked=len(direct),
            parent_module=parent_links.get(module),outside_registered_ids_found=0,
            all_individual_trajectory_inputs_reaudited=False))
    unreviewed=[
        dict(container="E13/ALL_PREDICTIONS_FROZEN and per-world feedback/service traces",reason="55MB manifest and many replay packets excluded; checked canonical protocol plus five input packets instead"),
        dict(container="E14/TOKEN_MATCHES_FROZEN and matched-prefix model packets",reason="35MB ledger and response data excluded; checked protocol/source manifest lineage to E13"),
        dict(container="E15/E16/E17 all model packets, labels, score arrays and per-run traces",reason="Only canonical source manifests and selected identity metadata used"),
        dict(container="Fewshot preliminary/episode result NPZ containers",reason="No reward/result arrays read; corrected protocol and frozen fresh input identity plan checked"),
        dict(container="CES native-author online trajectories and external datasets",reason="Original protocol states native full results cannot be replayed without trajectories; native text/correctness data are outside this MAB input audit"),
        dict(container="B1 public-method synthetic/native example data and Monte Carlo outcomes",reason="Outside the real GSM8K/BBH question-pool closure; source protocol and MAB frozen inputs checked"),
        dict(container="Remote archives, hidden temporary runtime copies, unrelated data roots",reason="Not enumerated or reviewed by this bounded local canonical metadata audit"),
        dict(container="Unregistered manual inspection, copied prompts or undocumented experiments",reason="UNKNOWN; no OS-wide access record exists"),
    ]
    details=dict(schema="private_historical_input_closure_details_v1",metadata_records=protocol_records,
                 identity_packet_records=packets,registered_source_anchors=anchors,
                 outside_registered_question_ids=sorted(extra_ids),outside_registered_canonical_hashes=sorted(extra_canonical),
                 unreviewed_containers=unreviewed)
    write_json(private/"HISTORICAL_INPUT_REGISTRY_CLOSURE_DETAILS.json",details)
    public_result=dict(
        schema="e18_historical_input_registration_closure_v1",status="NO_OUTSIDE_REGISTERED_INPUT_FOUND_IN_REVIEWED_CANONICAL_METADATA" if not extra_ids and not extra_canonical else "EXTRA_IDENTITY_REQUIRES_REVIEW",
        registered_questions=1708,reviewed_research_modules=len(modules),
        protocol_and_source_metadata_containers=sum(record.get("status")=="READ_METADATA_ONLY" for record in protocol_records),
        explicitly_checked_identity_or_index_containers=len(packets),
        outside_registered_question_ids=len(extra_ids),outside_registered_canonical_hashes=len(extra_canonical),
        modules=modules,scope_declaration_only_modules=declared_only,
        unreviewed_containers=unreviewed,
        manual_unregistered_input_use="UNKNOWN",all_project_possible_inputs_certified=False,
        interpretation="Reviewed local protocol/source/hash references and selected identity packets are consistent with the registered1708. This is not a full per-run or OS-wide proof of every historical input access.",
        rewards_decoded=0,gold_values_decoded=0,embedding_arrays_decoded=0,model_weights_decoded=0,
        protected_input_values_decoded_this_audit=0,new_llm_calls=0,new_encoder_forwards=0,
        existing_proposal_modified=False,existing_proposal_sha256=sha(private/"proposed_new2600_inputs_only.jsonl"),
        private_detail_sha256=sha(private/"HISTORICAL_INPUT_REGISTRY_CLOSURE_DETAILS.json"),
        source_sha256=sha(Path(__file__)),cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall,
    )
    write_json(public/"HISTORICAL_INPUT_REGISTRY_CLOSURE.json",public_result)
    text=f"""# 历史实验输入注册闭包核查

## 范围与方法

本次查阅{len(modules)}个研究模块的{public_result['protocol_and_source_metadata_containers']}份协议或来源元数据，额外检查{len(packets)}个身份或整数索引容器。范围覆盖原E1/E2、V2角色划分、条件增益与Fewshot修正、E3至E17、两个不同E9、NS、CES与B1。混合NPZ只解码显式允许的ID/canonical/历史索引成员，向量、评分、答案、权重及逐回放结果不解码。

## 结果

查阅范围内注册外原题ID为{len(extra_ids)}，注册外canonical hash为{len(extra_canonical)}。原E2的200/600矩阵来源manifest与注册角色一致；E8五折身份包、E4/E5/E7整数分割、CES15个输入准备包、NS历史ID、B1验证池ID、E10输入来源、E13五折输入索引，以及E17新输入冻结身份均落在既有1708注册内。已有2600拟议清单保持不变。

## 证据边界

旧机制中仅具冻结范围声明和父资产链的模块为{json.dumps(declared_only,ensure_ascii=False)}，已单列，未将其提升为逐运行完整验证。E13大型冻结清单、E14完整成本账本、海量模型/回放、外部原生例子和远端临时副本未逐条审阅。手工未登记审阅与未保存输入的操作范围为UNKNOWN，不能据此声称整个项目所有可能访问过的问题均已证明覆盖。

## 决定

当前未发现需要追加排除的新身份。若后续出现新的外部输入注册，应先核对与拟议2600的身份/近重复关系，再决定是否重新冻结候选。本次不修改候选、不读取reward/gold、不新增推理或编码。
"""
    with (public/"HISTORICAL_INPUT_REGISTRY_CLOSURE_ZH.md").open("x",encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({key:public_result[key] for key in ("status","reviewed_research_modules","protocol_and_source_metadata_containers","explicitly_checked_identity_or_index_containers","outside_registered_question_ids","scope_declaration_only_modules","existing_proposal_sha256")},ensure_ascii=False))


if __name__=="__main__":
    main()
