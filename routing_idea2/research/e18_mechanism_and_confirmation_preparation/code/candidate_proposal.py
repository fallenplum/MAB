"""Prepare unadmitted input-only candidates using private protected signatures."""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from math import sqrt
from pathlib import Path
import time

from audit_sources import (
    HEADER, INPUT_FIELDS, METADATA_FIELDS, NEAR_RULES, SEED, _keyed_hash,
    _private_signature, _rank, _screen_near_duplicates, _slots, canonical_hash,
    official_example_objects, project, read_json, sha, write_json, write_jsonl,
)


def protected_screen(rows,key,protected_records):
    """Compare one-way input signatures without reconstructing protected text."""
    by_slots=defaultdict(list)
    for record in protected_records:
        by_slots[record["critical_slots_hmac256"]].append(record)
    removed,review_edges=set(),[]
    slot_matches=cosine_checks=ordered_checks=0
    for index,row in enumerate(rows):
        if row["registered_excluded"]:
            continue
        matches=by_slots.get(_keyed_hash(key,_slots(row)),[])
        if not matches:
            continue
        candidate=_private_signature(row,key)
        first_counts=candidate["character_gram_counts"]
        first_norm=sqrt(sum(value*value for value in first_counts.values()))
        for protected in matches:
            slot_matches+=1
            equivalent=candidate["ordered_tokens_hmac256"]==protected["ordered_tokens_hmac256"]
            if equivalent:
                removed.add(index)
                review_edges.append(dict(candidate_question_id=row["question_id"],protected_question_id=protected["question_id"],
                    reason="ordered_token_equivalent_to_protected_input",disposition="EXCLUDE_PROPOSED_CANDIDATE"))
                continue
            second_counts=protected["character_gram_counts"]
            second_norm=sqrt(sum(value*value for value in second_counts.values()))
            numerator=sum(value*second_counts.get(token,0) for token,value in first_counts.items())
            cosine=numerator/(first_norm*second_norm) if first_norm and second_norm else 0.0
            cosine_checks+=1
            if cosine+1e-12<.985:
                continue
            ratio=SequenceMatcher(None,candidate["ordered_trigram_hmac256"],protected["ordered_trigram_hmac256"],autojunk=False).ratio()
            first=set(candidate["word_token_hmac256"])
            second=set(protected["word_token_hmac256"])
            jaccard=len(first&second)/len(first|second) if first|second else 1.0
            ordered_checks+=1
            if ratio>=.985 and jaccard>=.95:
                removed.add(index)
                review_edges.append(dict(candidate_question_id=row["question_id"],protected_question_id=protected["question_id"],
                    reason="protected_hash_signature_high_similarity",character_gram_count_cosine=cosine,
                    ordered_trigram_ratio=ratio,word_hash_jaccard=jaccard,disposition="EXCLUDE_PENDING_REVIEW"))
    return removed,review_edges,dict(
        protected_signature_records=len(protected_records),candidate_inputs_considered=sum(not row["registered_excluded"] for row in rows),
        matching_critical_slot_pairs=slot_matches,character_hash_cosine_pairs=cosine_checks,
        ordered_hash_pairs=ordered_checks,excluded_candidate_inputs=len(removed),
        protected_text_reconstructed=False,additional_protected_input_values_decoded=0,
    )


def safe_source_rows(root,source_manifest,exclusions,official_index):
    protected_ids={row["question_id"] for row in exclusions if row["split"]=="sealed_test"}
    excluded_ids={row["question_id"] for row in exclusions}
    excluded_hashes={row["canonical_question_sha256"] for row in exclusions}
    expected={row["question_id"]:row["canonical_question_sha256"] for row in exclusions}
    rows,scans=[],[]
    gsm_counts=Counter()
    with (root/"data/derived/gsm8k_questions.jsonl").open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            gsm_counts["physical_rows"]+=1
            header=HEADER.match(raw.removeprefix(b"\xef\xbb\xbf"))
            if header is None:
                raise ValueError("GSM qid header unavailable before protected skip")
            qid=header.group(1).decode("ascii")
            if qid in protected_ids:
                gsm_counts["protected_rows_skipped_before_input_decoding"]+=1
                continue
            metadata=project(raw,METADATA_FIELDS)
            if metadata["source_split"]!="train":
                gsm_counts["nonprimary_official_test_rows_metadata_only"]+=1
                continue
            row=project(raw,INPUT_FIELDS|{"source_file","source_index","source_revision"})
            row["canonical_question_sha256"]=canonical_hash(row["question"])
            if qid in expected and row["canonical_question_sha256"]!=expected[qid]:
                raise ValueError("Registered GSM input identity differs")
            row["source_frame"]="gsm8k_official_train_original_primary_frame"
            row["registered_excluded"]=qid in excluded_ids or row["canonical_question_sha256"] in excluded_hashes
            rows.append(row)
            gsm_counts["allowed_input_values_decoded"]+=1
    scans.append(dict(path="data/derived/gsm8k_questions.jsonl",**dict(gsm_counts),answer_values_decoded=0))
    canonical_positions=defaultdict(list)
    source_positions=defaultdict(dict)
    for item in official_index:
        canonical_positions[item["canonical_question_sha256"]].append(item)
        source_positions[item["source_file"]][int(item["source_index"])]=item
    duplicate_hashes={key for key,positions in canonical_positions.items() if len(positions)>1}
    duplicate_metadata=[dict(canonical_question_sha256=key,
        source_locations=[dict(source_file=item["source_file"],source_index=item["source_index"],question_id=item["question_id"]) for item in positions])
        for key,positions in canonical_positions.items() if key in duplicate_hashes]
    for source in source_manifest["official_bbh"]:
        counts=Counter()
        for index,raw in enumerate(official_example_objects(root/source["source_file"])):
            counts["physical_example_objects"]+=1
            item=source_positions[source["source_file"]][index]
            if item["protected_sealed"]:
                counts["protected_rows_skipped_before_input_decoding"]+=1
                continue
            if item["canonical_question_sha256"] in duplicate_hashes:
                counts["duplicate_official_input_positions_skipped"]+=1
                continue
            value=project(raw,{"input"})
            if set(value)!={"input"} or canonical_hash(value["input"])!=item["canonical_question_sha256"]:
                raise ValueError("External official position mapping and projected input differ")
            row={key:item[key] for key in ("question_id","task","subtask","source_split","source_file","source_index","source_frame","canonical_question_sha256")}
            row["source_revision"]="9ee07bd481feebf959a6b59d61ea57bdcf30964d"
            row["question"]=value["input"]
            if row["question_id"] in expected and row["canonical_question_sha256"]!=expected[row["question_id"]]:
                raise ValueError("Registered BBH canonical identity differs")
            row["registered_excluded"]=row["question_id"] in excluded_ids or row["canonical_question_sha256"] in excluded_hashes
            rows.append(row)
            counts["allowed_input_values_decoded"]+=1
        if counts["physical_example_objects"]!=source["examples"]:
            raise ValueError("Official source position count differs")
        scans.append(dict(path=source["source_file"],**dict(counts),answer_values_decoded=0))
    if len({row["question_id"] for row in rows})!=len(rows) or len({row["canonical_question_sha256"] for row in rows})!=len(rows):
        raise ValueError("Safe source pool contains repeated original IDs or canonical inputs")
    return rows,scans,duplicate_metadata


def build_proposal(root,output):
    started_cpu,started_wall=time.process_time(),time.perf_counter()
    public,private=output/"public",output/"private"
    if (public/"DATA_AUDIT.json").exists() or (private/"proposed_new2600_inputs_only.jsonl").exists():
        raise FileExistsError("Existing proposed cohort is immutable")
    inventory=read_json(public/"DATA_SOURCE_INVENTORY.json")
    source_manifest=read_json(private/"SOURCE_MANIFEST.json")
    exclusions=read_json(private/"REGISTERED_EXCLUSION_METADATA.json")["records"]
    for source in source_manifest["files"]:
        if sha(root/source["path"])!=source["sha256"]:
            raise ValueError("Source changed after inventory")
    protected_root=private/"protected_input_fingerprints"
    access=read_json(protected_root/"AUTHORIZED_ACCESS_RECEIPT.json")
    if access["status"]!="AUTHORIZED_ONCE_ONLY_INPUT_FINGERPRINT_AUDIT_COMPLETE" or access["unique_protected_questions"]!=200:
        raise ValueError("Protected one-time audit is incomplete")
    for name,digest in (("PROTECTED_INPUT_SIGNATURES.json",access["signature_sha256"]),
                        ("OFFICIAL_BBH_INPUT_IDENTITY_INDEX.json",access["bbh_position_index_sha256"])):
        if sha(protected_root/name)!=digest:
            raise ValueError("Protected fingerprint artifact changed")
    # Key material and signatures stay private and are never model inputs.
    key=(protected_root/"FINGERPRINT_KEY.bin").read_bytes()
    if len(key)!=32:
        raise ValueError("Invalid private fingerprint key")
    signatures=read_json(protected_root/"PROTECTED_INPUT_SIGNATURES.json")["records"]
    official_index=read_json(protected_root/"OFFICIAL_BBH_INPUT_IDENTITY_INDEX.json")["records"]
    write_json(public/"NEAR_DUPLICATE_RULES_FROZEN.json",dict(NEAR_RULES,
        protected_authorization="One user-authorized isolated input-only fingerprint pass; no plaintext retained",
        candidate_builder_sha256=sha(Path(__file__)),parser_source_sha256=sha(Path(__file__).with_name("audit_sources.py"))))
    rows,scans,official_duplicates=safe_source_rows(root,source_manifest,exclusions,official_index)
    print("E18_SAFE_SOURCE_INPUT_PROJECTION_COMPLETE",len(rows),flush=True)
    format_groups,pending,near_edges=_screen_near_duplicates(rows)
    print("E18_NONPROTECTED_INPUT_DUPLICATE_SCREEN_COMPLETE",len(near_edges),flush=True)
    format_removed=set()
    clusters=[]
    for group in format_groups:
        registered=[index for index in group if rows[index]["registered_excluded"]]
        representative=None if registered else min(group,key=lambda index:_rank("format-representative",rows[index]["canonical_question_sha256"]))
        format_removed.update(index for index in group if index!=representative)
        clusters.append(dict(question_ids=[rows[index]["question_id"] for index in group],
            representative=None if representative is None else rows[representative]["question_id"],registered_input_present=bool(registered)))
    protected_removed,protected_edges,protected_summary=protected_screen(rows,key,signatures)
    print("E18_PROTECTED_SIGNATURE_COMPARISON_COMPLETE",protected_summary["matching_critical_slot_pairs"],flush=True)
    eligible=[row for index,row in enumerate(rows) if not row["registered_excluded"] and index not in format_removed and index not in pending and index not in protected_removed]
    gsm=sorted((row for row in eligible if row["task"]=="gsm8k"),key=lambda row:_rank("gsm1300",row["canonical_question_sha256"]))
    bbh_by_subtask=defaultdict(list)
    for row in eligible:
        if row["task"]=="bbh":
            bbh_by_subtask[row["subtask"]].append(row)
    subtasks=sorted(bbh_by_subtask)
    if len(gsm)<1300 or len(subtasks)!=27:
        raise ValueError("Insufficient candidate task coverage after fixed screens")
    extra=set(sorted(subtasks,key=lambda name:_rank("bbh1300-extra-quota",name))[:4])
    quotas={name:49 if name in extra else 48 for name in subtasks}
    selected=gsm[:1300]
    for subtask in subtasks:
        group=sorted(bbh_by_subtask[subtask],key=lambda row:_rank("bbh1300",row["canonical_question_sha256"]))
        if len(group)<quotas[subtask]:
            raise ValueError("Insufficient BBH quota for fixed subtask "+subtask)
        selected.extend(group[:quotas[subtask]])
    proposed=[]
    for row in selected:
        copied={key:value for key,value in row.items() if key!="registered_excluded"}
        copied["status"]="PROPOSED_NOT_ADMITTED"
        copied["independence_qualification"]="PREDECLARED_INPUT_DUPLICATE_SCREEN_COMPLETE; SEMANTIC_INDEPENDENCE_NOT_PROVEN"
        copied["source_layer_admission"]="PROPOSED_SOURCE_FRAME_EXTENSION_NOT_ADMITTED" if copied["source_frame"]=="bbh_official_full_source_frame_expansion" else "PROPOSED_ORIGINAL_FRAME_NOT_ADMITTED"
        proposed.append(copied)
    excluded_ids={row["question_id"] for row in exclusions}
    excluded_hashes={row["canonical_question_sha256"] for row in exclusions}
    if len(proposed)!=2600 or Counter(row["task"] for row in proposed)!={"gsm8k":1300,"bbh":1300}:
        raise ValueError("Proposed2600 task quotas differ")
    if len({row["question_id"] for row in proposed})!=2600 or len({row["canonical_question_sha256"] for row in proposed})!=2600:
        raise ValueError("Proposed original question identities repeat")
    if any(row["question_id"] in excluded_ids or row["canonical_question_sha256"] in excluded_hashes for row in proposed):
        raise ValueError("Proposed candidate overlaps registered cohorts")
    if any({"reference","gold","reward","score","answer","target"}&set(row) for row in proposed):
        raise ValueError("Input-only proposal has forbidden fields")
    candidate_path=private/"proposed_new2600_inputs_only.jsonl"
    write_jsonl(candidate_path,proposed)
    source_extension=[row for row in proposed if row["source_frame"]=="bbh_official_full_source_frame_expansion"]
    original_frame=[row for row in proposed if row["source_frame"]!="bbh_official_full_source_frame_expansion"]
    write_jsonl(private/"PROPOSED_ORIGINAL_SOURCE_FRAME_INPUTS.jsonl",original_frame)
    write_jsonl(private/"PROPOSED_BBH_SOURCE_FRAME_EXTENSION_INPUTS.jsonl",source_extension)
    write_json(private/"PROPOSED_2600_CANDIDATE_MANIFEST.json",dict(
        status="PROPOSED_NOT_ADMITTED",target_questions=2600,admitted_questions=0,input_only_jsonl_sha256=sha(candidate_path),
        fixed_selection_seed=SEED,ordering="GSM1300 by fixed canonical rank, then BBH by sorted subtask and fixed canonical rank",
        selection_uses_model_predictions=False,selection_uses_reward_or_cost=False,
        bbh_subtask_quotas=quotas,records=[{key:value for key,value in row.items() if key!="question"} for row in proposed],
    ))
    write_json(private/"INPUT_DUPLICATE_AUDIT_DETAILS.json",dict(
        official_exact_duplicate_groups=official_duplicates,format_equivalence_clusters=clusters,
        unprotected_high_similarity_edges=near_edges,protected_signature_edges=protected_edges,
        protected_screen_summary=protected_summary,near_edges_not_transitively_merged=True,
        protected_signatures_not_used_for_training=True,
    ))
    input_schema=sorted(set().union(*(set(row) for row in proposed)))
    report=dict(
        schema="e18_input_only_data_audit_v2",status="PROPOSED_NOT_ADMITTED_INPUT_SOURCE_AUDIT_COMPLETE",
        prior_inventory_status=inventory["status"],prior_block_resolved_by="User-authorized once-only isolated200-input fingerprint pass",
        registered_cohorts=inventory["registered_cohort_counts"],registered_exclusion_questions=1708,
        primary_source_target=inventory["primary_source_target"],source_difference_from_sealed=inventory["sealed_source_target_difference"],
        official_source_counts=dict(gsm8k_train=7473,gsm8k_test=1319,bbh_test_examples=6511,bbh_subtasks=27,old_llmrouterbench_bbh_subpool=1078),
        original_frame_upper_bounds_before_near_screen=inventory["main_original_source_exact_metadata_upper_bounds"],
        safe_input_pool_questions=len(rows),registered_excluded_in_safe_pool=sum(row["registered_excluded"] for row in rows),
        official_input_duplicate_groups=len(official_duplicates),official_duplicate_positions=sum(len(group["source_locations"]) for group in official_duplicates),
        format_equivalence_clusters=len(clusters),format_duplicate_rows_excluded=len(format_removed),
        high_similarity_review_edges=len(near_edges),review_endpoint_questions=len(pending),
        protected_signature_screen=protected_summary,
        eligible_after_all_fixed_input_screens=dict(Counter(row["task"] for row in eligible)),
        safe_old_bbh_remaining_after_screens=sum(row["task"]=="bbh" and row["source_frame"]=="bbh_old_llmrouterbench_common_subpool" for row in eligible),
        proposed_questions=2600,proposed_task_counts=dict(Counter(row["task"] for row in proposed)),
        proposed_source_frame_counts=dict(Counter(row["source_frame"] for row in proposed)),
        proposed_bbh_subtask_counts=dict(sorted(Counter(row["subtask"] for row in proposed if row["task"]=="bbh").items())),
        predeclared_bbh_quotas=quotas,input_only_schema=input_schema,
        candidate_input_sha256=sha(candidate_path),candidate_metadata_sha256=sha(private/"PROPOSED_2600_CANDIDATE_MANIFEST.json"),
        original_frame_input_sha256=sha(private/"PROPOSED_ORIGINAL_SOURCE_FRAME_INPUTS.jsonl"),
        expanded_bbh_frame_input_sha256=sha(private/"PROPOSED_BBH_SOURCE_FRAME_EXTENSION_INPUTS.jsonl"),
        duplicate_rule_sha256=sha(public/"NEAR_DUPLICATE_RULES_FROZEN.json"),
        source_audit_code_sha256=sha(Path(__file__).with_name("audit_sources.py")),candidate_builder_code_sha256=sha(Path(__file__)),
        protected_authorized_access=dict(unique_protected_questions=access["unique_protected_questions"],
            actual_protected_input_values_decoded=access["actual_protected_input_value_occurrences_decoded"],
            actual_protected_input_values_by_task=access["actual_protected_input_values_by_task"],
            protected_plaintext_persisted=False,protected_embedding_forwards=0,protected_training_rows=0,
            answer_values_decoded=0,fresh_reward_values_read=0,additional_build_stage_protected_input_decodes=0,
            sealed_directory_opened=False,private_authorized_receipt_sha256=sha(protected_root/"AUTHORIZED_ACCESS_RECEIPT.json")),
        sampling_frame_warning="Official BBH full6511 expands beyond the old1078-question LLMBench source frame. Report existing-fresh/original-frame and official-coverage-extension strata separately.",
        pretraining_exposure_warning="Both official datasets are public. Input disjointness does not establish absence of language-model pretraining exposure.",
        near_duplicate_limitation="Fixed high-threshold syntactic and one-way fingerprint screens do not prove semantic independence or eliminate every paraphrase.",
        completeness_scope="Registered project1308 plus fresh400 are excluded. Newly disclosed unregistered manual-use cohorts require an appended exclusion registry before admission.",
        planned_total_with_existing_fresh400=3000,admitted_new_questions=0,new_llm_calls=0,new_downloads=0,new_encoder_forwards=0,
        confidence_intervals=0,bootstrap_runs=0,scans=scans,
        cpu_s=time.process_time()-started_cpu,wall_s=time.perf_counter()-started_wall,
        authorized_fingerprint_cpu_s=access["cpu_s"],authorized_fingerprint_wall_s=access["wall_s"],
    )
    write_json(public/"DATA_AUDIT.json",report)
    chinese=f"""# E18 数据来源与拟议2600题审计

## 研究问题与设置

计划由现有fresh400加新增2600组成3000题候选评价池。主来源沿现有fresh400，GSM使用官方train，BBH使用官方test。原sealed200中的100道GSM来自官方test，来源层不同，继续保留原角色。

官方GSM train有7473题，test有1319题。BBH官方27子任务共6511条，旧LLMRouterBench六模型共同子池只有1078题。原1308题注册及fresh400构成1708题排除全集。按身份扣除后，GSM主来源上限6719题，旧BBH子池上限224题。扩大BBH覆盖需要单列来源层。

## 实际执行与数值结果

已完成一次经用户明确授权的隔离输入指纹审计，实际处理保护题输入200题，GSM与BBH各100。仅保存私有HMAC签名与官方BBH行号映射，未展示或另存保护题原文，未读取答案，未编码保护题向量，未用于训练。隔离进程已退出，候选构建通过外部索引在解码前跳过这些保护题。

安全输入池共{len(rows)}题。官方重复输入簇{len(official_duplicates)}，格式等价簇{len(clusters)}，非保护文本近重复待审边{len(near_edges)}。与200保护题单向签名比较筛除{len(protected_removed)}个候选输入。各项比较保留数字、实体与运算符，不把共享模板的不同实例直接归并，局部相似关系没有取传递闭包。

已生成2600题输入清单，GSM1300与BBH1300。BBH27子任务按固定哈希规则分配为23个子任务48题、4个子任务49题。来源层计数为{json.dumps(report['proposed_source_frame_counts'],ensure_ascii=False)}。选择只使用来源、输入规范化及预固定哈希rank，未读取模型预测、费用或正确性。

## 结论与权限

清单状态保持PROPOSED_NOT_ADMITTED，新增执行数据集入组为0、LLM/API调用为0、下载为0。原来源层与官方BBH覆盖扩展层另有独立私有清单，可分别报告。当前完成的是预声明输入去重规则检查，不能据此证明全部语义改写不存在，也不能证明公开基准从未进入语言模型预训练。

## 优势与风险

优势是完整来源hash、原题ID与canonical去重可回放，保护题只经过一次隔离指纹步骤，采样不依赖收益。风险是BBH抽样框扩大、近重复规则的覆盖边界，以及未注册人工审阅题可能需要补充排除。规则保证不将明显改变数字、实体或运算符的题型实例机械合并为一个样本。

## 下一步

可对拟议2600输入进行冻结MiniLM前向与已冻结策略的输入盲动作/调用并集预算。生成与评分仍需按候选审批与预算执行规则推进。现有fresh400与官方BBH扩展来源层分别统计，sealed200保留最终角色。
"""
    with (public/"DATA_AUDIT_ZH.md").open("x",encoding="utf-8") as stream:
        stream.write(chinese)
    write_json(private/"DATA_AUDIT_COMPLETION.json",dict(status=report["status"],
        public_report_sha256=sha(public/"DATA_AUDIT.json"),chinese_report_sha256=sha(public/"DATA_AUDIT_ZH.md"),
        candidate_input_sha256=sha(candidate_path),candidate_metadata_sha256=report["candidate_metadata_sha256"],
        candidate_questions=2600,admitted_new_questions=0,protected_input_audit_receipt_sha256=sha(protected_root/"AUTHORIZED_ACCESS_RECEIPT.json")))
    print(json.dumps({key:report[key] for key in ("status","proposed_questions","proposed_task_counts","proposed_source_frame_counts","candidate_input_sha256","admitted_new_questions")},ensure_ascii=False))
