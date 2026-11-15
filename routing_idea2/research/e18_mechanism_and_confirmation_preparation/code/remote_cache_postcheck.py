"""Reconcile returned unscored SSH cache metadata against frozen local identities."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from cache_audit import _fresh_identities, read_metadata, sha256, write_json


def key(row):
    return row["canonical_id"], row["member_index"], row["generation_seed"]


def audit(asset_root, output_root):
    asset_root, output_root = Path(asset_root), Path(output_root)
    private, public = output_root / "private", output_root / "public"
    access = []
    original_audit = read_metadata(private / "CACHE_AUDIT.json", access)
    local_path = private / "VERIFIED_MEMBER_CACHE_KEYS.json"
    if sha256(local_path) != original_audit["private_cache_keys_sha256"]:
        raise ValueError("Previously verified local cache changed")
    local = read_metadata(local_path, access)["keys"]
    local_cost = read_metadata(private / "MEMBER_PHYSICAL_LEDGER.json", access)["rows"]
    source_path = private / "REMOTE_CACHE_INVENTORY_NEW_ENDPOINT_v2.json"
    source = read_metadata(source_path, access)
    if not source["authenticated"] or source["status"] != "AUTHENTICATED_READ_ONLY_INVENTORY":
        raise ValueError("Successful pinned read-only inventory is required")
    if source["observed_pin"] != source["expected_pin"]:
        raise ValueError("Actual remote host pin differs from the trusted pin")
    inventory = source["inventory"]
    if inventory["conflicting_logical_keys"] or inventory["physical_attempt_conflicts"]:
        raise ValueError("Returned inventory contains unresolved identity conflicts")
    identities = _fresh_identities(asset_root / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/fresh_dev_test_inputs.npz")
    by_local = {key(row): row for row in local}
    by_cost = {key(row): row for row in local_cost}
    model_fields = ("model_fingerprint", "model_id", "model_revision", "member_prompt_version", "member_mode")
    expected_models = {}
    question_sources = {}
    for row in local:
        expected = tuple(row[field] for field in model_fields)
        member = row["member_index"]
        if member in expected_models and expected_models[member] != expected:
            raise ValueError("Local model identity is inconsistent")
        expected_models[member] = expected
        question = row["question_id"]
        if question in question_sources and question_sources[question] != row["source_question_sha256"]:
            raise ValueError("Local raw-question source is inconsistent")
        question_sources[question] = row["source_question_sha256"]
    valid, rejected = [], []
    for row in inventory["unique_member_cache_keys"]:
        reason = None
        question = row["question_id"]
        if identities.get(question) != row["canonical_id"]:
            reason = "canonical_identity"
        elif question_sources.get(question) != row["source_question_sha256"]:
            reason = "raw_question_source_hash"
        elif tuple(row[field] for field in model_fields) != expected_models.get(row["member_index"]):
            reason = "frozen_model_or_prompt_identity"
        elif row["generation_seed"] not in (42, 137, 2024) or row["sample_index"] != 0 or row["sampling_control"] is not False:
            reason = "generation_contract"
        elif not math.isfinite(float(row["source_model_wall_s"])) or row["source_model_wall_s"] < 0:
            reason = "unknown_source_wall"
        elif row["tokens"] != row["prompt_tokens"] + row["completion_tokens"]:
            reason = "token_ledger_arithmetic"
        if reason:
            rejected.append(dict(reason=reason, record=row))
        else:
            valid.append(row)
    remote = {key(row): row for row in valid}
    if len(remote) != len(valid):
        raise ValueError("Duplicate remote logical outputs after validation")
    overlap = set(remote) & set(by_local)
    for identity in overlap:
        old, new = by_local[identity], remote[identity]
        for field in ("record_sha256", "answer_sha256", "job_id", "physical_attempt_id", "source_question_sha256"):
            if old[field] != new[field]:
                raise ValueError("Remote output conflicts with frozen local canonical output")
        for field in ("prompt_tokens", "completion_tokens", "tokens"):
            if by_cost[identity][field] != new[field]:
                raise ValueError("Remote actual resource count differs from verified local count")
        if not math.isclose(by_cost[identity]["source_model_wall_s"], new["source_model_wall_s"], rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError("Remote source-call duration differs from the verified local duration")
    union_keys = {identity: dict(row, storage="LOCAL_CANONICAL") for identity, row in by_local.items()}
    union_costs = {identity: dict(row, storage="LOCAL_CANONICAL") for identity, row in by_cost.items()}
    for identity, row in remote.items():
        if identity not in union_keys:
            union_keys[identity] = dict(row, storage="REMOTE_VERIFIED_NOT_LOCALLY_MIRRORED")
            union_costs[identity] = dict(row, model_calls=1, storage="REMOTE_VERIFIED_NOT_LOCALLY_MIRRORED")
    attempt_ids = [row["physical_attempt_id"] for row in union_costs.values()]
    if len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("A physical execution would be charged more than once")
    rejected_count = Counter(row["reason"] for row in rejected)
    coverage = Counter((row["member_index"], row["generation_seed"]) for row in union_keys.values())
    coverage_rows = [dict(member_index=member, generation_seed=seed,
        verified_outputs=coverage[member, seed], remaining_unverified=400-coverage[member, seed],
        scope="Inspected original native/variant workspaces plus frozen local imports")
        for member in range(6) for seed in (42, 137, 2024)]
    totals = {field: sum(row[field] for row in union_costs.values())
              for field in ("prompt_tokens", "completion_tokens", "tokens", "source_model_wall_s")}
    existing_keys = private / "RECONCILED_MEMBER_CACHE_KEYS_v2.json"
    physical_path = private / "RECONCILED_MEMBER_PHYSICAL_LEDGER_v2.json"
    write_json(existing_keys, dict(schema="E18_reconciled_local_remote_member_keys_v2",
        primary_key_fields=["canonical_id", "member_index", "generation_seed"],
        keys=list(union_keys.values()), new_remote_raw_text_copied=False,
        transport_note="New remote raw files require exact-SHA local mirroring before a local aggregation runner can consume them"))
    write_json(physical_path, dict(schema="E18_unique_verified_local_remote_physical_costs_v2",
        rows=list(union_costs.values()), physical_deduplication="physical_attempt_id"))
    write_json(private / "REMOTE_CACHE_POSTCHECK_REJECTIONS_v2.json", dict(rejected=rejected))
    result = dict(schema="E18_remote_cache_postcheck_v2", status="PASS" if not rejected else "PARTIAL_REJECTED",
        remote_inventory_sha256=sha256(source_path),
        remote_inventory_source_sha256=sha256(Path(__file__).with_name("remote_cache_audit_v2.py")),
        host_hash=source["host_hash"], authenticated=True, host_pin_verified=True,
        originally_local_outputs=len(local), remote_record_validated_outputs=len(valid),
        overlapping_identical_local_outputs=len(overlap), newly_available_remote_outputs=len(set(remote)-set(by_local)),
        verified_logical_union=len(union_keys), unique_physical_executions=len(attempt_ids),
        verified_actual_spent=totals, source_question_hash_checks=len(valid), model_prompt_generation_checks=len(valid),
        rejected_counts=dict(rejected_count), coverage=coverage_rows,
        full6x3x400_remaining_unverified=sum(row["remaining_unverified"] for row in coverage_rows),
        remaining_actual_token_cost=None, actual_unverified_aggregate_cost=None,
        aggregate_payloads_opened=inventory["aggregate_payloads_opened"],
        aggregate_scope="Only project-root directory metadata inspected; aggregate result payload locations remain UNKNOWN",
        other_project_locations="UNKNOWN; no exhaustive whole-server scan claimed",
        fresh_gold_or_reward_reads=0, module_sealed_paths_opened=0, new_generation_jobs=0,
        remote_writes=0, new_local_models=0,
        local_promotion_scope="Record/identity validation passed; remote answers were hashed but not copied to canonical local raw stores",
        reconciled_cache_keys_sha256=sha256(existing_keys), reconciled_physical_ledger_sha256=sha256(physical_path),
        source_sha256=sha256(__file__))
    write_json(private / "REMOTE_CACHE_POSTCHECK_v2.json", result)
    write_json(public / "REMOTE_CACHE_POSTCHECK_v2.json", result)
    lines = ["# E18 新端点远端缓存更新", "", "## 最新连接与范围", "",
        "用户提供的新端点已通过原ed25519 pin核验并成功认证。远端操作限定原MAB项目的SFTP目录、文件metadata及无评分成员输出读取。旧端点的timeout记录独立保留。公开结果仅保存host散列。", "",
        "## 已核验覆盖", "",
        f"原本地{len(local)}个成员输出与远端记录逐项相同。新增可用远端输出{len(set(remote)-set(by_local))}个，logical并集{len(union_keys)}个，物理attempt去重后{len(attempt_ids)}个。canonical、原题文本SHA、模型指纹、seed和prompt协议均通过后检。",
        "", "| 成员 | seed | 已验证输出 | 本次参考全集尚未验证 |", "|---|---:|---:|---:|"]
    for row in coverage_rows:
        lines.append(f"| {row['member_index']} | {row['generation_seed']} | {row['verified_outputs']} | {row['remaining_unverified']} |")
    lines.extend(["", "## 实际费用及存储状态", "",
        f"已验证物理输出的输入Token为{totals['prompt_tokens']}，输出为{totals['completion_tokens']}，合计{totals['tokens']}。这是历史已花费执行的去重并集，本次没有新增LLM调用。",
        "", "新增远端结果已保留身份与资源记录，原始答案没有复制到本地canonical raw stores。后续本地聚合器使用前需要按已固定文件SHA完成只读搬运。搬运可以复用这些输出，不需要重新调用成员。",
        "", "## 覆盖边界、优势与风险", "",
        "优势是新端点主机身份连续、旧本地结果逐条比对、重复物理执行不重复计费。风险是当前只扫原native及明确variant工作区。其它项目位置保持UNKNOWN，已验证数0不代表全服务器没有输出。聚合器仅取得项目根目录metadata，聚合结果载荷未打开，其缓存及实际费用保持UNKNOWN。",
        "", "六成员×三生成×400题是覆盖参考全集，后续真实所需调用由四套冻结策略的执行并集决定。未验证结果的实际Token不填0。本模块fresh标准答案、真实评分和sealed读取均为0，其他模块另行授权的输入指纹审计单独报告。"])
    with (public / "B4_REMOTE_CACHE_UPDATE_v2.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=Path(r"E:\语言模型组合路由"))
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args()
    output = arguments.output_root or arguments.asset_root / "research/e18_mechanism_and_confirmation_preparation"
    result = audit(arguments.asset_root, output)
    print(json.dumps(dict(status=result["status"], verified_logical_union=result["verified_logical_union"],
        newly_available_remote_outputs=result["newly_available_remote_outputs"],
        remaining_unverified=result["full6x3x400_remaining_unverified"], costs=result["verified_actual_spent"])))
