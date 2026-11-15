"""Independently verify local private mirrors before exposing E18 cache keys."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from materialize_remote_members_v4 import IDENTITY_FIELDS
from remote_cache_audit_v2 import _member_rows


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def logical(row):
    return row["canonical_id"], row["member_index"], row["generation_seed"]


def build(root):
    private, public = root / "private", root / "public"
    source_path = private / "RECONCILED_MEMBER_CACHE_KEYS_v2.json"
    receipt_path = private / "MEMBER_CACHE_MATERIALIZATION_v4.json"
    candidate_path = private / "LOCAL_MIRRORED_MEMBER_KEYS_v4.json"
    receipt = read(receipt_path)
    candidate = read(candidate_path)
    old_keys = read(source_path)["keys"]
    ledger_path = private / "RECONCILED_MEMBER_PHYSICAL_LEDGER_v2.json"
    ledger_rows = read(ledger_path)["rows"]
    candidates = candidate["keys"]
    expected = {row["member_index"]: row for row in old_keys}
    identities = {row["question_id"]: row["canonical_id"] for row in old_keys}
    old_by_logical = {logical(row): row for row in old_keys}
    state = receipt.get("materialization", {})
    if not receipt.get("authenticated") or receipt["status"] != "AUTHENTICATED_EXISTING_MEMBER_CACHE_MATERIALIZATION":
        raise ValueError("Private mirror transport did not authenticate and complete")
    if receipt["observed_pin"] != receipt["expected_pin"]:
        raise ValueError("Mirror transport host pin changed")
    if candidate["materialization_receipt_sha256"] != sha(receipt_path.read_bytes()):
        raise ValueError("Private mirror candidate does not bind its transport receipt")
    if state["source_keys_sha256"] != sha(source_path.read_bytes()):
        raise ValueError("Mirror scope does not bind the frozen V2 keys")
    if len(candidates) != state["materialized_key_count"]:
        raise ValueError("Private mirror key count differs from receipt")
    files = {}
    verified_by_logical = {}
    for row in candidates:
        original = old_by_logical.get(logical(row))
        if original is None or original["storage"] != "REMOTE_VERIFIED_NOT_LOCALLY_MIRRORED":
            raise ValueError("Materialization added an unaudited key")
        if any(row[field] != original[field] for field in IDENTITY_FIELDS):
            raise ValueError("Materialization changed original key identity or physical cost")
        path = Path(row["local_source_path"])
        if path.parent.resolve() != (private / "verified_remote_cache_mirror").resolve():
            raise ValueError("Mirror key points outside the dedicated private directory")
        if path not in files:
            raw = path.read_bytes()
            if sha(raw) != original["source_sha256"] or sha(raw) != row["local_source_sha256"]:
                raise ValueError("Local mirror differs from the original remote file SHA")
            observed, status = _member_rows(raw, expected[row["member_index"]], identities,
                row["member_index"], row["generation_seed"])
            if observed is None:
                raise ValueError("Local mirror schema/record check failed")
            files[path] = dict(bytes=len(raw), sha256=sha(raw), rows={logical(item): item for item in observed})
        observed = files[path]["rows"].get(logical(row))
        if observed is None or any(observed[field] != row[field] for field in IDENTITY_FIELDS):
            raise ValueError("Local mirror payload does not match a materialized key")
        if logical(row) in verified_by_logical:
            raise ValueError("Duplicate materialized logical key")
        verified_by_logical[logical(row)] = row
    combined = [verified_by_logical.get(logical(row), row) for row in old_keys]
    if len({logical(row) for row in combined}) != len(old_keys):
        raise ValueError("Combined cache lost unique logical identities")
    if len({row["physical_attempt_id"] for row in ledger_rows}) != len(ledger_rows):
        raise ValueError("Existing physical cost ledger contains duplicate executions")
    if {row["physical_attempt_id"] for row in combined} != {row["physical_attempt_id"] for row in ledger_rows}:
        raise ValueError("Combined logical cache and physical cost ledger have different execution coverage")
    complete = len(candidates) == 2130 and len(files) == 8 and not state["source_changed_files"]
    postcheck = dict(schema="E18_existing_member_cache_materialization_postcheck_v4",
        postcheck="PASS_ALL_EXPECTED_PRIVATE_MIRRORS" if complete else "PASS_PARTIAL_PRIVATE_MIRRORS_WITH_EXPLICIT_MISSING_SOURCES",
        materialization_receipt_sha256=sha(receipt_path.read_bytes()), source_keys_sha256=sha(source_path.read_bytes()),
        candidate_keys_sha256=sha(candidate_path.read_bytes()), host_pin_match=True,
        historical_physical_cost_ledger_sha256=sha(ledger_path.read_bytes()),
        total_previously_verified_member_keys=len(old_keys), original_local_canonical_keys=3200,
        expected_new_remote_keys=2130, mirrored_new_keys=len(candidates), mirrored_source_files=len(files),
        mirrored_bytes=sum(row["bytes"] for row in files.values()),
        source_changed_files=state["source_changed_files"], all_expected_keys_materialized=complete,
        storage_counts=dict(Counter(row["storage"] for row in combined)),
        mirrored_existing_prompt_tokens=sum(row["prompt_tokens"] for row in candidates),
        mirrored_existing_completion_tokens=sum(row["completion_tokens"] for row in candidates),
        mirrored_existing_total_tokens=sum(row["tokens"] for row in candidates),
        combined_existing_prompt_tokens=sum(row["prompt_tokens"] for row in ledger_rows),
        combined_existing_completion_tokens=sum(row["completion_tokens"] for row in ledger_rows),
        combined_existing_total_tokens=sum(row["tokens"] for row in ledger_rows),
        postcheck_engineering_repair="首次后检在任何写入前因旧local key无费用字段停止；改从既有独立物理费用账本汇总并核验执行ID集合，镜像字节与执行任务均未改变。",
        new_model_inference_tokens=0, original_local_member_cache_files_modified=0,
        fresh_gold_or_reward_value_decodes=0, sealed_paths_opened_by_this_module=0,
        new_generation_jobs=0, scoring_jobs=0, remote_commands_started=0, remote_writes=0,
        answer_payloads_publicly_exported=0,
        transport_note="只同步既有成员生成字节；历史执行Token保持原账本身份，本次SFTP没有新增模型推理Token。",
    )
    save(private / "MEMBER_CACHE_MATERIALIZATION_POSTCHECK_v4.json", postcheck)
    save(public / "MEMBER_CACHE_MATERIALIZATION_POSTCHECK_v4.json", postcheck)
    save(private / "RECONCILED_MEMBER_CACHE_KEYS_LOCAL_v4.json", dict(
        schema="E18_reconciled_member_cache_keys_local_v4", materialization_postcheck_sha256=sha(
            (private / "MEMBER_CACHE_MATERIALIZATION_POSTCHECK_v4.json").read_bytes()), keys=combined))
    report = f"""# B4 已有远端成员缓存本地物化 V4

## 完成状态

**本次将 {len(candidates):,} 个已通过 V2 身份核验的远端成员结果，保存为 E18 私有目录内的 {len(files)} 份完整字节镜像。** 逐文件 SHA256、未评分 schema、record SHA、题目身份、生成 seed、模型与 prompt、答案 SHA 和实测资源均再次匹配。

原有 3,200 个本地 canonical 缓存保持原路径。合并索引共 {len(combined):,} 个逻辑 key，其中本次已物化 {len(candidates):,} 个，尚未本地物化 {2130-len(candidates):,} 个已核验远端 key。镜像合计 {postcheck['mirrored_bytes']:,} 字节。来源 SHA 变化文件数为 {state['source_changed_files']}，变化来源没有按旧审计值直接入库。

模型输出字节和逐题索引仅存放在私有目录，公开交付只保留汇总。V2/V3 历史审计和原 8 份本地输出均保留。

## 成本与使用边界

本次镜像对应已发生的成员生成 Token 为 {postcheck['mirrored_existing_total_tokens']:,}，包含输入 {postcheck['mirrored_existing_prompt_tokens']:,} 和输出 {postcheck['mirrored_existing_completion_tokens']:,}。合并全部已核验成员结果对应历史 Token 为 {postcheck['combined_existing_total_tokens']:,}。这些执行费用继续按原物理执行身份入账。

本次 SFTP 同步新增模型推理 Token 为 0，新增生成/评分任务为 0，远端命令和远端写入为 0。本模块 fresh gold/reward 值解码为 0，sealed 文件打开为 0。聚合缓存范围结论仍以 V3 为准，在已扫描边界内可确认复用的 fresh 聚合为 0。

私有镜像不直接构成完整 fresh 评价结果。冻结策略所需输出的实际并集、尚缺成员和聚合调用由主流程的预算文件记录。
"""
    with (public / "B4_LOCAL_CACHE_MATERIALIZATION_v4.md").open("x", encoding="utf-8") as stream:
        stream.write(report)
    return postcheck


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root)
    print(json.dumps(dict(postcheck=result["postcheck"], materialized_keys=result["mirrored_new_keys"],
        materialized_files=result["mirrored_source_files"], materialized_bytes=result["mirrored_bytes"])))
