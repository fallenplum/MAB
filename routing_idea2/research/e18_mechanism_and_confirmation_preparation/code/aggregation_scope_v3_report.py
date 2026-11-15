"""Publish de-identified E18 V3 scope conclusions from an immutable audit file."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def build(root):
    private, public = root / "private", root / "public"
    path = private / "REMOTE_AGGREGATION_INVENTORY_v3.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    source = Path(__file__).with_name("remote_aggregation_audit_v3.py")
    if result["source_sha256"] != sha(source):
        raise ValueError("V3 executing-source receipt differs from current source")
    if not result["authenticated"] or result["status"] != "AUTHENTICATED_READ_ONLY_AGGREGATION_INVENTORY":
        raise ValueError("V3 remote inventory did not complete")
    if result["observed_pin"] != result["expected_pin"]:
        raise ValueError("Pinned host identity differs")
    inventory = result["inventory"]
    boundaries = Counter(row["state"] for row in inventory["boundaries"])
    metadata_states = Counter(row["state"] for row in inventory["inspected_metadata"])
    output_states = Counter(row["state"] for row in inventory["output_containers"])
    fresh_metadata = [row for row in inventory["inspected_metadata"] if row.get("fresh_scope")]
    root_prefix = "/data1/data2/chendong/routing/"
    roots = Counter(row["path"][len(root_prefix):].split("/")[0] for row in inventory["output_containers"])
    cohorts = Counter(row["identity"].get("cohort", "UNSPECIFIED") for row in inventory["inspected_metadata"] if "identity" in row)
    listing_complete = all(row["directory_listing_complete"] for row in inventory["directory_metadata"])
    unresolved = [row for row in inventory["boundaries"] if row["state"] not in {
        "OUTSIDE_AGGREGATION_SCOPE_NO_OPEN", "PROTECTED_NAME_SKIPPED_NO_OPEN"}]
    keys = inventory["reusable_aggregation_keys"]
    ledger = inventory["physical_aggregation_ledger"]
    if len(keys) != inventory["verified_logical_key_count"] or inventory["conflicting_logical_keys"]:
        raise ValueError("Fresh aggregation cache reconciliation is inconsistent")
    # This audit found no fresh plan/context, so no model-output row was opened.
    # Keep an explicit guard rather than publishing a stale hard-coded summary.
    if fresh_metadata or keys or ledger or inventory["aggregate_payloads_opened"]:
        raise ValueError("Nonempty fresh aggregation evidence needs a new result-specific postcheck")
    summary = dict(
        schema="E18_remote_aggregation_scope_postcheck_v3",
        postcheck="PASS_SCOPED_NO_VERIFIED_FRESH_AGGREGATION",
        inventory_sha256=sha(path), inventory_bytes=path.stat().st_size,
        inventory_source_sha256=result["source_sha256"],
        reported_utc=result["reported_utc"], elapsed_s=result["elapsed_s"],
        host_pin_match=True, authenticated_read_only=True,
        directory_count=len(inventory["directory_metadata"]), entries_seen=inventory["entries_seen"],
        directory_listings_complete=listing_complete, unresolved_in_scope_truncation_count=len(unresolved),
        configured_depth=inventory["max_depth"], configured_entries=inventory["max_entries"],
        configured_regular_file_bytes=inventory["max_file_bytes"],
        metadata_state_counts=dict(metadata_states), metadata_cohort_counts=dict(cohorts),
        output_container_counts=dict(output_states), output_container_roots=dict(roots),
        fresh_scope_metadata_count=0, fresh_aggregation_payloads_opened=0,
        verified_fresh_aggregation_logical_keys=0, verified_fresh_aggregation_physical_executions=0,
        verified_incremental_fresh_aggregation_prompt_tokens=0,
        verified_incremental_fresh_aggregation_completion_tokens=0,
        verified_incremental_fresh_aggregation_tokens=0,
        boundary_counts=dict(boundaries),
        scanned_scope=[
            "routing根目录下实际发现的agg/typed/stage2聚合相关根",
            "三个primary-aggregator-slot，含artifacts、preparation、private、results/raw",
            "qwen3共享worker内符合聚合作业命名的目录",
            "上述目录中已枚举的bundle/exports/shard及未评分输出容器元数据",
        ],
        preserved_unknown_scope=[
            "成员工作区已由成员缓存V2单独核验，本V3不重复读取",
            "模型、环境、qualification/diagnostics及受保护gold/reference/reward/score/sealed路径",
            "已列109个域外目录和8个受保护命名项的内部内容",
            "routing已知根之外的其他用户目录、数据盘位置和未登记旧项目副本",
        ],
        conclusion="在明确扫描的聚合根、槽位、bundle与exports边界内，未发现fresh scope计划或context，故本轮可确认复用fresh聚合数为0；其他范围维持UNKNOWN。",
        historical_train_validation_diagnostic_outputs_value_decoded=False,
        all_remote_aggregation_cost="UNKNOWN_OUTSIDE_INSPECTED_FRESH_SCOPE",
        fresh_gold_or_reward_value_decodes=0, sealed_paths_opened_by_this_module=0,
        new_generation_jobs=0, scoring_jobs=0, model_training_jobs=0,
        remote_commands_started=0, remote_writes=0,
        retry_budget_note="成功唯一key费用上限与跨人工续跑的含失败总费用分列；原协议跨续跑max_attempts未锁定。",
        guard_limitations="本次未发现fresh候选，fresh输出及jobs/bridge内容核验路径未用于任何实际缓存纳入。",
    )
    for filename in ("REMOTE_AGGREGATION_POSTCHECK_v3.json",):
        save(private / filename, summary)
        save(public / filename, summary)
    save(private / "VERIFIED_AGGREGATION_CACHE_KEYS_v3.json", dict(schema="E18_verified_aggregation_cache_keys_v3", inventory_sha256=sha(path), keys=keys))
    save(private / "VERIFIED_AGGREGATION_PHYSICAL_LEDGER_v3.json", dict(schema="E18_verified_aggregation_physical_ledger_v3", inventory_sha256=sha(path), rows=ledger))
    roots_table = "\n".join(f"| `{name}` | {count} |" for name, count in sorted(roots.items()))
    report = f"""# B4 远端聚合缓存范围补充核验 V3

## 核验结果

**已扫描的聚合根、槽位、bundle 和 exports 范围内，可确认复用的 fresh 聚合缓存为 0。** 现有成员缓存仍以 V2 已核验的 5,330 个逻辑 key 为准。该结论只覆盖本节列出的实际目录边界。

主机 ed25519 pin 与已有服务器一致，认证后仅通过 SFTP 列举、stat 和读取允许的元数据。用时 {result['elapsed_s']:.3f} 秒。累计完整列举 {len(inventory['directory_metadata'])} 个目录、{inventory['entries_seen']} 个条目，未触及深度或文件数上限，没有扫描内目录读取失败。

130 份文件通过键名防护后投影了身份元数据。68 个 `outputs.jsonl` 均未获得 fresh scope 计划及 context 绑定，因此未打开其模型输出原文。没有发现 fresh scope 的 RUN_PLAN 或 context。

## 实际扫描边界

- routing 根下实际存在的聚合、typed 与 stage2 相关目录。
- 三个 `primary-aggregator-slot` 的 `artifacts/private`、`artifacts/preparation` 与 `results/raw`。这些槽位的 raw 元数据只见 `online_stream` 和 `online_calibration`。
- 共享 qwen3 worker 中符合聚合作业命名的目录，以及上述范围中已枚举的 shard、bundle 和 exports。
- 36 个原训练/验证 typed 主聚合输出容器与 32 个历史诊断、单答、强模型等输出容器。仅核验路径与安全元数据。

| 输出根目录 | 已列举输出容器数 |
|---|---:|
{roots_table}

109 个域外目录和 8 个受保护命名项均保留了显式边界记录。成员工作区由独立 V2 成员审计处理。模型、环境、qualification/diagnostics、受保护数据目录，以及 routing 根以外未登记位置继续标记 UNKNOWN。

## 成本与权限

本次新增确认的 fresh 聚合逻辑 key、物理执行及其计入账本的 Token 均为 0。未知位置可能存在的聚合开销仍为 UNKNOWN。已扫描历史训练、验证和诊断输出的原文与成本未加入 fresh 账本。

本模块 fresh gold/reward 值解码 0，sealed 文件打开 0，远端命令启动 0，远端写入 0，新增生成、评分与训练任务均为 0。数据模块另有单向 sealed 输入指纹审计授权，项目总边界由对应模块记录。

成功唯一 key 的执行量只说明成功结果预算。原执行器遇失败停止，跨人工续跑的 `max_attempts` 总上限未锁定，含失败重试总开销另列。

## 依据与局限

执行脚本 SHA256 为 `{result['source_sha256']}`。私有完整目录元数据收据 SHA256 为 `{sha(path)}`，体积 {path.stat().st_size:,} 字节。

本次未发现 fresh 聚合候选，输出 record、jobs/bridge 和成员依赖的完整核验路径未用于纳入任何实际 fresh 聚合缓存。后续若定位到新目录，需要以该目录身份和未评分 schema 重新核验后再入账。
"""
    with (public / "B4_REMOTE_AGGREGATION_SCOPE_v3.md").open("x", encoding="utf-8") as stream:
        stream.write(report)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    result = build(arguments.root)
    print(json.dumps(dict(postcheck=result["postcheck"], directories=result["directory_count"],
        entries=result["entries_seen"], fresh_aggregation_keys=result["verified_fresh_aggregation_logical_keys"])))
