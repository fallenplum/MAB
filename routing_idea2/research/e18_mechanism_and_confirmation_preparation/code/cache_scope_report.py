"""Record the exact aggregate-cache search boundary and original retry contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cache_audit import read_metadata, sha256, write_json


def build(output_root):
    output_root = Path(output_root)
    private, public = output_root / "private", output_root / "public"
    postcheck = read_metadata(private / "REMOTE_CACHE_POSTCHECK_v2.json", [])
    inventory_path = private / "REMOTE_CACHE_INVENTORY_NEW_ENDPOINT_v2.json"
    if sha256(inventory_path) != postcheck["remote_inventory_sha256"]:
        raise ValueError("The inspected remote metadata changed")
    inventory = read_metadata(inventory_path, [])["inventory"]
    retry_path = private / "ORIGINAL_RETRY_BOUNDS_AUDIT.json"
    retry = read_metadata(retry_path, [])
    project_listings = [row for row in inventory["directory_metadata"] if "project_root" in row]
    member_listings = [row for row in inventory["directory_metadata"] if "workspace" in row]
    aggregate_named_entries = []
    for listing in project_listings:
        for row in listing.get("entries", []):
            name = row["name"]
            if any(term in name for term in ("typed", "aggregat", "aggcal")):
                aggregate_named_entries.append(dict(parent=listing["project_root"], name=name,
                    kind=row["kind"], observed="Parent-entry name/stat only",
                    child_directory_listed=False, result_payload_opened=False))
    result = dict(schema="E18_aggregate_cache_scope_and_retry_contract_v2", status="RECORDED_SCOPE",
        current_remote_inventory="AUTHENTICATED_PINNED_READ_ONLY",
        host_hash=postcheck["host_hash"], remote_postcheck_sha256=sha256(private / "REMOTE_CACHE_POSTCHECK_v2.json"),
        project_root_listings=[dict(path=row["project_root"], entries=len(row.get("entries", [])),
                                   content_scope="Directory names and lstat metadata only") for row in project_listings],
        member_workspaces=[dict(path=row["workspace"], state=row.get("state", "LISTED")) for row in member_listings],
        aggregate_related_parent_entries=aggregate_named_entries,
        verified_reusable_fresh_aggregate_outputs=0, aggregate_result_payloads_opened=0,
        meaning_of_zero="No fresh400 unscored aggregate output has passed identity verification in this audit",
        global_fresh_aggregate_cache_state="UNKNOWN",
        not_inspected=["Contents beneath primary-aggregator-slot0/1/2",
            "Result files beneath typed aggregation directories",
            "Unscored aggregate exports/archives beyond the returned parent directory metadata",
            "Other MAB project copies and other data-root locations",
            "Member export archives not represented by the inspected workspace JSONL"],
        original_retry_contract=retry, retry_proof_sha256=sha256(retry_path),
        cost_interpretation="A successful unique-output cap does not bound later manually authorized recovery attempts",
        full_process_attempt_count_cap=None, full_process_retry_inclusive_token_cap=None,
        automatic_paid_retry_count=0,
        module_fresh_gold_or_reward_reads=0, module_sealed_paths_opened=0,
        new_generation_or_scoring_jobs=0, source_sha256=sha256(__file__))
    write_json(private / "AGGREGATION_SCOPE_AND_RETRY_v2.json", result)
    write_json(public / "AGGREGATION_SCOPE_AND_RETRY_v2.json", result)
    lines = ["# 聚合缓存查验范围与失败重试预算", "", "## 新端点已检查的范围", "",
        "新端点已通过原主机pin并完成只读认证。本次实际扫描的两个项目目录如下，目录内只列名称、文件类型、大小和mtime。", ""]
    for row in result["project_root_listings"]:
        lines.append(f"- `{row['path']}`，{row['entries']}个父级目录项。")
    lines.extend(["", "另检查六成员原native及DeepSeek两个明确variant工作区的成员JSONL。主机地址只保存在私有握手收据，公开结果用host散列标识。", "",
        "## 聚合结果的身份结论", "",
        "本次已核验可复用fresh400未评分聚合输出为0。这个0表示尚未取得身份核验通过的聚合输出，全集是否存在其他缓存保持UNKNOWN。当前没有打开聚合结果载荷。",
        "", "父目录metadata中已经看到typed train/validation、旧诊断及strong/heterogeneous聚合目录，以及primary-aggregator-slot0/1/2三个目录项。该观察只覆盖名称与stat，并未进入这些目录读取聚合文件。其它MAB副本、其它数据根、尚未展开的export档案也保持UNKNOWN。",
        "", "## 原失败重试约束", "",
        "原native成员worker与正式stage2聚合worker均无自动付费重试。首次失败会停止并要求核对，unknown-cost/PENDING journal和共享成本fence阻止未经核对的再执行。",
        "", "member的max-new-jobs仅限制一次invocation内的新成功调用数，默认未设数值。底层Store没有累计max_attempts参数。36份正式聚合RUN_PLAN没有max_attempts/max_retries/retry_limit/attempt_cap/max_calls字段。跨人工恢复及重新授权后的总attempt数值上限尚未锁定。",
        "", "因此，完成全部唯一成功key所需的调用数与Token上限，应明确标为one-success资源口径。包括后续人工批准重试的全流程hard cap在原协议中缺少数值约束，当前记为null。未知失败成本需要保留并核对，不能因自动重试为0而填成0。",
        "", "## 优势与限制", "",
        "优势是区分已观察目录、已核验输出和未覆盖位置，并将成功唯一输出预算与失败恢复预算分别记录。限制是本次没有完整扫描所有聚合payload或导出档案。缓存存在性仍有未完成核查范围，当前结论限于已记录的只读观察。模块内fresh标准答案、真实评分、sealed文件读取及新生成任务均为0。"])
    with (public / "B4_AGGREGATION_SCOPE_AND_RETRY_v2.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
        default=Path(r"E:\语言模型组合路由\research\e18_mechanism_and_confirmation_preparation"))
    arguments = parser.parse_args()
    result = build(arguments.output_root)
    print(json.dumps(dict(status=result["status"], verified_fresh_aggregate_outputs=0,
                          full_process_attempt_count_cap=None)))
