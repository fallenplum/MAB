"""Read-only identity and actual-cost audit of eight existing fresh member files.

JSON property names are inspected before any value is decoded. Containers with
reference, gold, correctness or reward payload fields are rejected. Model
answers are hashed for identity, while private execution keys and public cost
summaries are written separately. No scorer or inference backend is imported.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np


FORBIDDEN_KEYS = frozenset({
    "y", "gold", "reference", "references", "reference_answer", "reference_answers",
    "correct_answer", "ground_truth", "groundtruth", "target", "targets", "label",
    "labels", "reward", "rewards", "score", "scores", "is_correct", "correct",
    "aggregate_scores", "member_scores", "password", "passwd", "private_key", "api_key",
})
JOB_FIELDS = frozenset({
    "job_id", "question_id", "task", "subtask", "split", "question", "model_alias",
    "sample_index", "seed", "model_fingerprint", "model_id", "model_revision",
    "member_mode", "prompt_version", "source_question_sha256", "messages", "sampling_control",
})
ROW_FIELDS = frozenset({
    "job_id", "question_id", "task", "subtask", "split", "model_alias", "sample_index",
    "seed", "model_fingerprint", "member_mode", "sampling_control", "prompt_version",
    "source_question_sha256", "started_at_unix", "attempt_id", "status", "cost_status",
    "result", "scoring_isolated_outside_generation_runner", "record_sha256",
})
RESULT_FIELDS = frozenset({
    "text", "prompt_tokens", "completion_tokens", "wall_time_s", "model_id", "model_revision",
    "seed", "backend", "peak_cuda_bytes", "raw_usage", "model_identity_verified",
    "completion_reached_cap", "finish_reason",
})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    # The original ExecutionStore uses these exact JSON serialization settings.
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def json_property_names(raw):
    """Inspect string keys without decoding any JSON value, including answers."""
    keys = set()
    position = 0
    while position < len(raw):
        if raw[position] != 34:
            position += 1
            continue
        start = position
        position += 1
        while position < len(raw):
            if raw[position] == 92:
                position += 2
                continue
            if raw[position] == 34:
                position += 1
                break
            position += 1
        else:
            raise ValueError("Unterminated JSON string during schema-only inspection")
        end = position
        while position < len(raw) and raw[position] in b" \r\n\t":
            position += 1
        if position < len(raw) and raw[position] == 58:
            keys.add(json.loads(raw[start:end]))
    return keys


def _checked_bytes(path, access):
    path = Path(path)
    if any("sealed" in part.casefold() for part in path.parts):
        raise ValueError("Sealed paths are outside this audit")
    raw = path.read_bytes()
    keys = json_property_names(raw)
    blocked = sorted(key for key in keys if key.casefold() in FORBIDDEN_KEYS)
    access.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                       schema_keys=sorted(keys), blocked_payload_keys=blocked))
    if blocked:
        raise ValueError("Protected payload key found before value decoding in " + path.name)
    return raw


def read_metadata(path, access):
    return json.loads(_checked_bytes(path, access))


def read_unscored_rows(path, access):
    raw = _checked_bytes(path, access)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Actual token counts must be nonnegative integers")
    return value


def _version_from_source(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return node.value.value
    raise ValueError("Pinned source version constant was not found")


def _fresh_identities(path):
    allowed = {"question_ids.npy", "canonical_hashes.npy", "task_labels.npy",
               "subtask_labels.npy", "embeddings.npy"}
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != allowed:
            raise ValueError("Unexpected member in the input-only fresh archive")
    with np.load(path, allow_pickle=False) as archive:
        question_ids = archive["question_ids"].copy()
        canonical = archive["canonical_hashes"].copy()
    if len(question_ids) != 400 or len(set(question_ids)) != 400 or len(set(canonical)) != 400:
        raise ValueError("Fresh input identities must contain400 unique original questions")
    return dict(zip(question_ids.tolist(), canonical.tolist()))


def _verify_frozen_preparation(folder, access):
    receipt = read_metadata(folder / "CONFIRMATION_PREPARATION_FROZEN.json", access)
    verified = []
    for row in receipt["private_artifacts"]:
        path = Path(row["path"])
        if "sealed" in str(path).casefold() or sha256(path) != row["sha256"]:
            raise ValueError("Original E17 preparation artifact changed")
        if path.stat().st_size != row["bytes"]:
            raise ValueError("Original E17 preparation artifact size changed")
        verified.append(dict(name=path.name, sha256=row["sha256"], bytes=row["bytes"]))
    with zipfile.ZipFile(folder / "FULL600_MODELS.npz") as archive:
        model_schema = archive.namelist()
    return dict(status=receipt["status"], files=verified, model_archive_members=model_schema,
                model_payload_decoded=False)


def audit(asset_root, output_root):
    asset_root, output_root = Path(asset_root), Path(output_root)
    private, public = output_root / "private", output_root / "public"
    before_cpu, before_wall = time.process_time(), time.perf_counter()
    accesses = []
    e17 = asset_root / "research/e17_supervision_distortion_audit/private/confirmation_input_only"
    frozen = _verify_frozen_preparation(e17, accesses)
    original_manifest = read_metadata(e17 / "INPUT_AND_CACHE_MANIFEST.json", accesses)
    fresh_path = asset_root / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/fresh_dev_test_inputs.npz"
    identities = _fresh_identities(fresh_path)
    declared_input = next(row for row in original_manifest["files"] if row["path"] == str(fresh_path))
    if sha256(fresh_path) != declared_input["sha256"]:
        raise ValueError("Fresh input-only identity archive changed")
    pool_path = asset_root / "configs/pool_v1.json"
    pool = read_metadata(pool_path, accesses)
    models = {int(row["index"]): row for row in pool["models"]}
    member_source = asset_root / "src/budget_router/member_execution.py"
    parser_source = asset_root / "src/budget_router_v2/scoring.py"
    member_prompt = _version_from_source(member_source, "MEMBER_PROMPT_VERSION")
    parser_version = _version_from_source(parser_source, "SCORER_VERSION")
    cache_keys, ledger, summaries = [], [], []
    attempts_seen, logical_seen = set(), set()
    aggregator_fingerprints = set()
    for item in original_manifest["cache_importer_receipts"]:
        receipt_path = Path(item["receipt"]["path"])
        if sha256(receipt_path) != item["receipt"]["sha256"]:
            raise ValueError("Original importer receipt changed")
        receipt = read_metadata(receipt_path, accesses)
        output_path = Path(item["canonical_output_path"])
        if sha256(output_path) != item["canonical_output_declared_sha256"]:
            raise ValueError("Canonical member output changed")
        if receipt["canonical_output"]["sha256"] != item["canonical_output_declared_sha256"]:
            raise ValueError("Importer and E17 disagree about the member payload")
        plan_path = receipt_path.with_name("EXECUTION_PLAN.json")
        if sha256(plan_path) != receipt["plan_sha256"] or receipt["plan_sha256"] != item["source_plan_sha256"]:
            raise ValueError("Generation plan binding changed")
        if sha256(receipt_path.with_name("worker.py")) != receipt["assignment_claim"]["worker_sha256"]:
            raise ValueError("Original physical worker changed")
        jobs = read_unscored_rows(receipt_path.with_name("jobs.jsonl"), accesses)
        if any(set(job) != JOB_FIELDS for job in jobs):
            raise ValueError("Fresh member jobs do not have the strict input-only schema")
        jobs_by_id = {job["job_id"]: job for job in jobs}
        if len(jobs_by_id) != 400:
            raise ValueError("A complete imported phase must contain400 unique jobs")
        outputs = read_unscored_rows(output_path, accesses)
        member, seed = int(item["member_index"]), int(item["generation_seed"])
        model = models[member]
        phase_attempts, phase_questions = set(), set()
        phase_prompt = phase_completion = 0
        phase_wall = 0.
        for row in outputs:
            if set(row) != ROW_FIELDS or set(row["result"]) != RESULT_FIELDS:
                raise ValueError("Unknown member-result schema")
            body = {key: value for key, value in row.items() if key != "record_sha256"}
            if digest(body) != row["record_sha256"]:
                raise ValueError("Original member record checksum failed")
            job = jobs_by_id[row["job_id"]]
            if digest({key: value for key, value in job.items() if key != "job_id"}) != job["job_id"]:
                raise ValueError("Original member job checksum failed")
            for key in ("question_id", "task", "subtask", "split", "model_alias", "sample_index", "seed",
                        "model_fingerprint", "member_mode", "sampling_control", "prompt_version", "source_question_sha256"):
                if row[key] != job[key]:
                    raise ValueError("Member output identity does not match its exact submitted job")
            if row["status"] != "success" or row["cost_status"] != "recorded_actual_usage":
                raise ValueError("Incomplete or unaccounted execution cannot become a reusable cache key")
            if not row["scoring_isolated_outside_generation_runner"] or row["split"] != "fresh_dev_test":
                raise ValueError("Unexpected score access or dataset identity")
            if row["question_id"] not in identities or row["seed"] != seed or row["sample_index"] != 0:
                raise ValueError("Fresh original question or generation identity differs")
            if row["model_fingerprint"] != model["fingerprint"] or row["model_fingerprint"] != receipt["logical_model_fingerprint"]:
                raise ValueError("Physical/logical model fingerprint mismatch")
            result = row["result"]
            if result["model_id"] != model["model_id"] or result["model_revision"] != model["revision"]:
                raise ValueError("Pinned model identity differs from the physical result")
            if not result["model_identity_verified"] or result["seed"] != seed or row["prompt_version"] != member_prompt:
                raise ValueError("Model, seed or prompt identity was not verified")
            if hashlib.sha256(job["question"].encode()).hexdigest() != row["source_question_sha256"]:
                raise ValueError("Question-only source hash mismatch")
            if not isinstance(result["text"], str) or not result["text"].strip():
                raise ValueError("Empty generated member answer")
            attempt = row["attempt_id"]
            logical = (identities[row["question_id"]], member, seed)
            if attempt in attempts_seen or logical in logical_seen:
                raise ValueError("Duplicate physical cost or logical cache key")
            attempts_seen.add(attempt)
            logical_seen.add(logical)
            phase_attempts.add(attempt)
            phase_questions.add(row["question_id"])
            prompt = _integer(result["prompt_tokens"])
            completion = _integer(result["completion_tokens"])
            wall = float(result["wall_time_s"])
            if not math.isfinite(wall) or wall < 0:
                raise ValueError("Invalid actual physical execution duration")
            phase_prompt += prompt
            phase_completion += completion
            phase_wall += wall
            identity = dict(canonical_id=logical[0], question_id=row["question_id"],
                member_index=member, generation_seed=seed, model_fingerprint=row["model_fingerprint"],
                model_id=result["model_id"], model_revision=result["model_revision"],
                member_prompt_version=member_prompt, member_mode=row["member_mode"],
                sample_index=0, sampling_control=row["sampling_control"],
                source_question_sha256=row["source_question_sha256"],
                answer_sha256=hashlib.sha256(result["text"].encode()).hexdigest(),
                record_sha256=row["record_sha256"], job_id=row["job_id"], physical_attempt_id=attempt,
                source_path=str(output_path), source_sha256=item["canonical_output_declared_sha256"])
            cache_keys.append(identity)
            ledger.append(dict(**identity, prompt_tokens=prompt, completion_tokens=completion,
                tokens=prompt + completion, source_model_wall_s=wall, model_calls=1))
        resources = receipt["resources"]
        if len(outputs) != 400 or len(phase_questions) != 400:
            raise ValueError("Imported phase lacks complete original-question coverage")
        if phase_attempts != set(resources["physical_attempt_ids"]):
            raise ValueError("Receipt physical execution identity differs")
        if any(resources[name] != value for name, value in
               (("actual_attempt_rows", 400), ("successful_jobs", 400), ("unknown_cost_rows", 0),
                ("prompt_tokens", phase_prompt), ("completion_tokens", phase_completion))):
            raise ValueError("Actual source-resource ledger differs from its receipt")
        if not math.isclose(resources["model_call_wall_seconds_sum"], phase_wall, rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError("Physical source-wall duration differs from its receipt")
        if receipt["fresh_gold_decoded"] or receipt["gold_scoring_ops"] or receipt["sealed_payloads_decoded"]:
            raise ValueError("Imported member phase has a forbidden scoring-access claim")
        aggregator_fingerprints.add(receipt["aggregator_fingerprint"])
        summaries.append(dict(member_index=member, alias=item["alias"], generation_seed=seed,
            complete_original_questions=400, successful_member_calls=400,
            prompt_tokens=phase_prompt, completion_tokens=phase_completion, tokens=phase_prompt + phase_completion,
            source_model_wall_s=phase_wall, importer_sha256=item["receipt"]["sha256"],
            output_sha256=item["canonical_output_declared_sha256"], source_plan_sha256=receipt["plan_sha256"],
            model_fingerprint=receipt["logical_model_fingerprint"]))
    if len(summaries) != 8 or len(aggregator_fingerprints) != 1:
        raise ValueError("The frozen eight-member import cohort or aggregator identity differs")
    expected_slots = {(member, seed) for member in range(6) for seed in (42, 137, 2024)}
    complete_slots = {(row["member_index"], row["generation_seed"]) for row in summaries}
    missing = [dict(member_index=member, generation_seed=seed, original_questions=400,
                    actual_tokens=None, state="NO_COMPLETE_LOCAL_IMPORT_IN_FROZEN_COHORT")
               for member, seed in sorted(expected_slots - complete_slots)]
    write_json(private / "VERIFIED_MEMBER_CACHE_KEYS.json", dict(
        schema="E18_verified_unscored_member_cache_keys_v1", primary_key_fields=["canonical_id", "member_index", "generation_seed"],
        keys=cache_keys, actual_output_text_stored=False, gold_or_reward_payload_reads=0))
    write_json(private / "MEMBER_PHYSICAL_LEDGER.json", dict(
        schema="E18_unique_paid_member_execution_ledger_v1", rows=ledger,
        physical_deduplication="attempt_id", prior_global_receipt_totals_added=False))
    write_json(private / "CACHE_SCHEMA_AND_SOURCE_ACCESS.json", dict(files=accesses,
        input_npz_decoded_members=["question_ids", "canonical_hashes"], fresh_reward_payload_reads=0,
        sealed_paths_opened=0))
    remote_path = private / "REMOTE_CACHE_STATUS.json"
    remote = read_metadata(remote_path, []) if remote_path.exists() else None
    spent = dict(member_calls=len(ledger), prompt_tokens=sum(row["prompt_tokens"] for row in ledger),
        completion_tokens=sum(row["completion_tokens"] for row in ledger), tokens=sum(row["tokens"] for row in ledger),
        source_model_wall_s=sum(row["source_model_wall_s"] for row in ledger))
    result = dict(schema="E18_B4_existing_cache_identity_and_cost_v1", status="LOCAL_MEMBER_COHORT_VERIFIED_REMOTE_UNKNOWN",
        question_count=400, frozen_model_preparation=frozen, complete_member_files=summaries,
        unique_physical_execution_count=len(attempts_seen), verified_cache_key_count=len(cache_keys),
        actual_spent_verified_members=spent, missing_full_pool_slots=missing,
        missing_scope="Full6 x3 x400 coverage reference only; four-policy required union is frozen separately",
        missing_actual_tokens=None, new_e18_model_calls=0, new_e18_gold_reveals=0,
        verified_fresh_aggregation_cache_count=0, actual_unverified_aggregation_tokens=None,
        aggregation_scope="Member-only audit; absence of verified aggregate metadata is not proof of no remote outputs",
        aggregator_fingerprint=next(iter(aggregator_fingerprints)),
        pinned_member_prompt=dict(version=member_prompt, source_sha256=sha256(member_source)),
        pinned_parser=dict(version=parser_version, source_sha256=sha256(parser_source), applied_to_fresh_answers=False),
        remote_status="UNKNOWN", remote_receipt_sha256=sha256(remote_path) if remote else None,
        fresh_rewards_read=0, fresh_gold_read=0, sealed_payloads_opened=0,
        actual_resource_collection_authorized_before_cache_audit=True,
        execution_deduplication="3200 globally unique attempt_ids; no sum of repeated global history receipt totals",
        confidence_intervals=0, bootstrap_runs=0, model_fits=0,
        private_cache_keys_sha256=sha256(private / "VERIFIED_MEMBER_CACHE_KEYS.json"),
        private_physical_ledger_sha256=sha256(private / "MEMBER_PHYSICAL_LEDGER.json"),
        source_sha256=sha256(__file__), cpu_seconds=time.process_time()-before_cpu,
        wall_seconds=time.perf_counter()-before_wall)
    write_json(private / "CACHE_AUDIT.json", result)
    write_json(public / "CACHE_AUDIT.json", result)
    _write_report(public / "B4_CACHE_AND_EXECUTION_COST.md", result)
    return result


def _write_report(path, result):
    lines = ["# E18 已有缓存身份与执行费用核验", "", "## 已验证范围", "",
        "原fresh400的8份完整成员输出已逐行核对。容器先检查键名，确认无标准答案和真实评分载荷后，读取模型输出及执行资源。题目身份绑定原input-only向量包，模型、seed、prompt、job及record哈希分别验证。模型答案仅存哈希，逐题执行身份留在私有账本。", "",
        "E17完整600题准备工件的5个文件哈希匹配，模型容器只检查成员名，权重载荷未解码。", "",
        "## 实际已花费成员执行", "",
        "| 成员 | seed | 已完成题数 | 输入Token | 输出Token | 总Token |", "|---|---:|---:|---:|---:|---:|"]
    for row in result["complete_member_files"]:
        lines.append(f"| {row['alias']} | {row['generation_seed']} | 400 | {row['prompt_tokens']} | {row['completion_tokens']} | {row['tokens']} |")
    cost = result["actual_spent_verified_members"]
    lines.extend(["", f"合计{cost['member_calls']}个唯一物理执行，输入{cost['prompt_tokens']}、输出{cost['completion_tokens']}，总{cost['tokens']} Token。原调用wall合计{cost['source_model_wall_s']:.4f}秒，属于逐调用历史耗时之和。它不提供并行端到端GPU时长。",
        "", "每份导入receipt还包含更大历史范围的全局费用快照。本次按唯一attempt_id和该phase实际资源核对，避免重复相加。", "",
        "## 缺失与新增预算", "",
        "完整六成员、三生成、400题的参考覆盖共有18格，本地冻结导入cohort有8格完整，另10格的实际Token记为未知。此参考覆盖不决定后续运行规模，四套冻结策略实际需要的调用并集另行封存。",
        "", "本核验已取得的fresh聚合缓存证据为0，尚未核验的聚合输出及其费用保持未知。成员生成与组合聚合分别计费，原题标准答案按canonical/reference身份去重。当前阶段新增LLM调用、gold揭示、评分及预测头拟合均为0。", "",
        "## 远端状态", "",
        "已有SSH端点及本地host pin已核对。授权升级后的SSH握手和一次短TCP连接均超时，远端缓存状态为UNKNOWN。连接未进入认证和文件读取阶段。本轮没有启动远端实验。具体连接证据保存在私有REMOTE_CACHE_STATUS.json。", "",
        "## 优势、风险与置信度", "",
        "优势是同时核对原题、模型、生成协议、实际输出与物理成本身份，所有方法可以合法共享同一缓存。风险是远端未连通，未验证其余完整或部分输出，缺失项的实际Token及未来聚合Token无法据现有证据确定。已核验本地成员身份与费用的置信度为高，远端完整性的置信度为低。",
        "", "本报告不评价成员答案正确性，全部结论属于身份与执行费用审计。"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=Path(r"E:\语言模型组合路由"))
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args()
    destination = arguments.output_root or arguments.asset_root / "research/e18_mechanism_and_confirmation_preparation"
    result = audit(arguments.asset_root, destination)
    print(json.dumps(dict(status=result["status"], verified_cache_keys=result["verified_cache_key_count"],
                          actual_spent=result["actual_spent_verified_members"]), ensure_ascii=False))
