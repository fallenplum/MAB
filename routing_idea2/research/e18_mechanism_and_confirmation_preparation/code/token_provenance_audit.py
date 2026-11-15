"""Read-only prompt-length distribution and physical-token source supplement."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from cache_audit import _fresh_identities, read_metadata, read_unscored_rows, sha256, write_json


INFERENCE_SOURCE_SHA256 = "3be35e1be1d30487e10b45fabbeae7254a5931c297359d9ef37adf80dd16d0ef"


def audit(asset_root, output_root):
    asset_root, output_root = Path(asset_root), Path(output_root)
    private, public = output_root / "private", output_root / "public"
    accesses = []
    ledger = read_metadata(private / "MEMBER_PHYSICAL_LEDGER.json", accesses)["rows"]
    manifest = read_metadata(asset_root / "research/e17_supervision_distortion_audit/private/confirmation_input_only/INPUT_AND_CACHE_MANIFEST.json", accesses)
    identities = _fresh_identities(asset_root / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/fresh_dev_test_inputs.npz")
    input_rows = read_unscored_rows(asset_root / "artifacts/private/routing_redesign_20261104/fresh_dev400/inputs_only.jsonl", accesses)
    raw_inputs = {row["question_id"]: row for row in input_rows}
    if len(raw_inputs) != 400 or set(raw_inputs) != set(identities):
        raise ValueError("Safe original question inputs and canonical identity archive differ")
    identity_matches = canonical_source_matches = source_question_matches = 0
    phases = defaultdict(list)
    for row in ledger:
        question = row["question_id"]
        source = raw_inputs[question]
        identity_matches += row["canonical_id"] == identities[question]
        canonical_source_matches += source["canonical_question_sha256"] == identities[question]
        source_question_matches += hashlib.sha256(source["question"].encode()).hexdigest() == row["source_question_sha256"]
        phases[(row["member_index"], row["generation_seed"])].append(row)
    if identity_matches != 3200 or canonical_source_matches != 3200 or source_question_matches != 3200:
        raise ValueError("Original input/canonical/source-question identities differ")
    inference = asset_root / "src/budget_router/inference.py"
    if sha256(inference) != INFERENCE_SOURCE_SHA256:
        raise ValueError("The inspected physical token-count implementation changed")
    source_matches = context_matches = software_false = context_software_false = 0
    provenance = []
    for item in manifest["cache_importer_receipts"]:
        receipt_path = Path(item["receipt"]["path"])
        receipt = read_metadata(receipt_path, accesses)
        plan_path = receipt_path.with_name("EXECUTION_PLAN.json")
        plan = read_metadata(plan_path, accesses)
        source = plan["original_source_sha256"]["src/budget_router/inference.py"]
        source_matches += source == INFERENCE_SOURCE_SHA256
        software_false += receipt["software_test_only"] is False
        contexts = []
        for entry in receipt["actual_contexts"]:
            path = Path(entry["path"])
            if sha256(path) != entry["sha256"]:
                raise ValueError("Original physical context changed")
            context = read_metadata(path, accesses)
            context_matches += context["inference_source_sha256"] == INFERENCE_SOURCE_SHA256
            context_software_false += context["software_test_only"] is False
            contexts.append(dict(source_sha256=entry["sha256"],
                inference_source_sha256=context["inference_source_sha256"],
                software_test_only=context["software_test_only"], role=context.get("role")))
        provenance.append(dict(member_index=item["member_index"], generation_seed=item["generation_seed"],
            receipt_sha256=item["receipt"]["sha256"], plan_sha256=sha256(plan_path),
            inference_source_sha256=source, software_test_only=receipt["software_test_only"],
            physical_contexts=contexts))
    if source_matches != 8 or context_matches != 8 or software_false != 8 or context_software_false != 8:
        raise ValueError("Physical token source or production-context status differs")
    distributions = []
    for (member, seed), rows in sorted(phases.items()):
        tokens = [row["prompt_tokens"] for row in rows]
        histogram = Counter(tokens)
        distributions.append(dict(member_index=member, generation_seed=seed,
            executions=len(rows), sum_prompt_tokens=sum(tokens), min=min(tokens), max=max(tokens),
            nunique=len(histogram), quartiles=np.percentile(tokens, [0, 25, 50, 75, 100]).tolist(),
            histogram=[dict(prompt_tokens=value, executions=count) for value, count in sorted(histogram.items())]))
    cross_seed = []
    for member in (0, 2):
        first = {row["canonical_id"]: row["prompt_tokens"] for row in phases[(member, 42)]}
        second = {row["canonical_id"]: row["prompt_tokens"] for row in phases[(member, 137)]}
        cross_seed.append(dict(member_index=member, matched_questions=len(first),
                               identical_prompt_lengths=first == second))
    result = dict(schema="E18_physical_token_provenance_and_identity_supplement_v1", status="PASS",
        phases=distributions, matched_seed_checks=cross_seed,
        canonical_id_matches_frozen_npz=identity_matches,
        safe_input_canonical_matches_npz=canonical_source_matches,
        raw_question_sha_matches_execution=source_question_matches,
        production_import_receipts=software_false, production_physical_contexts=context_software_false,
        source_matches=source_matches, physical_context_source_matches=context_matches,
        prompt_measurement="Runtime tokenizer input_ids.shape[-1] after template/tokenization",
        completion_measurement="Actual model.generate output suffix generated.numel()",
        measurement_source=dict(path=str(inference), sha256=INFERENCE_SOURCE_SHA256,
            input_count_lines=[215, 217], generation_call_line=232, generated_suffix_line=238,
            result_count_lines=[239, 240]),
        cost_classification="ACTUAL_EXECUTED_TOKEN_SEQUENCE_COUNTS_NO_ESTIMATION",
        source_bindings=provenance, new_tokenizer_or_model_calls=0,
        fresh_gold_or_rewards_read=0, sealed_files_opened=0,
        source_sha256=sha256(__file__))
    write_json(private / "TOKEN_PROVENANCE_SUPPLEMENT.json", result)
    write_json(private / "TOKEN_PROVENANCE_SOURCE_ACCESS.json", dict(files=accesses))
    write_json(public / "TOKEN_PROVENANCE_SUPPLEMENT.json", result)
    lines = ["# 成员Token计量来源补充", "", "## 逐phase分布", "",
        "8个phase输入Token分别随各题真实tokenizer输入长度变化，合计560000。每个phase并未分配固定70000 Token。", "",
        "| 成员 | seed | 执行数 | 输入Token合计 | 最小 | 最大 | 不同长度数 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in distributions:
        lines.append(f"| {row['member_index']} | {row['generation_seed']} | {row['executions']} | {row['sum_prompt_tokens']} | {row['min']} | {row['max']} | {row['nunique']} |")
    lines.extend(["", "## 测量来源", "",
        "物理Transformers执行器在tokenization后取input_ids实际张量长度作为prompt_tokens，实际model.generate后取生成后缀张量的numel作为completion_tokens。它们是运行时序列长度计数。当前核验没有重新执行tokenizer或模型。",
        "", "源码SHA与8份原执行计划、8份物理context逐项一致，8份import receipt及8份context均明确software_test_only=false。Qwen3和Granite在两个已有seed的400道同题上，输入长度均逐题完全相同。",
        "", "## 身份口径", "",
        "3200条cache key的canonical_id直接取原E17同一input-only NPZ中的canonical_hashes。再与安全原题输入文件的canonical字段及question文本SHA交叉核对，三项分别3200/3200匹配。该审核未使用标准答案或真实正确性。",
        "", "## 优势与限制", "",
        "优势是逐题长度分布、原物理生产context、源码绑定和导入账本相互核对。限制是历史调用没有在本阶段重新运行，验证依赖已保存且哈希一致的执行证据。这里的actual指模型输入与生成后缀序列计数，源wall合计仍无法替代真实GPU kernel时间。已核验来源的置信度为高。"])
    with (public / "B4_TOKEN_PROVENANCE.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=Path(r"E:\语言模型组合路由"))
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args()
    output = arguments.output_root or arguments.asset_root / "research/e18_mechanism_and_confirmation_preparation"
    result = audit(arguments.asset_root, output)
    print(json.dumps(dict(status=result["status"], canonical_matches=result["canonical_id_matches_frozen_npz"],
                          source_matches=result["source_matches"])))
