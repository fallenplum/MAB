"""Read-only discovery and replay of the existing train600 experiment assets.

This entry point never reads validation200, fresh400 or sealed200 payloads.
It writes aggregate provenance and replay results to the new E9 Bayes directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
ASSETS = Path(r"E:\语言模型组合路由")
E8 = "research/e8_decision_focused_combination_learning"
NS = "research/m4_noise_aware_set_learning"
INPUTS = "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/selector_train_inputs.npz"
OUTCOMES = "artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_outcomes.npz"
FOLDS = "research/conditional_gain_20261106/private/FOLDS_PRIVATE.npz"
EXPECTED_SHA = {
    INPUTS: "57e39747ec2756bcd79ab00885b3c08ad4f91fdb3ec900cbed15146420cd17c3",
    OUTCOMES: "a01230f2bed3e7f9d05713bf9e3e4f1cf44987b41c460f91e1d83bc1c29f191e",
    FOLDS: "6b433dcd2d8e6cd9938d2dd2d97261e18c3ae3d18325d7a04c6590e4d0600d50",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def archive(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Audit output already exists: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_record(path, root, role, permission):
    path = Path(path)
    record = {"root": str(root), "relative_path": str(path.relative_to(root)),
              "role": role, "permission": permission, "status": "MISSING"}
    if path.is_file():
        record.update(status="PRESENT", bytes=path.stat().st_size, sha256=sha(path))
    return record


def git_state(workspace):
    process = subprocess.run(
        ["git", "-c", f"safe.directory={workspace}", "-C", str(workspace), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False)
    if process.returncode == 0:
        return {"status": "LOCAL_CHECKOUT", "sha": process.stdout.strip()}
    return {"status": "NO_LOCAL_CHECKOUT", "sha": None,
            "reason": process.stderr.strip(), "no_branch_mutation_attempted": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=ASSETS)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "results/e9_bayes_gate")
    args = parser.parse_args()
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    root = args.asset_root
    records, public_checks = [], []
    required = [INPUTS, OUTCOMES, FOLDS,
                "configs/pool_v1.json",
                "results/decision_feedback_v2/s1_typed_aggcal_generation42_9ef6c02bc9fb/OPERATOR_FREEZE_LOCK_v1.json",
                "artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_matrix_receipt.json",
                f"{E8}/private/qualification/FOLD_PACKS_MANIFEST.json",
                f"{E8}/public/E8_PROTOCOL.json",
                f"{E8}/public/qualification/e8_assets.py",
                f"{E8}/public/training/e8_training_v1.py",
                f"{NS}/public/NS_SPO_PROTOCOL.json",
                f"{NS}/public/JOINT_NOISE_PROTOCOL.json",
                f"{NS}/public/JOINT_NOISE_COMPLETE_RESULTS.json",
                f"{NS}/public/JOINT_REPLICATE_PROVENANCE_AUDIT.json",
                f"{NS}/public/POSTERIOR_INTERPRETATION_LIMITS.md",
                f"{NS}/public/code/ns_training.py",
                f"{NS}/public/code/evaluate_ns.py"]
    for relative in required:
        records.append(file_record(root / relative, root, "historical required input", "READ_ONLY_TRAIN600_OR_SOURCE"))
    selections = {
        "e5_paired_gain_validation": ["E5_EXPERIMENT_CONFIG.json", "e5_common.py", "EXPERIMENT_SUMMARY.md", "E5_COMPLETE_RESULTS.json"],
        "e6_contrast_directed_feedback": ["E6_RUN_CONFIG.json", "prepare_e6.py", "EXPERIMENT_SUMMARY.md", "E6_COMPLETE_RESULTS.json"],
        "e7_opportunity_and_feedback_learning": ["E7_PROTOCOL.json", "e7_common.py", "05_E7_RESEARCH_DECISION.md", "E7_COMPLETE_RESULTS.json"],
        "e8_decision_focused_combination_learning": ["E8_PROTOCOL.json", "training/e8_training_v1.py", "04_MAIN_COMPARISONS.md", "E8_COMPLETE_RESULTS.json"],
        "e8_m4_mechanism_audit": ["M4_MECHANISM_PROTOCOL.json", "EXPERIMENT_SUMMARY.md", "M4_MECHANISM_COMPLETE_RESULTS.json"],
        "m4_noise_aware_set_learning": ["NS_SPO_PROTOCOL.json", "code/ns_training.py", "EXPERIMENT_SUMMARY.md", "NS_SPO_COMPLETE_RESULTS.json"],
    }
    for study, filenames in selections.items():
        public = root / "research" / study / "public"
        manifest_path = public / "PUBLIC_DELIVERY_MANIFEST.json"
        records.append(file_record(manifest_path, root, "historical publication manifest", "READ_ONLY_SOURCE"))
        published = read_json(manifest_path)["files"]
        for filename in filenames:
            path = public / filename
            record = file_record(path, root, "historical protocol/source/summary", "READ_ONLY_SOURCE")
            records.append(record)
            expected = published.get(filename, {}).get("sha256")
            matched = record.get("sha256") == expected if expected else None
            public_checks.append({"study": study, "relative": filename, "listed": expected is not None, "hash_matches": matched})
            if matched is False:
                raise AssertionError(f"Historical public asset hash mismatch: {path}")

    documents = ["MAB_E5-E8.1_阶段性实验汇报与后续研究方案讨论稿.docx", "MAB_E5_E8_1_formula_revised.docx"]
    document_reads = []
    for filename in documents:
        path = WORKSPACE / filename
        record = file_record(path, WORKSPACE, "user supplied stage report", "READ_ONLY_CONTEXT")
        records.append(record)
        if path.exists():
            with zipfile.ZipFile(path) as source:
                xml = ET.fromstring(source.read("word/document.xml"))
            text = "".join(xml.itertext())
            document_reads.append({"relative_path": filename, "sha256": record["sha256"],
                                   "paragraphs": len(xml.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p")),
                                   "extracted_character_count": len(text), "text_not_copied_to_output": True})
    if not document_reads:
        raise FileNotFoundError("Required stage report or formula revision not found")

    for relative, expected in EXPECTED_SHA.items():
        if sha(root / relative) != expected:
            raise AssertionError(f"Frozen source SHA mismatch: {relative}")
    inputs, outcomes, folds = archive(root / INPUTS), archive(root / OUTCOMES), archive(root / FOLDS)
    y = outcomes["aggregate_scores"].transpose(1, 0, 2)
    assert y.shape == (600, 3, 20) and np.isin(y, [0, 1]).all()
    assert inputs["embeddings"].shape == (600, 384)
    assert len(set(inputs["question_ids"].tolist())) == 600
    assert len(set(inputs["canonical_hashes"].tolist())) == 600
    assert np.array_equal(inputs["question_ids"], outcomes["question_ids"])
    assert np.array_equal(inputs["canonical_hashes"], outcomes["canonical_hashes"])
    assert outcomes["generation_seeds"].tolist() == [42, 137, 2024]
    assert Counter(inputs["task_labels"].tolist()) == {"gsm8k": 300, "bbh": 300}
    assert len(set(inputs["subtask_labels"][inputs["task_labels"] == "bbh"].tolist())) == 27
    fold_manifest = read_json(root / E8 / "private/qualification/FOLD_PACKS_MANIFEST.json")
    action_arrays = {name: np.full(600, -1, dtype=int) for name in ("M0_original", "M0_tuned", "M4", "NS_main")}
    split_checks = []
    inner_audit = []
    for fold in range(5):
        history, held = folds[f"fold{fold}_train"], folds[f"fold{fold}_holdout"]
        assert len(history) == 480 and len(held) == 120 and not set(history) & set(held)
        pack_base = root / E8 / f"private/qualification/fold_packs/fold{fold}"
        pack = archive(pack_base / "history480.npz")
        evaluation = archive(pack_base / "evaluation_inputs_only120.npz")
        assert set(evaluation) == {"E", "ids", "canonical", "global_indices"}
        assert np.array_equal(pack["global_indices"], history) and np.array_equal(evaluation["global_indices"], held)
        assert np.array_equal(pack["Y"], y[history])
        assert np.array_equal(pack["E"], inputs["embeddings"][history])
        assert np.array_equal(evaluation["E"], inputs["embeddings"][held])
        for role in ("history", "evaluation_inputs", "evaluation_metadata"):
            metadata = fold_manifest["folds"][fold][role]
            path = Path(metadata["path"])
            assert sha(path) == metadata["sha256"]
            records.append(file_record(path, root, role, "READ_ONLY_FOLD_LOCAL"))
        for method in ("M0_original", "M0_tuned", "M4"):
            base = root / E8 / f"private/training/fold{fold}/{method}"
            predictions = archive(base / "eval_predictions.npz")
            assert np.array_equal(predictions["eval_indices"], held)
            assert np.array_equal(predictions["actions"], predictions["raw_q"].argmax(axis=1))
            action_arrays[method][held] = predictions["actions"]
            for filename in ("eval_predictions.npz", "model_state.npz", "FIT_LOG.json", "FROZEN_MODEL_RECEIPT.json"):
                records.append(file_record(base / filename, root, f"frozen {method} {filename}", "READ_ONLY_HISTORICAL_MODEL"))
        log = read_json(root / E8 / f"private/training/fold{fold}/M0_tuned/FIT_LOG.json")
        fit_pairs = log["candidates"][0]["fits"]
        assert sorted(np.concatenate([np.asarray(row["held_rows"]) for row in fit_pairs]).tolist()) == list(range(480))
        assert all(not set(row["held_rows"]) & set(row["fitting_rows"]) for row in fit_pairs)
        inner_audit.append({"outer": fold, "folds": 3,
                            "source": f"{E8}/private/training/fold{fold}/M0_tuned/FIT_LOG.json",
                            "index_unit": "local rows in historical480",
                            "seed": 20261113 + fold, "selected": log["selected_parameters"]})
        base = root / NS / f"private/runtime/training/private/fold{fold}/NS_main"
        predictions = archive(base / "PREDICTIONS_FROZEN.npz")
        assert np.array_equal(predictions["evaluation_indices"], held)
        assert np.array_equal(predictions["actions"], predictions["scores"].argmax(axis=1))
        action_arrays["NS_main"][held] = predictions["actions"]
        for filename in ("PREDICTIONS_FROZEN.npz", "MODEL.npz", "FIT_LOG.json", "FROZEN_RECEIPT.json"):
            records.append(file_record(base / filename, root, f"frozen NS {filename}", "READ_ONLY_HISTORICAL_MODEL"))
        split_checks.append({"fold": fold, "history": 480, "evaluation": 120,
                             "ids_and_canonical_disjoint": not set(pack["canonical"]) & set(evaluation["canonical"]),
                             "raw_matrix_agreement": True, "evaluation_contains_no_rewards": True})
    replay = {}
    for method, actions in action_arrays.items():
        assert (actions >= 0).all()
        reward = y[np.arange(600)[:, None], np.arange(3)[None, :], actions[:, None]]
        counts = np.bincount(actions, minlength=20)
        replay[method] = {"accuracy": float(reward.mean()), "generation_accuracy": reward.mean(axis=0).tolist(),
                          "most_frequent_action": int(counts.argmax()), "most_frequent_count": int(counts.max()),
                          "actions_sha256": hashlib.sha256(actions.astype(np.int64).tobytes()).hexdigest()}
    for method, expected in (("M0_original", 1535 / 1800), ("M4", 1547 / 1800), ("NS_main", 1549 / 1800)):
        assert abs(replay[method]["accuracy"] - expected) < 1e-12
    assert replay["M4"]["most_frequent_count"] == 467
    assert replay["NS_main"]["most_frequent_count"] == 170
    same_actions = bool(np.array_equal(action_arrays["M0_original"], action_arrays["M0_tuned"]))

    # Presence-only checks, no list/read/hash of any held-out payload.
    boundaries = [{"path": "artifacts/private/routing_redesign_20261104/fresh_dev400",
                   "is_directory": (root / "artifacts/private/routing_redesign_20261104/fresh_dev400").is_dir(),
                   "operation": "is_dir only", "payload_read": False},
                  {"path": "data/jobs/sealed_test", "is_directory": (root / "data/jobs/sealed_test").is_dir(),
                   "operation": "is_dir only", "payload_read": False,
                   "interpretation": "Candidate dedicated path not authoritative. Protocol names sealed_test; contents intentionally unlocated."}]
    receipt_path = root / "research/e12_shared_feedback_learning/private/REMOTE_DELIVERY_VERIFIED.json"
    previous = read_json(receipt_path) if receipt_path.exists() else {}
    manifest = {"status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
                "assets": records, "public_manifest_checks": public_checks,
                "documents_read": document_reads, "source_git": git_state(WORKSPACE),
                "latest_known_remote_commit_from_receipt": previous.get("commit_sha"),
                "remote_commit_live_checked_in_this_script": False,
                "python": platform.python_version(), "executable": __import__("sys").executable,
                "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "psutil")},
                "data_shapes": {"aggregate_scores_source": [3, 600, 20], "internal": [600, 3, 20], "embeddings": [600, 384]},
                "task_counts": {"gsm8k": 300, "bbh": 300}, "bbh_subtasks": 27,
                "split_checks": split_checks, "inner_splits": inner_audit,
                "M0_original_and_tuned_actions_identical": same_actions,
                "protected_boundaries": boundaries,
                "new_physical_calls": 0, "new_gold": 0,
                "validation_fresh_sealed_payload_reads": 0,
                "cpu_seconds": time.process_time() - cpu_start,
                "wall_seconds": time.perf_counter() - wall_start}
    missing = [item["relative_path"] for item in records if item["status"] == "MISSING"]
    if missing:
        manifest["status"] = "BLOCKED"
        manifest["missing"] = missing
    output = args.output
    write_json(output / "00_asset_manifest.json", manifest)
    write_json(output / "00_historical_replay.json", {"status": "PASS", "methods": replay,
               "M0_original_and_tuned_actions_identical": same_actions,
               "new_fits": 0, "new_physical_calls": 0,
               "scores_are_cached_final_aggregate_rewards": True})
    report = """# E9 Bayes gate 输入与协议核查

## 范围与身份

本轮独立执行用户给出的 `MAB_E9_Codex_Experiment_Prompt.md`。现有 `e9_consistent_cost_sensitive_learning` 对应经验/后验均值与CSCE矩阵，方法和主比较不同。本轮使用独立的 `experiments/e9_bayes_gate` 与 `results/e9_bayes_gate`，旧研究资产保持只读。

已实际读取阶段汇报及同版本公式规范版。原始train600评分为3×600×20，转为题×生成×组合。评分、ID、canonical、MiniLM384向量、GSM/BBH各300与BBH27子任务映射一致。五折history480与evaluation120按原题和canonical互斥，历史包逐项与原矩阵一致，评价输入包没有奖励字段。

## 独立历史回放

| 冻结方法 | 准确率 | 最高频动作题数 |
|---|---:|---:|
| M0 original | 85.2778% | 55/600 |
| M4 exact SPO+ | 85.9444% | 467/600 |
| NS-SPO+ δ=0.05 | 86.0556% | 170/600 |

实际回放使用保存动作与真实最终聚合评分，不重新拟合、不生成。M0 original与tuned的600个冻结动作已逐项核对，结论见JSON。M0 tuned五外折均选择multiplier10，但其系数尺度与original仍不同。不得仅因动作相同宣称拟合程序相同。

## 合法预处理和内层配置

E8固定MiniLM-L6-v2 384维，合法拟合区PCA16(full)、StandardScaler及受罚常数，线性17×20共340参数。原正则λ=100×multiplier/(3×n×20)，候选multiplier为0.1、1、10，原M0固定raw_alpha100。内层三折由task/subtask分层，seed=20261113+outer；保存的FIT_LOG记录的是history480的局部行号。已核对每个内层训练/评价互斥、三评价分区覆盖history480。

## 生成与后验边界

原成员生成重复为42、137、2024，聚合seed固定42。同题每个生成块共享六成员输出，20组合之间存在相关性。seed编号本身不证明独立随机化。NS工作后验为有界联合截断Gaussian，噪声协方差20%对角收缩，历史区内部交叉拟合；512样本是诊断量，旧NS训练使用8样本。它不是精确Bernoulli后验，不自动提供真实潜在奖励区间校准或Bayes一致性。

## 来源及缺失事项

所有本轮依赖的原协议、源码、summary、公开manifest和私有冻结动作均记录真实路径、字节数及SHA256。所选历史公开项逐一与原manifest核验。专用sealed_test候选目录未定位不等于数据不存在，其协议来源仍保留；未搜索或读取其内容，fresh_dev400仅检查目录存在。validation200、fresh400和sealed200载荷读取次数均为0。

当前D工作区没有本地Git checkout。最近E12远端提交只引用已保存的核验收据，并非本脚本再次联网核验。分支创建留由主执行器处理，缺少本地Git不会改变实验数据或放宽评价权限。本轮不push、merge或重写历史。

## 核查状态

资产和历史主基准PASS后才允许本轮训练。数学资格、拟合和科学结论由各自真实结果给出，本文件不把输入核查PASS当作科学假设成立。若manifest列出关键MISSING项，状态为BLOCKED，应先补齐来源。
"""
    report_path = output / "00_protocol_audit.md"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "assets": len(records),
                      "public_checks": len(public_checks), "replay": replay,
                      "M0_actions_same": same_actions, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
