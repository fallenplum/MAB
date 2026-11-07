"""Train600-only raw execution, parser, provenance and duplicate-content audit.

Only whitelisted train result payloads and selected train jobs are decoded.
Other rows in mixed files may be hashed or header-scanned, never decoded as
answers. Source files and E3 are read-only. All derived artifacts stay in the
explicit output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import unicodedata

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

QUESTION_ID = re.compile(rb'(?<!\\)"question_id"\s*:\s*("(?:[^"\\]|\\.)*")')
RESULT_FIELD = re.compile(rb'(?<!\\)"result"\s*:')
SEEDS = (42, 137, 2024)
TEAMS = tuple(itertools.combinations(range(6), 3))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def physical(path):
    path = Path(path)
    text = str(path.resolve())
    if os.name == "nt" and len(text) >= 248 and not text.startswith("\\\\?\\"):
        return Path("\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text)
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf8")).hexdigest()


def canonical(question):
    return text_hash(" ".join(unicodedata.normalize("NFKC", question).split()))


def train_id(raw, allowed):
    match = QUESTION_ID.search(raw)
    require(match is not None, "Question ID unavailable before payload decoding")
    qid = json.loads(match.group(1))
    return qid, qid in allowed


class Files:
    """Memoized byte hashes and physical line offsets, without JSON decoding."""

    def __init__(self, root):
        self.root = root.resolve()
        self.records = {}
        self.offsets = {}
        self.handles = {}
        self.hash_bytes = 0
        self.selected_line_bytes = 0

    def path(self, record):
        path = Path(record["path"] if isinstance(record, dict) else record)
        path = path.resolve() if path.is_absolute() else (self.root / path).resolve()
        require(path.is_relative_to(self.root), "Source escaped asset root")
        return path

    def check(self, record, *, line_offsets=False):
        path = self.path(record)
        key = str(path)
        expected = record.get("sha256") if isinstance(record, dict) else None
        if key not in self.records or (line_offsets and key not in self.offsets):
            checksum, count = hashlib.sha256(), 0
            offsets = []
            with physical(path).open("rb") as stream:
                if line_offsets:
                    for raw in stream:
                        offsets.append(count)
                        count += len(raw)
                        checksum.update(raw)
                else:
                    for raw in iter(lambda: stream.read(1048576), b""):
                        count += len(raw)
                        checksum.update(raw)
            self.records[key] = {"path": key, "sha256": checksum.hexdigest(), "bytes": count}
            self.hash_bytes += count
            if line_offsets:
                self.offsets[key] = offsets
        require(expected is None or self.records[key]["sha256"] == expected, "Source byte hash mismatch")
        return path

    def selected_json_line(self, reference, allowed, expected_qid):
        path = self.check(reference, line_offsets=True)
        key = str(path)
        line = reference["line_index"]
        require(type(line) is int and 1 <= line <= len(self.offsets[key]), "Bad physical line reference")
        offset = self.offsets[key][line - 1]
        require("byte_offset" not in reference or reference["byte_offset"] == offset, "Source byte/line offset differs")
        if key not in self.handles:
            self.handles[key] = physical(path).open("rb")
        stream = self.handles[key]
        stream.seek(offset)
        raw = stream.readline()
        self.selected_line_bytes += len(raw)
        qid, permitted = train_id(raw, allowed)
        require(permitted and qid == expected_qid, "Selected source line is outside train whitelist")
        return json.loads(raw)

    def close(self):
        for handle in self.handles.values():
            handle.close()


def increment(table, key, parsed, cap, score):
    row = table.setdefault(key, {"rows": 0, "parsed": 0, "unparsed": 0, "cap_hits": 0, "correct": 0})
    row["rows"] += 1
    row["parsed"] += int(parsed)
    row["unparsed"] += int(not parsed)
    row["cap_hits"] += int(cap)
    row["correct"] += int(score)


def seed_duplicates(entries):
    """Content equality is descriptive, separate from execution identity."""
    output = {"triples": len(entries), "all_three_equal": 0, "all_three_distinct": 0,
              "at_least_one_equal_pair": 0, "pair_equal_counts": {"42_137": 0, "42_2024": 0, "137_2024": 0}}
    for values in entries.values():
        require(set(values) == set(SEEDS), "Cross-seed triple is incomplete")
        hashes = [values[seed] for seed in SEEDS]
        distinct = len(set(hashes))
        output["all_three_equal"] += int(distinct == 1)
        output["all_three_distinct"] += int(distinct == 3)
        output["at_least_one_equal_pair"] += int(distinct < 3)
        for left, right in ((42, 137), (42, 2024), (137, 2024)):
            output["pair_equal_counts"][f"{left}_{right}"] += int(values[left] == values[right])
    return output


def load_members(train, receipt, pool, references, files, member_messages):
    allowed = set(train["ids"].tolist())
    aliases = [model["alias"] for model in pool]
    sources = [record for record in receipt["sources"] if Path(record["path"]).suffix == ".jsonl"]
    require(len(sources) == 18, "Expected eighteen ordinary member source files")
    members, hashes = {}, defaultdict(dict)
    counts = Counter()
    for reference in sources:
        path = files.check(reference)
        with physical(path).open("rb") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                counts["physical_member_rows_header_checked"] += 1
                qid, permitted = train_id(raw, allowed)
                if not permitted:
                    counts["nontrain_member_payloads_skipped"] += 1
                    continue
                row = json.loads(raw)
                require(row["status"] == "success" and row["sampling_control"] is False, "Unexpected train member attempt")
                checksum = row["record_sha256"]
                require(digest({key: value for key, value in row.items() if key != "record_sha256"}) == checksum,
                        "Member execution record checksum differs")
                alias, seed = row["model_alias"], row["seed"]
                require(alias in aliases and seed in SEEDS, "Unknown member or generation seed")
                column, result = aliases.index(alias), row["result"]
                spec = pool[column]
                require(row["model_fingerprint"] == spec["fingerprint"] and result["model_id"] == spec["model_id"]
                        and result["model_revision"] == spec["revision"] and result["seed"] == seed
                        and result["model_identity_verified"] is True, "Member model/seed identity differs")
                question = references[qid]["question"]
                require(row["source_question_sha256"] == text_hash(question), "Member question differs from train reference input")
                job_fields = ("question_id", "task", "subtask", "split", "model_alias", "sample_index", "seed",
                              "model_fingerprint", "member_mode", "prompt_version", "source_question_sha256", "sampling_control")
                job = {key: row[key] for key in job_fields}
                job.update(question=question, model_id=spec["model_id"], model_revision=spec["revision"], messages=member_messages(question))
                require(digest(job) == row["job_id"], "Reconstructed question-only member job differs")
                key = (qid, seed, column)
                require(key not in members, "Duplicate ordinary member execution cell")
                members[key] = {"job_id": row["job_id"], "record_sha256": checksum,
                                "attempt_id": row["attempt_id"], "text_sha256": text_hash(result["text"])}
                hashes[(qid, column)][seed] = members[key]["text_sha256"]
                counts["train_member_results_decoded"] += 1
                counts["member_cap_hits"] += int(result["completion_reached_cap"] or result["completion_tokens"] >= 2048)
    require(len(members) == 600 * 3 * 6, "Train member coverage is incomplete")
    counts["member_score_requests"] = 0
    counts["member_score_observations"] = 0
    return members, seed_duplicates(hashes), dict(counts)


def audit(root, output):
    started, cpu = time.perf_counter(), time.process_time()
    os.environ["MAB_ASSET_ROOT"] = str(root)
    sys.path[:0] = [str(root / "research/decision_acquisition_e3/public"), str(root / "src"), str(root / "scripts/v2")]
    import assets
    from budget_router_v2 import scoring
    from budget_router.member_execution import member_messages
    from aggregation_sourceview import connect_database, stable_source_identity

    train, asset_refs = assets.load_assets(root)
    allowed = set(train["ids"].tolist())
    positions = {qid: i for i, qid in enumerate(train["ids"].tolist())}
    files = Files(root)
    databases = {}
    receipt_path = files.check(asset_refs["matrix_receipts"])
    receipt = json.loads(physical(receipt_path).read_text(encoding="utf-8-sig"))
    require(receipt["combinations"] == [list(team) for team in TEAMS], "Combination column mapping differs")
    pool_path = files.check(asset_refs["member_pool"])
    pool = json.loads(physical(pool_path).read_text(encoding="utf-8-sig"))["models"]
    require([model["index"] for model in pool] == list(range(6)), "Member pool list order differs")
    aliases = [model["alias"] for model in pool]
    freeze = json.loads(physical(files.check(asset_refs["operator_freeze"])).read_text(encoding="utf-8-sig"))
    scorer_path = files.check(root / "src/budget_router_v2/scoring.py")
    original_scorer_path = files.check(root / "src/budget_router/scoring.py")
    require(files.records[str(scorer_path)]["sha256"] == receipt["scorer_source_sha256"]
            and scoring.SCORER_VERSION == receipt["scorer_version"], "Frozen scoring source/version changed")
    reference_receipt_ref = next(record for record in receipt["sources"]
                                 if Path(record["path"]).name == "selector_train_references_v1.receipt.json")
    reference_receipt = json.loads(physical(files.check(reference_receipt_ref)).read_text(encoding="utf-8-sig"))
    reference_path = files.check(reference_receipt["projection"])
    references = {}
    with physical(reference_path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            require(row["question_id"] in allowed and row["question_id"] not in references, "Training reference projection contains foreign/duplicate rows")
            references[row["question_id"]] = row
    require(set(references) == allowed and len(references) == reference_receipt["rows"] == 600, "Train references incomplete")
    for qid, row in references.items():
        require(canonical(row["question"]) == train["canonical"][positions[qid]], "Recomputed canonical question hash differs")
    canonical_groups = Counter(canonical(row["question"]) for row in references.values())
    casefold_groups = Counter(" ".join(unicodedata.normalize("NFKC", row["question"]).casefold().split())
                             for row in references.values())
    members, member_duplicate_summary, member_counts = load_members(train, receipt, pool, references, files, member_messages)
    print(json.dumps({"phase": "members_verified", **member_counts}), flush=True)

    view_refs = [record for record in receipt["sources"] if Path(record["path"]).name == "SOURCEVIEW_RECEIPT.json"]
    require(len(view_refs) == 36, "Expected36 typed aggregate sourceviews")
    counts, reasons = Counter(), Counter()
    by_task, by_subtask, by_seed = {}, {}, {}
    cells, logical_ids, physical_ids, attempts = set(), set(), set(), set()
    member_uses = Counter()
    text_triples, input_triples, response_triples = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    mismatches, failures = [], []
    row_output = output / "RAW_AUDIT_ROWS.jsonl.gz"
    try:
        with gzip.GzipFile(filename=str(row_output), mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf8") as rows_out:
                for view_number, reference in enumerate(view_refs, 1):
                    view_receipt = json.loads(physical(files.check(reference)).read_text(encoding="utf-8-sig"))
                    source_path = files.path(view_receipt["sourceview"])
                    checksum = hashlib.sha256()
                    line_count = 0
                    with physical(source_path).open("rb") as stream:
                        for raw in stream:
                            checksum.update(raw)
                            if not raw.strip():
                                continue
                            line_count += 1
                            counts["physical_sourceview_rows_header_scanned"] += 1
                            match = RESULT_FIELD.search(raw)
                            require(match is not None, "Sourceview lacks payload boundary")
                            header = json.loads(raw[:match.start()] + b'"_payload_omitted":null}')
                            qid = header["question_id"]
                            if qid not in allowed:
                                counts["nontrain_aggregate_payloads_skipped_before_json_decode"] += 1
                                continue
                            row = json.loads(raw)
                            require(list(row)[-1] == "result" and row["stage"] == "stage2"
                                    and row["analysis_kind"] == "heterogeneous" and row["status"] == "SUCCESS"
                                    and row["software_test_only"] is False, "Unexpected aggregate scientific identity")
                            seed = row["member_generation_seed"]
                            require(seed in SEEDS and row["aggregation_seed"] == 42
                                    and row["full_operator_fingerprint"] == assets.OPERATOR_FP
                                    and row["logical_model_config_fp"] == freeze["logical_model_config_fingerprint"], "Aggregator generation/operator identity differs")
                            combo = TEAMS.index(tuple(sorted(aliases.index(alias) for alias in row["members"])))
                            q, g = positions[qid], SEEDS.index(seed)
                            cell = (q, g, combo)
                            require(cell not in cells, "Duplicate aggregate matrix cell")
                            cells.add(cell)
                            for value, collection in ((row["logical_job_id"], logical_ids), (row["physical_job_id"], physical_ids),
                                                       (row["physical_attempt_id"], attempts)):
                                require(value not in collection, "Execution identity reused across aggregate cells")
                                collection.add(value)
                            job = files.selected_json_line(row["logical_job_source"], allowed, qid)
                            require(digest(job) == row["logical_job_source"]["job_sha256"] and job["job_id"] == row["logical_job_id"]
                                    and job["members"] == row["members"] and job["seed"] == 42, "Logical aggregation job differs")
                            require(not ({"gold", "reference", "score", "member_scores", "aggregate_score"} & set(job)), "Forbidden label field in generation job")
                            for alias in row["members"]:
                                member_key = (qid, seed, aliases.index(alias))
                                provenance = job["member_provenance"][alias]
                                require(provenance["job_id"] == members[member_key]["job_id"]
                                        and provenance["record_sha256"] == members[member_key]["record_sha256"], "Aggregate points to different member execution")
                                member_uses[member_key] += 1
                            physical_job = files.selected_json_line(row["physical_job_source"], allowed, qid)
                            require(digest(physical_job) == row["physical_job_source"]["job_sha256"]
                                    and physical_job["job_id"] == row["physical_job_id"], "Physical job source differs")
                            raw_record = files.selected_json_line(row["raw_source"], allowed, qid)
                            raw_checksum = raw_record["record_sha256"]
                            require(digest({key: value for key, value in raw_record.items() if key != "record_sha256"}) == raw_checksum
                                    and raw_checksum == row["raw_source"]["record_sha256"], "Original physical output checksum differs")
                            body = raw_record["response"]
                            require(body["result"] == row["result"] and digest(body) == row["response_sha256"]
                                    and body["attempt_id"] == raw_record["attempt_id"] == row["physical_attempt_id"]
                                    and body["job_id"] == row["physical_job_id"] and body["status"] == "SUCCESS"
                                    and body["context_sha256"] == row["context_sha256"], "Raw output does not match selected sourceview")
                            request = {"attempt_id": row["physical_attempt_id"], "job_id": row["physical_job_id"],
                                       "model_fingerprint": row["physical_model_config_fp"], "split": physical_job["split"],
                                       "seed": physical_job["seed"], "messages": physical_job["messages"], "job_sha256": digest(physical_job)}
                            require(digest(request) == row["request_sha256"] == body["request_sha256"], "Physical request identity differs")
                            journal_path = files.check(row["journal_source"])
                            if str(journal_path) not in databases:
                                databases[str(journal_path)] = connect_database(journal_path, readonly=True, immutable=True)
                            journal = databases[str(journal_path)].execute(
                                "SELECT job_id,request_sha,status,response_json,response_sha,local_record_sha,started,finished FROM calls WHERE attempt_id=?",
                                (row["physical_attempt_id"],)).fetchone()
                            require(journal is not None and journal[0] == row["physical_job_id"] and journal[1] == row["request_sha256"]
                                    and journal[2] == "SUCCESS" and json.loads(journal[3]) == body
                                    and journal[4] == row["response_sha256"] and journal[5] == raw_checksum
                                    and math.isfinite(journal[6]) and math.isfinite(journal[7]) and journal[7] >= journal[6], "Selected raw response and read-only journal differ")
                            index_path = Path(view_receipt["index"]["path"])
                            index_key = str(index_path)
                            if index_key not in databases:
                                databases[index_key] = connect_database(index_path, readonly=True, immutable=True)
                            indexed = databases[index_key].execute("SELECT view_path,line_index,identity_json FROM sources WHERE logical_job_id=?",
                                                                  (row["logical_job_id"],)).fetchone()
                            require(indexed is not None and Path(indexed[0]) == source_path and indexed[1] == line_count
                                    and json.loads(indexed[2]) == stable_source_identity(row), "Execution source index differs")
                            result = row["result"]
                            require(result["seed"] == 42 and result["model_id"] == "Qwen/Qwen3-8B"
                                    and result["model_revision"] == "b968826d9c46dd6066d109eabc6255188de91218"
                                    and result["model_identity_verified"] is True and result["backend"] == "transformers", "Raw aggregator model identity differs")
                            for suffix in ("prompt_tokens", "completion_tokens", "wall_time_s"):
                                require(np.isclose(result[suffix], row["actual_resources"][suffix], rtol=1e-12, atol=1e-9)
                                        and np.isclose(result[suffix], train["cost"]["aggregate_" + suffix][g, q, combo], rtol=1e-12, atol=1e-9), "Raw model resource differs from matrix")
                            gold = references[qid]
                            scored = scoring.score_answer(gold["task"], result["text"], gold["reference"],
                                                          subtask=gold["subtask"], choices=gold["choices"])
                            cap = bool(result["completion_reached_cap"] or result["completion_tokens"] >= 2048)
                            value = float(scored.score)
                            if value != train["Y"][q, g, combo]:
                                mismatches.append({"question_index": q, "generation_seed": seed, "combo": combo,
                                                   "raw_rescored": value, "matrix_score": float(train["Y"][q, g, combo])})
                            counts["train_aggregate_payloads_decoded_and_raw_rescored"] += 1
                            counts["original_physical_outputs_verified"] += 1
                            counts["selected_journal_responses_decoded"] += 1
                            counts["selected_execution_index_rows_verified"] += 1
                            counts["parsed"] += int(scored.parsed)
                            counts["unparsed"] += int(not scored.parsed)
                            counts["nonempty"] += int(bool(result["text"].strip()))
                            counts["cap_hits"] += int(cap)
                            reasons[scored.reason] += 1
                            increment(by_task, gold["task"], scored.parsed, cap, value)
                            increment(by_subtask, gold["task"] + "/" + str(gold["subtask"] or ""), scored.parsed, cap, value)
                            increment(by_seed, str(seed), scored.parsed, cap, value)
                            text_sha = text_hash(result["text"])
                            text_triples[(q, combo)][seed] = text_sha
                            input_triples[(q, combo)][seed] = digest(job["messages"])
                            response_triples[(q, combo)][seed] = row["response_sha256"]
                            evidence = {"question_index": q, "generation_seed": seed, "combo": combo,
                                        "parsed": bool(scored.parsed), "score": value, "cap_hit": cap,
                                        "finish_reason": result["finish_reason"], "parser_reason": scored.reason,
                                        "text_sha256": text_sha, "logical_job_id": row["logical_job_id"],
                                        "physical_attempt_id": row["physical_attempt_id"], "response_sha256": row["response_sha256"],
                                        "sourceview_receipt_sha256": reference["sha256"], "sourceview_line": line_count}
                            rows_out.write(json.dumps(evidence, ensure_ascii=False, separators=(",", ":")) + "\n")
                            if not scored.parsed:
                                failures.append({key: evidence[key] for key in ("question_index", "generation_seed", "combo", "cap_hit", "finish_reason", "parser_reason", "text_sha256")})
                    require(line_count == view_receipt["sourceview"]["rows"]
                            and checksum.hexdigest() == view_receipt["sourceview"]["sha256"], "Sourceview row inventory/hash differs")
                    files.records[str(source_path)] = {"path": str(source_path), "sha256": checksum.hexdigest(), "bytes": physical(source_path).stat().st_size}
                    print(json.dumps({"phase": "aggregates", "sourceview": view_number, "total_sourceviews": len(view_refs),
                                      "train_outputs_verified": len(cells), "parsed": counts["parsed"], "unparsed": counts["unparsed"],
                                      "score_mismatches": len(mismatches), "elapsed_s": round(time.perf_counter() - started, 2)}), flush=True)
    finally:
        files.close()
        for connection in databases.values():
            connection.close()
    require(len(cells) == 36000 and len(logical_ids) == len(physical_ids) == len(attempts) == 36000, "Full real aggregate coverage incomplete")
    require(set(member_uses) == set(members) and set(member_uses.values()) == {10}, "Six-member execution reuse coverage differs")
    for row in (by_task, by_subtask, by_seed):
        for metrics in row.values():
            metrics["parsed_fraction"] = metrics["parsed"] / metrics["rows"]
            metrics["accuracy"] = metrics["correct"] / metrics["rows"]
    sources = list(files.records.values())
    return {"schema": "e4_train600_independent_raw_execution_audit_v1",
            "status": "PASS" if not mismatches else "RAW_LABEL_MISMATCHES_FOUND",
            "created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "asset_refs": asset_refs, "scorer_version": scoring.SCORER_VERSION,
            "scorer_source_sha256": files.records[str(scorer_path)]["sha256"],
            "underlying_original_scorer_sha256": files.records[str(original_scorer_path)]["sha256"],
            "gold_projection_sha256": reference_receipt["projection"]["sha256"],
            "counts": dict(counts), "member_counts": member_counts, "score_mismatches": mismatches,
            "parser_reason_counts": dict(reasons), "by_task": by_task, "by_subtask": by_subtask, "by_seed": by_seed,
            "unparsed_cases": failures,
            "seed_content_diagnostics": {"aggregate_full_text": seed_duplicates(text_triples),
                "aggregate_logical_input_messages": seed_duplicates(input_triples),
                "aggregate_full_response_objects": seed_duplicates(response_triples),
                "member_full_text": member_duplicate_summary,
                "interpretation": "Identical output text is content repetition; execution identity and seed provenance are audited separately"},
            "identity_and_reuse": {"unique_aggregate_logical_jobs": len(logical_ids), "unique_physical_jobs": len(physical_ids),
                "unique_physical_attempts": len(attempts), "member_dependency_links": sum(member_uses.values()),
                "unique_member_executions": len(members), "uses_per_member_execution": dict(Counter(member_uses.values())),
                "lexicographic_20_combination_columns_verified": True, "member_pool_aliases_in_order": aliases},
            "question_identity": {"train_questions": 600, "canonical_recomputed": 600,
                "canonical_mismatches": 0, "unique_canonical_hashes": len(canonical_groups),
                "normalization_duplicate_groups": sum(value > 1 for value in canonical_groups.values()),
                "casefold_normalization_duplicate_groups": sum(value > 1 for value in casefold_groups.values()),
                "canonical_rule": "SHA256(NFKC(question) with whitespace collapsed); case-sensitive; question field only",
                "semantic_near_duplicate_audit_performed": False},
            "permissions": {"allowed_payload_split": "selector_train", "allowed_question_count": 600,
                "nontrain_results_json_decoded": 0, "validation_payloads_decoded": 0, "fresh_payloads_decoded": 0,
                "sealed_payloads_decoded": 0, "physical_byte_scan_of_mixed_sources": True,
                "physical_byte_scan_is_not_semantic_payload_decode": True, "new_llm_calls": 0,
                "new_judge_calls": 0, "member_scores_requested": 0, "original_files_modified": False},
            "row_evidence": {"file": str(row_output), "rows": 36000,
                "sha256": hashlib.sha256(row_output.read_bytes()).hexdigest(), "raw_texts_exported": False},
            "source_files_byte_verified": sources,
            "runtime": {"elapsed_s": time.perf_counter() - started, "process_cpu_s": time.process_time() - cpu,
                "hashed_bytes": files.hash_bytes, "selected_source_line_bytes": files.selected_line_bytes},
            "limitations": ["Reference answers are the frozen train-only projection; this audit does not independently re-derive official benchmark truth",
                "Canonical normalization detects textual equality under NFKC/whitespace, not semantic duplicates",
                "Model execution evidence is retained local raw output and journal provenance; weights and GPU inference are not rerun",
                "Equal member or aggregator text across seeds does not by itself establish copied executions"]}


def notes(result):
    c = result["counts"]
    return "\n".join([
        "# E4-A1 原始输出真实性与评分复核", "", "## 核查范围", "",
        "仅使用train600合法题ID及训练专用真值。混合sourceview在result字段之前筛选题ID，原始物理输出与job均在完整JSON解析前再次核对train题ID。", "",
        "## 实际结果", "", f"审查状态为{result['status']}。独立复核{c['train_aggregate_payloads_decoded_and_raw_rescored']}条聚合输出及原始输出、journal与执行索引。",
        f"实际解析成功{c['parsed']}条，失败{c['unparsed']}条，截断{c['cap_hits']}条。评分与冻结矩阵不一致{len(result['score_mismatches'])}条。", "",
        "完整分任务、子任务、生成种子、输出重复、源码与源文件哈希保存在RAW_AUDIT_RESULTS.json。逐输出脱敏行级证据保存在RAW_AUDIT_ROWS.jsonl.gz。", "",
        "## 成员与种子身份", "", "核对10800个独立成员执行，每个被10个合法异构组合复用。聚合器的随机种子固定42，三个生成重复由成员生成种子42、137、2024定义。成员正确性评分请求为0。", "",
        "相同答案字节作为内容重复单列。请求、上下文、物理attempt、原始记录摘要及成员依赖分别核验，避免把相同短答案直接解释成复制seed。", "",
        "## 权限与成本", "", "原始缓存及E3文件保持只读。其他分区的混合文件字节会被哈希或跳行扫描，其他分区结果载荷的语义解码为0。新增LLM与judge调用均为0。", "",
        f"实际耗时{result['runtime']['elapsed_s']:.2f}秒，进程CPU {result['runtime']['process_cpu_s']:.2f}秒。", "",
        "## 优势与局限", "", "优势为逐输出重跑真实冻结parser，同时核对原输出、原始请求、journal、成员依赖和矩阵列映射。",
        "真值沿用冻结训练投影，官方真值的外部独立重建另需专门审查。NFKC和空白规范化只覆盖文本重复，语义近重复、模型预训练污染及重新GPU推理均未在本模块验证。", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"E:\语言模型组合路由"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = audit(args.root, args.output_dir)
    except Exception as error:
        result = {"schema": "e4_train600_independent_raw_execution_audit_v1", "status": "INCOMPLETE_AUDIT",
                  "error_type": type(error).__name__, "error": str(error), "new_llm_calls": 0}
        (args.output_dir / "RAW_AUDIT_RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        raise
    (args.output_dir / "RAW_AUDIT_RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (args.output_dir / "RAW_AUDIT_NOTES.md").write_text(notes(result), encoding="utf8")
    print(json.dumps({"phase": "COMPLETE", "status": result["status"], "counts": result["counts"],
                      "label_mismatches": len(result["score_mismatches"]), "runtime": result["runtime"]}), flush=True)


if __name__ == "__main__":
    main()
