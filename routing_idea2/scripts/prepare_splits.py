"""Freeze score-blind, task-balanced question membership; no inference or label scoring."""
from pathlib import Path
import argparse
import ast
import collections
import copy
import hashlib
import json
import os
import random
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261030
N, K = 6, 3
SPLIT_QUOTAS = {
    "qualification": 4,
    "e1_selection_calibration": 50,
    "e1_diagnostic_evaluation": 100,
    "online_calibration": 100,
    "online_stream": 300,
    "sealed_test": 100,
}
SPLIT_ORDER = list(SPLIT_QUOTAS)
ROW_FIELDS = {"question_id", "task", "subtask", "split", "source_split"}

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_question(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())

def canonical_qid(text):
    # No task or source-split prefix: catches cross-task/source duplicate questions.
    return hashlib.sha256(canonical_question(text).encode("utf-8")).hexdigest()

def local_seed(seed, label):
    return int.from_bytes(hashlib.sha256((str(seed) + "\n" + label).encode("utf-8")).digest()[:8], "big")

def project_record(raw):
    # Explicit allowlist; never access a reference answer or any model score.
    qid, task, question = raw.get("question_id"), raw.get("task"), raw.get("question")
    split, subtask = raw.get("source_split"), raw.get("subtask")
    if not isinstance(qid, str) or not qid or task not in {"gsm8k", "bbh"}:
        raise ValueError("Invalid input question ID/task")
    if not isinstance(question, str) or not canonical_question(question) or not isinstance(split, str):
        raise ValueError("Missing question text/source split")
    if task == "bbh" and (not isinstance(subtask, str) or not subtask):
        raise ValueError("BBH subtask must be officially verified before splitting")
    outputs = raw.get("member_outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("Invalid member-output container")
    return dict(question_id=qid, task=task, question=question, subtask=subtask,
                source_split=split, member_keys=tuple(sorted(outputs.keys())))

def assert_unique_inputs(records):
    qids, canonical = {}, {}
    for row in records:
        qid, key = row["question_id"], canonical_qid(row["question"])
        if qid in qids:
            raise ValueError("Duplicate original question ID: " + qid)
        if key in canonical:
            raise ValueError("NFKC/whitespace duplicate question across input pools: " + qid + " / " + canonical[key])
        qids[qid] = row
        canonical[key] = qid
    return qids

def output_row(row, split):
    return dict(question_id=row["question_id"], task=row["task"], subtask=row["subtask"],
                split=split, source_split=row["source_split"])

def validate_assignments(assignments, records):
    originals = assert_unique_inputs(records)
    selected_ids, selected_canonical = set(), set()
    counts = collections.Counter()
    bbh_subtasks = sorted({r["subtask"] for r in records if r["task"] == "bbh"})
    subcounts = {s: collections.Counter({sub: 0 for sub in bbh_subtasks}) for s in SPLIT_ORDER}
    global_bbh = collections.Counter({sub: 0 for sub in bbh_subtasks})
    for row in assignments:
        if set(row) != ROW_FIELDS:
            raise AssertionError("Manifest record fields must exclude text, references and scores")
        if row["question_id"] not in originals:
            raise AssertionError("Unknown original question ID")
        source = originals[row["question_id"]]
        if any(row[k] != source[k] for k in ("task", "subtask", "source_split")):
            raise AssertionError("Assignment source metadata changed")
        if row["question_id"] in selected_ids:
            raise AssertionError("Question ID appears in more than one split")
        key = canonical_qid(source["question"])
        if key in selected_canonical:
            raise AssertionError("Canonical question crosses split boundaries")
        selected_ids.add(row["question_id"])
        selected_canonical.add(key)
        counts[(row["task"], row["split"])] += 1
        if row["task"] == "gsm8k":
            expected = "test" if row["split"] == "sealed_test" else "train"
            if row["source_split"] != expected:
                raise AssertionError("GSM8K source train/test boundary violated")
        else:
            subcounts[row["split"]][row["subtask"]] += 1
            global_bbh[row["subtask"]] += 1
    for task in ("gsm8k", "bbh"):
        for split, quota in SPLIT_QUOTAS.items():
            if counts[(task, split)] != quota:
                raise AssertionError("Task-balanced split quota mismatch")
    if len(assignments) != 1308 or len(selected_ids) != 1308 or len(selected_canonical) != 1308:
        raise AssertionError("Frozen total must be 654 questions/task")
    for c in list(subcounts.values()) + [global_bbh]:
        if max(c.values()) - min(c.values()) > 1:
            raise AssertionError("BBH allocation is not balanced across all 27 subtasks")
    return dict(
        total_selected=len(assignments),
        counts={task: {s: counts[(task, s)] for s in SPLIT_ORDER} for task in ("gsm8k", "bbh")},
        original_question_ids_preserved=True,
        canonical_rule="sha256(Unicode NFKC(question), whitespace collapsed); no task/source prefix",
        input_count=len(records), input_unique_canonical_questions=len(originals),
        selected_unique_canonical_questions=len(selected_canonical),
        cross_split_qid_overlap=0, cross_split_canonical_overlap=0,
        bbh_subtask_counts={s: dict(sorted(c.items())) for s, c in subcounts.items()},
        bbh_subtask_total_counts=dict(sorted(global_bbh.items())),
        bbh_split_max_minus_min={s: max(c.values()) - min(c.values()) for s, c in subcounts.items()},
        bbh_total_max_minus_min=max(global_bbh.values()) - min(global_bbh.values()),
    )

def build_assignments(records, seed=SEED):
    assert_unique_inputs(records)
    assignments = []
    gsm = [r for r in records if r["task"] == "gsm8k"]
    if any(r["source_split"] not in {"train", "test"} for r in gsm):
        raise ValueError("Unexpected official GSM8K source split")
    train = sorted((r for r in gsm if r["source_split"] == "train"), key=lambda r: r["question_id"])
    test = sorted((r for r in gsm if r["source_split"] == "test"), key=lambda r: r["question_id"])
    random.Random(local_seed(seed, "gsm8k/train")).shuffle(train)
    random.Random(local_seed(seed, "gsm8k/test")).shuffle(test)
    needed_train = sum(v for s, v in SPLIT_QUOTAS.items() if s != "sealed_test")
    if len(train) < needed_train or len(test) < SPLIT_QUOTAS["sealed_test"]:
        raise ValueError("Insufficient official GSM8K source questions")
    cursor = 0
    for split, quota in SPLIT_QUOTAS.items():
        if split == "sealed_test":
            chosen = test[:quota]
        else:
            chosen = train[cursor:cursor + quota]
            cursor += quota
        assignments.extend(output_row(r, split) for r in chosen)
    groups = collections.defaultdict(list)
    for row in sorted((r for r in records if r["task"] == "bbh"), key=lambda r: r["question_id"]):
        groups[row["subtask"]].append(row)
    if len(groups) != 27:
        raise ValueError("Expected exactly 27 verified BBH subtasks")
    subtasks = sorted(groups)
    for sub in subtasks:
        random.Random(local_seed(seed, "bbh/" + sub)).shuffle(groups[sub])
    tie_order = list(subtasks)
    random.Random(local_seed(seed, "bbh/subtask-tie-order")).shuffle(tie_order)
    rank = {sub: i for i, sub in enumerate(tie_order)}
    global_counts = collections.Counter({sub: 0 for sub in subtasks})
    for split, quota in SPLIT_QUOTAS.items():
        split_counts = collections.Counter({sub: 0 for sub in subtasks})
        for _ in range(quota):
            available = [sub for sub in subtasks if groups[sub]]
            if not available:
                raise ValueError("Insufficient BBH questions")
            sub = min(available, key=lambda x: (global_counts[x], split_counts[x], rank[x]))
            row = groups[sub].pop()
            assignments.append(output_row(row, split))
            global_counts[sub] += 1
            split_counts[sub] += 1
    ordered = []
    for split in SPLIT_ORDER:
        selected = sorted((r for r in assignments if r["split"] == split), key=lambda r: r["question_id"])
        random.Random(local_seed(seed, "mixed-task-listing/" + split)).shuffle(selected)
        ordered.extend(selected)
    verification = validate_assignments(ordered, records)
    return ordered, verification

def meaningful_self_tests():
    models = ["fixture_model_" + str(i) for i in range(6)]
    raw = []
    for source_split, amount in (("train", 600), ("test", 160)):
        for i in range(amount):
            raw.append(dict(question_id="gsm8k:" + source_split + ":" + str(i), task="gsm8k", subtask=None,
                            question="Fixture gsm8k " + source_split + " item " + str(i),
                            source_split=source_split, reference="hidden", member_outputs={}))
    for sub in range(27):
        for i in range(30):
            raw.append(dict(question_id="bbh:" + str(sub) + ":" + str(i), task="bbh",
                            subtask="fixture_subtask_" + str(sub), source_split="test",
                            question="Fixture bbh subtask " + str(sub) + " item " + str(i),
                            reference="hidden", member_outputs={m: {"score": 0} for m in models}))
    projected = [project_record(r) for r in raw]
    first, stats = build_assignments(projected)
    reordered, _ = build_assignments(list(reversed(projected)))
    assert first == reordered, "Membership changed when input file order changed"
    altered = copy.deepcopy(raw)
    for r in altered:
        r["reference"] = "arbitrarily changed hidden answer"
        for value in r["member_outputs"].values():
            value["score"] = 999
    blind, _ = build_assignments([project_record(r) for r in altered])
    assert first == blind, "Assignment accessed reference answers/model scores"
    changed, _ = build_assignments(projected, SEED + 1)
    assert first != changed, "Fixture did not exercise seed-specific randomization"
    forged = copy.deepcopy(first)
    source = next(r for r in forged if r["task"] == "gsm8k" and r["split"] == "sealed_test")
    source["source_split"] = "train"
    try:
        validate_assignments(forged, projected)
    except AssertionError:
        pass
    else:
        raise AssertionError("Forbidden GSM8K source split mutation passed")
    duplicate = dict(projected[0])
    duplicate["question_id"] = "different_original_id"
    duplicate["question"] = "".join(chr(ord(c) + 0xFEE0) if 33 <= ord(c) <= 126 else c for c in duplicate["question"])
    try:
        build_assignments(projected + [duplicate])
    except ValueError:
        pass
    else:
        raise AssertionError("NFKC-equivalent question crossed input pools")
    return dict(status="PASS", synthetic_only=True, checks=[
        "Exact quotas: 654/task and 1308 total",
        "Original-ID and NFKC-canonical disjointness",
        "Deterministic under reversed source record order",
        "Changing references/model scores leaves assignments identical",
        "Different seed changes deterministic assignment",
        "GSM8K official test cannot be reassigned as training source",
        "NFKC-equivalent questions with different original IDs fail closed",
        "BBH all 27 subtasks balanced per split and globally to max-min <= 1",
    ])

def checked_runtime_split_names():
    source = ROOT / "src/budget_router/execution.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    found = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ALLOWED_SPLITS" for t in node.targets):
            found = ast.literal_eval(node.value)
    if found != set(SPLIT_QUOTAS):
        raise ValueError("Frozen names differ from execution.ALLOWED_SPLITS")
    return dict(path=source.relative_to(ROOT).as_posix(),sha256=sha_file(source),allowed_splits=sorted(found))

def read_sources():
    specifications = [
        ("data/derived/available_records.jsonl", "artifacts/audit/llmrouterbench/available_records_receipt.json", "bbh"),
        ("data/derived/gsm8k_questions.jsonl", "artifacts/audit/gsm8k_source_audit.json", "gsm8k"),
    ]
    pool_path = ROOT / "artifacts/audit/llmrouterbench/selected_pool_audit.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    expected_models = set(pool["selected_model_aliases"])
    if len(expected_models) != 6:
        raise ValueError("Selected cache pool must have six models")
    records, sources = [], []
    coverage = collections.defaultdict(collections.Counter)
    for relative, receipt_relative, task in specifications:
        path, receipt_path = ROOT / relative, ROOT / receipt_relative
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        actual = sha_file(path)
        if actual != receipt["sha256"]:
            raise ValueError("Input differs from the verified source preparation receipt: " + relative)
        count = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = project_record(json.loads(line))
                if row["task"] != task:
                    raise ValueError("Unexpected task in normalized source")
                if task == "bbh" and set(row["member_keys"]) != expected_models:
                    raise ValueError("BBH question lacks complete selected-pool coverage")
                coverage[task][len(row["member_keys"])] += 1
                records.append(row)
                count += 1
        sources.append(dict(path=relative,bytes=path.stat().st_size,sha256=actual,records=count,
                            receipt_path=receipt_relative,receipt_sha256=sha_file(receipt_path)))
    sources.append(dict(path=pool_path.relative_to(ROOT).as_posix(),sha256=sha_file(pool_path),role="Frozen-before-E1 candidate aliases and data-quality exclusions"))
    return records, sources, dict(model_aliases=sorted(expected_models),
        source_member_coverage={task:{str(k):v for k,v in sorted(c.items())} for task,c in coverage.items()},
        bbh_cached_members_per_question=6,gsm8k_member_outputs_missing=True,
        unknown_published_cache_weight_revisions=True)

def atomic_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    os.replace(temporary,path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-only",action="store_true")
    args = parser.parse_args()
    tests = meaningful_self_tests()
    if args.self_test_only:
        print(json.dumps(tests,ensure_ascii=False))
        return
    names = checked_runtime_split_names()
    records,sources,coverage = read_sources()
    assignments,checks = build_assignments(records)
    selected = {r["question_id"] for r in assignments}
    source_by_id = {r["question_id"]:r for r in records}
    selected_coverage = {
        task:dict(collections.Counter(str(len(source_by_id[r["question_id"]]["member_keys"])) for r in assignments if r["task"]==task))
        for task in ("gsm8k","bbh")
    }
    manifest = dict(schema_version="question-split-v1",seed=SEED,N=N,K=K,
        scientific_status="SPLIT_MEMBERSHIP_FROZEN; GSM8K_MEMBER_OUTPUTS_MISSING; E0_NOT_COMPLETE",
        quotas_per_task=SPLIT_QUOTAS,e1_total_questions=300,total_selected=1308,
        record_order="Within each split, deterministic seed-derived shuffle across both tasks; all records preserve original IDs.",
        selection_policy="Fixed seed, source membership and BBH subtask balance only. Reference answers and model scores are never accessed.",
        canonical_id_rule=checks["canonical_rule"],source_files=sources,pool_coverage=coverage,
        selected_pool_coverage=selected_coverage,bbh_subtask_allocation=checks["bbh_subtask_counts"],
        canonical_question_fingerprints={qid:canonical_qid(source_by_id[qid]["question"]) for qid in sorted(selected)},
        canonical_fingerprint_rule="sha256(UTF-8 of Unicode NFKC(question) with whitespace collapsed); task and source_split excluded",
        records=assignments)
    output = ROOT/"data/derived/split_manifest.json"
    atomic_json(output,manifest)
    on_disk = json.loads(output.read_text(encoding="utf-8"))
    assert on_disk == manifest
    validate_assignments(on_disk["records"],records)
    audit = dict(status="PASS_SPLIT_MEMBERSHIP; E0_GENERATION_STILL_MISSING",seed=SEED,
        manifest_path=output.relative_to(ROOT).as_posix(),manifest_sha256=sha_file(output),
        manifest_bytes=output.stat().st_size,source_files=sources,execution_split_contract=names,
        checks=checks,self_tests=tests,selected_pool_coverage=selected_coverage,
        source_input_counts=dict(collections.Counter(r["task"] for r in records)),
        unselected_source_counts=dict(collections.Counter(r["task"] for r in records if r["question_id"] not in selected)),
        output_record_fields=sorted(ROW_FIELDS),contains_question_text=False,
        contains_reference_answers=False,contains_model_scores=False,
        raw_cache_uploaded=False,no_new_model_inference=True,E0_complete=False,
        stage_boundaries="qualification and E1 selection-calibration are disjoint from E1 diagnostic evaluation; online-calibration, online-stream and sealed-test membership never overlap.")
    atomic_json(ROOT/"artifacts/audit/split_audit.json",audit)
    print(json.dumps({k:audit[k] for k in ("status","manifest_path","manifest_sha256","manifest_bytes","selected_pool_coverage","E0_complete")},ensure_ascii=False))
    print(json.dumps(checks["counts"],ensure_ascii=False))

if __name__ == "__main__":
    main()
