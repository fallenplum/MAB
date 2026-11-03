"""Preparation-layer regressions using synthetic records only; no downloads or model loading."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import importlib.util
import json
import sys
import unicodedata
import pytest

from budget_router import preparation as prep
from budget_router.inference import ModelSpec, configuration_fingerprint
from budget_router.execution import _digest, validate_job
from budget_router.member_execution import make_member_job

PROJECT = Path(__file__).resolve().parents[1]
QUOTAS = {"qualification": 4, "e1_selection_calibration": 50, "e1_diagnostic_evaluation": 100,
          "online_calibration": 100, "online_stream": 300, "sealed_test": 100}
BBH_SUBTASKS = ["boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
    "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate", "object_counting", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting"]
GOLD = "DO_NOT_SERIALIZE_GOLD_SENTINEL_987654321"

def put_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def put_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")

def canonical(question):
    text = " ".join(unicodedata.normalize("NFKC", question).split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def refresh_manifest_audit(root, manifest=None):
    path = root / "data/derived/split_manifest.json"
    if manifest is not None:
        put_json(path, manifest)
    put_json(root / "artifacts/audit/split_audit.json", {"manifest_sha256": prep.file_sha(path), "synthetic_test": True})

@pytest.fixture
def project(tmp_path):
    root = tmp_path
    aliases = ["member_" + str(i) for i in range(6)]
    pool = []
    for i, alias in enumerate(aliases):
        spec = ModelSpec("fixture/" + alias, "a" * 40, model_path=str(root / "models" / alias), quantization="nf4")
        config = "configs/" + alias + ".json"
        put_json(root / config, asdict(spec))
        pool.append(dict(index=i, alias=alias, config=config, model_id=spec.model_id, revision=spec.revision,
                         fingerprint=configuration_fingerprint(spec), historical_cache_alias="old_" + alias))
    put_json(root / "configs/pool_v1.json", {"status": "SYNTHETIC_ONLY", "models": pool})
    agg = ModelSpec("fixture/aggregator", "b" * 40, model_path=str(root / "models/aggregator"), quantization="nf4")
    put_json(root / "configs/aggregator_v1.json", asdict(agg))
    assignments, questions, fingerprints = [], {"gsm8k": [], "bbh": []}, {}
    for task in ("gsm8k", "bbh"):
        serial = 0
        for split, count in QUOTAS.items():
            for j in range(count):
                qid = task + ":synthetic:" + str(serial)
                subtask = BBH_SUBTASKS[serial % 27] if task == "bbh" else None
                source_split = "test" if task == "bbh" or split == "sealed_test" else "train"
                question = "Synthetic private question " + qid + " Unicode \u2028 separator \u0085 retained?"
                assignments.append(dict(question_id=qid, task=task, subtask=subtask, split=split, source_split=source_split))
                fingerprints[qid] = canonical(question)
                historical = {"old_" + alias: dict(raw_output="Historical answer " + alias, score=999999,
                                prompt_tokens=10, completion_tokens=4, cost=123.0,
                                source_file="data/raw/synthetic/" + alias + ".json") for alias in aliases}
                questions[task].append(dict(question_id=qid, task=task, subtask=subtask, question=question,
                    reference=GOLD, source_split=source_split, member_outputs=historical if task == "bbh" else {}))
                serial += 1
    sources = []
    for filename, receipt_name, task in (("available_records.jsonl", "available_source.json", "bbh"),
                                        ("gsm8k_questions.jsonl", "gsm_source.json", "gsm8k")):
        relative = "data/derived/" + filename
        receipt_relative = "artifacts/audit/" + receipt_name
        put_jsonl(root / relative, questions[task])
        put_json(root / receipt_relative, {"synthetic_only": True, "sha256": prep.file_sha(root / relative)})
        sources.append(dict(path=relative, sha256=prep.file_sha(root / relative),
                            receipt_path=receipt_relative, receipt_sha256=prep.file_sha(root / receipt_relative)))
    manifest = dict(schema_version="question-split-v1", seed=20261030, N=6, K=3,
        records=assignments, canonical_question_fingerprints=fingerprints, source_files=sources)
    refresh_manifest_audit(root, manifest)
    return root

class CharTokenizer:
    name_or_path = "synthetic-tokenizer"
    init_kwargs = {"revision": "b" * 40}
    def encode(self, text, **kwargs):
        return [ord(x) for x in text]
    def decode(self, tokens, **kwargs):
        return "".join(chr(x) for x in tokens)

@pytest.fixture
def fake_tokenizer(monkeypatch):
    factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: CharTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=factory))

def gate_fixture(root):
    aggregator = ModelSpec(**json.loads((root / "configs/aggregator_v1.json").read_text("utf-8")))
    lock = root / "protocol/E0_LOCK.json"
    put_json(lock, {"status": "SYNTHETIC_E0_TEST_ONLY", "pool_config_sha256": prep.file_sha(root / "configs/pool_v1.json")})
    gate = dict(status="PASS", synthetic_only=True, evidence={
        "split_manifest_sha256": prep.file_sha(root / "data/derived/split_manifest.json"),
        "aggregator_revision": aggregator.revision,
        "aggregator_fingerprint": configuration_fingerprint(aggregator),
        "e0_lock_sha256": prep.file_sha(lock)})
    put_json(root / "results/e1/e1_diagnostics.json", gate)
    return gate

def member_output_fixture(root, stage="qualification"):
    _, selected, questions, pool = prep.select_records(root, stage)
    for entry in pool:
        spec = prep.read_spec(root, entry)
        rows = []
        for assigned in selected:
            source = questions[assigned["question_id"]]
            job = make_member_job(question_id=source["question_id"], task=source["task"], subtask=source["subtask"],
                question=source["question"], model_alias=entry["alias"], seed=42, model_spec=spec,
                split=assigned["split"], sample_index=0)
            keys = ("job_id", "question_id", "task", "subtask", "split", "model_alias", "sample_index", "seed",
                    "model_fingerprint", "member_mode", "prompt_version", "source_question_sha256")
            row = {k: job[k] for k in keys}
            row.update(status="success", cost_status="recorded_actual_usage", synthetic_only=True,
                result=dict(text="Synthetic served answer " + entry["alias"], prompt_tokens=31, completion_tokens=7,
                    wall_time_s=0.01, model_id=spec.model_id, model_revision=spec.revision, seed=42,
                    backend=spec.backend, model_identity_verified=True))
            row["record_sha256"] = _digest(row)
            rows.append(row)
        put_jsonl(root / "results/raw" / stage / ("members_" + entry["alias"].lower() + ".jsonl"), rows)
    return selected, pool

def mutate_first_member(root, mutator, stage="qualification"):
    path = root / "results/raw" / stage / "members_member_0.jsonl"
    rows = prep.read_jsonl(path)
    mutator(rows[0])
    rows[0]["record_sha256"] = _digest({k:v for k,v in rows[0].items() if k!="record_sha256"})
    put_jsonl(path, rows)

def test_load_inputs_preserves_legal_unicode_jsonl_and_original_ids(project):
    manifest, rows, questions, pool = prep.load_inputs(project)
    assert len(rows) == len(questions) == 1308
    assert len(pool) == 6
    assert all("\u2028" in q["question"] and "\u0085" in q["question"] for q in questions.values())
    assert set(questions) == set(manifest["canonical_question_fingerprints"])

@pytest.mark.parametrize("kind", ["manifest", "source", "receipt", "canonical"])
def test_load_inputs_fails_closed_on_manifest_source_receipt_and_content_drift(project, kind):
    manifest_path = project / "data/derived/split_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    source_path = project / "data/derived/available_records.jsonl"
    if kind == "manifest":
        manifest["seed"] = 123
        put_json(manifest_path, manifest)
    elif kind == "receipt":
        put_json(project / manifest["source_files"][0]["receipt_path"], {"changed": True})
    else:
        rows = prep.read_jsonl(source_path)
        rows[0]["question"] += " changed"
        put_jsonl(source_path, rows)
        if kind == "canonical":
            manifest["source_files"][0]["sha256"] = prep.file_sha(source_path)
            refresh_manifest_audit(project, manifest)
    with pytest.raises(ValueError):
        prep.load_inputs(project)

def test_read_spec_binds_full_base_fingerprint_before_resampling(project):
    _, _, _, pool = prep.load_inputs(project)
    base = prep.read_spec(project, pool[0])
    sampled = prep.read_spec(project, pool[0], sampling=True)
    assert sampled.temperature == 0.7 and sampled.top_p == 0.95
    assert sampled.model_id == base.model_id
    config = project / pool[0]["config"]
    changed = asdict(base)
    changed["max_new_tokens"] += 1
    put_json(config, changed)
    with pytest.raises(ValueError, match="fingerprint"):
        prep.read_spec(project, pool[0], sampling=True)

def test_all_stage_names_and_exact_task_counts_and_sealed_guard(project):
    gate_fixture(project)
    for stage, count in (("qualification", 8), ("e1", 300), ("online_calibration", 200), ("online_stream", 600)):
        _, rows, _, _ = prep.select_records(project, stage)
        assert len(rows) == count
        assert sum(r["task"] == "gsm8k" for r in rows) == count // 2
    with pytest.raises(ValueError, match="sealed"):
        prep.select_records(project, "sealed_test")

@pytest.mark.parametrize("kind", ["task", "calibration_evaluation"])
def test_e1_rejects_wrong_task_or_cal_eval_quota_even_with_300_total(project, kind):
    path = project / "data/derived/split_manifest.json"
    m = json.loads(path.read_text("utf-8"))
    if kind == "task":
        bbh = next(r for r in m["records"] if r["task"]=="bbh" and r["split"]=="e1_diagnostic_evaluation")
        gsm = next(r for r in m["records"] if r["task"]=="gsm8k" and r["split"]=="qualification")
        bbh["split"], gsm["split"] = "qualification", "e1_diagnostic_evaluation"
    else:
        for task in ("gsm8k", "bbh"):
            row = next(r for r in m["records"] if r["task"]==task and r["split"]=="e1_selection_calibration")
            row["split"]="e1_diagnostic_evaluation"
    refresh_manifest_audit(project, m)
    with pytest.raises(ValueError, match="per-task/per-split"):
        prep.select_records(project, "e1")

@pytest.mark.parametrize("kind", ["partial_status", "split", "aggregator_configuration", "e0_lock", "pool"])
def test_online_gate_requires_exact_pass_and_current_split_full_config_lock_and_pool(project, kind):
    gate = gate_fixture(project)
    gate_path = project / "results/e1/e1_diagnostics.json"
    if kind == "partial_status":
        gate["status"]="PARTIAL"
        gate["gate"]={"status":"PASS"}
        put_json(gate_path, gate)
    elif kind == "split":
        gate["evidence"]["split_manifest_sha256"]="0"*64
        put_json(gate_path, gate)
    elif kind == "aggregator_configuration":
        path=project/"configs/aggregator_v1.json"
        spec=json.loads(path.read_text("utf-8"));spec["max_new_tokens"]+=1
        put_json(path,spec)
    elif kind == "e0_lock":
        gate["evidence"]["e0_lock_sha256"]="0"*64
        put_json(gate_path,gate)
    else:
        path=project/"configs/pool_v1.json";pool=json.loads(path.read_text("utf-8"));pool["changed_after_lock"]=True
        put_json(path,pool)
    with pytest.raises(ValueError):
        prep.select_records(project,"online_stream")

def test_collect_members_accepts_only_full_six_real_job_bound_outputs(project):
    records, pool=member_output_fixture(project)
    answers, provenance=prep.collect_members(project,"qualification",records,pool)
    assert len(answers)==8 and all(len(v)==6 for v in answers.values())
    assert all(len(v)==6 for v in provenance.values())
    path=project/"results/raw/qualification/members_member_0.jsonl"
    put_jsonl(path,prep.read_jsonl(path)[:-1])
    with pytest.raises(ValueError,match="All six"):
        prep.collect_members(project,"qualification",records,pool)

@pytest.mark.parametrize("field", ["job_id","split","seed","source_question_sha256","prompt_version","member_mode"])
def test_member_metadata_and_original_job_identity_cannot_drift_under_rehashed_record(project,field):
    records,pool=member_output_fixture(project)
    def change(row):
        row[field]=999 if field=="seed" else ("sealed_test" if field=="split" else "changed")
    mutate_first_member(project,change)
    with pytest.raises(ValueError):
        prep.collect_members(project,"qualification",records,pool)

@pytest.mark.parametrize("field", ["model_revision","model_identity_verified","prompt_tokens","completion_tokens","text"])
def test_member_result_identity_usage_and_nonempty_text_are_required(project,field):
    records,pool=member_output_fixture(project)
    def change(row):
        row["result"][field]={"model_revision":"f"*40,"model_identity_verified":False,"prompt_tokens":"missing",
                             "completion_tokens":999999,"text":""}[field]
    mutate_first_member(project,change)
    with pytest.raises(ValueError):
        prep.collect_members(project,"qualification",records,pool)

def test_member_success_cannot_hide_unknown_cost(project):
    records,pool=member_output_fixture(project)
    mutate_first_member(project,lambda row:row.update(cost_status="unknown_must_reconcile"))
    with pytest.raises(ValueError):
        prep.collect_members(project,"qualification",records,pool)

def test_terminal_failed_call_must_have_known_real_usage_even_when_all_successes_exist(project):
    records,pool=member_output_fixture(project)
    path=project/"results/raw/qualification/members_member_0.jsonl"
    rows=prep.read_jsonl(path)
    failure=dict(status="failure",job_id="other_attempt",cost_status="recorded_actual_usage",actual_usage=None)
    failure["record_sha256"]=_digest(failure)
    put_jsonl(path,[failure]+rows)
    with pytest.raises(ValueError,match="usage"):
        prep.collect_members(project,"qualification",records,pool)

def test_full_e1_jobs_bind_300_times_20_and_never_copy_references_or_scores(project,fake_tokenizer):
    records,pool=member_output_fixture(project,"e1")
    assert len(records)==300
    receipt=prep.prepare_aggregations(project,"e1")
    jobs=prep.read_jsonl(receipt["path"])
    assert receipt["jobs"]==6000 and len({j["job_id"] for j in jobs})==6000
    for job in jobs:
        validate_job(job,require_token_policy=True)
        assert GOLD not in json.dumps(job,ensure_ascii=False)
        assert len(job["members"])==3 and "score" not in job["member_provenance"]
    counts={}
    for job in jobs:
        counts.setdefault(job["question_id"],set()).add(tuple(job["members"]))
    assert len(counts)==300 and all(len(teams)==20 for teams in counts.values())
    by_split={s:sum(j["split"]==s for j in jobs) for s in ("e1_selection_calibration","e1_diagnostic_evaluation")}
    assert by_split=={"e1_selection_calibration":2000,"e1_diagnostic_evaluation":4000}

def test_historical_outputs_are_qualification_only_and_separate_from_real_jobs(project,fake_tokenizer):
    with pytest.raises(ValueError,match="Historical"):
        prep.prepare_aggregations(project,"e1",qualification_from_historical=True)
    historical=prep.prepare_aggregations(project,"qualification",qualification_from_historical=True)
    assert Path(historical["path"]).name=="historical_runtime_qualification.jsonl"
    jobs=prep.read_jsonl(historical["path"])
    assert len(jobs)==4 and all(j["task"]=="bbh" and j["split"]=="qualification" for j in jobs)
    assert all(p["historical_checkpoint_revision"].startswith("UNKNOWN") for j in jobs for p in j["member_provenance"].values())
    member_output_fixture(project,"qualification")
    formal=prep.prepare_aggregations(project,"qualification")
    assert Path(formal["path"]).name=="aggregations.jsonl" and formal["jobs"]==160
    assert GOLD not in Path(formal["path"]).read_text("utf-8")

def test_aggregator_alone_is_exactly_200_diagnostic_jobs_without_labels(project,fake_tokenizer):
    receipt=prep.prepare_alone(project)
    jobs=prep.read_jsonl(receipt["path"])
    assert len(jobs)==200
    assert all(j["split"]=="e1_diagnostic_evaluation" and not j["members"] for j in jobs)
    assert GOLD not in Path(receipt["path"]).read_text("utf-8")
    for job in jobs:
        validate_job(job,require_token_policy=True)

def load_cli():
    path=PROJECT/"scripts/prepare_jobs.py"
    module_spec=importlib.util.spec_from_file_location("preparation_cli_under_test",path)
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize("args", [
    ["--kind","alone","--stage","qualification"],
    ["--kind","alone","--stage","online_stream"],
    ["--kind","members","--historical-qualification"],
    ["--kind","aggregations","--stage","e1","--historical-qualification"],
])
def test_cli_rejects_stage_or_historical_mode_misuse_before_preparation(project,monkeypatch,args):
    cli=load_cli()
    monkeypatch.setattr(cli,"ROOT",project)
    for method in ("prepare_members","prepare_aggregations","prepare_alone"):
        monkeypatch.setattr(cli,method,lambda *a,**k:pytest.fail("Invalid CLI invoked job preparation"))
    monkeypatch.setattr(sys,"argv",["prepare_jobs"]+args)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code==2

def test_immutable_batches_reuse_identical_bytes_but_never_replace_different_jobs(project):
    path=project/"data/jobs/software_fixture.jsonl"
    rows=[{"question_id":"synthetic","text":"Unicode \u2028 inside legal JSON string"}]
    first=prep.immutable_jsonl(path,rows)
    assert prep.immutable_jsonl(path,rows)==first
    with pytest.raises(ValueError,match="immutable"):
        prep.immutable_jsonl(path,[{"question_id":"different"}])
