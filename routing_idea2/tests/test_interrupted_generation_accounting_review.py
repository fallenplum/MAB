"""SOFTWARE ONLY: store acceptance is not proof of actual interrupted usage.

No Torch, GPU, live process, or production execution store is accessed here.
"""
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from budget_router.inference import ModelSpec, GenerationFailure, GenerationResult
from budget_router.member_execution import MemberStore, make_member_job
from budget_router.preparation import validate_success_rows
import budget_router.preparation as preparation
from evaluate_real_e1 import _read_stores, _tokens


def setup_job():
    spec = ModelSpec("SOFTWARE_TEST_ONLY/review", "a" * 40,
                     model_path="SOFTWARE_TEST_ONLY_NO_LOAD", quantization="nf4")
    job = make_member_job(question_id="SOFTWARE_TEST_ONLY_109", task="gsm8k",
                          question="SOFTWARE TEST ONLY: one plus one?",
                          model_alias="REVIEW", seed=42, model_spec=spec,
                          split="e1_diagnostic_evaluation")
    return spec, job


class Backend:
    """Explicit synthetic measurements; never constructs a model or imports Torch."""
    def __init__(self, spec, failure=None):
        self.spec, self.failure, self.calls = spec, failure, 0

    def generate(self, messages, seed):
        self.calls += 1
        assert seed == 42
        if self.failure is not None:
            raise self.failure
        return GenerationResult("Final answer: 2", 127, 64, 19.25,
                                self.spec.model_id, self.spec.revision, seed,
                                self.spec.backend, model_identity_verified=True)


def audit(path, job):
    missing = set()
    expected = {job["question_id"]: {k: job[k] for k in ("task", "subtask", "split")}}
    result = _read_stores([path], expected, missing)
    return result, missing


def test_known_failure_retry_is_two_billed_calls_and_one_success(tmp_path):
    spec, job = setup_job()
    path = tmp_path / "SOFTWARE_TEST_ONLY.jsonl"
    failed = Backend(spec, GenerationFailure("synthetic_known_failure",
                     usage={"prompt_tokens": 127, "completion_tokens": 51}, wall_time_s=2760.125))
    row, cached = MemberStore(path).run(failed, job)
    assert row["status"] == "failure" and not cached
    ok = Backend(spec)
    success, cached = MemberStore(path).run(ok, job)
    assert success["status"] == "success" and not cached
    assert row["attempt_id"] != success["attempt_id"]
    assert row["job_id"] == success["job_id"]
    assert [r["status"] for r in validate_success_rows(path)] == ["failure", "success"]
    (successes, costs, calls, _, _), missing = audit(path, job)
    cost = costs[job["job_id"]]
    assert not missing and len(successes) == 1 and len(calls) == 2
    assert cost == {"calls": 2, "prompt_tokens": 254, "completion_tokens": 115,
                    "wall_time_s": 2779.375, "known": True,
                    "unknown_usage_calls": 0, "unknown_wall_time_calls": 0}
    assert _tokens(cost) == 369
    repeated, cached = MemberStore(path).run(ok, job)
    assert cached and repeated["record_sha256"] == success["record_sha256"]
    assert ok.calls == 1
    assert audit(path, job)[0][1] == costs


@pytest.mark.parametrize("usage", [None, {"prompt_tokens": 127, "completion_tokens": True},
                                  {"prompt_tokens": 127, "completion_tokens": 51.0}])
def test_unknown_usage_blocks_retry_and_does_not_become_known_zero(tmp_path, usage):
    spec, job = setup_job()
    path = tmp_path / "SOFTWARE_TEST_ONLY.jsonl"
    bad = Backend(spec, GenerationFailure("synthetic_unknown_usage", usage=usage, wall_time_s=12.))
    row, _ = MemberStore(path).run(bad, job)
    assert row["cost_status"] == "unknown_must_reconcile"
    ok = Backend(spec)
    with pytest.raises(RuntimeError, match="unknown cost"):
        MemberStore(path).run(ok, job)
    assert ok.calls == 0
    with pytest.raises(ValueError, match="Unreconciled"):
        validate_success_rows(path)
    (_, costs, _, _, _), missing = audit(path, job)
    assert "unreconciled_actual_call_costs" in missing
    assert costs[job["job_id"]]["unknown_usage_calls"] == 1
    assert _tokens(costs[job["job_id"]]) is None


@pytest.mark.parametrize("wall", [None, float("inf")])
def test_known_token_declaration_does_not_reconcile_missing_wall(tmp_path, wall):
    spec, job = setup_job()
    path = tmp_path / "SOFTWARE_TEST_ONLY.jsonl"
    # JSON storage rejects infinity, so absence is the persistable unknown case.
    if wall is not None:
        with pytest.raises(ValueError, match="JSON compliant"):
            MemberStore(path).run(Backend(spec, GenerationFailure("synthetic_unknown_wall",
                usage={"prompt_tokens": 127, "completion_tokens": 51}, wall_time_s=wall)), job)
        return
    row, _ = MemberStore(path).run(Backend(spec, GenerationFailure("synthetic_unknown_wall",
                usage={"prompt_tokens": 127, "completion_tokens": 51}, wall_time_s=wall)), job)
    assert row["cost_status"] == "recorded_actual_usage"
    # Store accepts a valid token declaration without measuring its provenance;
    # E1 independently refuses to classify the total cost as fully reconciled.
    validate_success_rows(path)
    (_, costs, _, _, _), missing = audit(path, job)
    assert "unreconciled_actual_call_costs" in missing
    assert costs[job["job_id"]]["unknown_wall_time_calls"] == 1
    assert _tokens(costs[job["job_id"]]) is None


def test_e1_member_collection_keeps_six_successes_and_preserves_failure(tmp_path, monkeypatch):
    spec, job0 = setup_job()
    qid = job0["question_id"]
    question = {"question_id": qid, "task": "gsm8k", "subtask": None,
                "question": job0["question"]}
    records = [{"question_id": qid, "task": "gsm8k", "subtask": None,
                "split": "e1_diagnostic_evaluation"}]
    pool = [{"alias": "REVIEW" + str(i)} for i in range(6)]
    monkeypatch.setattr(preparation, "load_inputs", lambda root: ({}, records, {qid: question}, pool))
    monkeypatch.setattr(preparation, "read_spec", lambda root, entry: spec)
    failed_path = None
    for i, entry in enumerate(pool):
        job = make_member_job(question_id=qid, task="gsm8k", question=question["question"],
              model_alias=entry["alias"], seed=42, model_spec=spec, split=records[0]["split"])
        path = tmp_path / "results" / "raw" / "e1" / ("members_" + entry["alias"].lower() + ".jsonl")
        if i == 0:
            failed_path = path
            MemberStore(path).run(Backend(spec, GenerationFailure("synthetic_known_failure",
                usage={"prompt_tokens": 127, "completion_tokens": 51}, wall_time_s=12.)), job)
        MemberStore(path).run(Backend(spec), job)
    answers, provenance = preparation.collect_members(tmp_path, "e1", records, pool)
    assert set(answers[qid]) == {p["alias"] for p in pool}
    assert len(provenance[qid]) == 6
    assert len(validate_success_rows(failed_path)) == 2


def test_full_e1_assembler_preserves_known_member_failure_without_changing_scores(tmp_path_factory):
    import json
    import test_real_e1_assembly as fixture
    root = fixture.corpus.__wrapped__(tmp_path_factory)
    source = root / "results/raw/e1/members_m0.jsonl"
    first = next(json.loads(line) for line in source.read_text("utf8").splitlines()
                 if json.loads(line)["split"] == "e1_diagnostic_evaluation")
    failed = {k: v for k, v in first.items() if k not in {"record_sha256", "result"}}
    failed.update(status="failure", attempt_id="SOFTWARE_TEST_ONLY_known_interrupt",
                  cost_status="recorded_actual_usage",
                  actual_usage={"prompt_tokens": 31, "completion_tokens": 7},
                  failure_wall_time_s=12.5)
    failure_path = root / "SOFTWARE_TEST_ONLY_failure.jsonl"
    fixture.write_rows(failure_path, [fixture.signed(failed)])
    member_paths = [failure_path] + [root / ("results/raw/e1/members_m" + str(i) + ".jsonl") for i in range(6)]
    baseline = fixture.evaluate(root)
    report = fixture.evaluate(root, member_stores=member_paths)
    assert report["status"] == "SOFTWARE_TEST_ONLY" and not report["research_complete"]
    assert report["missing_requirements"] == baseline["missing_requirements"]
    assert report["diagnostic_status"] == baseline["diagnostic_status"]
    assert report["physical_diagnostic_cost"]["unique_calls"] == 8801
    assert report["physical_diagnostic_cost"]["total_tokens"] == 39238
    assert report["physical_diagnostic_cost"]["known_call_wall_time_sum_s"] == pytest.approx(892.5)
    assert report["logical_standalone_service_cost"]["sum_calls"] == 24010
    assert report["logical_standalone_service_cost"]["sum_tokens"] == 84380
    assert report["physical_diagnostic_cost"]["by_kind"]["ordinary_members"]["unique_calls"] == 1801
    assert report["metrics"]["best_fixed_calibration"]["mean"] == 0
    assert report["metrics"]["diagnostic_oracle_upper_reference"]["mean"] == 1
    assert report["coverage"]["missing_members"] == report["coverage"]["missing_aggregates"] == 0


def test_e0_collects_actual_synthetic_1800_successes_binds_failure_and_keeps_cap_one(tmp_path, monkeypatch):
    import json
    from budget_router import protocol_lock as pl
    from test_preparation import project, put_jsonl
    import test_local_supervisor_lock as fixture
    root = fixture.lock_project.__wrapped__(project.__wrapped__(tmp_path), monkeypatch)
    manifest, records, questions, pool = pl.load_inputs(root)
    e1 = [r for r in records if r["split"] in {"e1_selection_calibration", "e1_diagnostic_evaluation"}]
    monkeypatch.setattr(preparation, "load_inputs", lambda root: (manifest, records, questions, pool))
    # Restore the actual collector; the imported qualification fixture stubs it.
    monkeypatch.setattr(pl, "collect_members", preparation.collect_members)
    first_path = None
    for mi, entry in enumerate(pool):
        spec = preparation.read_spec(root, entry)
        rows = []
        for qi, record in enumerate(e1):
            q = questions[record["question_id"]]
            job = make_member_job(question_id=q["question_id"], task=q["task"], subtask=q["subtask"],
                  question=q["question"], model_alias=entry["alias"], seed=42, model_spec=spec,
                  split=record["split"], sample_index=0)
            success = fixture.software_result(job, spec)
            if mi == qi == 0:
                failed = {k: v for k, v in success.items() if k not in {"record_sha256", "result"}}
                failed.update(status="failure", attempt_id="SOFTWARE_TEST_ONLY_e0_failure",
                              actual_usage={"prompt_tokens": 31, "completion_tokens": 7}, failure_wall_time_s=12.5)
                failed["record_sha256"] = fixture._digest(failed)
                rows.append(failed)
            rows.append(success)
        path = root / "results/raw/e1" / ("members_" + entry["alias"].lower() + ".jsonl")
        put_jsonl(path, rows)
        if mi == 0:
            first_path = path
    answers, _ = preparation.collect_members(root, "e1", e1, pool)
    assert len(answers) == 300 and all(len(v) == 6 for v in answers.values())
    assert len(validate_success_rows(first_path)) == 301
    lock = pl.freeze_e0(root)
    config = json.loads((root / "configs/experiment_v1.json").read_text("utf8"))
    assert config["budget"]["member_call_cap"] == config["budget"]["aggregator_call_cap"] == 1
    assert len(lock["member_outputs"]) == 6
    assert lock["member_outputs"][0]["sha256"] == pl.file_sha(first_path)
    assert pl.freeze_e0(root) == lock
    # No fixture weights are real; qualification and reference roundtrip are
    # software stubs from the pre-existing lock fixture. Never a production gate.


def test_operator_review_never_touches_processes_or_ledgers_even_with_claimed_approval(tmp_path):
    import hashlib
    import json
    import review_interrupted_call_recovery as operator
    plan = {"schema": "interrupted_call_operator_plan_review_v1", "execution_implemented": False,
            "new_user_authorization_obtained": False, "assessment": "CONDITIONAL_NOT_EXECUTABLE",
            "blocking_proofs": ["atomic_freeze", "gpu_context_retirement"]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf8")
    report = operator.review(path)
    assert not report["execution_ready"] and report["actions_performed"] == []
    with pytest.raises(PermissionError, match="New explicit user"):
        operator.review(path, execute=True)
    auth = {"schema": "explicit_interrupted_call_authorization_v1",
            "plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "explicit_user_approval": True, "user_message_evidence": "SOFTWARE_TEST_ONLY_NOT_ACTUAL_AUTHORIZATION"}
    auth_path = tmp_path / "authorization.json"
    auth_path.write_text(json.dumps(auth), encoding="utf8")
    with pytest.raises(RuntimeError, match="REVIEW_ONLY"):
        operator.review(path, execute=True, authorization=auth_path)
    assert set(p.name for p in tmp_path.iterdir()) == {"plan.json", "authorization.json"}
