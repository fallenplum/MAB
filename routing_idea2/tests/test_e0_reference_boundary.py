"""Synthetic E0 reference-boundary tests; never reads real data or runs a GPU."""
from collections import Counter
import json
from types import SimpleNamespace

import pytest

from budget_router import protocol_lock as pl
from budget_router.scoring import score_answer as real_score_answer
from test_preparation import project
from test_local_supervisor_lock import lock_project, assert_no_transaction


NONSEALED = (
    "qualification", "e1_selection_calibration", "e1_diagnostic_evaluation",
    "online_calibration", "online_stream",
)


class SealedQuestionsTrap(dict):
    def __init__(self, values, sealed_ids):
        super().__init__(values)
        self.sealed_ids = set(sealed_ids)
        self.accessed = []

    def __getitem__(self, key):
        if key in self.sealed_ids:
            raise AssertionError("E0 preflight accessed a sealed question mapping")
        self.accessed.append(key)
        return super().__getitem__(key)


def forbid_weights(*args, **kwargs):
    pytest.fail("Reference preflight must stop before model weight validation")


def assert_failed_freeze_preserved(root, config_before):
    assert not (root / "protocol/E0_LOCK.json").exists()
    assert (root / "configs/experiment_v1.json").read_bytes() == config_before
    assert_no_transaction(root)


def test_sealed_mapping_never_accessed_and_preflight_precedes_weight_validation(lock_project, monkeypatch):
    root = lock_project
    manifest, records, questions, pool = pl.load_inputs(root)
    sealed = {r["question_id"] for r in records if r["split"] == "sealed_test"}
    trapped = SealedQuestionsTrap(questions, sealed)
    monkeypatch.setattr(pl, "load_inputs", lambda *a, **k: (manifest, records, trapped, pool))
    events = []

    def score(*args, **kwargs):
        events.append("reference")
        return SimpleNamespace(parsed=True, score=1)

    monkeypatch.setattr(pl, "score_answer", score)
    monkeypatch.setattr(pl, "validate_local_model_source", lambda *a, **k: events.append("weights"))
    lock = pl.freeze_e0(root)
    expected = Counter(r["split"] for r in records if r["split"] in NONSEALED)
    count = sum(expected.values())
    assert events == ["reference"] * count + ["weights"] * 7
    assert not (sealed & set(trapped.accessed))
    assert lock["scorer_version"] == pl.SCORER_VERSION
    audit = lock["reference_preflight"]
    assert audit["status"] == "PASS_NON_SEALED_REFERENCE_ROUNDTRIP"
    assert audit["validated_splits"] == list(NONSEALED)
    assert audit["validated_questions_by_split"] == dict(expected)
    assert audit["validated_questions"] == 1108 == count
    assert audit["deferred_splits"] == ["sealed_test"]
    assert audit["deferred_questions"] == 200 == len(sealed)
    assert audit["sealed_questions_accessed_by_preflight"] is False
    assert audit["deferred_validation_stage"] == "released_final_grading_after_final_release_and_frozen_decisions"
    assert pl.freeze_e0(root) == lock
    assert_no_transaction(root)


@pytest.mark.parametrize("split", NONSEALED)
def test_invalid_nonsealed_reference_stops_before_weights_and_preserves_config(lock_project, monkeypatch, split):
    root = lock_project
    _, records, questions, _ = pl.load_inputs(root)
    chosen = next(r for r in records if r["split"] == split and r["task"] == "gsm8k")
    sentinel = "SYNTHETIC_INVALID_NUMERIC_REFERENCE"
    questions[chosen["question_id"]]["reference"] = sentinel
    calls = []

    def score(task, prediction, reference, **kwargs):
        calls.append(reference)
        if reference == sentinel:
            return real_score_answer(task, prediction, reference, **kwargs)
        return SimpleNamespace(parsed=True, score=1)

    monkeypatch.setattr(pl, "score_answer", score)
    monkeypatch.setattr(pl, "validate_local_model_source", forbid_weights)
    before = (root / "configs/experiment_v1.json").read_bytes()
    with pytest.raises(ValueError, match="Invalid reference answer"):
        pl.freeze_e0(root)
    assert sentinel in calls
    assert_failed_freeze_preserved(root, before)


def test_unknown_split_fails_before_question_lookup_or_weights(lock_project, monkeypatch):
    root = lock_project
    manifest, records, questions, pool = pl.load_inputs(root)
    unknown = {"split": "unreviewed_split", "question_id": "MUST_NOT_ACCESS_UNKNOWN_MAPPING"}
    trapped = SealedQuestionsTrap(questions, {unknown["question_id"]})
    monkeypatch.setattr(pl, "load_inputs", lambda *a, **k: (manifest, [unknown] + records, trapped, pool))
    monkeypatch.setattr(pl, "validate_local_model_source", forbid_weights)
    monkeypatch.setattr(pl, "score_answer", lambda *a, **k: pytest.fail("Unknown split must stop before scoring"))
    before = (root / "configs/experiment_v1.json").read_bytes()
    with pytest.raises(ValueError, match="Unknown split in E0 reference preflight"):
        pl.freeze_e0(root)
    assert trapped.accessed == []
    assert_failed_freeze_preserved(root, before)


def test_incomplete_member_coverage_stops_before_reference_preflight(lock_project, monkeypatch):
    root = lock_project
    monkeypatch.setattr(pl, "collect_members", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(pl, "score_answer", lambda *a, **k: pytest.fail("Incomplete member coverage reached reference scoring"))
    monkeypatch.setattr(pl, "validate_local_model_source", forbid_weights)
    before = (root / "configs/experiment_v1.json").read_bytes()
    with pytest.raises(ValueError, match="Complete real member coverage required"):
        pl.freeze_e0(root)
    assert_failed_freeze_preserved(root, before)


@pytest.mark.parametrize("version_case", ["mismatch", "missing"])
def test_actual_config_scorer_version_mismatch_stops_before_references_and_weights(lock_project, monkeypatch, version_case):
    root = lock_project
    path = root / "configs/experiment_v1.json"
    config = json.loads(path.read_text("utf8"))
    if version_case == "mismatch":
        config["scoring"]["version"] = "SYNTHETIC_WRONG_VERSION_" + pl.SCORER_VERSION
    else:
        del config["scoring"]["version"]
    path.write_text(json.dumps(config), encoding="utf8")
    before = path.read_bytes()
    monkeypatch.setattr(pl, "score_answer", lambda *a, **k: pytest.fail("Version mismatch reached reference scoring"))
    monkeypatch.setattr(pl, "validate_local_model_source", forbid_weights)
    with pytest.raises(ValueError, match="Experiment scoring version differs from current scorer"):
        pl.freeze_e0(root)
    assert_failed_freeze_preserved(root, before)
