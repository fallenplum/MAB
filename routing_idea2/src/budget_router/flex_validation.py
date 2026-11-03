"""Independent paid-information Flex receipts and full E4 technical acceptance.

Original-system reproduction and positive research evidence are never inferred
from completeness. Production rejects software-only assets and partial grids.
"""
from dataclasses import asdict, replace, dataclass
import json
import math
import os
from pathlib import Path
import uuid
import numpy as np

from .stage_validation import (read_json, read_jsonl, atomic_json, immutable_json,
    artifact_reference, verified_reference, canonical_digest, verify_input_identity,
    settings_from_config, validate_protocol_config, episode_specs, episode_name)
from .neighbor_validation import validate_e4_gate
from .simulation import MatrixEnvironment, _read_quality_for_evaluation
from .budget import ResourceCost
from .flex_training import (FLEX_STRATEGY, FLEX_VARIANT, FLEX_PARAMETERS,
    PaidFullWarmupRow, FrozenPCASmoothedFlexPolicy, train_flex_adapted, _training_data)
from .neighbor_policies import LegalMemberFeedback
from .neighbor_simulation import _evaluate_frozen_service

ARTIFACTS = ("summary.json", "traces.jsonl", "warmup_anonymous.json", "policy_checkpoint.json", "cost_curve.csv")
SOURCES = ("src/budget_router/flex_training.py", "src/budget_router/flex_simulation.py",
    "src/budget_router/flex_validation.py", "scripts/run_flex_reference.py",
    "scripts/evaluate_flex_reference.py", "protocol/flex_training_adaptation_manifest.json")
_FINAL_BATCH_TOKEN = object()


def locked_grid(config):
    validate_protocol_config(config)
    seeds = config["statistics"]["algorithm_stream_seeds"]
    rates = config["feedback"]["final_score_budget_per_100_questions"]
    delays = config["feedback"]["delays"]
    if len(seeds) != 5 or len(set(seeds)) != 5 or rates != [10, 25, 50, 100] or delays != [0, 5, 20]:
        raise ValueError("Flex requires the entire original five-seed/four-rate/three-delay grid")
    return {"stage": "flex_reference", "strategies": [FLEX_STRATEGY], "seeds": seeds, "rates": rates, "delays": delays}


def flex_settings(config):
    return replace(settings_from_config(config), warmup_mode="paid_member_prior_then_combo")


def make_identity(root, inputs, *, e3_gate_path, mandatory_e4_gate_path):
    gate, bindings = validate_e4_gate(mandatory_e4_gate_path, inputs)
    e3 = artifact_reference(e3_gate_path)
    if bindings["e3_gate_sha256"] != e3["sha256"] or gate.get("e3_gate", {}).get("sha256") != e3["sha256"]:
        raise ValueError("Flex mandatory E4 and E3 parents differ")
    config = read_json(inputs["assets"]["config"]["path"])
    value = {"schema": "flex_grid_identity_v1", "inputs": inputs, "grid": locked_grid(config),
        "settings": json.loads(json.dumps(asdict(flex_settings(config)))),
        "e3_gate": e3, "mandatory_e4_gate": artifact_reference(mandatory_e4_gate_path),
        "flex_sources": {name: artifact_reference(Path(root)/name) for name in SOURCES},
        "variant": FLEX_VARIANT, "training_parameters": dict(FLEX_PARAMETERS),
        "native_system": "NOT_RUN", "one_pinned_generation_cache": True,
        "information_regime": "paid all-six member warmup plus paid unused combo labels; static online service"}
    value["identity_sha256"] = canonical_digest(value)
    return value


def verify_identity(identity, *, parent=True):
    core = {key: value for key, value in identity.items() if key != "identity_sha256"}
    if identity.get("schema") != "flex_grid_identity_v1" or identity.get("identity_sha256") != canonical_digest(core):
        raise ValueError("Flex immutable grid identity changed")
    verify_input_identity(identity["inputs"])
    if set(identity.get("flex_sources", {})) != set(SOURCES): raise ValueError("Flex independent source closure incomplete")
    for reference in identity["flex_sources"].values(): verified_reference(reference)
    e3 = verified_reference(identity["e3_gate"])
    mandatory = verified_reference(identity["mandatory_e4_gate"])
    if parent:
        gate, bindings = validate_e4_gate(mandatory, identity["inputs"])
        if bindings["e3_gate_sha256"] != artifact_reference(e3)["sha256"]: raise ValueError("Flex upstream E3 mismatch")
    config = read_json(identity["inputs"]["assets"]["config"]["path"])
    if (identity["grid"] != locked_grid(config) or identity["settings"] != json.loads(json.dumps(asdict(flex_settings(config))))
            or identity.get("variant") != FLEX_VARIANT or identity.get("training_parameters") != FLEX_PARAMETERS
            or identity.get("native_system") != "NOT_RUN"):
        raise ValueError("Flex full-scale protocol, paid information or named loss variant changed")
    return identity


def _add(total, cost):
    if set(cost) != {"executions", "labels", "tokens"}: raise ValueError("Missing actual resource fields")
    for key in total:
        value = cost[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0: raise ValueError("Invalid actual resource count")
        total[key] += value


def validate_result(result, spec, identity):
    summary, checkpoint, traces, warmup = (result[key] for key in ("summary", "policy_checkpoint", "traces", "warmup"))
    if summary.get("status") != "COMPLETED_RESTRICTED_FEEDBACK_SIMULATION_ONLY" or checkpoint.get("software_test_only") is not False:
        raise ValueError("Software-only/partial Flex cannot be accepted as production")
    if any(summary.get(key) != value or checkpoint.get(key) != value for key, value in spec.items()): raise ValueError("Flex episode parameters changed")
    if spec.get("strategy") != FLEX_STRATEGY or summary.get("method_variant") != FLEX_VARIANT or checkpoint.get("method_variant") != FLEX_VARIANT:
        raise ValueError("Wrong Flex named adaptation")
    if summary.get("native_system") != "NOT_RUN" or checkpoint.get("native_system") != "NOT_RUN": raise ValueError("Native Flex claim forbidden")
    if summary.get("settings") != identity["settings"] or checkpoint.get("settings") != identity["settings"]: raise ValueError("Flex settings changed")
    if summary.get("n_questions") != 600 or summary.get("all_questions_served") is not True or len(traces) != 600:
        raise ValueError("All original 600 service questions required")
    if len({r["question_id"] for r in traces}) != 600 or [r["step"] for r in traces] != list(range(600)):
        raise ValueError("Flex missing, repeated or reordered stream question")
    if {task: sum(r["task"] == task for r in traces) for task in ("gsm8k", "bbh")} != {"gsm8k": 300, "bbh": 300}:
        raise ValueError("Flex must retain both original 300-question tasks")
    if (checkpoint.get("schema") != "flex_smoothed_frozen_policy_checkpoint_v1" or checkpoint.get("next_tick") != 600
            or checkpoint.get("frozen_as_of_tick") != 599 or checkpoint.get("pending_feedback_flushed") is not False
            or checkpoint.get("online_updates") is not False or summary.get("online_updates") is not False):
        raise ValueError("Missing frozen static Flex checkpoint")
    if checkpoint.get("stream_question_order") != [r["question_id"] for r in traces]: raise ValueError("Flex checkpoint stream order changed")
    FrozenPCASmoothedFlexPolicy.from_state_dict(checkpoint["policy_state"])
    config = read_json(identity["inputs"]["assets"]["config"]["path"])
    settings = flex_settings(config)
    expected_warmup = min(32, settings.warmup_max_labels, 600*spec["label_rate"]//100//5)
    prior = expected_warmup//2
    if len(warmup) != expected_warmup or summary.get("warmup_labels") != expected_warmup:
        raise ValueError("Flex complete paid warmup cap/schedule changed")
    warm_cost = {"executions": 0, "labels": 0, "tokens": 0}
    for position, row in enumerate(warmup):
        member_only = position < prior
        if (row.get("purchase_kind") != ("paid_all_six_member_reference" if member_only else "paid_combo_final_label")
                or row.get("feedback_units") != 1 or row.get("generated_member_labels") != (6 if member_only else 3)
                or row.get("aggregate_labels") != (0 if member_only else 1)
                or (member_only and row.get("combination_index") is not None)):
            raise ValueError("Flex paid information regime changed")
        _add(warm_cost, row["actual_cost"])
    if warm_cost != summary["warmup_cost"] or warm_cost["labels"] != expected_warmup: raise ValueError("Flex warmup cost not reconciled")
    report = checkpoint["policy_state"]["training_report"]
    ids = sorted(row["question_id"] for row in warmup[:prior])
    if (report != summary.get("training_report") or report.get("training_question_ids") != ids
            or report.get("training_rows") != prior or report.get("training_member_labels") != 6*prior
            or report.get("parameters") != FLEX_PARAMETERS or report.get("seed") != spec["seed"]
            or len(report.get("optimization_loss_trace", [])) != 200):
        raise ValueError("Flex training used different rows, supervision or objective")
    if summary.get("paid_member_reference_events") != prior or summary.get("warmup_combo_labels_used_for_training") != 0:
        raise ValueError("Flex extra/unused labels silently entered training")
    total = dict(warm_cost)
    for row in traces:
        if (row.get("alternate_index") is not None or row.get("action_kind") != "none"
                or row.get("requested_combination_labels") != 0 or row.get("policy_updated_after_current_question") is not False
                or row.get("prediction_snapshot_precedes_current_execution_and_feedback") is not True
                or row.get("service_frozen_before_audit") is not True): raise ValueError("Flex static serving/permission boundary changed")
        if row.get("purchase_cost") != {"executions": 0, "labels": 0, "tokens": 0}: raise ValueError("Unnecessary online Flex feedback purchased")
        if not isinstance(row.get("predicted_combination_scores_before_decision"), list) or len(row["predicted_combination_scores_before_decision"]) != 20:
            raise ValueError("Flex predecision diagnostics incomplete")
        if not isinstance(row.get("quality"), (float, int)) or not 0 <= row["quality"] <= 1 or row.get("candidate_coverage") not in (0, 1):
            raise ValueError("Invalid real aggregate quality or candidate coverage")
        if not math.isclose(row["aggregation_loss"], row["candidate_coverage"]-row["quality"], abs_tol=1e-12): raise ValueError("Coverage substituted for aggregate score")
        _add(total, row["service_cost"]); _add(total, row["purchase_cost"])
        if total != row["cumulative_cost"]: raise ValueError("Flex cumulative cost mismatch")
        if row.get("protected_future_service") != asdict(settings.caps.service(False)*(599-row["step"])):
            raise ValueError("Flex remaining normal service was not protected")
    budget = summary["budget"]
    if (total != budget["spent"] or budget != checkpoint["budget"] or total["labels"] != expected_warmup
            or any(total[k] > budget["limits"][k] for k in total) or budget["pending"]
            or budget["future_service"] != {"executions": 0, "labels": 0, "tokens": 0}): raise ValueError("Flex global cost not reconciled")
    if budget["limits"] != asdict(ResourceCost(settings.execution_limit, 600*spec["label_rate"]//100, settings.token_limit)):
        raise ValueError("Flex hard budget differs from original pinned limits")
    for key, expected in {"quality_micro": float(np.mean([r["quality"] for r in traces])),
        "candidate_coverage": float(np.mean([r["candidate_coverage"] for r in traces])),
        "aggregation_loss": float(np.mean([r["aggregation_loss"] for r in traces]))}.items():
        if summary.get(key) != expected: raise ValueError("Flex metric summary differs from its complete question traces")
    for broker_name in ("feedback", "warmup_feedback"):
        broker = checkpoint[broker_name]
        if broker.get("pending_values_serialized") is not False or broker.get("pending_paid_requests_without_values"):
            raise ValueError("Flex checkpoint contains future values or nonexistent pending purchases")
    if checkpoint["feedback"].get("requested"): raise ValueError("Frozen Flex requested online labels")
    if len(checkpoint.get("paid_unused_combo_labels", [])) != expected_warmup-prior: raise ValueError("Paid combo labels were not retained")
    assets, locked = identity["inputs"]["assets"], identity["inputs"]["locked_provenance"]
    for split in ("online", "calibration"):
        required = {"matrix_sha256": assets[split+"_matrix"]["sha256"], "e0_lock_sha256": locked["e0_lock"]["sha256"],
            "e1_gate_sha256": locked["e1_gate"]["sha256"], "pca_sha256": locked["pca_artifact"]["sha256"]}
        for container in (summary, checkpoint):
            if any(container.get(split+"_provenance", {}).get(key) != value for key, value in required.items()):
                raise ValueError("Flex differs from original paid matrices/locks")
    expected_sources = {name: identity["flex_sources"][name]["sha256"] for name in (
        "src/budget_router/flex_training.py", "src/budget_router/flex_simulation.py", "protocol/flex_training_adaptation_manifest.json")}
    for container in (summary, checkpoint):
        auth = container.get("parent_authorization", {})
        if (auth.get("e4_gate_sha256") != identity["mandatory_e4_gate"]["sha256"]
                or auth.get("e3_authorization", {}).get("gate_sha256") != identity["e3_gate"]["sha256"]
                or container.get("flex_sources") != expected_sources): raise ValueError("Flex parent/source binding changed")
    for key in ("controller_process_time_s", "controller_wall_time_s", "offline_evaluation_process_time_s", "offline_evaluation_wall_time_s"):
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0: raise ValueError("Missing actual CPU/wall measurements")
    json.dumps(result, allow_nan=False)
    return result


class FlexEpisodeStore:
    def __init__(self, directory, identity, *, resume=False, parent=True):
        self.directory = Path(directory).resolve(); self.identity = verify_identity(identity, parent=parent)
        if self.directory.exists():
            if not resume or read_json(self.directory/"grid_plan.json") != identity: raise ValueError("Flex resume identity changed")
        else:
            self.directory.mkdir(parents=True); atomic_json(self.directory/"grid_plan.json", identity)
        self.allowed = {episode_name(spec): spec for spec in episode_specs(identity["grid"])}

    def load(self, spec):
        name = episode_name(spec)
        if self.allowed.get(name) != spec: raise ValueError("Flex episode outside original full grid")
        folder = self.directory/name
        if not folder.exists(): return None
        receipt = read_json(folder/"episode_receipt.json")
        if receipt.get("schema") != "completed_flex_episode_v1" or receipt.get("spec") != spec or receipt.get("run_identity_sha256") != self.identity["identity_sha256"]:
            raise ValueError("Flex episode receipt identity changed")
        if set(receipt.get("artifacts", {})) != set(ARTIFACTS): raise ValueError("Flex artifact inventory incomplete")
        for name in ARTIFACTS:
            if receipt["artifacts"][name] != {**artifact_reference(folder/name), "path": name}: raise ValueError("Committed Flex artifact changed")
        result = {"summary": read_json(folder/"summary.json"), "traces": read_jsonl(folder/"traces.jsonl"),
            "warmup": read_json(folder/"warmup_anonymous.json"), "policy_checkpoint": read_json(folder/"policy_checkpoint.json")}
        return validate_result(result, spec, self.identity)

    def commit(self, result, spec, writer):
        validate_result(result, spec, self.identity)
        if self.load(spec) is not None: raise FileExistsError("Completed Flex episode cannot be overwritten")
        name = episode_name(spec); temporary = self.directory/("."+name+".pending."+uuid.uuid4().hex)
        writer(result, temporary)
        for artifact in ARTIFACTS:
            with (temporary/artifact).open("r+b") as stream: os.fsync(stream.fileno())
        references = {name: {**artifact_reference(temporary/name), "path": name} for name in ARTIFACTS}
        atomic_json(temporary/"episode_receipt.json", {"schema": "completed_flex_episode_v1", "spec": spec,
            "run_identity_sha256": self.identity["identity_sha256"], "artifacts": references, "research_complete": False})
        os.replace(temporary, self.directory/name)
        return self.load(spec)


def _recompute_episode(result, identity, online, calibration, replayed_training):
    """Auditor-only exact selected-cell costs/quality plus deterministic retraining."""
    summary = result["summary"]; seed, rate = summary["seed"], summary["label_rate"]
    config = read_json(identity["inputs"]["assets"]["config"]["path"]); settings = flex_settings(config)
    mapper = online.mapper; warm_n = min(32, settings.warmup_max_labels, 600*rate//100//5); prior_n = warm_n//2
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1001]))
    init_session = calibration.new_session(settings.caps); rows = []; retained = []
    for position, row in enumerate(rng.permutation(200)[:warm_n]):
        combo = int(rng.integers(20)); context = calibration.context(int(row)); member_only = position < prior_n
        actual = (init_session.execute_member_prior(int(row)) if member_only else init_session.execute(int(row), combo, include_evaluation=False))
        paid = asdict(actual["service"]+ResourceCost(0, 1, 0)); observed = result["warmup"][position]
        if observed["question_id"] != context.question_id or observed["combination_index"] != (None if member_only else combo) or observed["actual_cost"] != paid:
            raise ValueError("Flex warmup differs from actual predeclared selected paid source cells")
        if member_only:
            records = tuple(LegalMemberFeedback(context.question_id, i, float(calibration._arrays["member_scores"][row, i]), context.task,
                0, "broker-paid-member-reference:"+context.question_id, "online_calibration", True) for i in range(6))
            rows.append(PaidFullWarmupRow(context.question_id, context.phi, records))
        else:
            retained.append({"label_id": f"{context.question_id}:combo:{combo}", "question_id": context.question_id,
                "combination_index": combo, "value": _read_quality_for_evaluation(calibration, int(row), combo),
                "available_at": 0, "used_for_flex_training": False})
    if retained != result["policy_checkpoint"]["paid_unused_combo_labels"]: raise ValueError("Retained paid combo labels differ from executed source")
    _, labels, _, digest = _training_data(rows, now=0)
    report = summary["training_report"]
    if report["training_input_sha256"] != digest or report["original_exact_infinite_loss_rows"] != int(np.all(labels == 0, axis=1).sum()):
        raise ValueError("Flex paid training lineage or exact-infinite row count changed")
    key = (seed, rate)
    if key not in replayed_training: replayed_training[key] = train_flex_adapted(rows, seed=seed, now=0).state_dict()
    if result["policy_checkpoint"]["policy_state"] != replayed_training[key]:
        raise ValueError("Flex fitted state cannot be reproduced from only its paid full warmup")
    policy = FrozenPCASmoothedFlexPolicy.from_state_dict(replayed_training[key])
    session = online.new_session(settings.caps)
    for tick, row in enumerate(np.random.default_rng(seed).permutation(600)):
        context = online.context(int(row)); trace = result["traces"][tick]
        team = policy.select(context.phi); main = mapper.subsets.index(team)
        if trace["question_id"] != context.question_id or trace["main_index"] != main or trace["main_members"] != list(team):
            raise ValueError("Flex service was not produced by the frozen paid-only model")
        actual = session.execute(int(row), main, include_evaluation=False)
        if trace["service_cost"] != asdict(actual["service"]): raise ValueError("Flex source costs differ from recorded actual costs")
        evaluation = _evaluate_frozen_service(online, int(row), main)
        if any(trace[key] != evaluation[key] for key in ("quality", "candidate_coverage", "aggregation_loss")):
            raise ValueError("Flex quality/coverage not computed from its actual selected aggregate")
        kernel, competence = policy.kernel(context.phi)
        scores = [float(np.linalg.det(kernel[np.ix_(team, team)])) for team in mapper.subsets]
        if not np.allclose(trace["predicted_combination_scores_before_decision"], scores, atol=1e-12, rtol=1e-12) or not np.allclose(trace["predicted_member_scores_before_decision"], competence, atol=1e-12, rtol=1e-12):
            raise ValueError("Flex predecision predictions differ from its frozen fitted policy")
    per_task = {task: float(np.mean([r["quality"] for r in result["traces"] if r["task"] == task])) for task in ("bbh", "gsm8k")}
    if summary["per_task_quality"] != per_task or summary["quality_task_macro"] != float(np.mean(list(per_task.values()))):
        raise ValueError("Flex aggregate summary disagrees with source-recomputed question scores")


def verify_complete_flex_grid(directory, *, parent=True, recompute=True):
    directory = Path(directory).resolve(); identity = read_json(directory/"grid_plan.json")
    store = FlexEpisodeStore(directory, identity, resume=True, parent=parent)
    results, receipts = [], []
    for spec in episode_specs(identity["grid"]):
        result = store.load(spec)
        if result is None: raise ValueError("Incomplete Flex grid: "+episode_name(spec))
        results.append(result); receipts.append(artifact_reference(directory/episode_name(spec)/"episode_receipt.json"))
    if len(results) != 60: raise ValueError("Full Flex grid requires 60 original episodes")
    if recompute:
        assets = identity["inputs"]["assets"]
        online = MatrixEnvironment.from_npz(assets["online_matrix"]["path"], assets["online_manifest"]["path"], expected_split="online_stream")
        calibration = MatrixEnvironment.from_npz(assets["calibration_matrix"]["path"], assets["calibration_manifest"]["path"], expected_split="online_calibration")
        if online.software_test_only or calibration.software_test_only or len(online) != 600 or len(calibration) != 200:
            raise ValueError("Production Flex acceptance rejects toy/reduced matrices")
        replayed = {}
        for result in results: _recompute_episode(result, identity, online, calibration, replayed)
        if len(replayed) != 20: raise ValueError("Flex independent training replay did not cover all seed/rate warmups")
    return identity, results, receipts


def reference_evaluation(identity, results, receipts):
    if len(results) != 60 or len(receipts) != 60: raise ValueError("Incomplete reference evidence")
    return {"schema": "flex_reference_evaluation_v1", "run_identity_sha256": identity["identity_sha256"],
        "episodes_evaluated": 60, "question_support_per_episode": 600, "per_task_support": {"bbh": 300, "gsm8k": 300},
        "distinct_seed_rate_training_replays": 20, "independent_paid_only_training_reproduced": True,
        "actual_source_cost_and_final_aggregate_scores_recomputed": True,
        "variant": FLEX_VARIANT, "native_system": "NOT_RUN", "episode_receipts": receipts,
        "per_episode": [{"seed": r["summary"]["seed"], "label_rate": r["summary"]["label_rate"], "feedback_delay": r["summary"]["feedback_delay"],
            "quality_task_macro": r["summary"]["quality_task_macro"], "candidate_coverage": r["summary"]["candidate_coverage"],
            "aggregation_loss": r["summary"]["aggregation_loss"], "cost": r["summary"]["budget"]["spent"],
            "full_training_rows": r["summary"]["training_report"]["training_rows"],
            "exact_original_infinite_loss_rows": r["summary"]["training_report"]["original_exact_infinite_loss_rows"]} for r in results],
        "academic_conclusion": "CONTROLLED_REFERENCE_COMPLETE_NO_AUTOMATIC_POSITIVE_OR_NATIVE_CLAIM",
        "information_boundary": "paid-prior supervision differs from random-combo main warmup; same-prior attribution comparison belongs to the explicit E5 ablation",
        "risk": "small full-label warmup, smoothed loss, frozen PCA and fixedK3 differ from original FlexRouter"}


def _reference_bindings(identity, receipts):
    return {"input_identity_sha256": canonical_digest(identity["inputs"]),
        "mandatory_e4_gate_sha256": identity["mandatory_e4_gate"]["sha256"], "e3_gate_sha256": identity["e3_gate"]["sha256"],
        "flex_sources_sha256": canonical_digest(identity["flex_sources"]), "required_grid_sha256": canonical_digest(identity["grid"]),
        "episode_receipts_sha256": canonical_digest(receipts)}


def _check_grid_summary(summary, identity, results, receipts):
    if (summary.get("schema") != "flex_grid_summary_v1" or summary.get("run_identity_sha256") != identity["identity_sha256"]
            or summary.get("episode_receipts") != receipts or summary.get("grid") != identity["grid"]
            or summary.get("native_system") != "NOT_RUN" or summary.get("variant") != FLEX_VARIANT):
        raise ValueError("Flex runner summary missing complete committed grid")
    measurements = summary.get("resource_measurements_by_episode", [])
    if len(measurements) != 60: raise ValueError("Flex runner source-cost measurements incomplete")
    for observed, result in zip(measurements, results):
        actual = result["summary"]
        required = {"spec": {key: actual[key] for key in ("strategy", "seed", "label_rate", "feedback_delay")},
            "quality_task_macro": actual["quality_task_macro"], "candidate_coverage": actual["candidate_coverage"],
            "aggregation_loss": actual["aggregation_loss"], "actual_resources": actual["budget"]["spent"],
            "limits": actual["budget"]["limits"], "controller_process_time_s": actual["controller_process_time_s"],
            "controller_wall_time_s": actual["controller_wall_time_s"],
            "training_process_time_s": actual["runtime_measurements"]["training_process_time_s"],
            "source_service_execution_wall_time_s": actual["source_service_execution_wall_time_s"]}
        if observed != required: raise ValueError("Flex runner cost/quality summary disagrees with paid episodes")


def accept_flex_reference(directory, gate_path):
    identity, results, receipts = verify_complete_flex_grid(directory, parent=True, recompute=True)
    directory = Path(directory)
    summary = read_json(directory/"grid_summary.json")
    _check_grid_summary(summary, identity, results, receipts)
    evaluation_path = directory/"flex_reference_evaluation.json"
    immutable_json(evaluation_path, reference_evaluation(identity, results, receipts))
    gate = {"schema": "flex_reference_gate_v1", "status": "FLEX_REFERENCE_TECHNICAL_PASS",
        "scope": "technical_completeness_not_scientific_positivity", "software_test_only": False,
        "grid_plan": artifact_reference(directory/"grid_plan.json"), "grid_summary": artifact_reference(directory/"grid_summary.json"),
        "evaluation": artifact_reference(evaluation_path), "episode_receipts": receipts,
        "mandatory_e4_gate": identity["mandatory_e4_gate"], "e3_gate": identity["e3_gate"],
        "run_identity_sha256": identity["identity_sha256"], "bindings": _reference_bindings(identity, receipts),
        "variant": FLEX_VARIANT, "native_system": "NOT_RUN", "research_complete": False}
    immutable_json(gate_path, gate)
    return gate


def validate_flex_reference_gate(path, current_inputs):
    gate = read_json(path)
    if (gate.get("schema") != "flex_reference_gate_v1" or gate.get("status") != "FLEX_REFERENCE_TECHNICAL_PASS"
            or gate.get("scope") != "technical_completeness_not_scientific_positivity" or gate.get("software_test_only") is not False):
        raise ValueError("Independent real trained-Flex reference gate required")
    plan = verified_reference(gate["grid_plan"])
    identity, results, receipts = verify_complete_flex_grid(plan.parent, parent=True, recompute=True)
    if (identity["inputs"] != current_inputs or gate.get("run_identity_sha256") != identity["identity_sha256"]
            or gate.get("episode_receipts") != receipts or gate.get("mandatory_e4_gate") != identity["mandatory_e4_gate"]
            or gate.get("e3_gate") != identity["e3_gate"] or gate.get("variant") != FLEX_VARIANT or gate.get("native_system") != "NOT_RUN"):
        raise ValueError("Flex gate belongs to different inputs, parents or complete episodes")
    summary = read_json(verified_reference(gate["grid_summary"]))
    _check_grid_summary(summary, identity, results, receipts)
    if read_json(verified_reference(gate["evaluation"])) != reference_evaluation(identity, results, receipts):
        raise ValueError("Flex independently reconstructed evaluation differs")
    bindings = _reference_bindings(identity, receipts)
    if gate.get("bindings") != bindings: raise ValueError("Flex gate bindings changed")
    return gate, bindings


def validate_flex_episode(checkpoint_ref, episode_receipt_ref, grid_plan_ref, *, batch_authorization=None):
    plan = verified_reference(grid_plan_ref); identity = read_json(plan)
    if batch_authorization is not None:
        if not isinstance(batch_authorization, FlexFinalBatchAuthorization) or batch_authorization._token is not _FINAL_BATCH_TOKEN:
            raise ValueError("Unverified Flex final batch authorization")
        if artifact_reference(plan) != batch_authorization.plan_reference: raise ValueError("Flex final episode belongs to another authorized grid")
        batch_authorization._check_top_level()
    store = FlexEpisodeStore(plan.parent, identity, resume=True, parent=batch_authorization is None)
    receipt_path = verified_reference(episode_receipt_ref); receipt = read_json(receipt_path)
    expected = plan.parent/episode_name(receipt["spec"])/"episode_receipt.json"
    if receipt_path != expected.resolve(): raise ValueError("Flex episode outside its exact source grid")
    result = store.load(receipt["spec"])
    if result is None or verified_reference(checkpoint_ref) != (receipt_path.parent/"policy_checkpoint.json").resolve():
        raise ValueError("Flex checkpoint differs from its paid episode")
    return result


def make_e4_full_gate(path, *, mandatory_e4_gate, flex_reference_gate, current_inputs):
    _, mandatory_bindings = validate_e4_gate(mandatory_e4_gate, current_inputs)
    _, flex_bindings = validate_flex_reference_gate(flex_reference_gate, current_inputs)
    mandatory = artifact_reference(mandatory_e4_gate); flex = artifact_reference(flex_reference_gate)
    if flex_bindings["mandatory_e4_gate_sha256"] != mandatory["sha256"]: raise ValueError("Flex reference binds a different mandatory E4 gate")
    bindings = {**mandatory_bindings, "mandatory_e4_gate_sha256": mandatory["sha256"],
        "flex_reference_gate_sha256": flex["sha256"], "flex_sources_sha256": flex_bindings["flex_sources_sha256"]}
    gate = {"schema": "e4_full_technical_gate_v1", "status": "E4_FULL_TECHNICAL_PASS",
        "scope": "technical_completeness_not_scientific_positivity", "software_test_only": False,
        "mandatory_e4_gate": mandatory, "flex_reference_gate": flex, "bindings": bindings,
        "input_identity_sha256": canonical_digest(current_inputs), "native_systems": "NOT_RUN", "research_complete": False}
    immutable_json(path, gate)
    return gate


def validate_e4_full_gate(path, current_inputs):
    gate = read_json(path)
    if (gate.get("schema") != "e4_full_technical_gate_v1" or gate.get("status") != "E4_FULL_TECHNICAL_PASS"
            or gate.get("scope") != "technical_completeness_not_scientific_positivity" or gate.get("software_test_only") is not False):
        raise ValueError("Independent full E4 technical gate required")
    mandatory_path = verified_reference(gate["mandatory_e4_gate"]); flex_path = verified_reference(gate["flex_reference_gate"])
    _, mandatory_bindings = validate_e4_gate(mandatory_path, current_inputs)
    _, flex_bindings = validate_flex_reference_gate(flex_path, current_inputs)
    if flex_bindings["mandatory_e4_gate_sha256"] != gate["mandatory_e4_gate"]["sha256"]: raise ValueError("E4 full gate parent mismatch")
    bindings = {**mandatory_bindings, "mandatory_e4_gate_sha256": gate["mandatory_e4_gate"]["sha256"],
        "flex_reference_gate_sha256": gate["flex_reference_gate"]["sha256"], "flex_sources_sha256": flex_bindings["flex_sources_sha256"]}
    if gate.get("bindings") != bindings or gate.get("input_identity_sha256") != canonical_digest(current_inputs):
        raise ValueError("Full E4 gate source/input binding changed")
    return gate, bindings


@dataclass(frozen=True)
class FlexFinalBatchAuthorization:
    full_gate_reference: dict
    plan_reference: dict
    current_inputs_json: str
    bindings_json: str
    _token: object = None

    def _check_top_level(self):
        if self._token is not _FINAL_BATCH_TOKEN: raise ValueError("Unverified Flex final batch")
        verified_reference(self.full_gate_reference); verified_reference(self.plan_reference)

    def verify_unchanged(self):
        """One complete revalidation at batch end, not once per checkpoint."""
        self._check_top_level()
        _, bindings = validate_e4_full_gate(self.full_gate_reference["path"], json.loads(self.current_inputs_json))
        if bindings != json.loads(self.bindings_json): raise ValueError("Flex final batch dependencies changed")
        return True


def authorize_flex_final_batch(e4_full_gate, current_inputs):
    gate, bindings = validate_e4_full_gate(e4_full_gate, current_inputs)
    reference = read_json(verified_reference(gate["flex_reference_gate"]))
    plan = verified_reference(reference["grid_plan"])
    return FlexFinalBatchAuthorization(artifact_reference(e4_full_gate), artifact_reference(plan),
        json.dumps(current_inputs, sort_keys=True, allow_nan=False), json.dumps(bindings, sort_keys=True, allow_nan=False), _FINAL_BATCH_TOKEN)
