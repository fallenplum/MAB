"""Static paid-warmup Flex coverage reference; real aggregate quality is external.

Production needs independent mandatory E3/E4 acceptance. CPU training uses only
the first-half paid all-six reference events, never a full calibration scan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
import numpy as np

from .budget import BudgetLedger, ResourceCost
from .learner import cold_start_limit
from .simulation import (FeedbackBroker, validate_strategy_requirements, _clock,
    _elapsed, _add_timing, _broker_checkpoint)
from .neighbor_policies import LegalMemberFeedback
from .neighbor_simulation import (authorize_neighbor_pair, _check_authorization,
    _safe, _evaluate_frozen_service)
from .flex_training import (PaidFullWarmupRow, train_flex_adapted, FLEX_VARIANT,
    FLEX_STRATEGY)

_FLEX_TOKEN = object()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flex_source_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ("src/budget_router/flex_training.py", "src/budget_router/flex_simulation.py",
             "protocol/flex_training_adaptation_manifest.json")
    return {name: _sha(root/name) for name in names}


@dataclass(frozen=True)
class VerifiedFlexAuthorization:
    e3_authorization: object
    e4_gate_path: str
    e4_gate_sha256: str
    e4_bindings: object
    flex_sources: object
    _token: object = None

    def public_record(self):
        return {"schema": "verified_flex_parent_authorization_v1",
            "e3_authorization": self.e3_authorization.public_record(),
            "e4_gate_path": self.e4_gate_path, "e4_gate_sha256": self.e4_gate_sha256,
            "e4_bindings": dict(self.e4_bindings), "flex_sources": dict(self.flex_sources),
            "scope": "mandatory E3/E4 already accepted; this separate adapted coverage reference is not native FlexRouter"}


def authorize_flex_pair(online, calibration, *, e3_gate_path, e4_gate_path, current_inputs):
    from .neighbor_validation import validate_e4_gate
    e3 = authorize_neighbor_pair(online, calibration, gate_path=e3_gate_path, current_inputs=current_inputs)
    gate, bindings = validate_e4_gate(e4_gate_path, current_inputs)
    if (gate.get("schema") != "e4_mandatory_neighbors_gate_v1"
            or gate.get("status") != "E4_MANDATORY_NEIGHBORS_PASS"
            or gate.get("software_test_only") is not False
            or gate.get("scope") != "technical_completeness_not_scientific_positivity"):
        raise ValueError("Independent mandatory E4 technical acceptance required")
    if bindings.get("e3_gate_sha256") != e3.gate_sha256:
        raise ValueError("Flex E4 gate binds a different E3 parent")
    return VerifiedFlexAuthorization(e3, str(Path(e4_gate_path).resolve(strict=True)), _sha(e4_gate_path),
        MappingProxyType(dict(bindings)), MappingProxyType(flex_source_hashes()), _FLEX_TOKEN)


def _check_flex_authorization(online, calibration, authorization):
    if online.software_test_only:
        if authorization is not None: raise ValueError("Software tests cannot consume production Flex authorization")
        return None
    if not isinstance(authorization, VerifiedFlexAuthorization) or authorization._token is not _FLEX_TOKEN:
        raise ValueError("Use authorize_flex_pair; no boolean or inline gate")
    _check_authorization(online, calibration, authorization.e3_authorization)
    if _sha(authorization.e4_gate_path) != authorization.e4_gate_sha256:
        raise ValueError("E4 gate changed after full validation")
    if dict(authorization.flex_sources) != flex_source_hashes():
        raise ValueError("Flex implementation changed after authorization")
    return authorization.public_record()


def run_flex_simulation(online, calibration, *, seed=42, label_rate=25, delay=0,
                        settings, flex_authorization=None):
    """Full static service, with the identical E5 paid-prior warmup cost."""
    started = _clock()
    if label_rate not in (10, 25, 50, 100) or not isinstance(delay, int) or delay < 0:
        raise ValueError("Locked rate and nonnegative delay required")
    validate_strategy_requirements(online, calibration, "none", settings)
    if settings.warmup_mode != "paid_member_prior_then_combo" or settings.feedback_mode != "verifiable":
        raise ValueError("Flex training requires the predeclared paid_member_prior_then_combo ablation")
    if settings.main_feedback_natural:
        raise ValueError("This frozen no-online-feedback reference uses the locked no-natural-feedback setting")
    if not online.software_test_only:
        for environment in (online, calibration):
            permission = environment._manifest.get("feedback_permissions", {})
            if permission.get("mode") != "verifiable" or permission.get("generated_member_scores_on_reference_reveal") is not True:
                raise ValueError("Explicit authorized generated-member feedback permission required")
    authorization = _check_flex_authorization(online, calibration, flex_authorization)
    preflight = _elapsed(started)
    totals = {name: {"process_time_s": 0., "wall_time_s": 0.} for name in ("controller", "environment_replay", "offline_evaluation")}
    started = _clock(); n = len(online); mapper = online.mapper
    total_labels = n*label_rate//100
    ledger = BudgetLedger(ResourceCost(settings.execution_limit, total_labels, settings.token_limit))
    normal = settings.caps.service(False); ledger.reserve_future_service(normal*n)
    rng = np.random.default_rng(seed); order = rng.permutation(n)
    init_rng = np.random.default_rng(np.random.SeedSequence([seed, 1001]))
    initializer = calibration.new_session(settings.caps)
    init_broker = FeedbackBroker(initializer, delay=0, feedback_mode="verifiable")
    session = online.new_session(settings.caps)
    broker = FeedbackBroker(session, delay=delay, feedback_mode="verifiable")
    initial_max = min(cold_start_limit(total_labels, settings.warmup_max_labels), len(calibration))
    prior_target = initial_max//2
    rows, warmup, unused_combo_labels = [], [], []
    _add_timing(totals["controller"], _elapsed(started))
    for position, row in enumerate(init_rng.permutation(len(calibration))[:initial_max]):
        started = _clock(); member_only = position < prior_target
        upper = (ResourceCost(6*settings.caps.member_calls, 1,
                    6*settings.caps.member_calls*(settings.caps.member_prompt+settings.caps.member_completion))
                 if member_only else settings.caps.service(True))
        if not ledger.can_reserve(upper):
            _add_timing(totals["controller"], _elapsed(started)); break
        # Match primary RNG consumption even when the all-six event has no team.
        combo = int(init_rng.integers(20)); rid = f"warmup:{position}"
        ledger.reserve(rid, upper)
        _add_timing(totals["controller"], _elapsed(started))
        replay = _clock()
        outcome = (initializer.execute_member_prior(int(row)) if member_only
                   else initializer.execute(int(row), combo, include_evaluation=False))
        _add_timing(totals["environment_replay"], _elapsed(replay))
        started = _clock(); context = calibration.context(int(row))
        ledger.settle(rid, outcome["service"]+ResourceCost(0, 1, 0))
        _add_timing(totals["controller"], _elapsed(started))
        replay = _clock()
        if member_only: init_broker.request_member_prior_reference(int(row), now=0)
        else: init_broker.request(int(row), (combo,), now=0)
        labels, members = init_broker.deliver(0)
        _add_timing(totals["environment_replay"], _elapsed(replay))
        started = _clock()
        if member_only:
            if labels or len(members) != 6 or any(qid != context.question_id for qid, _, _ in members):
                raise ValueError("Paid all-six reference event is not complete")
            records = tuple(LegalMemberFeedback(qid, int(index), float(value), context.task, 0,
                "broker-paid-member-reference:"+context.question_id, "online_calibration", True)
                for qid, index, value in members)
            rows.append(PaidFullWarmupRow(context.question_id, context.phi, records))
        else:
            unused_combo_labels.extend({"label_id": label.label_id, "question_id": label.question_id,
                "combination_index": combo, "value": float(label.value), "available_at": int(label.available_at),
                "used_for_flex_training": False} for label in labels)
        warmup.append({"question_id": context.question_id, "combination_index": None if member_only else combo,
            "purchase_kind": "paid_all_six_member_reference" if member_only else "paid_combo_final_label",
            "feedback_units": 1, "generated_member_labels": len(members), "aggregate_labels": len(labels),
            "actual_cost": asdict(outcome["service"]+ResourceCost(0, 1, 0)),
            "source_service_execution_wall_time_s": outcome["source_service_execution_wall_time_s"]})
        _add_timing(totals["controller"], _elapsed(started))
    started = _clock(); training_start = _clock()
    policy = train_flex_adapted(rows, seed=seed, now=0)
    training_time = _elapsed(training_start)
    initialization_cost = asdict(ledger.spent)
    _add_timing(totals["controller"], _elapsed(started))
    initialization_controller = dict(totals["controller"])
    frozen_state = policy.state_dict()
    traces = []
    for tick, row in enumerate(order):
        timing = {name: {"process_time_s": 0., "wall_time_s": 0.} for name in totals}
        started = _clock(); context = online.context(int(row))
        team = policy.select(context.phi); main = mapper.subsets.index(team)
        sid = f"service:{tick}"; ledger.reserve_service(sid, normal)
        _add_timing(timing["controller"], _elapsed(started))
        evaluating = _clock(); kernel, competence = policy.kernel(context.phi)
        scores = [float(np.linalg.det(kernel[np.ix_(subset, subset)])) for subset in mapper.subsets]
        snapshot = {"predicted_combination_scores_before_decision": scores,
            "predicted_combination_working_variance_before_decision": None,
            "service_selection_values_before_decision": scores,
            "prediction_supported_combination_indices": list(range(20)),
            "predicted_member_scores_before_decision": competence.tolist(),
            "prediction_semantics": "DPP subset determinant proxy; not predicted final aggregate quality or oracle candidate coverage"}
        _add_timing(timing["offline_evaluation"], _elapsed(evaluating))
        replay = _clock(); outcome = session.execute(int(row), main, include_evaluation=False)
        _add_timing(timing["environment_replay"], _elapsed(replay))
        started = _clock(); ledger.settle(sid, outcome["service"])
        # Deliberately no request/deliver/update: incomplete three-member online
        # feedback is not purchased by this static six-label supervised reference.
        _add_timing(timing["controller"], _elapsed(started))
        evaluating = _clock(); evaluated = _evaluate_frozen_service(online, int(row), main)
        _add_timing(timing["offline_evaluation"], _elapsed(evaluating))
        trace = {"step": tick, "question_id": context.question_id, "task": context.task,
            "main_index": main, "main_members": list(team), "alternate_index": None, "action_kind": "none", **evaluated,
            "policy_label_count_before_decision": 6*len(rows), "policy_label_count_semantics": "paid complete warmup member labels; frozen",
            "policy_member_label_counts_before_decision": [len(rows)]*6,
            "arrived_labels_before_decision": 0, "immediate_labels_after_freeze": 0,
            "requested_combination_labels": 0, "pending_labels": 0,
            "legally_revealed_generated_member_feedback": 0, "predicted_incremental_cost": 0.,
            "actual_incremental_cost": 0., "predicted_cost_error": 0.,
            "information_gain_proxy": 0., "directional_information_evaluated": False, "acquisition_score": 0.,
            "service_cost": asdict(outcome["service"]), "purchase_cost": asdict(ResourceCost()),
            "cumulative_cost": asdict(ledger.spent), "protected_future_service": asdict(ledger.future_service),
            "source_service_execution_wall_time_s": outcome["source_service_execution_wall_time_s"],
            "source_audit_execution_wall_time_s": 0., "service_frozen_before_audit": True,
            "prediction_snapshot_precedes_current_execution_and_feedback": True,
            "policy_updated_after_current_question": False, **snapshot}
        for name, measurement in timing.items():
            _add_timing(totals[name], measurement)
            for key, value in measurement.items(): trace[name+"_"+key] = value
        traces.append(_safe(trace))
    if policy.state_dict() != frozen_state: raise RuntimeError("Frozen Flex policy changed during service")
    evaluating = _clock()
    per_task = {task: float(np.mean([r["quality"] for r in traces if r["task"] == task])) for task in sorted({r["task"] for r in traces})}
    _add_timing(totals["offline_evaluation"], _elapsed(evaluating))
    exporting = _clock()
    checkpoint = {"schema": "flex_smoothed_frozen_policy_checkpoint_v1", "software_test_only": online.software_test_only,
        "strategy": FLEX_STRATEGY, "method_variant": FLEX_VARIANT, "seed": int(seed), "label_rate": label_rate,
        "feedback_delay": delay, "next_tick": n, "frozen_as_of_tick": n-1, "pending_feedback_flushed": False,
        "serving_policy": "frozen_fixedK3_dpp_schur", "policy_state": frozen_state,
        "feature_map": {"n_models": 6, "k": 3, "phi_dimension": online.phi_dimension,
            "basis_dim": mapper.basis_dim, "combinations": [list(team) for team in mapper.subsets]},
        "settings": asdict(settings), "budget": ledger.snapshot(),
        "rng_states": {"policy": rng.bit_generator.state, "warmup": init_rng.bit_generator.state},
        "feedback": _broker_checkpoint(broker), "warmup_feedback": _broker_checkpoint(init_broker),
        "paid_unused_combo_labels": unused_combo_labels,
        "stream_question_order": [online.context(int(row)).question_id for row in order],
        "online_provenance": online.public_provenance(), "calibration_provenance": calibration.public_provenance(),
        "parent_authorization": authorization, "flex_sources": flex_source_hashes(),
        "online_updates": False, "native_system": "NOT_RUN"}
    checkpoint = _safe(checkpoint); export_time = _elapsed(exporting)
    measurements = {name+"_"+key: value for name, measurement in totals.items() for key, value in measurement.items()}
    measurements.update({"preflight_"+key: value for key, value in preflight.items()})
    measurements.update({"initialization_controller_"+key: value for key, value in initialization_controller.items()})
    measurements.update({"training_"+key: value for key, value in training_time.items()})
    measurements.update({"checkpoint_export_"+key: value for key, value in export_time.items()})
    measurements["scope"] = "Actual CPU FP64 training/controller clocks, environment replay and evaluator separate; no real GPU work or inferred GPU kernel seconds"
    source_walls = [r["source_service_execution_wall_time_s"] for r in traces]
    summary = {"status": "SOFTWARE_TEST_ONLY" if online.software_test_only else "COMPLETED_RESTRICTED_FEEDBACK_SIMULATION_ONLY",
        "research_complete": False, "native_system": "NOT_RUN", "strategy": FLEX_STRATEGY, "method_variant": FLEX_VARIANT,
        "seed": int(seed), "label_rate": label_rate, "feedback_delay": delay, "n_questions": n,
        "all_questions_served": len(traces) == n, "quality_micro": float(np.mean([r["quality"] for r in traces])),
        "quality_task_macro": float(np.mean(list(per_task.values()))), "per_task_quality": per_task,
        "candidate_coverage": float(np.mean([r["candidate_coverage"] for r in traces])),
        "aggregation_loss": float(np.mean([r["aggregation_loss"] for r in traces])),
        "warmup_labels": len(warmup), "warmup_cost": initialization_cost,
        "warmup_mode": "paid_member_prior_then_combo", "paid_member_reference_events": len(rows),
        "prior_member_labels_used": 6*len(rows), "warmup_arrived_combo_labels": len(unused_combo_labels),
        "warmup_combo_labels_used_for_training": 0, "arrived_original_labels": len(unused_combo_labels),
        "member_feedback_records_exposed": len(init_broker.member_history), "member_labels_used_by_policy": 6*len(rows),
        "feedback_mode": "verifiable", "pending_paid_labels_at_stream_end": 0,
        "training_report": policy.training_report, "online_updates": False,
        "online_delay_has_no_effect_on_frozen_no_feedback_policy": True,
        "label_allocation": "same E5 paid prior warmup cost; static policy buys no unused incomplete online labels",
        "budget": ledger.snapshot(), "settings": asdict(settings), "generation_replicates": 1,
        "stream_seed_is_not_new_generation": True, "cost_prediction_uses_current_hidden_lengths": False,
        "runtime_measurements": measurements, "controller_process_time_s": totals["controller"]["process_time_s"],
        "controller_wall_time_s": totals["controller"]["wall_time_s"],
        "offline_evaluation_process_time_s": totals["offline_evaluation"]["process_time_s"],
        "offline_evaluation_wall_time_s": totals["offline_evaluation"]["wall_time_s"],
        "source_service_execution_wall_time_s": None if any(v is None for v in source_walls) else sum(source_walls),
        "source_audit_execution_wall_time_s": 0., "policy_checkpoint_schema": checkpoint["schema"],
        "parent_authorization": authorization, "flex_sources": flex_source_hashes(),
        "online_provenance": online.public_provenance(), "calibration_provenance": calibration.public_provenance()}
    return {"summary": _safe(summary), "traces": traces, "warmup": warmup, "policy_checkpoint": checkpoint}
