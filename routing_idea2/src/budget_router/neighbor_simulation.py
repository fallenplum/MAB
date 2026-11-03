"""E4 CPU replay engine: restricted receipts, shared paid warmup, hard budget.

Only the environment/session/broker and evaluator access hidden matrices. The
neighbour policies receive public context, paid arrived labels and past costs.
No model execution, final-test release, or trained FlexRouter is implemented.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import numpy as np

from .budget import BudgetLedger, ResourceCost
from .learner import cold_start_limit
from .neighbor_policies import (
    CESPaidWarmupPolicy, C2MABVPolicy, LegalMemberFeedback, LegalComboFeedback,
    PastMemberExecution, structural_reward,
)
from .simulation import (
    FeedbackBroker, validate_strategy_requirements, _clock, _elapsed,
    _add_timing, _broker_checkpoint, _read_quality_for_evaluation,
)

NEIGHBOR_STRATEGIES = ("ces", "c2mabv_awc", "c2mabv_suc", "c2mabv_aic")
_VALIDATION_TOKEN = object()
_E3_EXTRA_BINDINGS = ("config_sha256", "required_grid_sha256", "algorithm_sources_sha256",
                      "e2_gate_sha256", "e3_grid_plan_sha256", "e3_grid_summary_sha256",
                      "e3_episode_receipts_sha256")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def neighbor_source_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ("src/budget_router/neighbor_simulation.py", "src/budget_router/neighbor_policies.py",
             "protocol/neighbor_implementation_manifest.json")
    return {name: _sha(root/name) for name in names}


def _environment_bindings(online, calibration):
    return {"online_matrix_sha256": getattr(online, "_matrix_file_sha256", None),
            "online_manifest_sha256": getattr(online, "_manifest_file_sha256", None),
            "calibration_matrix_sha256": getattr(calibration, "_matrix_file_sha256", None),
            "calibration_manifest_sha256": getattr(calibration, "_manifest_file_sha256", None),
            "e0_lock_sha256": online._manifest.get("e0_lock", {}).get("sha256"),
            "e1_gate_sha256": online._manifest.get("e1_gate", {}).get("sha256"),
            "pca_sha256": online._manifest.get("pca_artifact", {}).get("sha256"),
            "pool_config_sha256": online._manifest.get("pool_config", {}).get("sha256")}


@dataclass(frozen=True)
class VerifiedE3Authorization:
    gate_path: str
    gate_sha256: str
    bindings: object
    neighbor_sources: object
    _token: object = None

    def public_record(self):
        return {"schema": "verified_e3_neighbor_authorization_v1", "gate_path": self.gate_path,
                "gate_sha256": self.gate_sha256, "bindings": dict(self.bindings),
                "neighbor_sources": dict(self.neighbor_sources),
                "validation": "strict independent E3 validator; technical completeness, not scientific positivity"}


def authorize_neighbor_pair(online, calibration, *, gate_path, current_inputs):
    """Validate the complete upstream E2/E3 evidence once before an E4 grid.

    The external validator checks all 300 E2 and 420 E3 episodes and sources.
    E4 source hashes remain independent, avoiding changes to the old grid identity.
    """
    from .stage_validation import validate_e3_gate
    gate, bindings = validate_e3_gate(gate_path, current_inputs)
    if (gate.get("schema") != "e3_budget_mechanism_gate_v1" or gate.get("status") != "E3_TECHNICAL_PASS"
            or gate.get("software_test_only") is not False
            or gate.get("scope") != "technical_completeness_not_scientific_positivity"):
        raise ValueError("A production artifact-backed E3 technical PASS is required")
    expected = _environment_bindings(online, calibration)
    if any(not _digest(value) or bindings.get(key) != value for key, value in expected.items()):
        raise ValueError("Verified E3 gate belongs to different loaded environments")
    if any(not _digest(bindings.get(key)) for key in _E3_EXTRA_BINDINGS):
        raise ValueError("E3 complete-grid/source bindings are missing")
    return VerifiedE3Authorization(str(Path(gate_path).resolve(strict=True)), _sha(gate_path),
                                   MappingProxyType(dict(bindings)), MappingProxyType(neighbor_source_hashes()),
                                   _VALIDATION_TOKEN)


def _check_authorization(online, calibration, authorization):
    if online.software_test_only:
        if authorization is not None:
            raise ValueError("Software tests cannot consume a production E3 authorization")
        return None
    if not isinstance(authorization, VerifiedE3Authorization) or authorization._token is not _VALIDATION_TOKEN:
        raise ValueError("Use authorize_neighbor_pair; boolean/inline E3 PASS is not authorization")
    if _sha(authorization.gate_path) != authorization.gate_sha256:
        raise ValueError("E3 gate changed after independent validation")
    if any(authorization.bindings.get(key) != value for key, value in _environment_bindings(online, calibration).items()):
        raise ValueError("E3 authorization environment binding changed")
    if dict(authorization.neighbor_sources) != neighbor_source_hashes():
        raise ValueError("Neighbour implementation changed after authorization")
    return authorization.public_record()


def _safe(value):
    if isinstance(value, dict): return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(item) for item in value]
    if isinstance(value, np.ndarray): return _safe(value.tolist())
    if isinstance(value, np.generic): return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def _member_receipts(labels, members, contexts, generated, *, split, warmup):
    authorization = {label.question_id: label for label in labels}
    result = []
    for qid, index, value in members:
        if qid not in authorization or qid not in contexts or index not in generated.get(qid, ()):
            raise ValueError("Broker member feedback lacks a matching paid generated reference event")
        label = authorization[qid]
        result.append(LegalMemberFeedback(qid, int(index), float(value), contexts[qid].task,
            int(label.available_at), "broker-original:"+label.label_id, split, warmup))
    return result


def _cost_receipts(context, outcome, *, split, tick, warmup, upper):
    result = []
    for kind, index, tokens in outcome["cost_observations"]:
        if kind != "member": continue
        value = 1.+.001*int(tokens)
        if value > upper+1e-12:
            raise ValueError("Actual member cost exceeds its prelocked bound")
        result.append(PastMemberExecution(f"{context.question_id}:member:{index}:logical-paid",
            context.question_id, int(index), value, tick, split, warmup))
    return result


def _combo_receipt(label, mapper, *, split):
    index = int(label.label_id.rsplit(":combo:", 1)[1])
    return LegalComboFeedback(label.label_id, label.question_id, mapper.subsets[index], float(label.value),
        int(label.available_at), "broker-original:"+label.label_id, split)


def _evaluate_frozen_service(environment, row, main, *, reward_model=None):
    """Evaluator-only: these values never enter the policy without the broker."""
    quality = _read_quality_for_evaluation(environment, row, main)
    members = environment.mapper.subsets[main]
    values = environment._arrays["member_scores"][row]
    coverage = float(np.any(values[list(members)] == 1.))
    return {"quality": quality, "candidate_coverage": coverage,
            "aggregation_loss": coverage-quality,
            "realized_native_structural_reward_for_evaluation_only": None if reward_model is None else structural_reward(members, values, reward_model)}


def _ces_snapshot(policy, phi, mapper):
    phis = policy.member_features(phi)
    sets = np.asarray([phis[list(team)].sum(axis=0) for team in mapper.subsets])
    means = sets@policy.core.w_hat
    variance = np.einsum("ij,jk,ik->i", sets, policy.core.V_inv, sets)
    return {"predicted_combination_scores_before_decision": means.tolist(),
            "predicted_combination_working_variance_before_decision": variance.tolist(),
            "service_selection_values_before_decision": (means+policy.alpha*np.sqrt(np.maximum(variance, 0.))).tolist(),
            "prediction_supported_combination_indices": list(range(20)),
            "prediction_semantics": "CES shared feature aggregate-reward estimate; selection uses greedy set marginal, not global max or independentTopK",
            "predicted_member_scores_before_decision": None}


def run_neighbor_simulation(online, calibration, *, strategy, seed=42, label_rate=25,
                            delay=0, settings, e3_authorization=None):
    """Full prospective stream; main-only paid feedback, no audit/rejection draw."""
    preflight_start = _clock()
    if strategy not in NEIGHBOR_STRATEGIES: raise ValueError("Unknown neighbour strategy")
    if label_rate not in (10, 25, 50, 100) or not isinstance(delay, int) or delay < 0:
        raise ValueError("Locked feedback rate and nonnegative delay required")
    validate_strategy_requirements(online, calibration, "none", settings)
    if settings.feedback_mode != "verifiable" or settings.warmup_mode != "random_combo":
        raise ValueError("E4 policies require the same verifiable paid random-combo warmup")
    if settings.caps.member_calls != 1:
        raise ValueError("Prelocked C2 member cost bound assumes one member call per cell")
    if not online.software_test_only:
        for environment in (online, calibration):
            permission = environment._manifest.get("feedback_permissions", {})
            if permission.get("mode") != "verifiable" or permission.get("generated_member_scores_on_reference_reveal") is not True:
                raise ValueError("Production neighbours require explicit generated-member feedback permission")
    authorization = _check_authorization(online, calibration, e3_authorization)
    preflight = _elapsed(preflight_start)
    totals = {name: {"process_time_s": 0., "wall_time_s": 0.} for name in ("controller", "environment_replay", "offline_evaluation")}
    started = _clock()
    n = len(online); total_labels = n*label_rate//100; mapper = online.mapper
    ledger = BudgetLedger(ResourceCost(settings.execution_limit, total_labels, settings.token_limit))
    normal = settings.caps.service(settings.main_feedback_natural)
    ledger.reserve_future_service(normal*n)
    rng = np.random.default_rng(seed)
    init_rng = np.random.default_rng(np.random.SeedSequence([seed, 1001]))
    order = rng.permutation(n)
    # Paper cost observations require a known [0,1] bound, fixed before outputs.
    cost_upper = 1.+.001*(settings.caps.member_prompt+settings.caps.member_completion)
    policy = (CESPaidWarmupPolicy(online.phi_dimension, cost_upper_bound=cost_upper)
              if strategy == "ces" else C2MABVPolicy(reward_model=strategy.rsplit("_", 1)[1].upper(),
                  budget_threshold=1.5, delta=1./n, cost_upper_bound=cost_upper))
    initializer = calibration.new_session(settings.caps)
    init_broker = FeedbackBroker(initializer, delay=0, feedback_mode="verifiable")
    session = online.new_session(settings.caps)
    broker = FeedbackBroker(session, delay=delay, feedback_mode="verifiable")
    calibration_contexts, calibration_generated, warmup_outcomes, warmup_labels = {}, {}, [], []
    contexts, generated = {}, {}
    warmup = []
    initial_max = min(cold_start_limit(total_labels, settings.warmup_max_labels), len(calibration))
    _add_timing(totals["controller"], _elapsed(started))
    for position, row in enumerate(init_rng.permutation(len(calibration))[:initial_max]):
        started = _clock(); upper = settings.caps.service(True)
        if not ledger.can_reserve(upper):
            _add_timing(totals["controller"], _elapsed(started)); break
        combo = int(init_rng.integers(20)); rid = f"warmup:{position}"
        ledger.reserve(rid, upper)
        _add_timing(totals["controller"], _elapsed(started))
        replay = _clock(); outcome = initializer.execute(int(row), combo, include_evaluation=False)
        _add_timing(totals["environment_replay"], _elapsed(replay))
        started = _clock()
        context = calibration.context(int(row)); qid = context.question_id
        calibration_contexts[qid] = context; calibration_generated[qid] = mapper.subsets[combo]
        ledger.settle(rid, outcome["service"]+ResourceCost(0, 1, 0))
        _add_timing(totals["controller"], _elapsed(started))
        replay = _clock(); init_broker.request(int(row), (combo,), now=0)
        labels, members = init_broker.deliver(0)
        _add_timing(totals["environment_replay"], _elapsed(replay))
        started = _clock()
        member_records = _member_receipts(labels, members, calibration_contexts, calibration_generated,
                                          split="online_calibration", warmup=True)
        cost_records = _cost_receipts(context, outcome, split="online_calibration", tick=0,
                                      warmup=True, upper=cost_upper)
        if strategy == "ces":
            policy.observe_paid_warmup(member_records, now=0); policy.observe_warmup_costs(cost_records, now=0)
        else:
            policy.observe_members(member_records, now=0); policy.observe_costs(cost_records, now=0)
        warmup_labels.extend(labels); warmup_outcomes.append((context, combo, outcome))
        warmup.append({"question_id": qid, "combination_index": combo, "purchase_kind": "paid_combo_final_label",
            "feedback_units": 1, "generated_member_labels": len(members), "aggregate_labels": len(labels),
            "actual_cost": asdict(outcome["service"]+ResourceCost(0, 1, 0)),
            "source_service_execution_wall_time_s": outcome["source_service_execution_wall_time_s"]})
        _add_timing(totals["controller"], _elapsed(started))
    started = _clock()
    if strategy == "ces":
        policy.freeze_profile()
        for context, combo, outcome in warmup_outcomes:
            policy.register_execution(context.question_id, context.phi, mapper.subsets[combo],
                context.question_id+":warmup-combo", paid=True, split="online_calibration")
        for label in warmup_labels: policy.observe_combo(_combo_receipt(label, mapper, split="online_calibration"), now=0)
    initialization_cost = asdict(ledger.spent)
    _add_timing(totals["controller"], _elapsed(started))
    initialization_controller = dict(totals["controller"])
    traces = []

    def consume(labels, members, tick):
        if strategy == "ces":
            for label in labels: policy.observe_combo(_combo_receipt(label, mapper, split="online_stream"), now=tick)
        else:
            policy.observe_members(_member_receipts(labels, members, contexts, generated,
                split="online_stream", warmup=False), now=tick)

    for tick, row in enumerate(order):
        timing = {name: {"process_time_s": 0., "wall_time_s": 0.} for name in totals}
        replay = _clock(); arrived, revealed = broker.deliver(tick)
        _add_timing(timing["environment_replay"], _elapsed(replay))
        started = _clock(); consume(arrived, revealed, tick)
        context = online.context(int(row))
        before_count = len(policy._seen_combos) if strategy == "ces" else int(policy.reward_counts.sum())
        before_members = [int(policy._cap_count[i].sum()) for i in range(6)] if strategy == "ces" else policy.reward_counts.tolist()
        if strategy == "ces":
            team = policy.select(context.phi); selection = None
        else:
            selection = policy.select(now=tick, rng=rng); team = selection["members"]
        main = mapper.subsets.index(team)
        sid, pid = f"service:{tick}", f"feedback:{tick}"
        ledger.reserve_service(sid, normal)
        buy = not settings.main_feedback_natural and ledger.can_reserve(ResourceCost(0, 1, 0))
        if buy: ledger.reserve(pid, ResourceCost(0, 1, 0))
        # Purchase plan and service are frozen before current outputs/labels.
        _add_timing(timing["controller"], _elapsed(started))
        evaluation = _clock()
        if strategy == "ces": snapshot = _ces_snapshot(policy, context.phi, mapper)
        else:
            scores = [structural_reward(subset, selection["reward_means"], policy.reward_model) for subset in mapper.subsets]
            snapshot = {"predicted_combination_scores_before_decision": scores,
                "predicted_combination_working_variance_before_decision": None,
                "service_selection_values_before_decision": None,
                "prediction_supported_combination_indices": list(range(20)),
                "prediction_semantics": "native structural reward proxy, NOT predicted frozen aggregate final quality",
                "predicted_member_scores_before_decision": selection["reward_means"].tolist(),
                "native_structural_selection_before_decision": _safe(selection),
                "infinite_zero_observation_radii_serialized_as_null": True}
        _add_timing(timing["offline_evaluation"], _elapsed(evaluation))
        replay = _clock(); outcome = session.execute(int(row), main, include_evaluation=False)
        _add_timing(timing["environment_replay"], _elapsed(replay))
        started = _clock(); qid = context.question_id
        contexts[qid] = context; generated[qid] = team
        actual_service = outcome["service"]+ResourceCost(0, int(settings.main_feedback_natural), 0)
        ledger.settle(sid, actual_service)
        purchase = ResourceCost(0, int(buy), 0)
        if buy: ledger.settle(pid, purchase)
        if strategy == "ces":
            policy.register_execution(qid, context.phi, team, qid+":main-combo", paid=True, split="online_stream")
        else:
            # Executed cost is immediately available, unlike delayed correctness.
            policy.observe_costs(_cost_receipts(context, outcome, split="online_stream", tick=tick,
                warmup=False, upper=cost_upper), now=tick)
        requested = (main,) if buy or settings.main_feedback_natural else ()
        _add_timing(timing["controller"], _elapsed(started))
        replay = _clock(); broker.request(int(row), requested, now=tick)
        immediate, immediate_members = broker.deliver(tick)
        _add_timing(timing["environment_replay"], _elapsed(replay))
        started = _clock(); consume(immediate, immediate_members, tick)
        scalar = float(sum(weight*count for weight, count in zip(settings.cost_weights,
            (purchase.executions, purchase.labels, purchase.tokens))))
        _add_timing(timing["controller"], _elapsed(started))
        evaluation = _clock(); evaluated = _evaluate_frozen_service(online, int(row), main,
            reward_model=None if strategy == "ces" else policy.reward_model)
        _add_timing(timing["offline_evaluation"], _elapsed(evaluation))
        trace = {"step": tick, "question_id": qid, "task": context.task,
            "main_index": main, "main_members": list(team), "alternate_index": None,
            "action_kind": "main_only" if buy else "none", **evaluated,
            "policy_label_count_before_decision": before_count,
            "policy_label_count_semantics": "arrived aggregate labels" if strategy == "ces" else "arrived member labels",
            "policy_member_label_counts_before_decision": before_members,
            "arrived_labels_before_decision": len(arrived), "immediate_labels_after_freeze": len(immediate),
            "requested_combination_labels": len(requested), "pending_labels": broker.pending_labels,
            "legally_revealed_generated_member_feedback": len(revealed)+len(immediate_members),
            "predicted_incremental_cost": float(settings.cost_weights[1]) if buy else 0.,
            "actual_incremental_cost": scalar, "predicted_cost_error": 0.,
            "information_gain_proxy": 0., "directional_information_evaluated": False,
            "acquisition_score": 0., "service_cost": asdict(actual_service), "purchase_cost": asdict(purchase),
            "cumulative_cost": asdict(ledger.spent), "protected_future_service": asdict(ledger.future_service),
            "source_service_execution_wall_time_s": outcome["source_service_execution_wall_time_s"],
            "source_audit_execution_wall_time_s": 0., "service_frozen_before_audit": True,
            "prediction_snapshot_precedes_current_execution_and_feedback": True,
            "rounding_rejection_resampling": False, **snapshot}
        for name, measurement in timing.items():
            _add_timing(totals[name], measurement)
            for key, value in measurement.items(): trace[name+"_"+key] = value
        traces.append(_safe(trace))
    evaluation = _clock()
    per_task = {task: float(np.mean([v["quality"] for v in traces if v["task"] == task])) for task in sorted({v["task"] for v in traces})}
    _add_timing(totals["offline_evaluation"], _elapsed(evaluation))
    exporting = _clock()
    checkpoint = {"schema": "neighbor_restricted_feedback_policy_checkpoint_v1",
        "software_test_only": online.software_test_only, "strategy": strategy, "seed": int(seed),
        "label_rate": label_rate, "feedback_delay": delay, "next_tick": n, "frozen_as_of_tick": n-1,
        "pending_feedback_flushed": False, "policy_state": policy.state_dict(),
        "serving_policy": "ces_set_ucb" if strategy == "ces" else "c2_structural_reward",
        "feature_map": {"n_models": 6, "k": 3, "phi_dimension": online.phi_dimension,
            "basis_dim": mapper.basis_dim, "combinations": [list(team) for team in mapper.subsets]},
        "settings": asdict(settings), "budget": ledger.snapshot(),
        "rng_states": {"policy": rng.bit_generator.state, "warmup": init_rng.bit_generator.state},
        "feedback": _broker_checkpoint(broker), "warmup_feedback": _broker_checkpoint(init_broker),
        "stream_question_order": [online.context(int(row)).question_id for row in order],
        "online_provenance": online.public_provenance(), "calibration_provenance": calibration.public_provenance(),
        "e3_authorization": authorization, "neighbor_sources": neighbor_source_hashes(),
        "restore_boundary": "Only already arrived policy targets are present; reauthorize E3 and exact sources before replay continuation. Pending values absent and never flushed for frozen holdout."}
    checkpoint = _safe(checkpoint); export_time = _elapsed(exporting)
    measurements = {name+"_"+key: value for name, measurement in totals.items() for key, value in measurement.items()}
    measurements.update({"preflight_"+key: value for key, value in preflight.items()})
    measurements.update({"initialization_controller_"+key: value for key, value in initialization_controller.items()})
    measurements.update({"checkpoint_export_"+key: value for key, value in export_time.items()})
    measurements["scope"] = "Actual local CPU/process and wall clocks; controller excludes environment replay, evaluator snapshots/quality and checkpoint export; source model wall-time sum is not GPU kernel time."
    source_walls = [v["source_service_execution_wall_time_s"] for v in traces]
    summary = {"status": "SOFTWARE_TEST_ONLY" if online.software_test_only else "COMPLETED_RESTRICTED_FEEDBACK_SIMULATION_ONLY",
        "research_complete": False, "native_system": "NOT_RUN", "strategy": strategy, "seed": int(seed),
        "label_rate": label_rate, "feedback_delay": delay, "n_questions": n,
        "all_questions_served": len(traces) == n, "quality_micro": float(np.mean([v["quality"] for v in traces])),
        "quality_task_macro": float(np.mean(list(per_task.values()))), "per_task_quality": per_task,
        "candidate_coverage": float(np.mean([v["candidate_coverage"] for v in traces])),
        "aggregation_loss": float(np.mean([v["aggregation_loss"] for v in traces])),
        "warmup_labels": len(warmup), "warmup_cost": initialization_cost, "warmup_mode": "random_combo",
        "warmup_arrived_combo_labels": len(warmup_labels),
        "warmup_feedback_protocol": "common paid offline online_calibration combo labels; delay zero; arrived before stream",
        "arrived_original_labels": len(warmup_labels)+sum(v["arrived_labels_before_decision"]+v["immediate_labels_after_freeze"] for v in traces),
        "pending_paid_labels_at_stream_end": broker.pending_labels,
        "member_feedback_records_exposed": len(init_broker.member_history)+len(broker.member_history),
        "member_labels_used_by_policy": int(policy._cap_count.sum()) if strategy == "ces" else int(policy.reward_counts.sum()),
        "feedback_mode": "verifiable", "budget": ledger.snapshot(), "settings": asdict(settings),
        "cost_prediction_uses_current_hidden_lengths": False, "label_allocation": "same paid common warmup, then main-only paid labels until cap; no extra audit",
        "generation_replicates": 1, "stream_seed_is_not_new_generation": True,
        "native_structural_reward_is_not_final_aggregate_quality": True,
        "cost_upper_bound": cost_upper, "cost_scalar": "1 + .001*(actual member input+output tokens), divided by locked upper bound",
        "c2_soft_budget_threshold": None if strategy == "ces" else 1.5,
        "c2_rounding_does_not_enforce_single_round_cost": strategy != "ces",
        "rounding_rejection_resampling": False, "runtime_measurements": measurements,
        "controller_process_time_s": totals["controller"]["process_time_s"], "controller_wall_time_s": totals["controller"]["wall_time_s"],
        "offline_evaluation_process_time_s": totals["offline_evaluation"]["process_time_s"],
        "offline_evaluation_wall_time_s": totals["offline_evaluation"]["wall_time_s"],
        "source_service_execution_wall_time_s": None if any(v is None for v in source_walls) else sum(source_walls),
        "source_audit_execution_wall_time_s": 0.,
        "policy_checkpoint_schema": checkpoint["schema"], "e3_authorization": authorization,
        "neighbor_sources": neighbor_source_hashes(),
        "online_provenance": online.public_provenance(), "calibration_provenance": calibration.public_provenance()}
    return {"summary": _safe(summary), "traces": traces, "warmup": warmup, "policy_checkpoint": checkpoint}
