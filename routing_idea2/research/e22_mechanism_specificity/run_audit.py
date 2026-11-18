"""Fixed matched-row controls on the original train600 outer folds."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp

CODE = Path(__file__).resolve().parent
ASSETS = Path(r"E:\语言模型组合路由")
E19 = CODE.parent / "e19_symmetric_tie_constraint"
P3 = CODE.parent / "e21_p3_external_mechanism"
sys.path.insert(0, str(E19))
sys.path.insert(0, str(P3))
import run_e19 as old
from run_internal import summarize
from row_weighted import fit
from geometry import analyze

SEEDS = (2026111801, 2026111802, 2026111803)
CONDITIONS = ("B", "C", "Uniform") + tuple(f"Matched{seed}" for seed in SEEDS)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_binding():
    return {name: digest(CODE / name) for name in ("PROTOCOL.json", "run_audit.py", "row_weighted.py", "geometry.py")}


def controls(fold, phi, empirical, task, subtask, canonical):
    tied = np.all(empirical == empirical[:, :1], axis=1)
    norms = np.linalg.norm(phi[:, 1:], axis=1)
    groups = defaultdict(list)
    for index, label in enumerate(zip(task, subtask)):
        groups[label].append(index)
    strata = []
    for positions in groups.values():
        positions = np.array(positions)
        median = np.median(norms[positions])
        strata.extend([part for part in (positions[norms[positions] <= median],
                                         positions[norms[positions] > median]) if len(part)])
    weights = dict(B=np.ones(len(phi)), C=(~tied).astype(float),
                   Uniform=np.full(len(phi), 1-float(tied.mean())))
    for seed in SEEDS:
        weight = np.ones(len(phi))
        for positions in strata:
            count = int(tied[positions].sum())
            order = sorted(positions, key=lambda index: hashlib.sha256(
                f"{seed}|{fold}|{canonical[index]}".encode()).digest())
            weight[order[:count]] = 0
            if np.sum(weight[positions] == 0) != count:
                raise AssertionError("Matched stratum counts differ")
        weights[f"Matched{seed}"] = weight
    if any(abs(weight.sum()-weights["C"].sum()) > 1e-9 for key, weight in weights.items() if key != "B"):
        raise AssertionError("Matched effective training masses differ")
    report = dict(history_questions=len(phi), tied_questions=int(tied.sum()),
                  matched_strata=len(strata), pure_stratum_questions=int(sum(
                      len(part) for part in strata if tied[part].all() or (~tied[part]).all())),
                  controls={})
    for name, weight in weights.items():
        report["controls"][name] = dict(
            weight_sum=float(weight.sum()), zero_rows=int((weight == 0).sum()),
            removed_tied_mass=float(((1-weight)*tied).sum()),
            removed_nontied_mass=float(((1-weight)*(~tied)).sum()),
            tied_zero_overlap=int(((weight == 0) & tied).sum()),
            fraction_of_tied_mass_removed=float(((1-weight)*tied).sum()/tied.sum()),
        )
    return weights, report


def reuse(fold, family, packet):
    if family == "tspo":
        root = ASSETS / "research/e19_symmetric_tie_constraint/private" / f"fold{fold}"
        receipt = read(root / "PREDICTIONS_FROZEN.json")
        arrays = []
        for branch in ("B", "C"):
            path = root / f"{branch}_PREDICTIONS.npz"
            if digest(path) != receipt["predictions"][branch]["sha256"]:
                raise ValueError("Frozen E19 endpoint model changed")
            with np.load(path, allow_pickle=False) as archive:
                if not np.array_equal(packet["H"], archive["H"]) or not np.array_equal(packet["E"], archive["E"]):
                    raise ValueError("Frozen endpoint split differs")
                arrays.append(archive["W"].copy())
        return np.array(arrays)
    root = ASSETS / "research/e21_p3_external_mechanism/internal_run_v1/private"
    receipt = read(root / "PREDICTIONS_FROZEN.json")
    path = root / f"fold{fold}_PREDICTIONS.npz"
    if digest(path) != receipt["folds"][fold]["predictions_sha256"]:
        raise ValueError("Frozen P3 endpoint model changed")
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["E"], packet["E"]):
            raise ValueError("Frozen P3 split differs")
        return archive["W"].copy()


def listnet_gradient(phi, empirical, coefficient, weights):
    target = np.exp(empirical-logsumexp(empirical, axis=1, keepdims=True))
    scores = phi @ coefficient
    prediction = np.exp(scores-logsumexp(scores, axis=1, keepdims=True))
    return weights[:, None]*(prediction-target)


def history_setup():
    protocol = read(E19 / "E19_PROTOCOL.json")
    audit = read(E19 / "FROZEN_TSPO_REUSE_AUDIT.json")
    if digest(old.DEFAULT_INPUTS) != old.INPUT_SHA or digest(old.DEFAULT_REWARDS) != old.REWARD_SHA:
        raise ValueError("Frozen original train600 source differs")
    with np.load(old.DEFAULT_INPUTS, allow_pickle=False) as archive:
        metadata = {name: archive[name].copy()
                    for name in ("task_labels", "subtask_labels", "canonical_hashes")}
    return protocol, audit, metadata


def train(args):
    protocol, evidence, metadata = history_setup()
    binding = source_binding()
    root = args.root / "private"
    root.mkdir(parents=True, exist_ok=False)
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    folds = []
    for fold in range(5):
        packet, provenance = old.load_packet(fold, protocol, evidence)
        history = packet["H"]
        rewards, _ = old.read_reward_rows(old.DEFAULT_REWARDS, history)
        empirical = rewards.mean(axis=0)
        weights, match = controls(fold, packet["phi_h"], empirical,
                                  metadata["task_labels"][history], metadata["subtask_labels"][history],
                                  metadata["canonical_hashes"][history])
        folder = root / f"fold{fold}"
        folder.mkdir()
        old.write_npz(folder / "MASKS_FROZEN.npz", **weights, H=history)
        summaries, artifacts = {}, []
        for family in ("tspo", "listnet"):
            existing = reuse(fold, family, packet)
            coefficient_map = dict(zip(("B", "C"), existing))
            row_gradients, qualifications = {}, {}
            lam = protocol["regularization_by_fold"][fold]
            for name in CONDITIONS:
                if name in ("B", "C") and family == "listnet":
                    row_gradients[name] = listnet_gradient(packet["phi_h"], empirical, coefficient_map[name], weights[name])
                else:
                    coefficient, diag, row_gradient = fit(
                        family, packet["phi_h"], empirical, lam, weights[name])
                    qualifications[name] = diag
                    if name in ("B", "C"):
                        delta = packet["phi_h"] @ (coefficient-coefficient_map[name])
                        delta -= delta.mean(axis=1, keepdims=True)
                        if np.max(np.abs(delta)) > 1e-7:
                            raise ValueError("TSPO dual recovery changed original centered predictions")
                        diag["reused_centered_prediction_error"] = float(np.abs(delta).max())
                    else:
                        coefficient_map[name] = coefficient
                    row_gradients[name] = row_gradient
            geometry, geometry_arrays = analyze(
                family, packet["phi_h"], packet["phi_e"], empirical, lam, existing[0], existing[1],
                row_gradients["B"], row_gradients["C"])
            parameter_norm = {key: float(np.linalg.norm(value)) for key, value in coefficient_map.items()}
            scores = np.array([packet["phi_e"] @ coefficient_map[key] for key in CONDITIONS])
            path = folder / f"{family}_FROZEN.npz"
            old.write_npz(path, coefficients=np.array([coefficient_map[key] for key in CONDITIONS]),
                          E=packet["E"], scores=scores, conditions=np.array(CONDITIONS),
                          **geometry_arrays)
            artifacts.append(dict(family=family, filename=path.name, sha256=digest(path)))
            baseline_gradient = row_gradients["B"]
            removed_pressure = {}
            for key, weight in weights.items():
                pressure = packet["phi_h"].T @ ((1-weight)[:, None]*baseline_gradient) / len(history)
                removed_pressure[key] = float(np.linalg.norm(pressure))
            summaries[family] = dict(geometry=geometry, fits=qualifications,
                                    parameter_norm=parameter_norm, removed_data_gradient_norm_at_B=removed_pressure)
        record = dict(fold=fold, lambda_value=lam, matching=match, families=summaries,
                      frozen_input_sha256=provenance["input_packet_sha256"], artifacts=artifacts,
                      masks_sha256=digest(folder / "MASKS_FROZEN.npz"))
        write(folder / "FIT_SUMMARY.json", record)
        folds.append(record)
        print(json.dumps(dict(fold=fold, status="FROZEN")), flush=True)
    receipt = dict(status="ALL_CONTROLS_AND_GEOMETRY_FROZEN", source_binding=binding, folds=folds,
                   new_control_fits=40, dual_recovery_fits=10, reused_endpoint_heads=20,
                   fit_cpu_seconds=time.process_time()-started_cpu, fit_wall_seconds=time.perf_counter()-started_wall,
                   new_llm_calls=0, evaluation_rewards_read=0)
    write(root / "FROZEN.json", receipt)


def evaluate(args):
    protocol, evidence, metadata = history_setup()
    receipt = read(args.root / "private/FROZEN.json")
    if receipt["source_binding"] != source_binding():
        raise ValueError("Frozen audit source changed")
    for item in receipt["folds"]:
        for artifact in item["artifacts"]:
            if digest(args.root / "private" / f"fold{item['fold']}" / artifact["filename"]) != artifact["sha256"]:
                raise ValueError("Frozen audit artifact changed")
    indices, scores_by_family = [], defaultdict(list)
    for item in receipt["folds"]:
        fold = item["fold"]
        for family in ("tspo", "listnet"):
            with np.load(args.root / "private" / f"fold{fold}" / f"{family}_FROZEN.npz", allow_pickle=False) as archive:
                scores_by_family[family].append(archive["scores"].copy())
                if family == "tspo":
                    indices.append(archive["E"].copy())
    ids = np.concatenate(indices)
    if not np.array_equal(np.sort(ids), np.arange(600)):
        raise ValueError("The five evaluation folds do not partition train600")
    rewards, access = old.read_reward_rows(old.DEFAULT_REWARDS, ids)
    task, subtask = metadata["task_labels"][ids], metadata["subtask_labels"][ids]
    summaries = {}
    for family, blocks in scores_by_family.items():
        scores = np.concatenate(blocks, axis=1)
        control_metrics = {}
        for index, name in enumerate(CONDITIONS[1:], 1):
            pair = scores[[0, index]]
            control_metrics[name] = dict(
                primary=summarize(pair, task, subtask, rewards, 1e-8),
                raw_exact=summarize(pair, task, subtask, rewards, 0))
        utilities = np.array([np.einsum("qa,gqa->gq", old.uniform_policy(condition, 1e-8)["probabilities"], rewards)
                              for condition in scores])
        per_fold = []
        for fold in range(5):
            selected = slice(fold*120, (fold+1)*120)
            per_fold.append(dict(fold=fold, accuracies_pct={
                name: 100*old.task_macro(utilities[index][:, selected], task[selected])
                for index, name in enumerate(CONDITIONS)}))
        matched_mean = utilities[3:].mean(axis=0)
        c_minus_matched = 100*old.task_macro(utilities[1]-matched_mean, task)
        c_minus_uniform = 100*old.task_macro(utilities[1]-utilities[2], task)
        summaries[family] = dict(
            control_metrics=control_metrics, C_minus_Uniform_pp=c_minus_uniform,
            C_minus_MatchedMean_pp=c_minus_matched, per_fold=per_fold,
            baseline_accuracy_pct=100*old.task_macro(utilities[0], task),
            accuracies_pct={name: 100*old.task_macro(utilities[index], task) for index, name in enumerate(CONDITIONS)})
    result = dict(schema="mab_specificity_audit_results_v1", status="COMPLETE", source_binding=receipt["source_binding"],
                  folds=receipt["folds"], outcome_results=summaries, training_questions=600,
                  evaluation_scope="Old train600 development OOF", recorded_generation_events_per_policy=1800,
                  new_control_fits=40, dual_recovery_fits=10, reused_endpoint_heads=20,
                  fit_cpu_seconds=receipt["fit_cpu_seconds"], fit_wall_seconds=receipt["fit_wall_seconds"],
                  new_llm_calls=0, new_llm_tokens=0, bootstrap_repetitions=0)
    write(args.root / "public/RESULTS.json", result)
    print(json.dumps({key: {field: value for field, value in row.items() if field in
                           ("C_minus_Uniform_pp", "C_minus_MatchedMean_pp", "accuracies_pct")}
                      for key, row in summaries.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("fit", "evaluate"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    (train if args.phase == "fit" else evaluate)(args)
