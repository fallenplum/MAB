"""Evaluate all predeclared E15 cells after joint prediction freeze."""
from __future__ import annotations

from collections import defaultdict
import time

from e15_common import (
    CAPS, CODE, DESIGNS, E13, INPUTS, LOGGERS, METHODS, OUTCOMES, PRIMARY_CAP,
    PRIVATE, PUBLIC, SEEDS, STRUCTURES, load_npz, measured, read_json, sha256,
    verify_protocol, write_csv, write_json,
)
import numpy as np


def aggregate(rows, keys, values):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, selected in sorted(grouped.items()):
        row = dict(zip(keys, key))
        row["repeated_measurement_rows"] = len(selected)
        for name in values:
            available = [r[name] for r in selected if r[name] is not None]
            row[name] = float(np.mean(available)) if available else None
        output.append(row)
    return output


def flatten_cost(record):
    total = record["ledger"]["total"]
    result = {k: record[k] for k in ("design", "token_cap", "fold", "logger_seed", "generation_seed", "structure", "questions", "common_questions", "actual_tokens", "remaining_tokens", "next_package_tokens", "complete_pool_exhausted")}
    result["remaining_fraction"] = record["remaining_tokens"]/record["token_cap"]
    result.update(total)
    result.update({"full_plan__"+k:v for k,v in record["full_plan_ledger"]["total"].items()})
    return result


def evaluate():
    protocol = verify_protocol()
    cpu, wall = time.process_time(), time.perf_counter()
    frozen = read_json(PRIVATE / "ALL_PREDICTIONS_FROZEN.json")
    if frozen["status"] != "FROZEN_BEFORE_E15_EVALUATION":
        raise ValueError("Incomplete joint freeze")
    for row in frozen["files"] + frozen["random_worlds"]:
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("Frozen prediction/feedback changed")
    write_json(PRIVATE / "EVALUATOR_FREEZE_RECEIPT.json", dict(
        protocol_sha256=sha256(PUBLIC / "E15_PROTOCOL.json"),
        freeze_sha256=sha256(PRIVATE / "ALL_PREDICTIONS_FROZEN.json"),
        evaluator_sha256=sha256(__file__), all_predictions_checked=True))
    old = read_json(E13 / "public/E13_PROTOCOL.json")
    if sha256(OUTCOMES) != old["source_assets"][1]["sha256"]:
        raise ValueError("Original outcomes changed")
    with np.load(OUTCOMES, allow_pickle=False) as data:
        y = data["aggregate_scores"].copy()
    with np.load(INPUTS, allow_pickle=False) as data:
        task, subtask = data["task_labels"].copy(), data["subtask_labels"].copy()
    mean_y = y.mean(axis=0)
    actions = load_npz(PRIVATE / "ACTIONS_FROZEN.npz")["actions"]
    hits = np.empty(actions.shape, dtype=np.int8)
    for gi in range(3):
        hits[:, :, :, gi] = y[gi, np.arange(600), actions[:, :, :, gi]]
    np.savez_compressed(PRIVATE / "PAIRED_OUTCOMES.npz", hits=hits, actions=actions, original_rewards=y)
    quality, directions = [], []
    for di, design in enumerate(DESIGNS):
        for bi, cap in enumerate(CAPS):
            for si, structure in enumerate(STRUCTURES):
                for mi, method in enumerate(METHODS):
                    values = hits[di, bi, :, :, si, mi]
                    for label, mask in [("all", np.ones(600,dtype=bool)),("gsm8k",task=="gsm8k"),("bbh",task=="bbh")]:
                        quality.append(dict(design=design, token_cap=cap, structure=structure, method=method,
                            task=label, original_questions=int(mask.sum()), measurement_events=int(values[:,:,mask].size),
                            accuracy_pct=float(100*values[:,:,mask].mean())))
                    for kind, labels in (("generation", SEEDS), ("logger",LOGGERS)):
                        for j,label in enumerate(labels):
                            v = values[:,j] if kind=="generation" else values[j]
                            directions.append(dict(design=design,token_cap=cap,structure=structure,method=method,
                                slice_type=kind,slice=str(label),accuracy_pct=float(100*v.mean())))
                    for fold in range(5):
                        e = load_npz(E13/f"private/inputs/fold{fold}/INPUTS_ONLY.npz")["E"]
                        directions.append(dict(design=design,token_cap=cap,structure=structure,method=method,
                            slice_type="fold",slice=str(fold),accuracy_pct=float(100*values[:,:,e].mean())))
                    for label in np.unique(subtask):
                        mask = subtask==label
                        directions.append(dict(design=design,token_cap=cap,structure=structure,method=method,
                            slice_type="subtask",slice=str(label),accuracy_pct=float(100*values[:,:,mask].mean())))
    write_csv(PUBLIC / "QUALITY_MATRIX.csv", quality)
    write_csv(PUBLIC / "DIRECTION_RESULTS.csv", directions)
    comparisons, decompositions = [], []
    for bi,cap in enumerate(CAPS):
        for mi,method in enumerate(METHODS):
            for alternative in (1,2):
                diff = hits[0,bi,:,:,alternative,mi]-hits[0,bi,:,:,0,mi]
                comparisons.append(pair_result(cap,method,STRUCTURES[alternative]+"-Main",diff,task))
            diff = hits[0,bi,:,:,1,mi]-hits[0,bi,:,:,2,mi]
            comparisons.append(pair_result(cap,method,"NeighborPair-RandomPair",diff,task))
            for alternative in (1,2):
                full_diff = hits[0,bi,:,:,alternative,mi]-hits[0,bi,:,:,0,mi]
                common_diff = hits[1,bi,:,:,alternative,mi]-hits[1,bi,:,:,0,mi]
                alt_coverage = hits[0,bi,:,:,alternative,mi]-hits[1,bi,:,:,alternative,mi]
                main_coverage = hits[0,bi,:,:,0,mi]-hits[1,bi,:,:,0,mi]
                if not np.array_equal(full_diff, common_diff+alt_coverage-main_coverage):
                    raise ValueError("Coverage decomposition identity failed")
                decompositions.append(dict(token_cap=cap,method=method,alternative=STRUCTURES[alternative],
                    total_quality_difference_pp=float(100*full_diff.mean()),
                    extra_combination_effect_at_common_questions_pp=float(100*common_diff.mean()),
                    alternative_added_coverage_set_pp=float(100*alt_coverage.mean()),
                    main_added_coverage_set_pp=float(100*main_coverage.mean()),identity_exact=True))
            common_nr=hits[1,bi,:,:,1,mi]-hits[1,bi,:,:,2,mi]
            comparisons.append(pair_result(cap,method,"CommonCoverage NeighborPair-RandomPair",common_nr,task))
        interaction=(hits[0,bi,:,:,1,1]-hits[0,bi,:,:,0,1])-(hits[0,bi,:,:,1,0]-hits[0,bi,:,:,0,0])
        comparisons.append(pair_result(cap,"Group-minus-RawCentered","NeighborPair-Main interaction",interaction,task))
    write_csv(PUBLIC / "PAIRED_COMPARISONS.csv", comparisons)
    write_csv(PUBLIC / "COVERAGE_DECOMPOSITION.csv", decompositions)
    records=read_json(PRIVATE/"DESIGN_RECORDS_FROZEN.json")
    costs=[flatten_cost(r) for r in records]
    write_csv(PUBLIC/"EXECUTION_COSTS.csv",costs)
    cost_fields=[k for k,v in costs[0].items() if isinstance(v,(int,float)) and k not in ("token_cap","fold","logger_seed","generation_seed")]
    cost_summary=aggregate(costs,("design","token_cap","structure"),cost_fields)
    write_csv(PUBLIC/"COST_SUMMARY.csv",cost_summary)
    cap_rows, feedback_rows = capability_diagnostics(records,y,mean_y)
    write_csv(PRIVATE/"CAPABILITY_GROUP_RESULTS.csv",cap_rows)
    write_csv(PRIVATE/"FEEDBACK_GROUP_SUPPORT.csv",feedback_rows)
    cap_fields=("table_mae_three_mean","centered_table_mse_three_mean","table_binary_mse",
        "deployed_binary_mse","deployed_accuracy_pct","empirical_selection_loss_three_mean")
    capability_summary=aggregate(cap_rows,("design","token_cap","structure","method","task"),cap_fields)
    write_csv(PUBLIC/"CAPABILITY_ERROR_SUMMARY.csv",capability_summary)
    feedback_summary=aggregate(feedback_rows,("design","token_cap","structure","task"),
        ("feedback_questions","feedback_labels","observed_action_cells","zero_support_fraction","mean_action_support","two_action_reward_disagreement_rate"))
    write_csv(PUBLIC/"SUPPORT_AND_COVERAGE.csv",feedback_summary)
    capability_links=link_capability_quality(cap_rows)
    write_csv(PUBLIC/"CAPABILITY_QUALITY_LINK.csv",capability_links)
    primary=[r for r in comparisons if r["token_cap"]==PRIMARY_CAP]
    g=next(r for r in primary if r["method"]=="Group" and r["comparison"]=="NeighborPair-Main")
    nr=next(r for r in primary if r["method"]=="Group" and r["comparison"]=="NeighborPair-RandomPair")
    interaction=next(r for r in primary if r["method"]=="Group-minus-RawCentered")
    curve_positive=sum(r["difference_pp"]>0 for r in comparisons if r["method"]=="Group" and r["comparison"]=="NeighborPair-Main" and r["token_cap"]!=PRIMARY_CAP)
    gates=dict(main_practical_gain=g["difference_pp"]>=1.0,neighbor_over_random=nr["difference_pp"]>0,
        learner_interaction=interaction["difference_pp"]>0,positive_generation_count=g["positive_generation_count"]>=2,
        positive_fold_count=g["positive_fold_count"]>=3,other_curve_points_positive=curve_positive>=2)
    strongest_other=max(r["accuracy_pct"] for r in quality if r["design"]=="TokenMatched" and r["token_cap"]==PRIMARY_CAP and r["task"]=="all" and r["method"]!="Group")
    group_neighbor=next(r["accuracy_pct"] for r in quality if r["design"]=="TokenMatched" and r["token_cap"]==PRIMARY_CAP and r["task"]=="all" and r["method"]=="Group" and r["structure"]=="NeighborPair")
    result=dict(status="COMPLETE",evidence=protocol["evidence"],primary_token_cap=PRIMARY_CAP,
        methods=METHODS,structures=STRUCTURES,token_caps=CAPS,quality=quality,comparisons=comparisons,
        coverage_decomposition=decompositions,cost_summary=cost_summary,
        capability_error_summary=capability_summary,feedback_support_summary=feedback_summary,
        capability_quality_links=capability_links,
        primary_gate=dict(conditions=gates,all_passed=all(gates.values()),
            group_neighbor_minus_strongest_other_learner_pp=group_neighbor-strongest_other,
            interpretation="descriptive development screen; no error-rate guarantee"),
        next_stage="FROZEN_CONFIRMATION_PLANNING_ONLY" if all(gates.values()) else "STOP_AUTOMATIC_EXPANSION_RETAIN_ALL_RESULTS",
        independent_questions=600,confidence_intervals=0,bootstrap_runs=0,hypothesis_tests=0,
        new_physical_llm_calls=0,fresh400_reads=0,sealed200_reads=0,
        unique_learner_packets=frozen["unique_learner_packets"],newly_fitted_packets=frozen["newly_fitted_packets"],
        legacy_reused_packets=frozen["legacy_reused_packets"],evaluation_cost=measured(cpu,wall))
    write_json(PUBLIC/"E15_COMPLETE_RESULTS.json",result)
    write_json(PRIVATE/"EVALUATION_RECEIPT.json",dict(status="COMPLETE",result_sha256=sha256(PUBLIC/"E15_COMPLETE_RESULTS.json"),
        freeze_sha256=sha256(PRIVATE/"ALL_PREDICTIONS_FROZEN.json"),confidence_intervals=0,bootstrap_runs=0,**measured(cpu,wall)))
    print("E15_EVALUATION_COMPLETE",flush=True)


def pair_result(cap,method,comparison,diff,task):
    generation_net=diff.sum(axis=(0,2)).astype(int)
    fold_net=[]
    for fold in range(5):
        e=load_npz(E13/f"private/inputs/fold{fold}/INPUTS_ONLY.npz")["E"]
        fold_net.append(int(diff[:,:,e].sum()))
    return dict(token_cap=cap,method=method,comparison=comparison,
        difference_pp=100*int(diff.sum())/diff.size,
        gsm8k_difference_pp=100*int(diff[:,:,task=="gsm8k"].sum())/diff[:,:,task=="gsm8k"].size,
        bbh_difference_pp=100*int(diff[:,:,task=="bbh"].sum())/diff[:,:,task=="bbh"].size,
        positive_generation_count=int(np.sum(generation_net>0)),positive_fold_count=sum(v>0 for v in fold_net),
        generation_net_correct=generation_net.tolist(),fold_net_correct=fold_net,
        corrected_events=None if "interaction" in comparison else int(np.sum(diff>0)),
        damaged_events=None if "interaction" in comparison else int(np.sum(diff<0)),
        positive_difference_events=int(np.sum(diff>0)),negative_difference_events=int(np.sum(diff<0)),
        net_correct_events=int(diff.sum()),
        repeated_measurement_events=int(diff.size),independent_questions=600)


def capability_diagnostics(records,y,mean_y):
    capabilities,feedback=[],[]
    for record in records:
        inputs=load_npz(E13/f"private/inputs/fold{record['fold']}/INPUTS_ONLY.npz")
        packet=load_npz(record["model_path"])
        e=inputs["E"];gi=record["generation_index"]
        count=packet["group_count"]
        query=packet["observed_local_queries"];action=packet["observed_actions"];r=packet["observed_rewards"]
        ident={k:record[k] for k in ("design","token_cap","structure","fold","logger_seed","generation_seed")}
        for group in range(28):
            mask=inputs["oracle_group_e"]==group
            if not mask.any():
                continue
            hmask=inputs["group_h"][query]==group
            unique=np.unique(query[hmask])
            disagreement=[]
            for q in unique:
                vals=r[(query==q)&hmask]
                if len(vals)==2:disagreement.append(float(vals[0]!=vals[1]))
            feedback.append(dict(**ident,task=str(inputs["group_keys"][group,0]),subtask=str(inputs["group_keys"][group,1]),
                feedback_questions=len(unique),feedback_labels=int(hmask.sum()),observed_action_cells=int(np.sum(count[group]>0)),
                zero_support_fraction=float(np.mean(count[group]==0)),mean_action_support=float(count[group].mean()),
                two_action_reward_disagreement_rate=float(np.mean(disagreement)) if disagreement else None))
            target=mean_y[e[mask]].mean(axis=0)
            for mi,method in enumerate(METHODS):
                scores=packet["scores"][mi,mask]
                table=(packet["own__C"][group] if method=="Group" else packet["sr__group_scores"][group]
                       if method=="SaveRouterFixedMask" else scores.mean(axis=0))
                chosen=packet["actions"][mi,mask]
                binary=y[gi,e[mask]]
                selected=binary[np.arange(len(chosen)),chosen]
                centered_error=(table-table.mean())-(target-target.mean())
                capabilities.append(dict(**ident,method=method,task=str(inputs["group_keys"][group,0]),
                    subtask=str(inputs["group_keys"][group,1]),evaluation_questions=int(mask.sum()),
                    capability_source="native group table, residual excluded" if method=="SaveRouterFixedMask" else
                        "group table" if method=="Group" else "mean content scores within heldout group",
                    table_mae_three_mean=float(np.abs(table-target).mean()),
                    centered_table_mse_three_mean=float(np.square(centered_error).mean()),
                    table_binary_mse=float(np.square(table[None,:]-binary).mean()),
                    deployed_binary_mse=float(np.square(scores-binary).mean()),
                    deployed_accuracy_pct=float(100*selected.mean()),
                    empirical_selection_loss_three_mean=float(np.mean(mean_y[e[mask]].max(axis=1)-mean_y[e[mask],chosen])),
                    selected_support_mean=float(count[group,chosen].mean())))
    return capabilities,feedback


def link_capability_quality(rows):
    index={(r["design"],r["token_cap"],r["fold"],r["logger_seed"],r["generation_seed"],r["method"],r["task"],r["subtask"],r["structure"]):r for r in rows}
    pairs=defaultdict(list)
    for key,alt in index.items():
        if alt["structure"]=="Main":
            continue
        base=index[key[:-1]+("Main",)]
        pairs[(alt["design"],alt["token_cap"],alt["method"],alt["task"],alt["structure"])].append((
            alt["centered_table_mse_three_mean"]-base["centered_table_mse_three_mean"],
            alt["deployed_accuracy_pct"]-base["deployed_accuracy_pct"]))
    out=[]
    for key,values in sorted(pairs.items()):
        error,quality=np.array(values).T
        corr=float(np.corrcoef(error,quality)[0,1]) if np.std(error)>0 and np.std(quality)>0 else None
        out.append(dict(zip(("design","token_cap","method","task","alternative"),key)),)
        out[-1].update(repeated_group_measurements=len(values),
            mean_centered_mse_change=float(error.mean()),mean_group_accuracy_change_pp=float(quality.mean()),
            both_error_down_quality_up=int(np.sum((error<0)&(quality>0))),
            error_down_quality_not_up=int(np.sum((error<0)&(quality<=0))),
            error_quality_change_correlation=corr,interpretation="descriptive repeated-measure correlation")
    return out


if __name__=="__main__":
    evaluate()
