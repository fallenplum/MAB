"""Evaluate the locked noise, repeat allocation and representation controls."""
from __future__ import annotations

import itertools
import time
from collections import defaultdict
from e16_common import (ASSETS,E13,INPUTS,LOGGERS,OUTCOMES,PACKAGE_BUDGETS,PRIVATE,PUBLIC,REPS,SEEDS,
    load_npz,read_json,sha256,timing,verify_protocol,write_csv,write_json)
from variance_decomposition import decompose_rewards
import numpy as np


def grouped_mean(rows,keys,fields):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    output=[]
    for key,selected in sorted(groups.items()):
        row=dict(zip(keys,key));row["measurement_rows"]=len(selected)
        row.update({f:float(np.mean([r[f] for r in selected])) for f in fields})
        output.append(row)
    return output


def main():
    verify_protocol();cpu,wall=time.process_time(),time.perf_counter()
    freeze=read_json(PRIVATE/"ALL_PREDICTIONS_FROZEN.json")
    if freeze["status"]!="FROZEN_BEFORE_E16_OUTER_EVALUATION":raise ValueError("All predictions required")
    for r in freeze["files"]:
        if sha256(r["path"])!=r["sha256"]:raise ValueError("Frozen result changed")
    write_json(PRIVATE/"EVALUATOR_FREEZE_RECEIPT.json",dict(
        protocol_sha256=sha256(PUBLIC/"E16_PROTOCOL.json"),
        prediction_sha256=sha256(PRIVATE/"ALL_PREDICTIONS_FROZEN.json")))
    with np.load(OUTCOMES,allow_pickle=False) as z:y=z["aggregate_scores"].copy()
    x=load_npz(INPUTS);task=x["task_labels"];subtask=x["subtask_labels"]
    a=decompose_rewards(y,task,subtask)
    write_json(PUBLIC/"A_VARIANCE_COMPLETE.json",a)
    a_rows=[]
    for representation,data in a["representations"].items():
        scopes={"all":data["global"]}
        for name,g in data["groupings"].items():scopes[name+"_within"]=g["pooled_within_groups"]
        for scope,r in scopes.items():
            w=r.get("within_noise_variance_w",r.get("df_weighted_within_noise_variance_w"))
            signal=np.asarray(r["estimated_signal_variance"])
            observed=np.asarray(r["observed_between_question_variance"])
            correction=np.asarray(r["noise_correction_w_over_L"])
            a_rows.append(dict(representation=representation,scope=scope,
                independent_questions=r["n_original_questions"],dimensions=len(signal),
                mean_observed_variance=float(observed.mean()),mean_generation_noise=float(np.mean(w)),
                mean_noise_of_three_mean=float(correction.mean()),mean_signal_estimate=float(signal.mean()),
                negative_signal_estimates=int(np.sum(signal<0)),positive_signal_estimates=int(np.sum(signal>0))))
    write_csv(PUBLIC/"A_VARIANCE_SUMMARY.csv",a_rows)
    cross=[];i,j=np.triu_indices(20,1);d=y[:,:,i]-y[:,:,j]
    for held in range(3):
        training=d[np.arange(3)!=held].mean(axis=0);target=d[held]
        for label,mask in (("all",np.ones(600,dtype=bool)),("gsm8k",task=="gsm8k"),("bbh",task=="bbh")):
            for gap in (.5,1.0):
                chosen=(np.abs(training[mask])==gap)
                test=target[mask][chosen];train=training[mask][chosen]
                same=int(np.sum(train*test>0));reverse=int(np.sum(train*test<0));ties=int(np.sum(test==0))
                cross.append(dict(held_generation=SEEDS[held],task=label,two_seed_abs_gap=gap,
                    strict_training_pair_events=int(chosen.sum()),same_direction=same,reversed=reverse,held_tie=ties,
                    same_fraction_all_strict=same/len(test) if len(test) else None,
                    same_fraction_both_strict=same/(same+reverse) if same+reverse else None))
    write_csv(PUBLIC/"A_CROSS_SEED_DIRECTION.csv",cross)
    actions=load_npz(PRIVATE/"ACTIONS_FROZEN.npz")
    bhits=np.stack([y[g,np.arange(600),actions["B"]] for g in range(3)],axis=-2)
    chits=np.stack([y[g,np.arange(600),actions["C"]] for g in range(3)],axis=-2)
    np.savez_compressed(PRIVATE/"PAIRED_EVALUATION.npz",B=bhits,C=chits,mean_rewards=y.mean(axis=0))
    b_results=[];b_directions=[]
    for bi,budget in enumerate(PACKAGE_BUDGETS):
        for ri,length in enumerate((1,2,3)):
            hit=bhits[bi,:,:,ri]
            for label,mask in (("all",np.ones(600,dtype=bool)),("gsm8k",task=="gsm8k"),("bbh",task=="bbh")):
                b_results.append(dict(package_budget=budget,n=budget//length,L=length,task=label,
                    accuracy_pct=float(100*hit[:,:,:,mask].mean()),independent_questions=int(mask.sum()),
                    repeated_evaluation_events=int(hit[:,:,:,mask].size)))
            for g in range(3):b_directions.append(dict(package_budget=budget,n=budget//length,L=length,
                evaluation_generation=SEEDS[g],accuracy_pct=float(100*hit[:,:,g].mean())))
    write_csv(PUBLIC/"B_QUALITY.csv",b_results);write_csv(PUBLIC/"B_SEED_RESULTS.csv",b_directions)
    b_compare=[]
    for bi,budget in enumerate(PACKAGE_BUDGETS):
        for candidate,reference in ((0,2),(0,1),(1,2)):
            diff=bhits[bi,:,:,candidate]-bhits[bi,:,:,reference]
            b_compare.append(dict(package_budget=budget,comparison=f"L{candidate+1}-L{reference+1}",
                difference_pp=100*int(diff.sum())/diff.size,
                corrected_events=int(np.sum(diff>0)),damaged_events=int(np.sum(diff<0)),
                net_correct_events=int(diff.sum()),independent_questions=600,
                generation_direction=[int(np.sum(diff[:,:,g])) for g in range(3)],
                logger_direction=[int(np.sum(diff[l])) for l in range(3)]))
    write_csv(PUBLIC/"B_PAIRED_COMPARISONS.csv",b_compare)
    costs=[]
    for r in read_json(PRIVATE/"B_DESIGN_AND_COST.json"):
        costs.append(dict(package_budget=r["package_budget"],L=r["L"],n=r["n"],fold=r["fold"],
            logger=r["logger"],rotation=r["rotation"],**r["resources"]))
    write_csv(PUBLIC/"B_EXECUTION_COSTS.csv",costs)
    fields=list(costs[0])[6:]
    b_costs=grouped_mean(costs,("package_budget","L","n"),fields)
    write_csv(PUBLIC/"B_COST_SUMMARY.csv",b_costs)
    c_results=[];c_slices=[];c_metrics=[]
    for mi,rep in enumerate(REPS):
        hit=chits[mi]
        for label,mask in (("all",np.ones(600,dtype=bool)),("gsm8k",task=="gsm8k"),("bbh",task=="bbh")):
            c_results.append(dict(representation=rep,task=label,accuracy_pct=float(100*hit[:,mask].mean()),
                                  independent_questions=int(mask.sum())))
        for g in range(3):c_slices.append(dict(representation=rep,slice_type="generation",slice=str(SEEDS[g]),accuracy_pct=float(100*hit[g].mean())))
        for fold in range(5):
            old=load_npz(E13/f"private/inputs/fold{fold}/INPUTS_ONLY.npz");e=old["E"]
            packet=load_npz(PRIVATE/f"C/fold{fold}/{rep}_MODEL.npz")
            target=y[:,e].mean(axis=0);scores=packet["scores"]
            c_slices.append(dict(representation=rep,slice_type="fold",slice=str(fold),accuracy_pct=float(100*hit[:,e].mean())))
            c_metrics.append(dict(representation=rep,fold=fold,
                absolute_mse=float(np.square(scores-target).mean()),
                centered_mse=float(np.square((scores-scores.mean(axis=1,keepdims=True))-(target-target.mean(axis=1,keepdims=True))).mean()),
                empirical_selection_loss=float(np.mean(target.max(axis=1)-target[np.arange(120),packet["actions"]]))))
        for label in np.unique(subtask):c_slices.append(dict(representation=rep,slice_type="subtask",slice=str(label),accuracy_pct=float(100*hit[:,subtask==label].mean())))
    write_csv(PUBLIC/"C_QUALITY.csv",c_results);write_csv(PUBLIC/"C_SEED_FOLD_SUBTASK.csv",c_slices)
    write_csv(PUBLIC/"C_VALUE_ERRORS.csv",c_metrics)
    c_comparisons=[]
    for candidate,reference in ((1,0),(3,1),(3,2),(4,3)):
        diff=chits[candidate]-chits[reference]
        c_comparisons.append(dict(comparison=REPS[candidate]+"-"+REPS[reference],
            difference_pp=100*int(diff.sum())/diff.size,
            corrected_events=int(np.sum(diff>0)),damaged_events=int(np.sum(diff<0)),
            net_correct_events=int(diff.sum()),independent_questions=600,
            generation_direction=[int(diff[g].sum()) for g in range(3)]))
    write_csv(PUBLIC/"C_PAIRED_COMPARISONS.csv",c_comparisons)
    selections=read_json(PRIVATE/"C_SELECTION_RECORDS.json")
    public_selection=[{k:v for k,v in r.items() if k not in ("model_path","model_sha256","common_history_resources")} for r in selections]
    write_json(PUBLIC/"C_MODEL_SELECTION.json",public_selection)
    history_cost=[dict(fold=f,**next(r["common_history_resources"] for r in selections if r["fold"]==f)) for f in range(5)]
    write_csv(PUBLIC/"C_COMMON_HISTORY_COST.csv",history_cost)
    e10receipt=read_json(ASSETS/"research/e10_input_prefill_matching_audit/private/features/FEATURE_FREEZE_RECEIPT.json")
    cost=dict(frozen_training=freeze,evaluation=timing(cpu,wall),
        historical_qwen_encoding={k:e10receipt[k] for k in ("requests","formal_input_tokens","formal_gpu_seconds","model_load_wall_seconds")},
        current_new_encoder_forwards=0,current_new_llm_generation_calls=0,confidence_intervals=0,bootstrap_runs=0)
    # Do not publish private file manifests through the computation summary.
    cost["frozen_training"]={k:v for k,v in freeze.items() if k!="files"}
    write_json(PUBLIC/"COMPUTE_AND_HISTORY_COST.json",cost)
    result=dict(status="COMPLETE",evidence="REUSED_TRAIN600_EXPLORATORY_DEVELOPMENT",original_questions=600,
        A=a_rows,B=dict(quality=b_results,comparisons=b_compare,costs=b_costs),
        C=dict(quality=c_results,comparisons=c_comparisons,metrics=c_metrics),
        B_actual_fits=freeze["B_actual_fits"],C_actual_fits=freeze["C_actual_fits"],qualification_fits=5,
        confidence_intervals=0,bootstrap_runs=0,new_llm_generation_calls=0,new_encoder_forwards=0,
        validation200_reads=0,fresh400_reads=0,sealed200_reads=0)
    write_json(PUBLIC/"E16_COMPLETE_RESULTS.json",result)
    write_json(PRIVATE/"EVALUATION_RECEIPT.json",dict(status="COMPLETE",result_sha256=sha256(PUBLIC/"E16_COMPLETE_RESULTS.json"),**timing(cpu,wall)))
    print("E16_EVALUATION_COMPLETE",flush=True)


if __name__=="__main__":main()
