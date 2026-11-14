"""Freeze repetition-budget and input-representation diagnostic predictions."""
from __future__ import annotations

import argparse
import ast
import io
import time
import unittest

from e16_common import (
    ASSETS,CODE,E8,E10,E13,INPUTS,LOGGERS,MULTIPLIERS,OUTCOMES,PACKAGE_BUDGETS,
    PRIVATE,PUBLIC,QWEN,QWEN_SHA,REPS,SEEDS,array_hash,load_input_features,load_npz,
    read_json,selected_labels,sha256,source_files,timing,verify_protocol,write_csv,write_json,
)
from learning import balanced_order,feature_designs,fit_ridge,seed_subset
import numpy as np


def prepare():
    cpu,wall=time.process_time(),time.perf_counter()
    PUBLIC.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    for path in source_files():ast.parse(path.read_text(encoding="utf-8"))
    log=io.StringIO()
    tested=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.discover(str(CODE/"tests")))
    (PRIVATE/"QUALIFICATION.log").write_text(log.getvalue(),encoding="utf-8")
    if not tested.wasSuccessful():raise RuntimeError(log.getvalue())
    if sha256(QWEN)!=QWEN_SHA:raise ValueError("Original Qwen feature cache changed")
    original=read_json(E13/"public/E13_PROTOCOL.json")
    for path,item in ((INPUTS,original["source_assets"][0]),(OUTCOMES,original["source_assets"][1])):
        if sha256(path)!=item["sha256"]:raise ValueError("Original frozen source changed")
    inputs,qwen=load_input_features()
    manifest=read_json(E10/"private/input/PREFILL_INPUT_MANIFEST.json")
    # The exact record structure is checked without exposing question identities.
    records=manifest if isinstance(manifest,list) else manifest.get("records",manifest.get("requests",manifest.get("rows")))
    if records is None:raise ValueError("Unrecognized original Qwen input provenance")
    if isinstance(records,dict):records=list(records.values())
    if len(records)!=600:raise ValueError("Qwen question manifest length changed")
    for i,row in enumerate(records):
        if row["global_index"]!=i or row["question_id"]!=inputs["question_ids"][i] or row["canonical_id"]!=inputs["canonical_hashes"][i]:
            raise ValueError("Qwen and MiniLM identities differ")
    if not all(np.isfinite(qwen[k]).all() for k in ("full","matched")):
        raise ValueError("Nonfinite frozen Qwen input vectors")
    different=np.any(qwen["full"]!=qwen["matched"],axis=1)
    if not np.array_equal(different,qwen["truncated"]):raise ValueError("Unexpected visible-input mismatch scope")
    files=[]
    def add(path):files.append(dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size))
    for path in (INPUTS,OUTCOMES,QWEN,E10/"private/input/PREFILL_INPUT_MANIFEST.json",E10/"private/features/FEATURE_FREEZE_RECEIPT.json",E13/"public/E13_PROTOCOL.json"):
        add(path)
    split_arrays={}
    restore=[];all_e=[]
    for fold in range(5):
        input_path=E13/f"private/inputs/fold{fold}/INPUTS_ONLY.npz"
        if sha256(input_path)!=original["input_packets"][fold]["sha256"]:
            raise ValueError("Original outerfold preprocessing packet changed")
        add(input_path);p=load_npz(input_path)
        if len(p["H"])!=480 or len(p["E"])!=120 or set(p["H"])&set(p["E"]):
            raise ValueError("Illegal outerfold scope")
        all_e.extend(p["E"].tolist())
        split_arrays[f"f{fold}_H"]=p["H"];split_arrays[f"f{fold}_E"]=p["E"]
        fitlog=E8/f"private/training/fold{fold}/M0_tuned/FIT_LOG.json"
        model_path=fitlog.with_name("model_state.npz")
        add(fitlog);add(model_path)
        source=read_json(fitlog)
        for j,inner in enumerate(source["candidates"][0]["fits"]):
            tr=np.array(inner["fitting_rows"],dtype=int);va=np.array(inner["held_rows"],dtype=int)
            if len(tr)!=320 or len(va)!=160 or set(tr)&set(va) or set(tr)|set(va)!=set(range(480)):
                raise ValueError("Original legal inner fold changed")
            split_arrays[f"f{fold}_i{j}_train"]=tr;split_arrays[f"f{fold}_i{j}_validation"]=va
        for logger in LOGGERS:
            split_arrays[f"f{fold}_l{logger}_order"]=balanced_order(p["H"],inputs["task_labels"],logger,fold)
        old=load_npz(model_path)
        multiplier=float(source["selected_parameters"]["lambda_multiplier"])
        coefficient,residual=fit_ridge(p["phi_h"],selected_labels(p["H"],[0,1,2]).mean(axis=0),100*multiplier/3)
        difference=float(np.max(np.abs(p["phi_e"]@(coefficient-old["coefficient"]))))
        if difference>1e-10 or not np.array_equal(np.argmax(p["phi_e"]@coefficient,axis=1),np.argmax(p["phi_e"]@old["coefficient"],axis=1)):
            raise ValueError("Original direct20 Ridge recovery failed")
        restore.append(dict(fold=fold,max_score_error=difference,normal_equation_residual=residual,actions_identical=True))
    if len(all_e)!=600 or len(set(all_e))!=600:raise ValueError("Outer questions do not partition600")
    np.savez_compressed(PRIVATE/"SPLITS_AND_ORDERS_FROZEN.npz",**split_arrays)
    write_json(PRIVATE/"INPUT_MANIFEST.json",dict(files=files))
    write_json(PUBLIC/"QUALIFICATION.json",dict(status="PASS",tests=tested.testsRun,all_source_syntax_valid=True,
        original_ridge_recovery=restore,qualification_ridge_fits=5,qwen_question_mapping_valid=600,
        qwen_full_matched_different_rows=int(qwen["truncated"].sum()),**timing(cpu,wall)))
    protocol=dict(schema="E16_signal_repetition_representation_v1",branch="codex/e16-signal-replication-representation",
        parent_commit="39ba966be8fce97867b23ecd4f0ae1a5f502e862",
        evidence="REUSED_TRAIN600_EXPLORATORY_DEVELOPMENT",questions=600,generation_seeds=SEEDS,
        models=6,team_size=3,actions=20,aggregator="frozen V2 Qwen3",outerfolds=5,
        A=dict(targets="raw actions, within-question centered actions, all190 actual pair differences",
            formula="observed question-mean variance minus mean within-question generation variance /3",
            negative_estimates="retain untruncated",interpretation="method of moments under conditionally uncorrelated repeats",
            within_subtask="SSE/pooled df minus df-weighted within noise/3"),
        B=dict(question_seed_package_budgets=PACKAGE_BUDGETS,primary_budget=480,repeats=(1,2,3),
            main_cells=[dict(n=480,L=1),dict(n=240,L=2),dict(n=160,L=3)],logger_seeds=LOGGERS,
            selected_seed_subsets="all three singles/all three pairs/all-three once, rotations share cached all-three model",
            features="original H480 input-fitted PCA16 held fixed; unlabelled history inputs common",
            ridge="individual scalar-generation squared loss +100*entire coefficient norm; mean target alpha100/L",
            primary_comparison="L1 n480 minus L3 n160; L2 secondary",cost="equal nL gives same six-member/20-aggregate package count, gold and actual tokens separately reported"),
        C=dict(representations=REPS,primary_comparisons=["MiniLM64-MiniLM16","FusionMatched64-MiniLM64"],
            qwen_main="matched visible input, last-token frozen block24 vector",full_fusion="predeclared sensitivity,6 input differences",
            training="full H480 x3 rewards, direct20 linear ridge only",parameters=dict(MiniLM16=340,all_64=1300),
            scaling="unit variance per fitted PC; fusion32+32 directly concatenate, total variance64",
            preprocessing="PCA/scaler refitted inside each legitimate training fold",
            inner_folds="original E8 M0_tuned fitting_rows/held_rows",lambda_multipliers=MULTIPLIERS,
            mean_target_alpha="100*multiplier/3",selection="mean inner actual routing quality, tie stronger regularization",
            regularization_warning="same numerical alpha differs in effective complexity; PCA-only design can make candidate action rankings identical"),
        roles="old train600 OOF; validation200/fresh400/sealed200 not read",
        confidence_intervals=0,bootstrap_runs=0,new_language_model_calls=0,new_embedding_forwards=0,
        next_stage="diagnostic reporting only; no automatic new acquisition or generation",
        source_hashes={str(p.relative_to(CODE)):sha256(p) for p in source_files()},
        split_sha256=sha256(PRIVATE/"SPLITS_AND_ORDERS_FROZEN.npz"),
        input_manifest_sha256=sha256(PRIVATE/"INPUT_MANIFEST.json"))
    write_json(PUBLIC/"E16_PROTOCOL.json",protocol)
    write_json(PRIVATE/"PROTOCOL_FREEZE_RECEIPT.json",dict(status="FROZEN",protocol_sha256=sha256(PUBLIC/"E16_PROTOCOL.json")))
    print("E16_PROTOCOL_FROZEN",flush=True)


def package_resources(rows,seeds):
    with np.load(OUTCOMES,allow_pickle=False) as z:
        index=(np.asarray(seeds)[:,None],np.asarray(rows)[None,:])
        r=dict(gold=len(rows),member_calls=int(z["member_execution_counts"][index].sum()),
            aggregator_calls=int(z["aggregate_execution_counts"][index].sum()),scoring_ops=20*len(rows)*len(seeds))
        for stem in ("member","aggregate"):
            for field in ("prompt_tokens","completion_tokens"):
                value=z[stem+"_"+field][np.asarray(seeds)[:,None],np.asarray(rows)[None,:]].sum()
                r[stem+"_"+field]=int(value)
        r["input_tokens"]=r["member_prompt_tokens"]+r["aggregate_prompt_tokens"]
        r["output_tokens"]=r["member_completion_tokens"]+r["aggregate_completion_tokens"]
        r["tokens"]=r["input_tokens"]+r["output_tokens"]
    return r


def freeze():
    verify_protocol()
    if (PRIVATE/"ALL_PREDICTIONS_FROZEN.json").exists():raise FileExistsError("Already frozen")
    cpu,wall=time.process_time(),time.perf_counter()
    x,qwen=load_input_features();splits=load_npz(PRIVATE/"SPLITS_AND_ORDERS_FROZEN.npz")
    actions_b=np.full((2,3,3,3,600),-1,dtype=np.int16)
    actions_c=np.full((5,600),-1,dtype=np.int16)
    b_records=[];c_records=[];all_files=[];b_cache={};b_fit_count=0;c_fit_count=0
    for fold in range(5):
        h,e=splits[f"f{fold}_H"],splits[f"f{fold}_E"]
        old=load_npz(E13/f"private/inputs/fold{fold}/INPUTS_ONLY.npz")
        local={int(q):i for i,q in enumerate(h)}
        for bi,budget in enumerate(PACKAGE_BUDGETS):
            for li,logger in enumerate(LOGGERS):
                order=splits[f"f{fold}_l{logger}_order"]
                for ri,length in enumerate((1,2,3)):
                    n=budget//length;rows=np.sort(order[:n])
                    for rotation in range(3):
                        seeds=seed_subset(length,rotation)
                        key=(fold,tuple(rows),tuple(seeds))
                        if key not in b_cache:
                            start_cpu,start_wall=time.process_time(),time.perf_counter()
                            mean=selected_labels(rows,seeds).mean(axis=0)
                            design=old["phi_h"][[local[int(q)] for q in rows]]
                            w,residual=fit_ridge(design,mean,100/length)
                            scores=old["phi_e"]@w;actions=np.argmax(scores,axis=1)
                            path=PRIVATE/f"B/fold{fold}/model{b_fit_count}.npz";path.parent.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(path,weights=w,scores=scores,actions=actions,history=rows,seeds=seeds,evaluation=e)
                            b_cache[key]=(actions,path)
                            all_files.append(dict(path=str(path),sha256=sha256(path)))
                            b_fit_count+=1
                            fit=dict(cpu_s=time.process_time()-start_cpu,wall_s=time.perf_counter()-start_wall,normal_residual=residual)
                        else:fit=dict(cpu_s=0.0,wall_s=0.0,normal_residual=None)
                        actions,path=b_cache[key]
                        actions_b[bi,li,rotation,ri,e]=actions
                        b_records.append(dict(fold=fold,package_budget=budget,logger=logger,rotation=rotation,L=length,n=n,
                            training_seed_indices=seeds.tolist(),model_path=str(path),model_sha256=sha256(path),
                            resources=package_resources(rows,seeds),fitting_cost=fit))
        c_root=PRIVATE/f"C/fold{fold}";c_root.mkdir(parents=True,exist_ok=True)
        histories=[]
        for j in range(3):
            tr=splits[f"f{fold}_i{j}_train"];va=splits[f"f{fold}_i{j}_validation"]
            designs,state=feature_designs(x["embeddings"],qwen["matched"],qwen["full"],h[tr],h[va])
            np.savez_compressed(c_root/f"INNER{j}_FEATURES.npz",history=h[tr],evaluation=h[va],**state)
            labels=selected_labels(h[tr],[0,1,2]).mean(axis=0)
            val=selected_labels(h[va],[0,1,2]).mean(axis=0)
            for rep in REPS:
                ph,pe=designs[rep]
                for multiplier in MULTIPLIERS:
                    w,residual=fit_ridge(ph,labels,100*multiplier/3);c_fit_count+=1
                    predicted=np.argmax(pe@w,axis=1)
                    quality=float(val[np.arange(len(va)),predicted].mean())
                    histories.append(dict(fold=fold,inner=j,representation=rep,multiplier=multiplier,
                        accuracy=quality,normal_residual=residual,actions=predicted.tolist()))
        outer,state=feature_designs(x["embeddings"],qwen["matched"],qwen["full"],h,e)
        np.savez_compressed(c_root/"OUTER_FEATURES.npz",history=h,evaluation=e,**state)
        labels=selected_labels(h,[0,1,2]).mean(axis=0)
        for mi,rep in enumerate(REPS):
            values={mult:np.mean([r["accuracy"] for r in histories if r["representation"]==rep and r["multiplier"]==mult]) for mult in MULTIPLIERS}
            best=max(values.values());chosen=max(m for m in MULTIPLIERS if values[m]>=best-1e-12)
            ph,pe=outer[rep]
            w,residual=fit_ridge(ph,labels,100*chosen/3);c_fit_count+=1
            scores=pe@w;actions=np.argmax(scores,axis=1)
            path=c_root/(rep+"_MODEL.npz")
            np.savez_compressed(path,weights=w,scores=scores,actions=actions,history=h,evaluation=e)
            actions_c[mi,e]=actions
            all_files.append(dict(path=str(path),sha256=sha256(path)))
            c_records.append(dict(fold=fold,representation=rep,multiplier=chosen,inner_quality=values,
                common_history_resources=package_resources(h,[0,1,2]),
                grid_boundary=chosen in (min(MULTIPLIERS),max(MULTIPLIERS)),parameters=int(w.size),
                weight_norm=float(np.linalg.norm(w)),normal_residual=residual,
                max_gram_off_diagonal=float(np.max(np.abs(ph.T@ph-np.diag(np.diag(ph.T@ph))))),
                train_total_feature_variance=float(np.var(ph[:,1:],axis=0).sum()),
                inner_grid_actions_identical=all(len({tuple(r["actions"]) for r in histories if r["representation"]==rep and r["inner"]==j})==1 for j in range(3)),
                model_path=str(path),model_sha256=sha256(path)))
        # Inner actions contain no public question IDs and remain in private.
        write_json(c_root/"INNER_SELECTION_LOG.json",histories)
        print(f"E16_FROZEN fold={fold} B_unique={b_fit_count} C_fits={c_fit_count}",flush=True)
    if np.any(actions_b<0) or np.any(actions_c<0):raise ValueError("Prediction coverage incomplete")
    np.savez_compressed(PRIVATE/"ACTIONS_FROZEN.npz",B=actions_b,C=actions_c)
    write_json(PRIVATE/"B_DESIGN_AND_COST.json",b_records)
    write_json(PRIVATE/"C_SELECTION_RECORDS.json",c_records)
    for path in (PRIVATE/"ACTIONS_FROZEN.npz",PRIVATE/"B_DESIGN_AND_COST.json",PRIVATE/"C_SELECTION_RECORDS.json"):
        all_files.append(dict(path=str(path),sha256=sha256(path)))
    for path in sorted((PRIVATE/"C").rglob("*")):
        if path.is_file() and not any(p["path"]==str(path) for p in all_files):all_files.append(dict(path=str(path),sha256=sha256(path)))
    write_json(PRIVATE/"ALL_PREDICTIONS_FROZEN.json",dict(status="FROZEN_BEFORE_E16_OUTER_EVALUATION",files=all_files,
        B_actual_fits=b_fit_count,C_actual_fits=c_fit_count,qualification_fits=5,
        protocol_sha256=sha256(PUBLIC/"E16_PROTOCOL.json"),new_llm_calls=0,new_encoder_forwards=0,**timing(cpu,wall)))
    print("E16_ALL_PREDICTIONS_FROZEN",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("prepare","freeze"))
    args=parser.parse_args();prepare() if args.command=="prepare" else freeze()
