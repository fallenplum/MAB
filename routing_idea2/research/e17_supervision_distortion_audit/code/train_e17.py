"""100 predeclared renamings and same-draw loss-order experiments."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor,as_completed
import io
import time
import unittest

from e17_common import (ASSETS,AUDIT,CODE,E8,E13,INPUTS,MC_COUNTS,METHODS,NS,OUTCOMES,
    PRIVATE,PUBLIC,SEEDS,array_hash,input_packet,labels,load_npz,posterior_path,read_json,
    relative_calibration,ridge,sha256,sources,timing,verify_protocol,write_json)
from objectives import fit_spo,fit_expected_spo
import numpy as np


def parameters(fold):
    m4=read_json(E8/f"private/training/fold{fold}/M4/FIT_LOG.json")
    mse=read_json(E8/f"private/training/fold{fold}/M0_tuned/FIT_LOG.json")
    return float(m4["final_diagnostics"]["regularization"]),100*float(mse["selected_parameters"]["lambda_multiplier"])/3


def prepare():
    cpu,wall=time.process_time(),time.perf_counter()
    PUBLIC.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    if (PUBLIC/"E17_PROTOCOL.json").exists():raise FileExistsError("Protocol is already frozen")
    for path in sources():ast.parse(path.read_text(encoding="utf-8"))
    stream=io.StringIO()
    tested=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(CODE/"tests")))
    (PRIVATE/"QUALIFICATION.log").write_text(stream.getvalue(),encoding="utf-8")
    if not tested.wasSuccessful():raise RuntimeError(stream.getvalue())
    files=[]
    def add(path,expected=None):
        digest=sha256(path)
        if expected is not None and digest!=expected:raise ValueError("Original source hash mismatch")
        files.append(dict(path=str(path),sha256=digest,bytes=path.stat().st_size))
    source=read_json(E13/"public/E13_PROTOCOL.json")
    add(INPUTS,source["source_assets"][0]["sha256"]);add(OUTCOMES,source["source_assets"][1]["sha256"])
    add(E13/"public/E13_PROTOCOL.json")
    add(ASSETS/"research/e14_mechanism_diagnostics/public/code/e14_common.py")
    add(NS/"public/NS_SPO_PROTOCOL.json");add(AUDIT/"public/M4_MECHANISM_PROTOCOL.json")
    add(AUDIT/"public/code/run_distribution_audit.py","d3334819abaee2af4b3501edd4d691c6c89ae03644e4c6ff1b4973eef5382b81")
    import clarabel
    dependency=__import__("pathlib").Path(clarabel.__file__).parent
    for path in sorted(dependency.glob("*")):
        if path.is_file() and path.suffix in (".py",".pyd"):add(path)
    rng=np.random.default_rng(20261115017)
    permutations=[np.arange(20)]
    while len(permutations)<101:
        p=rng.permutation(20)
        if not any(np.array_equal(p,q) for q in permutations):permutations.append(p)
    np.savez_compressed(PRIVATE/"PERMUTATIONS_FROZEN.npz",permutations=np.array(permutations))
    sample_checks=[]
    for fold in range(5):
        path=E13/f"private/inputs/fold{fold}/INPUTS_ONLY.npz"
        add(path,source["input_packets"][fold]["sha256"])
        p=input_packet(fold)
        for method in ("M4","M0_tuned"):
            for name in ("FIT_LOG.json","model_state.npz"):add(E8/f"private/training/fold{fold}/{method}/{name}")
        for rule in ("lexicographic","reverselex","uniform"):add(AUDIT/f"private/distribution/models/fold{fold}_full_spo_{rule}.npz")
        add(posterior_path(fold))
        post=load_npz(posterior_path(fold))
        if not np.array_equal(post["history_indices"],p["H"]):raise ValueError("Posterior history identity differs")
        if post["mu_samples"].shape!=(480,8,20) or post["mu_samples_alt"].shape!=(480,8,20):raise ValueError("Actual MC draws unavailable")
        if np.any(post["fallback_mask"]):raise ValueError("Historical fallback requires explicit stratification")
        sample_checks.append(dict(fold=fold,actual_samples=8,available_joint_pool=16,
            unique_max_samples=int(np.sum((post["mu_samples"]==post["mu_samples"].max(axis=2,keepdims=True)).sum(axis=2)==1)),
            mean8_vs_mean512_mae=float(np.mean(np.abs(post["mu_samples"].mean(axis=1)-post["posterior_mean"])))))
        add(NS/f"private/runtime/training/private/fold{fold}/NS_delta0/MODEL.npz")
        add(NS/f"private/runtime/training/private/fold{fold}/NS_main/MODEL.npz")
    write_json(PRIVATE/"INPUT_MANIFEST.json",dict(files=files))
    protocol=dict(schema="E17_supervision_distortion_controlled_audit_v1",
        branch="codex/e17-supervision-distortion-audit",parent_commit="1f012ae7b9af54adc6a68fcf4417fab81a269982",
        evidence="REUSED_TRAIN600_MECHANISM_DEVELOPMENT",questions=600,outerfolds=5,history=480,evaluation=120,
        members=6,K=3,actions=20,generation_seeds=SEEDS,features="same original MiniLM PCA16/scaler; penalized17x20 head",
        permutation=dict(random_permutations=100,identity_control=1,seed=20261115017,
            encoding="column j in renamed coordinates represents physical team permutation[j]",
            methods=METHODS,training_targets="Ridge empirical means; SPO numeric first true champion; TSPO equal exact-maximizer weights",
            inference_modes=["semantic_exact","numeric_exact","semantic_near","numeric_near"],
            near_tolerance=1e-8,primary_inference="semantic_exact; numerical-near1e-8 sensitivity always reported",
            semantic_training_control="canonical SPO identity solution mapped exactly; mathematical same-objective control, not refitted100 times",
            regularization="frozen original method/fold values; no tuning",solver="Clarabel0.11.1 exact epigraph tol1e-9/max1000/thread1"),
        loss_order=dict(primary_MC=8,MC_counts=MC_COUNTS,pool="actual cached8 draws followed by pre-existing disjoint alt8",
            primary_targets="E[plain SPO+] delta0 versus SPO+(mean of precisely same draws)",
            B1_identity="both operators identical, one model reused",B8_expected="verified original NS_delta0 cache",
            original_NS_main="frozen delta.05 set expectation, background only",
            references=["frozen Ridge","empirical SPO","empirical uniform TSPO","frozen NS.05"],
            regression_metrics="same training-only nonnegative common scale on centered scores; rankings and deployment scores untouched",
            true_mean_and_regret="known-truth synthetic only; real-data empirical three-repeat loss marked separately"),
        confirmation=dict(fresh400_role="input-only mechanism generalization and frozen call-union/power plan until execution decision",
            sealed200="final sealed, no reward access",fresh_accuracy="pending prerequisite/cache/budget assessment",
            effect_pp_primary=1.0,effect_pp_sensitivity=.5,power_target=.8,alpha_one_sided=.05,
            no_automatic_question_expansion=True),
        statistics="paired original-question point differences, seed/task breakdowns and permutation distributions; no CIs or bootstrap",
        evaluation_source_policy="training sources frozen now; evaluator and confirmation source hashes must be separately frozen before opening outer rewards",
        source_hashes={str(p.relative_to(CODE)):sha256(p) for p in sources()},
        permutation_sha256=sha256(PRIVATE/"PERMUTATIONS_FROZEN.npz"),
        original_sample_audit=sample_checks,new_physical_llm_calls=0,new_encoder_forwards=0,confidence_intervals=0,bootstrap_runs=0)
    write_json(PUBLIC/"E17_PROTOCOL.json",protocol)
    write_json(PRIVATE/"PROTOCOL_FREEZE_RECEIPT.json",dict(status="FROZEN",protocol_sha256=sha256(PUBLIC/"E17_PROTOCOL.json")))
    write_json(PUBLIC/"QUALIFICATION.json",dict(status="PASS",tests=tested.testsRun,solver_version=clarabel.__version__,all_source_syntax_valid=True,**timing(cpu,wall)))
    print("E17_PROTOCOL_FROZEN",flush=True)


def permutation_case(fold,index,permutation,p,reward,regularization,ridge_alpha):
    root=PRIVATE/f"permutations/fold{fold}/p{index:03d}"
    root.mkdir(parents=True,exist_ok=True)
    output=root/"MODEL.npz";receipt_path=root/"FROZEN.json"
    if receipt_path.exists():
        receipt=read_json(receipt_path)
        if receipt["protocol_sha256"]!=sha256(PUBLIC/"E17_PROTOCOL.json") or sha256(output)!=receipt["sha256"]:raise ValueError("Cached permutation differs")
        return receipt
    all_w=[];all_scores=[];diagnostics=[];calibrations=[]
    for method in METHODS:
        cpu,wall=time.process_time(),time.perf_counter()
        if method=="Ridge":w,diag=ridge(p["phi_h"],reward[:,permutation],ridge_alpha)
        else:w,diag=fit_spo(p["phi_h"],reward[:,permutation],regularization,
                           rule="lexicographic" if method=="SPO_coordinate" else "uniform")
        mapped=np.empty_like(w);mapped[:,permutation]=w
        scores=p["phi_e"]@mapped
        if index==0:
            reference=(E8/f"private/training/fold{fold}/M0_tuned/model_state.npz" if method=="Ridge" else
                       AUDIT/f"private/distribution/models/fold{fold}_full_spo_{'lexicographic' if method=='SPO_coordinate' else 'uniform'}.npz")
            old=load_npz(reference)["coefficient"]
            error=float(np.max(np.abs(p["phi_e"]@(mapped-old))))
            if error>1e-7:raise ValueError("Original canonical head recovery failed")
            diag["canonical_recovery_score_max_error"]=error
        all_w.append(mapped);all_scores.append(scores)
        diagnostics.append(dict(method=method,**diag,**timing(cpu,wall)))
        calibrations.append(relative_calibration(p["phi_h"],mapped,reward))
    np.savez_compressed(output,weights=np.stack(all_w),scores=np.stack(all_scores),
                        permutation=permutation,history=p["H"],evaluation=p["E"])
    receipt=dict(status="FROZEN",fold=fold,permutation_index=index,sha256=sha256(output),
        protocol_sha256=sha256(PUBLIC/"E17_PROTOCOL.json"),diagnostics=diagnostics,calibrations=calibrations)
    write_json(receipt_path,receipt);return receipt


def run_permutation_fold(fold,stop_after=None):
    p=input_packet(fold);reward=labels(p["H"]);regularization,ridge_alpha=parameters(fold)
    permutations=load_npz(PRIVATE/"PERMUTATIONS_FROZEN.npz")["permutations"]
    receipts=[]
    for index,perm in enumerate(permutations):
        receipts.append(permutation_case(fold,index,perm,p,reward,regularization,ridge_alpha))
        if index%10==0:print(f"E17_PERM fold={fold} p={index}/100",flush=True)
        if stop_after is not None and index>=stop_after:break
    return dict(fold=fold,models=len(receipts),receipts=receipts)


def run_order_fold(fold):
    p=input_packet(fold);reward=labels(p["H"]);regularization,_=parameters(fold)
    post=load_npz(posterior_path(fold));pool=np.concatenate((post["mu_samples"],post["mu_samples_alt"]),axis=1)
    receipts=[]
    for count in MC_COUNTS:
        samples=pool[:,:count]
        for operator in ("LossThenMean","MeanThenLoss"):
            path=PRIVATE/f"loss_order/fold{fold}/B{count}/{operator}.npz";path.parent.mkdir(parents=True,exist_ok=True)
            receipt_path=path.with_suffix(".json")
            if receipt_path.exists():
                old=read_json(receipt_path)
                if sha256(path)!=old["sha256"]:raise ValueError("Order model changed")
                receipts.append(old);continue
            cpu,wall=time.process_time(),time.perf_counter()
            reuse=None
            if count==8 and operator=="LossThenMean":
                source=NS/f"private/runtime/training/private/fold{fold}/NS_delta0/MODEL.npz"
                w=load_npz(source)["coefficient"];diag=dict(reused_source_sha256=sha256(source));reuse="frozen NS_delta0"
            elif count==1 and operator=="MeanThenLoss":
                source=path.with_name("LossThenMean.npz")
                w=load_npz(source)["weights"];diag=dict(reused_source_sha256=sha256(source));reuse="B1 exact operator identity"
            elif operator=="LossThenMean":w,diag=fit_expected_spo(p["phi_h"],samples,regularization)
            else:w,diag=fit_spo(p["phi_h"],samples.mean(axis=1),regularization)
            np.savez_compressed(path,weights=w,scores=p["phi_e"]@w,history=p["H"],evaluation=p["E"],
                                sample_hash=np.array(array_hash(samples)))
            receipt=dict(status="FROZEN",fold=fold,MC_samples=count,operator=operator,
                protocol_sha256=sha256(PUBLIC/"E17_PROTOCOL.json"),sample_sha256=array_hash(samples),
                sha256=sha256(path),reused=reuse is not None,reuse_role=reuse,diagnostics=diag,
                calibration=relative_calibration(p["phi_h"],w,reward),
                mean_MC_vs_empirical_MAE=float(np.abs(samples.mean(axis=1)-reward).mean()),
                mean_MC_vs_512_summary_MAE=float(np.abs(samples.mean(axis=1)-post["posterior_mean"]).mean()),
                true_posterior_mean_error=None,true_population_regret=None,**timing(cpu,wall))
            write_json(receipt_path,receipt);receipts.append(receipt)
        print(f"E17_ORDER fold={fold} B={count}",flush=True)
    return dict(fold=fold,models=len(receipts),receipts=receipts)


def freeze(workers):
    verify_protocol()
    if (PRIVATE/"ALL_PREDICTIONS_FROZEN.json").exists():raise FileExistsError("Already frozen")
    cpu,wall=time.process_time(),time.perf_counter();permutations=[];orders=[]
    # A real identity/random-permutation pilot is retained in the same immutable cache.
    pilot=run_permutation_fold(0,stop_after=1)
    if not (PRIVATE/"RESOURCE_PILOT.json").exists():write_json(PRIVATE/"RESOURCE_PILOT.json",dict(status="PASS",models=pilot["models"],outer_rewards_read=0))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(run_permutation_fold,f):f for f in range(5)}
        for future in as_completed(futures):permutations.append(future.result())
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(run_order_fold,f):f for f in range(5)}
        for future in as_completed(futures):orders.append(future.result())
    files=[]
    for path in sorted((PRIVATE/"permutations").rglob("MODEL.npz")):files.append(dict(path=str(path),sha256=sha256(path)))
    for path in sorted((PRIVATE/"loss_order").rglob("*.npz")):files.append(dict(path=str(path),sha256=sha256(path)))
    write_json(PRIVATE/"ALL_PREDICTIONS_FROZEN.json",dict(status="FROZEN_BEFORE_OUTER_REWARD_EVALUATION",
        permutation_folds=permutations,order_folds=orders,files=files,
        protocol_sha256=sha256(PUBLIC/"E17_PROTOCOL.json"),new_llm_calls=0,**timing(cpu,wall)))
    print("E17_ALL_PREDICTIONS_FROZEN",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("prepare","freeze"));parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args();prepare() if args.command=="prepare" else freeze(args.workers)
