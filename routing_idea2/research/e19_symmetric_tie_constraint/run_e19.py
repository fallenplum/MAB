"""E19 P0/P1 runner with selected-row rewards and prediction-before-E freeze.

No fitting occurs at import. The first P0 run is the four planned formal fits.
P1 reuses fold0 and fits only the eight remaining declared heads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np


CODE = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
DEFAULT_REWARDS = ASSETS / "artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_outcomes.npz"
DEFAULT_INPUTS = ASSETS / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/selector_train_inputs.npz"
REWARD_SHA = "a01230f2bed3e7f9d05713bf9e3e4f1cf44987b41c460f91e1d83bc1c29f191e"
INPUT_SHA = "57e39747ec2756bcd79ab00885b3c08ad4f91fdb3ec900cbed15146420cd17c3"
WEIGHTED_SHA = "019c505c4a5ab2f90e532eda2842546c444c6480a182504f56c935f3527dd4da"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def reward_header(path, member="aggregate_scores.npy"):
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1,0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2,0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("Unsupported numeric NPY header version")
            if fortran or len(shape) != 3 or dtype.hasobject or dtype.kind not in "fibu":
                raise ValueError("Selected-row reader requires a C-order numeric generation/question/action member")
            offset = stream.tell()
            expected = offset + int(np.prod(shape))*dtype.itemsize
            if archive.getinfo(member).file_size != expected:
                raise ValueError("Reward member length differs from its numeric header")
            return dict(member=member, version=list(version), shape=list(shape),
                        dtype=dtype.str, fortran_order=fortran, data_offset=offset,
                        member_bytes=expected, decoded_reward_values=0)


def read_reward_rows(path, indices, *, member="aggregate_scores.npy"):
    """Decode requested C-order question rows only; consume other rows as bytes."""
    header = reward_header(path, member)
    indices = np.asarray(indices)
    generations, questions, actions = header["shape"]
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(indices) == 0 or len(set(indices.tolist())) != len(indices)
            or np.any(indices < 0) or np.any(indices >= questions)):
        raise ValueError("Requested reward question rows must be unique in-range integer indices")
    dtype = np.dtype(header["dtype"])
    result = np.empty((generations, len(indices), actions), dtype=dtype)
    lookup = {int(row): local for local,row in enumerate(indices)}
    row_bytes = actions*dtype.itemsize
    raw_skipped = decoded_bytes = 0
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as stream:
            if len(stream.read(header["data_offset"])) != header["data_offset"]:
                raise ValueError("Reward member header was truncated")
            for generation in range(generations):
                for row in range(questions):
                    raw = stream.read(row_bytes)
                    if len(raw) != row_bytes:
                        raise ValueError("Numeric reward row was truncated")
                    if row in lookup:
                        result[generation, lookup[row]] = np.frombuffer(raw, dtype=dtype, count=actions)
                        decoded_bytes += len(raw)
                    else:
                        raw_skipped += len(raw)
            if stream.read(1):
                raise ValueError("Unexpected bytes after numeric reward rows")
    return result, dict(header=header, requested_question_rows=len(indices),
                        requested_indices_sha256=array_hash(indices.astype(np.int64)),
                        decoded_generation_rows=generations*len(indices),
                        decoded_reward_values=result.size, decoded_data_bytes=decoded_bytes,
                        skipped_raw_data_bytes=raw_skipped,
                        full_question_array_decoded=len(indices) == questions)


def uniform_policy(scores, tolerance):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Policy scores must be a finite question/action matrix")
    active = scores >= scores.max(axis=1, keepdims=True)-tolerance
    probabilities = active/active.sum(axis=1, keepdims=True)
    sorted_scores = np.sort(scores, axis=1)
    margin = sorted_scores[:, -1]-sorted_scores[:, -2]
    return dict(scores=scores, centered_scores=scores-scores.mean(axis=1, keepdims=True),
                probabilities=probabilities, argmax=scores.argmax(axis=1), margin=margin,
                active_count=active.sum(axis=1), strict=margin > tolerance)


def expected_coupled_changes(p_b, p_c, rewards):
    """Permutation-equivariant maximal same-action overlap correction coupling."""
    overlap = np.minimum(p_b, p_c)
    rho = np.maximum(0.0, 1.0-overlap.sum(axis=1))
    e_b, e_c = p_b-overlap, p_c-overlap
    b_wrong = np.einsum("qa,gqa->gq", e_b, 1-rewards)
    b_right = np.einsum("qa,gqa->gq", e_b, rewards)
    c_wrong = np.einsum("qa,gqa->gq", e_c, 1-rewards)
    c_right = np.einsum("qa,gqa->gq", e_c, rewards)
    corrections = np.divide(b_wrong*c_right, rho[None,:], out=np.zeros_like(b_wrong), where=rho[None,:] > 1e-12)
    damages = np.divide(b_right*c_wrong, rho[None,:], out=np.zeros_like(b_right), where=rho[None,:] > 1e-12)
    utility_b = np.einsum("qa,gqa->gq", p_b, rewards)
    utility_c = np.einsum("qa,gqa->gq", p_c, rewards)
    error = float(np.max(np.abs(corrections-damages-(utility_c-utility_b))))
    if error > 1e-12:
        raise AssertionError("Expected coupled net corrections differ from marginal utility")
    return corrections, damages, utility_b, utility_c, error


def task_macro(values, labels):
    return float(np.mean([np.asarray(values)[:, labels == task].mean() for task in ("gsm8k","bbh")]))


def point_metrics(utilities, corrections, damages, rewards, b, c, labels, subtasks, tolerance):
    strict_both = (b["margin"] > tolerance) & (c["margin"] > tolerance)
    strict_change = strict_both & (b["argmax"] != c["argmax"])
    rows = np.arange(len(labels))
    reward_b = rewards[:, rows, b["argmax"]]
    reward_c = rewards[:, rows, c["argmax"]]
    strict_corrections = int(((reward_b == 0)&(reward_c == 1)&strict_both[None,:]).sum())
    strict_damages = int(((reward_b == 1)&(reward_c == 0)&strict_both[None,:]).sum())
    result = dict(questions=len(labels), B_task_macro_accuracy=task_macro(utilities[0], labels),
                  C_task_macro_accuracy=task_macro(utilities[1], labels),
                  C_minus_B_pp=100*task_macro(utilities[1]-utilities[0],labels),
                  expected_corrections=float(corrections.sum()), expected_damages=float(damages.sum()),
                  expected_net_corrections=float((corrections-damages).sum()),
                  expected_counts_are_fractional_coupling_expectations=True,
                  strict_both_unique_questions=int(strict_both.sum()), strict_changed_questions=int(strict_change.sum()),
                  strict_integer_corrections=strict_corrections, strict_integer_damages=strict_damages,
                  strict_integer_net_corrections=strict_corrections-strict_damages,
                  by_task={}, by_generation_seed={}, bbh_subtasks={},
                  B_numerical_tied_questions=int((b["active_count"] > 1).sum()),
                  C_numerical_tied_questions=int((c["active_count"] > 1).sum()))
    for task in ("gsm8k","bbh"):
        mask = labels == task
        result["by_task"][task] = dict(questions=int(mask.sum()), B_accuracy=float(utilities[0][:,mask].mean()),
            C_accuracy=float(utilities[1][:,mask].mean()), gain_pp=100*float((utilities[1]-utilities[0])[:,mask].mean()))
    for index, seed in enumerate((42,137,2024)):
        result["by_generation_seed"][str(seed)] = dict(B_accuracy=task_macro(utilities[0][index:index+1],labels),
            C_accuracy=task_macro(utilities[1][index:index+1],labels),
            gain_pp=100*task_macro((utilities[1]-utilities[0])[index:index+1],labels))
    for subtask in sorted(set(subtasks[labels == "bbh"].tolist())):
        mask = (labels == "bbh")&(subtasks == subtask)
        result["bbh_subtasks"][str(subtask)] = dict(questions=int(mask.sum()), B_accuracy=float(utilities[0][:,mask].mean()),
            C_accuracy=float(utilities[1][:,mask].mean()), gain_pp=100*float((utilities[1]-utilities[0])[:,mask].mean()))
    result["B_bbh_subtask_macro_accuracy"] = float(np.mean([v["B_accuracy"] for v in result["bbh_subtasks"].values()]))
    result["C_bbh_subtask_macro_accuracy"] = float(np.mean([v["C_accuracy"] for v in result["bbh_subtasks"].values()]))
    return result, strict_both, strict_change


def load_packet(fold, protocol, audit):
    row = next(item for item in audit["folds"] if item["fold"] == fold)
    path = Path(row["input_packet_path"])
    if sha(path) != row["input_packet_sha256"]:
        raise ValueError("Frozen fold-local input packet bytes differ")
    with np.load(path, allow_pickle=False) as archive:
        packet = {name: archive[name].copy() for name in protocol["input_packet_fields_allowed"]}
    h, e = packet["H"].astype(np.int64), packet["E"].astype(np.int64)
    if (len(h) != protocol["history_questions_per_fold"] or len(e) != protocol["evaluation_questions_per_fold"]
            or set(h)&set(e) or len(set(h)|set(e)) != protocol["independent_development_questions"]
            or packet["phi_h"].shape != (len(h), protocol["head_shape"][0])
            or packet["phi_e"].shape != (len(e), protocol["head_shape"][0])
            or not np.all(packet["phi_h"][:,0] == 1) or not np.all(packet["phi_e"][:,0] == 1)):
        raise ValueError("Frozen fold partition or constant-feature geometry differs")
    if protocol["regularization_by_fold"][fold] != row["frozen_lambda"]:
        raise ValueError("Protocol regularization differs from its original fitting source")
    return packet, row


def run_fold(fold, protocol, protocol_sha, study, audit, inputs, qualify_reverse):
    from weighted_objectives import fit_weighted_tspo, objective_components, STRICT_SOLVER_SETTINGS
    for key in ("tol_gap_abs","tol_gap_rel","tol_feas","max_iter","max_threads"):
        if STRICT_SOLVER_SETTINGS[key] != protocol["solver"][key]:
            raise ValueError("Actual weighted solver settings differ from protocol")
    private = study/"private"/f"fold{fold}"
    private.mkdir(parents=True, exist_ok=False)
    packet, source = load_packet(fold, protocol, audit)
    h, e = packet["H"].astype(np.int64), packet["E"].astype(np.int64)
    observed, access = read_reward_rows(DEFAULT_REWARDS, h)
    if observed.shape != (3,480,20) or not np.isin(observed,(0,1)).all():
        raise ValueError("History reward units differ from original three binary repeats")
    write_json(private/"H_REWARD_ACCESS.json", dict(stage="H_ONLY_BEFORE_FITS", **access))
    targets = observed.mean(axis=0)
    regularization = protocol["regularization_by_fold"][fold]
    tolerance = protocol["deployment"]["absolute_score_tie_tolerance"]
    branch_states, diagnostics, outputs = {}, {}, {}
    orders = [("canonical",None)]
    if qualify_reverse:
        orders.append(("reverse", np.arange(protocol["actions"]-1,-1,-1)))
    for order, permutation in orders:
        for branch in ("B","C"):
            name = branch if order == "canonical" else branch+"_reverse"
            coefficient, diag = fit_weighted_tspo(packet["phi_h"], targets, regularization,
                protocol["branches"][branch]["beta"], permutation=permutation,
                gamma=protocol["gamma"], strict_tolerances=True)
            score_h, score_e = packet["phi_h"]@coefficient, packet["phi_e"]@coefficient
            state = uniform_policy(score_e, tolerance)
            exact = uniform_policy(score_e, 0.0)
            path = private/(name+"_PREDICTIONS.npz")
            write_npz(path, W=coefficient, scores_h=score_h, centered_scores_h=score_h-score_h.mean(axis=1,keepdims=True),
                scores_e=score_e, centered_scores_e=state["centered_scores"], probabilities=state["probabilities"],
                raw_exact_probabilities=exact["probabilities"], argmax=state["argmax"], full_margin=state["margin"],
                numerical_argmax_count=state["active_count"], raw_exact_argmax_count=exact["active_count"],
                H=h, E=e, **{key:packet[key] for key in ("pca_mean","pca_components","scaler_mean","scaler_scale")})
            write_json(private/(name+"_FIT_LOG.json"), dict(branch=branch, order=order,
                regularization=regularization, gamma=protocol["gamma"], diagnostics=diag))
            branch_states[name] = state
            outputs[name] = dict(path=str(path),sha256=sha(path))
            diagnostics[name] = diag
            print(json.dumps(dict(event="FORMAL_FIT_COMPLETE",fold=fold,branch=name,status=diag["status"],
                                  objective=diag["objective"],iterations=diag["iterations"])),flush=True)
    b, c = diagnostics["B"], diagnostics["C"]
    difference = b["F0"]-c["F0"]
    jt_difference = c["J_T"]-b["J_T"]
    limit = protocol["P0"]["objective_and_kkt_tolerance"]
    checks = dict(endpoint_lower=difference >= -limit, endpoint_upper=difference <= jt_difference+limit,
                  endpoint_JT_direction=jt_difference >= -limit)
    numerical_fields = ("direct_epigraph_error","primal_violation","dual_violation","stationarity_inf",
                        "complementarity_inf","primal_dual_objective_gap")
    checks["all_KKT_and_gap"] = all(diag[field] <= limit for diag in diagnostics.values() for field in numerical_fields)
    mappings = {}
    if qualify_reverse:
        for branch in ("B","C"):
            with np.load(outputs[branch]["path"],allow_pickle=False) as original, np.load(outputs[branch+"_reverse"]["path"],allow_pickle=False) as reverse:
                error_h=float(np.max(np.abs(original["centered_scores_h"]-reverse["centered_scores_h"])))
                error_e=float(np.max(np.abs(original["centered_scores_e"]-reverse["centered_scores_e"])))
                prob_error=float(np.max(np.abs(original["probabilities"]-reverse["probabilities"])))
            mappings[branch]=dict(centered_score_h_max_error=error_h,centered_score_e_max_error=error_e,
                                  uniform_probability_max_error=prob_error)
            checks[branch+"_mapped_centered"] = max(error_h,error_e) <= protocol["P0"]["centered_score_mapping_tolerance"]
            checks[branch+"_mapped_uniform_policy"] = prob_error <= protocol["P0"]["centered_score_mapping_tolerance"]
    qualification=dict(status="PASS" if all(checks.values()) else "FAIL",fold=fold,checks=checks,
        F0_B_minus_F0_C=difference, JT_C_minus_JT_B=jt_difference, mappings=mappings,
        real_fits=len(diagnostics),denominator=len(h),gamma=protocol["gamma"],
        B_beta=protocol["branches"]["B"]["beta"], C_beta=protocol["branches"]["C"]["beta"],
        inequality_tolerance=limit, diagnostics=diagnostics)
    write_json(private/"NUMERICAL_QUALIFICATION.json",qualification)
    if not all(checks.values()):
        raise AssertionError("Fold qualification failed; evaluation rewards remain undecoded")
    receipt=dict(schema="e19_fold_predictions_frozen_v1",status="FROZEN_BEFORE_E_REWARDS",fold=fold,
        protocol_sha256=protocol_sha,runner_source_sha256=sha(__file__),weighted_source_sha256=sha(CODE/"weighted_objectives.py"),
        input_packet_sha256=source["input_packet_sha256"], source_reward_npz_sha256=REWARD_SHA,
        H_indices_sha256=array_hash(h),E_indices_sha256=array_hash(e),predictions=outputs,
        numerical_qualification_sha256=sha(private/"NUMERICAL_QUALIFICATION.json"),
        fresh_or_sealed_reward_reads=0,evaluation_reward_rows_decoded_before_freeze=0)
    frozen_path=private/"PREDICTIONS_FROZEN.json"
    write_json(frozen_path,receipt)
    # Opening E is deliberately after the immutable prediction/qualification receipt.
    if read_json(frozen_path)["status"] != "FROZEN_BEFORE_E_REWARDS":
        raise ValueError("Prediction freeze receipt is unavailable")
    for item in outputs.values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("A frozen prediction file changed before evaluation access")
    evaluation, e_access=read_reward_rows(DEFAULT_REWARDS,e)
    if not np.isin(evaluation,(0,1)).all():
        raise ValueError("Evaluation rewards must be binary")
    write_json(private/"E_REWARD_ACCESS.json",dict(stage="E_ONLY_AFTER_PREDICTION_FREEZE",prediction_receipt_sha256=sha(frozen_path),**e_access))
    state_b,state_c=branch_states["B"],branch_states["C"]
    corr,damage,utility_b,utility_c,coupling_error=expected_coupled_changes(state_b["probabilities"],state_c["probabilities"],evaluation)
    exact_utilities=[]
    for branch in ("B","C"):
        with np.load(outputs[branch]["path"],allow_pickle=False) as saved:
            exact_utilities.append(np.einsum("qa,gqa->gq",saved["raw_exact_probabilities"],evaluation))
    metrics,strict_both,strict_changed=point_metrics(np.stack((utility_b,utility_c)),corr,damage,evaluation,
        state_b,state_c,inputs["task_labels"][e],inputs["subtask_labels"][e],tolerance)
    metrics.update(fold=fold,coupling_net_identity_max_error=coupling_error,
                   raw_exact_B_accuracy=task_macro(exact_utilities[0],inputs["task_labels"][e]),
                   raw_exact_C_accuracy=task_macro(exact_utilities[1],inputs["task_labels"][e]),
                   frozen_prediction_receipt_sha256=sha(frozen_path), bootstrap_completed=False)
    write_npz(private/"EVALUATION.npz",E=e,rewards=evaluation,utilities=np.stack((utility_b,utility_c)),
        raw_exact_utilities=np.stack(exact_utilities),expected_corrections=corr,expected_damages=damage,
        strict_both_unique=strict_both,strict_changed=strict_changed,
        probabilities=np.stack((state_b["probabilities"],state_c["probabilities"])),
        argmax=np.stack((state_b["argmax"],state_c["argmax"])),
        margins=np.stack((state_b["margin"],state_c["margin"])),
        task_labels=inputs["task_labels"][e],subtask_labels=inputs["subtask_labels"][e])
    write_json(study/"public"/f"FOLD{fold}_POINT_RESULTS.json",metrics)
    return qualification,metrics


def combine(study,inputs,protocol_sha,protocol):
    positions=[]
    all_arrays={}
    for fold in range(5):
        path=study/"private"/f"fold{fold}"/"EVALUATION.npz"
        receipt=read_json(path.parent/"PREDICTIONS_FROZEN.json")
        if receipt["protocol_sha256"] != protocol_sha:
            raise ValueError("Cross-fold protocol identity differs")
        with np.load(path,allow_pickle=False) as packet:
            positions.append(packet["E"])
            for key in packet.files:
                if key != "E":all_arrays.setdefault(key,[]).append(packet[key].copy())
    indices=np.concatenate(positions)
    if len(indices)!=600 or not np.array_equal(np.sort(indices),np.arange(600)):
        raise ValueError("Five evaluation folds are not an exact600-question partition")
    order=np.argsort(indices)
    axes=dict(rewards=1,utilities=2,raw_exact_utilities=2,expected_corrections=1,expected_damages=1,
              strict_both_unique=0,strict_changed=0,probabilities=1,argmax=1,margins=1,task_labels=0,subtask_labels=0)
    merged={key:np.take(np.concatenate(values,axis=axes[key]),order,axis=axes[key]) for key,values in all_arrays.items()}
    merged["E"]=np.arange(600)
    write_npz(study/"private/OOF_EVALUATION.npz",**merged)
    b=dict(argmax=merged["argmax"][0],margin=merged["margins"][0],active_count=(merged["probabilities"][0]>0).sum(axis=1))
    c=dict(argmax=merged["argmax"][1],margin=merged["margins"][1],active_count=(merged["probabilities"][1]>0).sum(axis=1))
    metrics,_,_=point_metrics(merged["utilities"],merged["expected_corrections"],merged["expected_damages"],
        merged["rewards"],b,c,merged["task_labels"],merged["subtask_labels"],
        protocol["deployment"]["absolute_score_tie_tolerance"])
    metrics.update(status="P1_ALL_FIVE_FOLDS_POINTS_COMPLETE",main_fits=10,qualification_fits=2,
                   protocol_sha256=protocol_sha, bootstrap_completed=False,
                   analysis_scope="Repeatedly used train600 OOF development; generation repeats stay clustered by original question")
    write_json(study/"public/P1_POINT_RESULTS.json",metrics)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("inspect","p0","p1"),required=True)
    parser.add_argument("--protocol",type=Path,default=CODE/"E19_PROTOCOL.json")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--study-root",type=Path,default=ASSETS/"research/e19_symmetric_tie_constraint")
    args=parser.parse_args()
    if args.phase=="inspect":
        print(json.dumps(reward_header(DEFAULT_REWARDS)))
        return
    protocol_sha=sha(args.protocol)
    if args.protocol_sha256 and protocol_sha!=args.protocol_sha256:
        raise ValueError("Protocol SHA differs")
    protocol=read_json(args.protocol)
    if protocol["schema"]!="e19_symmetric_tie_constraint_protocol_v1" or protocol["gamma"]!=0:
        raise ValueError("Unexpected E19 protocol")
    if sha(CODE/"weighted_objectives.py")!=WEIGHTED_SHA or sha(DEFAULT_REWARDS)!=REWARD_SHA or sha(DEFAULT_INPUTS)!=INPUT_SHA:
        raise ValueError("Frozen weighted operator or historical numeric source differs")
    for name,expected in protocol.get("source_hashes",{}).items():
        if sha(CODE/name)!=expected:raise ValueError("Protocol source changed: "+name)
    if reward_header(DEFAULT_REWARDS)["shape"]!=[3,600,20] or reward_header(DEFAULT_REWARDS)["dtype"]!="<f8":
        raise ValueError("The exact frozen reward header differs")
    audit=read_json(CODE/"FROZEN_TSPO_REUSE_AUDIT.json")
    with np.load(DEFAULT_INPUTS,allow_pickle=False) as packet:
        inputs={key:packet[key].copy() for key in ("question_ids","canonical_hashes","task_labels","subtask_labels")}
    if args.phase=="p0":
        qualification,metrics=run_fold(0,protocol,protocol_sha,args.study_root,audit,inputs,True)
        write_json(args.study_root/"public/P0_QUALIFICATION.json",dict(qualification,status="PASS_P0_GATE",protocol_sha256=protocol_sha,
                   P0_formal_fits=4,P1_allowed=True,scientific_utility_gate_used=False))
        print(json.dumps(dict(status="PASS_P0_GATE",point_gain_pp=metrics["C_minus_B_pp"])),flush=True)
    else:
        gate=read_json(args.study_root/"public/P0_QUALIFICATION.json")
        if gate["status"]!="PASS_P0_GATE" or gate["protocol_sha256"]!=protocol_sha:
            raise ValueError("P1 requires the same protocol's successful planned P0")
        for fold in protocol["P1"]["remaining_folds"]:
            run_fold(fold,protocol,protocol_sha,args.study_root,audit,inputs,False)
        metrics=combine(args.study_root,inputs,protocol_sha,protocol)
        print(json.dumps(dict(status=metrics["status"],point_gain_pp=metrics["C_minus_B_pp"])),flush=True)


if __name__=="__main__":
    main()
