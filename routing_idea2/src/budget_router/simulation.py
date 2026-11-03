"""Prospective restricted-feedback simulation; synthetic tests are not research.

The environment owns current scores and actual token lengths. Policies receive
only phi, arrived labels, and historical/cap cost predictions. Production assets
must pass provenance/E1 gates; E3 acquisition also requires E2_PASS.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import hashlib
import heapq
import json
import time
import numpy as np
from .features import SubsetFeatureMap
from .learner import OriginalLabel, SharedRidgeRegressor, cold_start_limit
from .budget import ResourceCost, CostPrediction, BudgetLedger, BudgetExceeded
from .acquisition import AcquisitionPlan, choose_main, plan_acquisition

STRATEGIES=("fixed_combination","fixed_with_paid_monitoring","contextual_topk","none","random","periodic","max_uncertainty","max_uncertainty_per_cost","no_cost","decision","linucb","thompson")
DEFAULT_SEEDS=(42,137,2024,2025,2026)
REQUIRED_ARRAYS=("question_ids","task_labels","phi","aggregate_scores","member_scores",
 "member_prompt_tokens","member_completion_tokens","aggregate_prompt_tokens","aggregate_completion_tokens")

def _hash_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def _digest(x):
    return isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdef" for c in x)

def _reference(base,record,name):
    if not isinstance(record,dict) or not record.get("path") or not _digest(record.get("sha256")):
        raise ValueError(f"Missing provenance reference: {name}")
    path=Path(record["path"])
    if not path.is_absolute(): path=base/path
    if not path.is_file() or _hash_file(path)!=record["sha256"]:
        raise ValueError(f"Missing or changed provenance artifact: {name}")
    return path

def _validate_stage_bindings(manifest,e0,e1):
    evidence=e1.get("evidence",{})
    expected={"e0_lock_sha256":manifest["e0_lock"]["sha256"],
              "split_manifest_sha256":manifest["split_manifest"]["sha256"],
              "pool_config_sha256":manifest["pool_config"]["sha256"],
              "aggregator_fingerprint":manifest["aggregator"]["fingerprint"],
              "scorer_version":manifest["scorer"]["version"]}
    if any(evidence.get(k)!=v for k,v in expected.items()):
        raise ValueError("E1 PASS does not bind the same E0/split/pool/full aggregator configuration/scorer")
    for key in ("pool_config_sha256","split_manifest_sha256","aggregator_fingerprint"):
        if e0.get(key)!=expected[key]:
            raise ValueError("E0 lock differs from the full online execution configuration")
    scorer_hashes={v["sha256"] for v in manifest["sources"] if v["kind"]=="scorer_source"}
    if evidence.get("scorer_source_sha256") not in scorer_hashes:
        raise ValueError("Online scorer implementation differs from E1 PASS")

def _validate_locked_split(locked, question_ids, fingerprints, split, pca):
    records=locked.get("records")
    canonical=locked.get("canonical_question_fingerprints")
    if not isinstance(records,list) or not isinstance(canonical,dict):
        raise ValueError("Locked split requires records and canonical_question_fingerprints")
    identifiers=[r["question_id"] for r in records]
    if len(set(identifiers))!=len(identifiers) or set(canonical)!=set(identifiers):
        raise ValueError("Locked split has duplicate/missing question identities")
    values=[canonical[q] for q in identifiers]
    if not all(_digest(v) for v in values) or len(set(values))!=len(values):
        raise ValueError("Canonical questions leak across locked splits")
    expected={r["question_id"] for r in records if r["split"]==split}
    if set(question_ids)!=expected or len(question_ids)!=len(expected):
        raise ValueError("Matrix must match the entire locked split, without cross-stage substitution")
    if list(fingerprints)!=[canonical[q] for q in question_ids]:
        raise ValueError("Matrix fingerprints do not match the locked question identities")
    fit=pca.get("fit_question_ids")
    training={r["question_id"] for r in records if r["split"]=="online_calibration"}
    if not isinstance(fit,list) or not fit or len(set(fit))!=len(fit) or not set(fit)<=training:
        raise ValueError("PCA fit question IDs must be explicitly confined to online_calibration")

@dataclass(frozen=True)
class CallCaps:
    member_prompt:int=4096
    member_completion:int=256
    aggregate_prompt:int=4096
    aggregate_completion:int=256
    member_calls:int=1
    aggregate_calls:int=1

    def __post_init__(self):
        if any(not isinstance(v,int) or v<0 for v in asdict(self).values()) or min(self.member_calls,self.aggregate_calls)<1:
            raise ValueError("Nonnegative token caps and positive call caps required")

    def service(self,natural=False):
        return ResourceCost(3*self.member_calls+self.aggregate_calls,int(natural),
            3*self.member_calls*(self.member_prompt+self.member_completion)
            +self.aggregate_calls*(self.aggregate_prompt+self.aggregate_completion))

    def alternate(self,labels):
        return ResourceCost(self.member_calls+self.aggregate_calls,labels,
            self.member_calls*(self.member_prompt+self.member_completion)
            +self.aggregate_calls*(self.aggregate_prompt+self.aggregate_completion))

@dataclass(frozen=True)
class SimulationSettings:
    execution_limit:int
    token_limit:int
    caps:CallCaps=CallCaps()
    cost_weights:tuple=(1.,1.,.001)
    warmup_max_labels:int=32
    ridge:float=1.
    noise_variance:float=1.
    bootstrap_replicates:int=200
    bootstrap_min_labels:int=32
    bootstrap_min_questions:int=4
    kappa:float=2.
    threshold:float=0.
    period:int=5
    main_feedback_natural:bool=False
    feedback_mode:str="verifiable"
    cost_prediction:str="history"
    warmup_mode:str="random_combo"
    stream_order:str="iid"

    def __post_init__(self):
        ResourceCost(self.execution_limit,0,self.token_limit)
        if len(self.cost_weights)!=3 or not np.isfinite(self.cost_weights).all() or min(self.cost_weights)<0 or not any(self.cost_weights):
            raise ValueError("Explicit execution/label/token scalar weights required")
        if self.feedback_mode not in {"verifiable","final_only"} or self.cost_prediction not in {"history","caps"}:
            raise ValueError("Unsupported feedback/cost prediction mode")
        if self.stream_order not in {"iid","gsm8k_then_bbh","bbh_then_gsm8k"}:
            raise ValueError("Unsupported declared task order")
        if self.warmup_mode not in {"random_combo", "paid_member_prior_then_combo"}:
            raise ValueError("Unsupported warmup mode")
        if self.warmup_mode == "paid_member_prior_then_combo" and self.feedback_mode != "verifiable":
            raise ValueError("Member-prior ablation requires legitimately reference-verifiable feedback")
        if self.bootstrap_replicates<2: raise ValueError("Need at least two bootstrap draws")

@dataclass(frozen=True)
class QuestionContext:
    question_id:str
    task:str
    phi:np.ndarray

class MatrixEnvironment:
    """Private matrices: never hand this object or its arrays to a policy."""
    def __init__(self,arrays,*,split,manifest,software_test_only=False):
        missing=set(REQUIRED_ARRAYS)-set(arrays)
        if missing: raise ValueError(f"Missing real matrix assets: {sorted(missing)}")
        self._arrays={k:np.array(v,copy=True) for k,v in arrays.items()}
        self.split=split
        self.software_test_only=bool(software_test_only)
        self._manifest=dict(manifest)
        ids,tasks=self._arrays["question_ids"],self._arrays["task_labels"]
        if ids.ndim!=1 or tasks.shape!=ids.shape or ids.dtype.kind not in "US" or tasks.dtype.kind not in "US":
            raise ValueError("Aligned non-object string question IDs/task labels required")
        if not len(ids) or len(set(ids.tolist()))!=len(ids) or any(not str(x) for x in ids):
            raise ValueError("Unique nonempty question IDs required")
        n=len(ids); phi=self._arrays["phi"]
        if phi.ndim!=2 or phi.shape[0]!=n or not np.isfinite(phi).all() or not np.allclose(phi[:,0],1.):
            raise ValueError("phi must be frozen train-only PCA features with leading constant")
        for key,width in (("aggregate_scores",20),("member_scores",6)):
            value=self._arrays[key]
            if value.shape!=(n,width) or not np.isfinite(value).all() or np.any((value<0)|(value>1)):
                raise ValueError(f"Invalid actual score matrix: {key}")
        for stem,width in (("member",6),("aggregate",20)):
            for kind in ("prompt_tokens","completion_tokens"):
                key=f"{stem}_{kind}"; value=self._arrays[key]
                if value.shape!=(n,width) or value.dtype.kind not in "iu" or np.any(value<0):
                    raise ValueError(f"Invalid actual billed tokens: {key}")
            key=f"{stem}_execution_counts"
            if key not in self._arrays:
                if not software_test_only and manifest.get("single_call_per_cell") is not True:
                    raise ValueError("Actual execution counts required for retry/failure accounting")
                self._arrays[key]=np.ones((n,width),dtype=int)
            value=self._arrays[key]
            if value.shape!=(n,width) or value.dtype.kind not in "iu" or np.any(value<1):
                raise ValueError(f"Invalid actual execution counts: {key}")
            key=stem+"_wall_time_s"
            if key in self._arrays:
                value=self._arrays[key]
                if value.shape!=(n,width) or value.dtype.kind not in "fiu" or not np.isfinite(value).all() or np.any(value<0):
                    raise ValueError(f"Invalid source execution wall times: {key}")
        fingerprints=manifest.get("question_fingerprints")
        if software_test_only:
            fingerprints=[hashlib.sha256(("software:"+str(x)).encode()).hexdigest() for x in ids]
        if not isinstance(fingerprints,list) or len(fingerprints)!=n or not all(_digest(x) for x in fingerprints) or len(set(fingerprints))!=n:
            raise ValueError("Unique canonical question fingerprints required")
        self._fingerprints=tuple(fingerprints)
        self.mapper=SubsetFeatureMap(6,3)

    @classmethod
    def from_npz(cls,matrix_path,manifest_path,*,expected_split):
        """Strict provenance loader; see module and CLI interface documentation."""
        if expected_split not in {"online_calibration","online_stream"}:
            raise ValueError("E1/final-test matrices cannot enter online simulation")
        matrix_path,manifest_path=Path(matrix_path),Path(manifest_path)
        if not matrix_path.is_file() or not manifest_path.is_file():
            raise ValueError("Real matrix/provenance assets are absent")
        manifest_bytes=manifest_path.read_bytes()
        m=json.loads(manifest_bytes.decode("utf-8-sig"))
        actual_matrix_sha256=_hash_file(matrix_path)
        actual_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest()
        if m.get("software_test_only") or m.get("split")!=expected_split or m.get("matrix_sha256")!=actual_matrix_sha256:
            raise ValueError("Matrix split/hash is not qualified for production")
        if m.get("e2_gate") is not None:
            raise ValueError("Legacy inline E2 gate is not an authorization; validate a separate --e2-gate against both original assets")
        flags=("all_combinations_executed","real_frozen_llm_aggregation","aggregate_scores_from_executed_aggregator",
               "e1_isolated","sealed_test_isolated","failure_costs_included")
        if any(m.get(flag) is not True for flag in flags):
            raise ValueError("Real aggregation/isolated provenance required; no member-score surrogate")
        pool=m.get("model_pool",[])
        if len(pool)!=6 or any(not v.get("model_id") or not v.get("revision") for v in pool) or len({v["model_id"] for v in pool})!=6:
            raise ValueError("Exactly six models with pinned revisions required")
        agg=m.get("aggregator",{})
        if not agg.get("model_id") or not agg.get("revision") or not _digest(agg.get("prompt_sha256")) or not _digest(agg.get("fingerprint")) or not m.get("cache_revision"):
            raise ValueError("Aggregator/model/cache provenance incomplete")
        if not m.get("scorer",{}).get("version") or m["scorer"].get("no_model_calls") is not True:
            raise ValueError("This schema requires pinned local verifier; billed judge calls need richer accounting")
        pca=m.get("pca",{})
        if pca.get("fit_split")!="online_calibration" or not pca.get("encoder_id") or not pca.get("encoder_revision"):
            raise ValueError("Frozen encoder/train-only calibration PCA provenance required")
        base=manifest_path.parent
        locked_split=None; e0_receipt=None; e1_receipt=None; locked_pool=None
        for name in ("e0_lock","e1_gate","split_manifest","pca_artifact","pool_config"):
            path=_reference(base,m.get(name),name)
            if name=="split_manifest": locked_split=json.loads(path.read_text(encoding="utf-8-sig"))
            if name=="pool_config": locked_pool=json.loads(path.read_text(encoding="utf-8-sig"))
            if name=="e0_lock": e0_receipt=json.loads(path.read_text(encoding="utf-8-sig"))
            if name=="e1_gate": e1_receipt=json.loads(path.read_text(encoding="utf-8-sig"))
            if name in {"e0_lock","e1_gate"}:
                accepted={"PASS","LOCKED","E0_PASS","E0_PASS_AND_LOCK"} if name=="e0_lock" else {"PASS","E1_PASS","PASS_E1"}
                if json.loads(path.read_text(encoding="utf-8-sig")).get("status") not in accepted:
                    raise ValueError(f"Required stage gate has not passed: {name}")
        sources=m.get("sources",[])
        if not {"member_outputs","aggregate_outputs","scorer_source"}.issubset({v.get("kind") for v in sources}):
            raise ValueError("Missing member/aggregate/scorer source hashes")
        for record in sources: _reference(base,record,record.get("kind","source"))
        _validate_stage_bindings(m,e0_receipt,e1_receipt)
        rows=locked_pool.get("models",[])
        if [v.get("index") for v in rows]!=list(range(6)) or [(v.get("model_id"),v.get("revision")) for v in rows]!=[(v["model_id"],v["revision"]) for v in m["model_pool"]]:
            raise ValueError("NPZ model-column order differs from frozen pool")
        if m.get("combinations")!=[list(v) for v in SubsetFeatureMap(6,3).subsets]:
            raise ValueError("Aggregate score columns must follow locked lexicographic combinations")
        # Input JSON cannot self-authorize E3. Only the external strict stage
        # validator may attach a bound gate to the two in-memory environments.
        for key in ("e2_gate_sha256","e2_gate_bindings"):
            m.pop(key,None)
        m["e2_pass"]=False
        with np.load(matrix_path,allow_pickle=False) as data:
            arrays={key:data[key] for key in data.files}
        environment=cls(arrays,split=expected_split,manifest=m)
        environment._matrix_file_sha256=actual_matrix_sha256
        environment._manifest_file_sha256=actual_manifest_sha256
        _validate_locked_split(locked_split,environment._arrays["question_ids"].tolist(),m["question_fingerprints"],expected_split,m["pca"])
        return environment

    @classmethod
    def for_software_test(cls,arrays,*,split):
        return cls(arrays,split=split,manifest={"software_test_only":True},software_test_only=True)

    def __len__(self): return len(self._arrays["question_ids"])
    @property
    def phi_dimension(self): return self._arrays["phi"].shape[1]

    def context(self,row):
        qid=hashlib.sha256(("routing-question-v1:"+str(self._arrays["question_ids"][row])).encode()).hexdigest()
        phi=self._arrays["phi"][row].astype(float,copy=True); phi.flags.writeable=False
        return QuestionContext(qid,str(self._arrays["task_labels"][row]),phi)

    def new_session(self,caps): return _EnvironmentSession(self,caps)

    def public_provenance(self):
        if self.software_test_only: return {"software_test_only":True}
        m=self._manifest
        result={"matrix_sha256":m["matrix_sha256"],"e0_lock_sha256":m["e0_lock"]["sha256"],
                "e1_gate_sha256":m["e1_gate"]["sha256"],"pca_sha256":m["pca_artifact"]["sha256"],
                "model_pool":m["model_pool"],"cache_revision":m["cache_revision"],"aggregator":m["aggregator"],
                "scorer":m["scorer"],"sources":[{"kind":v["kind"],"sha256":v["sha256"]} for v in m["sources"]]}
        if m.get("e2_pass"):
            result.update(e2_gate_sha256=m.get("e2_gate_sha256"),e2_gate_bindings=m.get("e2_gate_bindings"))
        return result

class _EnvironmentSession:
    def __init__(self,environment,caps):
        self.environment,self.caps=environment,caps
        self._executed={}

    def _cell_cost(self,row,stem,index):
        a=self.environment._arrays
        calls=int(a[f"{stem}_execution_counts"][row,index])
        prompt=int(a[f"{stem}_prompt_tokens"][row,index])
        completion=int(a[f"{stem}_completion_tokens"][row,index])
        max_calls=self.caps.member_calls if stem=="member" else self.caps.aggregate_calls
        pcap=self.caps.member_prompt if stem=="member" else self.caps.aggregate_prompt
        ccap=self.caps.member_completion if stem=="member" else self.caps.aggregate_completion
        if calls>max_calls or prompt>calls*pcap or completion>calls*ccap:
            raise BudgetExceeded("Actual source cell violates locked execution/token cap")
        return ResourceCost(calls,0,prompt+completion)

    def execute_member_prior(self,row):
        """Preplanned all-six generation on calibration only; no combo is executed."""
        if self.environment.split != "online_calibration":
            raise ValueError("Member-only prior purchases are confined to paid calibration")
        if row in self._executed:
            raise ValueError("Question already executed")
        cost=ResourceCost(); observations=[]
        for index in range(6):
            cell=self._cell_cost(row,"member",index);cost+=cell
            observations.append(("member",index,cell.tokens))
        self._executed[row]={"main_index":None,"generated":set(range(6)),"executed":set()}
        values=self.environment._arrays.get("member_wall_time_s")
        return {"service":cost,"audit":ResourceCost(),"cost_observations":observations,
            "source_service_execution_wall_time_s":None if values is None else float(values[row].sum()),
            "source_audit_execution_wall_time_s":0.}

    def execute(self,row,main_index,alternate_index=None,*,include_evaluation=True):
        if row in self._executed: raise ValueError("Main output already frozen for this question")
        subsets=self.environment.mapper.subsets
        main=subsets[main_index]
        alternate=None if alternate_index is None else subsets[alternate_index]
        if alternate is not None and (len(set(alternate)-set(main))!=1 or len(set(main)-set(alternate))!=1):
            raise ValueError("Audit must replace exactly one member")
        service=ResourceCost(); observations=[]
        for model in main:
            cost=self._cell_cost(row,"member",model); service+=cost
            observations.append(("member",model,cost.tokens))
        cost=self._cell_cost(row,"aggregate",main_index); service+=cost
        observations.append(("aggregate",main_index,cost.tokens))
        audit=ResourceCost(); generated=set(main); executed={main_index}
        if alternate is not None:
            model=next(iter(set(alternate)-set(main)))
            mc=self._cell_cost(row,"member",model); ac=self._cell_cost(row,"aggregate",alternate_index)
            audit=mc+ac
            observations.extend((("member",model,mc.tokens),("aggregate",alternate_index,ac.tokens)))
            generated.add(model); executed.add(alternate_index)
        self._executed[row]={"main_index":main_index,"generated":generated,"executed":executed}
        result={"service":service,"audit":audit,"cost_observations":observations}
        if include_evaluation:
            result["main_quality_for_evaluation_only"]=_read_quality_for_evaluation(self.environment,row,main_index)
        def wall(stem,index):
            values=self.environment._arrays.get(stem+"_wall_time_s")
            return None if values is None else float(values[row,index])
        main_walls=[wall("member",model) for model in main]+[wall("aggregate",main_index)]
        audit_walls=[] if alternate is None else [wall("member",next(iter(set(alternate)-set(main)))),wall("aggregate",alternate_index)]
        result["source_service_execution_wall_time_s"]=None if any(v is None for v in main_walls) else sum(main_walls)
        result["source_audit_execution_wall_time_s"]=None if any(v is None for v in audit_walls) else sum(audit_walls)
        return result

class FeedbackBroker:
    """Only authorized executed original labels arrive; legal member scores are exposed."""
    def __init__(self,session,*,delay,feedback_mode="verifiable"):
        if not isinstance(delay,int) or delay<0: raise ValueError("Nonnegative integer delay required")
        if feedback_mode not in {"verifiable","final_only"}: raise ValueError("Unknown feedback mode")
        self.session,self.delay,self.feedback_mode=session,delay,feedback_mode
        self._queue=[]; self._sequence=0; self._requested=set(); self.member_history=[]
        self._member_reference_requested=set()

    @property
    def pending_labels(self): return sum(len(event[2]) for event in self._queue)

    def request(self,row,combination_indices,*,now):
        state=self.session._executed.get(row)
        if state is None: raise ValueError("Cannot score before execution")
        indices=tuple(combination_indices)
        if len(set(indices))!=len(indices) or any(i not in state["executed"] or (row,i) in self._requested for i in indices):
            raise ValueError("Only executed, not-yet-requested original labels are eligible")
        if not indices: return
        env=self.session.environment; context=env.context(row); features=env.mapper.transform(context.phi); due=now+self.delay
        labels=tuple(OriginalLabel(f"{context.question_id}:combo:{i}",context.question_id,features[i],
                     float(env._arrays["aggregate_scores"][row,i]),due) for i in indices)
        members=tuple((context.question_id,j,float(env._arrays["member_scores"][row,j])) for j in sorted(state["generated"])) if self.feedback_mode=="verifiable" else ()
        self._requested.update((row,i) for i in indices)
        heapq.heappush(self._queue,(due,self._sequence,labels,members)); self._sequence+=1

    def request_member_prior_reference(self,row,*,now):
        """One paid reference reveal exposes all six generated member labels.

        This is a separately named E5 reference-event accounting adapter. It
        buys no aggregate label and cannot be used to learn aggregate rewards.
        """
        state=self.session._executed.get(row)
        if self.feedback_mode!="verifiable" or self.session.environment.split!="online_calibration":
            raise ValueError("Paid member references require verifiable calibration")
        if state is None or state["generated"]!=set(range(6)) or state["executed"]:
            raise ValueError("Reference requires pre-executed all-six member-only answers")
        if row in self._member_reference_requested:
            raise ValueError("Member reference already purchased")
        env=self.session.environment;context=env.context(row)
        members=tuple((context.question_id,j,float(env._arrays["member_scores"][row,j])) for j in range(6))
        heapq.heappush(self._queue,(now+self.delay,self._sequence,(),members))
        self._sequence+=1;self._member_reference_requested.add(row)

    def deliver(self,now):
        labels=[]; members=[]
        while self._queue and self._queue[0][0]<=now:
            _,_,arrived,revealed=heapq.heappop(self._queue); labels.extend(arrived); members.extend(revealed)
        self.member_history.extend(members)
        return tuple(labels),tuple(members)

class _PastCostPredictor:
    def __init__(self,settings):
        self.settings=settings; self._totals=defaultdict(float); self._counts=defaultdict(int)

    def update(self,observations):
        for stem,index,tokens in observations:
            self._totals[(stem,index)]+=tokens; self._counts[(stem,index)]+=1

    def _tokens(self,stem,index):
        c=self.settings.caps
        upper=(c.member_prompt+c.member_completion)*c.member_calls if stem=="member" else (c.aggregate_prompt+c.aggregate_completion)*c.aggregate_calls
        key=(stem,index)
        return self._totals[key]/self._counts[key] if self.settings.cost_prediction=="history" and self._counts[key] else upper

    def predictions(self,mapper,main_index):
        s=self.settings; w=s.cost_weights
        result={"main_only":CostPrediction(max(w[1],1e-12),ResourceCost(0,1,0))}
        base=set(mapper.subsets[main_index]); labels=1 if s.main_feedback_natural else 2
        for subset in mapper.subsets:
            new=set(subset)-base
            if len(new)!=1 or len(base-set(subset))!=1: continue
            j=next(iter(new)); index=mapper.subsets.index(subset)
            estimate=w[0]*(s.caps.member_calls+s.caps.aggregate_calls)+w[1]*labels
            estimate+=w[2]*(self._tokens("member",j)+self._tokens("aggregate",index))
            result[subset]=CostPrediction(max(float(estimate),1e-12),s.caps.alternate(labels))
        return result

def _fixed_from_arrived_warmup(labels,*,now,n_combinations=20):
    """Empirical best OBSERVED combo from arrived, paid calibration labels only."""
    observed=defaultdict(list); seen={}; pending=0
    for label in labels:
        if label.available_at>now:
            pending+=1; continue
        if label.kind!="original" or ":combo:" not in label.label_id:
            raise ValueError("Fixed-combination selection requires original combination labels")
        index=int(label.label_id.rsplit(":combo:",1)[1])
        if not 0<=index<n_combinations or not np.isfinite(label.value) or not 0<=label.value<=1:
            raise ValueError("Invalid arrived warmup combination label")
        identity=(label.question_id,index,float(label.value),label.available_at)
        if label.label_id in seen:
            if seen[label.label_id]!=identity:
                raise ValueError("Conflicting warmup label ID")
            continue
        seen[label.label_id]=identity
        observed[index].append(float(label.value))
    means={i:float(np.mean(values)) for i,values in observed.items()}
    selected=min(means,key=lambda i:(-means[i],i)) if means else 0
    return selected,{"basis":"paid_arrived_online_calibration_labels_only" if means else "no_arrived_warmup_labels_lexicographic_default",
        "selected_index":selected,"arrived_labels_used":len(seen),"pending_labels_ignored":pending,
        "observed_combination_indices":sorted(means),"observed_empirical_means":means,
        "unobserved_combination_policy":"excluded; no optimistic imputation","locked_before_stream":True}

class _ContextualMemberRidges:
    """Six phi-only regressors: only broker-authorized GENERATED member labels.

    There is no member-score-to-aggregate-reward construction. The common
    environment still determines service quality from actual aggregate scores.
    """
    def __init__(self,phi_dimension,*,ridge=1.,noise_variance=1.):
        self._models=[SharedRidgeRegressor(phi_dimension,ridge=ridge,noise_variance=noise_variance) for _ in range(6)]
        self._phi_dimension=phi_dimension
        self._contexts={}; self._generated=defaultdict(set); self._seen={}

    @property
    def label_counts(self):
        return tuple(model.label_count for model in self._models)

    def record_generated(self,context,model_indices):
        phi=np.asarray(context.phi,dtype=float)
        if phi.shape!=(self._phi_dimension,) or not np.isfinite(phi).all():
            raise ValueError("Generated question has invalid frozen phi")
        indices=tuple(model_indices)
        if any(not isinstance(i,(int,np.integer)) or not 0<=i<6 for i in indices):
            raise ValueError("Generated member index outside frozen six-model pool")
        old=self._contexts.get(context.question_id)
        if old is not None and not np.array_equal(old,phi):
            raise ValueError("Question phi changed after generation")
        self._contexts[context.question_id]=phi.copy()
        self._generated[context.question_id].update(int(i) for i in indices)

    def observe(self,member_feedback,*,now):
        normalized=[]; batch={}
        for qid,model_index,value in member_feedback:
            if qid not in self._contexts or model_index not in self._generated[qid]:
                raise ValueError("Member feedback for an ungenerated answer is forbidden")
            value=float(value)
            if not np.isfinite(value) or not 0<=value<=1:
                raise ValueError("Invalid legal member quality")
            key=(qid,int(model_index))
            old=self._seen.get(key,batch.get(key))
            if old is not None and old!=value:
                raise ValueError("Conflicting original generated-member feedback")
            batch[key]=value; normalized.append((key,value))
        added=0
        for (qid,index),value in normalized:
            if (qid,index) in self._seen: continue
            label=OriginalLabel(f"{qid}:member:{index}",qid,self._contexts[qid],value,now)
            self._models[index].add_label(label,now=now)
            self._seen[(qid,index)]=value; added+=1
        return added

    def predict(self,phi):
        return np.array([float(model.predict(phi)) for model in self._models])

    def choose_subset(self,phi):
        scores=self.predict(phi)
        return tuple(sorted(int(i) for i in np.argsort(-scores,kind="stable")[:3]))

def _validate_pair(online,calibration,strategy):
    if online.split!="online_stream" or calibration.split!="online_calibration":
        raise ValueError("Initialization requires online_calibration, never E1/final data")
    if set(online._fingerprints)&set(calibration._fingerprints):
        raise ValueError("Online/calibration question leakage")
    if online.phi_dimension!=calibration.phi_dimension or online.software_test_only!=calibration.software_test_only:
        raise ValueError("Incompatible environments")
    if not online.software_test_only:
        for key in ("model_pool","cache_revision","aggregator","scorer","pca_artifact","pca","split_manifest","pool_config","e0_lock","e1_gate"):
            if online._manifest.get(key)!=calibration._manifest.get(key):
                raise ValueError(f"Frozen provenance mismatch: {key}")
        if strategy in {"random","periodic","max_uncertainty","max_uncertainty_per_cost","no_cost","decision"}:
            bound=online._manifest.get("e2_gate_bindings")
            gate=online._manifest.get("e2_gate_sha256")
            if (online._manifest.get("e2_pass") is not True or calibration._manifest.get("e2_pass") is not True
                or not _digest(gate) or calibration._manifest.get("e2_gate_sha256")!=gate
                or not isinstance(bound,dict) or calibration._manifest.get("e2_gate_bindings")!=bound):
                raise ValueError("E3 acquisition requires the same externally validated artifact-backed E2_PASS on both environments")
            expected={"online_matrix_sha256":getattr(online,"_matrix_file_sha256",None),
                "online_manifest_sha256":getattr(online,"_manifest_file_sha256",None),
                "calibration_matrix_sha256":getattr(calibration,"_matrix_file_sha256",None),
                "calibration_manifest_sha256":getattr(calibration,"_manifest_file_sha256",None),
                "e0_lock_sha256":online._manifest.get("e0_lock",{}).get("sha256"),
                "e1_gate_sha256":online._manifest.get("e1_gate",{}).get("sha256"),
                "pca_sha256":online._manifest.get("pca_artifact",{}).get("sha256"),
                "pool_config_sha256":online._manifest.get("pool_config",{}).get("sha256")}
            if any(not _digest(value) or bound.get(key)!=value for key,value in expected.items()):
                raise ValueError("E2 gate is bound to different or unverified online/calibration assets")
            if any(not _digest(bound.get(key)) for key in ("config_sha256","required_grid_sha256","algorithm_sources_sha256")):
                raise ValueError("E2 gate must bind its validated config, complete required grid and algorithm sources")

def validate_strategy_requirements(online,calibration,strategy,settings):
    """Preflight before any purchase or result directory creation."""
    if strategy not in STRATEGIES+("static_cold_start",):
        raise ValueError("Unknown strategy")
    _validate_pair(online,calibration,strategy)
    if strategy=="contextual_topk":
        if settings.feedback_mode!="verifiable":
            raise ValueError("contextual_topk is not applicable to final_only without legal member feedback")
        for environment in (online,calibration):
            if environment.software_test_only: continue
            permission=environment._manifest.get("feedback_permissions",{})
            if permission.get("mode")!="verifiable" or permission.get("generated_member_scores_on_reference_reveal") is not True:
                raise ValueError("Production contextual_topk requires explicit authorized generated-member feedback permissions")

def _clock():
    return time.process_time(),time.perf_counter()


def _elapsed(start):
    return {"process_time_s":max(0.,time.process_time()-start[0]),
            "wall_time_s":max(0.,time.perf_counter()-start[1])}


def _add_timing(target,measurement):
    for key,value in measurement.items():target[key]+=value


def _read_quality_for_evaluation(environment,row,combination_index):
    """Evaluator-only read after the policy has frozen its service and purchase."""
    return float(environment._arrays["aggregate_scores"][row,combination_index])


def _checkpoint_json_value(value):
    """Keep checkpoint state identical before/after its JSON persistence."""
    if isinstance(value,dict):return {str(key):_checkpoint_json_value(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_checkpoint_json_value(item) for item in value]
    if isinstance(value,np.ndarray):return _checkpoint_json_value(value.tolist())
    if isinstance(value,np.generic):return value.item()
    return value


def _ridge_checkpoint(model):
    return {"feature_dim":model.feature_dim,"ridge":model.ridge,
        "noise_variance":model.noise_variance,"bootstrap_min_labels":model.bootstrap_min_labels,
        "bootstrap_min_questions":model.bootstrap_min_questions,
        "design":model.design.tolist(),"design_inverse":model.design_inverse.tolist(),
        "rhs":model._rhs.tolist(),"theta":model.theta.tolist(),"prior_mean":model.prior_mean.tolist(),
        "arrived_original_labels":[{"label_id":v.label_id,"question_id":v.question_id,
            "features":v.features.tolist(),"value":float(v.value),"available_at":int(v.available_at),
            "kind":v.kind} for v in model.labels]}


def _broker_checkpoint(broker):
    """Pending entries contain NO future label values, only paid request metadata.

    A continuation can rehydrate these requests only from the same bound private
    environment when their due tick arrives. Frozen final evaluation must not
    deliver them merely because wall-clock time has elapsed.
    """
    env=broker.session.environment
    pending=[]
    for due,sequence,labels,members in sorted(broker._queue):
        qids={label.question_id for label in labels}|{str(item[0]) for item in members}
        if len(qids)!=1:raise ValueError("Feedback queue mixed different questions")
        pending.append({"available_at":int(due),"request_sequence":int(sequence),
            "question_id":next(iter(qids)),
            "label_ids":[label.label_id for label in labels],
            "combination_indices":[int(label.label_id.rsplit(":combo:",1)[1]) for label in labels],
            "generated_member_indices":sorted({int(item[1]) for item in members})})
    executed=[]
    for row,state in sorted(broker.session._executed.items()):
        executed.append({"row_index":int(row),"question_id":env.context(row).question_id,
            "main_index":None if state["main_index"] is None else int(state["main_index"]),"generated_member_indices":sorted(state["generated"]),
            "executed_combination_indices":sorted(state["executed"])})
    return {"feedback_mode":broker.feedback_mode,"delay":broker.delay,
        "sequence":broker._sequence,"pending_paid_requests_without_values":pending,
        "requested":[{"row_index":int(row),"combination_index":int(index)} for row,index in sorted(broker._requested)],
        "executed":executed,"pending_paid_labels":broker.pending_labels,
        "pending_values_serialized":False,
        "paid_member_reference_rows":sorted(broker._member_reference_requested)}


def _prediction_snapshot(learner,member_predictor,context,mapper,fixed_selection,strategy,statistics):
    if strategy in {"fixed_combination","fixed_with_paid_monitoring"}:
        means=fixed_selection["observed_empirical_means"]
        scores=[None if index not in means else float(means[index]) for index in range(len(mapper.subsets))]
        return {"predicted_combination_scores_before_decision":scores,
            "prediction_supported_combination_indices":sorted(int(i) for i in means),
            "predicted_combination_working_variance_before_decision":None,
            "service_selection_values_before_decision":None,
            "prediction_semantics":"paid_warmup_observed_combo_empirical_mean_only_not_full_calibration_best",
            "predicted_member_scores_before_decision":None}
    if member_predictor is not None:
        members=member_predictor.predict(context.phi)
        scores=[float(np.mean(members[list(subset)])) for subset in mapper.subsets]
        return {"predicted_combination_scores_before_decision":scores,
            "prediction_supported_combination_indices":list(range(len(mapper.subsets))),
            "predicted_combination_working_variance_before_decision":None,
            "service_selection_values_before_decision":None,
            "prediction_semantics":"mean_member_prediction_ranking_proxy_not_aggregate_reward_or_probability",
            "predicted_member_scores_before_decision":members.tolist()}
    return {"predicted_combination_scores_before_decision":list(statistics["predicted_means"]),
        "prediction_supported_combination_indices":list(range(len(mapper.subsets))),
        "predicted_combination_working_variance_before_decision":statistics["working_variances"],
        "service_selection_values_before_decision":list(statistics["selection_values"]),
        "prediction_semantics":"shared_second_order_aggregate_quality_regression_raw_unclipped",
        "predicted_member_scores_before_decision":None}


def run_simulation(online,calibration,*,strategy,seed=42,label_rate=25,delay=0,settings):
    """One complete prospective stream with paid common warmup and hard budgets.

    fixed_combination freezes the empirical winner among paid observed warmup
    combinations and buys no unneeded online monitoring. The old monitoring
    version remains explicit as fixed_with_paid_monitoring. none is a learning
    second-order baseline with paid main feedback, not a no-feedback policy.

    Prediction snapshots precede current outputs and feedback. Only the offline
    evaluator reads unpurchased quality. Controller CPU/wall time is separated
    from environment replay, diagnostic snapshots/evaluation and state export.
    """
    preflight_start=_clock()
    if strategy not in STRATEGIES+("static_cold_start",):raise ValueError("Unknown strategy")
    if label_rate not in (10,25,50,100):raise ValueError("Primary rates are 10/25/50/100")
    validate_strategy_requirements(online,calibration,strategy,settings)
    if settings.warmup_mode!="random_combo" and strategy in {"fixed_combination","fixed_with_paid_monitoring","contextual_topk","static_cold_start"}:
        raise ValueError("Paid member-prior ablation is defined for shared second-order policies only")
    preflight=_elapsed(preflight_start)
    totals={name:{"process_time_s":0.,"wall_time_s":0.} for name in
        ("controller","environment_replay","offline_evaluation")}
    controller_start=_clock()
    n=len(online);total_labels=n*label_rate//100
    ledger=BudgetLedger(ResourceCost(settings.execution_limit,total_labels,settings.token_limit))
    normal=settings.caps.service(settings.main_feedback_natural)
    ledger.reserve_future_service(normal*n)
    mapper=online.mapper
    learner=SharedRidgeRegressor(online.phi_dimension*mapper.basis_dim,ridge=settings.ridge,
        noise_variance=settings.noise_variance,bootstrap_min_labels=settings.bootstrap_min_labels,
        bootstrap_min_questions=settings.bootstrap_min_questions)
    rng=np.random.default_rng(seed)
    init_rng=np.random.default_rng(np.random.SeedSequence([seed,1001]))
    bootstrap_rng=np.random.default_rng(np.random.SeedSequence([seed,1002]))
    order=rng.permutation(n)
    if settings.stream_order!="iid":
        tasks=("gsm8k","bbh") if settings.stream_order=="gsm8k_then_bbh" else ("bbh","gsm8k")
        if {online.context(int(row)).task for row in order}!=set(tasks):
            raise ValueError("Task-shift run requires exactly the two frozen tasks")
        order=np.asarray([row for task in tasks for row in order if online.context(int(row)).task==task],dtype=int)
    initializer=calibration.new_session(settings.caps)
    init_broker=FeedbackBroker(initializer,delay=0,feedback_mode=settings.feedback_mode)
    session=online.new_session(settings.caps)
    broker=FeedbackBroker(session,delay=delay,feedback_mode=settings.feedback_mode)
    costs=_PastCostPredictor(settings);warmup=[];initial_arrived=[]
    member_predictor=_ContextualMemberRidges(online.phi_dimension,ridge=settings.ridge,
        noise_variance=settings.noise_variance) if strategy=="contextual_topk" else None
    initial_max=min(cold_start_limit(total_labels,settings.warmup_max_labels),len(calibration))
    prior_reference_target=initial_max//2 if settings.warmup_mode=="paid_member_prior_then_combo" else 0
    ability_prior=_ContextualMemberRidges(online.phi_dimension,ridge=settings.ridge,
        noise_variance=settings.noise_variance) if prior_reference_target else None
    prior_installed=False;prior_reference_count=0
    _add_timing(totals["controller"],_elapsed(controller_start))
    for position,row in enumerate(init_rng.permutation(len(calibration))[:initial_max]):
        controller_start=_clock()
        member_only=position<prior_reference_target
        upper=(ResourceCost(6*settings.caps.member_calls,1,
            6*settings.caps.member_calls*(settings.caps.member_prompt+settings.caps.member_completion))
            if member_only else settings.caps.service(True))
        if not ledger.can_reserve(upper):
            _add_timing(totals["controller"],_elapsed(controller_start));break
        combo=int(init_rng.integers(20));rid=f"warmup:{position}"
        ledger.reserve(rid,upper)
        _add_timing(totals["controller"],_elapsed(controller_start))
        replay_start=_clock()
        outcome=(initializer.execute_member_prior(int(row)) if member_only
            else initializer.execute(int(row),combo,include_evaluation=False))
        _add_timing(totals["environment_replay"],_elapsed(replay_start))
        controller_start=_clock()
        if member_predictor is not None:member_predictor.record_generated(calibration.context(int(row)),mapper.subsets[combo])
        ledger.settle(rid,outcome["service"]+ResourceCost(0,1,0))
        _add_timing(totals["controller"],_elapsed(controller_start))
        replay_start=_clock()
        if member_only:init_broker.request_member_prior_reference(int(row),now=0)
        else:init_broker.request(int(row),(combo,),now=0)
        labels,member_labels=init_broker.deliver(0)
        _add_timing(totals["environment_replay"],_elapsed(replay_start))
        controller_start=_clock();initial_arrived.extend(labels)
        if member_only:
            ability_prior.record_generated(calibration.context(int(row)),range(6))
            ability_prior.observe(member_labels,now=0);prior_reference_count+=1
        elif ability_prior is not None and not prior_installed:
            # Explicit prior on aggregate predictor coefficients, NOT aggregate
            # pseudo-observations: average contextual member capability is only
            # a cold-start proxy, not the actual frozen aggregator's reward.
            prior=np.zeros((online.phi_dimension,mapper.basis_dim))
            for index,model in enumerate(ability_prior._models):prior[:,1+index]=model.theta/3.
            learner=SharedRidgeRegressor(online.phi_dimension*mapper.basis_dim,
                ridge=settings.ridge,noise_variance=settings.noise_variance,
                bootstrap_min_labels=settings.bootstrap_min_labels,
                bootstrap_min_questions=settings.bootstrap_min_questions,prior_mean=prior.ravel())
            prior_installed=True
        for label in labels:learner.add_label(label,now=0)
        if member_predictor is not None:member_predictor.observe(member_labels,now=0)
        costs.update(outcome["cost_observations"])
        warmup.append({"question_id":calibration.context(int(row)).question_id,
            "combination_index":None if member_only else combo,
            "purchase_kind":"paid_all_six_member_reference" if member_only else "paid_combo_final_label",
            "feedback_units":1,"generated_member_labels":len(member_labels),
            "aggregate_labels":len(labels),"actual_cost":asdict(outcome["service"]+ResourceCost(0,1,0))})
        _add_timing(totals["controller"],_elapsed(controller_start))
    if ability_prior is not None and prior_reference_count and not prior_installed:
        raise BudgetExceeded("Paid-prior warmup could not afford any planned real-combination warmup; do not silently fall back")
    controller_start=_clock()
    initialization_cost=asdict(ledger.spent)
    initial_member_counts=list(member_predictor.label_counts) if member_predictor is not None else None
    fixed_index,fixed_selection=_fixed_from_arrived_warmup(initial_arrived,now=0)
    traces=[];draws=None;draws_at=-1
    _add_timing(totals["controller"],_elapsed(controller_start))
    initialization_controller=dict(totals["controller"])
    for tick,row in enumerate(order):
        tick_timing={name:{"process_time_s":0.,"wall_time_s":0.} for name in totals}
        replay_start=_clock();arrived,revealed=broker.deliver(tick)
        _add_timing(tick_timing["environment_replay"],_elapsed(replay_start))
        controller_start=_clock()
        for label in arrived:learner.add_label(label,now=tick)
        if member_predictor is not None:member_predictor.observe(revealed,now=tick)
        context=online.context(int(row));statistics={}
        is_fixed=strategy in {"fixed_combination","fixed_with_paid_monitoring"}
        features=None if is_fixed or member_predictor is not None else mapper.transform(context.phi)
        main_policy=strategy if strategy in {"linucb","thompson"} else "greedy"
        if is_fixed:main=fixed_index
        elif member_predictor is not None:main=mapper.subsets.index(member_predictor.choose_subset(context.phi))
        else:main=choose_main(learner,features,policy=main_policy,rng=rng,decision_statistics=statistics)
        member_counts_before=list(member_predictor.label_counts) if member_predictor is not None else None
        predecision_count=learner.label_count
        if strategy in {"decision","no_cost"} and draws_at!=learner.label_count:
            draws=learner.bootstrap_coefficients(settings.bootstrap_replicates,rng=bootstrap_rng);draws_at=learner.label_count
        sid,pid=f"service:{tick}",f"feedback:{tick}"
        ledger.reserve_service(sid,normal)
        if (strategy in {"fixed_combination","static_cold_start"}
            or settings.main_feedback_natural and strategy in {"fixed_with_paid_monitoring","contextual_topk","none"}):
            plan=AcquisitionPlan(main,mapper.subsets[main],None,None,"none",(),0.,ResourceCost(),0.,0.,(),strategy)
        elif strategy in {"none","fixed_with_paid_monitoring","contextual_topk"} and not settings.main_feedback_natural:
            paid=CostPrediction(max(settings.cost_weights[1],1e-12),ResourceCost(0,1,0))
            permitted=paid.upper_bound.fits_within(ledger.available)
            plan=AcquisitionPlan(main,mapper.subsets[main],None,None,"main_only" if permitted else "none",
                (mapper.subsets[main],) if permitted else (),paid.scalar if permitted else 0.,
                paid.upper_bound if permitted else ResourceCost(),0.,0.,(),strategy)
        else:
            predictions=costs.predictions(mapper,main)
            plan=plan_acquisition(learner,features,mapper.subsets,main_index=main,
                strategy="none" if strategy in {"fixed_with_paid_monitoring","contextual_topk"} else strategy,
                main_feedback_expected=settings.main_feedback_natural,predicted_costs=predictions,
                available=ledger.available,bootstrap_coefficients=draws,kappa=settings.kappa,
                threshold=settings.threshold,round_index=tick,period=settings.period,rng=rng)
        if plan.action_kind!="none":ledger.reserve(pid,plan.reservation)
        _add_timing(tick_timing["controller"],_elapsed(controller_start))
        # Capture policy predictions while its current-answer/label permissions are still closed.
        evaluation_start=_clock()
        # Diagnostic support bookkeeping is not an algorithmic decision cost;
        # read IDs only, without cloning every stored feature vector.
        trained_combinations=sorted({int(label_id.rsplit(":combo:",1)[1]) for label_id in learner._labels})
        snapshot=_prediction_snapshot(learner,member_predictor,context,mapper,fixed_selection,strategy,statistics)
        _add_timing(tick_timing["offline_evaluation"],_elapsed(evaluation_start))
        replay_start=_clock();outcome=session.execute(int(row),main,plan.alternate_index,include_evaluation=False)
        _add_timing(tick_timing["environment_replay"],_elapsed(replay_start))
        controller_start=_clock()
        if member_predictor is not None:
            generated=set(mapper.subsets[main])
            if plan.alternate_index is not None:generated.update(mapper.subsets[plan.alternate_index])
            member_predictor.record_generated(context,generated)
        service_actual=outcome["service"]+ResourceCost(0,int(settings.main_feedback_natural),0)
        ledger.settle(sid,service_actual)
        requested=[main] if settings.main_feedback_natural else []
        requested.extend(mapper.subsets.index(subset) for subset in plan.feedback_subsets)
        if len(set(requested))!=len(requested):raise ValueError("Shared original main label purchased twice")
        purchase_actual=outcome["audit"]+ResourceCost(0,len(plan.feedback_subsets),0)
        if plan.action_kind!="none":ledger.settle(pid,purchase_actual)
        _add_timing(tick_timing["controller"],_elapsed(controller_start))
        replay_start=_clock();broker.request(int(row),requested,now=tick);immediate,immediate_members=broker.deliver(tick)
        _add_timing(tick_timing["environment_replay"],_elapsed(replay_start))
        controller_start=_clock()
        for label in immediate:learner.add_label(label,now=tick)
        if member_predictor is not None:member_predictor.observe(immediate_members,now=tick)
        costs.update(outcome["cost_observations"])
        actual_scalar=sum(w*v for w,v in zip(settings.cost_weights,(purchase_actual.executions,purchase_actual.labels,purchase_actual.tokens)))
        _add_timing(tick_timing["controller"],_elapsed(controller_start))
        evaluation_start=_clock();quality=_read_quality_for_evaluation(online,int(row),main)
        _add_timing(tick_timing["offline_evaluation"],_elapsed(evaluation_start))
        trace={"step":tick,"question_id":context.question_id,"task":context.task,
            "main_index":main,"main_members":list(mapper.subsets[main]),"alternate_index":plan.alternate_index,
            "action_kind":plan.action_kind,"quality":quality,
            "policy_label_count_before_decision":predecision_count,"policy_member_label_counts_before_decision":member_counts_before,
            "arrived_labels_before_decision":len(arrived),"arrived_combination_indices_before_decision":trained_combinations,
            "immediate_labels_after_freeze":len(immediate),"requested_combination_labels":len(requested),
            "pending_labels":broker.pending_labels,"legally_revealed_generated_member_feedback":len(revealed)+len(immediate_members),
            "predicted_incremental_cost":plan.predicted_cost,"actual_incremental_cost":actual_scalar,
            "predicted_cost_error":actual_scalar-plan.predicted_cost,"information_gain_proxy":plan.information_gain,
            "directional_information_evaluated":strategy in {"decision","no_cost"},
            "acquisition_score":plan.score,"service_cost":asdict(service_actual),"purchase_cost":asdict(purchase_actual),
            "cumulative_cost":asdict(ledger.spent),"protected_future_service":asdict(ledger.future_service),
            "source_service_execution_wall_time_s":outcome["source_service_execution_wall_time_s"],
            "source_audit_execution_wall_time_s":outcome["source_audit_execution_wall_time_s"],
            "service_frozen_before_audit":True,"prediction_snapshot_precedes_current_execution_and_feedback":True,**snapshot}
        for name,measurement in tick_timing.items():
            _add_timing(totals[name],measurement)
            for key,value in measurement.items():trace[name+"_"+key]=value
        traces.append(trace)
    evaluation_start=_clock()
    per_task={task:float(np.mean([v["quality"] for v in traces if v["task"]==task])) for task in sorted({v["task"] for v in traces})}
    _add_timing(totals["offline_evaluation"],_elapsed(evaluation_start))
    export_start=_clock()
    member_state=None if member_predictor is None else {
        "models":[_ridge_checkpoint(model) for model in member_predictor._models],
        "generated_contexts":{qid:phi.tolist() for qid,phi in member_predictor._contexts.items()},
        "generated_members":{qid:sorted(indices) for qid,indices in member_predictor._generated.items()},
        "arrived_seen_keys":[{"question_id":qid,"member_index":index} for qid,index in sorted(member_predictor._seen)]}
    checkpoint={"schema":"restricted_feedback_policy_checkpoint_v1","software_test_only":online.software_test_only,
        "strategy":strategy,"seed":int(seed),"label_rate":label_rate,"feedback_delay":delay,
        "next_tick":n,"frozen_as_of_tick":n-1,"pending_feedback_flushed":False,
        "serving_policy":"fixed" if is_fixed else "independent_topk" if member_predictor is not None else main_policy,
        "linucb_alpha":1.0,"coefficient_interpretation":"Predictive ridge solution; fixed-K main and pair effects are not separately identified.",
        "feature_map":{"n_models":6,"k":3,"basis_dim":mapper.basis_dim,"phi_dimension":online.phi_dimension,
            "combinations":[list(v) for v in mapper.subsets]},
        "settings":asdict(settings),"shared_ridge":_ridge_checkpoint(learner),"member_predictor":member_state,
        "fixed_selection":fixed_selection if is_fixed else None,
        "rng_states":{"policy":rng.bit_generator.state,"warmup":init_rng.bit_generator.state,"bootstrap":bootstrap_rng.bit_generator.state},
        "cached_bootstrap":{"arrived_label_count":draws_at,"coefficients":None if draws is None else draws.tolist()},
        "cost_history":[{"kind":stem,"index":int(index),"total_tokens":value,"observation_count":costs._counts[(stem,index)]}
            for (stem,index),value in sorted(costs._totals.items())],
        "feedback":_broker_checkpoint(broker),"warmup_feedback":_broker_checkpoint(init_broker),
        "budget":ledger.snapshot(),"stream_question_order":[online.context(int(row)).question_id for row in order],
        "online_provenance":online.public_provenance(),"calibration_provenance":calibration.public_provenance(),
        "restore_boundary":"Pending values are absent. Rehydrate only against the exact bound environment at an authorized due tick; frozen sealed evaluation must not flush them."}
    checkpoint=_checkpoint_json_value(checkpoint)
    export_timing=_elapsed(export_start)
    measurements={name+"_"+key:value for name,measurement in totals.items() for key,value in measurement.items()}
    measurements.update({"initialization_controller_"+key:value for key,value in initialization_controller.items()})
    measurements.update({"checkpoint_export_"+key:value for key,value in export_timing.items()})
    measurements.update({"preflight_"+key:value for key,value in preflight.items()})
    measurements["scope"]="Actual local process CPU and wall clocks; controller excludes private matrix replay, evaluator prediction snapshots/quality reads and checkpoint export; model execution wall times are separate source measurements, not inferred GPU kernel seconds."
    source_service=[v["source_service_execution_wall_time_s"] for v in traces]
    source_audit=[v["source_audit_execution_wall_time_s"] for v in traces]
    summary={"status":"SOFTWARE_TEST_ONLY" if online.software_test_only else "COMPLETED_RESTRICTED_FEEDBACK_SIMULATION_ONLY",
        "research_complete":False,"strategy":strategy,"seed":int(seed),"label_rate":label_rate,"feedback_delay":delay,
        "n_questions":n,"all_questions_served":len(traces)==n,"quality_micro":float(np.mean([v["quality"] for v in traces])),
        "quality_task_macro":float(np.mean(list(per_task.values()))),"per_task_quality":per_task,
        "warmup_labels":len(warmup),"warmup_cost":initialization_cost,
        "warmup_feedback_protocol":("common paid offline online_calibration combo labels; delay zero; arrived before stream"
            if settings.warmup_mode=="random_combo" else "E5 paid reference-event adapter: first half all-six member references, remaining combo final labels; distinct calibration questions; one feedback unit each; no combo pseudo-labels"),
        "warmup_mode":settings.warmup_mode,"stream_order":settings.stream_order,"paid_member_reference_events":prior_reference_count,
        "prior_member_labels_used":0 if ability_prior is None else sum(ability_prior.label_counts),
        "explicit_member_prior_installed":prior_installed,
        "warmup_arrived_combo_labels":len(initial_arrived),"fixed_selection":fixed_selection if is_fixed else None,
        "fixed_definition":"budgeted paid observed-warmup empirical winner; not full-calibration oracle" if is_fixed else None,
        "fixed_online_feedback":"paid_monitoring" if strategy=="fixed_with_paid_monitoring" else "no_voluntary_online_monitoring" if strategy=="fixed_combination" else None,
        "predictor":"six_independent_phi_ridges" if member_predictor is not None else "fixed_observed_warmup_combo" if is_fixed else "shared_second_order_ridge",
        "initial_member_label_counts":initial_member_counts,"final_member_label_counts":list(member_predictor.label_counts) if member_predictor is not None else None,
        "member_labels_used_by_policy":sum(member_predictor.label_counts) if member_predictor is not None else 0,
        "member_feedback_prior":("none; only broker-authorized scores of paid generated answers" if ability_prior is None else
            "explicit ridge prior mean from paid all-six member contextual fits divided by K; not aggregate training labels; bootstrap centered on identical prior"),
        "e2_research_complete":False,"arrived_original_labels":learner.label_count,
        "pending_paid_labels_at_stream_end":broker.pending_labels,"member_feedback_records_exposed":len(init_broker.member_history)+len(broker.member_history),
        "feedback_mode":settings.feedback_mode,"budget":ledger.snapshot(),"settings":asdict(settings),
        "cost_prediction_uses_current_hidden_lengths":False,"none_definition":"no alternate; paid main-only feedback",
        "label_allocation":"strategy controlled under global cap; no forced equal audit count",
        "evaluation":"hidden matrix scores assess frozen service; never revealed without broker permission",
        "generation_replicates":1,"stream_seed_is_not_new_generation":True,
        "runtime_measurements":measurements,"controller_process_time_s":totals["controller"]["process_time_s"],
        "controller_wall_time_s":totals["controller"]["wall_time_s"],
        "offline_evaluation_process_time_s":totals["offline_evaluation"]["process_time_s"],
        "offline_evaluation_wall_time_s":totals["offline_evaluation"]["wall_time_s"],
        "source_service_execution_wall_time_s":None if any(v is None for v in source_service) else sum(source_service),
        "source_audit_execution_wall_time_s":None if any(v is None for v in source_audit) else sum(source_audit),
        "source_execution_wall_time_scope":"Sum of the selected pinned source attempts, including known failures; physical full-cache generation is reported separately.",
        "policy_checkpoint_schema":checkpoint["schema"],
        "online_provenance":online.public_provenance(),"calibration_provenance":calibration.public_provenance()}
    return {"summary":summary,"traces":traces,"warmup":warmup,"policy_checkpoint":checkpoint}

def save_simulation(result,output_dir):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/"summary.json").write_text(json.dumps(result["summary"],ensure_ascii=False,indent=2),encoding="utf8")
    with (output_dir/"traces.jsonl").open("w",encoding="utf8") as stream:
        for row in result["traces"]: stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    (output_dir/"warmup_anonymous.json").write_text(json.dumps(result["warmup"],indent=2),encoding="utf8")
    (output_dir/"policy_checkpoint.json").write_text(json.dumps(result["policy_checkpoint"],ensure_ascii=False,allow_nan=False),encoding="utf8")

def paired_question_bootstrap(results,*,reference="none",replicates=2000,seed=20261030,confidence=.95):
    """Question-stratified pairing, conditional on the pinned generation cache."""
    if replicates<2 or not 0<confidence<1: raise ValueError("Invalid bootstrap settings")
    grouped=defaultdict(dict)
    for result in results:
        s=result["summary"]; grouped[(s["label_rate"],s["feedback_delay"],s["strategy"])][s["seed"]]=result
    output=[]; rng=np.random.default_rng(seed)
    for (rate,delay,strategy),runs in grouped.items():
        if strategy==reference: continue
        controls=grouped.get((rate,delay,reference))
        if not controls or set(controls)!=set(runs): raise ValueError("Pairing requires matching strategies/seeds")
        per_question=defaultdict(list); tasks={}
        for stream_seed,result in runs.items():
            treated={r["question_id"]:r for r in result["traces"]}
            control={r["question_id"]:r for r in controls[stream_seed]["traces"]}
            if treated.keys()!=control.keys(): raise ValueError("Methods must serve the same entire question set")
            for qid in treated:
                if treated[qid]["task"]!=control[qid]["task"]: raise ValueError("Task mismatch")
                tasks[qid]=treated[qid]["task"]
                per_question[qid].append(treated[qid]["quality"]-control[qid]["quality"])
        strata={task:np.array([np.mean(v) for qid,v in per_question.items() if tasks[qid]==task]) for task in sorted(set(tasks.values()))}
        draws=np.mean([rng.choice(v,(replicates,len(v)),replace=True).mean(axis=1) for v in strata.values()],axis=0)
        alpha=(1-confidence)/2
        output.append({"strategy":strategy,"reference":reference,"label_rate":rate,"delay":delay,
            "task_macro_quality_difference":float(np.mean([v.mean() for v in strata.values()])),
            "question_paired_ci":[float(v) for v in np.quantile(draws,[alpha,1-alpha])],
            "stream_seeds":sorted(runs),"question_count":len(per_question),"bootstrap_replicates":replicates,
            "scope":"one pinned generation cache; paired question units; stream seeds averaged within question",
            "research_complete":False})
    return output

