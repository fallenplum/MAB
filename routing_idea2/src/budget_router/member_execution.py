"""Independent question-only member execution, separate from aggregation jobs.

Reference answers/scores are never accepted here. Formal v1 model outputs are
regenerated under pinned local NF4 specs; historical answer caches are not rewards.
"""
from __future__ import annotations
from dataclasses import asdict
import math
from .inference import ModelSpec,GenerationFailure,configuration_fingerprint,_validated_usage
from .execution import ExecutionStore,ALLOWED_SPLITS,_digest,_text_hash
import time
import uuid

MEMBER_PROMPT_VERSION="independent_user_only_brief_final_v2"
MEMBER_INSTRUCTION=(
    "Solve the original problem independently. Give brief reasoning, then end with a single line "
    "exactly of the form 'Final answer: <answer>'. For multiple choice, use the option letter "
    "in parentheses. For numeric questions, use only the final number on that line. "
    "Do not request or assume access to a reference answer."
)
MEMBER_FIELDS={"job_id","question_id","task","subtask","split","question","model_alias","sample_index",
               "seed","model_fingerprint","model_id","model_revision","member_mode","prompt_version",
               "source_question_sha256","messages","sampling_control"}


def member_mode(spec:ModelSpec,*,sampling_control=False):
    spec.validate()
    if spec.backend!="transformers" or spec.quantization!="nf4" or not spec.model_path:
        raise ValueError("Formal members require pinned local Transformers NF4 model specs")
    if not isinstance(sampling_control,bool):raise ValueError("sampling_control must be an explicit bool")
    sampled=abs(spec.temperature-.7)<1e-12 and abs(spec.top_p-.95)<1e-12
    if sampling_control:
        if not sampled:raise ValueError("Strong control requires shared temp0.7/top_p0.95 protocol")
        return "strong_model_resample"
    if spec.temperature==0:return "independent_deterministic"
    if sampled:return "independent_sampled"
    raise ValueError("Members allow legacy temp0 or shared temp0.7/top_p0.95 only")


def member_messages(question):
    return [{"role":"user","content":MEMBER_INSTRUCTION+"\n\nOriginal problem:\n"+question}]


def make_member_job(*,question_id:str,task:str,question:str,model_alias:str,seed:int,
                    model_spec:ModelSpec,split:str,subtask=None,sample_index=0,sampling_control=False):
    mode=member_mode(model_spec,sampling_control=sampling_control)
    if not isinstance(question_id,str) or not question_id or split not in ALLOWED_SPLITS or task not in {"gsm8k","bbh"}:
        raise ValueError("Unknown member split/task")
    if task=="bbh" and not subtask:raise ValueError("BBH member jobs require the pinned subtask")
    if not isinstance(question,str) or not question.strip() or not isinstance(model_alias,str) or not model_alias:
        raise ValueError("Question and model alias are required")
    if isinstance(sample_index,bool) or not isinstance(sample_index,int) or sample_index not in {0,1,2}:
        raise ValueError("sample_index must be 0,1,2")
    if not sampling_control and sample_index!=0:
        raise ValueError("Ordinary cached member generation has sample_index=0")
    if sampling_control and seed==42:
        raise ValueError("Strong controls require fresh seeds, not the ordinary cache seed42")
    if isinstance(seed,bool) or not isinstance(seed,int):raise ValueError("Integer generation seed required")
    row=dict(question_id=question_id,task=task,subtask=subtask,split=split,question=question,model_alias=model_alias,
        sample_index=sample_index,seed=seed,model_fingerprint=configuration_fingerprint(model_spec),
        model_id=model_spec.model_id,model_revision=model_spec.revision,member_mode=mode,sampling_control=sampling_control,
        prompt_version=MEMBER_PROMPT_VERSION,source_question_sha256=_text_hash(question),
        messages=member_messages(question))
    row["job_id"]=_digest(row)
    return row


def validate_member_job(job,spec=None):
    if set(job)!=MEMBER_FIELDS:raise ValueError("Question-only member schema rejects labels, scores and arbitrary fields")
    if job["job_id"]!=_digest({key:value for key,value in job.items() if key!="job_id"}):
        raise ValueError("Member job content hash mismatch")
    if job["split"] not in ALLOWED_SPLITS or job["task"] not in {"gsm8k","bbh"} or not job["question_id"]:
        raise ValueError("Invalid member question/task/split")
    if job["task"]=="bbh" and not job["subtask"]:raise ValueError("BBH subtask required")
    if job["prompt_version"]!=MEMBER_PROMPT_VERSION or job["messages"]!=member_messages(job["question"]):
        raise ValueError("Member prompt must contain only the pinned instruction and original question")
    if _text_hash(job["question"])!=job["source_question_sha256"]:
        raise ValueError("Member source question hash mismatch")
    if isinstance(job["sample_index"],bool) or not isinstance(job["sample_index"],int) or job["sample_index"] not in {0,1,2}:
        raise ValueError("Invalid member sample index")
    if isinstance(job["seed"],bool) or not isinstance(job["seed"],int):raise ValueError("Integer generation seed required")
    if spec is not None:
        if job["model_fingerprint"]!=configuration_fingerprint(spec) or job["model_id"]!=spec.model_id or job["model_revision"]!=spec.revision:
            raise ValueError("Member job/spec identity mismatch")
        if job["member_mode"]!=member_mode(spec,sampling_control=job["sampling_control"]):raise ValueError("Member/control sampling protocol mismatch")
    if not isinstance(job["sampling_control"],bool):raise ValueError("Explicit sampling_control bool required")
    if not job["sampling_control"] and job["sample_index"]!=0:
        raise ValueError("Ordinary member sample index must be zero")
    if job["sampling_control"] and (job["member_mode"]!="strong_model_resample" or job["seed"]==42):
        raise ValueError("Strong-control role and fresh generation seed required")
    if not job["sampling_control"] and job["member_mode"]=="strong_model_resample":
        raise ValueError("Strong-control mode requires explicit sampling_control")
    if job["member_mode"] not in {"independent_deterministic","independent_sampled","strong_model_resample"}:
        raise ValueError("Unknown member generation mode")
    return job["job_id"]


def validate_resample_groups(jobs):
    groups={}
    for job in jobs:
        if job["member_mode"]=="strong_model_resample":
            key=(job["question_id"],job["model_alias"],job["model_fingerprint"],job["split"])
            groups.setdefault(key,[]).append(job)
    for rows in groups.values():
        if len(rows)!=3 or {row["sample_index"] for row in rows}!={0,1,2} or len({row["seed"] for row in rows})!=3 or any(row["seed"]==42 for row in rows):
            raise ValueError("Strong-model control requires 3 distinct sample indices and generation seeds per question")


class MemberStore(ExecutionStore):
    """Reuse durable storage primitives, independently validate question-only jobs."""
    def _cached(self,job,spec):
        row=self.completed.get(job["job_id"])
        if row is None:return None
        for key in ("question_id","task","subtask","split","model_alias","sample_index","seed",
                    "model_fingerprint","member_mode","sampling_control","prompt_version","source_question_sha256"):
            if row.get(key)!=job.get(key):raise ValueError("Cached member result does not match the submitted job")
        result=row.get("result",{})
        if result.get("model_id")!=spec.model_id or result.get("model_revision")!=spec.revision or not result.get("model_identity_verified"):
            raise ValueError("Cached member model identity is unverified")
        if _validated_usage(result) is None:raise ValueError("Cached member usage is missing")
        return row

    def run(self,backend,job):
        spec=backend.spec
        validate_member_job(job,spec)
        with self._lock():
            self._reload()
            cached=self._cached(job,spec)
            if cached is not None:return cached,True
            if job["job_id"] in self.unreconciled:
                raise RuntimeError("Prior member failure has unknown cost; reconcile before retry")
            common={key:job.get(key) for key in ("job_id","question_id","task","subtask","split","model_alias",
                "sample_index","seed","model_fingerprint","member_mode","sampling_control","prompt_version","source_question_sha256")}
            common.update(started_at_unix=time.time(),attempt_id=uuid.uuid4().hex)
            try:
                result=backend.generate(job["messages"],job["seed"])
                usage=_validated_usage({"prompt_tokens":result.prompt_tokens,"completion_tokens":result.completion_tokens})
                if usage is None:raise GenerationFailure("invalid_member_usage")
                if not result.model_identity_verified or result.model_id!=spec.model_id or result.model_revision!=spec.revision or result.backend!=spec.backend:
                    raise GenerationFailure("member_model_identity_mismatch",usage=usage)
                if result.seed!=job["seed"] or result.prompt_tokens>spec.max_input_tokens or result.completion_tokens>spec.max_new_tokens:
                    raise GenerationFailure("member_protocol_limit_mismatch",usage=usage)
                if not isinstance(result.text,str) or not math.isfinite(result.wall_time_s) or result.wall_time_s<0:
                    raise GenerationFailure("invalid_member_result",usage=usage)
                data=asdict(result)
                data["completion_reached_cap"]=result.completion_tokens>=spec.max_new_tokens
                row={**common,"status":"success","cost_status":"recorded_actual_usage","result":data,
                     "scoring_isolated_outside_generation_runner":True}
            except Exception as error:
                usage=_validated_usage(error.usage if isinstance(error,GenerationFailure) else None)
                row={**common,"status":"failure","error_type":type(error).__name__,
                    "error":"Member execution failed; provider response and exception details intentionally not logged",
                    "reason_code":error.reason_code if isinstance(error,GenerationFailure) else "unclassified_failure",
                    "actual_usage":usage,"cost_status":"recorded_actual_usage" if usage is not None else "unknown_must_reconcile",
                    "failure_wall_time_s":error.wall_time_s if isinstance(error,GenerationFailure) else None}
            return self._append_unlocked(row),False
