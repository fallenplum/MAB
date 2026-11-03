"""Pinned, frozen real-model execution with verifiable identity and honest usage."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import os
import re
import time
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    backend: str = "transformers"
    model_path: str | None = None
    device: str = "cuda"
    dtype: str = "float16"
    max_input_tokens: int = 6144
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 300.0
    quantization: str | None = None
    enable_thinking: bool | None = None

    def validate(self):
        if not self.model_id or not self.revision or self.revision.lower() in {"main", "latest", "pending"}:
            raise ValueError("A fixed model identifier and exact revision are required")
        if self.backend not in {"transformers", "openai_compatible"}:
            raise ValueError("Unsupported execution backend")
        if self.backend == "transformers" and not re.fullmatch(r"[0-9a-fA-F]{40}", self.revision):
            raise ValueError("Transformers execution requires a full pinned repository commit")
        if any(isinstance(n,bool) or not isinstance(n,int) or n<=0 for n in (self.max_input_tokens,self.max_new_tokens)):
            raise ValueError("Token limits must be positive integers")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported locked compute dtype")
        if self.quantization not in {None, "nf4"}:
            raise ValueError("Only locked NF4 or unquantized execution is supported")
        if self.quantization == "nf4" and not re.fullmatch(r"cuda(?::\d+)?",self.device):
            raise ValueError("The NF4 protocol requires an explicit CUDA device")
        if not math.isfinite(self.temperature) or self.temperature<0 or not 0<self.top_p<=1:
            raise ValueError("Invalid generation settings")
        if not math.isfinite(self.timeout_s) or self.timeout_s<=0:
            raise ValueError("Invalid timeout")
        if self.api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",self.api_key_env):
            raise ValueError("Configure a key variable name, never a key value")
        if self.enable_thinking is not None and not isinstance(self.enable_thinking,bool):
            raise ValueError("enable_thinking must be an explicit bool or None")
        if self.backend == "openai_compatible" and self.enable_thinking is not None:
            raise ValueError("Thinking control requires pinned local chat-template execution")
        if self.backend == "openai_compatible":
            url=urlsplit(self.base_url or "")
            if url.scheme not in {"http","https"} or not url.netloc or url.username or url.password or url.query or url.fragment:
                raise ValueError("Explicit service URL without embedded credentials/query required")


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    wall_time_s: float
    model_id: str
    model_revision: str
    seed: int
    backend: str
    peak_cuda_bytes: int | None = None
    raw_usage: dict[str, Any] | None = None
    model_identity_verified: bool = False
    completion_reached_cap: bool | None = None
    finish_reason: str | None = None


class GenerationFailure(RuntimeError):
    """Safe error code plus known usage; never attach provider bodies or credentials."""
    def __init__(self, reason_code, *, usage=None, wall_time_s=None, planned_prompt_tokens=None):
        if not re.fullmatch(r"[a-z_]+",reason_code):
            reason_code="generation_failed"
        super().__init__(reason_code)
        self.reason_code=reason_code
        self.usage=usage
        self.wall_time_s=wall_time_s
        self.planned_prompt_tokens=planned_prompt_tokens


class InputTooLong(GenerationFailure):
    pass


def _file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def validate_local_model_source(spec: ModelSpec, receipt_path=None):
    """Validate the completed downloader receipt and actual local bytes.

    Custom local_dir downloads do not inherit identity from their folder name.
    Receipts follow scripts/download_inference_models.py. No network is used.
    """
    spec.validate()
    if not spec.model_path:
        return None
    model_dir=Path(spec.model_path).resolve(strict=True)
    root=model_dir.parent.parent
    receipts=[Path(receipt_path)] if receipt_path else sorted((root/"artifacts/audit").glob("*_model_download.json"))
    receipt=None
    for candidate in receipts:
        data=json.loads(candidate.read_text(encoding="utf-8-sig"))
        if data.get("repo_id")==spec.model_id and data.get("revision")==spec.revision:
            if Path(data.get("path","")).resolve()!=model_dir:
                continue
            receipt=data
            break
    if receipt is None or not receipt.get("completed_at_unix") or not receipt.get("files"):
        raise ValueError("Completed local model source receipt matching repository/revision/path required")
    verified=set()
    for record in receipt["files"]:
        relative=Path(record["path"])
        if relative.is_absolute():
            raise ValueError("Receipt inventory paths must be project-relative")
        target=(root/relative).resolve(strict=True)
        if not target.is_relative_to(model_dir):
            raise ValueError("Model receipt inventory escaped its locked directory")
        if target.stat().st_size!=record["bytes"] or _file_hash(target)!=record["sha256"]:
            raise ValueError("Local model bytes differ from completed download receipt")
        if record.get("lfs_sha256") and record["lfs_sha256"]!=record["sha256"]:
            raise ValueError("Recorded model weight identity disagrees with LFS source")
        verified.add(target.relative_to(model_dir).as_posix())
    if "config.json" not in verified or "tokenizer_config.json" not in verified:
        raise ValueError("Model/tokenizer configuration missing from verified source inventory")
    if not ({"tokenizer.json","vocab.txt"}&verified or {"vocab.json","merges.txt"}<=verified):
        raise ValueError("Verified tokenizer vocabulary is incomplete")
    if "model.safetensors.index.json" in verified:
        index=json.loads((model_dir/"model.safetensors.index.json").read_text("utf-8"))
        required=set(index.get("weight_map",{}).values())
        if not required or not required<=verified:
            raise ValueError("Pinned safetensor shard inventory is incomplete")
    elif "model.safetensors" not in verified:
        raise ValueError("Verified complete safetensors model/index is required")
    return receipt


def _validated_usage(usage):
    if not isinstance(usage,dict):
        return None
    keys=("prompt_tokens","completion_tokens")
    if any(isinstance(usage.get(k),bool) or not isinstance(usage.get(k),int) or usage[k]<0 for k in keys):
        return None
    result={k:usage[k] for k in keys}
    if "total_tokens" in usage:
        total=usage["total_tokens"]
        if isinstance(total,bool) or not isinstance(total,int) or total!=sum(result.values()):
            return None
        result["total_tokens"]=total
    # Never store arbitrary service-supplied fields under the name raw_usage.
    return result


def chat_template_text(tokenizer,messages,spec):
    options={"tokenize":False,"add_generation_prompt":True}
    if spec.enable_thinking is not None:options["enable_thinking"]=spec.enable_thinking
    return tokenizer.apply_chat_template(messages,**options)


class TransformersBackend:
    def __init__(self, spec: ModelSpec):
        spec.validate()
        self.source_receipt=validate_local_model_source(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.spec=spec
        self.torch=torch
        location=spec.model_path or spec.model_id
        kwargs={"local_files_only":bool(spec.model_path),"trust_remote_code":False}
        if not spec.model_path:kwargs["revision"]=spec.revision
        self.tokenizer=AutoTokenizer.from_pretrained(location,**kwargs)
        dtype=getattr(torch,spec.dtype)
        loading=dict(torch_dtype=dtype,use_safetensors=True,attn_implementation="sdpa",
                     low_cpu_mem_usage=True,device_map={"":spec.device})
        if spec.quantization=="nf4":
            from transformers import BitsAndBytesConfig
            loading["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
        self.model=AutoModelForCausalLM.from_pretrained(location,**loading,**kwargs)
        # For a hub snapshot the resolved config commit must match the pinned one.
        if not spec.model_path and getattr(self.model.config,"_commit_hash",None)!=spec.revision:
            raise ValueError("Resolved model commit differs from pinned revision")
        if spec.quantization=="nf4" and not getattr(self.model,"is_loaded_in_4bit",False):
            raise ValueError("Requested NF4 model was not actually loaded in 4bit")
        wanted=torch.device(spec.device)
        wanted_index=wanted.index if wanted.index is not None else (torch.cuda.current_device() if wanted.type=="cuda" else None)
        for device in getattr(self.model,"hf_device_map",{}).values():
            observed=torch.device(f"cuda:{device}" if isinstance(device,int) else device)
            index=observed.index if observed.index is not None else wanted_index
            if observed.type!=wanted.type or index!=wanted_index:
                raise ValueError("Unexpected model device/offload differs from locked execution")
        self.model.eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        if self.model.training or any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("Model must be frozen and in evaluation mode")
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token_id=self.tokenizer.eos_token_id

    def generate(self,messages:list[dict[str,str]],seed:int)->GenerationResult:
        torch=self.torch
        text=chat_template_text(self.tokenizer,messages,self.spec)
        inputs=self.tokenizer(text,return_tensors="pt",add_special_tokens=False)
        n_input=int(inputs.input_ids.shape[-1])
        if n_input>self.spec.max_input_tokens:
            raise InputTooLong("input_limit_before_model_call",usage={"prompt_tokens":0,"completion_tokens":0},planned_prompt_tokens=n_input)
        inputs={key:value.to(self.spec.device) for key,value in inputs.items()}
        torch.manual_seed(int(seed))
        cuda=self.spec.device.startswith("cuda")
        if cuda:
            torch.cuda.manual_seed_all(int(seed))
            torch.cuda.reset_peak_memory_stats(self.spec.device)
            torch.cuda.synchronize(self.spec.device)
        kwargs=dict(max_new_tokens=self.spec.max_new_tokens,do_sample=self.spec.temperature>0,
                    pad_token_id=self.tokenizer.pad_token_id,use_cache=True)
        if self.spec.temperature>0:kwargs.update(temperature=self.spec.temperature,top_p=self.spec.top_p)
        start=time.perf_counter()
        try:
            with torch.inference_mode():output=self.model.generate(**inputs,**kwargs)
            if cuda:torch.cuda.synchronize(self.spec.device)
        except Exception as error:
            raise GenerationFailure("local_generation_failed_unknown_usage",wall_time_s=time.perf_counter()-start,
                                    planned_prompt_tokens=n_input) from None
        elapsed=time.perf_counter()-start
        generated=output[0,n_input:]
        return GenerationResult(text=self.tokenizer.decode(generated,skip_special_tokens=True),prompt_tokens=n_input,
            completion_tokens=int(generated.numel()),wall_time_s=elapsed,model_id=self.spec.model_id,
            model_revision=self.spec.revision,seed=int(seed),backend=self.spec.backend,
            peak_cuda_bytes=int(torch.cuda.max_memory_allocated(self.spec.device)) if cuda else None,
            model_identity_verified=True,completion_reached_cap=int(generated.numel())>=self.spec.max_new_tokens,
            finish_reason="length_cap" if int(generated.numel())>=self.spec.max_new_tokens else "model_stopped")


class OpenAICompatibleBackend:
    """Explicit endpoint only. Provider must attest both model ID and revision."""
    def __init__(self,spec:ModelSpec):
        spec.validate()
        self.spec=spec

    def generate(self,messages:list[dict[str,str]],seed:int)->GenerationResult:
        import requests
        headers={"Content-Type":"application/json"}
        if self.spec.api_key_env:
            key=os.environ.get(self.spec.api_key_env)
            if not key:raise GenerationFailure("key_environment_unset_before_request",usage={"prompt_tokens":0,"completion_tokens":0})
            headers["Authorization"]="Bearer "+key
        body=dict(model=self.spec.model_id,messages=messages,max_tokens=self.spec.max_new_tokens,
                  temperature=self.spec.temperature,top_p=self.spec.top_p,seed=int(seed))
        start=time.perf_counter()
        try:
            response=requests.post(self.spec.base_url.rstrip("/")+"/chat/completions",json=body,
                                   headers=headers,timeout=self.spec.timeout_s)
            response.raise_for_status()
            payload=response.json()
        except Exception:
            raise GenerationFailure("service_request_failed_unknown_usage",wall_time_s=time.perf_counter()-start) from None
        elapsed=time.perf_counter()-start
        usage=_validated_usage(payload.get("usage"))
        if usage is None:
            raise GenerationFailure("actual_token_usage_missing_or_invalid",wall_time_s=elapsed)
        revision=payload.get("model_revision",payload.get("revision"))
        if payload.get("model")!=self.spec.model_id or revision!=self.spec.revision:
            raise GenerationFailure("served_model_identity_unverified",usage=usage,wall_time_s=elapsed)
        if usage["prompt_tokens"]>self.spec.max_input_tokens:
            raise InputTooLong("service_input_limit_exceeded",usage=usage,wall_time_s=elapsed)
        if usage["completion_tokens"]>self.spec.max_new_tokens:
            raise GenerationFailure("service_output_limit_exceeded",usage=usage,wall_time_s=elapsed)
        try:text=payload["choices"][0]["message"]["content"]
        except (KeyError,IndexError,TypeError):
            raise GenerationFailure("invalid_service_output",usage=usage,wall_time_s=elapsed) from None
        if not isinstance(text,str):
            raise GenerationFailure("invalid_service_output",usage=usage,wall_time_s=elapsed)
        return GenerationResult(text=text,prompt_tokens=usage["prompt_tokens"],completion_tokens=usage["completion_tokens"],
            wall_time_s=elapsed,model_id=payload["model"],model_revision=revision,seed=int(seed),backend=self.spec.backend,
            raw_usage=usage,model_identity_verified=True,
            completion_reached_cap=usage["completion_tokens"]>=self.spec.max_new_tokens,
            finish_reason=payload["choices"][0].get("finish_reason") if payload["choices"][0].get("finish_reason") in {"stop","length","content_filter","tool_calls",None} else "other")


def make_backend(spec:ModelSpec):
    spec.validate()
    return TransformersBackend(spec) if spec.backend=="transformers" else OpenAICompatibleBackend(spec)


def configuration_fingerprint(spec:ModelSpec)->str:
    spec.validate()
    data=asdict(spec)
    # Preserve pre-existing aggregator fingerprints when no thinking option is set.
    if data.get("enable_thinking") is None:data.pop("enable_thinking",None)
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
