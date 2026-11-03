"""Pure, source-bound proof of a settled generation prefix.

No process, file, CUDA, or ledger access occurs here. A trusted executor must
capture raw py-spy JSON while its own verified freeze lease is continuously
active. The ticket callbacks are trusted executor code, never JSON trust flags.
They must bind current process identity, capture bytes and generation profile.

Stock py-spy cannot read torch.Tensor native shape. We derive its shape from
pinned source invariants and a next-forward caller/callee stack, and explicitly
label that fact. This does not certify OS suspension, GPU retirement, or wall
accounting; callers must independently prove those before an actual abort.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import PureWindowsPath
import re
from typing import Callable, Mapping


class SnapshotProofError(ValueError):
    pass


_TOKEN = object()
_KEYS = ("inference", "generation", "forward", "member")


def _fail(reason):
    raise SnapshotProofError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail("invalid integer: " + name)
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        _fail("invalid positive finite value: " + name)
    return float(value)


def _path(value):
    if not isinstance(value, str) or not value or ".." in PureWindowsPath(value).parts:
        _fail("invalid source path")
    return str(PureWindowsPath(value)).casefold()


def _same_ast(node, text):
    other = ast.parse(text).body[0]
    return ast.dump(node, include_attributes=False) == ast.dump(other, include_attributes=False)


def _method(tree, class_name, name):
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    if len(classes) != 1:
        _fail("source class missing or ambiguous: " + class_name)
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(methods) != 1:
        _fail("source method missing or ambiguous: " + name)
    return methods[0]


def _writes(function, name):
    return [n for n in ast.walk(function) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)]


def audit_source_invariants(source_texts, source_paths, expected_source_sha256):
    """Check exact hashes and the source invariant used to derive shapes."""
    if any(set(value) != set(_KEYS) for value in (source_texts, source_paths, expected_source_sha256)):
        _fail("the four pinned source bindings are required")
    trees = {}
    hashes = {}
    for key in _KEYS:
        value = source_texts[key]
        if isinstance(value, bytes):
            raw = value
            text = raw.decode("utf-8-sig")
        elif isinstance(value, str):
            raw = value.encode("utf8")
            text = value.lstrip("\ufeff")
        else:
            _fail("source must be exact bytes or a lossless string")
        digest = _sha(raw)
        if not re.fullmatch("[0-9a-f]{64}", expected_source_sha256[key]) or digest != expected_source_sha256[key]:
            _fail("source hash mismatch: " + key)
        _path(source_paths[key])
        hashes[key] = digest
        try:
            trees[key] = ast.parse(text)
        except SyntaxError:
            _fail("pinned source is not valid Python: " + key)
    sample = _method(trees["generation"], "GenerationMixin", "_sample")
    helper = _method(trees["generation"], "GenerationMixin", "_has_unfinished_sequences")
    loops = [n for n in ast.walk(sample) if isinstance(n, ast.While)]
    if len(loops) != 1:
        _fail("exactly one sample while loop is required")
    loop = loops[0]
    if not isinstance(loop.test, ast.Call) or not isinstance(loop.test.func, ast.Attribute) or loop.test.func.attr != "_has_unfinished_sequences":
        _fail("sample while condition changed")
    initial = [n for n in sample.body if _same_ast(n, "batch_size, cur_len = input_ids.shape")]
    cats = [n for n in loop.body if _same_ast(n, "input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)")]
    increments = [n for n in loop.body if _same_ast(n, "cur_len += 1")]
    flags = [n for n in loop.body if _same_ast(n, "this_peer_finished = unfinished_sequences.max() == 0")]
    if not (len(initial) == len(cats) == len(increments) == len(flags) == 1):
        _fail("sample shape invariant changed")
    if len(_writes(sample, "input_ids")) != 1 or len(_writes(sample, "cur_len")) != 2 or len(_writes(sample, "batch_size")) != 1:
        _fail("unexpected input_ids/cur_len/batch_size write")
    if not (initial[0].lineno < loop.lineno < cats[0].lineno < flags[0].lineno < increments[0].lineno):
        _fail("sample update order changed")
    # Direct tensor resizing is incompatible even if an assignment counter fits.
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and isinstance(n.func.value, ast.Name) and n.func.value.id == "input_ids"
           for n in ast.walk(sample)):
        _fail("unexpected input_ids method call")
    calls = []
    for statement in loop.body:
        if isinstance(statement, ast.If) and isinstance(statement.test, ast.Name) and statement.test.id == "is_prefill":
            calls += [n for n in statement.orelse if _same_ast(n, "outputs = model_forward(**model_inputs, return_dict=True)")]
    if len(calls) != 1 or calls[0].lineno >= cats[0].lineno:
        _fail("next-forward call is not unique")
    helper_body = [n for n in helper.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    if (len(helper_body) != 2 or not isinstance(helper_body[0], ast.If)
        or not isinstance(helper_body[0].test, ast.Name) or helper_body[0].test.id != "synced_gpus"
        or len(helper_body[0].orelse) != 1
        or not _same_ast(helper_body[0].orelse[0], "if this_peer_finished:\n    return False")
        or not _same_ast(helper_body[1], "return True")):
        _fail("unfinished-sequence bool synchronization branch changed")
    backend = _method(trees["inference"], "TransformersBackend", "generate")
    expected_input = [n for n in backend.body if _same_ast(n, "n_input = int(inputs.input_ids.shape[-1])")]
    transfers = [n for n in backend.body if _same_ast(n, "inputs = {key: value.to(self.spec.device) for key, value in inputs.items()}")]
    if len(expected_input) != 1 or len(transfers) != 1 or len(_writes(backend, "n_input")) != 1 or len(_writes(backend, "inputs")) != 2:
        _fail("backend original-input length/transfer invariant changed")
    if not expected_input[0].lineno < transfers[0].lineno:
        _fail("backend input order changed")
    forward = _method(trees["forward"], "Qwen3ForCausalLM", "forward")
    store = _method(trees["member"], "MemberStore", "run")
    return {"hashes": hashes, "paths": {k: _path(source_paths[k]) for k in _KEYS},
            "sample_call_line": calls[0].lineno, "sample_range": (sample.lineno, sample.end_lineno),
            "backend_range": (backend.lineno, backend.end_lineno),
            "forward_range": (forward.lineno, forward.end_lineno),
            "store_range": (store.lineno, store.end_lineno),
            "shape_method": "derived_from_pinned_source_invariant_not_read_from_native_tensor"}


_PROFILE_FALSE = ("token_healing", "inputs_embeds", "assisted_decoding", "custom_logits_processors",
                  "custom_stopping_criteria", "custom_generation_hooks", "compiled_generation")


def _profile(callback, source_binding):
    if not callable(callback):
        _fail("profile validator must be trusted executor code, not a JSON declaration")
    value = callback()
    if not isinstance(value, dict):
        _fail("profile validator did not return a mapping")
    if (value.get("source_binding_sha256") != source_binding or value.get("decoder_only") is not True
        or value.get("num_beams") != 1 or isinstance(value.get("num_beams"), bool)
        or value.get("num_return_sequences") != 1 or isinstance(value.get("num_return_sequences"), bool)
        or value.get("prefill_chunk_size") is not None
        or any(value.get(key) is not False for key in _PROFILE_FALSE)):
        _fail("generation profile is not the source-verified single-sequence stock path")
    return dict(value)


class _FrozenCaptureTicket:
    __slots__ = ("_token", "raw", "raw_sha", "pid", "ctime", "nonce", "lease", "profile", "profile_check", "source", "software")
    def __init__(self, token, **kwargs):
        if token is not _TOKEN:
            _fail("capture tickets are issued by the trusted executor factory")
        self._token = token
        for key, value in kwargs.items():
            setattr(self, key, value)


def _lease(ticket):
    value = ticket.lease()
    if not isinstance(value, dict) or value.get("status") != "verified_closed_thread_set_all_user_suspended" or value.get("active") is not True:
        _fail("freeze lease is not active and verified")
    if (value.get("pid") != ticket.pid or value.get("create_time_ns") != ticket.ctime
        or value.get("nonce") != ticket.nonce or value.get("raw_dump_sha256") != ticket.raw_sha):
        _fail("freeze lease process/capture identity changed")
    tids = value.get("thread_ids")
    if not isinstance(tids, (list, tuple)) or not tids or len(set(tids)) != len(tids):
        _fail("invalid frozen native thread set")
    for tid in tids:
        _int(tid, "frozen thread", 1)
    return frozenset(tids)


def issue_frozen_capture_ticket(raw_dump: bytes, *, pid: int, create_time_ns: int, lease_nonce: str,
        lease_validator: Callable, source_texts: Mapping, source_paths: Mapping,
        expected_source_sha256: Mapping, profile_validator: Callable, software_test_only=False):
    """Issue an opaque ticket; executor callbacks must perform live verification.

    A callable accepting arbitrary JSON trust flags is NOT a valid executor.
    profile_validator must independently verify qualified source/config lineage.
    lease_validator must bind bytes captured in this exact still-active lease.
    This factory deliberately performs no target reads or process control.
    """
    if not isinstance(raw_dump, bytes) or not raw_dump or not callable(lease_validator):
        _fail("raw bytes and a trusted live lease validator are required")
    _int(pid, "pid", 1); _int(create_time_ns, "create_time_ns", 1)
    if not isinstance(lease_nonce, str) or not re.fullmatch("[a-zA-Z0-9_-]{16,128}", lease_nonce):
        _fail("a unique lease nonce is required")
    if type(software_test_only) is not bool:
        _fail("software_test_only must be explicit bool")
    sources = audit_source_invariants(source_texts, source_paths, expected_source_sha256)
    source_binding = _digest(sources["hashes"])
    profile = _profile(profile_validator, source_binding)
    ticket = _FrozenCaptureTicket(_TOKEN, raw=raw_dump, raw_sha=_sha(raw_dump), pid=pid, ctime=create_time_ns,
        nonce=lease_nonce, lease=lease_validator, profile=profile, profile_check=profile_validator,
        source=sources, software=software_test_only)
    _lease(ticket)
    return ticket


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            _fail("duplicate JSON field")
        result[key] = value
    return result


def _load(raw):
    try:
        value = json.loads(raw.decode("utf8"), object_pairs_hook=_pairs,
                           parse_constant=lambda v: _fail("nonfinite JSON number"))
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail("incomplete py-spy JSON: " + type(error).__name__)
    if not isinstance(value, list) or not value:
        _fail("py-spy JSON must contain thread traces")
    return value


def _locals(frame):
    values = frame.get("locals")
    if not isinstance(values, list):
        _fail("required frame locals are missing")
    result = {}
    for local in values:
        if not isinstance(local, dict) or not isinstance(local.get("name"), str) or local["name"] in result:
            _fail("invalid or duplicated frame local")
        result[local["name"]] = local.get("repr")
    return result


def _literal(values, name):
    value = values.get(name)
    if not isinstance(value, str) or any(marker in value.casefold() for marker in ("...", "truncated", "<unavailable", "?")):
        _fail("missing/truncated required local: " + name)
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        _fail("required local is not a complete literal: " + name)


def _anchor(frames, name, path, bounds=None):
    result = []
    for index, frame in enumerate(frames):
        if frame.get("name") == name and _path(frame.get("filename")) == path:
            line = _int(frame.get("line"), "frame line", 1)
            if bounds is not None and not bounds[0] <= line <= bounds[1]:
                continue
            result.append((index, frame))
    if len(result) != 1:
        _fail("required call anchor is missing or ambiguous: " + name)
    return result[0]


@dataclass(frozen=True)
class PrefixProof:
    pid: int
    create_time_ns: int
    os_thread_id: int
    job_id: str
    model_fingerprint: str
    seed: int
    planned_prompt_tokens: int
    committed_completion_tokens: int
    backend_start_qpc: float
    raw_dump_sha256: str
    source_binding_sha256: str
    shape_method: str
    software_test_only: bool
    original_attempt_id: str | None

    @property
    def prompt_tokens(self):
        return self.planned_prompt_tokens

    @property
    def completion_tokens(self):
        return self.committed_completion_tokens

    @property
    def evidence(self):
        return {"raw_dump_sha256": self.raw_dump_sha256,
                "source_binding_sha256": self.source_binding_sha256,
                "shape_method": self.shape_method,
                "original_attempt_id": self.original_attempt_id,
                "freeze_scope": "verified_closed_thread_set_all_user_suspended",
                "pid": self.pid, "create_time_ns": self.create_time_ns,
                "producer_os_thread_id": self.os_thread_id,
                "software_test_only": self.software_test_only,
                "does_not_prove_gpu_retirement_or_wall": True}


def prove_frozen_sample(ticket, *, expected_job_id, expected_model_fingerprint, expected_seed=42,
                        max_new_tokens, expected_input_tokens=None):
    """Return only a source/lease-bound prefix proof, never a failure ledger row."""
    if type(ticket) is not _FrozenCaptureTicket or ticket._token is not _TOKEN:
        _fail("an opaque capture ticket is required; JSON trust flags are not proof")
    native_tids = _lease(ticket)
    if _sha(ticket.raw) != ticket.raw_sha:
        _fail("capture bytes changed after ticket issuance")
    source = ticket.source
    source_binding = _digest(source["hashes"])
    if _profile(ticket.profile_check, source_binding) != ticket.profile:
        _fail("qualified generation profile changed")
    _int(expected_seed, "expected seed"); _int(max_new_tokens, "max_new_tokens", 1)
    if any(not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v)
           for v in (expected_job_id, expected_model_fingerprint)):
        _fail("exact job/config hashes are required")
    traces = _load(ticket.raw)
    candidates = []
    seen_tids = set()
    for trace in traces:
        if not isinstance(trace, dict) or trace.get("pid") != ticket.pid:
            _fail("trace process differs from freeze lease")
        tid = _int(trace.get("os_thread_id"), "trace OS thread", 1)
        if tid in seen_tids or tid not in native_tids:
            _fail("trace thread is duplicated or outside frozen thread set")
        seen_tids.add(tid)
        frames = trace.get("frames")
        if not isinstance(frames, list) or any(not isinstance(f, dict) for f in frames):
            _fail("invalid frame list")
        sample_frames = [f for f in frames if f.get("name") == "_sample"]
        if sample_frames:
            if len(sample_frames) != 1:
                _fail("multiple/reentrant sample producers")
            candidates.append((tid, frames))
    if len(candidates) != 1:
        _fail("exactly one sample producer is required")
    tid, frames = candidates[0]
    fi, forward = _anchor(frames, "forward", source["paths"]["forward"], source["forward_range"])
    si, sample = _anchor(frames, "_sample", source["paths"]["generation"], source["sample_range"])
    gi, generate = _anchor(frames, "generate", source["paths"]["inference"], source["backend_range"])
    mi, store = _anchor(frames, "run", source["paths"]["member"], source["store_range"])
    if not fi < si < gi < mi or sample["line"] != source["sample_call_line"]:
        _fail("not inside the next model_forward call; transition frames are forbidden")
    sl, gl, ml = _locals(sample), _locals(generate), _locals(store)
    batch = _int(_literal(sl, "batch_size"), "batch_size", 1)
    cur_len = _int(_literal(sl, "cur_len"), "cur_len", 1)
    n_input = _int(_literal(gl, "n_input"), "n_input", 1)
    if batch != 1 or _literal(sl, "is_prefill") is not False or _literal(sl, "synced_gpus") is not False or _literal(sl, "streamer") is not None:
        _fail("sample is not stock batch1/nonprefill/unsynced/no-streamer")
    if expected_input_tokens is not None and n_input != _int(expected_input_tokens, "expected input", 1):
        _fail("input length differs from independent tokenizer provenance")
    count = cur_len - n_input
    if not 1 <= count < max_new_tokens:
        _fail("prefix is not an unfinished post-prefill generation within cap")
    common = _literal(ml, "common")
    if not isinstance(common, dict):
        _fail("MemberStore common identity is missing")
    if (common.get("job_id") != expected_job_id or common.get("model_fingerprint") != expected_model_fingerprint
        or common.get("seed") != expected_seed or isinstance(common.get("seed"), bool)
        or _literal(gl, "seed") != expected_seed or isinstance(_literal(gl, "seed"), bool)
        or common.get("sample_index") != 0 or isinstance(common.get("sample_index"), bool)
        or common.get("split") not in {"e1_selection_calibration", "e1_diagnostic_evaluation"}):
        _fail("MemberStore/backend/job/seed identity mismatch")
    start = _finite(_literal(gl, "start"), "backend start QPC")
    attempt = common.get("attempt_id")
    if attempt is not None and (not isinstance(attempt, str) or not re.fullmatch("[0-9a-f]{32}", attempt)):
        _fail("original attempt UUID is malformed")
    if _lease(ticket) != native_tids:
        _fail("freeze thread set changed during proof")
    return PrefixProof(ticket.pid, ticket.ctime, tid, expected_job_id, expected_model_fingerprint,
        expected_seed, n_input, count, start, ticket.raw_sha, source_binding, source["shape_method"], ticket.software, attempt)


def inspect_pyspy_dump(raw_dump):
    """Describe raw capabilities without authorizing a prefix or any process action."""
    traces = _load(raw_dump)
    native_shapes = False
    for trace in traces:
        for frame in trace.get("frames", []):
            for local in frame.get("locals") or []:
                if local.get("name") == "input_ids" and isinstance(local.get("repr"), str):
                    native_shapes = native_shapes or "shape=" in local["repr"]
    return {"threads": len(traces), "ordinary_dump_proves_atomic_freeze": False,
            "ordinary_dump_reads_native_tensor_shape": False,
            "shape_like_text_seen_but_not_trusted": native_shapes,
            "committed_prefix_authorized": False,
            "required_method": "trusted_freeze_ticket_plus_pinned_source_invariant_and_next_forward_chain"}
