"""Approval-gated, single-question reference projection for future E18 scoring.

Only synthetic callers are exercised in this preparation round. Mixed source
files require explicit approval for raw-byte hashing/structural scanning. Values
outside the requested approved record are never JSON-decoded. Unsupported or
ambiguous physical formats fail closed. No source is opened at module import.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import threading

from e18_execution_adapter import canonical_digest, valid_sha, validate_scope_approval


FORMATS=frozenset({"normalized_qid_jsonl_v1","gsm8k_official_jsonl_v1","bbh_official_examples_v1"})
HEADER=re.compile(rb'^\s*\{\s*"question_id"\s*:\s*"([^"\\]+)"\s*[,}]')
SOURCE_FIELDS=frozenset({"source_id","format","path","sha256","task","source_split","subtask"})
RECORD_FIELDS=frozenset({"question_id","canonical_question_sha256","task","subtask","source_split","source_id","row_index"})
DENY_FIELDS=frozenset({"question_id","canonical_question_sha256"})
REFERENCE_BINDINGS=frozenset({"reference_sources_sha256","reference_records_sha256","sealed_exclusions_sha256","reference_version_sha256"})


def _canonical_input(question):
    import unicodedata
    if not isinstance(question,str) or not question.strip():
        raise ValueError("A nonempty original input is required")
    return hashlib.sha256(" ".join(unicodedata.normalize("NFKC",question).split()).encode()).hexdigest()


def build_reference_scope(records,sources,bindings,reference_version,sealed_exclusions,*,allow_structural_byte_scan=False):
    """Build input/source metadata only, without opening a reference source."""
    if not isinstance(reference_version,str) or not reference_version or type(allow_structural_byte_scan) is not bool:
        raise ValueError("Explicit reference version and byte-scan permission are required")
    normalized_sources=[]
    for item in sources:
        if set(item)!=SOURCE_FIELDS or item["format"] not in FORMATS or not valid_sha(item["sha256"]):
            raise ValueError("Unsupported or incompletely bound reference source")
        if not isinstance(item["source_id"],str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}",item["source_id"]):
            raise ValueError("A unique explicit reference source identifier is required")
        if item["task"] not in {"gsm8k","bbh"} or not isinstance(item["source_split"],str):
            raise ValueError("Reference source task/split metadata are required")
        if item["source_split"]!=("train" if item["task"]=="gsm8k" else "test"):
            raise ValueError("E18 reference source must retain GSM train and BBH test roles")
        path=Path(item["path"])
        if not path.is_absolute():
            raise ValueError("Reference sources require an absolute approved path")
        if any(part.casefold() in {"sealed","sealed_test","sealed200"} for part in path.parts):
            raise PermissionError("Sealed source directories cannot supply E18 references")
        if item["format"]=="gsm8k_official_jsonl_v1" and item["task"]!="gsm8k":
            raise ValueError("GSM source codec requires GSM task identity")
        if item["format"]=="bbh_official_examples_v1" and (item["task"]!="bbh" or not item["subtask"]):
            raise ValueError("Official BBH source requires its pinned subtask")
        normalized_sources.append({**item,"path":str(path)})
    source_map={item["source_id"]:item for item in normalized_sources}
    if not source_map or len(source_map)!=len(normalized_sources):
        raise ValueError("Reference source IDs must be unique")
    deny=[]
    for item in sealed_exclusions:
        if set(item)!=DENY_FIELDS or not isinstance(item["question_id"],str) or not item["question_id"] or not valid_sha(item["canonical_question_sha256"]):
            raise ValueError("Explicit sealed identity/canonical exclusions are required")
        deny.append(dict(item))
    denied_ids={item["question_id"] for item in deny}
    denied_hashes={item["canonical_question_sha256"] for item in deny}
    if not deny or len(denied_ids)!=len(deny) or len(denied_hashes)!=len(deny):
        raise ValueError("Nonempty unique sealed identity exclusions are mandatory")
    normalized_records=[]
    for item in records:
        if set(item)!=RECORD_FIELDS or not isinstance(item["question_id"],str) or not item["question_id"] or not valid_sha(item["canonical_question_sha256"]):
            raise ValueError("Reference records must be identity/locator metadata only")
        if type(item["row_index"]) is not int or item["row_index"]<0 or item["source_id"] not in source_map:
            raise ValueError("A reliable nonnegative approved source-row locator is required")
        if item["question_id"] in denied_ids or item["canonical_question_sha256"] in denied_hashes:
            raise PermissionError("Sealed identities cannot enter the reference scope")
        source=source_map[item["source_id"]]
        if item["task"]!=source["task"] or item["source_split"]!=source["source_split"]:
            raise ValueError("Reference record/source task or split differs")
        subtask=item["subtask"] or None
        if item["task"]=="gsm8k" and subtask is not None:
            raise ValueError("GSM reference metadata cannot acquire a BBH subtask")
        if item["task"]=="bbh" and (not isinstance(subtask,str) or not subtask):
            raise ValueError("BBH requires a pinned subtask evaluator")
        if source["subtask"] is not None and subtask!=source["subtask"]:
            raise ValueError("Reference source subtask differs")
        normalized_records.append({**item,"subtask":subtask})
    ids=[item["question_id"] for item in normalized_records]
    canonical=[item["canonical_question_sha256"] for item in normalized_records]
    positions=[(item["source_id"],item["row_index"]) for item in normalized_records]
    if not ids or len(set(ids))!=len(ids) or len(set(canonical))!=len(canonical) or len(set(positions))!=len(positions):
        raise ValueError("Duplicate reference qid, canonical identity or source row")
    target_digest=canonical_digest([[qid,value] for qid,value in zip(ids,canonical)])
    if bindings.get("target_ids_sha256")!=target_digest:
        raise ValueError("Reference target identities differ from the frozen target binding")
    additions=dict(reference_sources_sha256=canonical_digest(normalized_sources),
        reference_records_sha256=canonical_digest(normalized_records),sealed_exclusions_sha256=canonical_digest(deny),
        reference_version_sha256=canonical_digest(reference_version))
    if any(key in bindings and bindings[key]!=value for key,value in additions.items()):
        raise ValueError("Reference source/identity/version binding differs")
    return dict(schema="e18_scope_contract_v2",role="reference",cohort="e18_confirm",
        bindings={**bindings,**additions},reference_version=reference_version,
        question_keys=[dict(question_id=qid,canonical_question_sha256=value) for qid,value in zip(ids,canonical)],
        reference_records=normalized_records,reference_sources=normalized_sources,sealed_exclusions=deny,
        access_policy=dict(allow_structural_byte_scan=allow_structural_byte_scan,
            unapproved_reference_value_decoding=False,source_sha_before_reference_decode=True,
            cache_unit="canonical_question_sha256/reference_version",batch_reference_preload=False),
        authorization_state="PREPARATION_ONLY_UNTIL_EXPLICIT_READ_REFERENCE_APPROVAL")


class _Cursor:
    """Byte-level JSON structure traversal without decoding skipped values."""
    def __init__(self,data):
        self.data=data
        self.position=0

    def whitespace(self):
        while self.position<len(self.data) and self.data[self.position] in b" \t\r\n":
            self.position+=1

    def expect(self,value):
        self.whitespace()
        if self.data[self.position:self.position+1]!=value:
            raise ValueError("Unexpected reference source JSON structure")
        self.position+=1

    def string(self,decode=False):
        self.whitespace()
        start=self.position
        self.expect(b'"')
        escaped=False
        while self.position<len(self.data):
            value=self.data[self.position]
            self.position+=1
            if escaped:
                escaped=False
            elif value==92:
                escaped=True
            elif value==34:
                raw=self.data[start:self.position]
                return json.loads(raw) if decode else None
        raise ValueError("Unterminated reference source JSON string")

    def skip(self,depth=0):
        if depth>128:
            raise ValueError("Reference source nesting exceeds its parser limit")
        self.whitespace()
        value=self.data[self.position:self.position+1]
        if value==b'"':
            self.string()
            return
        if value in (b"{",b"["):
            closing=b"}" if value==b"{" else b"]"
            self.position+=1
            self.whitespace()
            if self.data[self.position:self.position+1]==closing:
                self.position+=1
                return
            while True:
                if value==b"{":
                    self.string();self.expect(b":")
                self.skip(depth+1)
                self.whitespace()
                next_value=self.data[self.position:self.position+1]
                if next_value==closing:
                    self.position+=1
                    return
                self.expect(b",")
        start=self.position
        while self.position<len(self.data) and self.data[self.position] not in b" \t\r\n,]}":
            self.position+=1
        if self.position==start:
            raise ValueError("Empty reference source JSON value")


def _project_object(raw,allowed,on_decoded=None):
    cursor=_Cursor(raw.removeprefix(b"\xef\xbb\xbf"))
    cursor.expect(b"{")
    result={};seen=set()
    while True:
        cursor.whitespace()
        if cursor.data[cursor.position:cursor.position+1]==b"}":
            cursor.position+=1
            break
        key=cursor.string(decode=True)
        if key in seen:
            raise ValueError("Duplicate key in approved reference record")
        seen.add(key)
        cursor.expect(b":");cursor.whitespace()
        start=cursor.position;cursor.skip();end=cursor.position
        if key in allowed:
            result[key]=json.loads(cursor.data[start:end])
            if on_decoded is not None:on_decoded(key)
        cursor.whitespace()
        if cursor.data[cursor.position:cursor.position+1]==b",":
            cursor.position+=1
        elif cursor.data[cursor.position:cursor.position+1]!=b"}":
            raise ValueError("Invalid approved reference record boundary")
    cursor.whitespace()
    if cursor.position!=len(cursor.data):
        raise ValueError("Trailing source data after approved reference record")
    return result


def _example_at(raw,wanted):
    cursor=_Cursor(raw.removeprefix(b"\xef\xbb\xbf"))
    cursor.expect(b"{")
    selected=None;seen=set();examples_count=0
    while True:
        cursor.whitespace()
        if cursor.data[cursor.position:cursor.position+1]==b"}":
            cursor.position+=1
            break
        key=cursor.string(decode=True)
        if key in seen:
            raise ValueError("Duplicate root key in official BBH source")
        seen.add(key);cursor.expect(b":")
        if key!="examples":
            cursor.skip()
        else:
            cursor.expect(b"[")
            while True:
                cursor.whitespace()
                if cursor.data[cursor.position:cursor.position+1]==b"]":
                    cursor.position+=1
                    break
                start=cursor.position
                if cursor.data[start:start+1]!=b"{":
                    raise ValueError("Official BBH examples must be objects")
                cursor.skip();end=cursor.position
                if examples_count==wanted:
                    selected=cursor.data[start:end]
                examples_count+=1;cursor.whitespace()
                if cursor.data[cursor.position:cursor.position+1]==b",":
                    cursor.position+=1
                elif cursor.data[cursor.position:cursor.position+1]!=b"]":
                    raise ValueError("Invalid official BBH examples boundary")
        cursor.whitespace()
        if cursor.data[cursor.position:cursor.position+1]==b",":
            cursor.position+=1
        elif cursor.data[cursor.position:cursor.position+1]!=b"}":
            raise ValueError("Invalid official BBH source boundary")
    cursor.whitespace()
    if cursor.position!=len(cursor.data) or "examples" not in seen or selected is None:
        raise ValueError("Approved official BBH source row is unavailable")
    return selected,examples_count


class E18ReferenceProvider:
    """Lazy hash-bound single-reference provider with per-question cache."""
    def __init__(self,scope_contract,approval_ref,*,software_test_only=False):
        self.scope=copy.deepcopy(scope_contract)
        self.approval_ref=copy.deepcopy(approval_ref)
        self.software_test_only=software_test_only
        rebuilt=build_reference_scope(self.scope["reference_records"],self.scope["reference_sources"],self.scope["bindings"],
            self.scope["reference_version"],self.scope["sealed_exclusions"],
            allow_structural_byte_scan=self.scope["access_policy"]["allow_structural_byte_scan"])
        if rebuilt!=self.scope:
            raise ValueError("Reference scope contains altered or unexpected metadata")
        validate_scope_approval(self.approval_ref,self.scope,"read_reference",software_test_only=software_test_only)
        self._records={row["question_id"]:row for row in self.scope["reference_records"]}
        self._sources={row["source_id"]:row for row in self.scope["reference_sources"]}
        self._cache={};self._failed_after_access=set();self._revealed=set();self._events=[]
        self._lock=threading.RLock()
        self._counts=dict(requests=0,reference_cache_hits=0,reference_values_decoded=0,
            reference_decode_attempts=0,
            source_scan_attempts=0,verified_source_scans=0,raw_source_bytes_streamed=0,
            physical_source_records_streamed=0,unapproved_encoded_record_values_skipped=0,
            denied_requests=0,failed_after_reference_access=0)

    def stats(self):
        with self._lock:
            return {**self._counts,"gold_reveal_accounts_consumed":len(self._revealed),
                    "distinct_references_cached":len(self._cache),"unapproved_reference_values_decoded":0,
                    "protected_input_values_decoded":0,"software_test_only":self.software_test_only}

    def audit_events(self):
        with self._lock:return copy.deepcopy(self._events)

    def _capture(self,record,source):
        if not self.scope["access_policy"]["allow_structural_byte_scan"]:
            raise PermissionError("Raw-byte source SHA/structure scanning has not been explicitly approved")
        self._counts["source_scan_attempts"]+=1
        checksum=hashlib.sha256();selected=None;physical_records=0
        path=Path(source["path"])
        if source["format"]=="bbh_official_examples_v1":
            payload=bytearray()
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(1048576),b""):
                    checksum.update(block);payload.extend(block);self._counts["raw_source_bytes_streamed"]+=len(block)
            if checksum.hexdigest()!=source["sha256"]:
                raise ValueError("Reference source SHA256 differs from the approved source")
            selected,physical_records=_example_at(bytes(payload),record["row_index"])
        else:
            seen=set()
            with path.open("rb") as stream:
                for line_index,raw in enumerate(stream):
                    checksum.update(raw);self._counts["raw_source_bytes_streamed"]+=len(raw)
                    if not raw.strip():continue
                    physical_records+=1
                    if source["format"]=="normalized_qid_jsonl_v1":
                        match=HEADER.match(raw.removeprefix(b"\xef\xbb\xbf"))
                        if match is None:
                            raise ValueError("Reliable qid-first metadata header is missing")
                        qid=match.group(1).decode("ascii")
                        if qid in seen:raise ValueError("Duplicate source qid before reference decoding")
                        seen.add(qid)
                        if qid==record["question_id"]:
                            if line_index!=record["row_index"]:
                                raise ValueError("Approved qid source-row locator differs")
                            selected=raw
                    elif line_index==record["row_index"]:
                        selected=raw
            if checksum.hexdigest()!=source["sha256"]:
                raise ValueError("Reference source SHA256 differs from the approved source")
            if selected is None:raise ValueError("Approved reference row is unavailable")
        self._counts["verified_source_scans"]+=1
        self._counts["physical_source_records_streamed"]+=physical_records
        self._counts["unapproved_encoded_record_values_skipped"]+=max(0,physical_records-1)
        return selected

    def get_reference(self,question_id,canonical_question_sha256,reference_version):
        with self._lock:
            self._counts["requests"]+=1
            key=(canonical_question_sha256,reference_version)
            began_access=False
            try:
                record=self._records.get(question_id)
                if record is None or record["canonical_question_sha256"]!=canonical_question_sha256 or reference_version!=self.scope["reference_version"]:
                    raise PermissionError("Unknown or out-of-scope question/canonical/reference version")
                validate_scope_approval(self.approval_ref,self.scope,"read_reference",software_test_only=self.software_test_only)
                if key in self._failed_after_access:
                    raise RuntimeError("Prior reference access failed; explicit reconciliation is required")
                if key in self._cache:
                    self._counts["reference_cache_hits"]+=1
                    return {**copy.deepcopy(self._cache[key]),"cache_hit":True,"reference_revealed_new":False}
                source=self._sources[record["source_id"]]
                raw=self._capture(record,source)
                if source["format"]=="normalized_qid_jsonl_v1":
                    input_meta=_project_object(raw,{"question_id","question","task","subtask","source_split"})
                    if input_meta.get("question_id")!=question_id or input_meta.get("task")!=record["task"] or (input_meta.get("subtask") or None)!=record["subtask"] or input_meta.get("source_split")!=record["source_split"]:
                        raise ValueError("Approved source row task/qid metadata differ")
                    question=input_meta.get("question")
                    reference_field="reference"
                else:
                    input_key="question" if source["format"]=="gsm8k_official_jsonl_v1" else "input"
                    question=_project_object(raw,{input_key}).get(input_key)
                    reference_field="answer" if source["format"]=="gsm8k_official_jsonl_v1" else "target"
                if _canonical_input(question)!=canonical_question_sha256:
                    raise ValueError("Approved source-row input canonical differs before reference decoding")
                # This is the first point at which the requested reference value
                # can be JSON-decoded. The unique account is charged even if the
                # source value is malformed and the attempt is then fenced.
                self._revealed.add(key);began_access=True
                self._counts["reference_decode_attempts"]+=1
                def decoded(field):
                    if field==reference_field:self._counts["reference_values_decoded"]+=1
                payload=_project_object(raw,{reference_field,"choices"},on_decoded=decoded)
                reference=payload.get(reference_field)
                if not isinstance(reference,str) or not reference.strip():
                    raise ValueError("The approved reference is not nonempty text")
                if source["format"]=="gsm8k_official_jsonl_v1":
                    if "####" not in reference:raise ValueError("Official GSM reference lacks its frozen final-answer delimiter")
                    reference=reference.rsplit("####",1)[1].strip()
                    if not reference:raise ValueError("Official GSM final reference is empty")
                source_identity=dict(source_id=source["source_id"],source_sha256=source["sha256"],
                    source_format=source["format"],source_row_index=record["row_index"],
                    selected_raw_record_sha256=hashlib.sha256(raw).hexdigest(),
                    question_id=question_id,canonical_question_sha256=canonical_question_sha256,reference_version=reference_version)
                result=dict(question_id=question_id,canonical_question_sha256=canonical_question_sha256,
                    reference_version=reference_version,task=record["task"],subtask=record["subtask"],
                    reference=reference,choices=payload.get("choices"),source_identity_hash=canonical_digest(source_identity))
                self._cache[key]=copy.deepcopy(result)
                self._events.append(dict(event="APPROVED_REFERENCE_PROJECTED",identity_hash=canonical_digest([question_id,*key]),
                    source_identity_hash=result["source_identity_hash"],reference_revealed_new=True))
                return {**result,"cache_hit":False,"reference_revealed_new":True}
            except Exception as error:
                self._counts["denied_requests"]+=1
                if began_access:
                    self._failed_after_access.add(key);self._counts["failed_after_reference_access"]+=1
                self._events.append(dict(event="REFERENCE_ACCESS_REJECTED",error_type=type(error).__name__,
                    reference_access_account_consumed=began_access))
                raise
