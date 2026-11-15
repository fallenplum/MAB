"""Input-only E18 source inventory and unadmitted candidate preparation.

The sealed split is handled exclusively through external identity metadata. This
entry point never opens a sealed directory or a sealed input/gold payload. Mixed
historical JSONL sources are projected with a field allowlist; reference/member
output values are syntax-skipped. Official BBH example bodies are only counted
and source-indexed, never decoded, until an external sealed-position map exists.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import time
import unicodedata

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


DEFAULT_ROOT = Path(r"E:\语言模型组合路由")
STUDY_NAME = "e18_mechanism_and_confirmation_preparation"
SEED = 20261116018
DECODER = json.JSONDecoder()
HEADER = re.compile(rb'^\s*\{\s*"question_id"\s*:\s*"([^"\\]+)"\s*[,}]')
INPUT_FIELDS = {"question_id", "task", "subtask", "source_split", "question"}
METADATA_FIELDS = {"question_id", "task", "subtask", "source_split", "source_index", "source_file",
                   "source_question_ids", "subtask_source_file", "source_revision", "subtask_source_revision"}
NEAR_RULES = dict(
    schema="e18_input_duplicate_rules_v1", seed=SEED,
    exact="SHA256 of Unicode NFKC text with whitespace collapsed; no task/source prefix",
    token_equivalence="Ordered casefolded word/number/operator tokens match exactly; only punctuation/case formatting varies",
    critical_slots="Ordered numeric literals, quoted strings, capitalized word tokens, and math/logical operators must match",
    near_candidate_filter="same task and BBH subtask plus identical critical-slot signature",
    character_tfidf_cosine_minimum=0.985, character_sequence_ratio_minimum=0.985,
    word_token_jaccard_minimum=0.95,
    near_edges="Review candidates only; no union-find/transitive merging of pairwise similar texts",
    protected_comparison="Private HMAC slot gate, ordered-token equivalence, hashed character3-5gram count cosine>=.985, ordered-trigram ratio>=.985, word-token hash Jaccard>=.95",
    selection="Exclude all high-similarity review endpoints from text-ready proposals; do not count duplicate token-equivalence groups multiple times",
    limitations="High-threshold syntactic screening does not certify absence of semantic paraphrases. Sealed near-duplicate screening requires external precomputed signatures or separately authorized inputs.",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def canonical_hash(text):
    return hashlib.sha256(canonical(text).encode()).hexdigest()


def _skip_space(text, position):
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def skip_value(text, position):
    """Locate a JSON value's end without decoding any of its content."""
    position = _skip_space(text, position)
    depth, quoted, escaped = 0, False, False
    for index in range(position, len(text)):
        character = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
        elif character in "]}":
            if depth == 0:
                return index
            depth -= 1
            if depth == 0 and text[position] in "[{":
                return index+1
        elif character == "," and depth == 0:
            return index
    if depth or quoted:
        raise ValueError("Unterminated JSON value")
    return len(text)


def object_fields(text):
    position = _skip_space(text, 0)
    if text[position] != "{":
        raise ValueError("Expected JSON object")
    position += 1
    seen = set()
    while True:
        position = _skip_space(text, position)
        if text[position] == "}":
            return
        key, position = DECODER.raw_decode(text, position)
        if not isinstance(key, str) or key in seen:
            raise ValueError("Duplicate or invalid JSON object key")
        seen.add(key)
        position = _skip_space(text, position)
        if text[position] != ":":
            raise ValueError("Missing JSON separator")
        start = _skip_space(text, position+1)
        end = skip_value(text, start)
        yield key, start, end
        position = _skip_space(text, end)
        if text[position] == ",":
            position += 1
        elif text[position] != "}":
            raise ValueError("Malformed JSON object boundary")


def project(raw, allowed):
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    result = {}
    for key, start, end in object_fields(text):
        if key in allowed:
            result[key] = DECODER.raw_decode(text, start)[0]
    return result


def official_example_count(path):
    """Count whole examples structurally; do not decode input or target values."""
    text = Path(path).read_bytes().decode("utf-8-sig")
    location = [(start,end) for key,start,end in object_fields(text) if key == "examples"]
    if len(location) != 1:
        raise ValueError("Official BBH source requires exactly one examples array")
    position, end = location[0]
    if text[position] != "[":
        raise ValueError("Official examples field is not an array")
    count = 0
    position += 1
    while True:
        position = _skip_space(text, position)
        if text[position] == "]":
            return count
        if text[position] != "{":
            raise ValueError("Official example is not an object")
        position = _skip_space(text, skip_value(text, position))
        count += 1
        if text[position] == ",":
            position += 1
        elif text[position] != "]" or position >= end:
            raise ValueError("Malformed official examples array")


def official_example_objects(path):
    """Yield encoded whole example objects, leaving value selection to caller."""
    text=Path(path).read_bytes().decode("utf-8-sig")
    location=[(start,end) for key,start,end in object_fields(text) if key=="examples"]
    if len(location)!=1:
        raise ValueError("Expected one official examples array")
    position,end=location[0]
    if text[position]!="[":
        raise ValueError("Expected official examples array")
    position+=1
    while True:
        position=_skip_space(text,position)
        if text[position]=="]":
            return
        start=position
        stop=skip_value(text,start)
        if text[start]!="{":
            raise ValueError("Official example is not an object")
        yield text[start:stop]
        position=_skip_space(text,stop)
        if text[position]==",":
            position+=1
        elif text[position]!="]" or position>=end:
            raise ValueError("Malformed official example boundary")


def _keyed_hash(key,value):
    encoded=json.dumps(value,ensure_ascii=False,separators=(",", ":")).encode()
    return hmac.new(key,encoded,hashlib.sha256).hexdigest()


def _private_signature(row,key):
    text=canonical(row["question"]).casefold()
    grams=Counter(text[index:index+width] for width in (3,4,5) for index in range(max(0,len(text)-width+1)))
    tokens=_tokens(text)
    return dict(
        critical_slots_hmac256=_keyed_hash(key,_slots(row)),
        ordered_tokens_hmac256=_keyed_hash(key,tokens),
        character_gram_counts={_keyed_hash(key,value):count for value,count in grams.items()},
        ordered_trigram_hmac256=[_keyed_hash(key,text[index:index+3]) for index in range(max(0,len(text)-2))],
        word_token_hmac256=sorted({_keyed_hash(key,token) for token in tokens}),
    )


def protected_fingerprints(root,output):
    """One authorized isolated process, with no protected plaintext persisted."""
    started_cpu,started_wall=time.process_time(),time.perf_counter()
    private=output/"private"
    target=private/"protected_input_fingerprints"
    if target.exists():
        raise FileExistsError("Protected one-time audit already started; do not repeat it")
    metadata=read_json(private/"REGISTERED_EXCLUSION_METADATA.json")["records"]
    protected={row["question_id"]:row for row in metadata if row["split"]=="sealed_test"}
    by_canonical={row["canonical_question_sha256"]:row for row in metadata}
    if len(protected)!=200:
        raise ValueError("External protected identity count differs")
    source_manifest=read_json(private/"SOURCE_MANIFEST.json")
    target.mkdir(parents=True)
    write_json(target/"AUTHORIZED_SOURCE_FREEZE.json",dict(
        status="AUTHORIZED_ONCE_ONLY_INPUT_FINGERPRINT_AUDIT_STARTED",
        authorization="User explicitly allowed the one-time input fingerprint audit through root on2026-11-16",
        code_sha256=sha(Path(__file__)),protected_identity_source_sha256=sha(private/"REGISTERED_EXCLUSION_METADATA.json"),
        permitted="Input text transiently projected for identity/fingerprint only; no plaintext output, answer, encoder, training, or model selection",
    ))
    key=os.urandom(32)
    # Reconstruct the old subpool's identity mapping using only metadata for its
    # protected records. No protected input is decoded from this mixed source.
    old_subpool={}
    with (root/"data/derived/available_records.jsonl").open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            match=HEADER.match(raw.removeprefix(b"\xef\xbb\xbf"))
            if match is None:
                raise ValueError("Missing external source qid header")
            qid=match.group(1).decode("ascii")
            if qid in protected:
                old_subpool[protected[qid]["canonical_question_sha256"]]=qid
            else:
                row=project(raw,INPUT_FIELDS)
                old_subpool[canonical_hash(row["question"])]=qid
    signatures={}
    occurrences=Counter()
    official_index=[]
    for source in source_manifest["official_bbh"]:
        path=root/source["source_file"]
        if sha(path)!=source["source_sha256"]:
            raise ValueError("Official source changed before isolated fingerprint pass")
        for index,raw in enumerate(official_example_objects(path)):
            value=project(raw,{"input"})
            if set(value)!={"input"} or not isinstance(value["input"],str):
                raise ValueError("Official input-only projection differs")
            question=value["input"]
            fingerprint=canonical_hash(question)
            original=by_canonical.get(fingerprint)
            generated="bbh:"+hashlib.sha256(("bbh\ntest\n"+" ".join(question.split())).encode()).hexdigest()
            qid=old_subpool.get(fingerprint,original["question_id"] if original else generated)
            protected_row=protected.get(qid)
            official_index.append(dict(
                question_id=qid,canonical_question_sha256=fingerprint,task="bbh",subtask=source["subtask"],
                source_split="test",source_file=source["source_file"],source_sha256=source["source_sha256"],
                source_index=index,protected_sealed=protected_row is not None,
                source_frame="bbh_old_llmrouterbench_common_subpool" if fingerprint in old_subpool else "bbh_official_full_source_frame_expansion",
            ))
            if protected_row is not None:
                if protected_row["canonical_question_sha256"]!=fingerprint or protected_row["task"]!="bbh" or protected_row["subtask"]!=source["subtask"]:
                    raise ValueError("Protected BBH metadata and official position differ")
                row=dict(question=question,task="bbh",subtask=source["subtask"])
                signatures[qid]=dict(question_id=qid,canonical_question_sha256=fingerprint,task="bbh",subtask=source["subtask"],**_private_signature(row,key))
                occurrences["bbh"]+=1
            del question,value
    if len(official_index)!=6511:
        raise ValueError("Official input index coverage differs")
    with (root/"data/derived/gsm8k_questions.jsonl").open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            match=HEADER.match(raw.removeprefix(b"\xef\xbb\xbf"))
            if match is None:
                raise ValueError("Missing GSM external qid header")
            qid=match.group(1).decode("ascii")
            if qid not in protected:
                continue
            row=project(raw,INPUT_FIELDS)
            expected=protected[qid]
            fingerprint=canonical_hash(row["question"])
            if row["task"]!="gsm8k" or fingerprint!=expected["canonical_question_sha256"] or row["source_split"]!="test":
                raise ValueError("Protected GSM metadata differs")
            signatures[qid]=dict(question_id=qid,canonical_question_sha256=fingerprint,task="gsm8k",subtask=None,**_private_signature(row,key))
            occurrences["gsm8k"]+=1
            del row
    if set(signatures)!=set(protected) or occurrences!={"gsm8k":100,"bbh":100}:
        raise ValueError("Authorized pass must match exactly200 protected original inputs")
    with (target/"FINGERPRINT_KEY.bin").open("xb") as stream:
        stream.write(key)
    write_json(target/"PROTECTED_INPUT_SIGNATURES.json",dict(schema="private_keyed_input_signatures_v1",records=list(signatures.values()),plaintext_persisted=False))
    write_json(target/"OFFICIAL_BBH_INPUT_IDENTITY_INDEX.json",dict(schema="external_bbh_source_position_identity_v1",records=official_index,
        input_text_persisted=False,answer_values_decoded=0,protected_identity_matches=100))
    receipt=dict(
        status="AUTHORIZED_ONCE_ONLY_INPUT_FINGERPRINT_AUDIT_COMPLETE",unique_protected_questions=200,
        actual_protected_input_value_occurrences_decoded=sum(occurrences.values()),
        actual_protected_input_values_by_task=dict(occurrences),official_bbh_input_values_processed=6511,
        protected_plaintext_displayed_or_persisted=False,protected_embedding_forwards=0,protected_training_rows=0,
        reference_answer_values_decoded=0,fresh_reward_values_read=0,sealed_directory_opened=False,
        signature_sha256=sha(target/"PROTECTED_INPUT_SIGNATURES.json"),
        bbh_position_index_sha256=sha(target/"OFFICIAL_BBH_INPUT_IDENTITY_INDEX.json"),
        code_sha256=sha(Path(__file__)),cpu_s=time.process_time()-started_cpu,wall_s=time.perf_counter()-started_wall,
    )
    write_json(target/"AUTHORIZED_ACCESS_RECEIPT.json",receipt)
    print(json.dumps({key:receipt[key] for key in ("status","unique_protected_questions","actual_protected_input_value_occurrences_decoded","reference_answer_values_decoded")},ensure_ascii=False))


def _source(root, relative, expected=None):
    path = root/relative
    actual = sha(path)
    if expected and actual != expected:
        raise ValueError("Source no longer matches historical receipt: "+relative)
    return dict(path=relative, sha256=actual, bytes=path.stat().st_size)


def _count_records(rows, *fields):
    counts = Counter(tuple(str(row.get(field) or "") for field in fields) for row in rows)
    return [dict(zip(fields,key), questions=count) for key,count in sorted(counts.items())]


def _safe_bbh_index_metadata(root, sealed_ids):
    """Read external metadata only, including for sealed identities."""
    count = 0
    sealed_records = []
    indices = []
    with (root/"data/derived/available_records.jsonl").open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            row = project(raw, METADATA_FIELDS)
            count += 1
            values = row.get("source_question_ids", {})
            indices.extend(value for value in values.values() if isinstance(value, int))
            if row["question_id"] in sealed_ids:
                sealed_records.append(row)
    return dict(
        source_records=count, sealed_external_metadata_records=len(sealed_records),
        available_fields=sorted({key for row in sealed_records for key in row}),
        source_question_id_type="LLMRouterBench per-model record index; no validated official BBH within-file example index",
        numeric_source_indices_min=min(indices) if indices else None,
        numeric_source_indices_max=max(indices) if indices else None,
        official_position_mapping_qualified=False,
        sealed_input_or_reference_fields_decoded=0,
    ), sealed_records


def inventory(root, output):
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    public, private = output/"public", output/"private"
    old_path = root/"data/derived/split_manifest.json"
    audit = read_json(root/"artifacts/audit/split_audit.json")
    old = read_json(old_path)
    if sha(old_path) != audit["manifest_sha256"]:
        raise ValueError("Original split manifest changed")
    expected_fields = {"question_id", "task", "subtask", "split", "source_split"}
    if len(old["records"]) != 1308 or any(set(row) != expected_fields for row in old["records"]):
        raise ValueError("Old split metadata contains unexpected fields or membership")
    fresh_path = root/"artifacts/private/routing_redesign_20261104/fresh_dev400/SELECTION_MANIFEST_v1.json"
    input_receipt = read_json(root/"artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/INPUT_ASSETS_RECEIPT.json")
    fresh = read_json(fresh_path)
    if sha(fresh_path) != input_receipt["fresh_selection"]["sha256"] or len(fresh["records"]) != 400:
        raise ValueError("Original fresh400 selection metadata changed")
    forbidden = {"question", "reference", "answer", "reward", "score", "target"}
    if any(forbidden & set(row) for row in fresh["records"]):
        raise ValueError("Fresh selection metadata contains forbidden fields")
    exclusions = []
    for row in old["records"]:
        exclusions.append(dict(row, canonical_question_sha256=old["canonical_question_fingerprints"][row["question_id"]],
                               role="original_sealed_metadata_only" if row["split"]=="sealed_test" else "registered_historical_development"))
    for row in fresh["records"]:
        exclusions.append(dict(row, split="fresh_dev400", role="existing_fresh_input_cohort_reserved"))
    if len({row["question_id"] for row in exclusions}) != 1708 or len({row["canonical_question_sha256"] for row in exclusions}) != 1708:
        raise ValueError("Registered cohort identities overlap")
    sealed_ids = {row["question_id"] for row in exclusions if row["split"]=="sealed_test"}
    index_check, sealed_metadata = _safe_bbh_index_metadata(root, sealed_ids)
    bbh_receipt = read_json(root/"data/raw/bbh_official/source_receipt.json")
    gsm_receipt = read_json(root/"data/raw/gsm8k_official/source_receipt.json")
    official_bbh = []
    sources = [_source(root,"data/derived/split_manifest.json",audit["manifest_sha256"]),
               _source(root,"artifacts/audit/split_audit.json"), _source(root,str(fresh_path.relative_to(root))),
               _source(root,"data/raw/bbh_official/source_receipt.json"),
               _source(root,"data/raw/gsm8k_official/source_receipt.json")]
    for source in audit["source_files"]:
        if source["path"] in ("data/derived/available_records.jsonl", "data/derived/gsm8k_questions.jsonl"):
            sources.append(_source(root,source["path"],source["sha256"]))
    for record in bbh_receipt["files"]:
        sources.append(_source(root,record["path"],record["sha256"]))
        official_bbh.append(dict(source_file=record["path"],source_sha256=record["sha256"],
                                 subtask=Path(record["source_path"]).stem,
                                 examples=official_example_count(root/record["path"]),
                                 input_values_decoded=0, target_values_decoded=0))
    if len(official_bbh) != 27 or sum(row["examples"] for row in official_bbh) != 6511:
        raise ValueError("Official BBH structural example count differs")
    gsm = read_json(root/"artifacts/audit/gsm8k_source_audit.json")
    for record in gsm_receipt["files"]:
        # Count only physical JSONL rows, never decode the mixed raw payload.
        sources.append(_source(root,record["path"],record["sha256"]))
        with (root/record["path"]).open("rb") as stream:
            count = sum(1 for raw in stream if raw.strip())
        if count != gsm["counts"][Path(record["source_path"]).stem]:
            raise ValueError("Official GSM8K source row count differs")
    write_json(private/"REGISTERED_EXCLUSION_METADATA.json",dict(status="METADATA_ONLY_EXCLUSIONS",records=exclusions,
               sealed_records_source="external original split manifest only; no sealed path enumerated"))
    write_json(private/"SEALED_EXTERNAL_INDEX_AUDIT.json",dict(summary=index_check, external_metadata=sealed_metadata))
    write_json(private/"SOURCE_MANIFEST.json",dict(files=sources,official_bbh=official_bbh))
    result = dict(
        schema="e18_data_source_inventory_v1", status="INVENTORY_COMPLETE_BBH_OFFICIAL_INPUT_PROJECTION_BLOCKED",
        registered_cohort_counts=_count_records(exclusions,"split","task","source_split"),
        old_registered_questions=1308, fresh_reserved_questions=400, registered_exclusion_union=1708,
        sealed_questions=200, sealed_directories_opened=0, sealed_payloads_opened=0,
        official_gsm_counts=gsm["counts"], official_bbh_examples=6511, official_bbh_subtasks=27,
        official_bbh_per_subtask=[dict(subtask=row["subtask"],examples=row["examples"]) for row in official_bbh],
        old_bbh_cached_subpool=1078,
        primary_source_target="GSM8K official train plus BBH official test, matching existing fresh400",
        sealed_source_target_difference="Original sealed200 uses100 GSM8K official test and100 BBH test; it is a different GSM source stratum.",
        main_original_source_exact_metadata_upper_bounds=dict(gsm_train=6719, bbh_old_cached_subpool=224),
        broader_official_bbh_source_frame=dict(examples=6511, proposed_additional_questions=1300,
            layer="official BBH expanded coverage beyond old LLMRouterBench common subset",
            position_exclusion_status="BLOCKED_EXTERNAL_SEALED_INDEX"),
        alternate_gsm_test_target=dict(official_test=1319,reserved_sealed_test=100,metadata_upper_bound=1219,
            warning="Cannot supply1500 test-stratum GSM questions; official train is a separately disclosed extension for that different target."),
        sealed_external_index_audit=index_check,
        safe_input_sources=[
            dict(path="data/derived/gsm8k_questions.jsonl",schema=sorted(INPUT_FIELDS),projection="qid header and metadata first; never decode a sealed row or any reference; primary train only"),
            dict(path="data/derived/available_records.jsonl",schema=sorted(INPUT_FIELDS),projection="qid header excludes sealed before question projection; all other fields syntax-skipped"),
            dict(path="artifacts/private/routing_redesign_20261104/fresh_dev400/inputs_only.jsonl",schema=sorted(INPUT_FIELDS|{"canonical_question_sha256"}),projection="already input-only; no reward/answer fields allowed"),
        ],
        additional_used_cohort_scope="All original registered1308 plus fresh400; E1-E17 canonical pipeline reuses these cohorts. An unregistered manual inspection would require an appended external exclusion registry.",
        unresolved=["No validated external official BBH example-index mapping for sealed identities.",
                    "No external sealed text near-duplicate signatures; exact identity exclusion alone cannot complete near-duplicate acceptance."],
        source_manifest_sha256=sha(private/"SOURCE_MANIFEST.json"),
        new_llm_calls=0,new_downloads=0,gold_values_decoded=0, confidence_intervals=0,bootstrap_runs=0,
        cpu_s=time.process_time()-start_cpu,wall_s=time.perf_counter()-start_wall,
    )
    write_json(public/"DATA_SOURCE_INVENTORY.json",result)
    print(json.dumps({key:result[key] for key in ("status","official_gsm_counts","official_bbh_examples","main_original_source_exact_metadata_upper_bounds")},ensure_ascii=False))


def _tokens(text):
    return tuple(re.findall(r"\d+(?:\.\d+)?|[^\W\d_]+|[+\-*/=<>≤≥≠×÷\[\](){}%^&|~]",canonical(text).casefold(),re.UNICODE))


def _slots(row):
    text = canonical(row["question"])
    return (row["task"],row.get("subtask") or "",
            tuple(re.findall(r"[+\-]?\d+(?:[.,/]\d+)*",text)),
            tuple(re.findall(r'"([^"\n]+)"|“([^”\n]+)”',text)),
            tuple(re.findall(r"\b[A-Z][A-Za-z]*\b",text)),
            tuple(re.findall(r"[+\-*/=<>≤≥≠×÷\[\](){}%^&|~]",text)))


def _screen_near_duplicates(rows):
    """Conservative syntactic screens; never merge a similarity relation."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    token_groups, slot_groups = defaultdict(list), defaultdict(list)
    for index,row in enumerate(rows):
        token_groups[_tokens(row["question"])].append(index)
        slot_groups[_slots(row)].append(index)
    exact_format_groups = [group for group in token_groups.values() if len(group)>1]
    pending, edges = set(), []
    for indices in slot_groups.values():
        if len(indices)<2:
            continue
        texts = [canonical(rows[index]["question"]).casefold() for index in indices]
        matrix = TfidfVectorizer(analyzer="char",ngram_range=(3,5),norm="l2").fit_transform(texts)
        similarity = (matrix@matrix.T).tocoo()
        for first,second,cosine in zip(similarity.row,similarity.col,similarity.data):
            if first>=second or cosine+1e-12<NEAR_RULES["character_tfidf_cosine_minimum"]:
                continue
            a,b=indices[first],indices[second]
            if _tokens(rows[a]["question"])==_tokens(rows[b]["question"]):
                continue
            ratio=SequenceMatcher(None,texts[first],texts[second],autojunk=False).ratio()
            left,right=set(_tokens(texts[first])),set(_tokens(texts[second]))
            jaccard=len(left&right)/len(left|right)
            if ratio>=NEAR_RULES["character_sequence_ratio_minimum"] and jaccard>=NEAR_RULES["word_token_jaccard_minimum"]:
                pending.update((a,b))
                edges.append(dict(first_question_id=rows[a]["question_id"],second_question_id=rows[b]["question_id"],
                                  character_tfidf_cosine=float(cosine),character_sequence_ratio=ratio,word_jaccard=jaccard,
                                  disposition="PENDING_REVIEW_NO_TRANSITIVE_EQUIVALENCE"))
    return exact_format_groups,pending,edges


def _rank(label,value):
    return hashlib.sha256(f"{SEED}\n{label}\n{value}".encode()).hexdigest()


def build_proposal(root,output):
    from candidate_proposal import build_proposal as prepare
    return prepare(root,output)


def self_test():
    raw=b'{"question_id":"fixture", "question":"What is 2+2?", "reference":"SECRET", "member_outputs":{"x":{"raw_output":"SECRET"}}, "task":"gsm8k", "subtask":null,"source_split":"train"}'
    projected=project(raw,INPUT_FIELDS)
    if set(projected)!=INPUT_FIELDS or "SECRET" in json.dumps(projected):
        raise AssertionError("Forbidden-value projection failed")
    if _tokens("A has 2 apples.")==_tokens("A has 3 apples."):
        raise AssertionError("Different numerical instances collapsed")
    if _tokens("Alice has 2 apples.")==_tokens("Bob has 2 apples."):
        raise AssertionError("Different entity instances collapsed")
    if _tokens("2+3")==_tokens("2-3"):
        raise AssertionError("Different operators collapsed")
    if _tokens("A has 2 apples.")!=_tokens("a has 2 apples!"):
        raise AssertionError("Formatting equivalence control failed")
    body=' {"examples":[{"input":"not decoded","target":"secret"},{"input":"not decoded2","target":"secret2"}], "other":{}} '
    locations=[(key,start,end) for key,start,end in object_fields(body)]
    if [key for key,_,_ in locations]!=["examples","other"]:
        raise AssertionError("Structural source-field parsing failed")
    print("SOURCE_PROJECTION_AND_INSTANCE_PRESERVATION_CONTROLS_PASS")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("self-test","inventory","protected-fingerprints","build-proposal"))
    parser.add_argument("--asset-root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    destination=args.output or args.asset_root/"research"/STUDY_NAME
    if args.stage=="self-test":
        self_test()
    elif args.stage=="inventory":
        inventory(args.asset_root,destination)
    elif args.stage=="protected-fingerprints":
        protected_fingerprints(args.asset_root,destination)
    else:
        build_proposal(args.asset_root,destination)
