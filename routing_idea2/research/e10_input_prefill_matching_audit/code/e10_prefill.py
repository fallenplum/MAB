"""Frozen Qwen3 input-only prefill extraction, with qualified block-24 stopping."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(r'E:\语言模型组合路由\research\e10_input_prefill_matching_audit')
MODEL = Path(r'E:\语言模型组合路由\models\Qwen--Qwen3-8B')
BLOCK = 24


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def put(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


class CapturedBlock(Exception):
    def __init__(self, vector):
        self.vector = vector


def normalized(value):
    value = value.to(torch.float32)
    return torch.nn.functional.normalize(value, p=2, dim=-1)


def extract(model, encoded, stop=True):
    """Return pre-final-norm block output at each last nonpadding token."""
    mask = encoded['attention_mask']
    position = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = position.masked_fill(mask == 0, -1).max(dim=1).values
    if torch.any(last < 0):
        raise ValueError('Empty input')
    captured = []

    def hook(module, inputs, output):
        state = output[0] if isinstance(output, tuple) else output
        vector = normalized(state[torch.arange(len(state), device=state.device), last]).clone()
        if stop:
            raise CapturedBlock(vector)
        captured.append(vector)

    handle = model.layers[BLOCK - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(**encoded, use_cache=False, output_hidden_states=not stop, return_dict=True)
        if stop:
            raise AssertionError('Block stopping hook did not execute')
        reference = normalized(output.hidden_states[BLOCK][torch.arange(len(mask), device=mask.device), last])
        return captured[0], reference
    except CapturedBlock as complete:
        return complete.vector
    finally:
        handle.remove()


def qualify(model, tokenizer):
    start = time.perf_counter()
    texts = ['A small synthetic input.', 'Another synthetic input with a few more words.']

    def encode(values, side):
        tokenizer.padding_side = side
        return tokenizer(values, padding=True, truncation=False, add_special_tokens=True,
                         return_tensors='pt').to('cuda:0')

    comparisons = []
    batch = encode(texts, 'left')
    hook_full, library_full = extract(model, batch, stop=False)
    early = extract(model, batch)
    comparisons.append(dict(name='hook_vs_hidden_states24', maximum_absolute_error=float((hook_full-library_full).abs().max()),
                            exact=bool(torch.equal(hook_full, library_full))))
    comparisons.append(dict(name='full36_vs_stop24_same_batch', maximum_absolute_error=float((early-library_full).abs().max()),
                            exact=bool(torch.equal(early, library_full))))
    right = extract(model, encode(texts, 'right'))
    single = torch.cat([extract(model, encode([text], 'left')) for text in texts])
    for name, values in [('left_vs_right_padding', right), ('batch_vs_single', single)]:
        comparisons.append(dict(name=name, maximum_absolute_error=float((values-early).abs().max()),
            maximum_l2_error=float(torch.linalg.vector_norm(values-early, dim=-1).max()),
            minimum_cosine=float((values*early).sum(dim=-1).min())))
    tokenizer.padding_side = 'left'
    assert all(item['exact'] for item in comparisons[:2])
    assert all(item['maximum_l2_error'] <= 0.005 and item['minimum_cosine'] >= 0.99998 for item in comparisons[2:])
    return dict(status='PASS', synthetic_texts=2, forward_batches=5,
        full36_batches=1, stop24_batches=4, synthetic_question_processing_events=8,
        target_block_one_based=BLOCK, hidden_states_index=BLOCK,
        location='block24 output before block25 and before final RMSNorm',
        comparisons=comparisons, wall_seconds=time.perf_counter()-start,
        exact_full_vs_early_required=True, numeric_padding_l2_tolerance=0.005,
        numeric_padding_cosine_minimum=0.99998, answer_tokens_generated=0), early.cpu().numpy()


def main(manifest):
    protocol_path = ROOT / 'public/E10_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    protocol_hash = sha(protocol_path)
    freeze = json.loads((ROOT/'private/PROTOCOL_FREEZE_RECEIPT.json').read_text(encoding='utf-8'))
    assert freeze['protocol_sha256'] == protocol_hash
    assert freeze['sources']['e10_prefill.py'] == sha(Path(__file__))
    lineage_path = ROOT/'private/MODEL_AND_PARENT_LINEAGE.json'
    stat_lock_path = ROOT/'private/MODEL_FILE_STAT_LOCK.json'
    assert sha(lineage_path) == freeze['model_lineage_sha256']
    stat_lock = json.loads(stat_lock_path.read_text())
    assert sha(stat_lock_path) == freeze['model_stat_lock_sha256']
    lineage = json.loads(lineage_path.read_text())
    assert lineage['model_revision'] == protocol['prefill']['revision']
    asset_root = ROOT.parent.parent
    for item in lineage['model_files']:
        path = asset_root/item['relative']
        current = path.stat()
        cached = stat_lock['files'][item['relative']]
        assert current.st_size == item['bytes']
        if current.st_mtime_ns != cached['mtime_ns']:
            assert sha(path) == item['sha256'], 'Frozen model bytes changed'
    if (ROOT/'private/features/FEATURE_FREEZE_RECEIPT.json').exists():
        receipt = json.loads((ROOT/'private/features/FEATURE_FREEZE_RECEIPT.json').read_text())
        assert receipt['protocol_sha256'] == protocol_hash
        assert receipt['feature_sha256'] == sha(ROOT/'private/features/PREFILL_FEATURES_FROZEN.npz')
        print('PREFILL_CACHE_ALREADY_COMPLETE', flush=True)
        return
    rows = json.loads(Path(manifest).read_text(encoding='utf-8-sig'))
    assert isinstance(rows, list) and len(rows) == 600
    assert [row['global_index'] for row in rows] == list(range(600))
    expected_input_hash = freeze['prefill_input_manifest_sha256']
    assert sha(manifest) == expected_input_hash
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert protocol['prefill']['tokenization']['add_special_tokens'] is True
    config = json.loads((MODEL/'config.json').read_text())
    max_context = config['max_position_embeddings']
    requests = []
    for row in rows:
        assert hashlib.sha256(row['text'].encode()).hexdigest() == row['text_sha256']
        requests.append(('full', row, row['text']))
    truncated = [row for row in rows if row['truncated']]
    if truncated:
        if not all(row['mapping_valid'] for row in truncated):
            put(ROOT/'private/R_MATCH_UNAVAILABLE.json', dict(status='UNAVAILABLE_PREFIX_MAPPING', rows=len(truncated)))
        else:
            requests += [('match', row, row['visible_prefix_text']) for row in truncated]
    for variant, row, text in requests:
        length = len(tokenizer(text, add_special_tokens=True, truncation=False)['input_ids'])
        if length > max_context:
            put(ROOT/'private/PREFILL_CONTEXT_BLOCK.json', dict(status='BLOCKED', variant=variant,
                global_index=row['global_index'], input_tokens=length, maximum=max_context))
            raise RuntimeError('Canonical input exceeds frozen checkpoint context')

    torch.manual_seed(20261114101)
    torch.set_num_threads(1)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    load_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModel.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={'': 'cuda:0'},
        attn_implementation='sdpa', quantization_config=quantization).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-load_started
    load_peak_allocated = torch.cuda.max_memory_allocated()
    load_peak_reserved = torch.cuda.max_memory_reserved()
    assert len(model.layers) == 36 and model.config.hidden_size == 4096
    qualification_wall_start = time.perf_counter()
    qualification_start, qualification_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    qualification_start.record()
    qualification, synthetic_reference = qualify(model, tokenizer)
    # All formal extraction aborts at this block before later layers execute.
    # Removing unexecuted modules reduces resident VRAM after the exact qualification.
    del model.layers[BLOCK:]
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    synthetic_texts = ['A small synthetic input.', 'Another synthetic input with a few more words.']
    synthetic_encoded = tokenizer(synthetic_texts, padding=True, truncation=False, add_special_tokens=True,
        return_tensors='pt').to('cuda:0')
    prefix_only = extract(model, synthetic_encoded).cpu().numpy()
    assert np.array_equal(prefix_only, synthetic_reference)
    qualification['comparisons'].append(dict(name='removed_unexecuted_layers_vs_full_resident', exact=True,
        maximum_absolute_error=0.0))
    qualification.update(forward_batches=6, stop24_batches=5, synthetic_question_processing_events=10)
    qualification['synthetic_input_tokens'] = 5 * sum(len(tokenizer(text, add_special_tokens=True)['input_ids']) for text in synthetic_texts)
    qualification_end.record()
    torch.cuda.synchronize()
    qualification.update(created_utc=now(), source_sha256=sha(Path(__file__)), protocol_sha256=protocol_hash,
        model_revision=protocol['prefill']['revision'], model_load_wall_seconds=load_seconds,
        model_load_peak_allocated_bytes=load_peak_allocated, model_load_peak_reserved_bytes=load_peak_reserved,
        gpu_seconds=qualification_start.elapsed_time(qualification_end)/1000,
        total_qualification_wall_seconds=time.perf_counter()-qualification_wall_start,
        gpu_timing_scope='CUDA event elapsed interval including intervening host scheduling, not sum of kernel durations',
        torch_version=torch.__version__, transformers_version=__import__('transformers').__version__,
        bitsandbytes_version=__import__('bitsandbytes').__version__, gpu_name=torch.cuda.get_device_name(0))
    qualification_path = ROOT/'private/logs'/('PREFILL_QUALIFICATION_'+str(time.time_ns())+'.json')
    put(qualification_path, qualification)
    if not (ROOT/'public/PREFILL_FEATURE_QUALIFICATION.json').exists():
        put(ROOT/'public/PREFILL_FEATURE_QUALIFICATION.json', qualification)
    torch.cuda.reset_peak_memory_stats()
    records = []
    cache_hits = 0
    newly_processed = 0
    for variant, row, text in requests:
        folder = ROOT/f'private/features/{variant}'
        folder.mkdir(parents=True, exist_ok=True)
        filename = folder/f'{row["global_index"]:03d}.npy'
        receipt_path = filename.with_suffix('.json')
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if receipt_path.exists():
            record = json.loads(receipt_path.read_text(encoding='utf-8'))
            assert record['protocol_sha256'] == protocol_hash and record['text_sha256'] == text_hash
            assert record['feature_sha256'] == sha(filename)
            records.append(record)
            cache_hits += 1
            continue
        start = time.perf_counter()
        cpu = time.process_time()
        encoded = tokenizer([text], return_tensors='pt', truncation=False,
            add_special_tokens=True, padding=False).to('cuda:0')
        cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        cuda_start.record()
        vector = extract(model, encoded)
        cuda_end.record()
        torch.cuda.synchronize()
        vector = vector[0].cpu().numpy()
        latency = time.perf_counter()-start
        assert vector.shape == (4096,) and vector.dtype == np.float32 and np.isfinite(vector).all()
        assert abs(float(np.linalg.norm(vector))-1.0) < 1e-5
        if filename.exists():
            raise RuntimeError('Unsealed prior attempt requires explicit quarantine')
        np.save(filename, vector, allow_pickle=False)
        record = dict(global_index=row['global_index'], variant=variant, protocol_sha256=protocol_hash,
            text_sha256=text_hash, input_tokens=int(encoded['attention_mask'].sum()),
            token_ids_sha256=hashlib.sha256(encoded['input_ids'].cpu().numpy().tobytes()).hexdigest(),
            feature_sha256=sha(filename), completed_utc=now(), gpu_milliseconds=cuda_start.elapsed_time(cuda_end),
            selector_feature_wall_seconds=latency, cpu_seconds=time.process_time()-cpu,
            transformer_blocks_executed=24, forward_batches=1,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            answer_tokens_generated=0)
        put(receipt_path, record)
        records.append(record)
        newly_processed += 1
        print(json.dumps(dict(completed=len(records), total=len(requests), variant=variant,
            index=row['global_index'], tokens=record['input_tokens'], seconds=latency)), flush=True)
    full = np.stack([np.load(ROOT/f'private/features/full/{i:03d}.npy', allow_pickle=False) for i in range(600)])
    matched = full.copy()
    for row in truncated:
        path = ROOT/f'private/features/match/{row["global_index"]:03d}.npy'
        if path.exists():
            matched[row['global_index']] = np.load(path, allow_pickle=False)
    artifact = ROOT/'private/features/PREFILL_FEATURES_FROZEN.npz'
    if not artifact.exists():
        np.savez_compressed(artifact, full=full, matched=matched,
            global_indices=np.arange(600), truncated=np.array([row['truncated'] for row in rows]))
    ready = ROOT/'private/features/FEATURE_FREEZE_RECEIPT.json'
    put(ready, dict(status='COMPLETE', completed_utc=now(), protocol_sha256=protocol_hash,
        feature_sha256=sha(artifact), requests=len(requests), n_truncated=len(truncated),
        this_run_new_question_processing_events=newly_processed, this_run_cache_hits=cache_hits,
        r_match_nontruncated_feature_reuse=600-len(truncated) if truncated else 0,
        r_match_available=not truncated or all(row['mapping_valid'] for row in truncated),
        model_load_wall_seconds=load_seconds, gpu_name=torch.cuda.get_device_name(0),
        formal_input_tokens=sum(item['input_tokens'] for item in records),
        formal_gpu_seconds=sum(item['gpu_milliseconds'] for item in records)/1000,
        formal_latency_seconds_sum=sum(item['selector_feature_wall_seconds'] for item in records),
        formal_cpu_seconds_sum=sum(item['cpu_seconds'] for item in records),
        formal_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        formal_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        per_question_receipt_sha256={f'{item["variant"]}/{item["global_index"]}':sha(ROOT/f'private/features/{item["variant"]}/{item["global_index"]:03d}.json') for item in records},
        transformer_blocks_per_formal_forward=24, no_lm_head_evaluated=True,
        model_lineage_sha256=sha(lineage_path), model_stat_lock_sha256=sha(stat_lock_path),
        weight_identity_verification='P0 all-file SHA256 followed by extraction-time unchanged size/mtime cache, changed mtime forces SHA256',
        qualification_receipt=str(qualification_path), answer_tokens_generated=0))
    print('PREFILL_FEATURES_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    main(parser.parse_args().manifest)
