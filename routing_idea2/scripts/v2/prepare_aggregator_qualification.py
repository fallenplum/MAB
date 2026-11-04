"""Prepare label-free real qualification inputs and development-role metadata for v2."""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import argparse
import hashlib
import json
import os
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
QWEN3_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
ROLE_SPLITS = {
    'aggregator_calibration': ('e1_selection_calibration', 100),
    'selector_train': ('online_stream', 600),
    'selector_validation': ('online_calibration', 200),
    'repeated_development_diagnostics': ('e1_diagnostic_evaluation', 200),
    'technical_qualification': ('qualification', 8),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_immutable(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, 'Existing prepared bytes differ: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    sys.path.insert(0, str(root / 'src'))
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    from budget_router.inference import ModelSpec, GenerationResult, configuration_fingerprint
    from budget_router.execution import make_job, validate_job, _validate_result, JOB_FIELDS
    from budget_router.member_execution import MemberStore, validate_member_job
    from budget_router.preparation import validate_success_rows
    from transformers import AutoTokenizer

    inputs = {}
    def source(relative):
        path = root / relative
        inputs[relative] = sha(path)
        return path

    manifest = read(source('data/derived/split_manifest.json'))
    require(all(set(row) == {'question_id', 'task', 'subtask', 'split', 'source_split'}
                for row in manifest['records']), 'Split metadata unexpectedly contains question/label fields')
    qualification = [row for row in manifest['records'] if row['split'] == 'qualification']
    require(Counter(row['task'] for row in qualification) == {'gsm8k': 4, 'bbh': 4},
            'Original qualification requires four questions per task')
    allowed = {row['question_id']: row for row in qualification}
    pool = read(source('configs/pool_v1.json'))['models']
    require(len(pool) == 6, 'Original six-member pool required')
    original_config = read(source('configs/aggregator_v1.json'))
    original_spec = ModelSpec(**original_config)
    original_jobs = jsonl(source('data/jobs/qualification/aggregator.jsonl'))
    for job in original_jobs:
        validate_job(job, original_spec, require_token_policy=True)
        require(job['split'] == 'qualification' and job['question_id'] in allowed,
                'Original runtime jobs crossed qualification membership')

    member_jobs = {}
    member_rows = {}
    questions = {}
    specs = {}
    for entry in pool:
        alias = entry['alias']
        spec = ModelSpec(**read(source(entry['config'])))
        spec.validate()
        specs[alias] = spec
        path = source('data/jobs/qualification/members_' + alias.lower() + '.jsonl')
        jobs = jsonl(path)
        raw = source('results/raw/qualification/members_' + alias.lower() + '.jsonl')
        rows = validate_success_rows(raw)
        store = MemberStore(raw)
        require(len(jobs) == len(rows) == len(store.completed) == 8 and not store.unreconciled,
                'Real original qualification member cache is incomplete: ' + alias)
        member_jobs[alias] = {}
        member_rows[alias] = {}
        for job in jobs:
            validate_member_job(job, spec)
            qid = job['question_id']
            require(qid in allowed and job['split'] == 'qualification' and job['seed'] == 42 and
                    job['task'] == allowed[qid]['task'] and job['subtask'] == allowed[qid]['subtask'],
                    'Member job differs from original qualification metadata')
            canonical = ' '.join(unicodedata.normalize('NFKC', job['question']).split())
            require(hashlib.sha256(canonical.encode()).hexdigest() == manifest['canonical_question_fingerprints'][qid],
                    'Qualification question differs from locked canonical fingerprint')
            if qid in questions:
                require(questions[qid] == job['question'], 'Member question text differs across pool')
            questions[qid] = job['question']
            row = store._cached(job, spec)
            require(row is not None and row['cost_status'] == 'recorded_actual_usage',
                    'Original member cache replay failed')
            _validate_result(GenerationResult(**row['result']), spec, job)
            member_jobs[alias][qid] = job
            member_rows[alias][qid] = row
        require(set(member_jobs[alias]) == set(allowed), 'Qualification member question coverage differs')

    qwen3 = specs['Qwen3-8B']
    require(qwen3.model_id == 'Qwen/Qwen3-8B' and qwen3.revision == QWEN3_REVISION,
            'Original pinned Qwen3 identity differs')
    config = dict(original_config)
    config.update(model_id=qwen3.model_id, revision=qwen3.revision, model_path=qwen3.model_path,
                  max_input_tokens=6144, max_new_tokens=2048, temperature=0.0, top_p=1.0,
                  enable_thinking=False)
    candidate = ModelSpec(**config)
    candidate.validate()
    tokenizer = AutoTokenizer.from_pretrained(candidate.model_path, local_files_only=True, trust_remote_code=False)
    team = [entry['alias'] for entry in pool[:3]]
    require(len(set(team)) == 3 and all(set(job['members']) == set(team) for job in original_jobs),
            'Original runtime qualification K3 team differs')
    jobs = []
    for assigned in qualification:
        qid = assigned['question_id']
        provenance = {}
        for alias in team:
            row = member_rows[alias][qid]
            spec = specs[alias]
            provenance[alias] = {'job_id': row['job_id'], 'record_sha256': row['record_sha256'],
                                 'model_id': spec.model_id, 'revision': spec.revision,
                                 'model_fingerprint': row['model_fingerprint']}
        job = make_job(question_id=qid, task=assigned['task'], subtask=assigned['subtask'],
                       question=questions[qid], answers={alias: member_rows[alias][qid]['result']['text'] for alias in team},
                       seed=42, model_spec=candidate, split='qualification', member_provenance=provenance,
                       tokenizer=tokenizer, source_token_cap=512, head_tokens=128, tail_tokens=384)
        require(set(job) == JOB_FIELDS, 'New job schema differs')
        validate_job(job, candidate, require_token_policy=True)
        jobs.append(job)
    require(len({job['job_id'] for job in jobs}) == 8 and Counter(job['task'] for job in jobs) == {'gsm8k': 4, 'bbh': 4},
            'Real qualification must cover four unique questions per task')

    config_path = root / 'configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json'
    jobs_path = root / 'data/jobs/decision_feedback_v2/s1_qualification/aggregator_qwen3_8b_candidate_v1.jsonl'
    output_dir = root / 'results/decision_feedback_v2/s1_qualification'
    raw_path = output_dir / 'aggregator_qwen3_8b_candidate_v1.jsonl'
    receipt_path = output_dir / 'qualification_review_v1.json'
    write_immutable(config_path, json_bytes(config))
    write_immutable(jobs_path, b''.join((json.dumps(job, ensure_ascii=False, allow_nan=False) + '\n').encode() for job in jobs))

    roles = {}
    for role, (old_split, count) in ROLE_SPLITS.items():
        records = [dict(row, canonical_question_sha256=manifest['canonical_question_fingerprints'][row['question_id']])
                   for row in manifest['records'] if row['split'] == old_split]
        require(len(records) == count, 'Existing development role count differs: ' + role)
        roles[role] = {'old_split': old_split, 'count': count, 'task_counts': dict(Counter(row['task'] for row in records)),
                       'status': 'EXISTING_DEVELOPMENT_METADATA', 'records': records}
    source_audit = read(source('artifacts/audit/split_audit.json'))
    available = read(source('artifacts/audit/llmrouterbench/available_records_receipt.json'))
    remaining = {name: count - source_audit['checks']['bbh_subtask_total_counts'][name]
                 for name, count in available['bbh_subtask_mapping']['subtask_counts'].items()}
    require(len(remaining) == 27 and sum(remaining.values()) == 424 and min(remaining.values()) >= 8,
            'Metadata no longer supports balanced new BBH200')
    final = [row for row in manifest['records'] if row['split'] == 'sealed_test']
    require(len(final) == 200, 'Original final exclusion metadata differs')
    locked_ids = sorted(row['question_id'] for row in manifest['records'])
    locked_hashes = sorted(manifest['canonical_question_fingerprints'].values())
    proposal = {'schema': 'decision_feedback_v2_development_partition_proposal_v1', 'status': 'EXISTING_ROLES_READY_NEW400_NOT_SELECTED',
                'K': 3, 'all_prior_experiment_questions_are_development': True, 'roles': roles,
                'new_development_holdout': {'status': 'NOT_SELECTED', 'target_count': 400, 'target_task_counts': {'gsm8k': 200, 'bbh': 200},
                                          'question_ids': [], 'bbh_subtasks': 27, 'bbh_quota': '11 subtasks x8 plus16 x7; score-blind seeded choice',
                                          'bbh_remaining_per_subtask': remaining, 'required_exclusions': 'all1308 locked IDs and canonical hashes plus any additional exposed question'},
                'all_locked_exclusion_metadata': {'count': len(locked_ids), 'question_ids': locked_ids,
                                                  'canonical_question_sha256': locked_hashes},
                'sealed_final_exclusion': {'count': 200, 'question_ids_sha256': digest(sorted(row['question_id'] for row in final)),
                                          'canonical_hashes_sha256': digest(sorted(manifest['canonical_question_fingerprints'][row['question_id']] for row in final)),
                                          'role': 'metadata_exclusion_only'},
                'feature_fit_scope': 'fit new PCA/scalers only on selector_train600; validation200/newholdout400 transform only',
                'generation_replication_scope': 'same canonical question remains in one role across all real generation replicas; stream seeds are separate',
                'raw_question_source_files_read': False, 'sealed_question_text_read': False, 'gold_fields_written': False,
                'source_metadata_sha256': inputs['data/derived/split_manifest.json']}
    proposal_path = root / 'artifacts/private/routing_redesign_20261104/PARTITION_PROPOSAL_v2.json'
    write_immutable(proposal_path, json_bytes(proposal))
    prepared = {'schema': 'decision_feedback_v2_aggregator_qualification_preparation_v1', 'status': 'PREPARED_NOT_RUN',
                'script_sha256': sha(__file__), 'config': {'path': config_path.relative_to(root).as_posix(), 'sha256': sha(config_path),
                                                        'model_fingerprint': configuration_fingerprint(candidate)},
                'jobs': {'path': jobs_path.relative_to(root).as_posix(), 'sha256': sha(jobs_path), 'count': 8,
                         'task_counts': {'gsm8k': 4, 'bbh': 4}, 'job_ids': [job['job_id'] for job in jobs], 'schema_fields': sorted(JOB_FIELDS)},
                'source_sha256': inputs, 'partition_proposal': {'path': proposal_path.relative_to(root).as_posix(), 'sha256': sha(proposal_path)},
                'K': 3, 'team': team, 'candidate_token_policy': {'source_token_cap': 512, 'head_tokens': 128, 'tail_tokens': 384},
                'original_prompt_template_reused': True, 'old_historical_answers_reused': False,
                'real_qualification_member_answers_reused': True, 'tokenizer': {'class': type(tokenizer).__name__, 'model_revision': candidate.revision},
                'planned_output': raw_path.relative_to(root).as_posix(), 'planned_review_receipt': receipt_path.relative_to(root).as_posix(),
                'original_runner_command': [str(root / '.venv/Scripts/python.exe'), '-B', '-X', 'utf8', str(root / 'scripts/run_aggregation.py'),
                                            '--jobs', str(jobs_path), '--model-config', str(config_path), '--output', str(raw_path),
                                            '--qualification', '--max-jobs', '8'],
                'qualification_review_status': 'NOT_RUN', 'heterogeneous_all20_multigeneration_stage1_status': 'NOT_RUN',
                'heterogeneous_generation_cache_replicates_prepared': 0, 'new_model_calls': 0, 'scientific_completion': False,
                'missing_before_execution': ['root server runtime/mapping qualification for this exact candidate fingerprint'],
                'missing_before_formal_stage1': ['review real technical outputs', 'freeze operator after aggregator-calibration100 review',
                                                'prepare all20 heterogeneous teams and genuine generation replicas'],
                'new400_status': 'NOT_SELECTED'}
    preparation_path = output_dir / 'QUALIFICATION_PREPARATION_MANIFEST_v1.json'
    write_immutable(preparation_path, json_bytes(prepared))
    print(json.dumps({'status': prepared['status'], 'jobs': 8, 'task_counts': prepared['jobs']['task_counts'],
                      'model_fingerprint': configuration_fingerprint(candidate), 'manifest': str(preparation_path),
                      'new_model_calls': 0, 'new400_status': 'NOT_SELECTED'}, ensure_ascii=False))


if __name__ == '__main__':
    main()