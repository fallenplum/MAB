"""Locally review eight server qualification outputs, with qualification-only gold projection."""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_SHA = 'f26468547582b3cf0cbcc178f7bfa52a1dcbfa689003ea955ccd74fda1fabacd'
INPUT_NAMES = {'JOB_ID_MAP.json', 'QUALIFICATION_PLAN.json', 'logical_config.json', 'logical_jobs.jsonl',
               'physical_config.json', 'physical_jobs.jsonl', 'run_aggregator_qualification_v2.py'}
EXPORT_NAMES = {'EXPORT_MANIFEST.json', 'outputs.jsonl', 'status.json', 'journal_snapshot.sqlite3'} | {'inputs/' + name for name in INPUT_NAMES}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def write_immutable(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, 'Existing reviewed artifact differs: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())


def unpack(archive, expected_sha, destination, local_bundle):
    require(sha(archive) == expected_sha, 'Downloaded archive SHA differs')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        require(len(members) == len(EXPORT_NAMES) and {item.name for item in members} == EXPORT_NAMES,
                'Export member allowlist or uniqueness differs')
        require(all(item.isfile() and not item.issym() and not item.islnk() and 0 <= item.size <= 4 * 1024 * 1024
                    for item in members) and sum(item.size for item in members) < 16 * 1024 * 1024, 'Unsafe export member type/size')
        payloads = {item.name: tar.extractfile(item).read() for item in members}
        require(all(len(payloads[item.name]) == item.size for item in members), 'Export member length differs')
    manifest = json.loads(payloads['EXPORT_MANIFEST.json'])
    require(manifest['schema'] == 'v2_technical_qualification_closed_export_v1' and
            set(manifest['files']) == EXPORT_NAMES - {'EXPORT_MANIFEST.json'}, 'Export manifest schema/inventory differs')
    for name, item in manifest['files'].items():
        require(item['bytes'] == len(payloads[name]) and item['sha256'] == hashlib.sha256(payloads[name]).hexdigest(),
                'Export file hash differs: ' + name)
    for name in INPUT_NAMES:
        require(payloads['inputs/' + name] == (local_bundle / name).read_bytes(), 'Returned input differs: ' + name)
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        target = destination / name
        require(target.resolve().is_relative_to(destination.resolve()), 'Export path escaped destination')
        write_immutable(target, data)
    return manifest


def qualification_references(root, questions, private):
    """Physically stream JSONL bytes, projecting payloads only after the explicit qid allowlist."""
    header = re.compile(rb'^\s*\{\s*"question_id"\s*:\s*"([^"\\]+)"\s*[,}]')
    audit = read(root / 'artifacts/audit/split_audit.json')
    expected = {entry['path']: entry for entry in audit['source_files'] if entry['path'] in
                {'data/derived/available_records.jsonl', 'data/derived/gsm8k_questions.jsonl'}}
    references = {}
    scans = []
    for relative, task in [('data/derived/available_records.jsonl', 'bbh'), ('data/derived/gsm8k_questions.jsonl', 'gsm8k')]:
        count = selected = 0
        checksum = hashlib.sha256()
        with (root / relative).open('rb') as stream:
            for line in stream:
                checksum.update(line)
                if not line.strip():
                    continue
                count += 1
                match = header.match(line.removeprefix(b'\xef\xbb\xbf'))
                require(match is not None, 'Source qid metadata must precede payload fields')
                qid = match.group(1).decode('ascii')
                if qid not in questions:
                    continue
                require(qid not in references, 'Duplicate qualification reference ID')
                row = json.loads(line)
                require(row['question_id'] == qid and row['task'] == task and
                        row['question'] == questions[qid]['question'] and row['subtask'] == questions[qid]['subtask'],
                        'Qualification reference identity/question differs')
                references[qid] = {key: row.get(key) for key in
                                   ('question_id', 'task', 'subtask', 'source_split', 'question', 'reference', 'choices')}
                selected += 1
        require(checksum.hexdigest() == expected[relative]['sha256'] and count == expected[relative]['records'],
                'Normalized source hash/count differs from original split audit')
        scans.append({'path': relative, 'sha256': checksum.hexdigest(), 'physical_rows_streamed': count,
                      'qualification_payload_rows_parsed': selected, 'other_payload_rows_not_parsed': count - selected})
    require(set(references) == set(questions) and len(references) == 8, 'Exactly eight qualification references required')
    path = private / 'derived/decision_feedback_v2/qualification_reference_projection_v1.jsonl'
    write_immutable(path, b''.join((json.dumps(references[qid], ensure_ascii=False, allow_nan=False) + '\n').encode() for qid in sorted(references)))
    receipt = {'schema': 'qualification_only_reference_projection_v1', 'status': 'EXACT_8_QUALIFICATION_PAYLOADS_PROJECTED',
               'source_scans': scans, 'question_ids': sorted(references), 'projected_rows': 8,
               'projection_sha256': sha(path), 'physical_source_scan_includes_other_and_sealed_bytes': True,
               'semantic_question_reference_access_scope': 'only8 explicit original qualification IDs',
               'sealed_question_payloads_parsed': 0, 'sealed_references_scored': 0, 'load_inputs_called': False}
    write_immutable(path.with_suffix('.projection_receipt.json'), encoded(receipt))
    return references, {'path': str(path), **receipt}


def review(root, archive, archive_sha):
    sys.path.insert(0, str(root / 'src'))
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                      OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    from budget_router.inference import ModelSpec, GenerationResult, configuration_fingerprint, chat_template_text
    from budget_router.execution import validate_job, make_job, _validate_result
    from budget_router.member_execution import MemberStore, validate_member_job
    from budget_router.preparation import validate_success_rows
    from budget_router.scoring import score_answer, SCORER_VERSION
    from transformers import AutoTokenizer

    private = root / 'artifacts/private/routing_redesign_20261104'
    bundle = root / 'artifacts/private/server_migration_20261102/routing_v2_qualification_20261104_r2'
    returned = bundle / 'returned_qualification_outputs_v1'
    export = unpack(archive, archive_sha, returned, bundle)
    plan = read(bundle / 'QUALIFICATION_PLAN.json')
    plan_sha = sha(bundle / 'QUALIFICATION_PLAN.json')
    receipt = read(bundle / 'BUNDLE_RECEIPT.json')
    require(plan_sha == receipt['plan_sha256'] == export['plan_sha256'], 'Trusted plan binding differs')
    for name, expected in plan['files_sha256'].items():
        require(sha(bundle / name) == expected, 'Trusted input hash differs: ' + name)
    require(sha(bundle / 'run_aggregator_qualification_v2.py') == plan['worker_sha256'], 'Worker source differs')
    for name, expected in plan['original_source_sha256'].items():
        require(sha(root / name) == expected, 'Original scientific source differs: ' + name)
    logical = ModelSpec(**read(bundle / 'logical_config.json'))
    physical = ModelSpec(**read(bundle / 'physical_config.json'))
    logical.validate()
    physical.validate()
    require({key: value for key, value in asdict(logical).items() if key != 'model_path'} ==
            {key: value for key, value in asdict(physical).items() if key != 'model_path'}, 'Non-path physical spec difference')
    require(physical.temperature == 0 and physical.max_new_tokens == 2048 and physical.max_input_tokens == 6144 and
            physical.enable_thinking is False and configuration_fingerprint(logical) == plan['logical_fingerprint'] and
            configuration_fingerprint(physical) == plan['physical_fingerprint'], 'Frozen candidate protocol differs')
    logical_jobs = rows(bundle / 'logical_jobs.jsonl')
    physical_jobs = rows(bundle / 'physical_jobs.jsonl')
    bridge = read(bundle / 'JOB_ID_MAP.json')['jobs']
    require(len(logical_jobs) == len(physical_jobs) == len(bridge) == 8, 'Eight source jobs required')
    for original, job, entry in zip(logical_jobs, physical_jobs, bridge):
        validate_job(original, logical, require_token_policy=True)
        validate_job(job, physical, require_token_policy=False)
        require(job['split'] == original['split'] == 'qualification' and len(job['members']) == 3 and
                entry['logical_job_id'] == original['job_id'] and entry['physical_job_id'] == job['job_id'] and
                entry['logical_full_job_sha256'] == digest(original) and entry['physical_full_job_sha256'] == digest(job), 'Job bridge differs')
        expected = json.loads(json.dumps(original))
        expected['model_fingerprint'] = plan['physical_fingerprint']
        expected['candidate_token_policy']['tokenizer_model_fingerprint'] = plan['physical_fingerprint']
        expected['candidate_token_policy']['tokenizer_identity']['name_or_path'] = physical.model_path
        expected['job_id'] = digest({key: value for key, value in expected.items() if key != 'job_id'})
        require(expected == job, 'Unexpected physical job modification')
    require(len({job['job_id'] for job in physical_jobs}) == 8, 'Duplicate source jobs')
    metadata = read(root / 'data/derived/split_manifest.json')
    qualification = {row['question_id']: row for row in metadata['records'] if row['split'] == 'qualification'}
    require(set(qualification) == {job['question_id'] for job in logical_jobs}, 'Jobs differ from original qualification IDs')
    pool = read(root / 'configs/pool_v1.json')['models']
    team = logical_jobs[0]['members']
    answers = {qid: {} for qid in qualification}
    provenance = {qid: {} for qid in qualification}
    questions = {}
    member_sources = {}
    for entry in pool:
        if entry['alias'] not in team:
            continue
        alias = entry['alias']
        spec = ModelSpec(**read(root / entry['config']))
        jobs_path = root / 'data/jobs/qualification' / ('members_' + alias.lower() + '.jsonl')
        raw = root / 'results/raw/qualification' / ('members_' + alias.lower() + '.jsonl')
        require(len(validate_success_rows(raw)) == 8, 'Member source is incomplete')
        store = MemberStore(raw)
        for job in rows(jobs_path):
            validate_member_job(job, spec)
            qid = job['question_id']
            require(qid in qualification and job['split'] == 'qualification', 'Member source crossed qualification scope')
            row = store._cached(job, spec)
            require(row is not None, 'Member cache replay failed')
            _validate_result(GenerationResult(**row['result']), spec, job)
            if qid in questions:
                require(questions[qid]['question'] == job['question'], 'Question differs between members')
            questions[qid] = {'question': job['question'], 'subtask': job['subtask']}
            answers[qid][alias] = row['result']['text']
            provenance[qid][alias] = {'job_id': row['job_id'], 'record_sha256': row['record_sha256'],
                                     'model_id': spec.model_id, 'revision': spec.revision, 'model_fingerprint': row['model_fingerprint']}
        member_sources[str(raw.relative_to(root))] = sha(raw)
        member_sources[str(jobs_path.relative_to(root))] = sha(jobs_path)
    tokenizer = AutoTokenizer.from_pretrained(logical.model_path, local_files_only=True, trust_remote_code=False)
    prompt_lengths = {}
    for original in logical_jobs:
        qid = original['question_id']
        rebuilt = make_job(question_id=qid, task=original['task'], subtask=original['subtask'],
                           question=questions[qid]['question'], answers=answers[qid], seed=original['seed'],
                           model_spec=logical, split='qualification', tokenizer=tokenizer,
                           member_provenance=provenance[qid], source_token_cap=512, head_tokens=128, tail_tokens=384)
        require(rebuilt == original, 'Prompt/job differs from real qualification question/member reconstruction')
        prompt_lengths[qid] = len(tokenizer(chat_template_text(tokenizer, original['messages'], logical), add_special_tokens=False)['input_ids'])
    references, reference_projection = qualification_references(root, questions, private)
    for qid, item in references.items():
        check = score_answer(item['task'], item['reference'], item['reference'], subtask=item['subtask'], choices=item['choices'])
        require(check.parsed and check.score == 1, 'Qualification reference does not roundtrip through original scorer')

    status = read(returned / 'status.json')
    raw = returned / 'outputs.jsonl'
    result_rows = rows(raw)
    context = status['context']
    require(status['status'] == export['qualification_status'] == 'TECHNICAL_QUALIFICATION_OUTPUTS_RECORDED_PENDING_REVIEW' and
            status['successful_rows'] == status['actual_attempt_rows'] == status['requested_jobs'] == 8 and
            status['unknown_cost_rows'] == 0 and status['exit_code'] == 0 and status['output_sha256'] == sha(raw), 'Server terminal status differs')
    require(status['plan_sha256'] == plan_sha and status['worker_sha256'] == plan['worker_sha256'] and
            status['context_sha256'] == digest(context) and context['actual_model_spec'] == asdict(physical) and
            context['logical_fingerprint'] == plan['physical_fingerprint'] and
            context['original_logical_fingerprint'] == plan['logical_fingerprint'] and context['plan_sha256'] == plan_sha,
            'Server model/runtime context differs')
    require(len(result_rows) == 8 and len({row['attempt_id'] for row in result_rows}) == 8 and
            {row['job_id'] for row in result_rows} == {job['job_id'] for job in physical_jobs}, 'Result job/attempt coverage differs')
    database = sqlite3.connect((returned / 'journal_snapshot.sqlite3').resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        require(database.execute("SELECT value FROM metadata WHERE key='context_sha256'").fetchone() == (digest(context),), 'Journal context differs')
        calls = database.execute('SELECT attempt_id,job_id,request_sha,status,response_json,response_sha,local_record_sha,started,finished FROM calls').fetchall()
        events = [json.loads(row[0]) for row in database.execute('SELECT event_json FROM events ORDER BY id')]
    finally:
        database.close()
    require(len(calls) == 8 and all(row[3] == 'SUCCESS' for row in calls), 'Journal is incomplete, failed or unknown')
    by_attempt = {row[0]: row for row in calls}
    by_job = {job['job_id']: job for job in physical_jobs}
    by_bridge = {entry['physical_job_id']: entry for entry in bridge}
    usage = {'prompt_tokens': 0, 'completion_tokens': 0}
    wall = 0.0
    scores = []
    commits = {}
    for row in result_rows:
        require(row['schema'] == 'v2_physical_qualification_generation_record_v1' and row['status'] == 'success' and
                row['split'] == 'qualification' and row['cost_status'] == 'recorded_actual_usage' and
                row['record_sha256'] == digest({key: value for key, value in row.items() if key != 'record_sha256'}), 'Outer record integrity differs')
        job = by_job[row['job_id']]
        entry = by_bridge[row['job_id']]
        require(row['logical_job_id'] == entry['logical_job_id'] and row['question_id'] == job['question_id'] and
                row['task'] == job['task'] and row['plan_sha256'] == plan_sha and row['context_sha256'] == digest(context), 'Outer source binding differs')
        payload = {'attempt_id': row['attempt_id'], 'job_id': job['job_id'], 'model_fingerprint': plan['physical_fingerprint'],
                   'split': job['split'], 'seed': job['seed'], 'messages': job['messages'], 'job_sha256': digest(job)}
        call = by_attempt[row['attempt_id']]
        body = row['response']
        require(call[1] == job['job_id'] and call[2] == row['request_sha256'] == digest(payload) and
                body == json.loads(call[4]) and digest(body) == call[5] and call[6] == row['record_sha256'], 'Request/response/local journal commit differs')
        require(body['status'] == 'SUCCESS' and body['attempt_id'] == row['attempt_id'] and body['job_id'] == job['job_id'] and
                body['request_sha256'] == digest(payload) and body['model_fingerprint'] == plan['physical_fingerprint'] and
                body['context_sha256'] == digest(context), 'Response protocol binding differs')
        result = GenerationResult(**body['result'])
        _validate_result(result, physical, job)
        require(math.isfinite(result.wall_time_s) and result.wall_time_s > 0 and
                result.prompt_tokens == prompt_lengths[job['question_id']], 'Measured wall/prompt-token reconstruction differs')
        for key in usage:
            usage[key] += body['result'][key]
        wall += result.wall_time_s
        reference = references[job['question_id']]
        scored = score_answer(job['task'], result.text, reference['reference'], subtask=job['subtask'], choices=reference['choices'])
        scores.append({'question_id': job['question_id'], 'task': job['task'], 'subtask': job['subtask'],
                       'score': scored.score, 'parsed': scored.parsed, 'parse_reason': scored.reason,
                       'nonempty': bool(result.text.strip()), 'completion_tokens': result.completion_tokens,
                       'completion_reached_cap': result.completion_reached_cap, 'finish_reason': result.finish_reason})
        commits[row['attempt_id']] = row['record_sha256']
    require(usage == status['actual_usage'] and abs(wall - status['model_call_wall_seconds_sum']) < 1e-8 and
            max(prompt_lengths.values()) == status['maximum_prompt_tokens'], 'Usage/wall totals differ')
    initial = [event for event in events if event['payload'].get('boundary') == 'post_backend_initialization']
    success_events = [event for event in events if event['payload'].get('boundary') == 'post_persisted_success']
    require(len(events) == 9 and len(initial) == 1 and len(success_events) == 8 and
            {event['payload']['attempt_id'] for event in success_events} == set(commits), 'Journal housekeeping/commit inventory differs')
    require(all(event['payload']['local_record_sha256'] == commits[event['payload']['attempt_id']] and
                event['result']['context_sha256'] == digest(context) for event in success_events), 'Safe post-commit journal event differs')
    issues = [row['question_id'] for row in scores if not row['parsed'] or not row['nonempty'] or
              row['completion_reached_cap'] is not False or row['completion_tokens'] >= physical.max_new_tokens or
              row['finish_reason'] != 'model_stopped']
    output = root / 'results/decision_feedback_v2/s1_qualification/server_physical_qualification_outputs_r2.jsonl'
    write_immutable(output, raw.read_bytes())
    by_task = {task: {'questions': sum(row['task'] == task for row in scores),
                      'correct': sum(row['score'] for row in scores if row['task'] == task),
                      'parsed': sum(row['parsed'] for row in scores if row['task'] == task)} for task in ('gsm8k', 'bbh')}
    report = {'schema': 'decision_feedback_v2_server_aggregator_qualification_review_v1',
              'status': 'REQUIRES_REVIEW_OUTPUT_FORMAT_OR_CAP' if issues else 'PASS', 'scope': 'server technical qualification only',
              'reviewer_execution': 'local CPU', 'generation_execution': 'remote matched Linux A6000 worker',
              'reviewer_source_sha256': sha(__file__), 'archive_sha256': archive_sha,
              'export_manifest_sha256': sha(returned / 'EXPORT_MANIFEST.json'), 'plan_sha256': plan_sha,
              'logical_fingerprint': plan['logical_fingerprint'], 'physical_fingerprint': plan['physical_fingerprint'],
              'logical_config_sha256': sha(bundle / 'logical_config.json'), 'physical_config_sha256': sha(bundle / 'physical_config.json'),
              'worker_source_sha256': plan['worker_sha256'], 'server_pid': status['pid'], 'server_starttime_ticks': status['starttime_ticks'],
              'server_gpu': status['gpu'], 'runtime_versions': status['runtime_versions'],
              'successful_rows': 8, 'actual_attempt_rows': 8, 'unknown_cost_rows': 0, 'actual_usage': usage,
              'model_call_wall_seconds_sum': wall, 'maximum_prompt_tokens': max(prompt_lengths.values()),
              'correct': sum(row['score'] for row in scores), 'questions': 8, 'accuracy': sum(row['score'] for row in scores) / 8,
              'by_task': by_task, 'finish_reason_counts': dict(Counter(row['finish_reason'] for row in scores)),
              'completion_cap_hits': sum(row['completion_reached_cap'] is True for row in scores), 'format_review_question_ids': issues,
              'accuracy_is_not_a_technical_acceptance_threshold': True, 'per_question': scores,
              'scorer_version': SCORER_VERSION, 'scorer_source_sha256': sha(root / 'src/budget_router/scoring.py'),
              'prompt_reconstruction': {'status': 'EXACT_ORIGINAL_MAKE_JOB_MATCH', 'questions': 8,
                                        'sources': member_sources, 'gold_substring_heuristic_used': False},
              'reference_projection': reference_projection, 'journal_binding': '8 exact request/response/local records plus9 safe housekeeping events',
              'server_output_raw': {'path': str(output), 'sha256': sha(output), 'schema': 'v2_physical_qualification_generation_record_v1', 'original_bytes_preserved': True},
              'scientific_completion': False, 'full_stage1_completion': False, 'new_model_calls_by_this_reviewer': 0}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--archive-sha256', default=ARCHIVE_SHA)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    archive = args.archive or root / 'artifacts/private/server_migration_20261102/routing_v2_qualification_20261104_r2/qualification_outputs_v1.tar.gz'
    report = review(root, archive, args.archive_sha256)
    target = root / 'results/decision_feedback_v2/s1_qualification/QUALIFICATION_REVIEW_v1.json'
    write_immutable(target, encoded(report))
    print(json.dumps({'status': report['status'], 'correct': report['correct'], 'questions': 8,
                      'by_task': report['by_task'], 'finish_reason_counts': report['finish_reason_counts'],
                      'completion_cap_hits': report['completion_cap_hits'], 'actual_usage': report['actual_usage'],
                      'model_call_wall_seconds_sum': report['model_call_wall_seconds_sum'],
                      'report': str(target), 'report_sha256': sha(target), 'new_model_calls': 0}, ensure_ascii=False))
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())