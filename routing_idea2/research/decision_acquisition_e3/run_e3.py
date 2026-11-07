"""Leakage-safe prequential E3 replay, immutable per-path receipts, no LLM calls."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
WORK = ROOT / 'research/decision_acquisition_e3'
PUBLIC = WORK / 'public'
PRIVATE = WORK / 'private'
sys.path.insert(0, str(ROOT / 'src'))
from assets import load_assets, make_splits, Env, TEAMS, SEEDS, algorithm_seed, sha256, priced_units
from predictors import ModelTemplate, qualification as model_qualification
from acquisition import projected_scores, choose_candidate, BudgetPacing, fantasy_table
from budget_router_v2.acquisition import decision_value

CONFIG = {
    'schema': 'decision_acquisition_e3_prequential_v1', 'N': 6, 'K': 3, 'actions': 20,
    'alpha': 100., 'pca_components': 16, 'history_sizes': [120, 240, 480], 'main_history_size': 240,
    'folds': 5, 'generation_seeds': [42, 137, 2024], 'order_seeds': [17, 29, 43],
    'gold_budgets': [10, 20, 40, 60], 'primary_gold_budget': 20,
    'primary_predictor': 'P2', 'predictors': ['P1', 'P2', 'P3'],
    'fantasy_pairs': 8, 'reference_count': 54, 'warmup_gold': 2,
    'pacing': 'equal-width gold slots; observe first third, then first record, mandatory last; Random uniform per slot',
    'feedback_delay': 0, 'noise': '.5 I + .5 history-only within-question generation correlation; unit diagonal',
    'historical_likelihood': 'independent unit-noise original Ridge; frozen prior for correlated online updates',
    'common_algorithm_random_numbers_across_generation_seeds': True,
    'matched_random_scope': 'source-KG-schedule-conditioned random-neighbor diagnostic; not an independent same-permission deployable policy',
    'primary_comparison': 'P2/B1/decision_cost - ig_cost at gold20,history240',
    'secondary_comparisons': ['P2/B1/decision_cost - random', 'P2/B1/decision_cost - boundary_cost',
                              'P2/B2/decision_cost - ig', 'P2/B2/decision_cost - random_neighbor'],
    'quality_target_rule': 'No-feedback accuracy + .005 absolute, fixed before replay; unreachable is censored',
    'delete_groups': 20, 'new_llm_calls': 0,
    'sensitivity': 'history120/480 at gold20 for B1 and B2 Random,IG,DecisionCost; primary mechanism grid history240',
    'multiplicity': 'Holm over primary plus four declared secondary contrasts; descriptive trace intervals separately marked',
}
NEIGHBORS = {i: [j for j, other in enumerate(TEAMS) if len(set(team) ^ set(other)) == 2]
             for i, team in enumerate(TEAMS)}
_DATA = None
_SPLITS = None
_TEMPLATES = {}


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(pending, path)


def signature():
    files = ['run_e3.py', 'assets.py', 'predictors.py', 'acquisition.py']
    content = {'config': CONFIG, 'code': {name: sha256(Path(__file__).parent / name) for name in files}}
    content['v2_kg_sha256'] = sha256(ROOT / 'src/budget_router_v2/acquisition.py')
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest(), content


def initialize_worker():
    global _DATA, _SPLITS
    _DATA, _ = load_assets(ROOT)
    _SPLITS = make_splits(_DATA)


def template_for(meta, deleted=None):
    fold = _SPLITS['folds'][meta['fold']]
    history = fold['nested'][meta['history_size']]
    references = fold['references'][meta['history_size']]
    if deleted is not None:
        history = history[~np.isin(history, deleted)]
        references = references[~np.isin(references, deleted)]
    key = (meta['fold'], meta['history_size'], meta['predictor'], meta.get('delete_group'))
    if key not in _TEMPLATES:
        _TEMPLATES[key] = ModelTemplate(_DATA, history, meta['predictor'], references, CONFIG)
    weights = np.zeros(len(references))
    tasks = _DATA['task'][references]
    for task in ('gsm8k', 'bbh'):
        mask = tasks == task
        weights[mask] = .5 / mask.sum()
    return _TEMPLATES[key], weights


def predictions(posterior, rows):
    template = posterior.template
    coefficient = posterior.mean.reshape(template.question_dimension, template.action_dimension)
    return template.X[rows] @ coefficient @ template.B.T


def episode(train, template, order, meta, weights, matched=None, audit_future=True):
    run_clock = time.process_time()
    stage, method = meta['stage'], meta['method']
    diagnostic = method == 'full_information'
    posterior = template.clone(independent_noise=method == 'independent_noise')
    env = Env(train, template.history, order, SEEDS.index(meta['gen']), meta['B'], diagnostic=diagnostic)
    seed = algorithm_seed(meta['fold'], meta['order'], 0)
    rng = np.random.default_rng(seed)
    pacing = BudgetPacing(len(order), meta['B'], seed, CONFIG['warmup_gold'])
    frozen_gold = BudgetPacing(len(order), meta['B'], seed, CONFIG['warmup_gold'])
    fantasies = fantasy_table(seed, CONFIG['fantasy_pairs'])
    baseline = np.argmax(predictions(template.clone(), order), axis=1)
    traces, feedback = [], []
    gen = SEEDS.index(meta['gen'])
    for tick, q in enumerate(order):
        q = int(q)
        main = int(np.argmax(posterior.predict(q)))
        env.service(q, main)
        clock = time.process_time()
        choices, score, chosen_kg = [main], 0., 0.
        purchase = False
        candidate_count = 0
        effective_method = method
        if method in ('independent_noise', 'random_replacement', 'no_cost', 'main_only'):
            effective_method = 'decision' if method == 'no_cost' else 'decision_cost'
        if method == 'none':
            purchase = False
        elif method == 'full_observable':
            purchase = True
        elif diagnostic:
            purchase, choices = True, list(range(20))
        elif stage == 'A':
            purchase = pacing.decide(tick, 0., random=True)
        elif method == 'matched_random':
            event = matched[tick]
            purchase = event['purchase']
            if event['comparison']:
                choices = [main, int(rng.choice(NEIGHBORS[main]))]
        elif stage == 'B2' and not frozen_gold.decide(tick, 0., random=True):
            purchase = False
        elif stage != 'B2' and not pacing.eligible(tick):
            purchase = False
        elif tick < CONFIG['warmup_gold']:
            purchase = True
        else:
            include_neighbors = stage in ('B2', 'J', 'D') and method != 'main_only'
            if effective_method.startswith('random'):
                packages = [[main]] + [[main, neighbor] for neighbor in NEIGHBORS[main]] if include_neighbors else [[main]]
                values = [{'actions': package, 'kg': 0., 'extra_kg': 0.} for package in packages]
            else:
                values, _ = projected_scores(posterior, q, main, NEIGHBORS[main], weights, fantasies,
                                             include_neighbors, need_kg=effective_method.startswith('decision'))
            candidates = values[1:] if stage == 'B2' else values
            estimates = []
            for candidate in candidates:
                expected = env.predict_incremental(candidate['actions'], env.executed_current(), include_gold=stage != 'B2')
                if stage == 'B2':
                    expected['priced_units'] -= .001  # main scoring is a common sunk comparison cost
                estimates.append(expected['priced_units'])
            picked, score = choose_candidate(candidates, effective_method, estimates, rng, stage)
            choices = candidates[picked]['actions']
            chosen_kg = candidates[picked]['extra_kg' if stage == 'B2' else 'kg']
            candidate_count = len(candidates)
            purchase = True if stage == 'B2' else pacing.decide(tick, score, random=effective_method == 'random')
            if method == 'random_replacement' and len(choices) == 2:
                choices = [main, int(rng.choice(NEIGHBORS[main]))]
        planning_cpu = time.process_time() - clock
        update_cpu = 0.
        observed, incremental, future = [], None, None
        if purchase:
            remaining = np.asarray(order[tick+1:], dtype=int)
            before_actions = np.argmax(predictions(posterior, remaining), axis=1) if audit_future and len(remaining) else None
            arrival = env.buy(q, choices, diagnostic=diagnostic)
            purchase = arrival['accepted']
            incremental = arrival['resources']
            if arrival['observations']:
                observed = arrival['observations']
                clock = time.process_time()
                posterior.update(q, [item[0] for item in observed], [item[1] for item in observed])
                update_cpu = time.process_time() - clock
                feedback.append({'tick': tick, 'q': q, 'source_gen': meta['gen'],
                                 'actions': [item[0] for item in observed], 'raw_scores': [item[1] for item in observed],
                                 'received_before_update': True, 'resources': incremental})
                if before_actions is not None:
                    after_actions = np.argmax(predictions(posterior, remaining), axis=1)
                    before_reward = train['Y'][remaining, gen, before_actions]
                    after_reward = train['Y'][remaining, gen, after_actions]
                    future = {'remaining_questions': len(remaining), 'changed': int(np.sum(before_actions != after_actions)),
                              'corrected': int(np.sum(after_reward > before_reward)), 'harmed': int(np.sum(after_reward < before_reward)),
                              'net_reward': int(np.sum(after_reward - before_reward)),
                              'scope': 'auditor_only_frozen_snapshot_comparison_without_later_feedback_updates'}
        resource = env.ledger['total']
        traces.append({'tick': tick, 'q': q, 'task': str(train['task'][q]), 'sub': str(train['sub'][q]),
                       'main': main, 'no_feedback_main': int(baseline[tick]),
                       'reward': env.evaluation[-1]['quality'],
                       'no_feedback_reward': float(train['Y'][q, gen, baseline[tick]]),
                       'fixed_reward': float(train['Y'][q, gen, template.fixed_actions[q]]),
                       'oracle_reward': float(train['Y'][q, gen].max()),
                       'service_frozen_before_feedback': True, 'purchase': purchase, 'comparison': len(observed) == 2,
                       'actions': choices if purchase else [], 'predicted_score': score, 'chosen_kg': chosen_kg,
                       'candidate_count': candidate_count, 'plan_cpu_s': planning_cpu, 'update_cpu_s': update_cpu,
                       'cumulative_gold': resource['gold'], 'cumulative_member_calls': resource['member_calls'],
                       'cumulative_aggregate_calls': resource['aggregator_calls'], 'cumulative_tokens': resource['tokens'],
                       'ledger': {key: dict(value) for key, value in env.ledger.items()},
                       'future_feedback_influence': future})
    assert env.summary()['served_questions'] == len(order)
    assert env.summary()['gold_revealed'] <= meta['B']
    historical = {'gold': len(template.history), 'member_calls': len(template.history)*18,
                  'aggregator_calls': len(template.history)*60, 'tokens': 0, 'source_model_wall_s': 0.}
    for stem in ('member', 'aggregate'):
        for suffix in ('prompt_tokens', 'completion_tokens'):
            historical['tokens'] += int(train['cost'][stem+'_'+suffix][:, template.history].sum())
        historical['source_model_wall_s'] += float(train['cost'][stem+'_wall_time_s'][:, template.history].sum())
    return {'meta': dict(meta), 'traces': traces, 'feedback': feedback, 'ledger': env.summary(),
            'historical_initialization_resources': historical,
            'runtime': {'process_time_s': time.process_time()-run_clock, 'planning_cpu_s': sum(t['plan_cpu_s'] for t in traces),
                        'update_cpu_s': sum(t['update_cpu_s'] for t in traces), 'new_llm_calls': 0}}


def cache_path(meta, identity):
    digest = hashlib.sha256(json.dumps({'identity': identity, 'meta': meta}, sort_keys=True).encode()).hexdigest()
    return PRIVATE / 'runs' / (digest + '.json.gz')


def run_job(job):
    meta, identity = job['meta'], job['identity']
    path = cache_path(meta, identity)
    if path.exists():
        try:
            run = json.loads(gzip.decompress(path.read_bytes()))
            if run['identity'] == identity and run['meta'] == meta and len(run['traces']) == run['ledger']['stream_questions']:
                return {'path': str(path), 'meta': meta, 'reused': True, 'runtime': run['runtime']}
        except (OSError, ValueError, KeyError, EOFError):
            pass
    deleted = job.get('deleted')
    fit_clock = time.process_time()
    template, weights = template_for(meta, deleted)
    template_lookup_cpu = time.process_time() - fit_clock
    order = _SPLITS['folds'][meta['fold']]['orders'][meta['order']]
    if deleted is not None:
        order = order[~np.isin(order, deleted)]
    matched = None
    if meta['method'] == 'matched_random':
        source = dict(meta, method='decision_cost')
        result = run_job({'meta': source, 'identity': identity, 'deleted': deleted})
        matched = json.loads(gzip.decompress(Path(result['path']).read_bytes()))['traces']
    run = episode(_DATA, template, order, meta, weights, matched)
    run['runtime']['template_fit_or_cache_lookup_cpu_s'] = template_lookup_cpu
    run['identity'] = identity
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending')
    pending.write_bytes(gzip.compress(json.dumps(run, allow_nan=False, separators=(',', ':')).encode(), compresslevel=5, mtime=0))
    check = json.loads(gzip.decompress(pending.read_bytes()))
    assert check['meta'] == meta and check['identity'] == identity
    os.replace(pending, path)
    return {'path': str(path), 'meta': meta, 'reused': False, 'runtime': run['runtime']}


def qualification(train, splits):
    checked = model_qualification()
    fold = splits['folds'][0]
    template = ModelTemplate(train, fold['nested'][240], 'P2', fold['references'][240], CONFIG)
    posterior = template.clone()
    q = int(fold['stream'][0])
    main = int(np.argmax(posterior.predict(q)))
    fantasy = fantasy_table(20261107)
    values, _ = projected_scores(posterior, q, main, NEIGHBORS[main], fold['reference_weights'], fantasy)
    errors = []
    for index in (0, 1, 5):
        actions = values[index]['actions']
        canonical = decision_value(posterior.mean, posterior.cov, posterior.observation(q, actions),
                                   posterior.refF, posterior.noise[np.ix_(actions, actions)], fantasy[len(actions)],
                                   fold['reference_weights'])
        errors += [abs(values[index]['kg']-canonical['decision_kg']), abs(values[index]['ig']-canonical['ig_theta'])]
    assert max(errors) < 1e-10
    env = Env(train, template.history, fold['orders'][17], 0, 20)
    first = int(fold['orders'][17][0])
    env.service(first, main)
    same = env.predict_incremental([main], env.executed_current())
    assert same['member_calls'] == same['aggregator_calls'] == 0 and same['gold'] == 1
    bought = env.buy(first, [main, NEIGHBORS[main][0]])
    assert bought['resources']['gold'] == 1 and bought['resources']['member_calls'] == bought['resources']['aggregator_calls'] == 1
    assert env.buy(first, [main, NEIGHBORS[main][0]])['observations'] == []
    return {'status': 'PASS', 'predictor_checks': checked, 'canonical_v2_projected_max_error': max(errors),
            'checks': ['same_snapshot_kg_and_real_joint_update', 'main_gold_no_duplicate_generation_charge',
                       'neighbor_one_new_member_and_aggregate', 'gold_and_original_score_deduplication'],
            'scientific_success': False}


def jobs_for(stage, identity, quick=False):
    if stage == 'A':
        variants = [('none', 0), ('full_observable', 120), ('full_information', 120)]
        variants += [('random', budget) for budget in CONFIG['gold_budgets']]
    elif stage == 'B1':
        variants = list(itertools.product(['random', 'max_uncertainty', 'boundary', 'boundary_cost', 'ig', 'ig_cost', 'decision', 'decision_cost'], CONFIG['gold_budgets']))
    elif stage == 'B2':
        variants = list(itertools.product(['random_neighbor', 'value_uncertainty', 'pair_uncertainty', 'ig', 'decision', 'decision_cost'], CONFIG['gold_budgets']))
    elif stage == 'J':
        variants = list(itertools.product(['random', 'max_uncertainty', 'boundary_cost', 'ig', 'ig_cost', 'decision', 'decision_cost', 'matched_random'], CONFIG['gold_budgets']))
    elif stage == 'D':
        variants = list(itertools.product(['ig_cost', 'no_cost', 'main_only', 'independent_noise', 'random_replacement'], CONFIG['gold_budgets']))
    else:
        raise ValueError(stage)
    for predictor, fold, generation, order in itertools.product(CONFIG['predictors'], range(5), SEEDS, CONFIG['order_seeds']):
        if stage == 'D' and predictor == 'P3':
            continue
        if quick and (fold != 0 or generation != 42 or order != 17):
            continue
        for method, budget in variants:
            yield {'identity': identity, 'meta': {'predictor': predictor, 'fold': fold, 'gen': generation,
                                                 'order': order, 'history_size': 240, 'stage': stage,
                                                 'method': method, 'B': budget,
                                                 'algorithm_seed': algorithm_seed(fold, order, 0)}}


def sensitivity_jobs(identity):
    for stage, methods in (('B1', ('random', 'ig_cost', 'decision_cost')),
                           ('B2', ('random_neighbor', 'ig', 'decision_cost'))):
        for job in jobs_for(stage, identity):
            if job['meta']['B'] != 20 or job['meta']['method'] not in methods:
                continue
            for size in (120, 480):
                yield {'identity': identity, 'meta': dict(job['meta'], history_size=size)}


def delete_groups(train):
    groups = [[] for _ in range(20)]
    rng = np.random.default_rng(20261107)
    for task in ('gsm8k', 'bbh'):
        rows = rng.permutation(np.flatnonzero(train['task'] == task))
        for i, q in enumerate(rows):
            groups[i % 20].append(int(q))
    return [sorted(group) for group in groups]


def delete_jobs(identity):
    groups = delete_groups(_DATA)
    for stage, methods in (('B1', ('decision_cost', 'ig_cost', 'random', 'boundary_cost')),
                           ('B2', ('decision_cost', 'ig', 'random_neighbor'))):
        for job in jobs_for(stage, identity):
            if job['meta']['predictor'] != 'P2' or job['meta']['B'] != 20 or job['meta']['method'] not in methods:
                continue
            for number, deleted in enumerate(groups):
                yield {'identity': identity, 'meta': dict(job['meta'], delete_group=number), 'deleted': deleted}


def execute(jobs, workers, name):
    completed = []
    jobs = list(jobs)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as executor:
        batches = [[job for job in jobs if job['meta']['method'] != 'matched_random'],
                   [job for job in jobs if job['meta']['method'] == 'matched_random']]
        for batch in batches:
            futures = [executor.submit(run_job, job) for job in batch]
            for future in as_completed(futures):
                receipt = future.result()
                completed.append(receipt)
                if len(completed) % 10 == 0 or len(completed) == len(jobs):
                    put(PRIVATE / 'PROGRESS.json', {'stage': name, 'completed_paths': len(completed), 'required_paths': len(jobs),
                                                  'elapsed_s': time.perf_counter()-start, 'last': receipt['meta']})
                    put(PRIVATE / (name+'_INDEX.json'), completed)
                    print(name, len(completed), '/', len(jobs), round(time.perf_counter()-start, 1), flush=True)
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['qualify', 'precheck', 'A', 'B1', 'B2', 'J', 'D', 'sensitivity', 'delete', 'all'], default='precheck')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    initialize_worker()
    identity, sources = signature()
    if args.stage in ('qualify', 'precheck'):
        receipt = qualification(_DATA, _SPLITS)
        receipt['identity'] = identity
        put(PUBLIC / 'QUALIFICATION.json', receipt)
        print('NUMERICAL_AND_PERMISSION_QUALIFICATION_PASS', flush=True)
        if args.stage == 'qualify':
            return
        # One fold, one generation seed, one order; every mechanism and budget type.
        checks = []
        for stage in ('A', 'B1', 'B2', 'J', 'D'):
            variants = list(jobs_for(stage, identity, quick=True))
            checks += [item for item in variants if item['meta']['B'] in (0, 20, 120)]
        execute(checks, args.workers, 'PRECHECK')
        return
    config = {**CONFIG, 'identity': identity, 'source_hashes': sources,
              'split_metadata': _SPLITS['metadata'], 'classification': 'development_offline_replay'}
    config_path = PUBLIC / 'E3_RUN_CONFIG.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Frozen config differs'
    else:
        put(config_path, config)
    if args.stage == 'sensitivity':
        execute(sensitivity_jobs(identity), args.workers, 'SENSITIVITY')
        return
    if args.stage == 'delete':
        execute(delete_jobs(identity), args.workers, 'DELETE')
        return
    stages = ('A', 'B1', 'B2', 'J', 'D') if args.stage == 'all' else (args.stage,)
    for stage in stages:
        execute(jobs_for(stage, identity), args.workers, stage)


if __name__ == '__main__':
    main()
