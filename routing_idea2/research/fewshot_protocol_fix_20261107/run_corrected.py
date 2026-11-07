"""Exact support-set development evaluation. No LLM calls or fresh outcomes."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from models import METHODS, initialize, adapt, semantic_labels

ROOT = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
WORK = ROOT / 'research/fewshot_protocol_fix_20261107'
PUBLIC = WORK / 'public'
PRIVATE = WORK / 'private'
COMMON = ROOT / 'research/conditional_gain_20261106/repo/routing_idea2/research/conditional_gain_20261106/code'
sys.path.insert(0, str(COMMON))
from common import load, sha, TEAMS, SEEDS
RUN_SIGNATURE = None


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(pending, path)


def inputs(data, rows):
    return {name: data[name][rows] for name in ('E', 'ids', 'task', 'sub')}


def qualify(train, validation):
    rows = np.flatnonzero(validation['task'] == 'bbh')[:4]
    task = str(validation['sub'][rows[0]])
    query = inputs(validation, rows)
    state = initialize(train, query, task, 'unseen')
    excluded = semantic_labels(train) == task
    altered = copy.deepcopy(train)
    altered['Y'][excluded] = 1 - altered['Y'][excluded]
    altered['E'][excluded] = 99.0
    second = initialize(altered, query, task, 'unseen')
    assert not set(state.history_indices) & set(np.flatnonzero(excluded))
    assert np.array_equal(state.coefficients, second.coefficients)
    assert np.array_equal(state.action_embedding, second.action_embedding)
    support = np.asarray([0, 1])
    reward = validation['Y'][rows[support], 0]
    observed = np.ones((2, 20), dtype=bool)
    posterior = adapt(state, support, reward, observed, 'online_ridge')
    xs = state.query_features[support]
    history_rhs = state.precision @ state.coefficients
    explicit = np.linalg.solve(state.precision + xs.T @ xs, history_rhs + xs.T @ reward)
    error = float(np.max(np.abs(posterior - state.query_features @ explicit)))
    assert error < 1e-10
    restricted = np.zeros_like(observed)
    actions = np.argmax(state.query_prediction[support], axis=1)
    restricted[np.arange(2), actions] = True
    restricted_reward = np.where(restricted, reward, np.nan)
    prediction = adapt(state, support, restricted_reward, restricted, 'online_ridge')
    untouched = ~restricted.any(0)
    assert np.array_equal(prediction[:, untouched], state.query_prediction[:, untouched])
    for action in np.flatnonzero(restricted.any(0)):
        legal = restricted[:, action]
        xx = xs[legal]
        fitted = np.linalg.solve(state.precision + xx.T @ xx, history_rhs[:, action] + xx.T @ reward[legal, action])
        assert np.max(np.abs(prediction[:, action] - state.query_features @ fitted)) < 1e-10
    empty = np.asarray([], dtype=int)
    for method in METHODS:
        clean = method.removesuffix('_shuffle')
        baseline = adapt(state, empty, np.empty((0, 20)), np.empty((0, 20), bool), clean)
        if clean != 'fixed':
            assert np.array_equal(baseline, state.query_prediction)
        else:
            assert np.all(np.argmax(baseline, axis=1) == state.fixed_action)
    # A poisoned Query outcome has no route into initialization or adaptation.
    altered_validation = copy.deepcopy(validation)
    altered_validation['Y'][rows[2:]] = 1000
    third = initialize(train, inputs(altered_validation, rows), task, 'unseen')
    assert np.array_equal(state.query_prediction, third.query_prediction)
    for method in ('online_ridge', 'hierarchical', 'knn', 'adapter2', 'adapter4'):
        first = adapt(state, support, restricted_reward, restricted, method)
        second = adapt(third, support, restricted_reward, restricted, method)
        assert np.array_equal(first, second)
        poisoned = np.where(restricted, reward, 12345.0)
        third_prediction = adapt(state, support, poisoned, restricted, method)
        assert np.allclose(first, third_prediction, atol=1e-12)
    assert np.max(np.abs(state.action_embedding.sum(0))) < 1e-10
    assert np.allclose(state.action_embedding.T @ state.action_embedding, 20 * np.eye(4))
    # Compare the cached historical top-five shortcut with brute-force retrieval.
    history = state.history_indices
    history_residual = train['Y'][history].mean(1) - state.train_features @ state.coefficients
    all_distance = np.maximum(0.0, 1 - query['E'] @ train['E'][history].T)
    residual_feedback = reward - state.query_prediction[support]
    expected = state.query_prediction.copy()
    for q in range(len(rows)):
        for action in range(20):
            old_indices = np.argsort(all_distance[q])[:5]
            old_weights = 1 / np.maximum(all_distance[q, old_indices], .01)
            old = np.average(history_residual[old_indices, action], weights=old_weights)
            distances = np.r_[all_distance[q], state.query_distances[q, support]]
            residuals = np.r_[history_residual[:, action], residual_feedback[:, action]]
            chosen = np.argsort(distances)[:5]
            new = np.average(residuals[chosen], weights=1 / np.maximum(distances[chosen], .01))
            expected[q, action] += new - old
    brute_error = float(np.max(np.abs(expected - adapt(state, support, reward, observed, 'knn'))))
    assert brute_error < 1e-10
    return {'status': 'PASS', 'historical_posterior_max_error': error,
            'historical_knn_bruteforce_max_error': brute_error,
            'checks': ['unseen_scores_and_embeddings_excluded', 'query_outcome_perturbation_invariance',
                       'full_parameter_online_equals_explicit_refit', 'restricted_online_equals_explicit_refit',
                       'unobserved_feedback_invariance_all_updaters', 'k0_identity_all_updaters',
                       'k0_fixed_identity', 'learned_action_embedding_centered_orthogonal',
                       'historical_plus_support_knn_equals_bruteforce'],
            'scope': 'numerical_and_information_boundary_qualification_not_scientific_success'}


def cost_for_choices(data, rows, choices):
    """Cached realized cost for one chosen team; axes question, seed, action."""
    n = len(rows)
    result = np.zeros((n, 3, 20, 5))
    cost = data['cost']
    # Receipts store generation, question, member/combination.
    for action, team in enumerate(TEAMS):
        for j, field in enumerate(('execution_counts', 'prompt_tokens', 'completion_tokens', 'wall_time_s')):
            member = cost['member_' + field][:, rows, :][:, :, team].sum(axis=2).T
            aggregate = cost['aggregate_' + field][:, rows, action].T
            result[:, :, action, j] = member + aggregate
        result[:, :, action, 4] = 1.0
    return result


def evaluate_group(train, validation, name, mode, feedback, k, state, rows):
    n = len(rows)
    path = PRIVATE / f'{mode}_{feedback}_k{k}_{name}.npz'
    identity = hashlib.sha256(json.dumps({'run': RUN_SIGNATURE, 'task': name, 'mode': mode,
                                           'feedback': feedback, 'k': k, 'rows': rows.tolist(),
                                           'methods': METHODS, 'seeds': SEEDS}, sort_keys=True).encode()).hexdigest()
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                valid = str(saved['signature']) == identity
                valid &= np.array_equal(saved['question_ids'], validation['ids'][rows])
                valid &= saved['choices'].shape == (math.comb(n, k), 3, len(METHODS), n)
                valid &= saved['support_sets'].shape == (math.comb(n, k), k)
                valid &= saved['total'].shape == (9, len(METHODS), 8)
                valid &= saved['roles'].shape == (n, 9, len(METHODS), 8)
                valid &= saved['service_cost'].shape == (3, len(METHODS), 5)
                valid &= saved['support_cost'].shape == (3, 5)
                if valid:
                    size = len(saved['support_sets'])
                    count = size * (n - k)
                    return {'task': name, 'n': n, 'k': k, 'total': saved['total'], 'roles': saved['roles'],
                            'count': count, 'support_sets': size, 'gold_reveals_per_episode': k,
                            'service_cost_mean': saved['service_cost'] / count,
                            'support_cost_per_episode': saved['support_cost'] / size, 'private_file': path.name}
        except (KeyError, ValueError, OSError, EOFError, zipfile.BadZipFile):
            pass
    support_sets = np.asarray(list(itertools.combinations(range(n), k)), dtype=np.int16).reshape(-1, k) if k else np.empty((1, 0), np.int16)
    shape = (9, len(METHODS), 8)
    total = np.zeros(shape)
    roles = np.zeros((n,) + shape)
    choices = np.zeros((len(support_sets), 3, len(METHODS), n), dtype=np.int8)
    rewards = validation['Y'][rows]
    base_actions = np.argmax(state.query_prediction, axis=1)
    fixed_actions = np.full(n, state.fixed_action)
    actual_cost = cost_for_choices(validation, rows, None)
    service_cost = np.zeros((3, len(METHODS), 5))
    support_cost = np.zeros((3, 5))
    begin = time.perf_counter()
    for episode, support in enumerate(support_sets):
        query_mask = np.ones(n, dtype=bool)
        query_mask[support] = False
        query = np.flatnonzero(query_mask)
        observed = np.ones((k, 20), dtype=bool)
        if feedback == 'single':
            observed[:] = False
            observed[np.arange(k), base_actions[support]] = True
        for source in range(3):
            real = np.where(observed, rewards[support, source], np.nan)
            shuffled = np.where(observed, train['Y'][state.donor_indices[support], source], np.nan)
            for method_index, method in enumerate(METHODS):
                clock = time.perf_counter()
                predicted = adapt(state, support, shuffled if method.endswith('_shuffle') else real,
                                  observed, method.removesuffix('_shuffle'))
                elapsed = time.perf_counter() - clock
                selected = np.argmax(predicted, axis=1)
                choices[episode, source, method_index] = selected
                service_cost[source, method_index] += actual_cost[query, source, selected[query]].sum(0)
                for target in range(3):
                    reward = rewards[query, target, selected[query]]
                    baseline = rewards[query, target, base_actions[query]]
                    fixed = rewards[query, target, fixed_actions[query]]
                    metrics = np.column_stack((reward, (reward > baseline), (reward < baseline),
                                               (reward > fixed), (reward < fixed),
                                               np.full(len(query), elapsed / len(query)),
                                               rewards[query, target].max(1) - reward,
                                               selected[query] != base_actions[query]))
                    direction = source * 3 + target
                    summed = metrics.sum(0)
                    total[direction, method_index] += summed
                    roles[query, direction, method_index] += metrics
                    roles[support, direction, method_index] += summed
            if k:
                if feedback == 'single':
                    support_cost[source] += actual_cost[support, source, base_actions[support]].sum(0)
                else:
                    c = validation['cost']
                    for j, field in enumerate(('execution_counts', 'prompt_tokens', 'completion_tokens', 'wall_time_s')):
                        support_cost[source, j] += c['member_' + field][source, rows[support]].sum() + c['aggregate_' + field][source, rows[support]].sum()
                    support_cost[source, 4] += 20 * k
        if episode % 200 == 0 or episode + 1 == len(support_sets):
            put(PRIVATE / 'PROGRESS.json', {'phase': 'CPU_EXACT_ENUMERATION', 'mode': mode, 'feedback': feedback,
                                          'task': name, 'k': k, 'completed_support_sets': episode + 1,
                                          'total_support_sets': len(support_sets), 'elapsed_s': time.perf_counter() - begin})
    count = len(support_sets) * (n - k)
    assert np.max(np.abs(roles.sum(0) - (k + 1) * total)) < 1e-6
    assert np.allclose(total[:, 0, 0] / count, rewards[np.arange(n)[:, None], np.arange(3)[None, :], base_actions[:, None]].mean(0)[np.tile(np.arange(3), 3)])
    pending = path.with_suffix('.pending.npz')
    np.savez_compressed(pending, signature=np.asarray(identity), support_sets=support_sets,
                        choices=choices, total=total, roles=roles,
                        question_ids=validation['ids'][rows], donor_ids=train['ids'][state.donor_indices],
                        rows=rows, service_cost=service_cost, support_cost=support_cost)
    with np.load(pending, allow_pickle=False) as check:
        assert str(check['signature']) == identity and np.array_equal(check['total'], total)
    os.replace(pending, path)
    return {'task': name, 'n': n, 'k': k, 'total': total, 'roles': roles, 'count': count,
            'support_sets': len(support_sets), 'gold_reveals_per_episode': k, 'service_cost_mean': service_cost / count,
            'support_cost_per_episode': support_cost / len(support_sets), 'private_file': path.name}


def main():
    global RUN_SIGNATURE
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualify-only', action='store_true')
    args = parser.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    data, refs = load()
    train, validation = data['selector_train'], data['selector_validation']
    receipt = qualify(train, validation)
    receipt['inputs'] = {key: value['sha256'] for key, value in refs.items()}
    receipt['code_sha256'] = {p.name: sha(p) for p in PUBLIC.glob('*.py')}
    RUN_SIGNATURE = hashlib.sha256(json.dumps({'inputs': receipt['inputs'],
                                               'engine': {p.name: sha(p) for p in (PUBLIC / 'models.py', PUBLIC / 'run_corrected.py')}},
                                              sort_keys=True).encode()).hexdigest()
    receipt['cache_run_signature'] = RUN_SIGNATURE
    put(PUBLIC / 'QUALIFICATION.json', receipt)
    if args.qualify_only:
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        return
    groups = ['gsm8k'] + sorted(set(validation['sub'][validation['task'] == 'bbh']))
    summaries = {}
    costs = {}
    from stats import summarize
    gsm_cache = {}
    state_cache = {}
    for mode in ('seen', 'unseen'):
        for feedback in ('full20', 'single'):
            for k in (0, 1, 2):
                blocks = []
                for name in groups:
                    rows = np.flatnonzero(semantic_labels(validation) == name)
                    if name == 'gsm8k' and (feedback, k) in gsm_cache:
                        block = gsm_cache[(feedback, k)]
                    else:
                        state_key = (mode, name)
                        if state_key not in state_cache:
                            state_cache[state_key] = initialize(train, inputs(validation, rows), name, mode)
                        state = state_cache[state_key]
                        block = evaluate_group(train, validation, name, mode, feedback, k, state, rows)
                        if name == 'gsm8k':
                            gsm_cache[(feedback, k)] = block
                    blocks.append(block)
                key = f'{mode}/{feedback}/k{k}'
                summaries[key] = summarize(blocks[0], blocks[1:])
                costs[key] = [{field: (value.tolist() if isinstance(value, np.ndarray) else value)
                               for field, value in block.items() if field not in ('total', 'roles')} for block in blocks]
                put(PUBLIC / 'EXPERIMENT_RESULTS.json', summaries)
                put(PUBLIC / 'COSTS.json', {'actual_new_llm_calls': 0, 'cached_counterfactual_costs': costs,
                                         'cost_axes': ['member_plus_aggregate_calls', 'prompt_tokens', 'completion_tokens', 'gpu_wall_s', 'aggregate_calls'],
                                         'service_cost_seed_axes': 'three diagonal source_seed=target_seed evaluations only',
                                         'note': 'Kernel integration is an expectation over alternative episodes. Never add it as one deployment bill.'})
                print(key + ' COMPLETE', flush=True)
    put(PUBLIC / 'COMPLETION.json', {'status': 'COMPLETE_DEVELOPMENT', 'llm_calls': 0,
                                   'fresh400_scores_read': False, 'sealed200_read': False,
                                   'configurations': len(summaries), 'qualification': receipt['status']})
    put(PRIVATE / 'PROGRESS.json', {'phase': 'COMPLETE_DEVELOPMENT', 'configurations': len(summaries)})


if __name__ == '__main__':
    main()
