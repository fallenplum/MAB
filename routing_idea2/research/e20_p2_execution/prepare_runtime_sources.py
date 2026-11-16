"""Copy qualified E18 execution code into the P2 probability-scope namespace."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
E18 = WORKSPACE / "experiments/e18_confirmatory_execution"
OLD_ADAPTER = WORKSPACE / "experiments/e18_mechanism_and_confirmation_preparation/e18_execution_adapter.py"


BUILD_SCOPE = '''def build_scope_contract(
    question_records, policy_probabilities, bindings,
    member_execution_keys, aggregate_execution_keys,
):
    """Bind B/C probabilities and exactly their positive-support execution union."""
    records = list(question_records)
    ids = [row["question_id"] for row in records]
    canonical = [row["canonical_question_sha256"] for row in records]
    if not records or len(set(ids)) != len(ids) or len(set(canonical)) != len(canonical):
        raise ValueError("Unique original target question identities required")
    probabilities = [[[float(value) for value in row] for row in policy]
                     for policy in policy_probabilities]
    if len(probabilities) != 2 or any(len(policy) != len(records) for policy in probabilities):
        raise ValueError("Exactly two B/C question-by20 probability matrices required")
    for policy in probabilities:
        for row in policy:
            if len(row) != 20 or any(not math.isfinite(value) or value < 0 for value in row):
                raise ValueError("Finite nonnegative20-action probability vectors required")
            if abs(sum(row)-1.0) > 1e-12:
                raise ValueError("Each deployed policy probability row must sum to one")
    target_ids_sha = canonical_digest([[qid, digest] for qid, digest in zip(ids, canonical)])
    if bindings.get("target_ids_sha256") != target_ids_sha:
        raise ValueError("Target identity binding differs")
    members, aggregates = set(), set()
    for index, record in enumerate(records):
        actions = {action for action in range(20)
                   if any(policy[index][action] > 0 for policy in probabilities)}
        for action in actions:
            for seed in GENERATION_SEEDS:
                aggregates.add((record["question_id"], seed, action))
                members.update((record["question_id"], seed, member) for member in TEAMS[action])
    member_rows = list(map(tuple, member_execution_keys))
    aggregate_rows = list(map(tuple, aggregate_execution_keys))
    member_calls, aggregate_calls = set(member_rows), set(aggregate_rows)
    if len(member_calls) != len(member_rows) or len(aggregate_calls) != len(aggregate_rows):
        raise ValueError("Duplicate execution ownership keys are forbidden")
    if not member_calls <= members or not aggregate_calls <= aggregates:
        raise ValueError("Execution keys exceed the frozen positive-support union")
    return dict(
        schema="p2_scope_contract_v1", role="execution", cohort=COHORT,
        bindings=dict(bindings), transport_split=TRANSPORT_SPLIT,
        operator_fingerprint=OPERATOR_FP,
        question_keys=[dict(question_id=row["question_id"],
                            canonical_question_sha256=row["canonical_question_sha256"],
                            question_sha256=hashlib.sha256(row["question"].encode()).hexdigest(),
                            task=row["task"], subtask=row.get("subtask") or "") for row in records],
        frozen_policy_probabilities=probabilities,
        required_member_keys=[list(key) for key in sorted(members)],
        required_aggregate_keys=[list(key) for key in sorted(aggregates)],
        member_execution_keys=[list(key) for key in sorted(member_calls)],
        aggregate_execution_keys=[list(key) for key in sorted(aggregate_calls)],
        execution_limits=dict(member_generation=len(member_calls), aggregation=len(aggregate_calls)),
        generation_seeds=list(GENERATION_SEEDS), aggregation_seed=42,
        authorization_state="HASH_BOUND_USER_APPROVAL_REQUIRED",
    )


'''


def write(name, text):
    path = HERE / name
    if path.exists() and path.read_bytes() != text.encode():
        raise FileExistsError("Prepared source exists with different bytes " + name)
    if not path.exists():
        path.write_bytes(text.encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    source = OLD_ADAPTER.read_text(encoding="utf-8")
    original_sha = hashlib.sha256(OLD_ADAPTER.read_bytes()).hexdigest()
    if original_sha != "21e76f6a38fa5f2ed590a2b2068bda8255bc4ebc290dea9eb7c32a3a44a65bc4":
        raise ValueError("Qualified original adapter changed")
    start = source.index("def build_scope_contract(")
    end = source.index("class E18ExecutionAdapter:", start)
    source = source[:start] + BUILD_SCOPE + source[end:]
    source = source.replace('COHORT = "e18_confirm"', 'COHORT = "e20_p2_confirm"')
    source = source.replace('"e18_scope_contract_v2"', '"p2_scope_contract_v1"')
    source = source.replace('"e18_scope_approval_v2"', '"p2_scope_approval_v1"')
    source = source.replace('"e18_execution_task_receipt_v2"', '"p2_execution_task_receipt_v1"')
    source = source.replace('E18ExecutionAdapter', 'P2ExecutionAdapter')
    source = source.replace('self.scope["frozen_policy_actions"]', 'self.scope["frozen_policy_probabilities"]')
    source = source.replace('import json\n', 'import json\nimport math\nfrom p2_scope_freeze import freeze_scope\n', 1)
    old_digest = '''def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
'''
    new_digest = '''_SCOPE_DIGESTS = {}


def canonical_digest(value):
    cached = _SCOPE_DIGESTS.get(id(value))
    if cached is not None and cached[0] is value:
        return cached[1]
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
'''
    if old_digest not in source:
        raise ValueError("Original canonical digest source differs")
    source = source.replace(old_digest, new_digest)
    source = source.replace('self.scope = json.loads(json.dumps(scope_contract))',
                            'self.scope = freeze_scope(scope_contract)\n        _SCOPE_DIGESTS[id(self.scope)] = (self.scope, canonical_digest(self.scope))')
    source = source.replace('Isolated E18 scope factories and an approval-gated single-task worker.',
                            'P2 probability-support scope and approval-gated single-task execution.')
    source = source.replace('e18_confirm. Real backend construction is deferred until explicit approval.',
                            'e20_p2_confirm. Real backend construction requires explicit bound approval.')
    source = source.replace('            "reward",\n',
                            '            "reward",\n            "rewards",\n            "answer",\n            "answers",\n            "Y",\n            "target",\n            "targets",\n', 1)
    sha = write("p2_execution_adapter.py", source)
    freezer = (E18 / "cpu_scope_memo.py").read_text(encoding="utf-8")
    frozen_code = freezer[freezer.index("def _deny("):freezer.index("def original_digest(")]
    freeze_sha = write("p2_scope_freeze.py", '"""E18 read-only JSON containers reused for a single P2 scope snapshot."""\n\n' + frozen_code)
    copies = {}
    for old_name, new_name in (("e18_gpu_worker.py", "p2_member_worker.py"),
                               ("e18_aggregation_worker_v3.py", "p2_aggregation_worker.py")):
        text = (E18 / old_name).read_text(encoding="utf-8")
        text = text.replace('ADAPTER_SHA = "' + original_sha + '"', 'ADAPTER_SHA = "' + sha + '"')
        text = text.replace('adapter_module.E18ExecutionAdapter(', 'adapter_module.P2ExecutionAdapter(')
        text = text.replace('"e18_execution_adapter"', '"p2_execution_adapter"')
        copies[new_name] = write(new_name, text)
    receipt = dict(status="P2_SCOPE_NAMESPACE_ADAPTED", original_adapter_sha256=original_sha,
                   adapter_sha256=sha, freezer_sha256=freeze_sha, worker_sources=copies,
                   scientific_core_sources_changed=0, model_loading_paths_changed=0,
                   scope_probability_rows=2, new_llm_calls=0,
                   operational_changes=["P2 probability-support scope and cohort namespace", "Read-only scope canonical digest memo", "Workers bind new adapter SHA"])
    (HERE / "P2_RUNTIME_SOURCE_ADAPTATION.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
