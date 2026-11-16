"""Bind the admitted input-only P2 pool to production native worker manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

import p2_execution_adapter as adapter


CODE = Path(__file__).resolve().parent
PRIVATE = Path(r"E:\语言模型组合路由\research\e20_p2_execution\private")
REMOTE = "/data2/chendong/e20_p2_execution"
OLD_REMOTE = "/data1/data2/chendong/routing/research/e18_confirmatory_execution"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def main():
    pool_path = PRIVATE / "ADMITTED2400_INPUTS_ONLY.jsonl"
    prediction_path = PRIVATE / "P2_PREDICTIONS_FROZEN.npz"
    records = [json.loads(line) for line in pool_path.read_text(encoding="utf-8").splitlines()]
    scientific_scope = read(PRIVATE / "EXECUTION_SCOPE.json")
    with np.load(prediction_path, allow_pickle=False) as packet:
        probabilities = packet["probabilities"].copy()
        raw = packet["raw_exact_probabilities"].copy()
        qids = packet["question_ids"].tolist()
    if qids != [row["question_id"] for row in records] or probabilities.shape != (2, 2400, 20):
        raise ValueError("Frozen predictions and admitted pool identities differ")
    if np.any((raw > 0) & ~(probabilities > 0)):
        raise ValueError("Raw diagnostic contains an action outside the frozen main support")
    member_keys = scientific_scope["required_member_keys"]
    aggregate_keys = scientific_scope["required_aggregate_keys"]
    budget = dict(schema="p2_actual_execution_budget_v1", original_questions=2400,
                  member_generation=len(member_keys), aggregation=len(aggregate_keys),
                  conservative_success_token_cap=len(member_keys)*6144+len(aggregate_keys)*8192,
                  expected_cache_hits=0, new_llm_calls_executed=0)
    entries = {"inputs/ADMITTED2400_INPUTS_ONLY.jsonl": pool_path.read_bytes(),
               "inputs/ACTUAL_EXECUTION_BUDGET.json": encode(budget)}
    protocol_bytes = (CODE / "P2_PRESELECTION_PROTOCOL.json").read_bytes()
    bindings = dict(protocol_sha256=sha(protocol_bytes), target_inputs_sha256=sha(pool_path.read_bytes()),
                    frozen_actions_sha256=sha(prediction_path.read_bytes()),
                    budget_sha256=sha(entries["inputs/ACTUAL_EXECUTION_BUDGET.json"]),
                    target_ids_sha256=adapter.canonical_digest([[row["question_id"], row["canonical_question_sha256"]] for row in records]),
                    final_model_sha256="b8b786eeb0aaac6f869136f5353186c13f78aba0a716ebf0bcad145532d01f92",
                    final_prediction_sha256=sha(prediction_path.read_bytes()))
    scope = adapter.build_scope_contract(records, probabilities.tolist(), bindings, member_keys, aggregate_keys)
    if set(map(tuple, scope["required_member_keys"])) != set(map(tuple, member_keys)) or set(map(tuple, scope["required_aggregate_keys"])) != set(map(tuple, aggregate_keys)):
        raise ValueError("Runtime support union differs from independently frozen scientific scope")
    entries["inputs/EXECUTION_SCOPE_RUNTIME.json"] = encode(scope)
    approval = dict(schema="p2_scope_approval_v1", status="APPROVED", cohort=adapter.COHORT,
                    software_test_only=False, operations=["member_generation", "aggregation", "read_reference", "score_outputs"],
                    scope_sha256_by_role={"execution": adapter.canonical_digest(scope)},
                    bindings=bindings, approved_output_root=REMOTE+"/owned",
                    execution_limits=scope["execution_limits"],
                    authorization="User explicitly requested completion of P2 preparation and commencement of P2 execution after E20 new2400 proposal",
                    sealed200_answer_access=False, fresh400_reward_access=False)
    entries["inputs/AUTHORIZATION_v2.json"] = encode(approval)
    inventory = read(CODE / "SERVER_RUNTIME_INVENTORY_PRIVATE.json")
    mapping = {row["member_index"]: row for row in inventory["members"]}
    if any(row["status"] != "METADATA_MATCH" for row in mapping.values()):
        raise ValueError("Original physical member mapping metadata differs")
    helper = dict(path=OLD_REMOTE+"/runtime_assets/artifacts/private/server_migration_20261102/mapped_backend_v1.py",
                  sha256="4ea0b3f213e4abe70a388e5f3d86cfe92b86eabb1db4f550ea0c57758febb845")
    common = dict(software_test_only=False, asset_root=OLD_REMOTE+"/runtime_assets",
                  adapter_ref=dict(path=REMOTE+"/code/p2_execution_adapter.py", sha256=sha((CODE/"p2_execution_adapter.py").read_bytes())),
                  scope_ref=dict(path=REMOTE+"/inputs/EXECUTION_SCOPE_RUNTIME.json", sha256=sha(entries["inputs/EXECUTION_SCOPE_RUNTIME.json"])),
                  question_records_ref=dict(path=REMOTE+"/inputs/ADMITTED2400_INPUTS_ONLY.jsonl", sha256=sha(entries["inputs/ADMITTED2400_INPUTS_ONLY.jsonl"])),
                  approval_ref=dict(path=REMOTE+"/inputs/AUTHORIZATION_v2.json", sha256=sha(entries["inputs/AUTHORIZATION_v2.json"])),
                  owned_output_root=REMOTE+"/owned", install_signal_handlers=True)
    manifest_refs = []
    for index in range(6):
        keys = [key for key in member_keys if key[2] == index]
        relative = f"inputs/member{index}_tasks.json"
        entries[relative] = encode(keys)
        physical = mapping[index]
        manifest = dict(common, schema="e18_gpu_worker_manifest_v1", stage="member_generation", member_index=index,
                        task_keys_ref=dict(path=REMOTE+"/"+relative, sha256=sha(entries[relative])),
                        gpu_lock_root=REMOTE+f"/gpu-locks/member{index}",
                        backend=dict(mode="mapped", physical_root=physical["worker_root"],
                                     mapping_ref=dict(path=physical["mapping_path"], sha256=physical["mapping_sha256"]),
                                     helper_ref=helper))
        name=f"manifests/member{index}_v2.json"
        entries[name]=encode(manifest)
        manifest_refs.append(dict(profile=index, path=REMOTE+"/"+name, sha256=sha(entries[name]), tasks=len(keys)))
    entries["inputs/aggregation_tasks.json"] = encode(aggregate_keys)
    aggregate = dict(common, schema="e18_aggregation_worker_manifest_v3", stage="aggregation",
                     task_keys_ref=dict(path=REMOTE+"/inputs/aggregation_tasks.json", sha256=sha(entries["inputs/aggregation_tasks.json"])),
                     gpu_lock_root=REMOTE+"/gpu-locks/aggregation",
                     backend=dict(mode="mapped", physical_root=OLD_REMOTE+"/runtime_assets",
                                  mapping_ref=dict(path=OLD_REMOTE+"/runtime_assets/artifacts/preparation/e18_qwen3_aggregator_mapping.json",
                                                   sha256="dfae3e84c1988616775d7ccb0385189ebd0112db6e152dc52c2400366be15217"),
                                  helper_ref=helper))
    entries["manifests/aggregation_v2.json"] = encode(aggregate)
    entries["P2_PRESELECTION_PROTOCOL.json"] = protocol_bytes
    package = PRIVATE / "P2_PRODUCTION_INPUTS_AND_MANIFESTS_v2.zip"
    with zipfile.ZipFile(package, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    for name in ("inputs/EXECUTION_SCOPE_RUNTIME.json", "inputs/AUTHORIZATION_v2.json", "inputs/ACTUAL_EXECUTION_BUDGET.json"):
        target = PRIVATE / "runtime_launch_v2" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(entries[name])
    receipt = dict(schema="p2_production_launch_preparation_v1", status="READY_FOR_REMOTE_PREFLIGHT",
                   scope_canonical_sha256=adapter.canonical_digest(scope), bindings=bindings,
                   archive=dict(path=str(package), sha256=sha(package.read_bytes()), bytes=package.stat().st_size),
                   member_manifest_refs=manifest_refs,
                   aggregate_manifest_ref=dict(path=REMOTE+"/manifests/aggregation_v2.json", sha256=sha(entries["manifests/aggregation_v2.json"])),
                   budget=budget, generated_files=[dict(path=name,sha256=sha(data),bytes=len(data)) for name,data in entries.items()],
                   new_llm_calls=0, references_decoded=0, scientific_source_edits=0)
    (PRIVATE / "P2_PRODUCTION_LAUNCH_RECEIPT_v2.json").write_bytes(encode(receipt))
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
