"""Create an approval-only metadata dependency graph from frozen E18 actions.

This review artifact contains original identities and input-source references,
without input text, member answers, aggregate answers, labels, prompts, native
job identifiers, queue files, worker commands, or runnable model invocations.
It neither trains/encodes nor changes the frozen question/combination choices.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from e18_common import ASSETS, CODE, PRIVATE, PUBLIC, POLICIES, SEEDS, read_json, sha256, write_json
from audit_sources import canonical_hash, digest, object_fields, project


OPERATOR = "9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e"
STATE = "APPROVAL_REQUIRED"
PROPOSED_STATE = "PROPOSED_NOT_ADMITTED"
AGGREGATOR_INVOCATION_SEED = 42
ALLOWED_PROPOSAL_FIELDS = {
    "question_id", "task", "subtask", "source_split", "question", "canonical_question_sha256",
    "source_frame", "source_file", "source_index", "source_revision", "status",
    "independence_qualification", "source_layer_admission",
}
REQUIRED_INPUT_FIELDS = {"question_id", "task", "subtask", "source_split", "question", "canonical_question_sha256"}


def _node_id(kind,key):
    return "/".join((kind,)+tuple(map(str,key)))


def _keys(actions,teams,seeds):
    aggregate_keys=set()
    member_keys=set()
    for question in range(actions.shape[1]):
        for action in set(map(int,actions[:,question])):
            for generation in seeds:
                aggregate_keys.add((question,int(generation),action))
                member_keys.update((question,int(generation),int(member)) for member in teams[action])
    return member_keys,aggregate_keys


def build_review_dag(actions,teams,seeds,policies,identities):
    """Pure metadata construction from already selected actions and inputs."""
    actions=np.asarray(actions)
    teams=np.asarray(teams)
    if actions.ndim!=2 or actions.dtype.kind not in "iu" or actions.shape[0]!=4 or actions.shape[1]!=len(identities):
        raise ValueError("Exactly four aligned frozen integer-action policies are required")
    if teams.shape!=(20,3) or teams.dtype.kind not in "iu" or not np.array_equal(teams,np.array(list(combinations(range(6),3)))):
        raise ValueError("Frozen original physical six-model K=3 semantic order differs")
    if np.any(actions<0) or np.any(actions>=20):
        raise ValueError("Invalid frozen combination action")
    if tuple(map(int,seeds))!=(42,137,2024) or len(policies)!=4 or len(set(policies))!=4:
        raise ValueError("Original generation seeds and four unique policies are required")
    if len({row["question_id"] for row in identities})!=len(identities) or len({row["canonical_question_sha256"] for row in identities})!=len(identities):
        raise ValueError("Input question identities must be unique")
    members,aggregates=_keys(actions,teams,seeds)
    decisions={}
    clusters=[]
    member_users=defaultdict(set)
    aggregate_users=defaultdict(set)
    for question,identity in enumerate(identities):
        refs=[]
        for policy_index,policy in enumerate(policies):
            action=int(actions[policy_index,question])
            reference=f"decision/{question}/{policy_index}"
            decisions[reference]=dict(policy=policy,policy_index=policy_index,question_index=question,action_index=action,
                                      frozen_array_field="actions[policy_index,question_index]")
            refs.append(reference)
            for generation in seeds:
                aggregate_users[(question,int(generation),action)].add(reference)
                for member in teams[action]:
                    member_users[(question,int(generation),int(member))].add(reference)
        clusters.append(dict(
            question_index=question,question_id=identity["question_id"],
            canonical_question_sha256=identity["canonical_question_sha256"],
            input_reference=identity["input_reference"],input_row_fingerprint=identity["input_row_fingerprint"],
            source_frame=identity["source_frame"],task=identity["task"],subtask=identity["subtask"],
            admission_state=PROPOSED_STATE if question>=400 else "EXISTING_FRESH_EXECUTION_APPROVAL_REQUIRED",
            execution_state=STATE,member_generation_seeds=list(seeds),policy_decision_references=refs,
        ))
    member_nodes=[]
    for key in sorted(members):
        question,generation,member=key
        identity=identities[question]
        member_nodes.append(dict(
            node_id=_node_id("member",key),node_kind="MEMBER_OUTPUT_METADATA",member_key=list(key),
            question_id=identity["question_id"],canonical_question_sha256=identity["canonical_question_sha256"],
            input_reference=identity["input_reference"],input_row_fingerprint=identity["input_row_fingerprint"],
            policy_decision_references=sorted(member_users[key]),member_generation_seed=generation,
            seed_protocol_reference="frozen_member_generation_protocol",execution_state=STATE,
        ))
    aggregate_nodes=[]
    for key in sorted(aggregates):
        question,generation,action=key
        identity=identities[question]
        dependency_keys=[(question,generation,int(member)) for member in teams[action]]
        aggregate_nodes.append(dict(
            node_id=_node_id("aggregate",key),node_kind="AGGREGATE_OUTPUT_METADATA",aggregate_key=list(key),
            question_id=identity["question_id"],canonical_question_sha256=identity["canonical_question_sha256"],
            input_reference=identity["input_reference"],input_row_fingerprint=identity["input_row_fingerprint"],
            dependency_member_keys=[list(item) for item in dependency_keys],
            dependency_node_ids=[_node_id("member",item) for item in dependency_keys],
            policy_decision_references=sorted(aggregate_users[key]),
            member_generation_seed=generation,aggregator_invocation_seed=AGGREGATOR_INVOCATION_SEED,
            aggregator_operator_fingerprint=OPERATOR,seed_protocol_reference="frozen_aggregation_seed_protocol",
            execution_state=STATE,
        ))
    return dict(
        schema="e18_approval_only_execution_metadata_dag_v1",status=STATE,proposed_question_state=PROPOSED_STATE,
        native_worker_consumption_allowed=False,native_queue_registration_performed=False,
        frozen_decision_references=decisions,question_clusters=clusters,
        member_nodes=member_nodes,aggregate_nodes=aggregate_nodes,
        cache_reuse_application="FULL_LOGICAL_GRAPH_UNPRUNED; later approved materialization may apply a hash-bound reconciliation",
    )


def qualify_review_dag(graph,actions,teams,seeds):
    """Validate exact dependencies, policy coverage and two-layer topology."""
    actions=np.asarray(actions)
    teams=np.asarray(teams)
    expected_members,expected_aggregates=_keys(actions,teams,seeds)
    member_map={tuple(node["member_key"]):node for node in graph["member_nodes"]}
    aggregate_map={tuple(node["aggregate_key"]):node for node in graph["aggregate_nodes"]}
    if len(member_map)!=len(graph["member_nodes"]) or len(aggregate_map)!=len(graph["aggregate_nodes"]):
        raise ValueError("Repeated node key in review DAG")
    if set(member_map)!=expected_members or set(aggregate_map)!=expected_aggregates:
        raise ValueError("Review DAG differs from the frozen action union")
    used_members=set()
    policy_seed_links=0
    for key,node in aggregate_map.items():
        question,generation,action=key
        cluster=graph["question_clusters"][question]
        if node["node_id"]!=_node_id("aggregate",key) or node["node_kind"]!="AGGREGATE_OUTPUT_METADATA" or node["execution_state"]!=STATE:
            raise ValueError("Aggregate node identity or approval state differs")
        if any(node[field]!=cluster[field] for field in ("question_id","canonical_question_sha256","input_reference","input_row_fingerprint")):
            raise ValueError("Aggregate input identity differs from its original question cluster")
        dependencies=list(map(tuple,node["dependency_member_keys"]))
        expected=[(question,generation,int(member)) for member in teams[action]]
        if len(dependencies)!=3 or len(set(dependencies))!=3 or dependencies!=expected:
            raise ValueError("An aggregate must depend on exactly its three same-question/same-generation members")
        if node["dependency_node_ids"]!=[_node_id("member",item) for item in dependencies]:
            raise ValueError("Dependency identifiers disagree with dependency keys")
        if not set(dependencies).issubset(member_map):
            raise ValueError("Aggregate references an absent member")
        if node["aggregator_operator_fingerprint"]!=OPERATOR or node["aggregator_invocation_seed"]!=42:
            raise ValueError("Frozen aggregate operator/seed differs")
        users={f"decision/{question}/{policy}" for policy in range(4) if int(actions[policy,question])==action}
        if set(node["policy_decision_references"])!=users:
            raise ValueError("Frozen policy action coverage differs")
        policy_seed_links+=len(users)
        used_members.update(dependencies)
    if used_members!=expected_members:
        raise ValueError("An unneeded member was included in the logical graph")
    for key,node in member_map.items():
        question,generation,member=key
        cluster=graph["question_clusters"][question]
        if node["node_id"]!=_node_id("member",key) or node["node_kind"]!="MEMBER_OUTPUT_METADATA" or node["execution_state"]!=STATE:
            raise ValueError("Member node identity or approval state differs")
        if any(node[field]!=cluster[field] for field in ("question_id","canonical_question_sha256","input_reference","input_row_fingerprint")):
            raise ValueError("Member input identity differs from its original question cluster")
        users={f"decision/{question}/{policy}" for policy in range(4) if member in teams[int(actions[policy,question])]}
        if set(node["policy_decision_references"])!=users:
            raise ValueError("Member output consumers differ from frozen decisions")
        if node["member_generation_seed"]!=generation:
            raise ValueError("Member repeat identity differs")
    if len(graph["question_clusters"])!=actions.shape[1] or len(graph["frozen_decision_references"])!=4*actions.shape[1]:
        raise ValueError("Original question/policy decision cardinality differs")
    for question,cluster in enumerate(graph["question_clusters"]):
        if cluster["question_index"]!=question or cluster["member_generation_seeds"]!=list(seeds):
            raise ValueError("Question clusters do not retain all three dependent repeats")
        if cluster["policy_decision_references"]!=[f"decision/{question}/{policy}" for policy in range(4)]:
            raise ValueError("Question-level frozen decision references differ")
        for policy in range(4):
            reference=f"decision/{question}/{policy}"
            if graph["frozen_decision_references"][reference]["action_index"]!=int(actions[policy,question]):
                raise ValueError("A review decision changes a frozen action")
            for generation in seeds:
                if reference not in aggregate_map[(question,int(generation),int(actions[policy,question]))]["policy_decision_references"]:
                    raise ValueError("A policy selection lacks a required aggregate")
    if graph["status"]!=STATE or graph["proposed_question_state"]!=PROPOSED_STATE or graph["native_worker_consumption_allowed"] or graph["native_queue_registration_performed"]:
        raise ValueError("Review graph has acquired execution permission")
    return dict(
        status="PASS",independent_question_clusters=actions.shape[1],member_generation_repeats=len(seeds),
        question_generation_clusters=actions.shape[1]*len(seeds),gold_reveal_accounts=actions.shape[1],
        unique_member_nodes=len(member_map),unique_aggregate_nodes=len(aggregate_map),
        unique_total_nodes=len(member_map)+len(aggregate_map),dependency_edges=3*len(aggregate_map),
        frozen_policy_decisions=4*actions.shape[1],policy_generation_selection_coverage=policy_seed_links,
        all_policy_generation_selections_covered=policy_seed_links==4*actions.shape[1]*len(seeds),
        each_aggregate_has_exactly_three_members=True,all_dependencies_same_question_and_generation=True,
        topological_layers=["member output metadata","aggregate output metadata"],acyclic=True,
        question_repeats_counted_as_independent_questions=False,native_queue_writes=0,
    )


def _input_references(path,cohort):
    file_hash=sha256(path)
    result={}
    nonempty=0
    with Path(path).open("rb") as stream:
        for physical_line,raw in enumerate(stream):
            if not raw.strip():
                continue
            text=raw.decode("utf-8-sig")
            names={name for name,_,_ in object_fields(text)}
            allowed=ALLOWED_PROPOSAL_FIELDS if cohort=="proposed_new2600" else REQUIRED_INPUT_FIELDS
            if not REQUIRED_INPUT_FIELDS.issubset(names) or names-allowed:
                raise ValueError("Input-only source contains unexpected fields")
            row=project(text,allowed)
            fingerprint=canonical_hash(row["question"])
            if fingerprint!=row["canonical_question_sha256"] or row["question_id"] in result:
                raise ValueError("Input source identity or canonical content differs")
            if cohort=="proposed_new2600" and row["status"]!=PROPOSED_STATE:
                raise ValueError("Proposal source has changed admission state")
            source_frame=row.get("source_frame", "gsm8k_official_train_original_primary_frame" if row["task"]=="gsm8k" else "bbh_old_llmrouterbench_common_subpool")
            reference=dict(source_artifact=cohort,artifact_sha256=file_hash,
                           jsonl_nonempty_row_index=nonempty,jsonl_physical_line_index=physical_line,text_field="question")
            row_metadata=dict(question_id=row["question_id"],canonical_question_sha256=fingerprint,
                              task=row["task"],subtask=row["subtask"],source_split=row["source_split"],
                              source_frame=source_frame,input_reference=reference)
            row_metadata["input_row_fingerprint"]=digest(row_metadata)
            result[row["question_id"]]=row_metadata
            nonempty+=1
    return result,dict(path=str(path),sha256=file_hash,questions=nonempty,decoded_fields="input-only allowlist; text used transiently for existing canonical verification",input_text_persisted_in_graph=False)


def _generation_protocol(qualification):
    pool_path=ASSETS/"configs/pool_v1.json"
    pool=read_json(pool_path)
    member_records=[]
    common=None
    for model in sorted(pool["models"],key=lambda item:int(item["index"])):
        path=ASSETS/model["config"]
        config=read_json(path)
        settings={key:config[key] for key in ("temperature","top_p","max_input_tokens","max_new_tokens","quantization","enable_thinking")}
        expected=dict(temperature=.7,top_p=.95,max_input_tokens=4096,max_new_tokens=2048,quantization="nf4",enable_thinking=False)
        if settings!=expected:
            raise ValueError("Frozen member generation protocol differs")
        if common is not None and common!=settings:
            raise ValueError("Member generation protocol is not shared")
        common=settings
        member_records.append(dict(member_index=int(model["index"]),logical_model_fingerprint=model["fingerprint"],
                                   model_id=model["model_id"],revision=model["revision"],config_path=str(path),config_sha256=sha256(path)))
    operator=qualification["active_operator"]
    if operator["aggregator_fingerprint"]!=OPERATOR or operator["composite_identity_sha256"]!=OPERATOR:
        raise ValueError("Qualified frozen composite operator differs")
    config_path=ASSETS/"configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json"
    config=read_json(config_path)
    if config["temperature"]!=0 or config["top_p"]!=1 or config["max_input_tokens"]!=6144 or config["max_new_tokens"]!=2048:
        raise ValueError("Frozen aggregation generation configuration differs")
    return dict(
        frozen_member_generation_protocol=dict(member_generation_seeds=list(SEEDS),settings=common,members=member_records),
        frozen_aggregation_seed_protocol=dict(member_repeat_key_field="member_generation_seed",aggregator_invocation_seed=42,
            aggregator_operator_fingerprint=OPERATOR,base_model_configuration_fingerprint=operator["base_model_configuration_fingerprint"],
            prompt_version=operator["prompt_version"],config_path=str(config_path),config_sha256=sha256(config_path),
            config_settings={key:config[key] for key in ("temperature","top_p","max_input_tokens","max_new_tokens","quantization","enable_thinking")}),
    )


def main():
    cpu,wall=time.process_time(),time.perf_counter()
    union_path=PRIVATE/"FROZEN_CALL_UNION_3000.json"
    prediction_path=PRIVATE/"FROZEN3000_INPUT_PREDICTIONS.npz"
    budget_receipt=read_json(PRIVATE/"FROZEN_BUDGET_RECEIPT.json")
    if sha256(prediction_path)!=budget_receipt["predictions_sha256"]:
        raise ValueError("Frozen prediction artifact differs")
    with np.load(prediction_path,allow_pickle=False) as archive:
        if set(archive.files)!={"actions","new_head_scores","question_ids","canonical_hashes","policies"}:
            raise ValueError("Unexpected frozen input-prediction archive members")
        frozen={key:archive[key].copy() for key in ("actions","question_ids","canonical_hashes","policies")}
    if tuple(map(str,frozen["policies"]))!=POLICIES or frozen["actions"].shape!=(4,3000):
        raise ValueError("Frozen policy names/order/shape differ")
    union=read_json(union_path)
    if union["questions"]!=3000 or union["policies"]!=4 or tuple(union["seeds"])!=SEEDS or union["authorizes_evaluation"]:
        raise ValueError("Frozen union scope or permission differs")
    teams=np.asarray(union["teams"],dtype=np.int64)
    expected_members,expected_aggregates=_keys(frozen["actions"],teams,SEEDS)
    if set(map(tuple,union["member_keys"]))!=expected_members or set(map(tuple,union["aggregate_keys"]))!=expected_aggregates:
        raise ValueError("Frozen actions and frozen union disagree")
    if len(union["member_keys"])!=47286 or len(union["aggregate_keys"])!=30921 or union["gold_question_indices"]!=list(range(3000)):
        raise ValueError("Original frozen47286/30921/3000 scope changed")
    if union["counts"]!=dict(member_calls=47286,aggregate_calls=30921,scoring_ops=30921,gold_questions=3000):
        raise ValueError("Frozen resource counts differ")
    source_paths=dict(
        existing_fresh400=ASSETS/"artifacts/private/routing_redesign_20261104/fresh_dev400/inputs_only.jsonl",
        proposed_new2600=PRIVATE/"proposed_new2600_inputs_only.jsonl",
    )
    source_maps={}
    source_records={}
    for cohort,path in source_paths.items():
        source_maps[cohort],source_records[cohort]=_input_references(path,cohort)
    if len(source_maps["existing_fresh400"])!=400 or len(source_maps["proposed_new2600"])!=2600:
        raise ValueError("Input source cohort sizes differ")
    identities=[]
    for question,(qid,fingerprint) in enumerate(zip(frozen["question_ids"],frozen["canonical_hashes"])):
        cohort="existing_fresh400" if question<400 else "proposed_new2600"
        row=source_maps[cohort].get(str(qid))
        if row is None or row["canonical_question_sha256"]!=str(fingerprint):
            raise ValueError("Frozen question and original input-source identity differ")
        identities.append(row)
    graph=build_review_dag(frozen["actions"],teams,SEEDS,POLICIES,identities)
    topology=qualify_review_dag(graph,frozen["actions"],teams,SEEDS)
    qualification_path=PUBLIC/"FROZEN_MODEL_QUALIFICATION_v3.json"
    qualification=read_json(qualification_path)
    if qualification["status"]!="PASS":
        raise ValueError("Frozen operator/model qualification is unavailable")
    graph["generation_protocols"]=_generation_protocol(qualification)
    graph["source_artifacts"]=source_records
    graph["frozen_policy_decision_artifact"]=dict(path=str(prediction_path),sha256=sha256(prediction_path),
        decoded_npz_members=["actions","question_ids","canonical_hashes","policies"],scores_decoded=False)
    graph["frozen_logical_union_artifact"]=dict(path=str(union_path),sha256=sha256(union_path))
    graph["operator_qualification_artifact"]=dict(path=str(qualification_path),sha256=sha256(qualification_path))
    graph["counts"]=topology
    graph["reuse_option_bindings"]=[]
    for name in ("RECONCILED_MEMBER_CACHE_KEYS_LOCAL_v4.json","FINAL_VERIFIED_EXECUTION_REMAINDER.json"):
        path=PRIVATE/name
        if path.is_file():
            graph["reuse_option_bindings"].append(dict(path=str(path),sha256=sha256(path),applied_to_logical_graph=False,
                future_step="Explicit approved materialization revalidates reconciliation and may satisfy existing member/aggregate nodes"))
    graph["admission_and_execution_gate"]=dict(question_admission=PROPOSED_STATE,execution=STATE,
        source_frame_extension_acceptance=STATE,native_queue_write_authorized=False,label_or_reference_access_authorized=False)
    graph["preparation_cost"]=dict(head_fits=0,pca_or_scaler_fits=0,encoder_forwards=0,llm_calls=0,score_operations=0)
    # The private source-reference table points to originals. No question text or
    # answer is duplicated in this review graph.
    forbidden={"question","answer","reference","target","reward","gold","raw_output","prompt","job_id","logical_job_id","physical_job_id","owned_output_root","remote_bundle","worker_command"}
    def check_keys(value):
        if isinstance(value,dict):
            if forbidden&set(value):
                raise ValueError("Review graph acquired a payload/native-worker field")
            for item in value.values():check_keys(item)
        elif isinstance(value,list):
            for item in value:check_keys(item)
    check_keys(graph)
    output_root=PRIVATE/"proposed_execution_review_only"
    if output_root.exists() or (PUBLIC/"PROPOSED_EXECUTION_DAG_QUALIFICATION.json").exists():
        raise FileExistsError("Existing approval-only DAG is immutable")
    output_root.mkdir(parents=True)
    graph_path=output_root/"PROPOSED_EXECUTION_DAG_METADATA.json"
    write_json(graph_path,graph)
    result=dict(schema="e18_proposed_execution_dag_qualification_v1",status="PASS_APPROVAL_REQUIRED",
        admission_state=PROPOSED_STATE,topology=topology,graph_sha256=sha256(graph_path),
        graph_bytes=graph_path.stat().st_size,source_hashes=dict(
            frozen_call_union=sha256(union_path),frozen_policy_decisions=sha256(prediction_path),
            existing_fresh400_inputs=source_records["existing_fresh400"]["sha256"],
            proposed2600_inputs=source_records["proposed_new2600"]["sha256"],
            operator_qualification=sha256(qualification_path),preparation_code=sha256(Path(__file__))),
        payload_field_qualification="PASS_INPUT_TEXT_ANSWER_LABEL_NATIVE_JOB_FIELDS_ABSENT",
        public_identity_key_values_included=0,full_logical_graph_preserved=True,cache_reuse_applied=False,
        native_queue_writes=0,head_fits=0,encoder_forwards=0,llm_calls=0,score_operations=0,
        cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall)
    rendered=json.dumps(result,ensure_ascii=False)
    if any(str(value) in rendered for field in ("question_ids","canonical_hashes") for value in frozen[field]):
        raise ValueError("Public qualification includes a private input identity")
    write_json(PUBLIC/"PROPOSED_EXECUTION_DAG_QUALIFICATION.json",result)
    write_json(output_root/"REVIEW_DAG_FREEZE_RECEIPT.json",dict(
        status=STATE,graph_sha256=sha256(graph_path),public_qualification_sha256=sha256(PUBLIC/"PROPOSED_EXECUTION_DAG_QUALIFICATION.json"),
        source_sha256=sha256(Path(__file__)),native_queue_registration_performed=False,
    ))
    print(json.dumps({key:result[key] for key in ("status","topology","graph_sha256","graph_bytes","native_queue_writes","llm_calls")},ensure_ascii=False))


if __name__=="__main__":
    main()
