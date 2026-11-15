"""Synthetic topology and frozen-policy coverage controls for review-only DAGs."""
import copy
from itertools import combinations
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_proposed_execution_dag import build_review_dag, qualify_review_dag


class ReviewDagTests(unittest.TestCase):
    def setUp(self):
        self.actions=np.array([[0,19],[0,19],[1,19],[19,0]],dtype=int)
        self.teams=np.array(list(combinations(range(6),3)))
        self.seeds=(42,137,2024)
        self.policies=("Ridge","SPO_coordinate","TSPO_uniform","GlobalFixed")
        self.identities=[dict(question_id=f"fixture{q}",canonical_question_sha256=f"synthetic{q}",
            input_reference=dict(source_artifact="synthetic",jsonl_nonempty_row_index=q),
            input_row_fingerprint=f"fingerprint{q}",source_frame="synthetic",task="gsm8k",subtask=None) for q in range(2)]
        self.graph=build_review_dag(self.actions,self.teams,self.seeds,self.policies,self.identities)

    def test_shared_members_and_aggregates_deduplicate_exactly(self):
        result=qualify_review_dag(self.graph,self.actions,self.teams,self.seeds)
        self.assertEqual(result["unique_member_nodes"],36)
        self.assertEqual(result["unique_aggregate_nodes"],15)
        self.assertEqual(result["dependency_edges"],45)
        self.assertEqual(result["policy_generation_selection_coverage"],24)
        self.assertEqual(result["gold_reveal_accounts"],2)
        self.assertTrue(result["acyclic"])

    def test_missing_member_fails(self):
        self.graph["member_nodes"].pop()
        with self.assertRaises(ValueError):qualify_review_dag(self.graph,self.actions,self.teams,self.seeds)

    def test_cross_generation_dependency_fails(self):
        self.graph["aggregate_nodes"][0]["dependency_member_keys"][0][1]=137
        with self.assertRaises(ValueError):qualify_review_dag(self.graph,self.actions,self.teams,self.seeds)

    def test_duplicate_aggregate_key_fails(self):
        self.graph["aggregate_nodes"].append(copy.deepcopy(self.graph["aggregate_nodes"][0]))
        with self.assertRaises(ValueError):qualify_review_dag(self.graph,self.actions,self.teams,self.seeds)

    def test_missing_shared_policy_reference_fails(self):
        node=next(node for node in self.graph["aggregate_nodes"] if len(node["policy_decision_references"])>1)
        node["policy_decision_references"].pop()
        with self.assertRaises(ValueError):qualify_review_dag(self.graph,self.actions,self.teams,self.seeds)

    def test_aggregate_seed_is_distinct_from_member_repeat_identity(self):
        node=next(node for node in self.graph["aggregate_nodes"] if node["member_generation_seed"]==2024)
        self.assertEqual(node["aggregate_key"][1],2024)
        self.assertEqual(node["aggregator_invocation_seed"],42)

    def test_native_queue_permission_fails(self):
        self.graph["native_worker_consumption_allowed"]=True
        with self.assertRaises(ValueError):qualify_review_dag(self.graph,self.actions,self.teams,self.seeds)

    def test_member_node_identifier_and_original_identity_are_verified(self):
        self.graph["member_nodes"][0]["node_id"]="different-id"
        with self.assertRaises(ValueError):qualify_review_dag(self.graph,self.actions,self.teams,self.seeds)
        self.graph=build_review_dag(self.actions,self.teams,self.seeds,self.policies,self.identities)
        self.graph["member_nodes"][0]["canonical_question_sha256"]="different-input"
        with self.assertRaises(ValueError):qualify_review_dag(self.graph,self.actions,self.teams,self.seeds)

    def test_graph_does_not_contain_original_question_text_or_native_job_fields(self):
        forbidden={"question","answer","gold","reference","reward","job_id","logical_job_id","physical_job_id"}
        for collection in (self.graph["member_nodes"],self.graph["aggregate_nodes"],self.graph["question_clusters"]):
            for node in collection:self.assertFalse(forbidden&set(node))


if __name__=="__main__":
    unittest.main()
