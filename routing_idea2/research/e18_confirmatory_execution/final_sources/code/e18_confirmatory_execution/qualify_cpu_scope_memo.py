"""Synthetic qualification of the independent CPU scope identity memo wrapper."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

WRAPPER_PATH = Path(__file__).with_name("cpu_scope_memo.py")
WRAPPER_IMPORT_SHA256 = hashlib.sha256(WRAPPER_PATH.read_bytes()).hexdigest()

import cpu_scope_memo as memo_api

if hashlib.sha256(WRAPPER_PATH.read_bytes()).hexdigest() != WRAPPER_IMPORT_SHA256:
    raise RuntimeError("The wrapper source changed while it was imported")


def canonical_digest(value):
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_readonly(test, value):
    with test.assertRaises(TypeError):
        value["added"] = True
    with test.assertRaises(TypeError):
        value["nested"]["values"].append(4)


class FrozenScopeQualification(unittest.TestCase):
    def value(self):
        return {"nested": {"values": [1, {"flag": True}, None]}, "label": "合成"}

    def test_recursive_freeze_preserves_canonical_identity(self):
        source = self.value()
        frozen = memo_api.freeze_scope(source)
        self.assertIsInstance(frozen, memo_api.FrozenDict)
        self.assertIsInstance(frozen["nested"], memo_api.FrozenDict)
        self.assertIsInstance(frozen["nested"]["values"], memo_api.FrozenList)
        self.assertIsInstance(frozen["nested"]["values"][1], memo_api.FrozenDict)
        self.assertEqual(canonical_digest(frozen), canonical_digest(source))
        assert_readonly(self, frozen)

    def test_original_mutable_alias_cannot_change_frozen_scope(self):
        source = self.value()
        frozen = memo_api.freeze_scope(source)
        expected = canonical_digest(frozen)
        source["nested"]["values"].append("changed synthetic input")
        source["label"] = "changed"
        self.assertEqual(canonical_digest(frozen), expected)

    def test_deepcopy_preserves_recursive_readonly(self):
        frozen = memo_api.freeze_scope(self.value())
        duplicate = copy.deepcopy(frozen)
        self.assertEqual(duplicate, frozen)
        assert_readonly(self, duplicate)

    def test_dictionary_mutation_entry_points_are_rejected(self):
        operations = [
            lambda value: value.__setitem__("added", 1),
            lambda value: value.__delitem__("nested"),
            lambda value: value.clear(),
            lambda value: value.pop("nested"),
            lambda value: value.popitem(),
            lambda value: value.setdefault("added", 1),
            lambda value: value.update({"added": 1}),
            lambda value: value.__ior__({"added": 1}),
        ]
        for operation in operations:
            with self.subTest(operation=operations.index(operation)):
                frozen = memo_api.freeze_scope(self.value())
                with self.assertRaises(TypeError):
                    operation(frozen)

    def test_list_mutation_entry_points_are_rejected(self):
        operations = [
            lambda value: value.__setitem__(0, 8),
            lambda value: value.__setitem__(slice(None), [8]),
            lambda value: value.__delitem__(0),
            lambda value: value.append(8),
            lambda value: value.extend([8]),
            lambda value: value.insert(0, 8),
            lambda value: value.pop(),
            lambda value: value.remove(1),
            lambda value: value.clear(),
            lambda value: value.reverse(),
            lambda value: value.sort(),
            lambda value: value.__iadd__([8]),
            lambda value: value.__imul__(2),
        ]
        for index, operation in enumerate(operations):
            with self.subTest(operation=index):
                frozen = memo_api.freeze_scope([1, 2, 3])
                with self.assertRaises(TypeError):
                    operation(frozen)

    def test_refreezing_and_json_roundtrip_preserve_values(self):
        source = self.value()
        frozen = memo_api.freeze_scope(memo_api.freeze_scope(source))
        self.assertEqual(json.loads(json.dumps(frozen, ensure_ascii=False)), source)
        assert_readonly(self, frozen)

    def test_reinitialization_and_attributes_cannot_unseal_containers(self):
        for frozen in (memo_api.freeze_scope({"a": 1}), memo_api.freeze_scope([1])):
            with self.subTest(container=type(frozen).__name__):
                with self.assertRaises(TypeError):
                    frozen.__init__()
                with self.assertRaises(TypeError):
                    frozen._sealed = False
                with self.assertRaises(TypeError):
                    del frozen._sealed


class OriginalFixtureQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import qualify_final_scoring as fixture_api

        fixture = fixture_api.FinalScoringQualification(
            "test_toy_end_to_end_uses_frozen_scorer_and_sparse_projection"
        )
        cls.addClassCleanup(fixture.doCleanups)
        fixture.setUpClass()
        fixture.setUp()
        cls.fixture = fixture
        cls.final = fixture_api.final

    def context_manager(self, manifest=None):
        return memo_api.ScopeMemo(
            self.final,
            self.fixture.manifest if manifest is None else manifest,
            software_test_only=True,
        )

    def scoring_adapter(self, context):
        provider_api = self.final.load_reference_api(
            self.fixture.manifest["reference_adapter_ref"]
        )
        provider = provider_api.E18ReferenceProvider(
            context["reference_scope"],
            context["approval_ref"],
            software_test_only=True,
        )
        runtime = self.fixture.root / self._testMethodName
        scorer = context["validator"].SparseScoringAdapter(
            context["approval_ref"],
            context["scoring_scope"],
            provider,
            runtime / "journal.sqlite3",
            runtime / "receipts",
            software_test_only=True,
            scorer_source_root=context["adapter"].apis.root / "src",
        )
        self.addCleanup(scorer.close)
        return provider, scorer

    def test_registered_scope_and_mutable_equal_object_have_correct_digests(self):
        source = self.fixture.scope
        expected = canonical_digest(source)
        with self.context_manager() as memo:
            registered = memo.register_scope(source, "execution")
            self.assertIsInstance(registered, memo_api.FrozenDict)
            self.assertEqual(memo.digest(registered), expected)
            self.assertEqual(memo.digest(registered), expected)
            shadow = json.loads(json.dumps(registered))
            self.assertEqual(memo.digest(shadow), expected)
            shadow["authorization_state"] = "synthetic changed scope"
            self.assertEqual(memo.digest(shadow), canonical_digest(shadow))
            self.assertNotEqual(memo.digest(shadow), expected)
            self.assertIsInstance(memo.stats(), dict)

    def test_unregistered_nested_mutable_values_are_never_identity_cached(self):
        value = {"synthetic": [1, 2]}
        with self.context_manager() as memo:
            first = memo.digest(value)
            value["synthetic"].append(3)
            second = memo.digest(value)
            self.assertNotEqual(first, second)
            self.assertEqual(second, canonical_digest(value))

    def test_module_fallback_calls_captured_original_and_preserves_its_exception(self):
        original = self.fixture.module.canonical_digest
        calls = []
        sentinel = object()

        class SyntheticFallbackError(Exception):
            pass

        def spy(value):
            calls.append(value)
            if value is sentinel:
                raise SyntheticFallbackError("synthetic original callable failure")
            return original(value)

        with patch.object(self.fixture.module, "canonical_digest", spy):
            with self.context_manager():
                context = self.fixture.gate()
                calls.clear()
                scope = context["adapter"].scope
                self.assertEqual(
                    self.fixture.module.canonical_digest(scope), canonical_digest(scope)
                )
                self.assertEqual(calls, [])
                native = {"synthetic_native": [1, 2, 3]}
                self.assertEqual(
                    self.fixture.module.canonical_digest(native), canonical_digest(native)
                )
                self.assertEqual(len(calls), 1)
                self.assertIs(calls[0], native)
                with self.assertRaisesRegex(
                    SyntheticFallbackError, "synthetic original callable failure"
                ):
                    self.fixture.module.canonical_digest(sentinel)
                self.assertIs(calls[-1], sentinel)

    def test_nonfinite_unknown_value_keeps_original_digest_failure(self):
        with self.context_manager() as memo:
            with self.assertRaises(ValueError):
                memo.digest({"synthetic": float("nan")})

    def test_source_reference_hash_mismatch_is_rejected(self):
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["scoring_scope_ref"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            with self.context_manager(manifest):
                self.fail("A mismatched scope source was accepted")

    def test_all_existing_module_callables_restore_after_exception(self):
        modules = [
            self.final,
            self.final.worker,
            self.final.audit_startup16,
            self.fixture.module,
        ]
        originals = [
            (module, name, value)
            for module in modules
            for name, value in vars(module).items()
            if callable(value)
        ]
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            with self.context_manager():
                self.fixture.gate()
                raise RuntimeError("synthetic interruption")
        for module, name, value in originals:
            with self.subTest(module=module.__name__, attribute=name):
                self.assertIs(getattr(module, name), value)

    def test_original_and_wrapped_complete_gates_match(self):
        original = self.fixture.gate()
        with self.context_manager() as memo:
            wrapped = self.fixture.gate()
            self.assertEqual(wrapped["gate"], original["gate"])
            self.assertEqual(set(wrapped["members"]), set(original["members"]))
            self.assertEqual(wrapped["receipts"], original["receipts"])
            self.assertEqual(wrapped["costs"], original["costs"])
            self.assertEqual(wrapped["gate"]["reference_values_read"], 0)
            self.assertEqual(wrapped["gate"]["scorer_calls"], 0)
            memo.verify_integrity()

    def test_original_and_wrapped_toy_scoring_and_resume_match(self):
        import numpy as np

        plain_root = self.fixture.root / "cpu_memo_parity_plain"
        memo_root = self.fixture.root / "cpu_memo_parity_wrapped"
        plain = self.final.execute_scoring(
            self.fixture.manifest,
            self.fixture.gate(),
            software_test_only=True,
            runtime_root=plain_root,
        )
        with self.context_manager() as memo:
            context = self.fixture.gate()
            wrapped = self.final.execute_scoring(
                self.fixture.manifest,
                context,
                software_test_only=True,
                runtime_root=memo_root,
            )
            resumed = self.final.execute_scoring(
                self.fixture.manifest,
                context,
                software_test_only=True,
                runtime_root=memo_root,
            )
            self.assertEqual(wrapped, plain)
            self.assertEqual(resumed, wrapped)
            self.assertEqual(wrapped["distinct_gold_accounts"], 4)
            self.assertEqual(wrapped["scoring_model_calls"], 0)
            self.assertGreater(memo.stats()["scope_cache_hits"], 0)
            self.assertEqual(memo.stats()["native_or_job_objects_cached"], 0)
            self.assertEqual(memo.stats()["approval_checks_skipped"], 0)
            self.assertEqual(memo.stats()["reference_source_scans_skipped"], 0)
            memo.verify_integrity()
        with np.load(plain_root / "FINAL_POLICY_REWARDS.npz", allow_pickle=False) as original:
            with np.load(memo_root / "FINAL_POLICY_REWARDS.npz", allow_pickle=False) as wrapped:
                self.assertEqual(set(original.files), set(wrapped.files))
                for key in original.files:
                    np.testing.assert_array_equal(original[key], wrapped[key])
                self.assertEqual(original["rewards"].shape, (4, 3, 4))

    def test_cached_score_still_rejects_approval_file_tampering(self):
        with self.context_manager():
            context = self.fixture.gate()
            provider, scorer = self.scoring_adapter(context)
            receipt = context["receipts"][0]
            version = context["reference_scope"]["reference_version"]
            scorer.score_one(receipt, version)
            before = provider.stats()
            path = Path(context["approval_ref"]["path"])
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\n")
                with self.assertRaises(ValueError):
                    scorer.score_one(receipt, version)
            finally:
                path.write_bytes(original)
            self.assertEqual(provider.stats(), before)

    def test_operation_withdrawal_still_denies_registered_scope(self):
        with self.context_manager() as memo:
            context = self.fixture.gate()
            scope = memo.register_scope(context["scoring_scope"], "scoring")
            approval = json.loads(Path(context["approval_ref"]["path"]).read_text())
            approval["operations"].remove("score_outputs")
            reference = self.fixture.artifact(
                self._testMethodName + ".json", approval
            )
            with self.assertRaises(PermissionError):
                context["module"].validate_scope_approval(
                    reference, scope, "score_outputs", software_test_only=True
                )

    def test_cached_score_still_rejects_native_record_tampering(self):
        with self.context_manager():
            context = self.fixture.gate()
            provider, scorer = self.scoring_adapter(context)
            receipt = context["receipts"][0]
            version = context["reference_scope"]["reference_version"]
            scorer.score_one(receipt, version)
            before = provider.stats()
            changed = copy.deepcopy(receipt)
            changed["native_record"]["result"]["completion_tokens"] += 1
            with self.assertRaises(ValueError):
                scorer.score_one(changed, version)
            self.assertEqual(provider.stats(), before)

    def test_synthetic_approval_cannot_authorize_production_mode(self):
        with self.context_manager() as memo:
            context = self.fixture.gate()
            scope = memo.register_scope(context["scoring_scope"], "scoring")
            with self.assertRaises(PermissionError):
                context["module"].validate_scope_approval(
                    context["approval_ref"], scope, "score_outputs", software_test_only=False
                )

    def test_incomplete_gate_stops_before_reference_provider(self):
        with self.context_manager():
            context = self.fixture.gate()
            context["gate"] = {**context["gate"], "unique_member_keys": 1}
            with patch.object(
                self.final,
                "load_reference_api",
                side_effect=AssertionError("Incomplete gate reached the provider"),
            ):
                with self.assertRaises(PermissionError):
                    self.final.execute_scoring(
                        self.fixture.manifest,
                        context,
                        software_test_only=True,
                        runtime_root=self.fixture.root / self._testMethodName,
                    )

    def test_builtin_mutator_bypass_is_detected_before_publication(self):
        for kind in ("dict", "list"):
            with self.subTest(container=kind):
                output = self.fixture.root / (self._testMethodName + kind + ".json")
                with self.assertRaises(ValueError):
                    with self.context_manager() as memo:
                        context = self.fixture.gate()
                        scope = context["scoring_scope"]
                        if kind == "dict":
                            dict.__setitem__(scope, "synthetic_bypass", True)
                        else:
                            list.__setitem__(scope["generation_seeds"], 0, 999)
                        with self.assertRaises(ValueError):
                            memo.verify_integrity()
                        with self.assertRaises(ValueError):
                            self.final.write_new_json(output, {"synthetic": True})
                        self.assertFalse(output.exists())

    def test_changed_scope_source_is_detected_before_publication(self):
        output = self.fixture.root / (self._testMethodName + ".json")
        with self.context_manager() as memo:
            path = Path(self.fixture.manifest["scoring_scope_ref"]["path"])
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\n")
                with self.assertRaises(ValueError):
                    memo.verify_integrity()
                with self.assertRaises(ValueError):
                    self.final.write_new_json(output, {"synthetic": True})
                self.assertFalse(output.exists())
            finally:
                path.write_bytes(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Qualification records are append-only")
    if args.asset_root is not None:
        os.environ["MAB_ASSET_ROOT"] = str(args.asset_root)
    stream = io.StringIO()
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(case)
        for case in (FrozenScopeQualification, OriginalFixtureQualification)
    )
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    wrapper_unchanged = file_sha(memo_api.__file__) == WRAPPER_IMPORT_SHA256
    record = dict(
        schema="e18_cpu_scope_memo_qualification_v1",
        status="PASS" if result.wasSuccessful() and wrapper_unchanged else "FAIL",
        tests_run=result.testsRun,
        errors=len(result.errors),
        failures=len(result.failures),
        wrapper_source_sha256=WRAPPER_IMPORT_SHA256,
        wrapper_source_unchanged_during_qualification=wrapper_unchanged,
        qualification_source_sha256=file_sha(__file__),
        software_test_only=True,
        synthetic_question_count=4,
        formal_gold_reads=0,
        real_question_or_reward_reads=0,
        actual_model_calls=0,
        actual_jobs_submitted=0,
        actual_process_signals=0,
        wall_seconds=time.perf_counter() - started,
        scope="Immutable identity memo and original frozen gate/scoring on synthetic fixture",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(record, ensure_ascii=False))
    if not result.wasSuccessful() or not wrapper_unchanged:
        print(stream.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
