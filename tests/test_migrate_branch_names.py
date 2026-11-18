"""Offline risk tests. Every API operation is mocked; no credentials or network."""
from __future__ import annotations

import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import migrate_branch_names as migration


def manifest():
    return {
        "schema_version": 1,
        "repository": "fallenplum/MAB",
        "operation": "remove_codex_prefix_preserve_commit",
        "branches": [
            {"old": f"codex/stage-{index:02d}", "new": f"stage-{index:02d}",
             "sha": f"{index + 1:040x}"}
            for index in range(31)
        ],
    }


class FakeAPI:
    def __init__(self, entries):
        self.branches = {e.old: migration.Branch(e.old, e.sha) for e in entries}
        self.branches["main"] = migration.Branch("main", "f" * 40)
        self.branches["experiment/unchanged"] = migration.Branch("experiment/unchanged", "e" * 40)
        self.rename_calls = []
        self.list_calls = 0
        self.get_calls = []
        self.list_hook = None
        self.get_hook = None

    def list_branches(self):
        self.list_calls += 1
        if self.list_hook:
            replacement = self.list_hook(self)
            if replacement is not None:
                return replacement
        return dict(self.branches)

    def get_branch(self, name):
        self.get_calls.append(name)
        if self.get_hook:
            replacement = self.get_hook(self, name)
            if replacement is not None:
                return replacement
        return self.branches.get(name)

    def rename(self, entry):
        self.rename_calls.append(entry)
        old = self.branches.pop(entry.old)
        new = migration.Branch(entry.new, old.sha)
        self.branches[entry.new] = new
        return new


class FakeResponse:
    def __init__(self, value, status=200):
        self.body = json.dumps(value).encode()
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return self.body[:limit]


class FakeOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class MigrationRiskTests(unittest.TestCase):
    def setUp(self):
        self.entries = migration.parse_manifest(manifest())

    def run_migration(self, api, **kwargs):
        return migration.migrate(self.entries, api, emit=lambda record: None,
                                 sleep=lambda delay: None, **kwargs)

    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(migration.MigrationError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_single_reviewed_branch_can_be_renamed(self):
        value = manifest()
        value["expected_count"] = 1
        value["branches"] = value["branches"][:1]
        entries = migration.parse_manifest(value)
        api = FakeAPI(entries)
        result = migration.migrate(entries, api, apply=True, emit=lambda record: None,
                                   sleep=lambda delay: None)
        self.assertEqual(result["renamed"], 1)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(api.branches["main"].sha, "f" * 40)
        self.assertEqual(api.branches[entries[0].new].sha, entries[0].sha)

    def test_explicit_count_must_be_valid_and_match_entries(self):
        for count in (0, -1, True, "1", 101):
            with self.subTest(count=count):
                value = manifest()
                value["expected_count"] = count
                self.assert_code("INVALID_EXPECTED_COUNT", migration.parse_manifest, value)
        value = manifest()
        value["expected_count"] = 1
        self.assert_code("INVALID_BRANCH_COUNT", migration.parse_manifest, value)

    def test_single_entry_without_explicit_count_keeps_original_guard(self):
        value = manifest()
        value["branches"] = value["branches"][:1]
        self.assert_code("INVALID_BRANCH_COUNT", migration.parse_manifest, value)

    def test_default_dry_run_with_token_makes_no_posts(self):
        api = FakeAPI(self.entries)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            path.write_text(json.dumps(manifest()))
            with mock.patch.dict(os.environ, {"GH_TOKEN": "TEST_ONLY_TOKEN"}, clear=True):
                with mock.patch.object(migration, "GitHubAPI", return_value=api) as constructor:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(migration.main(["--manifest", str(path)]), 0)
            constructor.assert_called_once_with("TEST_ONLY_TOKEN", allow_write=False)
        self.assertEqual(api.rename_calls, [])
        self.assertEqual(api.list_calls, 1)

    def test_apply_without_token_stops_before_any_api_access(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            path.write_text(json.dumps(manifest()))
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(migration, "GitHubAPI") as constructor:
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr):
                        code = migration.main(["--manifest", str(path), "--apply"])
            self.assertEqual(code, 1)
            constructor.assert_not_called()
            self.assertEqual(json.loads(stderr.getvalue()), {"error": "TOKEN_REQUIRED"})

    def test_write_guard_also_blocks_direct_api_post(self):
        api = migration.GitHubAPI("TEST_ONLY_TOKEN")
        api._opener = FakeOpener([])
        self.assert_code("WRITE_DISABLED", api.rename, self.entries[0])
        self.assertEqual(api._opener.requests, [])

    def test_unsafe_manifests_are_rejected(self):
        cases = []
        value = manifest(); value["repository"] = "another/repo"
        cases.append((value, "REPOSITORY_NOT_ALLOWED"))
        value = manifest(); value["branches"].pop()
        cases.append((value, "INVALID_BRANCH_COUNT"))
        value = manifest(); value["branches"][0]["old"] = "experiment/stage-00"
        cases.append((value, "SOURCE_PREFIX_NOT_ALLOWED"))
        value = manifest(); value["branches"][0]["new"] = "different"
        cases.append((value, "DESTINATION_NOT_ALLOWED"))
        value = manifest(); value["branches"][0].update(old="codex/main", new="main")
        cases.append((value, "MAIN_NOT_ALLOWED"))
        value = manifest(); value["branches"][0].update(old="codex/codex/x", new="codex/x")
        cases.append((value, "DESTINATION_NOT_ALLOWED"))
        value = manifest(); value["branches"][0].update(old="codex/../x", new="../x")
        cases.append((value, "INVALID_BRANCH_NAME"))
        value = manifest(); value["branches"][0]["sha"] = "not-a-sha"
        cases.append((value, "INVALID_EXPECTED_SHA"))
        value = manifest(); value["branches"][-1] = deepcopy(value["branches"][0])
        cases.append((value, "DUPLICATE_BRANCH_MAPPING"))
        value = manifest(); value["branches"][1].update(old="codex/stage-00/child", new="stage-00/child")
        cases.append((value, "DESTINATION_MAPPING_REF_CONFLICT"))
        for value, code in cases:
            with self.subTest(code=code):
                self.assert_code(code, migration.parse_manifest, value)

    def test_late_preflight_collision_stops_before_first_post(self):
        api = FakeAPI(self.entries)
        entry = self.entries[-1]
        api.branches[entry.new] = migration.Branch(entry.new, entry.sha)
        self.assert_code("DESTINATION_ALREADY_EXISTS", self.run_migration, api, apply=True)
        self.assertEqual(api.rename_calls, [])

    def test_late_preflight_source_drift_stops_before_first_post(self):
        api = FakeAPI(self.entries)
        entry = self.entries[-1]
        api.branches[entry.old] = migration.Branch(entry.old, "d" * 40)
        self.assert_code("SOURCE_SHA_OR_NAME_CHANGED", self.run_migration, api, apply=True)
        self.assertEqual(api.rename_calls, [])

    def test_existing_ref_parent_is_a_collision(self):
        entries = list(self.entries)
        entries[0] = migration.Entry("codex/family/child", "family/child", entries[0].sha)
        api = FakeAPI(entries)
        api.branches["family"] = migration.Branch("family", "d" * 40)
        self.assert_code("DESTINATION_REF_CONFLICT", migration.preflight, entries, api.list_branches())
        self.assertEqual(api.rename_calls, [])

    def test_immediate_source_read_detects_sha_and_name_changes(self):
        for wrong in (
            migration.Branch(self.entries[0].old, "d" * 40),
            migration.Branch(self.entries[0].new, self.entries[0].sha),
        ):
            with self.subTest(branch=wrong.name):
                api = FakeAPI(self.entries)
                api.get_hook = lambda this, name: wrong if name == self.entries[0].old else None
                expected = "BRANCH_SHA_MISMATCH" if wrong.name == self.entries[0].old else "BRANCH_NAME_MISMATCH"
                self.assert_code(expected, self.run_migration, api, apply=True)
                self.assertEqual(api.rename_calls, [])

    def test_resume_skips_only_absent_old_and_matching_new(self):
        api = FakeAPI(self.entries)
        for entry in self.entries[:7]:
            api.rename(entry)
        api.rename_calls.clear()
        result = self.run_migration(api, apply=True)
        self.assertEqual(result["already_completed"], 7)
        self.assertEqual(result["renamed"], 24)
        self.assertEqual(api.rename_calls, list(self.entries[7:]))
        self.assertEqual(api.branches["main"].sha, "f" * 40)
        self.assertEqual(api.branches["experiment/unchanged"].sha, "e" * 40)
        for entry in self.entries:
            self.assertNotIn(entry.old, api.branches)
            self.assertEqual(api.branches[entry.new].sha, entry.sha)

    def test_resume_rejects_missing_pair_or_wrong_destination_sha(self):
        for state, code in (("missing", "SOURCE_AND_DESTINATION_MISSING"),
                            ("wrong", "RESUME_DESTINATION_CHANGED")):
            with self.subTest(state=state):
                api = FakeAPI(self.entries)
                entry = self.entries[-1]
                del api.branches[entry.old]
                if state == "wrong":
                    api.branches[entry.new] = migration.Branch(entry.new, "d" * 40)
                self.assert_code(code, self.run_migration, api, apply=True)
                self.assertEqual(api.rename_calls, [])

    def test_ambiguous_post_failure_is_not_retried_and_rerun_resumes(self):
        api = FakeAPI(self.entries)
        original = api.rename

        def lost_response(entry):
            original(entry)
            raise migration.MigrationError("HTTP_STATUS", 503)

        api.rename = lost_response
        self.assert_code("HTTP_STATUS", self.run_migration, api, apply=True)
        self.assertEqual(len(api.rename_calls), 1)
        api.rename = original
        result = self.run_migration(api, apply=True)
        self.assertEqual(result["already_completed"], 1)
        self.assertEqual(len(api.rename_calls), 31)
        self.assertEqual(len({entry.old for entry in api.rename_calls}), 31)

    def test_get_post_race_is_detected_after_write_and_stops(self):
        api = FakeAPI(self.entries)
        original = api.rename

        def concurrent_push(entry):
            api.branches[entry.old] = migration.Branch(entry.old, "d" * 40)
            return original(entry)

        api.rename = concurrent_push
        self.assert_code("BRANCH_SHA_MISMATCH", self.run_migration, api, apply=True)
        self.assertEqual(len(api.rename_calls), 1)
        # No false claim of atomicity: the moved branch contains the pushed SHA.
        self.assertEqual(api.branches[self.entries[0].new].sha, "d" * 40)

    def test_background_rename_propagation_only_retries_reads(self):
        api = FakeAPI(self.entries)
        entry = self.entries[0]
        pending = {"reads": 2}

        def stale_list(this):
            if entry.new in this.branches and pending["reads"]:
                pending["reads"] -= 1
                values = dict(this.branches)
                values[entry.old] = migration.Branch(entry.old, entry.sha)
                return values
            return None

        api.list_hook = stale_list
        sleeps = []
        result = migration.migrate(self.entries, api, apply=True, emit=lambda value: None,
                                   sleep=sleeps.append)
        self.assertEqual(sleeps, [1, 2])
        self.assertEqual(result["renamed"], 31)
        self.assertEqual(len(api.rename_calls), 31)

    def test_background_verification_is_bounded(self):
        api = FakeAPI(self.entries)
        entry = self.entries[0]

        def never_finishes(this):
            if entry.new in this.branches:
                values = dict(this.branches)
                values[entry.old] = migration.Branch(entry.old, entry.sha)
                return values
            return None

        api.list_hook = never_finishes
        sleeps = []
        self.assert_code("RENAME_VERIFICATION_INCOMPLETE", migration.migrate, self.entries, api,
                         apply=True, emit=lambda value: None, sleep=sleeps.append)
        self.assertEqual(sleeps, [1, 2, 4, 4, 4])
        self.assertEqual(len(api.rename_calls), 1)

    def test_final_listing_detects_reappearing_old_even_if_new_is_correct(self):
        api = FakeAPI(self.entries)
        complete_lists = {"count": 0}

        def final_source_reappears(this):
            if all(entry.new in this.branches for entry in self.entries):
                complete_lists["count"] += 1
                if complete_lists["count"] >= 2:
                    values = dict(this.branches)
                    entry = self.entries[0]
                    values[entry.old] = migration.Branch(entry.old, entry.sha)
                    return values
            return None

        api.list_hook = final_source_reappears
        self.assert_code("FINAL_SOURCE_STILL_PRESENT", self.run_migration, api, apply=True)
        self.assertEqual(len(api.rename_calls), 31)

    def test_http_redirects_are_not_followed_or_treated_as_absence(self):
        sentinel = "TEST_ONLY_SECRET"
        api = migration.GitHubAPI(sentinel)
        self.assertTrue(any(isinstance(h, migration.NoRedirect) for h in api._opener.handlers))
        for status in (301, 302, 307, 308, 403):
            with self.subTest(status=status):
                error = urllib.error.HTTPError(
                    "https://api.github.com/safe", status, sentinel,
                    {"Location": "https://elsewhere.invalid/" + sentinel},
                    io.BytesIO(sentinel.encode()),
                )
                api._opener = FakeOpener([error])
                caught = self.assert_code("HTTP_STATUS", api.get_branch, self.entries[0].old)
                self.assertEqual(caught.as_record(), {"error": "HTTP_STATUS", "http_status": status})
                self.assertNotIn(sentinel, str(caught))
                self.assertEqual(len(api._opener.requests), 1)
        handler = migration.NoRedirect()
        self.assertIsNone(handler.redirect_request(None, None, 302, "ignored", {}, "https://elsewhere.invalid"))

    def test_encoded_post_path_and_payload_stay_in_fixed_repository(self):
        entry = self.entries[0]
        api = migration.GitHubAPI("TEST_ONLY_TOKEN", allow_write=True)
        api._opener = FakeOpener([FakeResponse({"name": entry.new, "commit": {"sha": entry.sha}}, 201)])
        self.assertEqual(api.rename(entry), migration.Branch(entry.new, entry.sha))
        request = api._opener.requests[0]
        self.assertEqual(request.full_url, migration.API_ROOT + "/branches/codex%2Fstage-00/rename")
        self.assertEqual(request.method, "POST")
        self.assertEqual(json.loads(request.data), {"new_name": entry.new})
        self.assert_code("API_PATH_NOT_ALLOWED", api._request, "GET", "/secrets")
        self.assert_code("HTTP_METHOD_NOT_ALLOWED", api._request, "DELETE", "/branches/x")

    def test_branch_listing_is_paginated_without_server_link_urls(self):
        values = [{"name": f"branch-{index}", "commit": {"sha": f"{index + 1:040x}"}}
                  for index in range(101)]
        api = migration.GitHubAPI(None)
        api._opener = FakeOpener([FakeResponse(values[:100]), FakeResponse(values[100:])])
        branches = api.list_branches()
        self.assertEqual(len(branches), 101)
        self.assertEqual(
            [request.full_url for request in api._opener.requests],
            [migration.API_ROOT + "/branches?per_page=100&page=1",
             migration.API_ROOT + "/branches?per_page=100&page=2"],
        )

    def test_unexpected_errors_never_print_exception_or_token(self):
        sentinel = "TEST_ONLY_SECRET"
        api = FakeAPI(self.entries)
        api.list_branches = mock.Mock(side_effect=RuntimeError(sentinel))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            path.write_text(json.dumps(manifest()))
            with mock.patch.dict(os.environ, {"GITHUB_TOKEN": sentinel}, clear=True):
                with mock.patch.object(migration, "GitHubAPI", return_value=api):
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr):
                        code = migration.main(["--manifest", str(path), "--apply"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stderr.getvalue()), {"error": "UNEXPECTED_ERROR"})
        self.assertNotIn(sentinel, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

