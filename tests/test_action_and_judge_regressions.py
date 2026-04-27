"""V3 Regression Tests — Patch Applier and Reward.

Tests that were historically problematic in V2 (action dispatch bugs,
judge scoring inconsistencies) are replaced by V3 equivalents testing
the patch applier and deterministic reward signals.
"""

import tempfile
from pathlib import Path

import pytest

from models import FlakeForgeAction, FlakeForgeObservation, RunRecord, RewardBreakdown
from agent.unified_agent import extract_patch, extract_think
from server.FlakeForge_environment import FlakeForgeEnvironment
from server.docker_runner import DockerTestRunner
from server.patch_applier import parse_search_replace_hunks, apply_search_replace_patch
from server.reward import (
    compute_format_reward,
    compute_anti_hack_penalty,
    compute_stability_reward,
    compute_verifiable_reward,
)


class TestPatchApplierRegressions:
    """Regression tests for the V3 patch applier."""

    def test_extracts_patch_from_xml_markdown_fence(self):
        """Models sometimes wrap the required XML blocks in a Markdown fence."""
        response = """```xml
<think>
Root Cause: async_wait (confidence: 0.95)
Evidence: timeout.
</think>
<patch>
--- source.py
<<<<<<< SEARCH
timeout = 0.05
=======
timeout = 0.5
>>>>>>> REPLACE
</patch>
```"""

        assert "Root Cause: async_wait" in extract_think(response)
        assert extract_patch(response) == """--- source.py
<<<<<<< SEARCH
timeout = 0.05
=======
timeout = 0.5
>>>>>>> REPLACE"""

    def test_whitespace_fuzzy_match(self):
        """Patch with slightly different indentation should still apply."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "example.py"
            test_file.write_text("    x = 1\n    y = 2\n    z = 3\n")

            # Search text without leading spaces
            patch = """--- example.py
<<<<<<< SEARCH
x = 1
=======
x = 42
>>>>>>> REPLACE"""
            result = apply_search_replace_patch(Path(tmpdir), patch)
            # Should succeed via fuzzy match
            assert result["success"] is True

    def test_file_not_found_graceful(self):
        """Patching a nonexistent file should not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            patch = """--- nonexistent.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE"""
            result = apply_search_replace_patch(Path(tmpdir), patch)
            assert result["success"] is False

    def test_empty_search_block_ignored(self):
        """Empty search blocks should be silently skipped."""
        patch = """--- test.py
<<<<<<< SEARCH

=======
new_code
>>>>>>> REPLACE"""
        hunks = parse_search_replace_hunks(patch)
        assert len(hunks) == 0  # Empty search should be skipped

    def test_multi_hunk_atomicity(self):
        """If second hunk fails, first should be rolled back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.py"
            file1.write_text("original1 = True\n")

            patch = """--- file1.py
<<<<<<< SEARCH
original1 = True
=======
modified1 = True
>>>>>>> REPLACE

--- file2.py
<<<<<<< SEARCH
this_doesnt_exist
=======
replacement
>>>>>>> REPLACE"""
            result = apply_search_replace_patch(Path(tmpdir), patch)
            assert result["success"] is False

    def test_protected_file_patch_is_rejected(self):
        """Infrastructure/config files should not be patchable by model output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text("[tool.pytest.ini_options]\n", encoding="utf-8")

            patch = """--- pyproject.toml
<<<<<<< SEARCH
[tool.pytest.ini_options]
=======
[tool.pytest.ini_options]
addopts = "-k not flaky"
>>>>>>> REPLACE"""
            result = apply_search_replace_patch(Path(tmpdir), patch)

            assert result["success"] is False
            assert result["protected_file"] is True
            assert "protected_file" in str(result["error"])

    def test_noop_patch_is_marked(self):
        """A syntactically applied patch with no diff should be visible to reward."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.py"
            source.write_text("value = 1\n", encoding="utf-8")

            patch = """--- source.py
<<<<<<< SEARCH
value = 1
=======
value = 1
>>>>>>> REPLACE"""
            result = apply_search_replace_patch(Path(tmpdir), patch)

            assert result["success"] is True
            assert result["noop"] is True
            assert result["lines_changed"] == 0


class TestRunnerRegressions:
    """Regression tests for pytest result parsing."""

    def test_quiet_pytest_success_counts_as_passed(self):
        """Quiet pytest output may omit '1 passed', but return code 0 is success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (root / "pytest.ini").write_text("[pytest]\naddopts = -q\n", encoding="utf-8")
            (tests_dir / "test_ok.py").write_text(
                "def test_ok():\n    assert True\n",
                encoding="utf-8",
            )

            runner = DockerTestRunner(str(root))
            # This test targets pytest stdout/RC parsing, not per-repo pip bootstrap.
            runner._deps_checked = True
            runner._deps_ready = True
            result = runner.run_test("tests/test_ok.py::test_ok")

            assert result.passed is True


class TestSourceDetectionRegressions:
    """Regression tests for source-under-test discovery."""

    def test_src_layout_import_is_read_as_source_under_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package = root / "src" / "pkg"
            tests_dir = root / "tests"
            package.mkdir(parents=True)
            tests_dir.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "service.py").write_text(
                "def fetch():\n    return {'success': True}\n",
                encoding="utf-8",
            )
            (tests_dir / "test_service.py").write_text(
                "from pkg.service import fetch\n\n"
                "def test_fetch():\n"
                "    assert fetch()['success']\n",
                encoding="utf-8",
            )

            env = FlakeForgeEnvironment(
                repo_path=str(root),
                test_identifier="tests/test_service.py::test_fetch",
                runner=object(),
            )

            _, source = env._read_sources()

            assert "def fetch" in source


class TestRewardRegressions:
    """Regression tests for V3 reward signals."""

    def test_no_crash_on_empty_action(self):
        """Reward computation should handle empty action gracefully."""
        action = FlakeForgeAction(raw_response="")
        reward = compute_format_reward(action)
        assert reward == 0.0

    def test_stability_edge_case_100_percent(self):
        """100% pass rate gives terminal bonus."""
        reward = compute_stability_reward(0.5, 1.0)
        assert reward == 2.25

    def test_anti_hack_assertion_deletion(self):
        """Deleting assertions should be heavily penalized."""
        patch = """<<<<<<< SEARCH
    assert result == expected
    assert len(data) > 0
=======
    pass
>>>>>>> REPLACE"""
        penalty = compute_anti_hack_penalty(patch, ["test.py"], 3)
        assert penalty < -0.5  # Two assertions removed

    def test_anti_hack_broad_except(self):
        """Adding bare except: should be penalized."""
        patch = """<<<<<<< SEARCH
result = func()
=======
try:
    result = func()
except:
    result = None
>>>>>>> REPLACE"""
        penalty = compute_anti_hack_penalty(patch, ["test.py"], 5)
        assert penalty < 0

    def test_reward_penalizes_noop_patch(self):
        action = FlakeForgeAction(
            raw_response="...",
            think_text="Root Cause: async_wait (confidence: 0.9)",
            patch_text="<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 1\n>>>>>>> REPLACE",
            predicted_category="async_wait",
        )
        obs = FlakeForgeObservation(
            episode_id="e",
            test_identifier="tests/test_x.py::test_x",
            step=1,
            steps_remaining=1,
            test_function_source="",
            source_under_test="",
        )

        reward = compute_verifiable_reward(
            action=action,
            observation=obs,
            patch_result={
                "success": True,
                "files_modified": ["source.py"],
                "lines_changed": 0,
                "noop": True,
            },
            post_run_results=[{"passed": False, "error_type": "AssertionError"}],
            baseline_pass_rate=0.0,
            pre_entropy=0.0,
        )

        assert reward.noop_patch_penalty == -0.5
        assert reward.to_dict()["noop_patch"] == -0.5

    def test_reward_penalizes_regression(self):
        action = FlakeForgeAction(
            raw_response="...",
            think_text="Root Cause: nondeterminism (confidence: 0.9)",
            patch_text="<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE",
            predicted_category="nondeterminism",
        )
        obs = FlakeForgeObservation(
            episode_id="e",
            test_identifier="tests/test_x.py::test_x",
            step=1,
            steps_remaining=1,
            test_function_source="",
            source_under_test="",
        )

        reward = compute_verifiable_reward(
            action=action,
            observation=obs,
            patch_result={
                "success": False,
                "files_modified": [],
                "lines_changed": 0,
            },
            post_run_results=[{"passed": True, "error_type": None}],
            baseline_pass_rate=0.0,
            pre_entropy=0.0,
            regression_detected=True,
        )

        assert reward.regression_penalty < 0.0
        assert reward.terminal_bonus == 0.0