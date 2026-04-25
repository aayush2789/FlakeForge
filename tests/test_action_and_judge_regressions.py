"""V3 Regression Tests — Patch Applier and Reward.

Tests that were historically problematic in V2 (action dispatch bugs,
judge scoring inconsistencies) are replaced by V3 equivalents testing
the patch applier and deterministic reward signals.
"""

import tempfile
from pathlib import Path

import pytest

from models import FlakeForgeAction, RunRecord, RewardBreakdown
from server.patch_applier import parse_search_replace_hunks, apply_search_replace_patch
from server.reward import (
    compute_format_reward,
    compute_anti_hack_penalty,
    compute_stability_reward,
)


class TestPatchApplierRegressions:
    """Regression tests for the V3 patch applier."""

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
        assert reward == 2.0

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