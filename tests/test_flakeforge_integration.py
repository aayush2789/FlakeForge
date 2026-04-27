"""FlakeForge Integration Tests."""

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from models import (
    FlakeForgeAction,
    FlakeForgeObservation,
    FlakeForgeState,
    RunRecord,
    PatchRecord,
    RewardBreakdown,
    ROOT_CAUSE_TYPES,
    failure_mode_entropy,
)
from agent.unified_agent import (
    extract_think,
    extract_patch,
    extract_category_from_think,
    extract_confidence_from_think,
    infer_category_from_patch,
    build_unified_prompt,
)
from server.patch_applier import parse_search_replace_hunks, apply_search_replace_patch
from server.reward import (
    compute_format_reward,
    compute_compile_reward,
    compute_stability_reward,
    compute_causal_proximity_reward,
    compute_entropy_reward,
    compute_anti_hack_penalty,
    compute_reasoning_consistency,
)


class TestThinkPatchParsing:
    """Test extraction of <think> and <patch> blocks from model output."""

    def test_extract_both_blocks(self):
        response = """<think>
Root Cause: async_wait (confidence: 0.85)
Evidence: TimeoutError at test_flaky.py:12
Strategy: Increase timeout
</think>

<patch>
--- tests/test_flaky.py
<<<<<<< SEARCH
    result = await asyncio.wait_for(fetch(), timeout=0.05)
=======
    result = await asyncio.wait_for(fetch(), timeout=0.5)
>>>>>>> REPLACE
</patch>"""

        assert "Root Cause: async_wait" in extract_think(response)
        assert "<<<<<<< SEARCH" in extract_patch(response)

    def test_extract_single_json_response(self):
        response = """{
  "think": {
    "claims": [
      {
        "claim_id": "c1",
        "category": "async_wait",
        "entity": "fetch",
        "location": "tests/test_flaky.py::test_fetch",
        "ast_node_type": "Call",
        "polarity": "present",
        "predicted_effect": "The test should stop timing out.",
        "reason": "The wait_for timeout is too aggressive."
      }
    ],
    "confidence": 0.85
  },
  "patch": {
    "hunks": [
      {
        "hunk_id": "h1",
        "file": "tests/test_flaky.py",
        "search": "    result = await asyncio.wait_for(fetch(), timeout=0.05)",
        "replace": "    result = await asyncio.wait_for(fetch(), timeout=0.5)",
        "rationale": "Use a realistic timeout.",
        "addresses_claim": "c1"
      }
    ]
  }
}"""
        assert '"claims"' in extract_think(response)
        patch = extract_patch(response)
        assert "--- tests/test_flaky.py" in patch
        assert "<<<<<<< SEARCH" in patch
        assert ">>>>>>> REPLACE" in patch

    def test_extract_minimal_json_response(self):
        """7B-model friendly: only required fields, no optional keys."""
        response = """{
  "think": {
    "claims": [
      {
        "category": "async_wait",
        "entity": "fetch",
        "location": "tests/test_flaky.py::test_fetch",
        "polarity": "present",
        "reason": "Timeout too aggressive."
      }
    ],
    "confidence": 0.85
  },
  "patch": {
    "hunks": [
      {
        "file": "tests/test_flaky.py",
        "search": "    result = await asyncio.wait_for(fetch(), timeout=0.05)",
        "replace": "    result = await asyncio.wait_for(fetch(), timeout=0.5)"
      }
    ]
  }
}"""
        assert '"claims"' in extract_think(response)
        patch = extract_patch(response)
        assert "--- tests/test_flaky.py" in patch
        assert "<<<<<<< SEARCH" in patch
        assert ">>>>>>> REPLACE" in patch

    def test_extract_patch_normalizes_escaped_hunk_strings(self):
        response = """{
  "think": {
    "claims": [
      {
        "category": "concurrency",
        "entity": "submit",
        "location": "source.py::WorkerPool.submit",
        "polarity": "present",
        "reason": "Lock branch malformed"
      }
    ],
    "confidence": 0.85
  },
  "patch": {
    "hunks": [
      {
        "file": "source.py",
        "search": "        with self._lock:\\\\\\"",
        "replace": "        with self._lock:",
        "rationale": "Normalize malformed line"
      }
    ]
  }
}"""
        patch = extract_patch(response)
        assert "--- source.py" in patch
        assert "with self._lock:" in patch
        assert '\\"' not in patch

    def test_extract_empty_response(self):
        assert extract_think("") == ""
        assert extract_patch("") == ""

    def test_extract_malformed_blocks(self):
        response = "No XML blocks here"
        assert extract_think(response) == ""
        assert extract_patch(response) == ""

    def test_category_extraction(self):
        think = "Root Cause: async_wait (confidence: 0.85)"
        assert extract_category_from_think(think) == "async_wait"

    def test_category_normalization(self):
        assert extract_category_from_think("Root Cause: timeout") == "async_wait"
        assert extract_category_from_think("Root Cause: race") == "concurrency"
        assert extract_category_from_think("Root Cause: mock") == "mock_residue"
        assert extract_category_from_think("Root Cause: fixture") == "fixture_scope_leak"

    def test_confidence_extraction(self):
        think = "Root Cause: async_wait (confidence: 0.85)"
        assert extract_confidence_from_think(think) == 0.85

    def test_confidence_clamping(self):
        assert extract_confidence_from_think("confidence: 1.5") == 1.0
        assert extract_confidence_from_think("confidence: -0.5") == 0.0


class TestPatchParsing:
    """Test search/replace hunk parsing."""

    def test_single_hunk(self):
        patch = """--- tests/test_flaky.py
<<<<<<< SEARCH
    result = await asyncio.wait_for(fetch(), timeout=0.05)
=======
    result = await asyncio.wait_for(fetch(), timeout=0.5)
>>>>>>> REPLACE"""
        hunks = parse_search_replace_hunks(patch)
        assert len(hunks) == 1
        assert hunks[0].file_path == "tests/test_flaky.py"
        assert "timeout=0.05" in hunks[0].search_text
        assert "timeout=0.5" in hunks[0].replace_text

    def test_multiple_hunks(self):
        patch = """--- file1.py
<<<<<<< SEARCH
old1
=======
new1
>>>>>>> REPLACE

--- file2.py
<<<<<<< SEARCH
old2
=======
new2
>>>>>>> REPLACE"""
        hunks = parse_search_replace_hunks(patch)
        assert len(hunks) == 2
        assert hunks[0].file_path == "file1.py"
        assert hunks[1].file_path == "file2.py"

    def test_empty_patch(self):
        assert parse_search_replace_hunks("") == []
        assert parse_search_replace_hunks("   ") == []

    def test_apply_patch_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test_example.py"
            test_file.write_text("x = 1\ny = 2\nz = 3\n")

            patch = """--- test_example.py
<<<<<<< SEARCH
y = 2
=======
y = 42
>>>>>>> REPLACE"""
            result = apply_search_replace_patch(Path(tmpdir), patch)
            assert result["success"] is True
            assert result["lines_changed"] >= 1

            content = test_file.read_text()
            assert "y = 42" in content
            assert "y = 2" not in content


class TestRewardSignals:
    """Test individual reward signal computation."""

    def test_format_reward_perfect(self):
        action = FlakeForgeAction(
            raw_response="...",
            think_text="Root Cause: async_wait (confidence: 0.85)\nEvidence: ...",
            patch_text="<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE",
        )
        reward = compute_format_reward(action)
        assert reward == 1.0

    def test_format_reward_missing_think(self):
        action = FlakeForgeAction(
            raw_response="...",
            think_text="",
            patch_text="<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE",
        )
        reward = compute_format_reward(action)
        assert reward == 0.5  # Only patch block present

    def test_format_reward_empty(self):
        action = FlakeForgeAction(raw_response="garbage")
        assert compute_format_reward(action) == 0.0

    def test_compile_reward(self):
        assert compute_compile_reward(True) == 1.0
        assert compute_compile_reward(False) == -1.0
        assert compute_compile_reward(True, "SyntaxError") == -0.5

    def test_stability_full_fix(self):
        assert compute_stability_reward(0.5, 1.0) == 2.0

    def test_stability_improvement(self):
        reward = compute_stability_reward(0.3, 0.6)
        assert reward > 0

    def test_stability_regression(self):
        reward = compute_stability_reward(0.5, 0.2)
        assert reward < 0

    def test_stability_no_change(self):
        assert compute_stability_reward(0.5, 0.5) == -0.1

    def test_causal_proximity_same_file(self):
        reward = compute_causal_proximity_reward(
            patch_files=["tests/test_flaky.py"],
            failure_frontier="tests/test_flaky.py:12:test_func",
            call_chain=["test_func"],
            boundary_crossings=[],
        )
        assert reward == 1.0

    def test_causal_proximity_distant_file(self):
        reward = compute_causal_proximity_reward(
            patch_files=["utils/helpers.py"],
            failure_frontier="tests/test_flaky.py:12:test_func",
            call_chain=["test_func"],
            boundary_crossings=[],
        )
        assert reward < 0

    def test_entropy_reduction(self):
        reward = compute_entropy_reward(pre_entropy=0.8, post_entropy=0.2)
        assert reward > 0

    def test_entropy_increase(self):
        reward = compute_entropy_reward(pre_entropy=0.2, post_entropy=0.8)
        assert reward < 0

    def test_anti_hack_sleep_injection(self):
        patch = """<<<<<<< SEARCH
result = func()
=======
import time
time.sleep(1)
result = func()
>>>>>>> REPLACE"""
        penalty = compute_anti_hack_penalty(patch, ["test.py"], 3)
        assert penalty < 0

    def test_anti_hack_skip_decorator(self):
        patch = """<<<<<<< SEARCH
def test_flaky():
=======
@pytest.mark.skip
def test_flaky():
>>>>>>> REPLACE"""
        penalty = compute_anti_hack_penalty(patch, ["test.py"], 2)
        assert penalty <= -1.0

    def test_anti_hack_clean_patch(self):
        patch = """<<<<<<< SEARCH
timeout=0.05
=======
timeout=0.5
>>>>>>> REPLACE"""
        penalty = compute_anti_hack_penalty(patch, ["test.py"], 1)
        assert penalty == 0.0


class TestReasoningConsistency:
    """Test reasoning-patch consistency verification."""

    def test_exact_match(self):
        score = compute_reasoning_consistency(
            "async_wait", "async_wait", "root cause analysis", "timeout fix"
        )
        assert score == 0.5

    def test_related_match(self):
        score = compute_reasoning_consistency(
            "concurrency", "async_wait", "root cause analysis", "timeout fix"
        )
        assert score == 0.25

    def test_mismatch(self):
        score = compute_reasoning_consistency(
            "network", "fixture_scope_leak", "network issue", "fixture fix"
        )
        assert score == -0.5


class TestCategoryInference:
    """Test inferring root cause category from what the patch modifies."""

    def test_infer_async_wait(self):
        assert infer_category_from_patch("timeout=0.5") == "async_wait"

    def test_infer_concurrency(self):
        assert infer_category_from_patch("asyncio.Lock()") == "concurrency"

    def test_infer_fixture(self):
        assert infer_category_from_patch("yield fixture_data\n    teardown()") == "fixture_scope_leak"

    def test_infer_mock(self):
        assert infer_category_from_patch("monkeypatch.setattr") == "mock_residue"

    def test_infer_cache(self):
        assert infer_category_from_patch("cache_clear()") == "module_cache_pollution"


class TestFailureModeEntropy:
    """Test Shannon entropy calculation."""

    def test_single_error_type(self):
        runs = [
            RunRecord(passed=False, duration_ms=100, error_type="TimeoutError"),
            RunRecord(passed=False, duration_ms=100, error_type="TimeoutError"),
        ]
        assert failure_mode_entropy(runs) == 0.0

    def test_uniform_error_distribution(self):
        runs = [
            RunRecord(passed=False, duration_ms=100, error_type="TimeoutError"),
            RunRecord(passed=False, duration_ms=100, error_type="AssertionError"),
        ]
        assert failure_mode_entropy(runs) == 1.0

    def test_no_failures(self):
        runs = [
            RunRecord(passed=True, duration_ms=100),
            RunRecord(passed=True, duration_ms=100),
        ]
        assert failure_mode_entropy(runs) == 0.0


class TestObservationBuilding:
    """Test observation construction."""

    def test_observation_has_deep_signals(self):
        obs = FlakeForgeObservation(
            episode_id="test-001",
            test_identifier="tests/test_flaky.py::test_func",
            step=0,
            steps_remaining=8,
            test_function_source="def test_func(): pass",
            source_under_test="def func(): pass",
            module_cache_violations=["utils.py: @lru_cache on fetch"],
            async_contamination_alive=True,
        )
        assert len(obs.module_cache_violations) == 1
        assert obs.async_contamination_alive is True
        assert obs.step == 0

    def test_unified_prompt_includes_deep_signals(self):
        obs = FlakeForgeObservation(
            episode_id="test-001",
            test_identifier="tests/test_flaky.py::test_func",
            step=0,
            steps_remaining=8,
            test_function_source="def test_func(): pass",
            source_under_test="def func(): pass",
            module_cache_violations=["utils.py: @lru_cache on compute"],
            mock_residue_sites=["test_flaky.py:10 — patch() without context"],
        )
        prompt = build_unified_prompt(obs)
        assert "module-cache hot spots" in prompt
        assert "mock residue:" in prompt

    def test_observation_run_history_limit(self):
        runs = [RunRecord(passed=True, duration_ms=100) for _ in range(30)]
        obs = FlakeForgeObservation(
            episode_id="test-001",
            test_identifier="test",
            step=0,
            steps_remaining=8,
            test_function_source="",
            source_under_test="",
            run_history=runs,
        )
        assert len(obs.run_history) == 20  # Capped at 20


class TestRootCauseTypes:
    """Verify root cause taxonomy is complete."""

    def test_all_categories_present(self):
        assert "async_wait" in ROOT_CAUSE_TYPES
        assert "module_cache_pollution" in ROOT_CAUSE_TYPES
        assert "fixture_scope_leak" in ROOT_CAUSE_TYPES
        assert "mock_residue" in ROOT_CAUSE_TYPES
        assert "import_side_effect" in ROOT_CAUSE_TYPES
        assert "unknown" in ROOT_CAUSE_TYPES

    def test_category_count(self):
        assert len(ROOT_CAUSE_TYPES) == 13
