"""V3 SFT Data Generator — synthetic reasoning + patch training pairs.

Creates supervised fine-tuning data from iDFlakies/FlakyDoctor datasets
with ground-truth reasoning chains. Uses template expansion for generating
diverse think+patch training examples.

IMPORTANT — V3 format:
    Completions are single JSON objects matching the unified agent schema:
        {
          "think": { "claims": [...], "confidence": 0.9 },
          "patch": { "hunks": [...] }
        }
    This is the format the live agent produces and the format the reward
    function expects.  The legacy XML format (<think>, <patch>) is accepted
    by the reward function as a fallback but should NOT be used for new SFT
    data.

Research basis:
- iDFlakies ICST 2019: OD (order-dependent) and NOD (non-order-dependent) classification
- FlakyDoctor: Ground-truth patches for common flakiness patterns
- Luo FSE 2014: Root cause taxonomy
"""

from __future__ import annotations

import json
import re
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from models import ROOT_CAUSE_TYPES
except ImportError:
    try:
        from ..models import ROOT_CAUSE_TYPES
    except (ImportError, ValueError):
        try:
            from FlakeForge.models import ROOT_CAUSE_TYPES
        except ImportError:
            ROOT_CAUSE_TYPES = [
                "async_wait", "concurrency", "test_order_dependency", "resource_leak",
                "shared_state", "network", "platform_dependency", "nondeterminism",
                "import_side_effect", "module_cache_pollution", "fixture_scope_leak",
                "mock_residue", "unknown",
            ]


# ── Reasoning Template Library ───────────────────────────────────────────────

REASONING_TEMPLATES = {
    "async_wait": [
        """Root Cause: async_wait (confidence: {confidence})
Evidence: {error_type} at {file_path}:{line_num}, {async_primitive} with timeout={timeout_val}s
The coroutine {function_name}() has a race condition where the {resource} can take longer than {timeout_val}s under {condition}, causing intermittent {error_type}.
Strategy: {strategy}""",
        """Root Cause: async_wait (confidence: {confidence})
Evidence: Intermittent {error_type} in {function_name}, asyncio event loop not properly awaited
The test does not properly await all pending tasks before asserting, leading to non-deterministic {error_type} when the event loop is under load.
Strategy: {strategy}""",
    ],
    "concurrency": [
        """Root Cause: concurrency (confidence: {confidence})
Evidence: {error_type} in {function_name}, shared resource {resource} accessed without synchronization
Multiple threads/coroutines access {resource} through {function_name}(). Without proper locking, concurrent modifications cause {error_type}.
Strategy: {strategy}""",
    ],
    "test_order_dependency": [
        """Root Cause: test_order_dependency (confidence: {confidence})
Evidence: Test {test_name} fails when run after {dependent_test}, passes in isolation
The test depends on state set up by {dependent_test} in {shared_resource}. Running in different order skips this setup.
Strategy: {strategy}""",
    ],
    "shared_state": [
        """Root Cause: shared_state (confidence: {confidence})
Evidence: Global/module-level {resource} in {file_path} mutated by {function_name}
The {resource} is modified by test execution but never reset, causing subsequent tests to see stale state. The mutation happens at {file_path}:{line_num}.
Strategy: {strategy}""",
    ],
    "module_cache_pollution": [
        """Root Cause: module_cache_pollution (confidence: {confidence})
Evidence: @lru_cache on {function_name} in {file_path}, cache not cleared between tests
The cached return value from a previous test run bleeds into the current test. The cache key does not account for test-specific state.
Strategy: {strategy}""",
    ],
    "fixture_scope_leak": [
        """Root Cause: fixture_scope_leak (confidence: {confidence})
Evidence: {fixture_name} fixture with scope='{scope}' returns mutable {return_type} without yield teardown
The fixture returns a mutable object that is shared across {scope}-scoped tests. Modifications in one test affect subsequent tests.
Strategy: {strategy}""",
    ],
    "mock_residue": [
        """Root Cause: mock_residue (confidence: {confidence})
Evidence: patch('{patched_target}') at {file_path}:{line_num} without context manager or .stop()
The mock patch is started but never properly cleaned up, leaking the mock into subsequent test execution.
Strategy: {strategy}""",
    ],
    "nondeterminism": [
        """Root Cause: nondeterminism (confidence: {confidence})
Evidence: {source} in {function_name} produces non-deterministic output
The test relies on {source} which varies between runs. Without seeding or deterministic alternatives, assertions on exact values fail intermittently.
Strategy: {strategy}""",
    ],
    "resource_leak": [
        """Root Cause: resource_leak (confidence: {confidence})
Evidence: {resource_type} opened in {function_name} at {file_path}:{line_num} not closed in teardown
The {resource_type} accumulates across tests, eventually causing {error_type} when the system limit is reached.
Strategy: {strategy}""",
    ],
    "import_side_effect": [
        """Root Cause: import_side_effect (confidence: {confidence})
Evidence: Top-level execution in {file_path}: {side_effect_description}
Module-level code runs on import, creating non-deterministic behavior depending on import order and timing.
Strategy: {strategy}""",
    ],
    "network": [
        """Root Cause: network (confidence: {confidence})
Evidence: {error_type} in {function_name} calling {endpoint}
The test makes real network calls to {endpoint} which are subject to latency, DNS failures, and service availability.
Strategy: {strategy}""",
    ],
    "platform_dependency": [
        """Root Cause: platform_dependency (confidence: {confidence})
Evidence: {function_name} uses {platform_feature} which behaves differently on {platform}
The test assumes {assumption} which only holds on {expected_platform}.
Strategy: {strategy}""",
    ],
}

STRATEGY_TEMPLATES = {
    "async_wait": [
        "Increase timeout to accommodate {resource} contention latency.",
        "Add proper await for all pending tasks before assertions.",
        "Use asyncio.wait_for with a reasonable timeout instead of raw await.",
    ],
    "concurrency": [
        "Add asyncio.Lock() around access to shared {resource}.",
        "Use thread-safe data structure instead of plain {resource}.",
    ],
    "test_order_dependency": [
        "Add explicit setup in the test fixture to initialize required state.",
        "Make the test self-contained by creating its own {shared_resource}.",
    ],
    "shared_state": [
        "Reset {resource} in a fixture teardown or conftest autouse fixture.",
        "Use a deep copy of {resource} per test instead of shared reference.",
    ],
    "module_cache_pollution": [
        "Add {function_name}.cache_clear() to the test teardown.",
        "Replace @lru_cache with a test-aware caching mechanism.",
    ],
    "fixture_scope_leak": [
        "Convert fixture to use yield with proper teardown/reset.",
        "Narrow fixture scope from {scope} to function.",
    ],
    "mock_residue": [
        "Wrap patch in with-context manager for automatic cleanup.",
        "Add explicit .stop() in teardown or use addCleanup().",
    ],
    "nondeterminism": [
        "Seed the random generator with a fixed value for reproducibility.",
        "Assert on properties/ranges instead of exact values.",
    ],
    "resource_leak": [
        "Add try/finally or context manager to ensure {resource_type} is closed.",
        "Use a fixture with yield to manage {resource_type} lifecycle.",
    ],
    "import_side_effect": [
        "Move side-effectful code inside if __name__ == '__main__' guard.",
        "Defer initialization to a function called at runtime, not import time.",
    ],
    "network": [
        "Mock the network call using unittest.mock.patch or responses library.",
        "Add VCR cassette recording for deterministic replay.",
    ],
    "platform_dependency": [
        "Add platform detection and conditional assertions.",
        "Use os.path or pathlib for platform-independent path handling.",
    ],
}


def _parse_search_replace_to_hunks(patch_text: str, file_hint: str = "") -> List[Dict[str, str]]:
    """Convert a search/replace patch block into a list of hunk dicts for V3 JSON."""
    import re
    hunks: List[Dict[str, str]] = []
    # Pattern: optional "--- file\n" header, then <<<<<<< SEARCH … ======= … >>>>>>> REPLACE
    blocks = re.split(r"\n?---\s+", patch_text.strip())
    for block in blocks:
        if not block.strip():
            continue
        # Try to extract file name from first line
        lines = block.split("\n", 1)
        file_path = file_hint
        rest = block
        if len(lines) > 1 and not lines[0].startswith("<"):
            file_path = lines[0].strip() or file_hint
            rest = lines[1]

        search_match = re.search(r"<<<<<<< SEARCH\n(.*?)\n=======", rest, re.DOTALL)
        replace_match = re.search(r"=======\n(.*?)\n?>>>>>>> REPLACE", rest, re.DOTALL)

        if search_match and replace_match:
            hunks.append({
                "hunk_id": f"h{len(hunks)+1}",
                "file": file_path or "tests/test_flaky.py",
                "search": search_match.group(1),
                "replace": replace_match.group(1),
                "rationale": "",
                "addresses_claim": f"c{len(hunks)+1}",
            })
    return hunks


def generate_sft_example(
    category: str,
    template_vars: Dict[str, str],
    observation_text: str,
    patch_text: str,
    file_hint: str = "",
) -> Dict[str, Any]:
    """Generate a single SFT training example in V3 JSON format.

    The completion is a single JSON object with "think" and "patch" keys
    that matches exactly the unified agent's response schema.  This ensures
    the model fine-tuned on this data produces outputs parseable by
    the V3 reward system.

    Args:
        category: Root cause category (must be in ROOT_CAUSE_TYPES).
        template_vars: Variables to fill into the reasoning template.
        observation_text: The observation context (becomes the prompt).
        patch_text: Ground-truth patch in search/replace format.
        file_hint: Fallback file path for hunks when not encoded in patch_text.

    Returns:
        Training example dict with prompt/completion fields, or {} on failure.
    """
    templates = REASONING_TEMPLATES.get(category, [])
    strategies = STRATEGY_TEMPLATES.get(category, ["Fix the root cause."])

    if not templates:
        return {}

    template = random.choice(templates)
    strategy = random.choice(strategies)

    confidence = round(random.uniform(0.75, 0.95), 2)
    template_vars.setdefault("confidence", str(confidence))
    template_vars.setdefault(
        "strategy",
        strategy.format(**template_vars) if "{" in strategy else strategy,
    )

    try:
        reasoning_text = template.format(**template_vars)
    except KeyError:
        reasoning_text = template
        for key, val in template_vars.items():
            reasoning_text = reasoning_text.replace(f"{{{key}}}", val)

    # Build the V3 JSON "think" block
    think_block: Dict[str, Any] = {
        "claims": [
            {
                "claim_id": "c1",
                "category": category,
                "entity": template_vars.get("function_name", "unknown"),
                "location": (
                    f"{template_vars.get('file_path', 'tests/test_flaky.py')}"
                    f"::{template_vars.get('function_name', 'unknown')}"
                ),
                "ast_node_type": "FunctionDef",
                "polarity": "present",
                "predicted_effect": (
                    f"Fixing the {category} root cause should increase test pass rate to 1.0."
                ),
                "reason": reasoning_text[:200],
            }
        ],
        "confidence": confidence,
    }

    # Build the V3 JSON "patch" block
    hunks = _parse_search_replace_to_hunks(patch_text, file_hint=file_hint)
    if not hunks:
        # Minimal hunk so the example is still parseable
        hunks = [{
            "hunk_id": "h1",
            "file": file_hint or "tests/test_flaky.py",
            "search": "",
            "replace": patch_text.strip(),
            "rationale": f"Apply {category} fix.",
            "addresses_claim": "c1",
        }]

    patch_block: Dict[str, Any] = {"hunks": hunks}

    # Completion is a single JSON object — no XML, no markdown fences
    completion_obj: Dict[str, Any] = {"think": think_block, "patch": patch_block}
    completion = json.dumps(completion_obj, ensure_ascii=False, indent=2)

    return {
        "prompt": observation_text,
        "completion": completion,
        "category": category,
        "metadata": {
            "template_used": template[:50],
            "strategy": (strategy.format(**template_vars) if "{" in strategy else strategy)[:50],
            "format": "v3_json",
        },
    }


def generate_sft_dataset_from_manifest(
    manifest_path: Path,
    output_path: Path,
    augmentation_factor: int = 3,
) -> int:
    """Generate SFT dataset from a FlakeForge manifest file.

    The manifest should contain entries with:
    - test_identifier: The flaky test name
    - category: Root cause category
    - patch: Ground-truth fix
    - observation: Context for the agent

    Args:
        manifest_path: Path to JSON manifest
        output_path: Path to write JSONL output
        augmentation_factor: Number of augmented versions per entry

    Returns:
        Number of examples generated
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = manifest if isinstance(manifest, list) else manifest.get("entries", [])
    examples: List[Dict[str, Any]] = []

    for entry in entries:
        category = entry.get("category", "unknown")
        if category not in ROOT_CAUSE_TYPES:
            continue

        for _ in range(augmentation_factor):
            template_vars = {
                "error_type": entry.get("error_type", "AssertionError"),
                "file_path": entry.get("file_path", "tests/test_example.py"),
                "line_num": str(entry.get("line_num", 1)),
                "function_name": entry.get("function_name", "test_func"),
                "test_name": entry.get("test_identifier", "test_func"),
                "resource": entry.get("resource", "shared_data"),
                "timeout_val": str(entry.get("timeout", "0.5")),
                "async_primitive": entry.get("async_primitive", "asyncio.wait_for"),
                "condition": entry.get("condition", "high load"),
                "source": entry.get("source", "random.random()"),
                "resource_type": entry.get("resource_type", "file handle"),
                "endpoint": entry.get("endpoint", "http://api.example.com"),
                "platform_feature": entry.get("platform_feature", "os.path.sep"),
                "platform": entry.get("platform", "Windows"),
                "expected_platform": entry.get("expected_platform", "Linux"),
                "assumption": entry.get("assumption", "/ as path separator"),
                "shared_resource": entry.get("shared_resource", "database"),
                "dependent_test": entry.get("dependent_test", "test_setup"),
                "fixture_name": entry.get("fixture_name", "db_session"),
                "scope": entry.get("scope", "session"),
                "return_type": entry.get("return_type", "dict"),
                "patched_target": entry.get("patched_target", "module.Class.method"),
                "side_effect_description": entry.get("side_effect_description", "connects to database"),
            }

            example = generate_sft_example(
                category=category,
                template_vars=template_vars,
                observation_text=entry.get("observation", ""),
                patch_text=entry.get("patch", ""),
                file_hint=entry.get("file_path", ""),
            )

            if example:
                examples.append(example)

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return len(examples)
