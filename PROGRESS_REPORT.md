# FlakeForge Gym - Current Progress Report

Date: 2026-04-21
Date: 2026-04-22
Workspace: C:/CodingNest/FlakeForge

## 1) Overall Progress Snapshot

This repository now reflects the redesigned FlakeForge Gym architecture:

- Strict action and observation schema layer
- Server environment lifecycle (reset/step) and state tracking
- Test execution runner for repeated flaky-test validation
- Reward computation module
- Internal tooling module (AST summary, logging injection, patching hooks, diffing)
- FastAPI app wiring
- Async client wrapper with judge-score plumbing
- Docker image definition for running the environment server
- Seed flaky repositories plus fixed solution variants
- Integration contract test scaffold
- One deterministic environment
- One trainable model with two sequential LoRA roles at the system level
- One frozen judge for auxiliary scoring
- Seven-action space with strict validation
- Repeated validation and regression checking
- Evidence-grounded AST-aware tool layer
- Seed repositories for all requested flaky categories
- Integration contract test scaffold

## 2) High-Level Status by Area

### Completed

- Core type system and validation in models
- Server/client wiring for OpenEnv APIs
- Main environment loop skeleton (reset/step/action dispatch)
- Docker runner and reward module separation
- Seed repositories created for all requested categories
- End-to-end integration test file added
- Core type system and validation in models
- Environment state and step/reset flow
- Failure-pattern extraction and reward shaping
- AST-aware tools for source inspection, evidence grounding, and patch application
- Docker runner for repeated validation and regression checks
- Seed repositories created for all requested categories
- End-to-end integration test file added

### Partially Implemented

- Patch application is currently simplified in places (textual strategy and placeholders in some action branches)
- Similar-fix retrieval is currently an interface placeholder, not a full Chroma/FAISS-backed dataset pipeline
- Judge scoring is scaffolded in the client with a mock response path
- Retrieval is present with a local Chroma-backed path, but the external flaky-fix corpus ingestion still needs final dataset curation
- Judge prompting is scaffolded in the client and should be connected to a real frozen model API in the training stack
- Patch precision is substantially improved, but some edge cases for multi-file repairs and complex decorators still need runtime hardening

### Not Yet Fully Verified at Runtime

- Full docker-backed integration execution has not been run to completion in this progress pass
- Full docker-backed integration execution has not been run to completion in this progress pass
- Cross-file patch scenarios and final judge integration need end-to-end runtime confirmation

## 3) File-by-File Role Map

## Root Files

### __init__.py
Role:
- Package export surface for the environment.
- Exposes action/observation/client classes for external imports.

Status:
- Updated to export canonical FlakeForge names plus backward-compatible aliases.

### .gitignore
Role:
- Prevents committing runtime/generated artifacts (venv, pyc, cache files).

Status:
- Added and active.

### client.py
Role:
- Defines the OpenEnv async environment client class.
- Converts action objects to wire payloads.
- Parses server responses into typed observation/state objects.
- Keeps per-episode judge score records.

Status:
- Implemented with typed parsing and judge-score plumbing.
- Contains a placeholder judge call flow for future real API integration.

### models.py
Role:
- Single source of truth for action/observation/state and supporting records.
- Enforces strict action parameter validation (allowed values and required keys).

Status:
- Implemented with strict validators and required data structures.
- Includes backward-compatible class aliases.

### openenv.yaml
Role:
- OpenEnv manifest defining app entrypoint and runtime metadata.

Status:
- Present and unchanged as base environment metadata.

### pyproject.toml
Role:
- Python project metadata and dependency declarations.
- Dev dependency setup for tests.

Status:
- Updated to include server/runtime dependencies and async test support.

### pytest.ini
Role:
- Pytest configuration.
- Registers integration marker and asyncio mode.

Status:
- Added and active.

### README.md
Role:
- Human-facing project documentation.

Status:
- Still mostly starter-template oriented and should be refreshed to match current FlakeForge Gym behavior.
- Updated to describe the redesigned POMDP environment and file layout.

### uv.lock
Role:
- Lockfile for reproducible dependency resolution in uv workflows.

Status:
- Present from scaffold/tooling baseline.

## Server Package

### server/__init__.py
Role:
- Export surface for server-side environment class.

Status:
- Updated to export canonical and legacy class names.

### server/app.py
Role:
- Creates FastAPI app by binding environment instance and action/observation schemas through OpenEnv server factory.

Status:
- Implemented and wired to FlakeForge environment classes.

### server/FlakeForge_environment.py
Role:
- Main RL environment logic.
- Handles reset/step lifecycle and episode state.
- Dispatches each action type to execution logic.
- Runs repeated validation, regression checks, reward computation, and observation construction.

Status:
- Implemented as the core orchestration module.
- Some action patch behaviors are currently simplified and should be hardened for full production semantics.

### server/docker_runner.py
Role:
- Executes pytest commands for target tests.
- Supports repeated runs in thread pool.
- Performs regression checks outside the target test.

Status:
- Implemented and integrated with environment.

### server/reward.py
Role:
- Centralized reward function computation.
- Combines stability, judge, efficiency, regression, retry penalty, and terminal bonuses.

Status:
- Implemented and integrated into step flow.

### server/tools.py
Role:
- Internal utility/tool layer used by environment logic.
- Repository listing, bounded excerpt reads, AST summaries, logging injection, patch application, diff computation, and retrieval interface.

Status:
- Implemented with core function signatures and baseline behavior.
- Contains placeholder/simplified logic in retrieval and parts of patch operations.

### server/requirements.txt
Role:
- Runtime dependencies required inside server container.

Status:
- Expanded to include OpenEnv core + parser/transformation/retrieval dependencies.

### server/Dockerfile
Role:
- Builds environment server image.
- Installs dependencies and launches uvicorn on port 8000.
- Defines healthcheck endpoint probing.

Status:
- Replaced from template with a direct python:3.11-slim based runtime image.
- Functionally wired; can be further tuned for build speed and layer caching.

## Tests

### tests/test_flakeforge_integration.py
Role:
- Docker-backed integration contract test.
- Builds image, creates client from image, runs reset and two key actions, asserts key outputs.

Status:
- Added and syntax/import validated via collect-only.
- Full runtime execution still pending.

## Seed Repositories (Flaky Scenarios + Solutions)

Each seed repository follows the same pattern:

- flaky_module.py: Intentionally flaky behavior source
- tests/test_flaky.py: Flaky test target used by environment
- solution/flaky_module.py: Fixed source variant
- solution/tests/test_flaky.py: Stable fixed test

### seed_repos/timing_race/flaky_module.py
Role:
- Async timing race source behavior.

### seed_repos/timing_race/tests/test_flaky.py
Role:
- Flaky test with variable timeout causing intermittent timeout behavior.

### seed_repos/timing_race/solution/flaky_module.py
Role:
- Reduced latency fixed implementation.

### seed_repos/timing_race/solution/tests/test_flaky.py
Role:
- Stable solution test path.

### seed_repos/shared_state/flaky_module.py
Role:
- Mutable shared-state source behavior.

### seed_repos/shared_state/tests/test_flaky.py
Role:
- Flaky test with residual state contamination.

### seed_repos/shared_state/solution/flaky_module.py
Role:
- Fixed variant that resets/clears shared state before use.

### seed_repos/shared_state/solution/tests/test_flaky.py
Role:
- Stable solution test path.

### seed_repos/external_dependency/flaky_module.py
Role:
- External-call-dependent behavior with intermittent network dependency exposure.

### seed_repos/external_dependency/tests/test_flaky.py
Role:
- Flaky test asserting status under unstable external behavior.

### seed_repos/external_dependency/solution/flaky_module.py
Role:
- Fixed variant adding fallback exception handling.

### seed_repos/external_dependency/solution/tests/test_flaky.py
Role:
- Stable solution test path.

### seed_repos/order_dependency/flaky_module.py
Role:
- Simple database-like state requiring setup order.

### seed_repos/order_dependency/tests/test_flaky.py
Role:
- Flaky ordering dependency scenario where setup may or may not happen before assertion.

### seed_repos/order_dependency/solution/flaky_module.py
Role:
- Stable source baseline retained for deterministic setup usage.

### seed_repos/order_dependency/solution/tests/test_flaky.py
Role:
- Stable fixed test that always performs setup before assertion.

### seed_repos/resource_leak/flaky_module.py
Role:
- Resource leak via accumulating temporary file handles.

### seed_repos/resource_leak/tests/test_flaky.py
Role:
- Flaky test sensitive to leaked-handle growth.

### seed_repos/resource_leak/solution/flaky_module.py
Role:
- Fixed variant using managed context to avoid leak.

### seed_repos/resource_leak/solution/tests/test_flaky.py
Role:
- Stable solution test path.

### seed_repos/nondeterminism/flaky_module.py
Role:
- Nondeterministic random choice source behavior.

### seed_repos/nondeterminism/tests/test_flaky.py
Role:
- Flaky test asserting deterministic output without seed control.

### seed_repos/nondeterminism/solution/flaky_module.py
Role:
- Fixed variant with deterministic seed initialization.

### seed_repos/nondeterminism/solution/tests/test_flaky.py
Role:
- Stable solution test path.

### seed_repos/compound_timing_shared/flaky_module.py
Role:
- Compound case combining async timing and shared-state mutation.

### seed_repos/compound_timing_shared/tests/test_flaky.py
Role:
- Flaky test combining timeout variability and state contamination.

### seed_repos/compound_timing_shared/solution/flaky_module.py
Role:
- Fixed variant addressing both timing and state reset.

### seed_repos/compound_timing_shared/solution/tests/test_flaky.py
Role:
- Stable solution test path.

### seed_repos/compound_external_nondeterminism/flaky_module.py
Role:
- Compound case combining unstable external call and random behavior.

### seed_repos/compound_external_nondeterminism/tests/test_flaky.py
Role:
- Flaky test with mixed dependency and nondeterminism risk.

### seed_repos/compound_external_nondeterminism/solution/flaky_module.py
Role:
- Fixed variant with deterministic seed and external-call fallback.

### seed_repos/compound_external_nondeterminism/solution/tests/test_flaky.py
Role:
- Stable solution test path.

## Generated Tooling Cache Files

These are tooling runtime artifacts, not functional environment logic:

### .pytest_cache/.gitignore
Role:
- Keeps pytest cache directory structure under tool control.

### .pytest_cache/CACHEDIR.TAG
Role:
- Standard cache marker file.

### .pytest_cache/README.md
Role:
- Explains cache directory purpose.

### .pytest_cache/v/cache/lastfailed
Role:
- Stores last failed test metadata for pytest.

## 4) Current Gaps to Reach Full Spec Fidelity

1. Harden action patch semantics:
- Implement fully structured libcst transformations for all patch operations and target resolution from evidence.

2. Implement full retrieval backend:
- Replace get_similar_fixes placeholder with real Chroma/FAISS indexing and query path.

3. Wire real judge calls:
- Replace mock judge response in client with frozen-model API invocation and robust JSON parsing/retry.

4. Run full integration runtime:
- Execute docker build + test run end-to-end and fix any runtime mismatches.

## 5) Recommended Immediate Next Step

Run the integration test fully (not collect-only):

- docker build -t flakeforge-env:latest -f server/Dockerfile .
- pytest -q tests/test_flakeforge_integration.py

Then iterate on any runtime failures until this contract passes consistently.
