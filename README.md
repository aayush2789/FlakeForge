---
title: Flakeforge Environment Server
emoji: 🔕
colorFrom: pink
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# FlakeForge Gym

FlakeForge Gym is a POMDP-style reinforcement learning environment for training a single 3.5B code model with two sequential LoRA roles: an Analyzer that forms a hypothesis about a flaky CI failure, and a Fixer that chooses a repair action from a small, strict action space. The environment is deterministic; the model is the only learned component. A frozen judge model provides auxiliary shaping reward, and the Docker-isolated runner executes repeat validations.

## Architecture

- One trainable base model with two LoRA adapters
- One frozen judge model for hypothesis and patch scoring
- One deterministic environment that owns reset, step, validation, and state tracking
- Seven action types with low-cardinality parameters
- Structured observation JSON returned on every step

## Current Status

The core environment, reward, tools, Docker runner, seed repositories, and integration test scaffolding are in place. The remaining work is runtime hardening and final wiring against the judge and retrieval corpus at scale.

## Quick Start

Build the Docker image:

```bash
docker build -t flakeforge-env:latest -f server/Dockerfile .
```

Create a client from the image:

```python
from FlakeForge import FlakeForgeAction, FlakeForgeEnv

env = await FlakeForgeEnv.from_docker_image("flakeforge-env:latest")
try:
    observation = await env.reset()
    result = await env.step(
        FlakeForgeAction(
            action_type="GATHER_EVIDENCE",
            parameters={"injection_target": "test"},
        )
    )
finally:
    await env.close()
```

## Action Space

The environment supports exactly seven actions:

- `GATHER_EVIDENCE`
- `ADD_TIMING_GUARD`
- `ADD_SYNCHRONIZATION`
- `MOCK_DEPENDENCY`
- `RESET_STATE`
- `ADD_RETRY`
- `REVERT_LAST_PATCH`

Each action has strict, low-cardinality parameters and is validated before the server accepts it.

## Observation and State

The main state objects are defined in `models.py`:

- `FlakeForgeAction`: validated action payload sent by the agent
- `RunRecord`: one validation run result
- `Hypothesis`: analyzer output with category, confidence, evidence, and suggested action
- `PatchRecord`: record of one patch attempt
- `ASTSummary`: compact code-structure summary used by tools and observations
- `FlakeForgeObservation`: full episode observation payload returned by the environment
- `FlakeForgeState`: compact server-side episode state
- `FailurePattern`: structured failure diagnostics from repeated validation

## Tools and Runner

The server-side tooling layer in `server/tools.py` converts raw source into structured, token-efficient signals:

- `list_repo_structure(root_path)`
- `read_file_excerpt(path, start_line, end_line)`
- `parse_ast_summary(path)`
- `resolve_target_from_evidence(path, evidence)`
- `get_failure_pattern(run_records)`
- `inject_logging(path, injection_points)`
- `apply_ast_patch(path, patch_spec)`
- `compute_diff(original_path, patched_source)`
- `get_similar_fixes(root_cause_category, test_source, embedding_model)`

The test runner in `server/docker_runner.py` executes repeated pytest runs, parses failures, and checks regression scope outside the target test.

## Environment Loop

`server/FlakeForge_environment.py` handles:

- Deterministic reset with baseline validation
- Hypothesis tracking across steps
- Gated action execution
- Repeated validation after every step
- Regression checks across the rest of the repo
- Patch bookkeeping and revert support
- Reward aggregation metadata

## Reward

`server/reward.py` computes shaped reward from:

- Stability improvement over baseline
- Judge scores
- Evidence overuse penalty
- Semantic patch-efficiency penalty
- Regression penalty
- Retry abuse penalty
- Terminal success/timeout shaping

## Seed Repositories

The `seed_repos/` directory contains eight flaky scenarios plus fixed solution variants:

- `timing_race`
- `shared_state`
- `external_dependency`
- `order_dependency`
- `resource_leak`
- `nondeterminism`
- `compound_timing_shared`
- `compound_external_nondeterminism`

Each scenario includes a flaky module, flaky tests, and a matching solution variant.

## Project Layout

- `__init__.py`: package exports
- `client.py`: OpenEnv client wrapper
- `models.py`: schema and validation source of truth
- `openenv.yaml`: OpenEnv manifest
- `pyproject.toml`: dependency and build metadata
- `pytest.ini`: pytest config
- `README.md`: high-level project documentation
- `PROGRESS_REPORT.md`: implementation progress and file-role map
- `server/`: environment runtime code
- `seed_repos/`: flaky CI fixtures and fixed solution variants
- `tests/`: integration tests

## Validation

The integration test currently collects cleanly under pytest. Full Docker-backed execution is the next runtime validation step.
