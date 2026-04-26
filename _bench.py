"""Quick benchmark: reset + rollout timing with num_runs=20."""
import time
import json
from server.FlakeForge_environment import FlakeForgeEnvironment
from server.docker_runner import DockerTestRunner
from models import FlakeForgeAction
from agent.unified_agent import (
    extract_think, extract_patch,
    extract_category_from_think, extract_confidence_from_think,
)


def bench(repo, test_id, label, num_runs=20):
    print(f"\n=== {label}: {repo.split('/')[-1]} (num_runs={num_runs}) ===")
    runner = DockerTestRunner(repo)
    env = FlakeForgeEnvironment(
        repo_path=repo, test_identifier=test_id,
        max_steps=1, num_runs=num_runs, runner=runner,
    )

    t0 = time.time()
    obs = env.reset(preflight_quick_runs=5, preflight_confirm_runs=10)
    t_reset = time.time() - t0
    print(f"  reset (5+10 preflight): {t_reset:.1f}s")
    print(f"  should_train={obs.should_train}  baseline_pass_rate={obs.baseline_pass_rate:.2f}")
    if not obs.should_train:
        print(f"  REJECTED: {obs.done_reason}")
        return

    fake = json.dumps({
        "think": {"claims": [{"category": "shared_state", "entity": "x",
                              "location": "x.py::f", "polarity": "present",
                              "reason": "test"}], "confidence": 0.5},
        "patch": {"hunks": []},
    })
    action = FlakeForgeAction(
        raw_response=fake,
        think_text=extract_think(fake),
        patch_text=extract_patch(fake),
        predicted_category=extract_category_from_think(extract_think(fake)),
        predicted_confidence=extract_confidence_from_think(extract_think(fake)),
    )

    t1 = time.time()
    env.reset(preflight_quick_runs=1, preflight_confirm_runs=1)
    t_fast = time.time() - t1

    t2 = time.time()
    step_obs = env.step(action)
    t_step = time.time() - t2

    rollout = t_fast + t_step
    print(f"  fast_reset (1+1): {t_fast:.1f}s")
    print(f"  env.step ({num_runs} pytest runs): {t_step:.1f}s")
    print(f"  reward={step_obs.reward:.3f}  pass_rate={step_obs.current_pass_rate:.2f}")
    print(f"  ---")
    print(f"  1 rollout = {rollout:.1f}s")
    print(f"  G=3 rollouts (sequential per repo) = {3 * rollout:.1f}s")
    print(f"  full step (4 repos parallel): ~{t_reset:.0f}s reset + {3 * rollout:.0f}s rollouts = {t_reset + 3 * rollout:.0f}s")


bench("seed_repos/idoft/ljson__test_unique_check",
      "tests/test_ljson.py::test_unique_check", "EASY-1")
bench("seed_repos/idoft/bottle-neck__test_router_mount_pass",
      "test/test_routing.py::test_router_mount_pass", "EASY-2")
