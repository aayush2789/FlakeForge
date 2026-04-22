#!/usr/bin/env python3
"""
Test runner script for FlakeForge inference on the minimal timing race test repo.

This script:
1. Sets up the test environment
2. Runs the flaky test multiple times to show the flakiness (ground truth)
3. Launches the FlakeForge inference engine
4. Collects and displays results

Usage:
    python test_runner.py [--show-flakiness] [--run-inference] [--steps 14]
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to the test repo (this directory)
TEST_REPO_ROOT = Path(__file__).parent

# NVIDIA API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")

# Small models for testing (lower latency, faster inference)
# Analyzer/Fixer: Using smaller model for speed
ANALYZER_FIXER_MODEL = os.getenv("ANALYZER_FIXER_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
# Judge: Keep Minimax as requested
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "minimaxai/minimax-m2.7")

API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Inference configuration
INFERENCE_MAX_STEPS = int(os.getenv("INFERENCE_MAX_STEPS", "5"))  # Smaller for quick testing
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "900"))

# Environment
USE_DOCKER_IMAGE = os.getenv("USE_DOCKER_IMAGE", "0").strip().lower() in {"1", "true", "yes"}
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def show_flakiness(num_runs: int = 10) -> None:
    """Run the test multiple times to demonstrate flakiness."""
    print_header(f"GROUND TRUTH: Running test {num_runs} times to show flakiness")
    
    passed = 0
    failed = 0
    
    for i in range(1, num_runs + 1):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_flaky.py::test_flaky_simple", "-v", "--tb=short"],
            cwd=TEST_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        print(f"Run {i:2d}/10:  {status}")
    
    print(f"\nSummary: {passed} passed, {failed} failed out of {num_runs} runs")
    pass_rate = passed / num_runs if num_runs > 0 else 0
    print(f"Pass rate: {pass_rate:.1%} (should be lower due to flakiness)\n")


def validate_config() -> bool:
    """Validate that required configuration is set."""
    print_header("Configuration Check")
    
    checks = [
        ("Test repo exists", TEST_REPO_ROOT.exists()),
        ("NVIDIA_API_KEY set", bool(NVIDIA_API_KEY)),
        ("Analyzer/Fixer model", ANALYZER_FIXER_MODEL),
        ("Judge model (Minimax)", JUDGE_MODEL),
        ("API base URL", API_BASE_URL),
    ]
    
    all_ok = True
    for check_name, check_result in checks:
        if isinstance(check_result, bool):
            status = "✓" if check_result else "✗"
            print(f"  {status} {check_name}: {check_result}")
            if not check_result:
                all_ok = False
        else:
            print(f"  • {check_name}: {check_result}")
    
    print()
    
    if not all_ok:
        print("ERROR: Required configuration is missing!")
        print("\nSet environment variables:")
        print("  PowerShell:")
        print("    $env:NVIDIA_API_KEY='your-key-here'")
        print("    $env:INFERENCE_MAX_STEPS='5'  # for quick testing")
        print("  Bash:")
        print("    export NVIDIA_API_KEY='your-key-here'")
        print("    export INFERENCE_MAX_STEPS=5  # for quick testing")
        return False
    
    return True


async def run_inference() -> dict[str, Any] | None:
    """Launch the FlakeForge inference engine."""
    print_header("FlakeForge Inference")
    
    # Build environment variables for inference
    env = os.environ.copy()
    env.update({
        "NVIDIA_API_KEY": NVIDIA_API_KEY or "",
        "API_BASE_URL": API_BASE_URL,
        "MODEL_NAME": ANALYZER_FIXER_MODEL,  # Used for analyzer and fixer
        "JUDGE_MODEL": JUDGE_MODEL,
        "INFERENCE_MAX_STEPS": str(INFERENCE_MAX_STEPS),
        "TEMPERATURE": str(TEMPERATURE),
        "MAX_TOKENS": str(MAX_TOKENS),
        "ENV_BASE_URL": ENV_BASE_URL,
        "USE_DOCKER_IMAGE": "0",  # Use HTTP endpoint for now
    })
    
    # Find the inference script
    flakeforge_root = TEST_REPO_ROOT.parent.parent  # Go up from test_repos/timing_race_minimal -> FlakeForge
    inference_script = flakeforge_root / "inference.py"
    
    if not inference_script.exists():
        print(f"ERROR: Inference script not found at {inference_script}")
        return None
    
    print(f"Starting inference...")
    print(f"  Environment: {ENV_BASE_URL}")
    print(f"  Max steps: {INFERENCE_MAX_STEPS}")
    print(f"  Analyzer/Fixer model: {ANALYZER_FIXER_MODEL}")
    print(f"  Judge model: {JUDGE_MODEL}\n")
    
    try:
        # Run the actual inference script
        result = subprocess.run(
            [sys.executable, str(inference_script)],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Print inference output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            print(f"\nERROR: Inference failed with exit code {result.returncode}")
            return None
        
        # Try to parse the JSON summary from stdout (last JSON object)
        lines = result.stdout.strip().split('\n')
        for line in reversed(lines):
            if line.strip().startswith('{'):
                try:
                    return json.loads(line)
                except:
                    pass
        
        return None
    except Exception as e:
        print(f"ERROR running inference: {e}")
        return None


def print_instructions() -> None:
    """Print setup and usage instructions."""
    print_header("Quick Start Guide")
    
    print("Step 1: Install Dependencies")
    print("  pip install -r requirements.txt")
    print("  pip install pytest pytest-asyncio")
    
    print("\nStep 2: Verify Flakiness Locally")
    print("  python -m pytest tests/test_flaky.py::test_flaky_simple -v --count=10")
    
    print("\nStep 3: Run Inference (requires environment)")
    print("  Option A - Use Docker image:")
    print("    export USE_DOCKER_IMAGE=1")
    print("    export LOCAL_IMAGE_NAME=flakeforge-env:latest")
    print("    python test_runner.py --run-inference")
    
    print("\n  Option B - Use running server:")
    print("    # Terminal 1: Start server")
    print("    cd ../..")
    print("    uv run server --port 8000")
    print("    # Terminal 2: Run inference")
    print("    export ENV_BASE_URL='http://localhost:8000'")
    print("    python test_runner.py --run-inference")
    
    print("\nStep 4: Check Results")
    print("  cat outputs/flakeforge_summary_*.json")


def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FlakeForge inference on timing race repo")
    parser.add_argument("--show-flakiness", action="store_true", help="Run ground truth to show flakiness")
    parser.add_argument("--run-inference", action="store_true", help="Run FlakeForge inference")
    parser.add_argument("--steps", type=int, default=INFERENCE_MAX_STEPS, help="Max inference steps")
    parser.add_argument("--runs", type=int, default=10, help="Number of flakiness runs")
    
    args = parser.parse_args()
    
    # Always validate config
    if not validate_config():
        sys.exit(1)
    
    # Show flakiness by default
    if args.show_flakiness or (not args.run_inference):
        show_flakiness(args.runs)
    
    # Run inference if requested
    if args.run_inference:
        try:
            result = asyncio.run(run_inference())
            if result:
                print("\nInference completed successfully!")
                print(f"Results: {OUTPUT_DIR / f'flakeforge_summary_*.json'}")
        except KeyboardInterrupt:
            print("\n✗ Interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Error: {e}")
            sys.exit(1)
    else:
        # Show how to proceed
        print_instructions()


if __name__ == "__main__":
    main()
