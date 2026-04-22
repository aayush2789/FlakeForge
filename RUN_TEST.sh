#!/usr/bin/env bash
# FlakeForge Inference Test - Step by Step Execution Guide
# This script documents the exact steps to run inference

set -e

echo "=================================================="
echo "  FlakeForge Inference - Test Execution Script"
echo "=================================================="
echo ""

# Step 1: Validate prerequisites
echo "Step 1: Checking Prerequisites..."
echo "=================================="
python QUICKSTART.py
echo ""

# Step 2: Prepare test environment
echo "Step 2: Setting Up Test Environment"
echo "===================================="
mkdir -p outputs
cd test_repos/timing_race_minimal

echo "✓ Test repo directory: $(pwd)"
echo "✓ Test files present:"
ls -1 tests/test_flaky.py source.py 2>/dev/null || echo "  ✗ Files not found"
echo ""

# Step 3: Show inference configuration
echo "Step 3: Inference Configuration"
echo "==============================="
echo "Environment variables:"
echo "  API_BASE_URL=${API_BASE_URL:-https://integrate.api.nvidia.com/v1}"
echo "  MODEL_NAME=${MODEL_NAME:-nvidia/llama-3.1-nemotron-70b-instruct}"
echo "  ENV_BASE_URL=${ENV_BASE_URL:-http://localhost:8000}"
echo "  INFERENCE_MAX_STEPS=${INFERENCE_MAX_STEPS:-5}"
echo "  USE_DOCKER_IMAGE=${USE_DOCKER_IMAGE:-0}"
echo ""

# Step 4: Show what test_runner does
echo "Step 4: Test Runner Capabilities"
echo "=================================="
echo "Available commands:"
echo "  python test_runner.py"
echo "    --show-flakiness    Show ground truth (run test N times)"
echo "    --run-inference     Launch FlakeForge inference"
echo "    --steps N           Set inference steps (default: from env)"
echo "    --runs N            Set flakiness test runs (default: 10)"
echo ""

# Step 5: Ready
echo "Step 5: Ready to Run!"
echo "===================="
echo ""
echo "Choose one:"
echo ""
echo "Option A - Show Flakiness (quick, ~30 seconds):"
echo "  python test_runner.py --show-flakiness --runs 20"
echo ""
echo "Option B - Run Full Inference (requires server, ~2-5 minutes):"
echo "  # Terminal 1: Start server"
echo "  cd ../.."
echo "  export NVIDIA_API_KEY='your-key'"
echo "  uv run server --port 8000"
echo ""
echo "  # Terminal 2: Run inference"
echo "  export NVIDIA_API_KEY='your-key'"
echo "  export ENV_BASE_URL='http://localhost:8000'"
echo "  python test_runner.py --run-inference --steps 5"
echo ""
echo "For more options, see: SETUP.md"
echo ""
