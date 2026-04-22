# FlakeForge Test Repo Setup Instructions

## Quick Setup

### 1. Initialize Git Repository

```bash
cd test_repos/timing_race_minimal
git init
git config user.email "test@flakeforge.local"
git config user.name "FlakeForge Test"
git add .
git commit -m "Initial flaky test repository"
```

### 2. Verify the Flakiness Locally

```bash
# Install dependencies
pip install pytest pytest-asyncio

# Run the flaky test multiple times
python -m pytest tests/test_flaky.py::test_flaky_simple -v

# Run 10 times in a loop to see the flakiness
for i in {1..10}; do echo "=== Run $i ==="; python -m pytest tests/test_flaky.py::test_flaky_simple -v --tb=short; done
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root (or export them from terminal):

```bash
# NVIDIA credentials
export NVIDIA_API_KEY="your-nvidia-api-key"

# For analyzer and fixer (small models for speed)
export MODEL_NAME="nvidia/llama-3.1-nemotron-70b-instruct"  # Or: "meta-llama/llama-3-8b-instruct"

# For judge (Minimax as requested)
export JUDGE_MODEL="minimaxai/minimax-m2.7"

# API endpoint
export API_BASE_URL="https://integrate.api.nvidia.com/v1"

# Environment configuration
export ENV_BASE_URL="http://localhost:8000"
export USE_DOCKER_IMAGE="0"  # Set to "1" to use Docker

# For quicker testing
export INFERENCE_MAX_STEPS="5"
export MAX_TOKENS="900"
export TEMPERATURE="0.1"
```

### 4. Option A: Run Inference with Docker

```bash
# Build the FlakeForge Docker image
cd ../..
docker build -t flakeforge-env:latest -f server/Dockerfile .

# Run inference
cd test_repos/timing_race_minimal
export USE_DOCKER_IMAGE="1"
python test_runner.py --run-inference
```

### 5. Option B: Run Inference with Server

**Terminal 1: Start the FlakeForge Server**

```bash
cd ../..
export NVIDIA_API_KEY="your-key"
uv run server --port 8000
```

**Terminal 2: Run Inference**

```bash
cd test_repos/timing_race_minimal
export NVIDIA_API_KEY="your-key"
export ENV_BASE_URL="http://localhost:8000"
python test_runner.py --run-inference --steps 5
```

## What Happens During Inference

1. **Phase 1 - Analysis**: The Analyzer role examines the test failure and forms a hypothesis
   - Expected: Detects "TIMING_RACE" as root cause
   
2. **Phase 2 - Execution**: The Fixer role applies a repair action
   - Could be: `ADD_TIMING_GUARD`, `ADD_RETRY`, or `ADD_SYNCHRONIZATION`
   
3. **Validation**: The environment runs the test 20 times after each step
   - If pass rate reaches 1.0, episode ends successfully
   - Each step gets scored by the Judge (Minimax)

4. **Results**: Summary saved to `outputs/flakeforge_summary_<timestamp>.json`

## Expected Results

If everything works correctly:

- **Step 1**: GATHER_EVIDENCE - pass rate stays low (~30%)
- **Step 2**: ADD_TIMING_GUARD or ADD_RETRY - pass rate improves significantly 
- **Step 3+**: Validation runs showing 100% pass rate
- **Final pass rate**: Should reach 1.0 (100%)
- **Total reward**: Positive (since we fixed the flake)

## Troubleshooting

### Error: "Unable to connect to environment"
→ Make sure the server is running on port 8000 or that Docker image is built

### Error: "NVIDIA_API_KEY not set"
→ Set the environment variable: `export NVIDIA_API_KEY="your-key"`

### Test still flaky after "fix"
→ The repair action may not have been applied correctly
→ Check the logs in `outputs/flakeforge_inference_*.log`

### Models not found
→ NVIDIA endpoints may be temporarily unavailable
→ Try using a different model: `export MODEL_NAME="meta-llama/llama-3-70b-instruct"`

## Model Options (Small for Speed)

### For Analyzer/Fixer (choose one):
- `nvidia/llama-3.1-nemotron-70b-instruct` (recommended)
- `meta-llama/llama-3-8b-instruct`
- `mistralai/mixtral-8x7b-instruct-v0.1`

### For Judge:
- `minimaxai/minimax-m2.7` (as agreed)

## File Structure After Test

```
test_repos/timing_race_minimal/
├── source.py                 # Flaky async code
├── tests/
│   └── test_flaky.py        # Flaky tests
├── test_runner.py            # Test orchestration script
├── requirements.txt          # Python dependencies
├── pytest.ini                # Pytest config
├── README.md                 # This repo's docs
├── .env                      # Environment file (create manually)
├── .gitignore                # Git ignore patterns
└── outputs/                  # Results directory (created after runs)
    ├── flakeforge_inference_*.log
    └── flakeforge_summary_*.json
```

## Next Steps

1. Confirm flakiness locally with repeated runs
2. Set API key and update environment variables
3. Choose small models to keep latency low
4. Run test_runner.py to validate the setup
5. Once working, move to full training runs

Good luck! 🚀
