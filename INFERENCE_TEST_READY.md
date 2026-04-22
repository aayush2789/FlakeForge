# 🚀 FlakeForge Test Setup Complete

## ✅ Test Repository Created

A minimal test repo with a **confirmed timing race condition** has been created at:
```
d:\Workspace\FlakeForge\test_repos\timing_race_minimal\
```

### Flakiness Confirmed ✓

Running `test_fetch_should_complete()` 15 times showed:
- **2 passes** (13% success rate)
- **13 failures** (87% fail rate - timing timeout)

This is the **ground truth** we want FlakeForge to fix!

---

## 📁 What Was Created

```
FlakeForge/
├── inference.py                         # Main inference script ✓
├── QUICKSTART.py                        # Quick validation script ✓  
├── .env.template                        # Environment variables template ✓
└── test_repos/
    ├── SETUP.md                         # Detailed setup guide
    └── timing_race_minimal/             # Test repository
        ├── source.py                    # Flaky async code with 80% timeout race
        ├── tests/
        │   └── test_flaky.py            # Flaky test cases
        ├── test_runner.py               # Test orchestration script
        ├── pytest.ini                   # Pytest config
        ├── requirements.txt             # Dependencies
        ├── .gitignore                   # Git ignore
        ├── README.md                    # Repo documentation
        └── .git/                        # Git repository ✓
```

---

## 🔧 Quick Start (3 Steps)

### Step 1: Set Your NVIDIA API Key

```bash
# Copy the template
cp .env.template .env

# Edit .env and fill in your NVIDIA API key:
# NVIDIA_API_KEY=your-key-here
```

Or set it directly:
```bash
export NVIDIA_API_KEY="your-nvidia-api-key"
```

### Step 2: Validate Setup

```bash
# Run the quick start validation
python QUICKSTART.py
```

This will:
- ✓ Check Python packages
- ✓ Verify API key
- ✓ Run test 10 times to show flakiness
- ✓ Display next steps

### Step 3: Run Inference

Choose Option A or B below:

---

## 🏃 Run Inference

### Option A: Docker (Requires Docker)

```bash
# Build the FlakeForge container (one time)
docker build -t flakeforge-env:latest -f server/Dockerfile .

# Set environment
export NVIDIA_API_KEY="your-key"
export USE_DOCKER_IMAGE="1"
export INFERENCE_MAX_STEPS="5"  # Quick test

# Run inference
cd test_repos/timing_race_minimal
python test_runner.py --run-inference
```

### Option B: Local Server (Recommended for Testing)

**Terminal 1 - Start the Server:**
```bash
export NVIDIA_API_KEY="your-nvidia-key"
uv run server --port 8000
```

**Terminal 2 - Run Inference:**
```bash
export NVIDIA_API_KEY="your-nvidia-key"
export ENV_BASE_URL="http://localhost:8000"

cd test_repos/timing_race_minimal
python test_runner.py --run-inference --steps 5
```

---

## 📊 What Happens During Inference

### Flow:
```
Reset env
  ↓
Run test 10 times → Baseline pass rate (~13%)
  ↓
Loop for up to 5 steps:
  ├─ Phase 1: ANALYZER forms hypothesis
  │   └─ Predicts: TIMING_RACE (80% confidence)
  ├─ Phase 2: FIXER selects action
  │   └─ Action: ADD_TIMING_GUARD or ADD_RETRY
  ├─ Validate: Run test 20 times
  │   └─ Check if pass rate improved
  └─ Score: Judge evaluates hypothesis & patch
      └─ Use Minimax model for scoring
  ↓
End Episode
  ├─ Success: pass_rate = 100%
  └─ Summary saved to: outputs/flakeforge_summary_*.json
```

### Expected Results:

- **Step 1**: Baseline → ~13% pass rate
- **Step 2**: ADD_TIMING_GUARD → Pass rate jumps to ~85%+
- **Step 3**: Validation → Should reach 100% after fix applied
- **Judge Score**: High scores if analyzer identified TIMING_RACE correctly
- **Final Pass Rate**: Should be 100% (all 20 validation runs pass)

---

## 🔍 Model Configuration

We're using **small, fast models** for quick testing:

```
Analyzer/Fixer:  nvidia/llama-3.1-nemotron-70b-instruct
Judge:          minimaxai/minimax-m2.7 (as agreed)
```

Switch models by updating `.env`:
```bash
# For ultra-fast testing (but lower quality)
export MODEL_NAME="meta-llama/llama-3-8b-instruct"

# For better quality analysis
export MODEL_NAME="nvidia/llama-3.1-nemotron-70b-instruct"
```

---

## 📋 Troubleshooting

### ❌ "API Key not set"
```bash
export NVIDIA_API_KEY="your-key-here"
```

### ❌ "Unable to connect to environment"
```bash
# Make sure server is running:
lsof -i :8000

# If not, start it in another terminal:
uv run server --port 8000
```

### ❌ "Model not found"
Try a different model:
```bash
export MODEL_NAME="meta-llama/llama-3-8b-instruct"
```

### ❌ "Flaky test not showing flakiness"
The timeout values can vary. Increase the sample size:
```bash
python -c "
import asyncio, sys; sys.path.insert(0, '.')
from source import fetch_data_with_race
for i in range(1, 51):
    r = asyncio.run(fetch_data_with_race())
    print('✓' if r['success'] else '✗', end='')
print()  # Show 50-run pattern
"
```

---

## 📈 Next Steps

1. ✅ Validate flakiness with QUICKSTART.py
2. ✅ Set API key in .env
3. ⏭️  Run inference with test_runner.py
4. ⏭️  Check results in outputs/
5. ⏭️  Iterate with different small models
6. ⏭️  Once working, scale to full training episodes

---

## 🎯 Expected Output Files

After each inference run:

```
outputs/
├── flakeforge_inference_20260422_143022.log
│   └── Detailed step-by-step logs
└── flakeforge_summary_20260422_143022.json
    └── Complete episode summary with metrics
```

Exit the summary file to see:
```bash
cat outputs/flakeforge_summary_*.json | python -m json.tool
```

---

## 💡 Key Files Reference

| File | Purpose |
|------|---------|
| `inference.py` | Main inference script (Analyzer → Fixer loop) |
| `test_repos/timing_race_minimal/test_runner.py` | Test orchestration and validation |
| `test_repos/timing_race_minimal/source.py` | The flaky code (80% timeout race) |
| `test_repos/timing_race_minimal/tests/test_flaky.py` | Test cases (13% pass rate) |
| `.env.template` | Environment variables template |
| `QUICKSTART.py` | One-command validation of setup |

---

## 🏁 You're Ready!

```bash
# To get started immediately:
python QUICKSTART.py
```

This will validate everything and show you exactly which command to run next!

---

**Happy testing! 🚀**
