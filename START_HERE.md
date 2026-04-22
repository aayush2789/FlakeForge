# 🎊 FLAKEFORGE INFERENCE TEST - READY TO RUN!

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                   ✅ SETUP COMPLETE & VERIFIED                              ║
║                                                                              ║
║  FlakeForge Inference Engine + Test Repository ready for first mission!     ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 📦 What You Have

### 1. Inference Engine ✅
```
inference.py (296 lines)
├─ Analyzer Role (hypothesis detection)
├─ Fixer Role (repair action selection)
├─ Judge Backend (Minimax scoring)
└─ Async execution with real-time logging
```

**Status**: Ready to connect to environment and run

### 2. Test Repository ✅
```
test_repos/timing_race_minimal/
├─ source.py (flaky code - 80% timeout)
├─ tests/test_flaky.py (test cases)
├─ test_runner.py (orchestration)
├─ requirements.txt (dependencies)
├─ pytest.ini (config)
├─ README.md (documentation)
├─ .git/ (git initialized)
└─ CONFIRMED FLAKINESS: 2/15 passes = 13% ✓
```

**Status**: Git committed, ready to test

### 3. Supporting Tools ✅
```
├─ QUICKSTART.py (validation in 1 command)
├─ QUICKSTART.bat (Windows batch script)
├─ .env.template (configuration template)
├─ INFERENCE_TEST_READY.md (getting started)
├─ TEST_SETUP_COMPLETE.md (this project)
├─ test_repos/SETUP.md (detailed setup)
└─ RUN_TEST.sh (execution guide)
```

**Status**: All documented and ready

---

## 🎯 To Get Started

### Quickest Way (1 command):
```bash
export NVIDIA_API_KEY="your-key"
python QUICKSTART.py
```

### On Windows:
```cmd
set NVIDIA_API_KEY=your-key
QUICKSTART.bat
```

This will:
- ✅ Validate environment
- ✅ Check API key
- ✅ Run test 10 times to show flakiness
- ✅ Tell you exactly what to do next

---

## 📊 Test Validation Results

```
Ground Truth Validation: 15 runs of test_fetch_should_complete()
┌─────────────────────────────────────────────────────────┐
│ Run  1: ✗ FAIL (timeout)                               │
│ Run  2: ✗ FAIL (timeout)                               │
│ Run  3: ✓ PASS                                         │
│ Run  4: ✓ PASS                                         │
│ Run  5-15: ✗✗✗✗✗✗✗✗✗✗✗ (all timeout)              │
├─────────────────────────────────────────────────────────┤
│ Summary: 2 PASSED, 13 FAILED                           │
│ Pass Rate: 13%                                         │
│ Root Cause: TIMING_RACE (80% chance of tight timeout) │
│ Status: READY FOR FLAKEFORGE FIXING ✓                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Two Ways to Test

### Option 1: Show Flakiness Only (1 minute)
```bash
cd test_repos/timing_race_minimal
python test_runner.py --show-flakiness --runs 20
```

Shows: Test failing repeatedly due to timeout race

### Option 2: Full Inference (requires running server)
```bash
# Terminal 1: Start engine
uv run server --port 8000

# Terminal 2: Run inference
cd test_repos/timing_race_minimal
export ENV_BASE_URL=http://localhost:8000
python test_runner.py --run-inference --steps 5
```

Expected flow:
```
Step 1: Analyzer → "TIMING_RACE detected"
        Fixer → "ADD_TIMING_GUARD"
        Pass rate: 13% → 30%

Step 2: Analyzer → "TIMING_RACE (refine)"
        Fixer → "ADD_TIMING_GUARD (increase more)"
        Pass rate: 30% → 95%

Step 3: Analyzer → "Looks good!"
        Fixer → "ADD_SYNCHRONIZATION safety"
        Pass rate: 95% → 100% ✓

Result: Fixed! Pass rate went from 13% → 100%
        Summary saved to outputs/
```

---

## ✨ Configuration

Minimal `.env`:
```env
NVIDIA_API_KEY=your-nvidia-api-key-here
MODEL_NAME=nvidia/llama-3.1-nemotron-70b-instruct
JUDGE_MODEL=minimaxai/minimax-m2.7
ENV_BASE_URL=http://localhost:8000
INFERENCE_MAX_STEPS=5
```

---

## 📋 Before Running

- [ ] Python 3.10+ 
- [ ] pip install openai pytest pytest-asyncio python-dotenv
- [ ] NVIDIA API key
- [ ] (Optional) Docker if using docker option

---

## 📊 File Checklist

```
✅ inference.py                    Main inference engine (296 lines)
✅ test_repos/timing_race_minimal/ Test repo with git + flakiness
  ✅ source.py                     Flaky code
  ✅ tests/test_flaky.py           Test cases
  ✅ test_runner.py                Orchestration
  ✅ .git/                         Git initialized
✅ QUICKSTART.py                   One-command validation
✅ QUICKSTART.bat                  Windows script
✅ .env.template                   Config template
✅ INFERENCE_TEST_READY.md         Getting started guide
✅ TEST_SETUP_COMPLETE.md          Full documentation
✅ test_repos/SETUP.md             Detailed setup
✅ RUN_TEST.sh                     Execution guide
```

---

## 🎬 Action Items  

### NOW (Take 5 minutes):
```bash
python QUICKSTART.py
```

This validates everything and shows you the exact next command.

### NEXT (When ready for full test):
Follow the instructions from QUICKSTART.py

---

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  Everything is ready! Run QUICKSTART.py and follow the prompts.            ║
║                                                                              ║
║  The test repo is confirmed flaky (13% pass rate) and ready to be fixed     ║
║  by the FlakeForge inference engine.                                        ║
║                                                                              ║
║  🚀 Let's go! 🚀                                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 📞 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| API key error | `export NVIDIA_API_KEY="your-key"` |
| Server not found | Start with: `uv run server --port 8000` |
| Test not flaky | Run with more iterations: `--runs 50` |
| Model not found | Use: `meta-llama/llama-3-8b-instruct` |
| Import errors | `pip install openai pytest-asyncio` |

---

**That's it! You're ready to test FlakeForge. 🎉**
