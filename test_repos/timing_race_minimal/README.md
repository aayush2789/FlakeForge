# Timing Race Minimal Test Repo

A minimal test repository with a simple **timing race condition** for testing the FlakeForge inference pipeline.

## What's the Flake?

The `test_flaky_simple()` test in `tests/test_flaky.py` has a ~70% pass rate because:

1. The `fetch_data_with_race()` function in `source.py` uses a **0.05 second timeout** 80% of the time.
2. The underlying async operation takes ~0.15 seconds to complete.
3. This causes a **timing race**: the timeout fires before the operation finishes.

**Expected behavior**: Test should always pass.
**Actual behavior**: Test fails ~30% of the time due to timeout.

## Running the Test Manually

```bash
# Single run (might pass or fail)
python -m pytest tests/test_flaky.py::test_flaky_simple -v

# Multiple runs to see the flakiness
for i in {1..10}; do python -m pytest tests/test_flaky.py::test_flaky_simple -v; done
```

## The Fix

The agent (Analyzer + Fixer) should:
1. **Analyze** the code and identify a timing race condition as the root cause.
2. **Fix** by either:
   - Increasing the timeout in `fetch_data_with_race()` to >= 0.15s
   - Adding a `ADD_TIMING_GUARD` to give the operation more time
   - Using `ADD_RETRY` with backoff to handle transient timeouts

## File Structure

```
timing_race_minimal/
├── source.py                 # The code with the flaky operation
├── tests/
│   └── test_flaky.py        # Flaky test cases
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Integration with FlakeForge

This repo is meant to be tested by the FlakeForge inference engine. The engine should:

1. **Reset** the repo to a clean state
2. **Gather evidence** by running the test multiple times and analyzing logs
3. **Hypothesize** the root cause (TIMING_RACE)
4. **Execute fixes** using one or more repair actions
5. **Validate** that the test now passes consistently (20+ runs)

## Expected Episodes

- **Step 1**: GATHER_EVIDENCE (inject logging, understand failure pattern)
- **Step 2**: ADD_TIMING_GUARD (increase timeout or add delay before critical operation)
- **Step 3+**: Validate that the fix works (20 test runs with high pass rate)
