# moderate_load_jitter_flaky

A **moderate-difficulty** flaky test repo for FlakeForge.

## What's the Flake?

`tests/test_flaky.py::test_request_processing_should_succeed` has a **~30%
failure rate** (well under 0.5) caused by one moderate bug in `source.py`.

### Saturating Worker Pool (~30% failure)

`WorkerPool.submit()` performs its queue-capacity check **outside** the lock,
and simulates concurrent-caller jitter. About 30% of the time it pretends
another thread just grabbed the last slot, and silently returns `False`
(queue_full), even when the queue has plenty of room.

### Failure Rate

| State | Approx failure rate |
|---|---|
| Default flaky state | ~30% |
| Queue submit fixed | ~0% |

## Running the Test

```bash
# Single run
python -m pytest tests/test_flaky.py::test_request_processing_should_succeed -v

# 20 runs to measure flakiness
for i in {1..20}; do
  python -m pytest tests/test_flaky.py::test_request_processing_should_succeed -v --tb=no -q
done
```

On Windows PowerShell:

```powershell
1..20 | ForEach-Object {
    python -m pytest tests/test_flaky.py::test_request_processing_should_succeed --tb=no -q
}
```

## Running with FlakeForge

```powershell
$env:FF_REPO_PATH = "test_repos/moderate_load_jitter_flaky"
$env:FF_TEST_ID   = "tests/test_flaky.py::test_request_processing_should_succeed"
$env:INFERENCE_MAX_STEPS = "6"
python inference.py
```

## Expected Repair Episode

| Step | Action | Expected outcome |
|---|---|---|
| 1 | GATHER_EVIDENCE | Agent identifies `queue_full` failures |
| 2 | FIX_QUEUE_RACE | Remove the 30% jitter branch in `WorkerPool.submit()` | Pass rate ~100% |
| 3+ | VALIDATE | Repeated runs confirm stable behavior |

## File Structure

```
moderate_load_jitter_flaky/
├── source.py            # WorkerPool queue-submit flake + stable ConfigStore
├── tests/
│   └── test_flaky.py   # Primary flaky test + isolation helpers
├── pytest.ini
├── requirements.txt
└── README.md
```

## Resetting to the Flaky State

If a previous FlakeForge run patched `source.py`, run:

```bash
python test_repos/moderate_load_jitter_flaky/reset_demo.py
```
