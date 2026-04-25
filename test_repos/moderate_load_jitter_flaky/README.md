# moderate_load_jitter_flaky

A **moderate-difficulty** flaky test repo for FlakeForge.

## What's the Flake?

`tests/test_flaky.py::test_request_processing_should_succeed` has a **~35-40%
failure rate** (well under 0.5) caused by two independent bugs in `source.py`.

### Gate 1 – Saturating Worker Pool (~30% failure)

`WorkerPool.submit()` performs its queue-capacity check **outside** the lock,
and simulates concurrent-caller jitter. About 30% of the time it pretends
another thread just grabbed the last slot, and silently returns `False`
(queue_full), even when the queue has plenty of room.

### Gate 2 – Config Stale-Read Window (~15% failure)

`ConfigStore.read()` can return `None` ~15% of the time because a simulated
background refresh briefly sets `_data = None` before writing the new dict.
A reader that lands in that window gets `None` back and the request is rejected
with `config_stale`.

### Combined Failure Rate

| State | Approx failure rate |
|---|---|
| Both gates active (default) | ~35-40% |
| Gate 1 fixed (remove jitter) | ~15% |
| Both gates fixed | ~0% |

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
| 1 | GATHER_EVIDENCE | Agent identifies two error strings: `queue_full` and `config_stale` |
| 2 | FIX_GATE_1 | Remove the 30% jitter branch in `WorkerPool.submit()` | Pass rate ~60% |
| 3 | FIX_GATE_2 | Add a read lock or remove the None swap in `ConfigStore.read()` | Pass rate ~100% |
| 4+ | VALIDATE | 20 runs confirm ≥95% pass rate |

## File Structure

```
moderate_load_jitter_flaky/
├── source.py            # WorkerPool + ConfigStore with two flaky gates
├── tests/
│   └── test_flaky.py   # Primary flaky test + isolation helpers
├── pytest.ini
├── requirements.txt
└── README.md
```

## Resetting to the Flaky State

If a previous FlakeForge run patched `source.py`, restore it from git:

```bash
git checkout -- test_repos/moderate_load_jitter_flaky/source.py
```
