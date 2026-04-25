# Multi-Step Flaky Demo Repo

This target repo is meant for live FlakeForge progress demos.

The main target is:

```powershell
tests/test_flaky.py::test_profile_fetch_should_be_stable
```

It has two independent flaky gates:

1. `fetch_profile()` randomly uses a `0.03s` timeout around an async operation that takes about `0.10s`.
2. After the timeout is fixed, `build_payload()` still randomly returns `"stale-request"` instead of `"stable-request"`.

That shape usually gives a visible multi-step episode: the first patch improves or reveals the timeout gate, then a later patch fixes the nondeterministic payload.

Run it with:

```powershell
$env:FF_REPO_PATH = "test_repos/multi_step_flaky"
$env:FF_TEST_ID = "tests/test_flaky.py::test_profile_fetch_should_be_stable"
$env:INFERENCE_MAX_STEPS = "5"
python inference.py
```

If the repo was already patched by a previous run, reset `source.py` to the original flaky version before running again.

```powershell
python test_repos/multi_step_flaky/reset_demo.py
```
