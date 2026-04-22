from __future__ import annotations

import concurrent.futures
import re
import subprocess
import time
from pathlib import Path
from typing import List

try:
    from ..models import RunRecord
except ImportError:
    from models import RunRecord


class DockerTestRunner:
    """Runs pytest commands inside the active environment container context."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = Path(repo_path)

    def run_test(self, test_id: str, timeout_seconds: int = 5) -> RunRecord:
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                ["pytest", test_id, "--tb=short", "-q", "--no-header"],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=self.repo_path,
            )
            duration_ms = int((time.perf_counter() - start) * 1000)
            output = f"{proc.stdout}\n{proc.stderr}".strip()
            passed = proc.returncode == 0 and "1 passed" in output
            error_type = self._extract_error_type(output)
            error_message = self._extract_error_message(output)
            return RunRecord(
                passed=passed,
                duration_ms=duration_ms,
                error_type=error_type,
                error_message=error_message,
                stderr_excerpt=(proc.stderr or "")[-500:],
            )
        except subprocess.TimeoutExpired:
            duration_ms = int((time.perf_counter() - start) * 1000)
            return RunRecord(
                passed=False,
                duration_ms=duration_ms,
                error_type="TimeoutError",
                error_message=f"pytest timed out after {timeout_seconds}s",
                stderr_excerpt=None,
            )
        except Exception as exc:  # pragma: no cover
            duration_ms = int((time.perf_counter() - start) * 1000)
            return RunRecord(
                passed=False,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
                error_message=str(exc),
                stderr_excerpt=None,
            )

    def run_test_n_times(self, test_id: str, n: int, max_workers: int = 4) -> List[RunRecord]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda _: self.run_test(test_id), range(n)))

    def check_regressions(self, exclude_test_id: str, timeout_seconds: int = 30) -> bool:
        exclude_test_file = exclude_test_id.split("::", 1)[0]
        tests_root = self.repo_path / "tests"
        if not tests_root.exists():
            return False

        try:
            proc = subprocess.run(
                [
                    "pytest",
                    str(tests_root),
                    f"--ignore={exclude_test_file}",
                    "-x",
                    "-q",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=self.repo_path,
            )
            return proc.returncode != 0
        except subprocess.TimeoutExpired:
            return True

    @staticmethod
    def _extract_error_type(output: str) -> str | None:
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*Error|Exception)\b", output)
        return match.group(1) if match else None

    @staticmethod
    def _extract_error_message(output: str) -> str | None:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return None
        return lines[-1][:200]
