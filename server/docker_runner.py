from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import List

try:
    from ..models import RunRecord
except ImportError:
    from models import RunRecord

logger = logging.getLogger(__name__)


class DockerTestRunner:
    """Runs pytest either locally or inside a Docker sandbox.

    Mode selection is controlled by environment variables:
    - USE_DOCKER_IMAGE=1 enables sandbox mode
    - LOCAL_IMAGE_NAME sets the image (default: flakeforge-env:latest)
    """

    def __init__(self, repo_path: str) -> None:
        self.repo_path = Path(repo_path)
        self.use_docker_image = os.getenv("USE_DOCKER_IMAGE", "0").strip().lower() in {"1", "true", "yes"}
        self.local_image_name = os.getenv("LOCAL_IMAGE_NAME", "flakeforge-env:latest").strip() or "flakeforge-env:latest"
        self._docker_checked = False
        self._docker_available = False
        self._docker_unavailable_reason = ""
        self._warned_unavailable = False

    def _ensure_docker_available(self) -> bool:
        """One-time probe for docker CLI and image presence."""
        if self._docker_checked:
            return self._docker_available

        self._docker_checked = True
        try:
            cli = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if cli.returncode != 0:
                self._docker_available = False
                self._docker_unavailable_reason = (cli.stderr or cli.stdout or "docker CLI probe failed").strip()
                return False

            img = subprocess.run(
                ["docker", "image", "inspect", self.local_image_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self._docker_available = img.returncode == 0
            if not self._docker_available:
                self._docker_unavailable_reason = (
                    img.stderr or img.stdout or f"docker image '{self.local_image_name}' not found"
                ).strip()
            return self._docker_available
        except Exception as exc:
            self._docker_available = False
            self._docker_unavailable_reason = f"exception while probing docker: {exc}"
            return False

    def _maybe_warn_docker_unavailable(self) -> None:
        if self.use_docker_image and not self._warned_unavailable:
            self._warned_unavailable = True
            logger.warning(
                "USE_DOCKER_IMAGE is enabled but sandbox execution is unavailable; falling back to local pytest. reason=%s",
                self._docker_unavailable_reason or "unknown",
            )

    def _pytest_cmd(self, test_id: str) -> List[str]:
        return ["pytest", test_id, "--tb=short", "-q", "--no-header"]

    def _run_local_pytest(self, test_id: str, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self._pytest_cmd(test_id),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=self.repo_path,
        )

    def _run_docker_pytest(self, test_id: str, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        mount_src = str(self.repo_path.resolve())
        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--cpus", os.getenv("FF_DOCKER_CPUS", "1.0"),
            "--memory", os.getenv("FF_DOCKER_MEMORY", "2g"),
            "-v", f"{mount_src}:/workspace",
            "-w", "/workspace",
            self.local_image_name,
            *self._pytest_cmd(test_id),
        ]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=self.repo_path,
        )

    def run_test(self, test_id: str, timeout_seconds: int = 5) -> RunRecord:
        start = time.perf_counter()
        try:
            if self.use_docker_image and self._ensure_docker_available():
                proc = self._run_docker_pytest(test_id, timeout_seconds)
            else:
                if self.use_docker_image:
                    self._maybe_warn_docker_unavailable()
                proc = self._run_local_pytest(test_id, timeout_seconds)
            duration_ms = int((time.perf_counter() - start) * 1000)
            output = f"{proc.stdout}\n{proc.stderr}".strip()
            passed = proc.returncode == 0
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
        repo_root = self.repo_path.resolve()
        tests_root = repo_root / "tests"
        if not tests_root.exists():
            return False

        try:
            cmd = [
                "pytest",
                str(tests_root),
                f"--ignore={repo_root / exclude_test_file}",
                "-x",
                "-q",
            ]
            if self.use_docker_image and self._ensure_docker_available():
                mount_src = str(repo_root)
                proc = subprocess.run(
                    [
                        "docker", "run", "--rm",
                        "--network", "none",
                        "--cpus", os.getenv("FF_DOCKER_CPUS", "1.0"),
                        "--memory", os.getenv("FF_DOCKER_MEMORY", "2g"),
                        "-v", f"{mount_src}:/workspace",
                        "-w", "/workspace",
                        self.local_image_name,
                        *cmd,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=repo_root,
                )
            else:
                if self.use_docker_image:
                    self._maybe_warn_docker_unavailable()
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=repo_root,
                )
            # pytest exits with code 5 when every test file was excluded and
            # no tests were collected. That is not a regression; it just means
            # this tiny target repo only has the flaky test file.
            return proc.returncode not in (0, 5)
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
