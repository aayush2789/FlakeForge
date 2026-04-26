from __future__ import annotations

import concurrent.futures
import logging
import os
import sys
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
        # When True, prepend a `pip install` step so repos with their own deps work.
        # Enable with FF_AUTO_INSTALL_DEPS=1. Also implies network access (no --network none).
        self.auto_install_deps = os.getenv("FF_AUTO_INSTALL_DEPS", "0").strip().lower() in {"1", "true", "yes"}
        self._docker_checked = False
        self._docker_available = False
        self._docker_unavailable_reason = ""
        self._warned_unavailable = False
        self._deps_installed = False  # install once per runner instance, not per run

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
        return [
            "pytest", test_id,
            "--tb=short", "-q", "--no-header",
            # Strip project-level coverage/addopts that break collection in a
            # minimal sandbox (e.g. --cov-fail-under=100 in pydash's setup.cfg).
            "-p", "no:cov",
            "--override-ini=addopts=",
        ]

    def _pip_install_script(self) -> str:
        """Shell snippet (bash) that pip-installs whatever requirements files exist.

        Used only for Docker runs (Linux containers always have bash).
        """
        candidates = [
            "requirements.txt",
            "requirements-tests.txt",
            "requirements-test.txt",
            "test_requirements.txt",
            "requirements-dev.txt",
            "dev-requirements.txt",
        ]
        installs = " ".join(
            f'pip install -q -r {r} 2>/dev/null || true;'
            for r in candidates
        )
        return (
            f"{installs} "
            "[ -f setup.py ] && pip install -q -e . 2>/dev/null || true; "
            "[ -f pyproject.toml ] && pip install -q -e . 2>/dev/null || true; "
        )

    def _install_deps_local(self) -> None:
        """Install external dependencies from requirements files.

        Installs only once per runner instance; subsequent calls are no-ops.
        Does NOT do `pip install -e .` — the package itself is made importable
        via PYTHONPATH in the subprocess env (see _local_pytest_env).
        """
        if self._deps_installed:
            return
        install_timeout = int(os.getenv("FF_INSTALL_TIMEOUT_SECONDS", "120"))
        candidates = [
            "requirements.txt",
            "requirements-tests.txt",
            "requirements-test.txt",
            "test_requirements.txt",
            "requirements-dev.txt",
            "dev-requirements.txt",
        ]
        pip = [sys.executable, "-m", "pip", "install", "-q", "-r"]
        for req in candidates:
            req_path = self.repo_path / req
            if req_path.exists():
                try:
                    subprocess.run([*pip, str(req_path)], capture_output=True, cwd=self.repo_path, timeout=install_timeout)
                except subprocess.TimeoutExpired:
                    logger.warning("[Runner] pip install timed out for %s", req)
        self._deps_installed = True

    def _local_pytest_env(self) -> dict:
        """Build subprocess env that makes the package importable without pip install -e."""
        env = os.environ.copy()
        # Add both the repo root and the common src-layout dir so that packages
        # in flat or src/ layout are importable without an editable install.
        paths = [str(self.repo_path), str(self.repo_path / "src")]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(p for p in paths + ([existing] if existing else []))
        return env

    def _run_local_pytest(self, test_id: str, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        if self.auto_install_deps:
            self._install_deps_local()
        # Use sys.executable -m pytest so it always resolves to the active venv.
        pytest_cmd = [sys.executable, "-m", *self._pytest_cmd(test_id)]
        return subprocess.run(
            pytest_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=self.repo_path,
            env=self._local_pytest_env(),
        )

    def _run_docker_pytest(self, test_id: str, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        mount_src = str(self.repo_path.resolve())
        base_docker = [
            "docker", "run", "--rm",
            "--cpus", os.getenv("FF_DOCKER_CPUS", "1.0"),
            "--memory", os.getenv("FF_DOCKER_MEMORY", "2g"),
            "-v", f"{mount_src}:/workspace",
            "-w", "/workspace",
        ]
        if self.auto_install_deps:
            # Need network to pip-install; chain install + pytest in a single bash call.
            shell_cmd = self._pip_install_script() + " ".join(self._pytest_cmd(test_id))
            cmd = [*base_docker, self.local_image_name, "bash", "-c", shell_cmd]
        else:
            cmd = [*base_docker, "--network", "none", self.local_image_name, *self._pytest_cmd(test_id)]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=self.repo_path,
        )

    def run_test(self, test_id: str, timeout_seconds: int | None = None) -> RunRecord:
        if timeout_seconds is None:
            timeout_seconds = int(os.getenv("FF_TEST_TIMEOUT_SECONDS", "30"))
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

    def run_test_full_file(self, test_id: str, timeout_seconds: int | None = None) -> RunRecord:
        """Run the whole test *file* containing test_id, then parse whether test_id passed.

        This is needed for ORDER_DEPENDENCY / RESOURCE_LEAK / SHARED_STATE categories
        where the flakiness only manifests when other tests in the same file run first and
        leave behind shared state.  In this mode each call still returns a RunRecord for the
        single test, but the subprocess runs the full file so side-effects accumulate.
        """
        if timeout_seconds is None:
            timeout_seconds = int(os.getenv("FF_TEST_TIMEOUT_SECONDS", "30"))
        file_path = test_id.split("::")[0]
        # Run with -v so we can parse PASSED/FAILED per-test lines.
        verbose_pytest = [file_path, "--tb=short", "-v", "--no-header", "-p", "no:cov", "--override-ini=addopts="]
        start = time.perf_counter()
        try:
            if self.use_docker_image and self._ensure_docker_available():
                mount_src = str(self.repo_path.resolve())
                base_docker = [
                    "docker", "run", "--rm",
                    "--cpus", os.getenv("FF_DOCKER_CPUS", "1.0"),
                    "--memory", os.getenv("FF_DOCKER_MEMORY", "2g"),
                    "-v", f"{mount_src}:/workspace",
                    "-w", "/workspace",
                ]
                if self.auto_install_deps:
                    shell_cmd = self._pip_install_script() + "pytest " + " ".join(verbose_pytest)
                    proc = subprocess.run(
                        [*base_docker, self.local_image_name, "bash", "-c", shell_cmd],
                        capture_output=True, text=True, timeout=timeout_seconds, cwd=self.repo_path,
                    )
                else:
                    proc = subprocess.run(
                        [*base_docker, "--network", "none", self.local_image_name, "pytest", *verbose_pytest],
                        capture_output=True, text=True, timeout=timeout_seconds, cwd=self.repo_path,
                    )
            else:
                if self.use_docker_image:
                    self._maybe_warn_docker_unavailable()
                if self.auto_install_deps:
                    self._install_deps_local()
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", *verbose_pytest],
                    capture_output=True, text=True, timeout=timeout_seconds, cwd=self.repo_path,
                    env=self._local_pytest_env(),
                )
            duration_ms = int((time.perf_counter() - start) * 1000)
            output = f"{proc.stdout}\n{proc.stderr}".strip()
            # Parse whether the specific test function passed.
            test_fn = test_id.split("::")[-1]
            # pytest -v lines look like: "tests/foo.py::my_test PASSED" or "FAILED"
            passed = bool(re.search(rf"\b{re.escape(test_fn)}\b.*\bPASSED\b", output))
            if not passed and not re.search(rf"\b{re.escape(test_fn)}\b", output):
                # Test wasn't collected / not found — fall back to overall return code
                passed = proc.returncode == 0
            error_type = self._extract_error_type(output) if not passed else None
            error_message = self._extract_error_message(output) if not passed else None
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
                passed=False, duration_ms=duration_ms,
                error_type="TimeoutError",
                error_message=f"pytest timed out after {timeout_seconds}s",
                stderr_excerpt=None,
            )
        except Exception as exc:
            duration_ms = int((time.perf_counter() - start) * 1000)
            return RunRecord(
                passed=False, duration_ms=duration_ms,
                error_type=type(exc).__name__,
                error_message=str(exc),
                stderr_excerpt=None,
            )

    def check_regressions(self, exclude_test_id: str, timeout_seconds: int | None = None) -> bool:
        if timeout_seconds is None:
            timeout_seconds = int(os.getenv("FF_REGRESSION_TIMEOUT_SECONDS", os.getenv("FF_TEST_TIMEOUT_SECONDS", "30")))
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
