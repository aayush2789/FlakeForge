# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pillar 2 — Chaos-Amplified RL Environment.

Runs pytest suites under controlled resource stress (CPU, memory, network
latency) so that infrastructure-sensitive race conditions reliably surface
in the sandbox — not just in production.

Design rationale:
  Deep codebase flakiness (race conditions under CPU load, GC-pause timeouts,
  thundering-herd retries under network latency) only manifests when the host
  is stressed. An RL agent cannot learn to fix what it cannot reproduce.
  ChaosAmplifiedRunner injects stress-ng / tc netem before each test batch
  and annotates RunRecords with the active chaos profile.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from enum import Enum
from typing import List, Optional

try:
    from ..models import RunRecord
    from .docker_runner import DockerTestRunner
except ImportError:
    try:
        from FlakeForge.models import RunRecord  # type: ignore
        from FlakeForge.server.docker_runner import DockerTestRunner  # type: ignore
    except ImportError:
        from models import RunRecord  # type: ignore
        from server.docker_runner import DockerTestRunner  # type: ignore

logger = logging.getLogger(__name__)

# ── Chaos profiles ─────────────────────────────────────────────────────────────

class ChaosProfile(str, Enum):
    NONE = "none"         # V1 baseline — no stressor
    CPU = "cpu"           # 2 stress-ng workers at 100% CPU
    MEM = "mem"           # Consume 70% of available container RAM
    NET = "net"           # 50ms ± 20ms network latency via tc netem
    COMPOUND = "compound" # All three simultaneously (production nightmare mode)


# ── Stressor command templates ──────────────────────────────────────────────────

_STRESSOR_CMDS: dict[ChaosProfile, list[list[str]]] = {
    ChaosProfile.CPU: [
        ["stress-ng", "--cpu", "2", "--timeout", "120s", "--metrics-brief"],
    ],
    ChaosProfile.MEM: [
        ["stress-ng", "--vm", "1", "--vm-bytes", "70%", "--timeout", "120s", "--metrics-brief"],
    ],
    ChaosProfile.NET: [
        # Add latency — requires iproute2 (tc) to be installed and NET_ADMIN capability
        ["tc", "qdisc", "add", "dev", "eth0", "root", "netem",
         "delay", "50ms", "20ms", "distribution", "normal"],
    ],
    ChaosProfile.COMPOUND: [
        ["stress-ng", "--cpu", "2", "--vm", "1", "--vm-bytes", "60%", "--timeout", "120s"],
        ["tc", "qdisc", "add", "dev", "eth0", "root", "netem",
         "delay", "50ms", "20ms", "distribution", "normal"],
    ],
}

_NET_TEARDOWN_CMD = ["tc", "qdisc", "del", "dev", "eth0", "root"]


# ── Main class ──────────────────────────────────────────────────────────────────

class ChaosAmplifiedRunner(DockerTestRunner):
    """
    Extends DockerTestRunner with the ability to inject stress before each batch.

    Usage::

        runner = ChaosAmplifiedRunner("/app/seed_repos/cpu_timing_race")
        records = runner.run_test_n_times_chaos(
            test_id="tests/test_flaky.py::test_flaky_case",
            n=20,
            profile=ChaosProfile.CPU,
        )
    """

    STRESSOR_WARMUP_SECONDS: float = 2.0   # time to let stressor stabilise before tests
    STRESSOR_TEARDOWN_WAIT: float = 1.0    # time to let OS recover after stressor exits

    def run_test_n_times_chaos(
        self,
        test_id: str,
        n: int,
        profile: ChaosProfile,
        max_workers: int = 4,
    ) -> List[RunRecord]:
        """
        Run `test_id` `n` times while a chaos stressor is active.

        Falls back to the normal (no-stressor) path when:
          - profile is NONE
          - the stressor binary is not available (graceful degradation)
        """
        if profile == ChaosProfile.NONE:
            return self.run_test_n_times(test_id, n=n, max_workers=max_workers)

        stressor_procs = self._start_stressor(profile)
        if not stressor_procs:
            # Binaries unavailable — degrade gracefully and log warning
            logger.warning(
                "chaos_runner: stressor unavailable for profile=%s — "
                "falling back to clean run. Install stress-ng and iproute2 "
                "on the host to enable chaos amplification.",
                profile.value,
            )
            return self.run_test_n_times(test_id, n=n, max_workers=max_workers)

        try:
            time.sleep(self.STRESSOR_WARMUP_SECONDS)
            records = self.run_test_n_times(test_id, n=n, max_workers=max_workers)
        finally:
            self._stop_stressors(stressor_procs, profile)
            time.sleep(self.STRESSOR_TEARDOWN_WAIT)

        return records

    def is_infrastructure_sensitive(
        self,
        test_id: str,
        clean_pass_rate: float,
        profile: ChaosProfile = ChaosProfile.CPU,
        n: int = 10,
        sensitivity_threshold: float = 0.2,
    ) -> tuple[bool, float]:
        """
        Check whether a flake worsens significantly under chaos.

        Returns (is_sensitive: bool, chaos_pass_rate: float).
        Sensitive means: chaos_pass_rate < clean_pass_rate - threshold.
        """
        chaos_records = self.run_test_n_times_chaos(test_id, n=n, profile=profile)
        passed = sum(1 for r in chaos_records if r.passed)
        chaos_pass_rate = passed / len(chaos_records) if chaos_records else 0.0
        is_sensitive = chaos_pass_rate < (clean_pass_rate - sensitivity_threshold)
        return is_sensitive, chaos_pass_rate

    # ── Internal stressor management ────────────────────────────────────────────

    def _start_stressor(self, profile: ChaosProfile) -> List[subprocess.Popen]:
        """
        Launches the stressor(s) for the given profile.
        Returns empty list if the required binary is absent.
        """
        cmds = _STRESSOR_CMDS.get(profile, [])
        procs: List[subprocess.Popen] = []

        for cmd in cmds:
            binary = cmd[0]
            if not self._binary_available(binary):
                logger.debug("chaos_runner: binary '%s' not found — skipping stressor.", binary)
                self._stop_stressors(procs, profile)
                return []
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid if os.name != "nt" else None,
                )
                procs.append(proc)
                logger.debug("chaos_runner: started stressor pid=%d cmd=%s", proc.pid, cmd)
            except Exception as exc:
                logger.warning("chaos_runner: failed to start stressor %s: %s", cmd, exc)
                self._stop_stressors(procs, profile)
                return []

        return procs

    def _stop_stressors(
        self, procs: List[subprocess.Popen], profile: ChaosProfile
    ) -> None:
        """Terminate all stressor processes and clean up network rules."""
        for proc in procs:
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
                proc.wait(timeout=5)
            except Exception as exc:
                logger.debug("chaos_runner: error stopping stressor pid=%s: %s", proc.pid, exc)

        # Clean up tc netem rules for net-affecting profiles
        if profile in (ChaosProfile.NET, ChaosProfile.COMPOUND):
            if self._binary_available("tc"):
                try:
                    subprocess.run(
                        _NET_TEARDOWN_CMD,
                        capture_output=True,
                        timeout=5,
                        check=False,
                    )
                except Exception:
                    pass

    @staticmethod
    def _binary_available(name: str) -> bool:
        """Fast check: is the binary on PATH?"""
        try:
            subprocess.run(
                [name, "--version"],
                capture_output=True,
                timeout=2,
                check=False,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
