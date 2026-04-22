# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pillar 4 — Performance Regression Sentinel.

After the agent applies a deep architectural fix, this module checks whether
the fix introduced a meaningful performance slowdown by running a small,
deterministic benchmark and comparing its timing distribution against a
pre-fix baseline using the Mann-Whitney U statistical test.

Design rationale:
  A global threading.Lock on the DB connection will stop all flakiness, but
  serialise every write and make the app 10x slower. FlakeForge v1 cannot
  detect this — it only cares about pass/fail. The Sentinel closes that gap.

Reference:
  Mostafa et al., "Risk-Aware Batch Testing for Performance Regression
  Detection", arXiv 2025 — statistical Change-Point Detection on timing
  distributions for CI without full benchmark suites.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional scipy import — degrade gracefully when not installed
try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _scipy_stats = None  # type: ignore
    _SCIPY_AVAILABLE = False
    logger.warning(
        "perf_sentinel: scipy not available — "
        "falling back to median-ratio-only regression check."
    )

try:
    import numpy as _np
    _NUMPY_AVAILABLE = True
except ImportError:
    _np = None  # type: ignore
    _NUMPY_AVAILABLE = False


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PerformanceBaseline:
    """Timing distribution captured during reset() before any patches."""
    benchmark_test_id: str
    timing_distribution_ms: List[float]
    p50_ms: float
    p95_ms: float
    sample_count: int

    @classmethod
    def from_timings(cls, test_id: str, timings: List[float]) -> "PerformanceBaseline":
        if not timings:
            return cls(test_id, [], 0.0, 0.0, 0)
        sorted_t = sorted(timings)
        n = len(sorted_t)
        p50 = sorted_t[n // 2]
        p95 = sorted_t[int(n * 0.95)] if n > 1 else sorted_t[-1]
        return cls(
            benchmark_test_id=test_id,
            timing_distribution_ms=timings,
            p50_ms=p50,
            p95_ms=p95,
            sample_count=n,
        )


@dataclass
class SentinelResult:
    """Result of a single regression check."""
    is_regression: bool
    median_ratio: float          # post_median / baseline_median (1.0 = no change)
    p_value: Optional[float]     # Mann-Whitney p-value (None if scipy unavailable)
    post_p50_ms: float
    baseline_p50_ms: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regression": self.is_regression,
            "median_ratio": round(self.median_ratio, 3),
            "p_value": round(self.p_value, 4) if self.p_value is not None else None,
            "post_p50_ms": round(self.post_p50_ms, 1),
            "baseline_p50_ms": round(self.baseline_p50_ms, 1),
            "message": self.message,
        }

    def penalty(self) -> float:
        """
        Reward penalty for a detected regression.
        Formula: 10 × log(median_ratio), capped at 25.
        A 2× slowdown → 6.93 penalty.
        A 10× slowdown → 23.03 penalty (worse than the 15-pt regression penalty).
        """
        if not self.is_regression:
            return 0.0
        return min(25.0, 10.0 * math.log(max(1.0, self.median_ratio)))


# ── Main sentinel class ────────────────────────────────────────────────────────

class PerformanceSentinel:
    """
    Captures a timing baseline and checks for regressions after patches.

    Parameters
    ----------
    slowdown_threshold:
        Minimum median_ratio to consider a regression (default 1.5 = 50% slower).
    p_value_cutoff:
        Statistical significance level for the Mann-Whitney U test (default 0.05).
    n_benchmark_runs:
        Number of benchmark runs for both baseline and post-fix checks.
    """

    def __init__(
        self,
        slowdown_threshold: float = 1.5,
        p_value_cutoff: float = 0.05,
        n_benchmark_runs: int = 10,
    ) -> None:
        self.slowdown_threshold = slowdown_threshold
        self.p_value_cutoff = p_value_cutoff
        self.n_benchmark_runs = n_benchmark_runs
        self._baseline: Optional[PerformanceBaseline] = None

    @property
    def has_baseline(self) -> bool:
        return self._baseline is not None

    def capture_baseline(self, runner: Any, benchmark_test_id: str) -> PerformanceBaseline:
        """
        Run the benchmark test n times via `runner` and store the timing baseline.

        ``runner`` must expose a ``run_test_n_times(test_id, n)`` method that
        returns a list of RunRecord objects with a ``duration_ms`` field.
        """
        records = runner.run_test_n_times(benchmark_test_id, n=self.n_benchmark_runs)
        timings = [float(r.duration_ms) for r in records]
        self._baseline = PerformanceBaseline.from_timings(benchmark_test_id, timings)
        logger.debug(
            "perf_sentinel: baseline captured | test=%s p50=%.1fms p95=%.1fms n=%d",
            benchmark_test_id,
            self._baseline.p50_ms,
            self._baseline.p95_ms,
            self._baseline.sample_count,
        )
        return self._baseline

    def check_regression(
        self, runner: Any, benchmark_test_id: Optional[str] = None
    ) -> SentinelResult:
        """
        Run the benchmark again and compare against the stored baseline.

        Returns a SentinelResult. If no baseline has been captured or the
        benchmark test cannot be found, returns a safe no-regression result.
        """
        if self._baseline is None:
            return SentinelResult(
                is_regression=False,
                median_ratio=1.0,
                p_value=None,
                post_p50_ms=0.0,
                baseline_p50_ms=0.0,
                message="No baseline captured — skipping performance check.",
            )

        test_id = benchmark_test_id or self._baseline.benchmark_test_id
        try:
            post_records = runner.run_test_n_times(test_id, n=self.n_benchmark_runs)
        except Exception as exc:
            logger.warning("perf_sentinel: benchmark run failed: %s", exc)
            return SentinelResult(
                is_regression=False,
                median_ratio=1.0,
                p_value=None,
                post_p50_ms=0.0,
                baseline_p50_ms=self._baseline.p50_ms,
                message=f"Benchmark run error — skipping: {exc}",
            )

        post_timings = [float(r.duration_ms) for r in post_records]
        if not post_timings:
            return SentinelResult(
                is_regression=False,
                median_ratio=1.0,
                p_value=None,
                post_p50_ms=0.0,
                baseline_p50_ms=self._baseline.p50_ms,
                message="No timings collected — skipping.",
            )

        post_p50 = sorted(post_timings)[len(post_timings) // 2]
        baseline_p50 = self._baseline.p50_ms

        median_ratio = (post_p50 / baseline_p50) if baseline_p50 > 0 else 1.0

        # Statistical test (requires scipy)
        p_value: Optional[float] = None
        stat_significant = True   # default: rely only on median ratio when scipy absent
        if _SCIPY_AVAILABLE and len(self._baseline.timing_distribution_ms) >= 5:
            try:
                _, p_value = _scipy_stats.mannwhitneyu(
                    self._baseline.timing_distribution_ms,
                    post_timings,
                    alternative="less",   # H1: baseline < post (i.e. post is slower)
                )
                stat_significant = p_value < self.p_value_cutoff
            except Exception as exc:
                logger.debug("perf_sentinel: scipy test error: %s", exc)

        is_regression = stat_significant and (median_ratio > self.slowdown_threshold)

        if is_regression:
            pct = (median_ratio - 1.0) * 100
            msg = (
                f"PERFORMANCE REGRESSION: {pct:.1f}% slower "
                f"(baseline p50={baseline_p50:.1f}ms → post p50={post_p50:.1f}ms, "
                f"ratio={median_ratio:.2f}x, p={p_value:.4f if p_value is not None else 'N/A'})"
            )
        else:
            msg = (
                f"Performance OK: ratio={median_ratio:.2f}x "
                f"(baseline={baseline_p50:.1f}ms → post={post_p50:.1f}ms)"
            )

        logger.debug("perf_sentinel: %s", msg)
        return SentinelResult(
            is_regression=is_regression,
            median_ratio=median_ratio,
            p_value=p_value,
            post_p50_ms=post_p50,
            baseline_p50_ms=baseline_p50,
            message=msg,
        )
