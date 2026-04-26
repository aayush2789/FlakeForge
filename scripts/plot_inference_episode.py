#!/usr/bin/env python3
"""Visualize a FlakeForge inference episode JSON (stdout from `python inference.py`).

Generates a multi-panel figure: step rewards, pass rate & patch success, heatmap of
`reward_breakdown` keys, and key traces. Requires matplotlib.

  python scripts/plot_inference_episode.py \\
    -i data/inference_example_episode.json \\
    -o docs/assets/inference_episode_dashboard.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_episode(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _trajectory_matrix(
    episode: Dict[str, Any]
) -> Tuple[List[int], List[str], np.ndarray]:
    traj: List[Dict[str, Any]] = episode.get("trajectory") or []
    if not traj:
        raise ValueError("episode has empty trajectory")
    # Collect all breakdown keys; prefer trajectory rows
    key_set: set[str] = set()
    for row in traj:
        rb = row.get("reward_breakdown") or {}
        key_set |= set(rb.keys())
    for row in episode.get("reward_breakdown_history") or []:
        key_set |= set(row.keys())
    if "total" in key_set:
        keys = [k for k in sorted(key_set) if k != "total"] + ["total"]
    else:
        keys = sorted(key_set)
    n = len(traj)
    m = len(keys)
    Z = np.zeros((m, n), dtype=np.float64)
    steps: List[int] = []
    for j, row in enumerate(traj):
        steps.append(int(row.get("step", j + 1)))
        rb = {k: float(v) for k, v in (row.get("reward_breakdown") or {}).items()}
        for i, k in enumerate(keys):
            Z[i, j] = rb.get(k, np.nan)
    return steps, keys, Z


def plot_episode(episode: Dict[str, Any], out_path: Path, dpi: int = 150) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("ggplot")
        except OSError:
            pass

    traj: List[Dict[str, Any]] = episode.get("trajectory") or []
    _, keys, Z = _trajectory_matrix(episode)
    n = len(traj)
    x = np.arange(1, n + 1)
    steps_arr = [int(t.get("step", i)) for i, t in enumerate(traj, start=1)]

    rewards = [float(t.get("reward", 0.0)) for t in traj]
    cum = np.cumsum(rewards)
    pass_rates = [float(t.get("pass_rate", 0.0)) for t in traj]
    applied = [bool(t.get("patch_applied", False)) for t in traj]
    conf = [float(t.get("predicted_confidence", 0.0)) for t in traj]

    total_r = float(episode.get("total_reward", cum[-1] if len(cum) else 0.0))
    done = str(episode.get("done_reason", ""))
    final_pr = float(episode.get("final_pass_rate", pass_rates[-1] if pass_rates else 0.0))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor("#fafbfc")
    supt = (
        f"Inference episode  ·  total_reward={total_r:.2f}  ·  final_pass_rate={final_pr:.2f}  "
        f"·  done_reason={done}"
    )
    fig.suptitle(supt, fontsize=12, fontweight="600", color="#0f172a", y=1.01)

    # --- Panel: rewards + cumulative ---
    ax0 = axes[0, 0]
    c_bar = "#3b82f6"
    c_cum = "#b45309"
    ax0.bar(x, rewards, color=c_bar, alpha=0.85, edgecolor="white", linewidth=0.5, label="Step reward")
    ax0.set_xticks(x)
    ax0.set_xticklabels([str(s) for s in steps_arr], fontsize=9)
    ax0.set_xlabel("Environment step", fontsize=10)
    ax0.set_ylabel("Step reward", color=c_bar, fontsize=10)
    ax0.tick_params(axis="y", labelcolor=c_bar)
    ax0.axhline(0.0, color="#94a3b8", linewidth=0.8, linestyle="--")
    ax0_t = ax0.twinx()
    ax0_t.plot(x, cum, color=c_cum, linewidth=2.2, marker="o", markersize=5, label="Cumulative")
    ax0_t.set_ylabel("Cumulative reward", color=c_cum, fontsize=10)
    ax0_t.tick_params(axis="y", labelcolor=c_cum)
    ax0.set_title("Reward per step and cumulative", fontsize=11, loc="left", color="#1e293b")
    h1, l1 = ax0.get_legend_handles_labels()
    h2, l2 = ax0_t.get_legend_handles_labels()
    ax0.legend(h1 + h2, l1 + l2, loc="lower left", framealpha=0.95, fontsize=8)

    # --- Panel: pass rate + patch applied + confidence ---
    ax1 = axes[0, 1]
    ax1.fill_between(x, 0, pass_rates, color="#8b5cf6", alpha=0.15, step="mid")
    ax1.plot(x, pass_rates, color="#7c3aed", linewidth=2, marker="s", markersize=5, label="pass_rate")
    for i, ok in enumerate(applied):
        ax1.axvline(
            x[i],
            ymin=0.02,
            ymax=0.12,
            color=("#10b981" if ok else "#ef4444"),
            linewidth=3,
            alpha=0.9,
        )
    ax1.plot(x, conf, color="#0ea5e9", linewidth=1.5, linestyle="--", label="predicted_confidence", alpha=0.9)
    ax1.set_ylim(-0.05, 1.12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in steps_arr], fontsize=9)
    ax1.set_xlabel("Environment step", fontsize=10)
    ax1.set_ylabel("Rate / confidence", fontsize=10)
    ax1.set_title("Pass rate, confidence, patch applied (green=applied, red=failed)", fontsize=11, loc="left", color="#1e293b")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.95)

    # --- Panel: heatmap of breakdown components ---
    ax2 = axes[1, 0]
    vmax = max(np.nanmax(np.abs(Z)), 1e-6)
    im = ax2.imshow(
        Z,
        aspect="auto",
        cmap="RdYlBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax2.set_yticks(np.arange(len(keys)))
    ax2.set_yticklabels(keys, fontsize=8)
    ax2.set_xticks(np.arange(n))
    ax2.set_xticklabels([str(s) for s in steps_arr], fontsize=8, rotation=0)
    ax2.set_xlabel("Step", fontsize=10)
    ax2.set_title("Reward breakdown (rows = JSON keys, cols = step)", fontsize=11, loc="left", color="#1e293b")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.034, pad=0.04)
    cbar.set_label("Component value", fontsize=9)

    # --- Panel: key traces (selected keys) ---
    ax3 = axes[1, 1]
    palette = [
        ("stability", "#dc2626"),
        ("oracle_reasoning", "#059669"),
        ("compile", "#d97706"),
        ("regression", "#7c2d12"),
        ("format", "#4f46e5"),
    ]
    for name, color in palette:
        ys = [float((t.get("reward_breakdown") or {}).get(name, np.nan)) for t in traj]
        if all(np.isnan(ys)):
            continue
        ax3.plot(x, ys, "o-", color=color, linewidth=1.8, markersize=4, label=name, alpha=0.9)
    ax3.axhline(0.0, color="#94a3b8", linewidth=0.8, linestyle="--")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(s) for s in steps_arr], fontsize=9)
    ax3.set_xlabel("Environment step", fontsize=10)
    ax3.set_ylabel("Component value", fontsize=10)
    ax3.set_title("Traces: stability, oracle, compile, regression, format", fontsize=11, loc="left", color="#1e293b")
    ax3.legend(loc="best", fontsize=8, ncol=2, framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot inference episode JSON to PNG dashboard.")
    p.add_argument("-i", "--input", type=Path, required=True, help="Episode JSON (e.g. saved from stdout)")
    p.add_argument("-o", "--output", type=Path, default=Path("docs/assets/inference_episode_dashboard.png"))
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    episode = _load_episode(args.input)
    plot_episode(episode, args.output, dpi=args.dpi)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
