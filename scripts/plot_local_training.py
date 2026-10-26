# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.11.2", "numpy==2.4.6"]
# ///
"""Plot the completed first GRPO run, without loading a model."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from plot_experiments import BLUE, GRID, MUTED, ORANGE, plt, save, style

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    run = ROOT / "results/local-mps-grpo-now"
    summary = json.loads((run / "grpo-summary.json").read_text())
    assert summary["completed_steps"] == 100 and summary["reload_predictions_exact"]
    metrics = {row["step"]: row for row in rows(run / "grpo-metrics.jsonl") if "reward" in row}
    resources = {row["step"]: row for row in rows(run / "grpo-resources.jsonl")}
    episodes = defaultdict(list)
    for episode in rows(run / "grpo-episodes.jsonl"):
        episodes[episode["update_before_generation"] + 1].append(episode)
    steps = sorted(metrics)
    assert len(steps) == 100 and all(len(episodes[step]) == 4 for step in steps)
    style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.80, bottom=0.19, hspace=0.72, wspace=0.36)
    fig.text(0.07, 0.95, "First local GRPO run: 100 updates completed", fontsize=19, weight="bold")
    fig.text(
        0.07,
        0.895,
        "Qwen 1.5B / MPS · starts from the 20-example SFT warmup, not the full 8,000-example checkpoint",
        fontsize=10,
        color=MUTED,
    )
    specifications = [
        (
            [metrics[s]["reward"] for s in steps],
            "Training reward",
            "Mean reward across four rollouts",
            BLUE,
        ),
        (
            [np.mean([len(e["attempts"]) for e in episodes[s]]) for s in steps],
            "Decisions per rollout",
            "Mean attempted moves",
            ORANGE,
        ),
        (
            [metrics[s]["grad_norm"] for s in steps],
            "Optimizer gradient norm",
            "Norm before the optimizer step",
            BLUE,
        ),
        (
            [resources[s]["mps_driver_bytes"] / 2**30 for s in steps],
            "MPS driver allocation",
            "GiB at update-end readout (not peak)",
            BLUE,
        ),
    ]
    for ax, (values, title, ylabel, color) in zip(axes.flat, specifications, strict=True):
        ax.plot(steps, values, color=color, alpha=0.3, linewidth=1)
        rolling = np.convolve(values, np.ones(10) / 10, mode="valid")
        ax.plot(steps[9:], rolling, color=color, linewidth=2)
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(1, 100)
        ax.grid(axis="y", color=GRID)
        ax.set_ylim(bottom=min(0, min(values) - 0.2) if min(values) < 0 else 0)
    axes[0, 1].axhline(16, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[0, 1].set_ylim(0, 17)
    axes[0, 1].text(98, 16.4, "Episode cap", ha="right", fontsize=9, color=MUTED)
    zero_steps = [s for s in steps if metrics[s]["frac_reward_zero_std"] == 1]
    axes[1, 0].set_ylim(bottom=-0.04)
    axes[1, 0].scatter(zero_steps, [0] * len(zero_steps), color=ORANGE, zorder=4, s=25)
    fig.text(
        0.07,
        0.115,
        "Thin lines: recorded update values. Thick lines: trailing 10-update mean. Orange dots: tied-reward groups.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.075,
        "400 training rollouts; no full-game evaluation. Training reward is a short-episode proxy for gameplay quality.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.035,
        "Source: results/local-mps-grpo-now/ · all 100 updates shown",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "local-grpo-first-100")


if __name__ == "__main__":
    main()
