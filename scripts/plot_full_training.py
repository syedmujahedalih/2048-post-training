# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.11.2", "numpy==2.4.6"]
# ///
"""Plot the completed SFT/GRPO pipeline using saved measurements only."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.ticker import PercentFormatter
from plot_experiments import BLUE, GRID, MUTED, ORANGE, plt, save, style

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trend(ax, x, y, color, label=None, window=10):
    ax.plot(x, y, color=color, alpha=0.22, linewidth=0.8)
    smoothed = np.convolve(y, np.ones(window) / window, mode="valid")
    ax.plot(x[window - 1 :], smoothed, color=color, linewidth=2, label=label)


def sft_figure():
    run = ROOT / "results/local-mps-full-v1"
    metrics = [r for r in rows(run / "sft-metrics.jsonl") if "loss" in r]
    assert len(metrics) == 2000
    steps = [r["step"] for r in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.76, bottom=0.24, wspace=0.30)
    fig.text(0.07, 0.94, "Full SFT: every one of the 8,000 examples", fontsize=19, weight="bold")
    fig.text(
        0.07,
        0.86,
        "Qwen 1.5B / LoRA / MPS · one epoch · batch 4 · 2,000 optimizer updates · 174.4 training minutes",
        fontsize=10,
        color=MUTED,
    )
    trend(axes[0], steps, [r["loss"] for r in metrics], BLUE, window=100)
    trend(axes[1], steps, [r["mean_token_accuracy"] for r in metrics], ORANGE, window=100)
    axes[0].set_title("Completion-token loss", loc="left", pad=12)
    axes[0].set_ylabel("Cross entropy (training batches)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("Completion-token accuracy", loc="left", pad=12)
    axes[1].set_ylabel("Accuracy (training batches)")
    axes[1].set_ylim(0.80, 1.005)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set_xlabel("Optimizer update")
        ax.set_xlim(1, 2000)
        ax.grid(axis="y", color=GRID)
    fig.text(
        0.07,
        0.12,
        "Thin: recorded batches. Thick: trailing 100-update mean. These are training metrics, not held-out accuracy.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.07, 0.06, "Source: results/local-mps-full-v1/sft-metrics.jsonl", fontsize=9, color=MUTED
    )
    save(fig, "local-full-sft")


def grpo_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.79, bottom=0.18, hspace=0.66, wspace=0.34)
    fig.text(0.07, 0.95, "Two GRPO runs, different SFT starting points", fontsize=19, weight="bold")
    fig.text(
        0.07,
        0.90,
        "100 updates and 400 short training rollouts per run · 4 rollouts per board · 16-attempt cap",
        fontsize=10,
        color=MUTED,
    )
    for name, label, color in [
        ("local-mps-grpo-now", "After 20-example warmup", BLUE),
        ("local-mps-full-v1", "After 8,000-example SFT", ORANGE),
    ]:
        run = ROOT / "results" / name
        metrics = [r for r in rows(run / "grpo-metrics.jsonl") if "reward" in r]
        resources = {r["step"]: r for r in rows(run / "grpo-resources.jsonl")}
        groups = defaultdict(list)
        for episode in rows(run / "grpo-episodes.jsonl"):
            groups[episode["update_before_generation"] + 1].append(episode)
        steps = [r["step"] for r in metrics]
        series = [
            [r["reward"] for r in metrics],
            [np.mean([len(e["attempts"]) for e in groups[s]]) for s in steps],
            [np.mean([e["early_exit"] for e in groups[s]]) for s in steps],
            [resources[s]["mps_driver_bytes"] / 2**30 for s in steps],
        ]
        for ax, values in zip(axes.flat, series, strict=True):
            trend(ax, steps, values, color, label)
    titles = [
        "Training reward",
        "Attempted moves per rollout",
        "Rollouts stopping early",
        "MPS driver allocation",
    ]
    labels = [
        "Mean reward across 4 rollouts",
        "Mean attempted moves",
        "Fraction of 4 rollouts",
        "GiB at update end (not peak)",
    ]
    for ax, title, label in zip(axes.flat, titles, labels, strict=True):
        ax.set_title(title, loc="left", pad=10)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xlabel("Optimizer update")
        ax.set_xlim(1, 100)
        ax.grid(axis="y", color=GRID)
    axes[0, 0].set_ylim(bottom=-0.25)
    axes[0, 1].set_ylim(0, 17)
    axes[0, 1].axhline(16, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[1, 0].set_ylim(-0.04, 1.04)
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 1].set_ylim(bottom=0)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, names, loc="upper left", bbox_to_anchor=(0.07, 0.865), ncol=2, frameon=False
    )
    fig.text(
        0.07,
        0.10,
        "Thin: raw update values. Thick: trailing 10-update means. Training reward favors longer valid short episodes.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.055,
        "One training seed per path; repeated training boards. These curves do not establish complete-game strength.",
        fontsize=9.5,
        color=MUTED,
    )
    save(fig, "local-grpo-comparison")


def fixed_board_check():
    records = []
    expected_ids = None
    for label, path, key in [
        ("Original instruct model", "local-mps-full-v1/sft-summary.json", "before_validation"),
        ("20-example SFT warmup", "local-mps-pilot/sft-summary.json", "after_validation"),
        ("Warmup + GRPO", "local-mps-grpo-now/grpo-summary.json", "after_validation"),
        ("8,000-example SFT", "local-mps-full-v1/sft-summary.json", "after_validation"),
        ("Full SFT + GRPO", "local-mps-full-v1/grpo-summary.json", "after_validation"),
    ]:
        report = json.loads((ROOT / "results" / path).read_text())[key]
        predictions = report["predictions"]
        ids = [p["id"] for p in predictions]
        if expected_ids is None:
            expected_ids = ids
        assert ids == expected_ids and len(ids) == 12
        records.append(
            (
                label,
                sum(p["legal"] for p in predictions),
                sum(p["teacher_match"] for p in predictions),
            )
        )
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.74, bottom=0.25)
    fig.text(
        0.07, 0.94, "A small fixed-board check, not a game benchmark", fontsize=18, weight="bold"
    )
    fig.text(
        0.07,
        0.86,
        "Same 12 validation boards · greedy first move · 48-token generation cap",
        fontsize=10,
        color=MUTED,
    )
    y = np.arange(len(records))
    ax.barh(y - 0.16, [r[1] for r in records], height=0.28, color=BLUE, label="Legal output")
    ax.barh(
        y + 0.16,
        [r[2] for r in records],
        height=0.28,
        color=ORANGE,
        label="Teacher first-choice match",
    )
    for i, (_, legal, teacher) in enumerate(records):
        ax.text(legal + 0.15, i - 0.16, f"{legal}/12", va="center", fontsize=10)
        ax.text(teacher + 0.15, i + 0.16, f"{teacher}/12", va="center", fontsize=10)
    ax.set_yticks(y, [r[0] for r in records])
    ax.invert_yaxis()
    ax.set_xlim(0, 13.6)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xlabel("Boards out of 12")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=2, frameon=False, fontsize=10)
    fig.text(
        0.07,
        0.13,
        "Legal moves appear in the prompt. Teacher agreement is imitation, not proof of optimal play.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.075,
        "No complete-game evaluation or 2048 win-rate claim; one board changes the measured rate by 8.3 points.",
        fontsize=9.5,
        color=MUTED,
    )
    save(fig, "local-fixed-board-check")


if __name__ == "__main__":
    style()
    sft_figure()
    grpo_comparison()
    fixed_board_check()
