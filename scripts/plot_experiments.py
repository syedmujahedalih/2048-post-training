# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.11.2", "numpy==2.4.6"]
# ///
"""Render experimental figures from committed measurements; no model runs needed.

Run: uv run --managed-python --script scripts/plot_experiments.py
PNG previews and editable SVGs are written to docs/figures/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
INK = "#172A3A"
MUTED = "#566674"
GRID = "#E2E7EB"
BLUE = "#2855C5"
ORANGE = "#B96432"
GOLD = "#997316"


def load(path):
    return json.loads((ROOT / path).read_text())


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "2048-post-training",
            "savefig.dpi": 180,
        }
    )


def heading(fig, title, subtitle):
    fig.text(0.07, 0.95, title, fontsize=19, weight="bold", va="top")
    fig.text(0.07, 0.875, subtitle, fontsize=10, color=MUTED, va="top")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(
            OUT / f"{name}.{extension}", metadata={"Date": None} if extension == "svg" else {}
        )
    svg = OUT / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(f"docs/figures/{name}.png (+ SVG)")


def mps_overview():
    report = load("results/mps-trl-student-check.json")
    stages = [report["stages"][name] for name in ("sft", "grpo")]
    assert all(stage["status"] == "passed" for stage in stages)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.3))
    fig.subplots_adjust(left=0.09, right=0.94, top=0.71, bottom=0.36, wspace=0.48)
    heading(
        fig,
        "Qwen 1.5B: SFT and GRPO updates ran on MPS",
        "Apple M4 / 24 GB unified memory · TRL 1.16.0 · LoRA rank 4 · one observed update per stage",
    )
    specifications = [
        (
            "Observed training time",
            "Seconds (model loading excluded)",
            "training_seconds",
            1,
            "{:.2f} s",
        ),
        (
            "MPS driver allocation at readout",
            "GiB (snapshot, not peak memory)",
            "mps_driver_allocated_bytes",
            2**30,
            "{:.2f} GiB",
        ),
    ]
    for ax, (title, xlabel, field, divisor, label) in zip(axes, specifications, strict=True):
        values = [stage[field] / divisor for stage in stages]
        ax.barh([0, 1], values, color=[BLUE, ORANGE], height=0.45, zorder=3)
        ax.set_yticks([0, 1], ["SFT", "GRPO"])
        ax.set_ylim(1.6, -0.6)
        ax.set_xlim(0, max(values) * 1.35)
        ax.set_title(title, loc="left", pad=17)
        ax.set_xlabel(xlabel, labelpad=12, fontsize=10)
        ax.grid(axis="x", color=GRID, zorder=0)
        for y, value in enumerate(values):
            ax.text(value + max(values) * 0.035, y, label.format(value), va="center", weight="bold")
    fig.text(
        0.07,
        0.19,
        "Both checks: finite, nonzero MPS gradients · changed adapter weights · exact adapter save/load",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.13,
        "SFT: 1 example, 20 target tokens. GRPO: 4 outputs, capped at 8 tokens, synthetic reward.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.085,
        "One observation per workload; no timing error bars or learning curve. Allocation is not total system RAM.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.035,
        "Source: results/mps-trl-student-check.json · unsupported MPS operator fallback disabled",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "mps-trl-overview")


def mps_rewards():
    report = load("results/mps-trl-student-check.json")
    groups = report["stages"]["grpo"]["synthetic_grpo_rewards"]
    assert len(groups) == 1 and len(groups[0]) == 4
    rewards = np.array(groups[0])
    mean = rewards.mean()
    centered = rewards - mean
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.3))
    fig.subplots_adjust(left=0.10, right=0.94, top=0.69, bottom=0.29, wspace=0.43)
    heading(
        fig,
        "One GRPO group supplied a relative learning signal",
        "Actual recorded rewards from the MPS probe · synthetic token reward · no 2048 gameplay measured",
    )
    labels = ["1", "2", "3", "4"]
    axes[0].bar(range(4), rewards, color=BLUE, width=0.5, zorder=3)
    axes[0].axhline(mean, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.3)
    axes[0].set_ylim(0, 0.8)
    axes[0].set_title("Recorded synthetic rewards", loc="left", pad=24)
    axes[0].text(
        0.98,
        1.035,
        f"Group mean: {mean:.3f}",
        transform=axes[0].transAxes,
        ha="right",
        color=MUTED,
        fontsize=10,
    )
    axes[0].set_ylabel("Reward (arbitrary units)")
    axes[1].bar(
        range(4),
        centered,
        color=[BLUE if value >= 0 else ORANGE for value in centered],
        width=0.5,
        zorder=3,
    )
    axes[1].axhline(0, color=MUTED, linewidth=1)
    axes[1].set_ylim(-0.30, 0.34)
    axes[1].set_title("Centered rewards (calculated)", loc="left", pad=24)
    axes[1].set_ylabel("Reward minus group mean")
    for ax in axes:
        ax.set_xticks(range(4), labels)
        ax.set_xlabel("Sampled completion", labelpad=10)
        ax.grid(axis="y", color=GRID, zorder=0)
    for i, value in enumerate(rewards):
        axes[0].text(i, value + 0.035, f"{value:.3f}", ha="center", fontsize=10)
    for i, value in enumerate(centered):
        axes[1].text(
            i,
            value + (0.025 if value >= 0 else -0.025),
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
        )
    fig.text(
        0.07,
        0.17,
        "The right panel is a reconstruction from saved rewards, before any standard-deviation normalization.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.11,
        "This confirms variation in one sampled group; it does not show that the policy improved.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.035,
        "Source: results/mps-trl-student-check.json · all four completions shown",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "mps-grpo-reward-group")


def decider_scores():
    report = load("results/decider-afterstates/summary.json")
    players = ["random", "priority", "greedy", "decider", "teacher"]
    names = ["Random legal", "Fixed priority", "Greedy merge", "Frozen Decider", "Depth-2 teacher"]
    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    fig.subplots_adjust(left=0.20, right=0.84, top=0.75, bottom=0.29)
    heading(
        fig,
        "Decider pilot: individual game scores",
        "Six shared seeds per player · frozen MLX inference · standalone players, not adviser-assisted RL",
    )
    offsets = np.linspace(-0.13, 0.13, 6)
    for y, player in enumerate(players):
        info = report["gameplay"][player]
        scores = info["scores"]
        assert len(scores) == 6
        color = BLUE if player == "decider" else GOLD if player == "teacher" else MUTED
        for index, score in enumerate(scores):
            seed = report["game_seed_start"] + index
            trace = load(f"results/decider-afterstates/traces/{player}-{seed}.json")
            assert trace["score"] == score
            capped = trace["truncated"]
            ax.scatter(
                score,
                y + offsets[index],
                marker="s" if capped else "o",
                s=48,
                facecolors="none" if capped else color,
                edgecolors=color,
                linewidths=1.4,
                zorder=3,
            )
        mean = np.mean(scores)
        assert abs(mean - info["mean_score"]) < 1e-6
        ax.text(
            1.025,
            y,
            f"{mean:,.0f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=11,
            weight="bold",
        )
    ax.text(1.025, 1.06, "Mean score", transform=ax.transAxes, fontsize=10, color=MUTED)
    ax.set_yticks(range(5), names)
    ax.set_ylim(4.55, -0.55)
    ax.set_xscale("log")
    ax.set_xlim(400, 100000)
    ax.set_xticks([500, 1000, 2000, 10000, 50000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.minorticks_off()
    ax.set_xlabel("2048 score (logarithmic axis)", labelpad=12)
    ax.grid(axis="x", color=GRID, zorder=0)
    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", color=MUTED, linestyle="none", label="One game", markersize=6
            ),
            Line2D(
                [],
                [],
                marker="s",
                markerfacecolor="none",
                color=GOLD,
                linestyle="none",
                label="Capped partial score",
                markersize=6,
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.15, 0.135),
        frameon=False,
        ncol=3,
        fontsize=10,
    )
    fig.text(
        0.07,
        0.10,
        "Two teacher games hit the 3,000-decision cap. All other games ended normally. No confidence interval is inferred.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.04,
        "Source: results/decider-afterstates/summary.json and traces/ · seeds 700000–700005",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "decider-game-scores")


def decider_pairs():
    report = load("results/decider-afterstates/summary.json")
    decider = np.array(report["gameplay"]["decider"]["scores"])
    priority = np.array(report["gameplay"]["priority"]["scores"])
    differences = decider - priority
    assert differences.tolist() == report["paired_score_differences_decider_minus_priority"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fig.subplots_adjust(left=0.16, right=0.93, top=0.72, bottom=0.27)
    heading(
        fig,
        "Paired results vary across the six seeds",
        "Frozen Decider score minus fixed-priority score · positive values favor Decider",
    )
    ax.axvline(0, color=MUTED, linewidth=1.2)
    for y, difference in enumerate(differences):
        color = BLUE if difference > 0 else ORANGE
        ax.hlines(y, 0, difference, color=color, linewidth=2, alpha=0.5)
        ax.scatter(difference, y, s=65, color=color, zorder=3)
        ax.annotate(
            f"{difference:+,}",
            (difference, y),
            xytext=(10 if difference > 0 else -10, 0),
            textcoords="offset points",
            ha="left" if difference > 0 else "right",
            va="center",
            fontsize=10,
        )
    ax.set_yticks(range(6), [str(report["game_seed_start"] + i) for i in range(6)])
    ax.set_ylim(5.55, -0.55)
    ax.set_xlim(-1550, 1650)
    ax.set_xticks([-1000, 0, 1000])
    ax.set_xlabel("Paired score difference (points)", labelpad=12)
    ax.set_ylabel("Game seed", labelpad=12)
    ax.grid(axis="x", color=GRID, zorder=0)
    fig.text(
        0.07,
        0.16,
        f"Observed mean difference: {differences.mean():+.0f} points. Four positive pairs, two negative; n = 6.",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.10,
        "This exploratory pilot does not establish a reliable adviser benefit or measure adviser-assisted learning.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.04,
        "Source: results/decider-afterstates/summary.json · no new model runs",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "decider-paired-differences")


def main():
    style()
    mps_overview()
    mps_rewards()
    decider_scores()
    decider_pairs()


if __name__ == "__main__":
    main()
