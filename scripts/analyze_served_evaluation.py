# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.11.2", "numpy==2.4.6"]
# ///
"""Replay the served-model benchmark and plot its saved measurements."""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter
from plot_experiments import BLUE, GOLD, GRID, MUTED, ORANGE, plt, save, style

from posttrain2048.cli import replay
from posttrain2048.data import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/served-grpo-v1"
AGENTS = ["grpo", "random", "priority", "greedy", "teacher"]
LABELS = [
    "GRPO model",
    "Random legal",
    "Direction priority",
    "Immediate-score greedy",
    "Depth-2 teacher",
]


def interval(samples):
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    half /= 1 + z * z / count
    return [max(0, center - half), min(1, center + half)]


def analyze():
    manifest = json.loads((RUN / "manifest.json").read_text())
    summary = json.loads((RUN / "summary.json").read_text())
    records = read_jsonl(RUN / "test-positions.jsonl")
    source = read_jsonl(ROOT / "data/demonstrations/test.jsonl")
    assert len(records) == len(source) == 1000
    assert [r["id"] for r in records] == [r["id"] for r in source]
    assert len({r["id"] for r in records}) == 1000
    for record, original in zip(records, source, strict=True):
        assert record["board_id"] == original["board_id"]
        assert record["game_id"] == original["game_id"]
        action = record["decision"]["direction"]
        assert record["legal"] == (record["decision"]["error"] is None)
        assert record["teacher_match"] == (record["legal"] and action == original["teacher_move"])
        assert (
            max(record["teacher_scores"], key=record["teacher_scores"].get)
            == original["teacher_move"]
        )
    traces = {}
    for agent in AGENTS:
        traces[agent] = []
        for seed in manifest["game_seeds"]:
            trace = json.loads((RUN / "games" / f"{agent}-{seed}.json").read_text())
            assert trace["seed"] == seed and trace["max_steps"] == 3000
            replay(trace)
            if agent == "grpo":
                assert trace["requests"] == len(trace["decisions"])
                failures = [d for d in trace["decisions"] if d["decision"]["error"]]
                assert len(failures) == int(bool(trace["policy_failure"]))
                if failures:
                    assert failures[-1] == trace["decisions"][-1]
                    assert not trace["terminated"] and trace["ending"] == "policy_failure"
                else:
                    assert trace["terminated"] or trace["attempts"] == 3000
            traces[agent].append(trace)
    rng = np.random.default_rng(20261026)
    game_indices = rng.integers(
        0, len(manifest["game_seeds"]), size=(10000, len(manifest["game_seeds"]))
    )
    scores = {a: np.array([t["score"] for t in ts]) for a, ts in traces.items()}
    games = {}
    for agent in AGENTS:
        games[agent] = {
            "mean_score_95_bootstrap": interval(scores[agent][game_indices].mean(axis=1)),
            "reached_2048_95_wilson": wilson(
                summary["games"][agent]["reached_2048"], len(traces[agent])
            ),
        }
        if agent != "grpo":
            differences = scores["grpo"] - scores[agent]
            games[agent]["grpo_minus_baseline_mean_score"] = float(differences.mean())
            games[agent]["grpo_minus_baseline_95_paired_bootstrap"] = interval(
                differences[game_indices].mean(axis=1)
            )
    clusters = defaultdict(list)
    for record in records:
        clusters[record["game_id"]].append(record)
    assert len(clusters) == 11
    sizes = np.array([len(rs) for rs in clusters.values()])
    matches = np.array([sum(r["teacher_match"] for r in rs) for rs in clusters.values()])
    cluster_indices = rng.integers(0, len(clusters), size=(10000, len(clusters)))
    rates = matches[cluster_indices].sum(axis=1) / sizes[cluster_indices].sum(axis=1)
    legal = [r for r in records if r["legal"]]
    regrets = [
        (max(r["teacher_scores"].values()) - r["teacher_scores"][r["decision"]["direction"]])
        / max(abs(max(r["teacher_scores"].values())), 1)
        for r in legal
    ]
    analysis = {
        "positions_checked": len(records),
        "game_traces_replayed": sum(len(ts) for ts in traces.values()),
        "game_attempts_replayed": sum(t["attempts"] for ts in traces.values() for t in ts),
        "heldout_teacher_match_95_source_game_bootstrap": interval(rates),
        "heldout_teacher_relative_regret_legal_only": float(np.mean(regrets)) if regrets else None,
        "heldout_teacher_exact_best_value_matches": sum(
            r["teacher_scores"][r["decision"]["direction"]] == max(r["teacher_scores"].values())
            for r in legal
        ),
        "games": games,
        "uncertainty_method": (
            "10,000 bootstrap resamples. Test-position teacher-match intervals resample 11 source "
            "games; score intervals resample 30 evaluation seeds, paired across policies. "
            "Wilson intervals describe observed 2048-reaching frequency. These omit training-seed "
            "uncertainty and do not establish population-wide guarantees."
        ),
    }
    (RUN / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    return summary, analysis, records, traces


def figure(summary, analysis, records, traces):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    fig.subplots_adjust(left=0.19, right=0.95, top=0.80, bottom=0.15, wspace=0.55, hspace=0.75)
    fig.text(
        0.07, 0.95, "Served GRPO: unseen boards and new game seeds", fontsize=20, weight="bold"
    )
    fig.text(
        0.07,
        0.90,
        "Qwen 1.5B · BF16 · vLLM Metal / MLX on M4 · 1,000 test boards · 30 paired game seeds per policy",
        fontsize=10,
        color=MUTED,
    )
    ax = axes[0, 0]
    stages = ["all", "early", "middle", "late"]
    y = np.arange(4)
    data = summary["test_positions"]
    ax.barh(
        y - 0.16,
        [data[s]["legal_rate"] for s in stages],
        height=0.28,
        color=BLUE,
        label="Legal move",
    )
    ax.barh(
        y + 0.16,
        [data[s]["teacher_match_rate"] for s in stages],
        height=0.28,
        color=ORANGE,
        label="Teacher match",
    )
    ax.set_yticks(y, [f"{s.title()} (n={data[s]['positions']})" for s in stages])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Held-out first decisions", loc="left", pad=12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.03, 1.27), ncol=2, fontsize=9)
    ax.grid(axis="x", color=GRID)
    ax = axes[0, 1]
    for i, agent in enumerate(AGENTS):
        values = [t["score"] for t in traces[agent]]
        color = ORANGE if agent == "grpo" else BLUE
        jitter = np.linspace(-0.18, 0.18, len(values))
        ax.scatter(values, i + jitter, color=color, alpha=0.4, s=14)
        mean = summary["games"][agent]["mean_score"]
        low, high = analysis["games"][agent]["mean_score_95_bootstrap"]
        ax.errorbar(mean, i, xerr=[[mean - low], [high - mean]], color=color, marker="D", capsize=4)
    ax.set_yticks(range(5), LABELS, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlim(left=0)
    ax.set_xticks([0, 100, 1000, 10000, 100000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Score at ending (symlog scale)")
    ax.set_title("Game outcomes: every seed shown", loc="left", pad=12)
    ax.grid(axis="x", color=GRID)
    ax.text(
        0,
        -0.31,
        "Diamonds: means; bars: 95% bootstrap intervals.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
    )
    ax = axes[1, 0]
    left = np.zeros(5)
    for key, label, color in [
        ("game_over", "Game over", BLUE),
        ("policy_failures", "Policy failure", ORANGE),
        ("move_cap", "Move cap", GOLD),
    ]:
        values = np.array([summary["games"][a][key] for a in AGENTS])
        ax.barh(range(5), values, left=left, height=0.55, label=label, color=color)
        left += values
    ax.set_yticks(range(5), LABELS, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 30)
    ax.set_xlabel("Games out of 30")
    ax.set_title("Why play ended", loc="left", pad=12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.2, -0.30), ncol=3, fontsize=8.5)
    ax = axes[1, 1]
    for values, label, color in [
        ([r["latency_seconds"] for r in records], "Test-board requests", BLUE),
        (
            [d["latency_seconds"] for t in traces["grpo"] for d in t["decisions"]],
            "Gameplay requests",
            ORANGE,
        ),
    ]:
        values = np.sort(values)
        ax.step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            color=color,
            label=label,
        )
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Local HTTP request latency (seconds)")
    ax.set_ylabel("Fraction of requests completed")
    ax.set_title("Task latency, sequential warmed server", loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="y", color=GRID)
    fig.text(
        0.07,
        0.07,
        "Stateless greedy policy; 128-token allowance. No repairs or fallback actions. Safety cap: 3,000 attempted moves.",
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.035,
        "One trained checkpoint; no SFT comparison. Teacher agreement is imitation, not optimality. Test boards come from 11 games.",
        fontsize=9,
        color=MUTED,
    )
    save(fig, "served-grpo-v1")


if __name__ == "__main__":
    style()
    figure(*analyze())
