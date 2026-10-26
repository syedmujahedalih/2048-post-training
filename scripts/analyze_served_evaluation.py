# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.11.2", "numpy==2.4.6"]
# ///
"""Replay the served-model benchmark and plot its saved measurements."""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter
from plot_experiments import BLUE, GOLD, GRID, MUTED, ORANGE, plt, save, style

from posttrain2048.cli import replay
from posttrain2048.data import read_jsonl
from posttrain2048.game import board, legal_moves
from posttrain2048.serving_eval import parse_move_response

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


def training_parser_audit():
    # Exact non-streaming extraction regex in the installed vLLM 0.32.0 Hermes parser.
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
    accepted = []
    unclosed = 0
    early_count = 0
    for episode in read_jsonl(ROOT / "results/local-mps-full-v1/grpo-episodes.jsonl"):
        if not episode["early_exit"]:
            continue
        early_count += 1
        last = [m for m in episode["completion"] if m["role"] == "assistant"][-1]
        text = last.get("content", "")
        unclosed += int("<tool_call>" in text and "</tool_call>" not in text)
        try:
            calls = [json.loads(a or b) for a, b in pattern.findall(text)]
            normalized = [
                {
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
                }
                for c in calls
            ]
            final_board = (
                episode["attempts"][-1]["board"]
                if episode["attempts"]
                else episode["initial_board"]
            )
            parsed = parse_move_response(
                {"choices": [{"message": {"tool_calls": normalized}}]},
                legal_moves(board(final_board)),
            )
            if parsed["error"] is None:
                accepted.append(
                    {
                        "update": episode["update_before_generation"] + 1,
                        "example_id": episode["example_id"],
                        "direction": parsed["direction"],
                        "unclosed_wrapper": "</tool_call>" not in text,
                    }
                )
        except (KeyError, TypeError, ValueError):
            continue
    return {
        "reference_parser": "vLLM 0.32.0 Hermes2ProToolParser.extract_tool_calls, non-streaming",
        "extraction_regex": pattern.pattern,
        "training_early_exits": early_count,
        "unclosed_final_replies": unclosed,
        "static_legal_move_parses": len(accepted),
        "accepted": accepted,
        "limits": (
            "Static parsing of saved training reply text, not a new model rollout. "
            "Does not show these counterfactual trajectories would finish or score better. "
            "Served API calls do not expose raw closing-tag correctness."
        ),
    }


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
        "uniform_legal_expected_teacher_match": {
            stage: float(
                np.mean(
                    [
                        1 / len(r["legal_moves"])
                        for r in source
                        if stage == "all" or r["stage"] == stage
                    ]
                )
            )
            for stage in ("all", "early", "middle", "late")
        },
        "games": games,
        "training_tail_parser_audit": training_parser_audit(),
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
        y - 0.24,
        [data[s]["legal_rate"] for s in stages],
        height=0.20,
        color=BLUE,
        label="Legal",
    )
    ax.barh(
        y,
        [data[s]["teacher_match_rate"] for s in stages],
        height=0.20,
        color=ORANGE,
        label="Teacher match",
    )
    ax.barh(
        y + 0.24,
        [analysis["uniform_legal_expected_teacher_match"][s] for s in stages],
        height=0.20,
        color=GOLD,
        label="Random expected",
    )
    ax.set_yticks(y, [f"{s.title()} (n={data[s]['positions']})" for s in stages])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Held-out first decisions", loc="left", pad=12)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.03, 1.27),
        ncol=3,
        fontsize=8,
        handlelength=1,
        columnspacing=0.8,
    )
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
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.05, 1.31),
        ncol=3,
        fontsize=8.5,
        handlelength=1.2,
        columnspacing=1.1,
    )
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
        "Stateless greedy policy; 128-token allowance; Hermes parsing. No client-side repair or fallback. Cap: 3,000 moves.",
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


def replay_figure(traces):
    ordered = sorted(traces, key=lambda t: (t["score"], t["seed"]))
    trace = ordered[(len(ordered) - 1) // 2]
    frames = np.linspace(0, len(trace["steps"]), 6).round().astype(int)
    palette = {
        0: "#E7E3DD",
        2: "#EEE4DA",
        4: "#EDE0C8",
        8: "#F2B179",
        16: "#F59563",
        32: "#F67C5F",
        64: "#F65E3B",
        128: "#EDCF72",
        256: "#EDCC61",
        512: "#EDC850",
        1024: "#EDC53F",
        2048: "#EDC22E",
    }
    fig, axes = plt.subplots(2, 3, figsize=(10, 8.8))
    fig.subplots_adjust(left=0.07, right=0.95, top=0.81, bottom=0.12, hspace=0.37, wspace=0.21)
    fig.text(0.07, 0.95, "Actual gameplay: the median-scoring GRPO run", fontsize=18, weight="bold")
    fig.text(
        0.07,
        0.89,
        f"Seed {trace['seed']} · final recorded score {trace['score']:,} · highest tile {trace['max_tile']} · {trace['ending'].replace('_', ' ')}",
        fontsize=11,
        color=MUTED,
    )
    for ax, frame in zip(axes.flat, frames, strict=True):
        state = trace["initial_board"] if frame == 0 else trace["steps"][frame - 1]["board"]
        score = 0 if frame == 0 else trace["steps"][frame - 1]["score"]
        for row, values in enumerate(state):
            for column, value in enumerate(values):
                ax.add_patch(
                    Rectangle(
                        (column, 3 - row),
                        0.94,
                        0.94,
                        color=palette.get(value, GOLD),
                    )
                )
                if value:
                    ax.text(
                        column + 0.47,
                        3.47 - row,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=18,
                        weight="bold",
                        color="#172A3A",
                    )
        ax.set(xlim=(-0.04, 4), ylim=(-0.04, 4), aspect="equal")
        ax.axis("off")
        ax.set_title(f"Move {frame} · score {score:,}", fontsize=11, loc="left", pad=10)
    fig.text(
        0.07,
        0.06,
        "Selection: lower median of 30 scores at ending. All intermediate actions, spawns and scores replayed exactly.",
        fontsize=9,
        color=MUTED,
    )
    save(fig, "served-grpo-median-game")


if __name__ == "__main__":
    style()
    results = analyze()
    figure(*results)
    replay_figure(results[-1]["grpo"])
