# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.11.2", "numpy==2.4.6"]
# ///
"""Validate and compare original-instruct, SFT and GRPO served policies."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_served_evaluation import interval, wilson
from matplotlib.ticker import FuncFormatter, PercentFormatter
from plot_experiments import BLUE, GOLD, GRID, MUTED, ORANGE, plt, save, style

from posttrain2048.cli import replay
from posttrain2048.data import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/checkpoint-comparison-v1"
POLICIES = ["base", "sft", "grpo"]
LABELS = {"base": "Original instruct", "sft": "SFT", "grpo": "SFT + GRPO"}
BASELINES = ["random", "priority", "greedy", "teacher"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze():
    reference = json.loads((ROOT / "results/served-grpo-v1/manifest.json").read_text())
    baseline_analysis = json.loads((ROOT / "results/served-grpo-v1/analysis.json").read_text())
    source = read_jsonl(ROOT / "data/demonstrations/test.jsonl")
    assert len(source) == 1000
    model_records = {}
    model_traces = {}
    summaries = {}
    manifests = {}
    replayed = 0
    attempts = 0
    for policy in POLICIES:
        directory = ROOT / "results" / f"served-{policy}-v1"
        manifest = json.loads((directory / "manifest.json").read_text())
        manifests[policy] = manifest
        for key in (
            "backend",
            "decoding",
            "system",
            "tools",
            "tool_choice",
            "test_file_sha256",
            "test_positions",
            "seen_probe_ids",
            "game_seeds",
            "max_game_attempts",
            "gameplay_context",
            "failure_rule",
            "baseline_agents",
            "teacher_depth",
        ):
            assert manifest[key] == reference[key], f"Changed contract: {policy}/{key}"
        for name in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
            assert (
                manifest["checkpoint_manifest"][name]["sha256"]
                == reference["checkpoint_manifest"][name]["sha256"]
            )
        records = read_jsonl(directory / "test-positions.jsonl")
        assert len(records) == 1000 and len({r["id"] for r in records}) == 1000
        for original, record in zip(source, records, strict=True):
            assert original["id"] == record["id"]
            assert original["board_id"] == record["board_id"]
            assert original["teacher_move"] == record["teacher_move"]
            assert record["legal"] == (record["decision"]["error"] is None)
            assert record["teacher_match"] == (
                record["legal"] and record["decision"]["direction"] == original["teacher_move"]
            )
            assert (
                max(record["teacher_scores"], key=record["teacher_scores"].get)
                == original["teacher_move"]
            )
        model_records[policy] = records
        traces = []
        for seed in reference["game_seeds"]:
            trace = json.loads((directory / "games" / f"{policy}-{seed}.json").read_text())
            replay(trace)
            assert trace["seed"] == seed and trace["max_steps"] == 3000
            assert trace["requests"] == len(trace["decisions"])
            failures = [d for d in trace["decisions"] if d["decision"]["error"]]
            assert len(failures) == int(bool(trace["policy_failure"]))
            if failures:
                assert trace["decisions"][-1] == failures[-1]
                assert trace["ending"] == "policy_failure" and not trace["terminated"]
            else:
                assert trace["terminated"] or trace["attempts"] == 3000
            traces.append(trace)
            replayed += 1
            attempts += trace["attempts"]
        model_traces[policy] = traces
        summaries[policy] = json.loads((directory / "summary.json").read_text())
    for agent in BASELINES:
        for seed in reference["game_seeds"]:
            trace = json.loads(
                (ROOT / "results/served-grpo-v1/games" / f"{agent}-{seed}.json").read_text()
            )
            replay(trace)
            replayed += 1
            attempts += trace["attempts"]
    rng = np.random.default_rng(20261026)
    indices = rng.integers(0, 30, size=(10000, 30))
    score_arrays = {p: np.array([t["score"] for t in model_traces[p]]) for p in POLICIES}
    grouped_ids = defaultdict(list)
    for index, row in enumerate(source):
        grouped_ids[row["game_id"]].append(index)
    assert len(grouped_ids) == 11
    sizes = np.array([len(ids) for ids in grouped_ids.values()])
    correct = {
        p: np.array(
            [sum(model_records[p][i]["teacher_match"] for i in ids) for ids in grouped_ids.values()]
        )
        for p in POLICIES
    }
    cluster_indices = rng.integers(0, 11, size=(10000, 11))
    models = {}
    for policy in POLICIES:
        summary = summaries[policy]
        records = model_records[policy]
        models[policy] = {
            "label": LABELS[policy],
            "games": summary["games"][policy],
            "test_positions": summary["test_positions"],
            "seen_probe": summary["seen_probe"],
            "position_latency_seconds": summary["position_latency_seconds"],
            "game_latency_seconds": summary["game_latency_seconds"],
            "mean_score_95_bootstrap": interval(score_arrays[policy][indices].mean(axis=1)),
            "teacher_match_95_source_game_bootstrap": interval(
                correct[policy][cluster_indices].sum(axis=1) / sizes[cluster_indices].sum(axis=1)
            ),
            "reached_2048_95_wilson": wilson(summary["games"][policy]["reached_2048"], 30),
            "mean_position_completion_tokens": float(
                np.mean([r["usage"]["completion_tokens"] for r in records])
            ),
            "position_length_finishes": sum(
                r["decision"]["finish_reason"] == "length" for r in records
            ),
            "checkpoint": manifests[policy]["checkpoint"],
            "run": f"results/served-{policy}-v1",
        }
    contrasts = {}
    for after, before in (("sft", "base"), ("grpo", "sft"), ("grpo", "base")):
        difference = score_arrays[after] - score_arrays[before]
        point_difference = correct[after] - correct[before]
        contrasts[f"{after}_minus_{before}"] = {
            "before": before,
            "after": after,
            "mean_score_difference": float(difference.mean()),
            "mean_score_difference_95_paired_bootstrap": interval(difference[indices].mean(axis=1)),
            "seeds_with_higher_score": int(np.sum(difference > 0)),
            "seeds_tied_score": int(np.sum(difference == 0)),
            "seeds_with_lower_score": int(np.sum(difference < 0)),
            "teacher_match_rate_difference": float(point_difference.sum() / 1000),
            "teacher_match_rate_difference_95_source_game_bootstrap": interval(
                point_difference[cluster_indices].sum(axis=1) / sizes[cluster_indices].sum(axis=1)
            ),
        }
    OUT.mkdir(exist_ok=True)
    analysis = {
        "models": models,
        "contrasts": contrasts,
        "reference_run": "results/served-grpo-v1",
        "baseline_games": summaries["grpo"]["games"],
        "baseline_score_intervals": baseline_analysis["games"],
        "checks": {
            "matched_test_positions_per_checkpoint": 1000,
            "matched_game_seeds": reference["game_seeds"],
            "all_protocol_and_tokenizer_checks_passed": True,
            "game_traces_replayed": replayed,
            "game_attempts_replayed": attempts,
        },
        "uncertainty_method": "10,000 shared bootstrap resamples: 30 paired game seeds for scores; 11 paired source-game clusters for teacher agreement. Conditional on one training seed and these exported checkpoints.",
        "limits": [
            "Original means the pinned vendor instruct model before our SFT, not a raw pretrained model.",
            "Scores include policy failures and move caps; neither is natural game over.",
            "Differences measure this model-plus-parser agent; automatic Hermes parsing is shared.",
            "The fixed test set is reused for checkpoint comparison without tuning any checkpoint or decoding.",
            "A performance regression does not by itself establish reward hacking.",
        ],
    }
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    return analysis, model_traces


def figure(analysis, traces):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    fig.subplots_adjust(left=0.20, right=0.95, top=0.79, bottom=0.15, wspace=0.56, hspace=0.80)
    fig.text(0.07, 0.95, "What did SFT and GRPO actually change?", fontsize=21, weight="bold")
    fig.text(
        0.07,
        0.90,
        "Same Qwen revision · BF16 / vLLM Metal · same 1,000 boards · same 30 seeds · same prompts, parser and greedy decoding",
        fontsize=9.8,
        color=MUTED,
    )
    models = analysis["models"]
    ax = axes[0, 0]
    y = np.arange(3)
    ax.barh(
        y - 0.16,
        [models[p]["test_positions"]["all"]["legal_rate"] for p in POLICIES],
        height=0.28,
        color=BLUE,
        label="Legal API call",
    )
    ax.barh(
        y + 0.16,
        [models[p]["test_positions"]["all"]["teacher_match_rate"] for p in POLICIES],
        height=0.28,
        color=ORANGE,
        label="Teacher match",
    )
    ax.set_yticks(y, [LABELS[p] for p in POLICIES])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Held-out first decisions", loc="left", pad=12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.03, 1.27), ncol=2, fontsize=9)
    ax.grid(axis="x", color=GRID)
    ax = axes[0, 1]
    agents = [*POLICIES, *BASELINES]
    names = [
        LABELS.get(
            p,
            {
                "random": "Random legal",
                "priority": "Direction priority",
                "greedy": "Immediate-score greedy",
                "teacher": "Depth-2 teacher",
            }.get(p),
        )
        for p in agents
    ]
    for i, policy in enumerate(agents):
        if policy in POLICIES:
            values = [t["score"] for t in traces[policy]]
            summary = models[policy]["games"]
            bounds = models[policy]["mean_score_95_bootstrap"]
        else:
            values = [
                json.loads(
                    (ROOT / "results/served-grpo-v1/games" / f"{policy}-{seed}.json").read_text()
                )["score"]
                for seed in analysis["checks"]["matched_game_seeds"]
            ]
            summary = analysis["baseline_games"][policy]
            bounds = analysis["baseline_score_intervals"][policy]["mean_score_95_bootstrap"]
        color = ORANGE if policy == "grpo" else BLUE if policy == "sft" else MUTED
        ax.scatter(values, i + np.linspace(-0.15, 0.15, 30), color=color, alpha=0.35, s=12)
        mean = summary["mean_score"]
        ax.errorbar(
            mean,
            i,
            xerr=[[mean - bounds[0]], [bounds[1] - mean]],
            marker="D",
            capsize=3,
            color=color,
        )
    ax.set_yticks(range(7), names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlim(left=0)
    ax.set_xticks([0, 100, 1000, 10000, 100000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Score at ending (symlog scale)")
    ax.set_title("Every game seed; means with 95% intervals", loc="left", pad=12)
    ax.grid(axis="x", color=GRID)
    ax = axes[1, 0]
    for i, key in enumerate(("sft_minus_base", "grpo_minus_sft", "grpo_minus_base")):
        row = analysis["contrasts"][key]
        mean = row["mean_score_difference"]
        low, high = row["mean_score_difference_95_paired_bootstrap"]
        ax.errorbar(
            mean,
            i,
            xerr=[[mean - low], [high - mean]],
            color=BLUE if mean >= 0 else ORANGE,
            marker="o",
            capsize=5,
        )
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(3), ["Original → SFT", "SFT → GRPO", "Original → GRPO"])
    ax.invert_yaxis()
    ax.set_title("Stage effects on these 30 seeds", loc="left", pad=12)
    ax.set_xlabel("Mean score difference (paired 95% bootstrap)", fontsize=9)
    ax.grid(axis="x", color=GRID)
    ax = axes[1, 1]
    left = np.zeros(3)
    for key, label, color in (
        ("game_over", "Game over", BLUE),
        ("policy_failures", "Policy failure", ORANGE),
        ("move_cap", "Move cap", GOLD),
    ):
        values = np.array([models[p]["games"][key] for p in POLICIES])
        ax.barh(range(3), values, left=left, height=0.55, label=label, color=color)
        left += values
    ax.set_yticks(range(3), [LABELS[p] for p in POLICIES])
    ax.invert_yaxis()
    ax.set_xlim(0, 30)
    ax.set_xlabel("Games out of 30")
    ax.set_title("Why each policy stopped", loc="left", pad=12)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.03, 1.30),
        ncol=3,
        fontsize=8,
        handlelength=1,
        columnspacing=0.8,
    )
    fig.text(
        0.07,
        0.07,
        "Automatic Hermes parsing; no client-side repairs or fallbacks. Stateless board prompts; 128-token allowance; 3,000-move cap.",
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.035,
        "Intervals reflect evaluation seeds/source games, not repeated training. A regression alone does not establish reward hacking.",
        fontsize=9,
        color=MUTED,
    )
    save(fig, "checkpoint-comparison-v1")


if __name__ == "__main__":
    style()
    figure(*analyze())
