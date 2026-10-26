"""Replay saved trajectories and summarize the completed local training runs."""

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from posttrain2048.data import read_jsonl
from posttrain2048.game import ACTIONS, Game, board, rows

ROOT = Path(__file__).resolve().parents[1]


def early_exit_category(episode):
    """Describe the saved final reply; this does not infer why generation ended."""
    assistants = [m for m in episode["completion"] if m["role"] == "assistant"]
    last = assistants[-1] if assistants else {}
    text = last.get("content", "").strip()
    if last.get("tool_calls"):
        return "structured_tool_call"
    if not text:
        return "empty_tail"
    if "<tool_call>" in text and "</tool_call>" not in text:
        return "unclosed_tool_call"
    return "other_text"


def summarize_grpo(name):
    run = ROOT / "results" / name
    metrics = [r for r in read_jsonl(run / "grpo-metrics.jsonl") if "reward" in r]
    episodes = read_jsonl(run / "grpo-episodes.jsonl")
    groups = defaultdict(list)
    for episode in episodes:
        game = Game(episode["seed"], state=board(episode["initial_board"]))
        for attempt in episode["attempts"]:
            assert rows(game.board) == attempt["before"]
            if attempt["action"] in ACTIONS:
                step = game.step(attempt["action"])
                assert step.valid == attempt["valid"]
                assert step.score_gain == attempt["score_gain"]
                spawn = list(step.spawned) if step.spawned is not None else None
                assert spawn == attempt["spawned"]
            else:
                assert not attempt["valid"] and attempt["score_gain"] == 0
            assert rows(game.board) == attempt["board"]
            assert game.score == attempt["score"]
        assert game.score == episode["score_gain"] and game.done == episode["terminal"]
        invalid = sum(not a["valid"] for a in episode["attempts"])
        invalid += episode["tool_errors"] + int(episode["early_exit"])
        assert invalid == episode["invalid_attempts"]
        expected = math.log1p(game.score) - 0.1 * invalid
        assert math.isclose(expected, episode["reward"], abs_tol=1e-12)
        assert len(episode["attempts"]) <= episode["horizon"] == 16
        groups[episode["update_before_generation"] + 1].append(episode)
    assert len(metrics) == len(groups) == 100
    assert all(len(group) == 4 for group in groups.values())
    for metric in metrics:
        mean = statistics.mean(e["reward"] for e in groups[metric["step"]])
        assert math.isclose(mean, metric["reward"], abs_tol=1e-5)
        assert math.isfinite(metric["grad_norm"])
    summary = json.loads((run / "grpo-summary.json").read_text())
    assert summary["completed_steps"] == 100 and summary["reload_predictions_exact"]
    assert summary["adapter_tensor_roundtrip_exact"]
    assert all(g["finite"] for g in summary["gradients"])
    masks = read_jsonl(run / "grpo-mask-audit.jsonl")
    assert len(masks) == 100
    early = [e for e in episodes if e["early_exit"]]
    early_advantages = []
    for mask in masks:
        group = groups[mask["step"] + 1]
        group_mean = statistics.mean(e["reward"] for e in group)
        for episode, audit in zip(group, mask["rollouts"], strict=True):
            assert episode["example_id"] == audit["example_id"]
            advantage = audit["advantage"]
            assert math.isfinite(advantage)
            assert (advantage > 0) == (episode["reward"] > group_mean)
            assert (advantage < 0) == (episode["reward"] < group_mean)
            if episode["early_exit"]:
                early_advantages.append(advantage)
    assert len(early_advantages) == len(early)
    early_blocks = []
    for start in range(0, 100, 20):
        block = [e for e in episodes if start <= e["update_before_generation"] < start + 20]
        early_blocks.append(
            {
                "first_update": start + 1,
                "last_update": start + 20,
                "rollouts": len(block),
                "early_exits": sum(e["early_exit"] for e in block),
            }
        )
    winners = [
        e
        for group in groups.values()
        for e in group
        if e["reward"] == max(r["reward"] for r in group)
    ]
    result = {
        "updates": len(metrics),
        "rollouts": len(episodes),
        "replayed_attempts": sum(len(e["attempts"]) for e in episodes),
        "mean_attempts": statistics.mean(len(e["attempts"]) for e in episodes),
        "early_exits": sum(e["early_exit"] for e in episodes),
        "invalid_attempts": sum(e["invalid_attempts"] for e in episodes),
        "tool_errors": sum(e["tool_errors"] for e in episodes),
        "terminal_rollouts": sum(e["terminal"] for e in episodes),
        "reward_first20": statistics.mean(m["reward"] for m in metrics[:20]),
        "reward_last20": statistics.mean(m["reward"] for m in metrics[-20:]),
        "attempts_first20": statistics.mean(len(e["attempts"]) for e in episodes[:80]),
        "attempts_last20": statistics.mean(len(e["attempts"]) for e in episodes[-80:]),
        "tied_reward_groups": sum(m["frac_reward_zero_std"] == 1 for m in metrics),
        "zero_gradient_updates": sum(m["grad_norm"] == 0 for m in metrics),
        "training_seconds": summary["training_seconds"],
        "max_adapter_change": summary["max_adapter_weight_change"],
        "validation_legal": sum(p["legal"] for p in summary["after_validation"]["predictions"]),
        "validation_teacher_matches": sum(
            p["teacher_match"] for p in summary["after_validation"]["predictions"]
        ),
        "reward_audit": {
            "outcomes": {
                "early_exit": len(early),
                "game_over": sum(e["terminal"] for e in episodes),
                "nonterminal_horizon_complete": sum(
                    not e["terminal"] and not e["early_exit"] for e in episodes
                ),
            },
            "early_exit_mean_attempts": statistics.mean(len(e["attempts"]) for e in early),
            "early_exits_by_update_block": early_blocks,
            "early_exit_final_reply_categories": dict(
                sorted(Counter(early_exit_category(e) for e in early).items())
            ),
            "early_exit_recorded_advantages": {
                "positive": sum(a > 0 for a in early_advantages),
                "zero": sum(a == 0 for a in early_advantages),
                "negative": sum(a < 0 for a in early_advantages),
            },
            "early_exits_with_positive_reward": sum(e["reward"] > 0 for e in early),
            "early_exit_group_winners_including_ties": sum(e["early_exit"] for e in winners),
            "group_winners_including_ties": len(winners),
            "post_terminal_calls": sum(e["post_terminal_calls"] for e in episodes),
            "unexecuted_tail_calls": sum(e["unexecuted_tail_calls"] for e in episodes),
            "interpretation_limit": (
                "Reply categories do not establish intentional stopping or its cause. "
                "Positive absolute rewards are not the same as positive GRPO advantages."
            ),
        },
    }
    return result


def main():
    sft = json.loads((ROOT / "results/local-mps-full-v1/sft-summary.json").read_text())
    training = read_jsonl(ROOT / "data/demonstrations/train.jsonl")
    assert len(sft["training_ids"]) == len(set(sft["training_ids"])) == 8000
    assert set(sft["training_ids"]) == {r["id"] for r in training}
    assert sft["completed_steps"] == 2000 and sft["batch_size"] == 4
    assert sft["reload_predictions_exact"] and sft["adapter_tensor_roundtrip_exact"]
    assert all(g["finite"] for g in sft["gradients"])
    metrics = [
        r for r in read_jsonl(ROOT / "results/local-mps-full-v1/sft-metrics.jsonl") if "loss" in r
    ]
    assert len(metrics) == 2000 and metrics[-1]["epoch"] == 1
    result = {
        "sft": {
            "examples": 8000,
            "updates": 2000,
            "training_seconds": sft["training_seconds"],
            "loss_first100": statistics.mean(r["loss"] for r in metrics[:100]),
            "loss_last100": statistics.mean(r["loss"] for r in metrics[-100:]),
            "validation_legal": 12 * sft["after_validation"]["legal_rate"],
            "validation_teacher_matches": 12 * sft["after_validation"]["teacher_agreement"],
        },
        "warmup_grpo": summarize_grpo("local-mps-grpo-now"),
        "full_sft_grpo": summarize_grpo("local-mps-full-v1"),
        "evaluation_limit": "12 fixed one-move validation boards; no complete-game evaluation",
    }
    path = ROOT / "results/local-mps-full-v1/analysis.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
