"""Measure real local Decider predictions and paired seeded gameplay."""

import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path

from posttrain2048.cli import make_agent, play, replay
from posttrain2048.data import read_jsonl
from posttrain2048.decider import DeciderAgent
from posttrain2048.game import board, legal_moves
from posttrain2048.teacher import Teacher


def summarize(games):
    scores = [game["score"] for game in games]
    return {
        "games": len(games),
        "mean_score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "scores": scores,
        "max_tiles": [game["max_tile"] for game in games],
        "reached_2048": sum(game["reached_2048"] for game in games),
        "invalid_attempts": sum(game["invalid_attempts"] for game in games),
        "truncated": sum(game["truncated"] for game in games),
    }


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    agent = DeciderAgent(args.endpoint, args.mode)
    health = agent.health()
    teacher = Teacher(args.depth)
    candidates = read_jsonl(args.data / "validation.jsonl")
    random.Random(20261025).shuffle(candidates)
    selected = candidates[: args.positions]
    report = {
        "mode": args.mode,
        "health": health,
        "teacher_depth": args.depth,
        "hardware": {
            "machine": platform.machine(),
            "system": platform.system(),
            "chip": "Apple M4",
            "unified_memory_gb": 24,
        },
        "position_count": len(selected),
        "diagnostic_legal_mask": False,
        "gameplay_legal_mask": True,
        "max_steps": args.max_steps,
        "game_seed_start": args.seed,
    }
    records = []
    predictions_file = args.output / "predictions.jsonl"
    with predictions_file.open("w") as handle:
        for index, entry in enumerate(selected):
            state = board(entry["board"])
            decision = agent.decide(state, legal_only=False)
            record = {
                "board_id": entry["board_id"],
                "game_id": entry["game_id"],
                "stage": entry["stage"],
                "teacher_move": entry["teacher_move"],
                "legal_moves": list(legal_moves(state)),
                "decision": decision,
                "legal": decision["action"] in legal_moves(state),
                "teacher_agreement": decision["action"] == entry["teacher_move"],
            }
            scores = teacher.scores(state)
            record["teacher_scores"] = scores
            record["teacher_relative_regret"] = (
                (
                    (max(scores.values()) - scores[decision["action"]])
                    / max(abs(max(scores.values())), 1)
                )
                if record["legal"]
                else None
            )
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if (index + 1) % 16 == 0:
                print(f"Positions: {index + 1}/{len(selected)}", flush=True)
    latencies = [row["decision"]["elapsed_ms"] for row in records]
    report["positions"] = {
        "legal_rate": statistics.mean(row["legal"] for row in records),
        "teacher_agreement": statistics.mean(row["teacher_agreement"] for row in records),
        "first_request_ms": latencies[0],
        "warm_median_ms": statistics.median(latencies[1:]),
        "warm_p95_ms": sorted(latencies[1:])[int(0.95 * (len(latencies) - 2))],
        "teacher_relative_regret_mean": statistics.mean(
            row["teacher_relative_regret"] for row in records if row["legal"]
        ),
    }
    (args.output / "position_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    games = {}
    policies = ["random", "priority", "greedy", "teacher", "decider"] if args.games else []
    for name in policies:
        games[name] = []
        for index in range(args.games):
            seed = args.seed + index
            policy = agent if name == "decider" else make_agent(name, seed, args.depth)
            started = time.monotonic()
            trace = play(policy, seed, args.max_steps)
            replay(trace)
            trace["wall_seconds"] = time.monotonic() - started
            games[name].append(trace)
            path = args.output / "traces" / f"{name}-{seed}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(trace, separators=(",", ":")) + "\n")
            print(
                f"Game {name} seed={seed}: score={trace['score']} tile={trace['max_tile']} steps={trace['attempts']} seconds={trace['wall_seconds']:.2f}",
                flush=True,
            )
    report["gameplay"] = {name: summarize(traces) for name, traces in games.items()}
    if args.games:
        report["paired_score_differences_decider_minus_priority"] = [
            a["score"] - b["score"]
            for a, b in zip(games["decider"], games["priority"], strict=True)
        ]
    report["notes"] = [
        "Exploratory local inference benchmark, not an RL training result.",
        "Position diagnostics supply all four actions; gameplay supplies only legal actions to every agent.",
        "Afterstate mode exposes deterministic slides and immediate score gains, not future random spawns or teacher scores.",
        "Teacher agreement measures imitation, not long-term optimality.",
        "Small game count does not establish a statistically reliable adviser benefit.",
    ]
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/demonstrations"))
    parser.add_argument("--output", type=Path, default=Path("results/decider-afterstates"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8099")
    parser.add_argument("--mode", choices=["directions", "afterstates"], default="afterstates")
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--seed", type=int, default=700000)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=3000)
    main(parser.parse_args())
