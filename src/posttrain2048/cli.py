"""Play, record, and verify complete seeded games."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .agents import GreedyAgent, PriorityAgent, RandomAgent
from .game import Game, format_board, rows
from .teacher import Teacher


def make_agent(name: str, seed: int = 0, depth: int = 2):
    return {
        "random": lambda: RandomAgent(seed),
        "priority": PriorityAgent,
        "greedy": GreedyAgent,
        "teacher": lambda: Teacher(depth),
    }[name]()


def play(agent, seed: int, max_steps: int = 3000) -> dict:
    game = Game(seed)
    initial = rows(game.board)
    steps = []
    for _ in range(max_steps):
        if game.done:
            break
        result = game.step(agent.choose(game.board))
        steps.append(
            {
                **asdict(result),
                "board": rows(result.board),
                "spawned": list(result.spawned) if result.spawned else None,
                "score": game.score,
            }
        )
    return {
        "seed": seed,
        "initial_board": initial,
        "steps": steps,
        "score": game.score,
        "max_tile": max(game.board),
        "reached_2048": max(game.board) >= 2048,
        "attempts": game.attempts,
        "invalid_attempts": game.attempts - game.valid_steps,
        "terminated": game.done,
        "truncated": not game.done,
        "max_steps": max_steps,
    }


def replay(trace: dict, verbose: bool = False) -> None:
    game = Game(trace["seed"])
    assert rows(game.board) == trace["initial_board"], "Initial board mismatch"
    for index, expected in enumerate(trace["steps"]):
        step = game.step(expected["action"])
        actual = json.loads(
            json.dumps({**asdict(step), "board": rows(step.board), "score": game.score})
        )
        assert actual == expected, f"Replay mismatch at decision {index}"
        if verbose:
            print(f"\n{index + 1}: {step.action}, score={game.score}")
            print(format_board(game.board))
    assert game.score == trace["score"]
    assert max(game.board) == trace["max_tile"]
    assert game.done == trace["terminated"]
    assert game.attempts == trace["attempts"]
    assert game.attempts - game.valid_steps == trace["invalid_attempts"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    playing = commands.add_parser("play")
    playing.add_argument(
        "--agent", choices=["random", "priority", "greedy", "teacher"], default="random"
    )
    playing.add_argument("--seed", type=int, default=0)
    playing.add_argument("--max-steps", type=int, default=3000)
    playing.add_argument("--depth", type=int, default=2)
    playing.add_argument("--output", type=Path)
    replaying = commands.add_parser("replay")
    replaying.add_argument("path", type=Path)
    replaying.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.command == "replay":
        replay(json.loads(args.path.read_text()), args.verbose)
        print("Replay verified")
        return
    trace = play(make_agent(args.agent, args.seed, args.depth), args.seed, args.max_steps)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(trace, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in trace.items() if key not in {"steps", "initial_board"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
