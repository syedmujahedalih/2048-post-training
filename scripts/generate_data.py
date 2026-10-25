"""Generate reproducible, teacher-labeled reachable boards without model inference."""

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from posttrain2048.agents import PriorityAgent, RandomAgent
from posttrain2048.data import canonical_key, example, validate_dataset
from posttrain2048.game import Game
from posttrain2048.teacher import Teacher


def stage(state):
    maximum = max(state)
    return "early" if maximum <= 64 else "middle" if maximum <= 512 else "late"


def generate(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"{name}.jsonl").exists() for name in ("train", "validation", "test")):
        raise SystemExit("Output already contains data; choose a new directory")
    teacher = Teacher(args.depth)
    seen = set()
    started = time.monotonic()
    deadline = started + args.max_seconds
    game_summaries = []
    for split_index, (split, size) in enumerate(
        zip(("train", "validation", "test"), args.sizes, strict=True)
    ):
        count = 0
        game_index = 0
        with (output / f"{split}.jsonl").open("w") as handle:
            while count < size and time.monotonic() < deadline:
                seed = args.seed + split_index * 100000 + game_index
                policy_name = ("random", "priority", "teacher", "teacher", "teacher")[
                    game_index % 5
                ]
                if policy_name == "random":
                    agent = RandomAgent(seed + 700000)
                elif policy_name == "priority":
                    agent = PriorityAgent()
                else:
                    agent = teacher
                game = Game(seed)
                candidates = defaultdict(list)
                for _ in range(args.max_steps):
                    if game.done or time.monotonic() >= deadline:
                        break
                    candidates[stage(game.board)].append((game.valid_steps, game.score, game.board))
                    game.step(agent.choose(game.board))
                sample_rng = random.Random(seed + 900000)
                for items in candidates.values():
                    sample_rng.shuffle(items)
                selected = []
                # Round-robin stage sampling prevents long late-game runs dominating every game.
                while len(selected) < args.per_game and any(candidates.values()):
                    for name in ("early", "middle", "late"):
                        if candidates[name] and len(selected) < args.per_game:
                            selected.append(candidates[name].pop())
                game_id = f"{split}-{seed}"
                for step_index, score, state in selected:
                    key = canonical_key(state)
                    if key in seen:
                        continue
                    action = teacher.choose(state)
                    entry = example(
                        state,
                        action,
                        {
                            "id": f"{game_id}-{step_index}",
                            "game_id": game_id,
                            "seed": seed,
                            "split": split,
                            "step": step_index,
                            "score": score,
                            "stage": stage(state),
                            "behavior_policy": policy_name,
                            "teacher": "nneonneo-expectimax",
                            "teacher_depth": args.depth,
                        },
                    )
                    handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
                    count += 1
                    seen.add(key)
                    if count == size:
                        break
                game_summaries.append(
                    {
                        "game_id": game_id,
                        "seed": seed,
                        "policy": policy_name,
                        "score": game.score,
                        "max_tile": max(game.board),
                        "moves": game.valid_steps,
                        "terminated": game.done,
                    }
                )
                game_index += 1
                if game_index % 10 == 0 or count == size:
                    print(
                        f"{split}: {count}/{size}, {game_index} games, {time.monotonic() - started:.1f}s",
                        flush=True,
                    )
    (output / "games.jsonl").write_text("".join(json.dumps(row) + "\n" for row in game_summaries))
    report = validate_dataset(output)
    report.update(
        {
            "requested_sizes": args.sizes,
            "depth": args.depth,
            "seed": args.seed,
            "per_game": args.per_game,
            "max_steps": args.max_steps,
            "elapsed_seconds": time.monotonic() - started,
            "teacher_revision": (Path("vendor/2048-ai/UPSTREAM_COMMIT").read_text().strip()),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    report["complete"] = [
        report["splits"][split]["examples"] for split in ("train", "validation", "test")
    ] == args.sizes
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["complete"]:
        raise SystemExit("Time limit reached; partial data retained and marked incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demonstrations"))
    parser.add_argument("--sizes", type=int, nargs=3, default=[8000, 1000, 1000])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--per-game", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--max-seconds", type=int, default=3600)
    generate(parser.parse_args())
