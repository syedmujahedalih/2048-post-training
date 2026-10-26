"""Evaluate a fixed served checkpoint on held-out positions and new seeded games.

Runtime endpoints and complete HTTP responses remain in ignored local records.
Public records contain simulator evidence and inference settings, not ports.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import statistics
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from posttrain2048.cli import make_agent, play, replay
from posttrain2048.data import SYSTEM, TOOLS, observation, read_jsonl
from posttrain2048.game import Game, board, legal_moves, rows
from posttrain2048.serving_eval import parse_move_response
from posttrain2048.teacher import Teacher

ROOT = Path(__file__).resolve().parents[1]
DECODING = {
    "temperature": 0,
    "top_p": 1,
    "top_k": -1,
    "repetition_penalty": 1,
    "max_tokens": 128,
    "seed": 2048,
    "stream": False,
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize_positions(records):
    groups = {"all": records}
    for stage in ("early", "middle", "late"):
        groups[stage] = [r for r in records if r["stage"] == stage]
    result = {}
    for name, group in groups.items():
        if not group:
            continue
        result[name] = {
            "positions": len(group),
            "source_games": len({r["game_id"] for r in group}),
            "legal": sum(r["legal"] for r in group),
            "teacher_matches": sum(r["teacher_match"] for r in group),
            "legal_rate": statistics.mean(r["legal"] for r in group),
            "teacher_match_rate": statistics.mean(r["teacher_match"] for r in group),
            "errors": dict(Counter(r["decision"]["error"] for r in group if not r["legal"])),
            "latency_p50_seconds": statistics.median(r["latency_seconds"] for r in group),
            "latency_p95_seconds": quantile([r["latency_seconds"] for r in group], 0.95),
        }
    return result


def summarize_games(traces):
    return {
        "games": len(traces),
        "mean_score": statistics.mean(t["score"] for t in traces),
        "median_score": statistics.median(t["score"] for t in traces),
        "mean_executed_moves": statistics.mean(t["attempts"] for t in traces),
        "score_min": min(t["score"] for t in traces),
        "score_max": max(t["score"] for t in traces),
        "highest_tile": max(t["max_tile"] for t in traces),
        "reached_2048": sum(t["reached_2048"] for t in traces),
        "game_over": sum(t["terminated"] for t in traces),
        "policy_failures": sum(bool(t.get("policy_failure")) for t in traces),
        "move_cap": sum(t.get("ending") == "move_cap" for t in traces),
        "failure_reasons": dict(
            Counter(t["policy_failure"]["error"] for t in traces if t.get("policy_failure"))
        ),
    }


class Client:
    def __init__(self, port, model, private):
        self.url = f"http://127.0.0.1:{port}/v1"
        self.model = model
        self.private = private

    def request(self, route, payload=None):
        request = urllib.request.Request(
            self.url + route,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    def decide(self, state, unit):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": observation(state)},
            ],
            "tools": TOOLS,
            "tool_choice": "auto",
            **DECODING,
        }
        started = time.monotonic()
        response = self.request("/chat/completions", payload)
        elapsed = time.monotonic() - started
        decision = parse_move_response(response, legal_moves(state))
        append_json(
            self.private / "responses.jsonl",
            {"unit": unit, "request": payload, "response": response},
        )
        record = {
            "decision": decision,
            "latency_seconds": elapsed,
            "usage": response.get("usage"),
        }
        if decision["error"]:
            record["failed_message"] = response["choices"][0].get("message")
        return record


def served_game(client, seed, max_steps, output, private_status):
    game = Game(seed)
    initial = rows(game.board)
    steps = []
    decisions = []
    failure = None
    started = time.monotonic()
    for index in range(max_steps):
        if game.done:
            break
        result = client.decide(game.board, f"game:{seed}:decision:{index}")
        decisions.append(result)
        if result["decision"]["error"]:
            failure = result["decision"]
            if failure["error"] != "illegal_move":
                break
        step = game.step(result["decision"]["direction"])
        steps.append(
            {
                **asdict(step),
                "board": rows(step.board),
                "spawned": list(step.spawned) if step.spawned else None,
                "score": game.score,
            }
        )
        if failure:
            break
        if (index + 1) % 25 == 0:
            write_json(
                private_status,
                {"stage": "game", "seed": seed, "requests": index + 1, "score": game.score},
            )
            print(f"GRPO seed={seed}: move {index + 1}, score={game.score}", flush=True)
    ending = "game_over" if game.done else "policy_failure" if failure else "move_cap"
    trace = {
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
        "ending": ending,
        "policy_failure": failure,
        "requests": len(decisions),
        "decisions": decisions,
        "wall_seconds": time.monotonic() - started,
    }
    replay(trace)
    write_json(output, trace)
    print(
        f"GRPO seed={seed}: {ending}, score={game.score}, tile={max(game.board)}, requests={len(decisions)}",
        flush=True,
    )
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default="2048-grpo")
    parser.add_argument("--run", default="served-grpo-v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.run.replace("-", "").isalnum():
        raise ValueError("Run names must be alphanumeric or hyphenated")
    private = ROOT / ".local" / args.run
    private.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / ".local/served-evaluation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = ROOT / "results" / args.run
    if output.exists() and not args.resume:
        raise FileExistsError("Use a fresh run name or explicit --resume; results already exist")
    output.mkdir(exist_ok=True)
    (output / "games").mkdir(exist_ok=True)
    test_path = ROOT / "data/demonstrations/test.jsonl"
    training = read_jsonl(ROOT / "data/demonstrations/train.jsonl")
    grpo = json.loads((ROOT / "results/local-mps-full-v1/grpo-summary.json").read_text())
    training_ids = set(grpo["training_ids"])
    seen = [r for r in training if r["id"] in training_ids]
    test = read_jsonl(test_path)
    assert len(test) == 1000 and len(seen) == 32
    audit = json.loads((ROOT / "results/data-audit.json").read_text())
    assert digest(test_path) == audit["splits"]["test"]["sha256"]
    assert not {r["board_id"] for r in seen} & {r["board_id"] for r in test}
    seeds = list(range(800000, 800030))
    assert not set(seeds) & {
        r["seed"] for r in read_jsonl(ROOT / "data/demonstrations/games.jsonl")
    }
    conversion = json.loads(
        (ROOT / "results/local-mps-full-v1/serving-bf16-export.json").read_text()
    )
    manifest = {
        "run": args.run,
        "served_model": args.model,
        "checkpoint": conversion["output"],
        "checkpoint_manifest": conversion["output_manifest"],
        "backend": "vLLM Metal 0.32.0 / MLX GPU; BF16 weights",
        "decoding": DECODING,
        "system": SYSTEM,
        "tools": TOOLS,
        "tool_choice": "auto; no forced grammar, repair, retry or fallback",
        "test_file_sha256": digest(test_path),
        "test_positions": 1000,
        "seen_probe_ids": sorted(training_ids),
        "game_seeds": seeds,
        "max_game_attempts": 3000,
        "gameplay_context": "Fresh system+board prompt for each decision; no conversation history",
        "failure_rule": "Stop game on first malformed or illegal action; record separately from game over",
        "baseline_agents": ["random", "priority", "greedy", "teacher"],
        "teacher_depth": 2,
        "limits": [
            "One trained checkpoint and one training seed; no causal SFT-versus-GRPO comparison.",
            "Teacher match measures imitation of depth-2 expectimax, not optimality.",
            "Test positions are clustered in 11 source games, not 1,000 independent games.",
            "Stateless gameplay and 128-token budget differ from multi-turn 48-token GRPO generation.",
            "Latency includes local HTTP and prefill/decode; server already warm, single sequential client.",
            "Move-cap games and policy failures are not natural game overs.",
        ],
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Resume configuration differs"
    else:
        write_json(manifest_path, manifest)
    client = Client(args.port, args.model, private)
    models = client.request("/models")["data"]
    model = next(m for m in models if m["id"] == args.model)
    assert Path(model["root"]).resolve() == (ROOT / conversion["output"]).resolve()
    teacher = Teacher(2)
    status = private / "status.json"
    for split, entries in (("seen-probe", seen), ("test", test)):
        path = output / f"{split}-positions.jsonl"
        existing = read_jsonl(path) if path.exists() else []
        assert [r["id"] for r in existing] == [r["id"] for r in entries[: len(existing)]]
        for index, entry in enumerate(entries[len(existing) :], len(existing)):
            state = board(entry["board"])
            result = client.decide(state, f"{split}:{entry['id']}")
            values = teacher.scores(state)
            action = result["decision"]["direction"]
            legal = result["decision"]["error"] is None
            record = {
                "id": entry["id"],
                "board_id": entry["board_id"],
                "game_id": entry["game_id"],
                "stage": entry["stage"],
                "teacher_move": entry["teacher_move"],
                "legal": legal,
                "teacher_match": legal and action == entry["teacher_move"],
                "teacher_scores": values,
                **result,
            }
            append_json(path, record)
            if (index + 1) % 25 == 0 or index + 1 == len(entries):
                write_json(status, {"stage": split, "completed": index + 1, "total": len(entries)})
                print(f"{split}: {index + 1}/{len(entries)}", flush=True)
    grouped = defaultdict(list)
    for agent in ["grpo", *manifest["baseline_agents"]]:
        for index, seed in enumerate(seeds):
            target = output / "games" / f"{agent}-{seed}.json"
            if target.exists():
                trace = json.loads(target.read_text())
                replay(trace)
            elif agent == "grpo":
                write_json(
                    status,
                    {"stage": "games", "agent": agent, "completed": index, "total": len(seeds)},
                )
                trace = served_game(client, seed, 3000, target, status)
            else:
                started = time.monotonic()
                trace = play(make_agent(agent, seed, 2), seed, 3000)
                replay(trace)
                trace["wall_seconds"] = time.monotonic() - started
                trace["ending"] = "game_over" if trace["terminated"] else "move_cap"
                write_json(target, trace)
                print(f"{agent} seed={seed}: {trace['ending']}, score={trace['score']}", flush=True)
            grouped[agent].append(trace)
    test_records = read_jsonl(output / "test-positions.jsonl")
    seen_records = read_jsonl(output / "seen-probe-positions.jsonl")
    latencies = [r["latency_seconds"] for r in test_records]
    game_decisions = [d for t in grouped["grpo"] for d in t["decisions"]]
    summary = {
        "run": args.run,
        "test_positions": summarize_positions(test_records),
        "seen_probe": summarize_positions(seen_records),
        "games": {agent: summarize_games(traces) for agent, traces in grouped.items()},
        "position_latency_seconds": {
            "p50": statistics.median(latencies),
            "p95": quantile(latencies, 0.95),
            "mean": statistics.mean(latencies),
        },
        "game_latency_seconds": {
            "requests": len(game_decisions),
            "p50": statistics.median(d["latency_seconds"] for d in game_decisions),
            "p95": quantile([d["latency_seconds"] for d in game_decisions], 0.95),
        },
        "all_completed_games_replayed": True,
        "limits": manifest["limits"],
    }
    write_json(output / "summary.json", summary)
    write_json(status, {"stage": "complete", "run": args.run})
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
