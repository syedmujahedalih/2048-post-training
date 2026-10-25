"""Shared observation format and leakage-aware dataset validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .game import ACTIONS, Board, board, legal_moves, rows

SYSTEM = (
    "You play 2048 on a 4x4 board. Zero is an empty cell. A move slides all tiles "
    "in one direction; adjacent equal tiles merge once per move. Each merge adds "
    "its resulting tile value to the score. After a valid move, a random empty "
    "cell receives a 2 (90%) or a 4 (10%). Maximize the eventual game score. "
    "Choose a legal move by calling move with a direction. Do not add commentary."
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Slide the 2048 board in one direction and return the next observation.",
            "parameters": {
                "type": "object",
                "properties": {"direction": {"type": "string", "enum": list(ACTIONS)}},
                "required": ["direction"],
                "additionalProperties": False,
            },
        },
    }
]


def observation(state: Board) -> str:
    return (
        "Board (rows top to bottom):\n"
        + "\n".join(" ".join(map(str, row)) for row in rows(state))
        + "\nLegal moves: "
        + ", ".join(legal_moves(state))
    )


def canonical_key(state: Board) -> str:
    """Identify the same position under rotation/reflection across dataset splits."""
    variants = []
    matrix = rows(state)
    for _ in range(4):
        variants.append(tuple(value for row in matrix for value in row))
        variants.append(tuple(value for row in matrix for value in reversed(row)))
        matrix = [list(row) for row in zip(*matrix[::-1], strict=True)]
    return hashlib.sha256(json.dumps(min(variants)).encode()).hexdigest()


def example(state: Board, action: str, metadata: dict) -> dict:
    if action not in legal_moves(state):
        raise ValueError("Teacher action must be legal")
    return {
        **metadata,
        "board_id": canonical_key(state),
        "board": rows(state),
        "legal_moves": list(legal_moves(state)),
        "teacher_move": action,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": observation(state)},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "move", "arguments": {"direction": action}},
                    }
                ],
            },
        ],
        "tools": TOOLS,
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_dataset(directory: Path) -> dict:
    seen_boards: set[str] = set()
    seen_games: set[str] = set()
    summary = {}
    for split in ("train", "validation", "test"):
        entries = read_jsonl(directory / f"{split}.jsonl")
        ids: set[str] = set()
        games: set[str] = set()
        for row in entries:
            state = board(row["board"])
            expected = example(state, row["teacher_move"], {})
            for field in ("board_id", "legal_moves", "messages", "tools"):
                if row[field] != expected[field]:
                    raise ValueError(f"Incorrect {field} in {split}: {row.get('id')}")
            if row["board_id"] in ids:
                raise ValueError("Duplicate board within split")
            ids.add(row["board_id"])
            games.add(row["game_id"])
        if ids & seen_boards:
            raise ValueError("Board leakage across splits, including rotation/reflection")
        if games & seen_games:
            raise ValueError("Game leakage across splits")
        seen_boards.update(ids)
        seen_games.update(games)
        summary[split] = {
            "examples": len(entries),
            "games": len(games),
            "actions": dict(Counter(row["teacher_move"] for row in entries)),
            "stages": dict(Counter(row["stage"] for row in entries)),
            "behavior_policies": dict(Counter(row["behavior_policy"] for row in entries)),
            "sha256": hashlib.sha256((directory / f"{split}.jsonl").read_bytes()).hexdigest(),
        }
    return {"splits": summary, "board_leakage": 0, "game_leakage": 0, "symmetry_leakage": 0}
