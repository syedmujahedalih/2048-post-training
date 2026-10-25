"""Standard 2048 rules, with pure moves and reproducible random tile spawning."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

Board = tuple[int, ...]
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")  # Also the upstream solver's integer order.


def board(values: Iterable[int] | Iterable[Iterable[int]]) -> Board:
    items = list(values)
    if len(items) == 4 and all(isinstance(row, (list, tuple)) for row in items):
        if any(len(row) != 4 for row in items):
            raise ValueError("A board must have four rows of four cells")
        items = [value for row in items for value in row]
    if len(items) != 16:
        raise ValueError("A board must have exactly 16 cells")
    if any(type(v) is not int or v < 0 or (v and v & (v - 1)) for v in items):
        raise ValueError("Cells must be integer zeroes or positive powers of two")
    return tuple(items)


def rows(state: Board) -> list[list[int]]:
    return [list(state[i : i + 4]) for i in range(0, 16, 4)]


def merge_line(line: Iterable[int]) -> tuple[tuple[int, ...], int]:
    nonzero = [value for value in line if value]
    merged: list[int] = []
    score = 0
    i = 0
    while i < len(nonzero):
        if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
            value = nonzero[i] * 2
            merged.append(value)
            score += value
            i += 2
        else:
            merged.append(nonzero[i])
            i += 1
    return tuple(merged + [0] * (4 - len(merged))), score


def move(state: Board, action: str) -> tuple[Board, int]:
    """Apply a deterministic slide/merge, without spawning a new tile."""
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action!r}")
    result = list(state)
    score = 0
    for line in range(4):
        if action == "LEFT":
            indices = [4 * line + j for j in range(4)]
        elif action == "RIGHT":
            indices = [4 * line + j for j in reversed(range(4))]
        elif action == "UP":
            indices = [4 * j + line for j in range(4)]
        else:
            indices = [4 * j + line for j in reversed(range(4))]
        merged, gained = merge_line(state[index] for index in indices)
        for index, value in zip(indices, merged, strict=True):
            result[index] = value
        score += gained
    return tuple(result), score


def legal_moves(state: Board) -> tuple[str, ...]:
    return tuple(action for action in ACTIONS if move(state, action)[0] != state)


def format_board(state: Board) -> str:
    return "\n".join(
        " ".join(f"{value:5}" if value else "    ." for value in row) for row in rows(state)
    )


@dataclass(frozen=True)
class Step:
    board: Board
    action: str
    score_gain: int
    valid: bool
    done: bool
    spawned: tuple[int, int] | None  # Flat cell index, value.


class Game:
    def __init__(self, seed: int, state: Board | None = None, score: int = 0):
        self.seed = seed
        self.rng = random.Random(seed)
        self.board = board(state) if state is not None else (0,) * 16
        self.score = score
        self.attempts = 0
        self.valid_steps = 0
        if state is None:
            self._spawn()
            self._spawn()

    def _spawn(self) -> tuple[int, int]:
        empty = [i for i, value in enumerate(self.board) if value == 0]
        if not empty:
            raise RuntimeError("Cannot spawn into a full board")
        value = 2 if self.rng.random() < 0.9 else 4
        index = self.rng.choice(empty)
        cells = list(self.board)
        cells[index] = value
        self.board = tuple(cells)
        return index, value

    @property
    def done(self) -> bool:
        return not legal_moves(self.board)

    def step(self, action: str) -> Step:
        next_board, gained = move(self.board, action)
        self.attempts += 1
        valid = next_board != self.board
        spawned = None
        if valid:
            self.board = next_board
            self.score += gained
            self.valid_steps += 1
            spawned = self._spawn()
        return Step(self.board, action, gained, valid, self.done, spawned)
