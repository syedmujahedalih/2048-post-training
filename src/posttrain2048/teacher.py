"""ctypes bridge to a pinned, bounded-search version of nneonneo's solver."""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path

from .game import ACTIONS, Board, board, legal_moves

ROOT = Path(__file__).resolve().parents[2]


def pack(state: Board) -> int:
    if any(value > 32768 for value in state):
        raise ValueError("Upstream bitboard supports tiles up to 32768")
    return sum((value.bit_length() - 1 if value else 0) << (4 * i) for i, value in enumerate(state))


def unpack(encoded: int) -> Board:
    return board([1 << rank if (rank := (encoded >> (4 * i)) & 15) else 0 for i in range(16)])


class Teacher:
    def __init__(self, depth: int = 2, library: str | Path | None = None):
        if not 1 <= depth <= 4:
            raise ValueError("Choose a bounded search depth from 1 to 4")
        self.depth = depth
        suffix = "dylib" if platform.system() == "Darwin" else "so"
        path = Path(library) if library else ROOT / "build" / f"lib2048.{suffix}"
        if not path.exists():
            raise FileNotFoundError(
                f"Build the teacher first: python scripts/build_teacher.py ({path})"
            )
        self.lib = ctypes.CDLL(str(path))
        self.lib.init_tables.argtypes = []
        self.lib.init_tables.restype = None
        self.lib.execute_move.argtypes = [ctypes.c_int, ctypes.c_uint64]
        self.lib.execute_move.restype = ctypes.c_uint64
        self.lib.score_move_limited.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
        self.lib.score_move_limited.restype = ctypes.c_float
        self.lib.init_tables()

    def deterministic_move(self, state: Board, action: str) -> Board:
        return unpack(self.lib.execute_move(ACTIONS.index(action), pack(state)))

    def scores(self, state: Board) -> dict[str, float]:
        encoded = pack(state)
        return {
            action: float(self.lib.score_move_limited(encoded, ACTIONS.index(action), self.depth))
            for action in legal_moves(state)
        }

    def choose(self, state: Board) -> str:
        scores = self.scores(state)
        if not scores:
            raise ValueError("A terminal board has no teacher move")
        return max(scores, key=scores.get)
