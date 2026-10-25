"""Simple, explicitly labeled control policies."""

import random

from .game import Board, legal_moves, move


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, state: Board) -> str:
        return self.rng.choice(legal_moves(state))


class PriorityAgent:
    def choose(self, state: Board) -> str:
        legal = legal_moves(state)
        return next(action for action in ("DOWN", "LEFT", "RIGHT", "UP") if action in legal)


class GreedyAgent:
    def choose(self, state: Board) -> str:
        return max(
            legal_moves(state),
            key=lambda action: (move(state, action)[1], move(state, action)[0].count(0)),
        )
