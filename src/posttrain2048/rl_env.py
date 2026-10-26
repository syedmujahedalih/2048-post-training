"""A bounded 2048 environment for TRL's native per-rollout tool interface."""

from __future__ import annotations

import math
from typing import Literal

from .data import observation
from .game import ACTIONS, Game, board, rows


class Game2048Env:
    def __init__(self, instance_id: int = 0, horizon: int = 16):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self._instance_id = instance_id
        self._horizon = horizon
        self._resets = 0

    def reset(self, initial_board, episode_seed, example_id, **kwargs) -> None:
        self._resets += 1
        self._seed = int(episode_seed) + self._instance_id * 1_000_003 + self._resets * 10_000_019
        self._game = Game(self._seed, state=board(initial_board))
        self._initial_board = rows(self._game.board)
        self._example_id = example_id
        self._attempts = []
        self._post_terminal_calls = 0

    def move(self, direction: Literal["UP", "DOWN", "LEFT", "RIGHT"]) -> str:
        """Slide the 2048 board in one direction and return the next observation.

        Args:
            direction: The direction to slide: UP, DOWN, LEFT, or RIGHT.

        Returns:
            The board, legal actions, and whether the episode has ended.
        """
        if len(self._attempts) >= self._horizon or self._game.done:
            self._post_terminal_calls += 1
            return "Episode finished. Do not call more tools. Reply DONE."
        before = self._game.board
        valid = False
        gained = 0
        spawned = None
        if direction in ACTIONS:
            step = self._game.step(direction)
            valid, gained, spawned = step.valid, step.score_gain, step.spawned
        self._attempts.append(
            {
                "action": direction,
                "before": rows(before),
                "board": rows(self._game.board),
                "valid": valid,
                "score_gain": gained,
                "spawned": spawned,
                "score": self._game.score,
            }
        )
        ended = len(self._attempts) >= self._horizon or self._game.done
        status = (
            "Episode finished. Do not call more tools. Reply DONE."
            if ended
            else "Call move once for the next decision."
        )
        return observation(self._game.board) + f"\nScore gained: {self._game.score}\n{status}"

    def _finish(self, completion):
        tool_messages = [m for m in completion if m["role"] == "tool"]
        calls = [
            call
            for m in completion
            if m["role"] == "assistant"
            for call in (m.get("tool_calls") or [])
        ]
        errors = [
            m
            for m in tool_messages
            if m.get("name") != "move" or str(m.get("content", "")).startswith("{'error':")
        ]
        assert (
            len(tool_messages) == len(self._attempts) + len(errors) + self._post_terminal_calls
        ), "Executed tools and recorded transcript disagree"
        assert len(self._attempts) + len(errors) <= self._horizon, (
            "Parallel tool calls exceeded the attempt budget"
        )
        invalid = sum(not step["valid"] for step in self._attempts) + len(errors)
        early_exit = not self._game.done and len(self._attempts) + len(errors) < self._horizon
        # A non-action reply before the budget ends is an invalid decision attempt.
        invalid += int(early_exit)
        score = self._game.score
        return {
            "example_id": self._example_id,
            "instance_id": self._instance_id,
            "seed": self._seed,
            "horizon": self._horizon,
            "initial_board": self._initial_board,
            "attempts": self._attempts,
            "score_gain": score,
            "invalid_attempts": invalid,
            "tool_errors": len(errors),
            "early_exit": early_exit,
            "post_terminal_calls": self._post_terminal_calls,
            "unexecuted_tail_calls": len(calls) - len(tool_messages),
            "terminal": self._game.done,
            "reward": math.log1p(score) - 0.1 * invalid,
            "completion": completion,
        }
