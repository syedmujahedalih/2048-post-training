"""Local Strands Decider client. No cloud API, training, or action overrides."""

from __future__ import annotations

import json
import time
import urllib.request

from .data import SYSTEM, observation
from .game import ACTIONS, Board, legal_moves, move, rows


class DeciderAgent:
    def __init__(self, endpoint: str = "http://127.0.0.1:8099", mode: str = "afterstates"):
        if mode not in {"directions", "afterstates"}:
            raise ValueError(mode)
        self.endpoint = endpoint.rstrip("/")
        self.mode = mode
        self.calls = []

    def health(self) -> dict:
        with urllib.request.urlopen(self.endpoint + "/health", timeout=30) as response:
            return json.load(response)

    def decide(self, state: Board, legal_only: bool = True) -> dict:
        options = legal_moves(state) if legal_only else ACTIONS
        if not options:
            raise ValueError("Terminal board")
        if len(options) == 1:
            # The choice API requires at least two options. A forced legal move
            # does not need model inference; record this separately.
            return {
                "action": options[0],
                "forced": True,
                "elapsed_ms": 0.0,
                "probabilities": {options[0]: 1.0},
                "confidence": None,
            }
        criteria = {}
        for action in options:
            shifted, score = move(state, action)
            if self.mode == "afterstates":
                criteria[action] = {
                    "board_after_slide_before_random_spawn": rows(shifted),
                    "score_gain": score,
                    "valid": shifted != state,
                }
            else:
                criteria[action] = f"Slide every tile {action.lower()}."
        payload = {
            "state": SYSTEM + "\n\n" + observation(state),
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": "Which move best increases the eventual 2048 game score? Consider future playable moves and merges, as well as the immediate result. Select a legal move.",
                    "criteria": criteria,
                }
            },
        }
        started = time.perf_counter()
        request = urllib.request.Request(
            self.endpoint + "/v1/systemone",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        answer = result["answers"]["move"]
        if answer["choice"] not in options:
            raise ValueError("Decider returned a choice outside the supplied options")
        record = {
            "action": answer["choice"],
            "forced": False,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "probabilities": answer["probabilities"],
            "confidence": answer["confidence"],
            "usage": result.get("usage"),
            "server_latency_ms": result.get("latency_ms"),
        }
        self.calls.append(record)
        return record

    def choose(self, state: Board) -> str:
        return self.decide(state, legal_only=True)["action"]
