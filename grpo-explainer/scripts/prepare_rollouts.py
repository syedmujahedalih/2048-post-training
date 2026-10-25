"""Regenerate the illustrative boards with the project's actual simulator."""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))

from posttrain2048.game import Game, legal_moves  # noqa: E402

start = (128, 64, 32, 16, 64, 32, 16, 8, 8, 8, 4, 2, 0, 2, 0, 0)
rollouts = []
for index in range(4):
    game = Game(seed=81 + index, state=start)
    policy = random.Random(800 + index)
    steps = [{"board": list(start), "score": 0, "action": "START"}]
    for _ in range(16):
        if game.done:
            break
        action = policy.choice(legal_moves(game.board))
        game.step(action)
        steps.append({"board": list(game.board), "score": game.score, "action": action})
    rollouts.append(steps)

game = Game(seed=82, state=start)
step = game.step("LEFT")
result = {
    "start": start,
    "rollouts": rollouts,
    "left": {"board": step.board, "gain": step.score_gain},
}
(ROOT / "src/rollouts.json").write_text(json.dumps(result, indent=2) + "\n")
