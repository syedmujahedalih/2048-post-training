import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from posttrain2048.game import ACTIONS, board, move


def test_moves_and_scores_against_original_javascript():
    if not shutil.which("node"):
        pytest.skip("Node.js is required for the independent original-JS check")
    rng = random.Random(314)
    cases = [
        {"board": [rng.choice([0, 2, 4, 8, 16, 32, 64]) for _ in range(16)], "action": action}
        for _ in range(500)
        for action in ACTIONS
    ]
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_original.mjs"
    result = subprocess.run(
        ["node", str(script)],
        input="\n".join(map(json.dumps, cases)),
        text=True,
        check=True,
        capture_output=True,
    )
    observed = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(observed) == len(cases)
    for case, original in zip(cases, observed, strict=True):
        shifted, gained = move(board(case["board"]), case["action"])
        assert list(shifted) == original["board"]
        assert gained == original["score"]
