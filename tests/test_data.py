import json

import pytest

from posttrain2048.data import canonical_key, example, validate_dataset
from posttrain2048.game import board, rows


def test_symmetries_share_a_key():
    state = board([[2, 4, 8, 0], [16, 0, 2, 4], [0, 8, 32, 0], [0, 0, 0, 2]])
    rotated = board([list(row) for row in zip(*rows(state)[::-1], strict=True)])
    reflected = board([row[::-1] for row in rows(state)])
    assert canonical_key(state) == canonical_key(rotated) == canonical_key(reflected)


def test_reject_illegal_teacher_label():
    with pytest.raises(ValueError):
        example(board([2, 0, 0, 0] + [0] * 12), "UP", {})


def test_validation_detects_game_and_symmetric_board_leaks(tmp_path):
    state = board([2, 2, 0, 0] + [0] * 12)
    entry = example(state, "LEFT", {"game_id": "a", "stage": "early", "behavior_policy": "teacher"})
    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(entry) + "\n")
    with pytest.raises(ValueError, match="leakage"):
        validate_dataset(tmp_path)
