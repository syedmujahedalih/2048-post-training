import random

import pytest

from posttrain2048.game import ACTIONS, board, legal_moves, move
from posttrain2048.teacher import Teacher, pack, unpack


@pytest.fixture(scope="module")
def teacher():
    try:
        return Teacher(2)
    except FileNotFoundError:
        pytest.skip("Build native teacher with python scripts/build_teacher.py")


def test_bitboard_roundtrip():
    state = board([0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    assert unpack(pack(state)) == state


def test_python_moves_match_upstream_native_solver(teacher):
    rng = random.Random(412)
    for _ in range(2000):
        state = board([rng.choice([0, 0, 2, 4, 8, 16, 32, 64, 128, 1024]) for _ in range(16)])
        for action in ACTIONS:
            assert move(state, action)[0] == teacher.deterministic_move(state, action)


def test_teacher_always_chooses_a_legal_move(teacher):
    rng = random.Random(91)
    for _ in range(30):
        state = board([rng.choice([0, 2, 4, 8, 16, 32]) for _ in range(16)])
        if legal_moves(state):
            assert teacher.choose(state) in legal_moves(state)
