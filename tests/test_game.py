import random

import pytest

from posttrain2048.agents import RandomAgent
from posttrain2048.cli import play, replay
from posttrain2048.game import ACTIONS, Game, board, legal_moves, merge_line, move


@pytest.mark.parametrize(
    "line,expected,score",
    [
        ([2, 2, 2, 2], (4, 4, 0, 0), 8),
        ([2, 2, 4, 0], (4, 4, 0, 0), 4),
        ([4, 0, 4, 4], (8, 4, 0, 0), 8),
        ([0, 0, 0, 0], (0, 0, 0, 0), 0),
        ([2, 4, 8, 16], (2, 4, 8, 16), 0),
        ([4, 4, 8, 8], (8, 16, 0, 0), 24),
    ],
)
def test_each_tile_merges_at_most_once(line, expected, score):
    assert merge_line(line) == (expected, score)


@pytest.mark.parametrize(
    "action,expected",
    [
        ("LEFT", [4, 0, 0, 0] + [0] * 12),
        ("RIGHT", [0, 0, 0, 4] + [0] * 12),
        ("UP", [2, 2, 0, 0] + [0] * 12),
        ("DOWN", [0] * 12 + [2, 2, 0, 0]),
    ],
)
def test_direction_convention(action, expected):
    state = board([2, 2, 0, 0] + [0] * 12)
    assert move(state, action)[0] == tuple(expected)


def test_invalid_move_does_not_consume_rng_or_spawn():
    game = Game(123, board([2, 0, 0, 0] + [0] * 12))
    rng_state = game.rng.getstate()
    result = game.step("LEFT")
    assert not result.valid
    assert result.spawned is None
    assert game.rng.getstate() == rng_state
    assert game.score == 0
    assert game.attempts == 1 and game.valid_steps == 0


def test_score_and_mass_account_for_merge_and_spawn():
    game = Game(5, board([2, 2, 0, 0] + [0] * 12))
    result = game.step("LEFT")
    assert result.score_gain == game.score == 4
    assert sum(game.board) == 4 + result.spawned[1]


def test_terminal_and_full_mergeable_boards():
    terminal = board([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert legal_moves(terminal) == ()
    assert Game(1, terminal).done
    assert not Game(1, board([2, 2] + list(terminal[2:]))).done


def test_initial_tiles_and_seeded_trace():
    assert Game(42).board == Game(42).board
    assert sum(v != 0 for v in Game(42).board) == 2
    trace = play(RandomAgent(12), 99, 500)
    replay(trace)


def test_spawn_distribution():
    values = [Game(seed)._spawn()[1] for seed in range(10000)]
    assert 0.88 < values.count(2) / len(values) < 0.92


def test_move_conserves_mass_for_random_boards():
    rng = random.Random(72)
    for _ in range(500):
        state = board([rng.choice([0, 2, 4, 8, 16, 32]) for _ in range(16)])
        for action in ACTIONS:
            assert sum(move(state, action)[0]) == sum(state)


@pytest.mark.parametrize("cells", [[0] * 15, [3] + [0] * 15, [-2] + [0] * 15, [True] + [0] * 15])
def test_reject_bad_boards(cells):
    with pytest.raises(ValueError):
        board(cells)


def test_reject_ragged_board():
    with pytest.raises(ValueError, match="four rows"):
        board([[2, 2, 2], [2, 2, 2, 2, 2], [0] * 4, [0] * 4])
