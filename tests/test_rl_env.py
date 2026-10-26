import math

import pytest

from posttrain2048.game import Game, board, legal_moves
from posttrain2048.rl_env import Game2048Env

START = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def completion(actions, responses, tail=False):
    messages = []
    for action, response in zip(actions, responses, strict=True):
        messages += [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "move", "arguments": {"direction": action}},
                    }
                ],
            },
            {"role": "tool", "name": "move", "content": response},
        ]
    messages.append(
        {"role": "assistant", "content": "DONE"}
        if not tail
        else {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": "move", "arguments": {"direction": "UP"}}}
            ],
        }
    )
    return messages


def test_actual_transition_reward_and_tail_detection():
    env = Game2048Env(horizon=1)
    env.reset(START, 42, "example")
    response = env.move("LEFT")
    result = env._finish(completion(["LEFT"], [response], tail=True))
    assert result["score_gain"] == 4
    assert result["reward"] == math.log1p(4)
    assert result["unexecuted_tail_calls"] == 1
    assert result["invalid_attempts"] == 0
    assert "Episode finished" in response
    game = Game(result["seed"], state=board(START))
    assert game.step("LEFT").board == board(result["attempts"][0]["board"])


def test_rng_streams_are_isolated_and_reproducible():
    def run(instance):
        env = Game2048Env(instance_id=instance, horizon=2)
        env.reset(START, 9, "example")
        responses, actions = [], []
        for _ in range(2):
            action = legal_moves(env._game.board)[0]
            actions.append(action)
            responses.append(env.move(action))
        return env._finish(completion(actions, responses))

    assert run(0) == run(0)
    assert run(0)["seed"] != run(1)["seed"]


def test_invalid_and_early_exit_penalties():
    env = Game2048Env(horizon=2)
    env.reset(START, 9, "example")
    response = env.move("SIDEWAYS")
    result = env._finish(completion(["SIDEWAYS"], [response]))
    assert result["initial_board"] == result["attempts"][0]["board"]
    assert result["reward"] == -0.2
    assert result["invalid_attempts"] == 2


def test_post_terminal_call_does_not_change_game():
    env = Game2048Env(horizon=1)
    env.reset(START, 9, "example")
    first = env.move("LEFT")
    state = env._game.board
    second = env.move("RIGHT")
    result = env._finish(completion(["LEFT", "RIGHT"], [first, second]))
    assert env._game.board == state
    assert len(result["attempts"]) == 1
    assert result["post_terminal_calls"] == 1


def test_transcript_mismatch_is_rejected():
    env = Game2048Env(horizon=1)
    env.reset(START, 9, "example")
    env.move("LEFT")
    with pytest.raises(AssertionError, match="transcript"):
        env._finish([])
