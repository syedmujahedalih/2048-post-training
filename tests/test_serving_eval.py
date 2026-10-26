import json

import pytest

from posttrain2048.serving_eval import parse_move_response


def response(arguments=None, name="move", content=None):
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": content,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments or {"direction": "UP"}),
                            },
                        }
                    ],
                },
            }
        ]
    }


def test_legal_call_is_not_replaced_by_teacher_action():
    result = parse_move_response(response({"direction": "LEFT"}), ["UP", "LEFT"])
    assert result["direction"] == "LEFT" and result["error"] is None


def test_prose_claiming_a_move_does_not_count_as_executing_it():
    payload = {"choices": [{"message": {"content": 'I call move({"direction":"UP"})'}}]}
    assert parse_move_response(payload, ["UP"])["error"] == "no_tool_call"


@pytest.mark.parametrize("direction", [" up", "UP ", "up", "STAY", ["UP"]])
def test_bad_directions_are_not_silently_repaired(direction):
    assert parse_move_response(response({"direction": direction}), ["UP"])["error"]


def test_illegal_call_is_a_failure_even_if_well_formed():
    result = parse_move_response(response(), ["DOWN"])
    assert result["direction"] == "UP" and result["error"] == "illegal_move"


def test_parallel_calls_are_not_collapsed_into_one_move():
    payload = response()
    calls = payload["choices"][0]["message"]["tool_calls"]
    calls.append(calls[0].copy())
    assert parse_move_response(payload, ["UP"])["error"] == "multiple_tool_calls"


def test_extra_arguments_and_wrong_tools_are_rejected():
    assert parse_move_response(response({"direction": "UP", "score": 999}), ["UP"])["error"]
    assert parse_move_response(response(name="set_score"), ["UP"])["error"] == "wrong_tool"
