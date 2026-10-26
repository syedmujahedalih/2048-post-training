"""Strict evaluation of executable move calls, without repairs or fallbacks."""

from __future__ import annotations

import json

from .game import ACTIONS


def parse_move_response(response: dict, legal: tuple[str, ...] | list[str]) -> dict:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    result = {
        "direction": None,
        "error": None,
        "finish_reason": choice.get("finish_reason"),
        "has_commentary": bool(message.get("content")),
    }
    if not calls:
        result["error"] = "no_tool_call"
    elif len(calls) != 1:
        result["error"] = "multiple_tool_calls"
    else:
        function = calls[0].get("function") or {}
        if calls[0].get("type") != "function" or function.get("name") != "move":
            result["error"] = "wrong_tool"
        else:
            try:
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict) or set(arguments) != {"direction"}:
                    raise ValueError("Arguments must match the move schema exactly")
                direction = arguments["direction"]
                if not isinstance(direction, str) or direction not in ACTIONS:
                    result["error"] = "bad_direction"
                else:
                    result["direction"] = direction
                    if direction not in legal:
                        result["error"] = "illegal_move"
            except (KeyError, TypeError, ValueError):
                result["error"] = "bad_arguments"
    return result
