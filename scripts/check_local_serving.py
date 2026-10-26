"""Check the localhost server with one validation board; this is not a benchmark."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from posttrain2048.data import TOOLS, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True, help="Port from ignored local config")
    parser.add_argument("--model", default="2048-grpo")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    base_url = f"http://127.0.0.1:{args.port}/v1"
    models = request_json(f"{base_url}/models")
    if args.model not in {model["id"] for model in models["data"]}:
        raise ValueError(f"Server did not list model {args.model!r}")

    example = read_jsonl(ROOT / "data/demonstrations/validation.jsonl")[0]
    payload = {
        "model": args.model,
        "messages": example["messages"][:-1],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "repetition_penalty": 1,
        "max_tokens": 128,
        "seed": 2048,
        "stream": False,
    }
    started = time.monotonic()
    response = request_json(f"{base_url}/chat/completions", payload)
    choice = response["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls") or []
    direction = None
    parse_error = None
    if len(calls) == 1 and calls[0]["function"]["name"] == "move":
        try:
            direction = json.loads(calls[0]["function"]["arguments"])["direction"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            parse_error = str(error)
    legal = direction in example["legal_moves"]
    report = {
        "endpoint_scope": "loopback; address supplied in local configuration",
        "served_models": {
            "data": [
                {"id": item["id"], "max_model_len": item.get("max_model_len")}
                for item in models["data"]
            ]
        },
        "validation_example_id": example["id"],
        "legal_moves": example["legal_moves"],
        "request": payload,
        "response": response,
        "parsed_direction": direction,
        "parse_error": parse_error,
        "legal_single_move": legal,
        "elapsed_seconds": time.monotonic() - started,
        "limits": (
            "One validation board checks serving and unconstrained automatic tool parsing. "
            "It does not measure held-out accuracy, full-game quality, or MPS/MLX parity. "
            "No required tool choice, grammar, repair, or retry was applied."
        ),
    }
    if args.output:
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not legal:
        raise SystemExit("Server responded, but did not produce one legal move tool call")


if __name__ == "__main__":
    main()
