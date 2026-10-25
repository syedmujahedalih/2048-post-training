"""Validate the exact student template and loss boundary without loading weights."""

import argparse
import json
import statistics
from pathlib import Path

from huggingface_hub import model_info
from transformers import AutoTokenizer

from posttrain2048.data import read_jsonl

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=Path("data/demonstrations"))
parser.add_argument("--output", type=Path, default=Path("results/tokenizer-check.json"))
parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
args = parser.parse_args()
revision = model_info(args.model).sha
tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
lengths = []
targets = []
sample = None
for entry in read_jsonl(args.data / "train.jsonl"):
    messages = entry["messages"]
    tools = entry["tools"]
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tools=tools, add_generation_prompt=True, return_dict=False
    )
    full_ids = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=False, return_dict=False
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Training prompt and inference prefix disagree")
    target_ids = full_ids[len(prompt_ids) :]
    text = tokenizer.decode(target_ids)
    payload = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
    call = json.loads(payload)
    if call != {"name": "move", "arguments": {"direction": entry["teacher_move"]}}:
        raise ValueError("Rendered tool call differs from teacher action")
    lengths.append(len(full_ids))
    targets.append(len(target_ids))
    if sample is None:
        sample = {
            "id": entry["id"],
            "rendered_full": tokenizer.decode(full_ids),
            "prompt_tokens": len(prompt_ids),
            "target_tokens": len(target_ids),
            "target_text": text,
            "prompt_labels": "-100 (ignored)",
            "target_labels": target_ids,
        }
report = {
    "model": args.model,
    "tokenizer_revision": revision,
    "examples_checked": len(lengths),
    "max_tokens": max(lengths),
    "median_tokens": statistics.median(lengths),
    "target_token_lengths": sorted(set(targets)),
    "all_prefixes_match": True,
    "all_targets_roundtrip": True,
    "sample": sample,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "sample"}, indent=2))
