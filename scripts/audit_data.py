"""Validate all splits and write a reproducible 20-example inspection sheet."""

import argparse
import json
import random
from pathlib import Path

from posttrain2048.data import read_jsonl, validate_dataset
from posttrain2048.game import board
from posttrain2048.teacher import Teacher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=Path("data/demonstrations"))
args = parser.parse_args()
report = validate_dataset(args.data)
entries = read_jsonl(args.data / "train.jsonl")
selected = random.Random(2048).sample(entries, 20)
teacher = Teacher(2)
lines = [
    "# Demonstration inspection sheet",
    "",
    "Twenty reproducibly sampled training examples. No test examples are inspected for prompt tuning.",
    "",
    "Each target is a legal move and is recomputed using the pinned teacher.",
    "",
]
for entry in selected:
    state = board(entry["board"])
    scores = teacher.scores(state)
    assert teacher.choose(state) == entry["teacher_move"]
    lines += [
        f"## {entry['id']} ({entry['stage']}, {entry['behavior_policy']})",
        "",
        "```text",
        *[" ".join(f"{value:4}" for value in row) for row in entry["board"]],
        "```",
        "",
        f"Target: **{entry['teacher_move']}**. Legal: {', '.join(entry['legal_moves'])}.",
        "",
        f"Teacher action values (heuristic units): `{json.dumps({a: round(v, 2) for a, v in scores.items()})}`",
        "",
    ]
Path("docs/DATA_AUDIT.md").write_text("\n".join(lines))
report["sampled_teacher_labels_recomputed"] = len(selected)
Path("results/data-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
