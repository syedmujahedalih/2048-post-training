"""Export explicit prompt/completion rows for completion-only SFT loss."""

import argparse
import hashlib
import json
from pathlib import Path

from posttrain2048.data import read_jsonl, validate_dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=Path("data/demonstrations"))
parser.add_argument("--output", type=Path, default=Path("data/sft"))
args = parser.parse_args()
validate_dataset(args.data)
args.output.mkdir(parents=True, exist_ok=True)
manifest = {
    "source": str(args.data),
    "format": "conversational prompt/completion with tools",
    "completion_only_loss": True,
    "splits": {},
}
for split in ("train", "validation", "test"):
    entries = read_jsonl(args.data / f"{split}.jsonl")
    path = args.output / f"{split}.jsonl"
    with path.open("w") as out:
        for row in entries:
            out.write(
                json.dumps(
                    {
                        "prompt": row["messages"][:-1],
                        "completion": row["messages"][-1:],
                        "tools": row["tools"],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    manifest["splits"][split] = {
        "examples": len(entries),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
