# Demonstration dataset

The local run generated **10,000 unique positions** from **104 games**:

| Split | Examples | Games | Early / middle / late |
| --- | ---: | ---: | --- |
| Train | 8,000 | 82 | 3,690 / 2,726 / 1,584 |
| Validation | 1,000 | 11 | 439 / 363 / 198 |
| Test | 1,000 | 11 | 448 / 354 / 198 |

`early` means maximum tile <=64; `middle` means <=512; `late` means >512.
The behavior policies are random legal play, a fixed direction priority, and
the teacher. Every selected position, including random-player positions, is
labeled by the same bounded depth-2 expectimax teacher.

Generation and validation took about 14.3 seconds on this M4 Mac. This is an
observed run time, not a guarantee for other machines or search depths.

## Files and reproduction

- `data/demonstrations/*.jsonl`: complete examples, provenance, and tool schema.
- `data/demonstrations/games.jsonl`: behavior-game summaries.
- `data/demonstrations/manifest.json`: settings, solver revision, and file hashes.
- `data/sft/*.jsonl`: explicit conversational `prompt` / `completion` / `tools`
  rows, intended for completion-only SFT loss.
- `data/pilot/`: the preceding 1,000-example plumbing run. It is separate from
  the full dataset and must not be concatenated with it, because seeds overlap.
- `docs/DATA_AUDIT.md`: 20 reproducibly sampled examples and teacher action values.
- `results/data-audit.json`: full split checks and 20 recomputed teacher labels.
- `results/tokenizer-check.json`: exact rendered Qwen tool call and loss boundary.

```bash
uv run python scripts/build_teacher.py
uv run python scripts/generate_data.py --output data/reproduced
uv run python scripts/audit_data.py --data data/reproduced
uv run python scripts/export_sft.py --data data/reproduced --output data/reproduced-sft
```

The generator refuses to overwrite an existing dataset. Use a new output folder.

## Leakage controls

Each split uses disjoint whole games and seed ranges. A canonical board hash
identifies all eight rotations/reflections. Such a board is retained only once
across the entire dataset. The validation step checks both within-split and
cross-split duplicates, as well as game IDs. The stored audit reports zero
leakage by these checks. These checks do not establish independence of similar
board patterns or generalization by a future trained model.

Gameplay pilot seeds begin at 700000. Reserve seeds 800000–800029 for the later
final student evaluation; do not use those games for prompt or reward tuning.

## Training format

The target is an executable tool call, for example:

```text
<tool_call>
{"name": "move", "arguments": {"direction": "LEFT"}}
</tool_call>
```

The tokenizer check verified all 8,000 training examples against the pinned
Qwen tokenizer revision. The longest complete example is 361 tokens, the
median is 346, and each assistant target is 20 tokens including its terminator.
An initial 512-token SFT limit therefore covers these examples without truncation.

Use `data/sft/` with **completion-only loss**. Loading the `messages` format with
default whole-sequence loss would also train on the system instructions, board,
and schema. The standard Qwen chat template does not supply a generation mask
for assistant-only loss; the explicit prompt/completion boundary avoids relying
on that feature. Verify an actual trainer batch again on the rented GPU.

## Limits

- A depth-2 solver is a bounded teacher; labels are not proven optimal actions.
- Many positions have nearly tied action values. First-choice agreement can
  penalize a reasonable alternative; full-game outcomes remain necessary.
- Teacher behavior games can hit the 3,000-move cap. Their summaries record
  whether the game actually terminated.
- Solver positions are correlated within games. The 8,000 examples are not
  8,000 independent experiments.
- The retained official test split is not used to tune the Decider prompt.
