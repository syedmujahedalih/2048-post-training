# Original Qwen, SFT and GRPO: matched checkpoint comparison

The SFT-only and original-instruct controls are running locally. They reuse the
completed [GRPO benchmark](SERVED_EVALUATION.md) without changing its measurements.
This comparison separates the contributions of our SFT and GRPO stages.

The SFT position test is complete: 1,000/1,000 legal calls and 450 teacher
first-choice matches, compared with GRPO's 1,000 legal calls and 445 matches.
The two policies choose different directions on 83 boards: SFT alone matches
the teacher on 31, and GRPO alone on 26. The small net difference is not a
conclusion about game quality; the paired game comparison is still running.

## Controls

| Stage | Artifact | Additional project training |
| --- | --- | --- |
| Original instruct | Pinned Qwen2.5-1.5B-Instruct | None |
| SFT | Full 8,000-example adapter merged into the same original checkpoint | 2,000 SFT updates |
| SFT + GRPO | Previously evaluated merged checkpoint | SFT followed by 100 GRPO updates |

“Original” means the vendor's instruction-tuned model before this project's
training, rather than a raw pretrained model. All serving artifacts use BF16
weights, with verified tokenizer/chat-template hashes shared across stages.
The original and SFT adapters/master checkpoints remain preserved.

Each control uses the same 1,000 held-out boards, 32 diagnostic training-board
probes, 30 game seeds, prompts, legal-move observations, Hermes tool parser,
greedy decoding and 128-token allowance as GRPO. Gameplay resets the prompt for
each board and stops on the first malformed or illegal action. Scores at the
3,000-attempt cap are distinguished from genuine game over.

The four algorithmic baseline players are reused from their frozen, replayed
traces. Their policies and game seeds are unchanged; rerunning them would add
no information. Run folders stay separate:

- `results/served-base-v1/`: original-instruct control.
- `results/served-sft-v1/`: SFT-only control.
- `results/served-grpo-v1/`: completed GRPO/reference benchmark.

## Reproduction

The pipeline prepares non-overwriting local exports, validates their tensor
hashes, and serves one model at a time. It evaluates SFT first, original Qwen
second, then restores GRPO serving. It checks that frozen reference files are
unchanged. Runtime connection details and process records stay ignored.

```sh
.venv/bin/python scripts/run_checkpoint_comparison.py
```

Existing completed exports are verified and reused. Interrupted evaluations
resume with matching manifests; partial exports are protected from overwriting.
An exclusive local pipeline lock prevents duplicate runs. Network checks cover
the API and its internal engine listeners, not only the API bind address.

After both controls finish:

```sh
PYTHONPATH=src uv run --offline --managed-python --script scripts/analyze_checkpoint_comparison.py
```

The analysis verifies the shared test contract, replays all model and baseline
traces, and calculates paired stage differences. Score intervals resample 30
common seeds; teacher-agreement intervals resample 11 source-game clusters.
They do not capture variation from repeating training with new training seeds.

## Interpretation

The important contrasts are SFT minus original, GRPO minus SFT, and GRPO minus
original. A negative GRPO-minus-SFT score difference would show a regression
for this deployed checkpoint and protocol. It would not, by itself, establish
reward hacking. The matched controls keep the parser and agent loop fixed;
they do not test the multi-turn training protocol or all deployment settings.
