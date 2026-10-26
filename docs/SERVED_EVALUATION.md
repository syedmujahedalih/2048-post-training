# Served GRPO evaluation

The first held-out evaluation uses the completed GRPO model through the local
vLLM Metal API. The position test is complete; the 30-seed gameplay comparison
is still running. The protocol below was fixed before requesting test positions.

## Held-out first decisions

| Test boards | Legal move | Teacher first-choice match | Random-legal expected match |
| --- | ---: | ---: | ---: |
| All 1,000 | 1,000/1,000 | 445/1,000 (44.5%) | 28.73% |
| Early: 448 | 448/448 | 193/448 (43.08%) | 28.31% |
| Middle: 354 | 354/354 | 153/354 (43.22%) | 29.76% |
| Late: 198 | 198/198 | 99/198 (50.0%) | 27.82% |

The random-legal reference is the expected first-choice match rate for uniformly
sampling each board's legal moves: the mean of `1 / number_of_legal_moves`.
It is not a separately sampled model result. The model exceeds that reference
on these boards while leaving substantial room to imitate the bounded teacher.
The prompt explicitly lists legal moves, so legality alone is a weak strategy test.

The seen-board diagnostic matched the teacher on 17/32 GRPO training boards
(53.13%), all early-stage positions. The comparable held-out early-stage match
rate is 43.08%. The seen set is small and selected; this descriptive gap alone
does not establish overfitting or show that GRPO caused it.

Observed request latency on the warmed server was about 0.679 seconds at the
median and 0.687 seconds at the 95th percentile. Gameplay results and the final
replay-checked plots will follow when all reserved seeds finish.

## Fixed protocol

- Model: the BF16 serving copy of the merged Qwen2.5-1.5B-Instruct checkpoint;
  tensor hashes are recorded in `results/served-grpo-v1/manifest.json`.
- Backend: vLLM Metal 0.32.0 with MLX on Apple M4, 24 GB unified memory.
- Positions: all 1,000 test examples from 11 held-out source games. The dataset
  hash is checked against the saved data audit. Report early/middle/late boards
  separately, because their difficulty and distribution differ.
- Seen diagnostic: all 32 boards used by GRPO, reported separately from the test
  set. This small, selected early-board set is not a representative training set.
- Gameplay: 30 new seeds reserved in the dataset plan, shared across GRPO,
  random legal, direction-priority, immediate-score greedy and depth-2 teacher.
- Prompt: the existing system instruction, legal moves, tool schema and full
  current board. Each move starts a fresh conversation. Board state is sufficient
  for the game transition; this protocol avoids growing history and context limits.
- Decoding: greedy, 128-token allowance, automatic tool choice, no forced grammar,
  correction, retries or fallback moves. Schema errors and illegal directions stop
  the model's game and count as policy failures, separately from natural game over.
- Game safety cap: 3,000 attempted moves per policy and seed. Capped games are
  labeled, and score at ending includes failures and caps rather than pretending
  that every recorded game reached game over.
- Latency: one sequential client, warmed server, HTTP request duration including
  prefill and generation. This is task latency, not maximum serving throughput.

## Reproduction

Keep the running server's connection settings in ignored local configuration.
Do not add concrete ports, credentials, process IDs or raw runtime records to Git.

```sh
. .local/serving.env
.venv/bin/python scripts/evaluate_served_model.py --port "$PORT" --run served-grpo-v1
```

Use a new run name for a new experiment. Existing output is protected from
overwriting; `--resume` validates the same manifest, skips completed positions and
games, and restarts only an interrupted game from its seed. A local lock prevents
concurrent benchmark processes. Complete HTTP responses and runtime status stay
in ignored `.local/served-grpo-v1/`; saved public records omit connection details.

```sh
PYTHONPATH=src uv run --offline --managed-python --script scripts/analyze_served_evaluation.py
```

The analysis replays every game trace and checks every test ID and teacher label.
Its uncertainty intervals resample source games for test-board agreement and pair
game seeds across policies for score differences. They do not include uncertainty
from repeating model training.

## Interpretation limits

This run evaluates one checkpoint. Without a matching SFT-only evaluation, it
cannot establish that GRPO caused a benefit, overfitting or reward exploitation.
Teacher agreement measures imitation of a bounded solver, not optimal play.
The 1,000 positions are correlated within 11 games, not 1,000 independent trials.

Fresh-prompt greedy gameplay with a 128-token allowance differs from sampled,
multi-turn GRPO with 48 tokens per reply. Its policy-failure rate must not be
compared directly with the 185 early exits aggregated during training. Preserve
this run as the baseline; tuning on these findings requires new held-out games
for a later confirmation.
