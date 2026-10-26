# Served GRPO evaluation

The first held-out evaluation is running against the completed GRPO model through
the local vLLM Metal API. The protocol below was fixed before requesting the test
positions. Results and figures will be added after the run completes.

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
