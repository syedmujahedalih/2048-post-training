# Morning review and GPU session

Local preparation is separate from GPU training. No student SFT or RL run has
been performed, and no rented instance has been provisioned.

## Review these first

1. `README.md` and `docs/PROGRESS.md` for what is complete.
2. `docs/DATASET.md`, `docs/DATA_AUDIT.md`, and `results/tokenizer-check.json`.
3. `docs/DECIDER_REPORT.md` and its linked raw results. The pilot does not yet
   justify extra adviser-assisted RL spending; start with the core student run.
4. `docs/PLAN.md` for hypotheses, evaluation controls, and budget.

The test seeds 800000–800029 are reserved for the later student evaluation.

## Step 4: supervised fine-tuning

- Clone this private repository to a single rented Linux GPU instance with
  24 GB VRAM. Set up the CUDA training environment separately from macOS/MLX.
- Pin tested versions of PyTorch, Transformers, PEFT, TRL, and Datasets on that
  machine. The local Decider dependency snapshot is for inference on macOS;
  it is not a tested CUDA SFT/GRPO lockfile.
- Load `Qwen/Qwen2.5-1.5B-Instruct` and record the model revision. The tokenizer
  already checked locally is revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Measure the untouched model before modifying its weights.
- Load `data/sft/train.jsonl` and `data/sft/validation.jsonl` with Datasets.
  These contain explicit conversational prompt/completion rows plus tools.
- Inspect an actual trainer batch. Use completion-only loss: board, rules, and
  tool-schema prompt tokens should have label -100; the assistant move call
  should contribute to loss. There are 20 target tokens per checked example.
- First overfit 20 examples and verify adapter gradients/updates. Then try one
  initial epoch on the full 8,000-example training split. A 512-token example
  limit covers the locally checked examples, whose maximum is 361 tokens.
- Start with a modest LoRA rank (e.g. 16), small microbatch, and gradient
  accumulation. Measure memory before increasing batch size.
- Save/reload the adapter and evaluate on validation boards and validation games.

Training scripts are deliberately not claimed to be tested or ready yet. The
next coding work is the actual trainer setup and its first-batch checks.

## Step 5: online GRPO

- Wrap the verified simulator in TRL's stateful environment interface. Match the
  move-tool name, parameters, action parser, and observation convention used by
  SFT. The simulator exists; the TRL wrapper is still to implement and test.
- Begin from the SFT checkpoint, four trajectories per training starting board,
  and a 16-decision horizon. Use independently seeded random spawns per trajectory.
- Initial reward proposal: `log1p(score_gain) - 0.1 * invalid_attempts`. This is
  a proxy for learning on short episodes, not an established best reward for 2048.
- Count malformed/no-op actions against the horizon, and verify environment
  observations are masked out of the loss. Record entropy and within-group reward
  variance; four repeated moves with identical rewards provide little signal.
- Verify 10 updates and a checkpoint reload before a longer budgeted run.
- Track full-game validation separately from episode reward. The original plan
  also flags history/context differences between short training episodes and
  fresh-board inference prompts as a possible source of regression.

## Step 6: optional adviser experiment

Use the local Decider findings to decide whether this deserves compute. If
included, branch from the same SFT checkpoint and match training boards,
interaction count, and settings. Report the adviser's additional inference cost.
Keep Decider frozen and let the student sample its own actions. Evaluate advice
on/off and include SFT-plus-advice as an inference-only control.

## Step 7: evaluation and explanation

- Compare the untouched student, SFT, and RL checkpoints with random legal play,
  fixed-priority play, and the solver teacher under the same rules.
- Report scores, tile distribution, fraction reaching 2048, invalid moves,
  truncations, latency, and total cost. Use paired seed lists and distinguish
  evaluation noise from training-seed variability.
- Save configs, adapter versions, complete traces, and reproduction commands.
- A negative result is a valid result. One exploratory RL run is not enough to
  claim an algorithm is generally better.

## Resource boundary

The proposed first-experiment allowance is $20–40. Check the offered rental
price, run a short throughput test, and estimate total cost including evaluation
and idle time before starting a long job. The user will review and perform the
GPU phase; this local preparation has not purchased compute.
