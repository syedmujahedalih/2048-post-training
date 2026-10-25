# 2048 post-training lab

Learn post-training end to end by teaching a small language model to play 2048.

The experiment is **solver demonstrations → SFT → online GRPO → game evaluation**.
A frozen Strands Decider adviser is a conditional comparison, after measuring
whether it gives useful 2048 advice.

## Seven steps

1. Build and verify a seeded simulator and baseline players locally.
2. Generate solver-labeled demonstrations and verify the data locally.
3. Benchmark Strands Decider on the Mac.
4. Train a LoRA SFT adapter on a rented GPU.
5. Run short-episode online GRPO on that GPU.
6. Compare adviser-assisted RL if the local pilot supports it.
7. Evaluate full games and write up the findings.

**Local steps 1–3 are complete.** One-update SFT and synthetic-reward GRPO
compatibility checks also pass on the Mac's MPS GPU, including the actual 1.5B
Qwen student. Full SFT and game-environment RL remain to do; no rented compute
was used. See the [MPS check and its limits](docs/MPS_TRL_CHECK.md).

Before the GPU session, watch the narrated [GRPO and online RL explainer](grpo-explainer/README.md)
or read its [transcript](grpo-explainer/TRANSCRIPT.md). It uses 2048 to explain
rollouts, group-relative advantages, policy updates, clipping, and evaluation.

## Results to review

- **10,000 demonstrations:** 8,000 training / 1,000 validation / 1,000 test.
- **28 passing tests**, including independent checks against the original game
  and native solver. The core checks also pass GitHub CI on Linux.
- **Student tokenizer validated:** all 8,000 training rows render correct move
  calls with an explicit completion-loss boundary; maximum 361 tokens.
- **Decider tested locally:** 128 diagnostic positions and six paired-seed games
  on Apple M4 through MLX. Its mean score was 1,591, versus 1,487 for a direction
  priority baseline, 1,985 for immediate-score greedy play, and 52,158 for the
  solver teacher. The small pilot does not establish a useful adviser benefit.

Start with [the morning handoff](docs/GPU_HANDOFF.md),
[dataset notes](docs/DATASET.md), and [benchmark findings](docs/DECIDER_REPORT.md).
The original [project plan](docs/PLAN.md) and [progress log](docs/PROGRESS.md)
provide the full context.

## Local setup

```bash
uv sync --locked
uv run python scripts/build_teacher.py
uv run pytest
uv run posttrain2048 play --agent teacher --seed 42 --output results/my-game.json
uv run posttrain2048 replay results/my-game.json
```

The simulator and data-generation package use Python's standard library.
Decider runs in a separate environment to isolate its model dependencies.
See [local reproduction commands](docs/LOCAL_COMMANDS.md).

## License

This project's original code is licensed under the [MIT License](LICENSE),
which permits commercial and non-commercial use. Third-party components retain
their own licenses and copyright notices; see [attributions](docs/THIRD_PARTY.md).
