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

Steps 1–3 are the current work. Steps 4–7 are reserved for the user's review and
later GPU session. No rented resources are used for the local work.

See [the project plan](docs/PLAN.md) and [progress](docs/PROGRESS.md).

## Local setup

```bash
uv sync
uv run pytest
```

The simulator and data-generation package use Python's standard library.
Decider runs in a separate environment to isolate its model dependencies.
