# Weekend project: post-training a language model to play 2048

Status: local steps 1–3 are complete. See `PROGRESS.md`, `DATASET.md`, and
`DECIDER_REPORT.md` for measured results. One-update MPS compatibility probes
also pass; see `MPS_TRL_CHECK.md`. The full steps 4–7 remain planned, and no
cloud instance has been provisioned.

## Questions

1. Does supervised imitation of a solver improve a small language model's play?
2. Does online reinforcement learning improve the resulting SFT checkpoint?
3. If Strands Decider gives useful advice, does a frozen adviser help learning
   enough to justify its additional inference cost?

The required experiment is SFT followed by online RL and full-game evaluation.
The Decider comparison follows a local usefulness benchmark. DPO is a reading
and CPU loss exercise, outside the rented training runs.

## Initial choices

| Item | Choice |
| --- | --- |
| Student | `Qwen/Qwen2.5-1.5B-Instruct` |
| Training | LoRA; Hugging Face Transformers, PEFT, and TRL |
| GPU target | One NVIDIA GPU with 24 GB VRAM; verify with a smoke run |
| Local work | Simulator, solver labels, data checks, baseline agents, Decider pilot |
| GPU environment | Separate Linux environment with tested, pinned dependencies; do not reuse the root Mac/MLX environment |
| Observations | Compact text representation of the 4x4 board, rules, and allowed directions |
| Actions | One `move(direction)` call per turn; direction is UP, DOWN, LEFT, or RIGHT |
| SFT labels | The same move-call format used during RL |
| RL | TRL GRPO with a stateful game environment, starting with 16-decision episodes |
| Adviser | Frozen `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`, subject to pilot results |

Starting from an Instruct checkpoint means we are adapting an existing
instruction-following model. We are not recreating base-to-instruction training.
We will verify the selected tokenizer's tool formatting before generating the
full dataset; if a formatting adapter is required, SFT and RL must share it.

The GRPO environment integration is a build task, not a verified capability of
this repository. It must pass the trace and gradient checks below before longer
training. Short episodes bound conversation growth; full games are evaluated
with a fresh board prompt at each decision to keep context bounded. We will
report this context difference and inspect it as a possible source of regression.

## 1. Build and verify the game locally — approximately 2 hours

- Implement a seeded simulator and a small terminal replay viewer.
- Use standard 2048 rules: each tile merges at most once per move; only valid
  moves spawn tiles; score increments by the values of newly merged tiles.
- Continue after reaching 2048, until no legal moves remain or evaluation caps
  are reached. Record reaching 2048 separately.
- Verify merge examples, score accounting, no-op moves, terminal boards, spawn
  probabilities, and reproducibility. Compare deterministic move results with
  the original game's implementation.
- Add random-legal and simple fixed-priority legal-move baselines.

Checkpoint: replaying a seed and action list reproduces the same state and score.

## 2. Generate demonstrations — approximately 2 hours

- Integrate the open expectimax solver as the SFT teacher. Bound its search
  effort and record the solver revision and settings.
- Generate a 1,000-example pilot, then target 10,000 examples if generation fits
  a one-hour initial CPU budget. Treat these as proposed sizes, not proven needs.
- Collect reachable early-, middle-, and later-game boards from a mix of solver,
  heuristic, and random play; ask the solver to label each saved position.
- Save game ID, seed, board, legal moves, teacher move, prompt, and target call.
- Split games into training, validation, and held-out data, then remove exact
  board duplicates across splits. Reserve separate seeds for full-game testing.
- Audit 20 examples by hand. Check that target moves are legal and formatting
  round-trips through the tokenizer and move parser.

Checkpoint: versioned data manifest, leakage checks, example inspection, and a
verified tokenizer boundary exist before a longer GPU run.

## 3. Benchmark Decider locally — approximately 1 hour

- Run Decider locally on held-out validation boards and a small gameplay pilot.
  Compare legality, agreement with the solver, score, and latency with baselines.
  Agreement with the solver is diagnostic, not the definition of game skill.
- Add adviser hints to the RL comparison only if this pilot shows a useful
  signal. Record a weak or negative pilot rather than treating Decider as an
  established 2048 expert. Do not use its confidence as a reward.

Checkpoint: a Decider pilot report establishes whether the adviser experiment
has enough support to prioritize. The completed pilot did not establish this.

## 4. Train the SFT checkpoint — approximately 2 focused hours

- Record the untouched Instruct model's baseline before changing its weights.
- Run a small GPU smoke test and measure memory, speed, and estimated cost.
- First overfit 20 examples to verify prompt masking, target formatting, and
  adapter updates. Inspect a batch's tokens, labels, and loss mask.
- Train on the selected demonstrations for at most one initial epoch. Use
  validation results to decide whether an additional run is justified.
- Save the adapter, tokenizer/template settings, base revision, seed, dependency
  versions, configuration, and metrics. Reload and verify inference.

Checkpoint: the checkpoint produces executable move calls and is evaluated on
held-out boards and a small set of games. A working pipeline is required;
improvement over the starting model is a result to measure, not assume.

## 5. Connect online GRPO — approximately 2–3 hours

- Wrap the simulator as a stateful environment for TRL. Each episode begins
  from a saved training board, with freshly sampled student actions.
- Start with four trajectories per initial board, each capped at 16 decision
  attempts. Each trajectory has its own reproducible tile-spawn RNG stream.
- Keep the SFT model's LoRA weights trainable and the game code fixed.
- Initial episode reward: `log1p(score_gain) - 0.1 * invalid_attempts`.
  This is a short-horizon training proxy; report full-game performance separately.
- Malformed calls and no-op moves count as invalid attempts and consume the
  episode's decision budget. They do not spawn a tile or earn score.
- Log raw score gains, invalid attempts, early terminations, truncations,
  within-group reward variation, policy entropy, and gradient norms alongside
  aggregate reward. Verify that sampled groups provide a learning signal.
- Inspect one complete rollout and its loss mask: policy-generated actions
  contribute to learning; environment observations and frozen adviser text do
  not. Verify sampling probabilities correspond to the executed policy actions.
- Do not override sampled actions with the solver's or Decider's selection.
- Run 10 updates, save/reload, and verify that trainable weights changed and
  losses/gradients are finite before launching a longer run.
- Give the first baseline RL run a two-GPU-hour limit. Extend only after checking
  throughput and reward/quality diagnostics, within the overall budget.

Checkpoint: actual sampled gameplay produces real policy updates. Completion of
this integration is not evidence that GRPO improves the model.

## 6. Compare frozen Decider advice — conditional, approximately 2 hours

- Branch both RL arms from the identical SFT checkpoint: A gets no advice; B
  gets the frozen Decider's suggested direction in each observation.
- Use the same training-board distribution, episode horizon, reward, update
  settings, and number of environment interactions. Measure wall time and cost
  separately; advice has additional inference overhead.
- Match context/history handling between A and B. The student always samples
  its own action, conditioned on the observation it actually received.
- Include adviser inference in the compute accounting. First test whether the
  adviser and learner fit together; do not assume a second GPU is needed or
  that the original one-model memory estimate still applies.
- If integration or cost exceeds the remaining allowance, retain the completed
  local Decider benchmark and finish the core SFT-to-RL experiment.

Checkpoint: a comparison measures a clearly defined intervention. One training
seed per arm is an exploratory result; repeat with another seed if budget allows.

## 7. Evaluate and explain — approximately 2 hours

- Begin with 10 validation games per checkpoint for debugging. Use a separate
  fixed list of 30 test seeds for the final report, with a 3,000-decision cap per
  game. Report capped games explicitly.
- Evaluate random-legal, fixed-priority heuristic, solver teacher, untouched
  Instruct, SFT, and SFT-plus-RL policies with consistent rules and action parsing.
- For the adviser comparison, evaluate both trained checkpoints with advice on
  and off. Include the frozen SFT checkpoint with advice as a control for the
  direct inference-time benefit of advice.
- Report mean and median score, largest-tile distribution, fraction reaching
  2048, invalid-action rate, decision latency, GPU hours, and adviser overhead.
- Use paired comparisons on the same test-seed list. Report evaluation variation
  separately from the unmeasured variation between training seeds.
- Save all game traces, configurations, adapter checkpoints, plots, and commands.
- Produce a terminal replay and a short written account of findings, limitations,
  and the next experiment. Reaching 2048 consistently is a stretch outcome.

Checkpoint: another person can reload the checkpoint and reproduce evaluation.
An unsuccessful RL improvement or an unhelpful adviser is still a valid finding.

## Learning checkpoints

Before each major run, explain:

1. SFT: what counts as a target token, and which weights can change?
2. RL: what is a rollout, a reward, a group-relative advantage, and a policy update?
3. Evaluation: could increasing the training reward leave full-game play worse?
4. Adviser: did the student improve, or did advice improve the combined system?

For DPO, read the objective and implement a CPU toy calculation with preferred
and rejected response log probabilities. Inspect the reference-model ratios and
beta. No rented DPO run is part of this plan.

## Budget and schedule

- Working estimate: 12–16 focused hours for the core experiment; integration
  problems or a repeated adviser comparison can exceed that estimate.
- Proposed first experiment spending ceiling: $40, with a $20 review point.
  This is a planning recommendation, not user approval to purchase resources.
- Reference price checked October 24, 2026: Runpod lists a 24 GB RTX 4090 at
  $0.89/hour. Ten compute hours would be $8.90 before storage, tax, and other
  charges. Verify the offered instance price and availability before launch.
- Measure throughput and projected completion cost before a longer job. Account
  for setup, evaluation, retries, idle rental time, and persisted storage.
- Proposed compute allocation: 20% baseline/SFT, 40% core RL, 20% evaluation,
  20% adviser pilot or a justified rerun. Preserve the evaluation allocation.

## Immediate next implementation after local preparation

Review the completed local artifacts and `GPU_HANDOFF.md`. Implement and verify
the SFT trainer setup on the user's rented GPU, beginning with the untouched
student baseline and a 20-example overfit check. The local phase has stopped
before this work and before any rented resource is provisioned.

## Sources

- [Original 2048 implementation](https://github.com/gabrielecirulli/2048)
- [Open expectimax solver](https://github.com/nneonneo/2048-ai)
- [Student model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [TRL SFT](https://huggingface.co/docs/trl/en/sft_trainer)
- [TRL GRPO](https://huggingface.co/docs/trl/en/grpo_trainer)
- [TRL stateful environments](https://huggingface.co/docs/trl/en/openenv)
- [Strands Decider model and limitations](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610)
- [DPO objective](https://huggingface.co/docs/trl/en/dpo_trainer)
- [Runpod pricing](https://www.runpod.io/pricing)
