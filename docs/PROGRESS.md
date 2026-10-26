# Progress

## Repository initialized

- Created a separate local and private GitHub repository.
- Copied the agreed weekend plan and aligned its seven-step numbering with
  the README's execution sequence.
- Local machine: Apple M4 Mac mini, 24 GB unified memory, macOS 26.6.2.
- No cloud instances or training runs have been started.

## Simulator and solver checkpoint

- Seeded game engine, move/merge rules, score accounting, and terminal detection.
- Replay CLI verifies every saved transition and spawn.
- Bounded depth-2 native teacher with retained upstream MIT license and revision.
- Independent move/score comparisons against the original JavaScript game and
  move comparisons against the native solver.
- 28 local tests pass after adding data leakage and input checks.

## Dataset checkpoint

- Pilot: 1,000 examples, followed by 10,000 full demonstrations.
- Full split: 8,000 train / 1,000 validation / 1,000 test, from 104 whole games.
- No game, exact-board, or rotation/reflection duplicate leakage across splits.
- Explicit prompt/completion SFT export and 20-example teacher-label inspection.
- All 8,000 training examples checked with the real student tokenizer; matching
  prompt prefixes, legal rendered calls, max 361 tokens, 20 target tokens each.
- No student weights loaded and no model trained.

## Decider checkpoint

- Source and checkpoint versions recorded; native MLX serving works on the M4.
- 128-board afterstate diagnostic completed: 100% legal choices with simulator
  information provided, 35.2% agreement with the teacher's first choice.
- Six complete Decider games, compared on the same seed list with random,
  priority, greedy, and solver policies; every stored game trace replay-verified.
- Decider mean score 1,591; priority 1,487; greedy 1,985; solver 52,158.
- The pilot provides no clear evidence to justify extra adviser-assisted RL
  compute yet. Keep the results as a measured comparison and start GPU work with
  the unassisted SFT-to-RL path.
- A directions-only diagnostic checks the simpler input variant separately.
- Linux GitHub CI passed native compilation, lint/format, all 28 tests, the
  full dataset audit, and saved-game replay at commit `89f2529`.

## Handoff

- Local steps 1–3 are complete. See `GPU_HANDOFF.md` for steps 4–7.
- No paid cloud resources, student inference baseline, SFT, or RL run was started.
- This separate repository leaves the existing inference-engineering work intact.

## GRPO explainer

- Built a 4:52 narrated Remotion composition with 11 replayable chapters.
- Original teaching script, Mac Samantha narration, and measured phrase captions.
- Valid simulator rollouts, a worked mean/subtraction/normalization example,
  clipping and optional KL, credit assignment, and the proposed TRL experiment.
- Rendered and visually reviewed all chapters; corrected card overflow and
  early-terminal rollout handling. TypeScript/ESLint pass; narration lengths
  and caption boundaries match the 8,757-frame timeline.
- The preview is local. Source and transcript are versioned; generated system
  speech and review renders stay local. No student training or paid compute.

## MPS / TRL compatibility checkpoint — 2026-10-25

- Isolated `.venv-trl-mps` uses managed CPython 3.11.15, TRL 1.16.0, Torch
  2.14.1, Transformers 5.20.2, PEFT 0.21.2, Accelerate 1.15.0, Datasets 5.1.0.
- SFT and GRPO pass on a tiny Qwen2 and on the real Qwen2.5-1.5B-Instruct,
  with unsupported MPS operator fallback disabled.
- Checked nonzero finite MPS gradients, changed LoRA weights, and exact saved
  adapter tensor reload. SFT checks the audited 20-token move-call loss boundary.
- Each probe performs one update. GRPO uses four short text generations and a
  synthetic token reward; it does not play the stateful 2048 environment.
- Adapters are temporary and removed after checking. No production checkpoint,
  20-example overfit result, or improved gameplay is claimed. No paid compute.

## Visual experimental reports

- Added reproducible PNG/SVG plots for MPS timing/allocation and synthetic rewards.
- Added Decider individual-game scores and paired differences, including explicit
  markers for partial teacher games at the decision cap.
- Embedded figures in both reports and the README, with sample counts and limits.
- Recorded the reporting convention in `EXPERIMENT_WRITEUPS.md`: future reports
  should include useful plots from saved results, with regeneration commands.

## Local training launched

- Completed an 80-update, 20-example SFT warmup and verified fresh-checkpoint
  inference matches. This is recorded as a warmup, not the full SFT run.
- Added the stateful TRL game environment, seeded trajectory isolation, reward
  accounting, observation-mask audits, and unexecuted-tail masking.
- Started a detached pipeline: 100 real GRPO updates now, followed by a full
  8,000-example SFT epoch and another 100 GRPO updates from its checkpoint.
- See `LOCAL_TRAINING_RUN.md` for active logs, checkpoints, and the queue.

## First 100-update GRPO job completed

- Completed 100 updates and 400 rollouts from the 20-example SFT warmup adapter.
- Replay verified all 2,115 sampled move attempts and episode rewards.
- Saved/reloaded the adapter with identical validation predictions.
- Added `FIRST_LOCAL_GRPO.md` and a reproducible four-panel training plot.
- The pipeline automatically advanced into the full 8,000-example SFT epoch;
  the post-full-SFT GRPO job remains queued.
