# Progress

## Repository initialized

- Created a separate local and private GitHub repository.
- Copied the agreed weekend plan; the README's seven-step numbering is the
  execution sequence (the original plan groups some of these activities).
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
