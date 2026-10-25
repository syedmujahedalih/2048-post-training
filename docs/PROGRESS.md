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

## Decider checkpoint in progress

- Source and checkpoint versions recorded; native MLX serving works on the M4.
- 128-board afterstate diagnostic completed; full-game pilot is running.
- Final findings and the GPU-session handoff will be recorded after gameplay.
