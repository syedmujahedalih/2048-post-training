# Local training in progress

The authorized sequence runs independently of chat turns through
`scripts/run_full_pipeline.py` on Apple M4 / MPS, using the pinned environment
in `mps-trl-requirements.txt`. No rented GPU or paid service is used.

1. **Completed:** 100 real 2048 GRPO updates, starting from the saved
   20-example SFT warmup adapter. Four trajectories per starting board, capped
   at 16 decisions. This is not the adapter trained on the full dataset.
2. **Running:** one epoch over all 8,000 SFT examples, batch size 4 and 2,000
   optimizer updates, starting afresh from the original pretrained student.
3. **Queued:** 100 GRPO updates starting from that full SFT adapter.

The [first run's report and plots](FIRST_LOCAL_GRPO.md) include all 100 updates
and replay-verified trajectories.

Student: `Qwen/Qwen2.5-1.5B-Instruct`, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. LoRA rank 4 on q/v projections,
bfloat16, gradient checkpointing, native Transformers generation, no CPU
operator fallback. The MPS allocator uses its recommended working-set limit.

The game environment exposes only `move`. It uses independent seeded spawns
for each trajectory. Rewards are `log1p(score gained) - 0.1 * invalid attempts`;
invalid attempts include no-op moves, tool failures, and a non-action reply
before the episode ends. A rollout can end before 16 decisions if the model
stops calling tools. Model actions are not replaced by a teacher or adviser.
Simulator observations are masked from the loss; a final unexecuted tool call
at the turn cap is also masked. Episode traces and mask audits are saved.

## Progress and checkpoints

- `results/local-training-status.json`: live pipeline state and process ID
  (local only, ignored by Git).
- `logs/grpo-start-now.log`, `logs/sft-all-8000.log`, and
  `logs/grpo-after-full-sft.log`: streamed phase progress.
- `results/local-mps-grpo-now/`: first GRPO metrics, traces, and final report.
- `results/local-mps-full-v1/`: full SFT and post-SFT GRPO results.
- `checkpoints/local-mps-grpo-now/` and `checkpoints/local-mps-full-v1/`:
  local adapter and optimizer checkpoints, excluded from Git.

GRPO checkpoints every five updates; full SFT checkpoints every 100 updates.
The pipeline stops on a real process failure. Restarting it skips completed
phases and resumes available trainer checkpoints. The existing thread heartbeat
checks hourly, stays quiet during normal progress, and handles errors or final
reporting. Final reports should include plots from recorded metrics and state
evaluation limits; training reward alone is not evidence of better gameplay.

## Garbage collection

`gc` is Python's standard-library garbage collector. `gc.collect()` reclaims
unreachable Python object graphs between phases. `torch.mps.empty_cache()`
then releases unused GPU cache blocks. Neither changes the model's algorithm
or releases tensors that still have live references.
