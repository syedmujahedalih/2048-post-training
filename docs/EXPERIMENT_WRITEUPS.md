# Experimental write-up conventions

Reports should make the evidence visible alongside the finding. Include at
least one useful plot or diagram when the recorded data supports it.

- Generate measured plots from saved raw results, and commit the generator.
- Show units, model/backend, workload, and sample count. Include individual
  observations for small pilots and paired differences for paired experiments.
- State when observations are capped, truncated, synthetic, or derived.
- Keep snapshots, peak memory, and total system memory distinct. Do not turn a
  single timing observation into a throughput claim or uncertainty interval.
- Use learning curves only when multiple training checkpoints were recorded.
  One-update compatibility checks should show timing, memory, or learning-signal
  diagnostics instead. Different objectives' loss values are not directly comparable.
- Put a short interpretation and the practical limits beside each figure.
- Preserve the raw results and environment details, and provide the regeneration
  command. Use PNG for embedded previews and SVG for editable/scalable figures.

## Current figures

- [MPS/TRL check](MPS_TRL_CHECK.md): observed update times, memory snapshots,
  and the actual synthetic GRPO reward group.
- [Decider pilot](DECIDER_REPORT.md): all individual game scores, capped teacher
  outcomes, and six paired Decider-minus-priority differences.

```sh
uv run --managed-python --script scripts/plot_experiments.py
```

The script's dependency versions are pinned inline. It reads existing results
and writes `docs/figures/*.png` and `docs/figures/*.svg`; it performs no training
or model inference.
