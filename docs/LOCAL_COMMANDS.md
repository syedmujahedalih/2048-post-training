# Local reproduction

Run commands from the root of this repository. The core package needs Python
3.11, uv, a C++ compiler, and Node.js for the original-game parity test.

## Simulator and teacher

```bash
uv sync --locked
uv run python scripts/build_teacher.py
uv run pytest -q
uv run posttrain2048 play --agent teacher --seed 42 --output results/teacher-demo.json
uv run posttrain2048 replay results/teacher-demo.json --verbose
```

`replay` checks every saved board, spawn, action validity, and score against a
fresh seeded simulation. It does not call the policy again.

## Data

```bash
uv run python scripts/audit_data.py
uv run python scripts/export_sft.py
```

These commands audit and export the committed dataset. To regenerate it, use a
new directory as shown in `DATASET.md`; the generator refuses overwriting.

## Frozen Decider on Apple silicon

The exact working Python environment is recorded separately from the core
package. It includes MLX, the pinned Decider source revision, and the inference
dependencies. No model weights are in Git.
Keep the chosen `DECIDER_PORT` in an ignored `.local/decider.env` file.

```bash
uv venv --python 3.11 .venv-decider
uv pip install --python .venv-decider/bin/python -r docs/decider-macos-requirements.txt
. .local/decider.env
.venv-decider/bin/strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device mlx --port "$DECIDER_PORT"
```

The first load downloads the adapter and its Qwen3.5 base weights from Hugging
Face. In another terminal:

```bash
. .local/decider.env
uv run python scripts/benchmark_decider.py --endpoint "http://127.0.0.1:$DECIDER_PORT" --output results/my-decider-pilot
```

The benchmark sends requests only to the configured local server. It writes
raw board predictions, traces, summaries, and timings incrementally. Six games
are an exploratory pilot, not a reliable estimate of a win rate.

`directions` mode supplies action names/descriptions; `afterstates` mode also
supplies deterministic resulting boards and immediate merge scores. The latter
uses the simulator as a known transition model. Neither mode is given teacher
scores or future random tiles. Gameplay passes only legal options to every
agent. The 128-board diagnostic passes all four options to measure invalid choices.

The committed results record the exact checkpoint revision and code revision in
`results/decider-provenance.json`. To reproduce a historical run after upstream
changes `main`, first download that recorded revision and pass its local folder
to the server instead of the moving Hub ID.

## Student tokenizer, without downloading student weights

With the Decider environment installed:

```bash
.venv-decider/bin/python scripts/check_tokenizer.py
```

This validates the student chat template and completion-loss boundary. It does
not run or train the student model.
