# Local vLLM Metal serving

This serves the saved 2048 model on the Mac GPU using vLLM's Metal plugin and
its MLX execution backend. The original FP32 master checkpoint stays unchanged.
The server loads a separate BF16 serving copy. This is not GGUF or a quantized
export.

## Install

Use the [official stable Homebrew installation](https://docs.vllm.ai/projects/vllm-metal/en/latest/installation/):

```sh
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
```

The formula installs a private Python environment, independent of the project's
training environments. Installed for this experiment: Python 3.12.16,
vLLM 0.32.0+cpu, vLLM Metal 0.32.0, MLX 0.32.2, mlx-lm 0.32.0,
PyTorch 2.13.0, and Transformers 5.19.0. The `+cpu` tag names the upstream macOS
wheel; the Metal plugin supplies GPU execution.
The core engine can also print `device_config=cpu`; the relevant execution log
is `MLX device set to: Device(gpu, 0)` and native paged-attention Metal kernels.

## Prepare the BF16 serving copy

The installed vLLM Metal 0.32.0 plain MLX-LM loader preserves checkpoint tensor
dtypes. Its upstream `--dtype bfloat16` configuration does not cast FP32
safetensors weights, so using the FP32 master directly would keep FP32 weights
with BF16 KV cache. Create an explicit serving copy instead:

```sh
.venv-trl-mps/bin/python scripts/export_bf16_serving.py
```

This reads `checkpoints/local-mps-full-v1/merged-fp32` and writes
`checkpoints/local-mps-full-v1/serving-bf16`. It refuses existing outputs,
verifies every saved tensor against its BF16 cast, checks every tensor header,
and verifies that SHA-256 hashes of all original master files are unchanged.
The conversion manifest is `results/local-mps-full-v1/serving-bf16-export.json`.
BF16 rounding can change predictions; no exact behavioral parity is claimed.

## Start and stop

From the project root:

```sh
sh scripts/serve_local_vllm.sh
```

The API binds to `127.0.0.1:8000` and exposes model name `2048-grpo`. It loads
`checkpoints/local-mps-full-v1/serving-bf16`, uses one concurrent sequence,
an 8,192-token context, `--gpu-memory-utilization 0.30`, and an explicit
1,024-block KV override (448 MiB for this model's BF16 GQA cache). The installed
plugin sizes cache through its Metal planner, which ignored the core
`--kv-cache-memory-bytes` setting in the first launch. The server is offline for
Hugging Face access and disables usage telemetry. Installation itself downloads
the runtime and dependencies.

Use Ctrl-C to stop a foreground server. For a background server:

```sh
nohup sh scripts/serve_local_vllm.sh > logs/vllm-metal-grpo.log 2>&1 < /dev/null &
echo $!
```

Keep that PID and use `kill <PID>` to stop it. The startup script refuses an
occupied port before loading weights. Run only one model server at a time on
this 24 GB Mac; stop it and wait for its engine child to exit before switching
models. This setup does not register a login service or expose the API to other
computers.

```sh
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

`PORT`, `MODEL_DIR`, and `SERVED_MODEL_NAME` can be set before starting the script.
For an SFT comparison, merge the preserved SFT adapter into a separate directory
first, then serve that directory with the same options. Never overwrite the
GRPO export or compare different serving dtypes without recording the change.

## Verify one move

```sh
.venv/bin/python scripts/check_local_serving.py \
  --output results/local-mps-full-v1/serving-check.json
```

This sends the first validation board using the project's actual `SYSTEM`,
`TOOLS`, and saved chat template. It stores the full request and response,
including returned tool calls, completion finish reason, and usage. The script
refuses to overwrite an existing output file. It does not inspect the test set.

Hermes automatic tool parsing matches
[Qwen2.5's tool-call format](https://docs.vllm.ai/en/latest/features/tool_calling/#qwen-models).
The request uses `tool_choice=auto`, greedy decoding, repetition penalty 1,
and an explicit 128-token output budget. No required-tool grammar, repair,
or retry hides invalid replies. `--generation-config vllm` prevents the base
checkpoint's sampling defaults from silently changing those settings.

One successful move validates connectivity and parsing only. It establishes
neither backend numerical parity nor held-out gameplay quality. Formal SFT
versus GRPO evaluation must use identical prompts, parsing, dtype, output
budgets, and game seeds; log malformed replies and early terminations as
observations rather than silently fixing them.

## Verification recorded on October 26, 2026

The active server is API PID `98302`, engine PID `98311`, with startup log
`logs/vllm-metal-grpo-bf16.log`. These PIDs identify this launch and change on
restart. Stop this launch with `kill 98302`.

The log confirms `Device(gpu, 0)`, the `serving-bf16` checkpoint, native Metal
paged-attention kernels, and **1,024 cache blocks / 0.44 GiB** across 28 layers.
All 338 checkpoint tensor headers are BF16; the saved tensor payload totals
3,087,428,608 bytes. The FP32 master's file hashes were unchanged after conversion.

`/health` and `/v1/models` succeeded. One unconstrained validation request
returned `move({"direction": "UP"})`, a legal move, with finish reason
`tool_calls` and 19 completion tokens from a 128-token allowance. Its observed
request duration was 1.03 seconds; this single request is not a throughput
benchmark or an accuracy estimate. The full record is
`results/local-mps-full-v1/serving-check.json`.

The first launch's validation is retained separately as
`serving-check-initial-fp32-weights.json`. That launch used preserved FP32 weights
despite requesting BF16 upstream, and its actual cache was 4.47 GiB despite
requesting 512 MiB via the ignored byte-budget option. It was stopped before
the final BF16 server started. Those two requests should not be treated as a
controlled model-quality comparison.
