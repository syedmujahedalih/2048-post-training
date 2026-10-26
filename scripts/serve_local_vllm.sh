#!/bin/sh
# Serve the existing merged checkpoint on this Mac; do not modify its weights.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL_DIR=${MODEL_DIR:-"$ROOT/checkpoints/local-mps-full-v1/serving-bf16"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-2048-grpo}
PORT=${PORT:-8000}

if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "Missing merged Hugging Face checkpoint: $MODEL_DIR" >&2
    exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
    echo "Install vLLM Metal with the official Homebrew formula first; see docs/LOCAL_SERVING.md." >&2
    exit 1
fi
if /usr/sbin/lsof -nP "-iTCP:$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $PORT already has a listener; refusing to load a second model." >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export VLLM_MLX_DEVICE=gpu

# The installed MLX loader preserves unquantized checkpoint tensor dtypes;
# --dtype alone does not cast FP32 master weights. Load the separate BF16 copy.
# A 1,024-block override bounds actual BF16 GQA cache allocation to 448 MiB.
exec vllm serve "$MODEL_DIR" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 512 \
    --gpu-memory-utilization 0.30 \
    --num-gpu-blocks-override 1024 \
    --generation-config vllm \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
