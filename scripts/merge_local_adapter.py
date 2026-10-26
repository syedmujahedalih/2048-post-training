"""Export the completed adapter into an FP32 Hugging Face master checkpoint, offline."""

import gc
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from posttrain2048.data import TOOLS, read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    started = time.monotonic()
    adapter = ROOT / "checkpoints/local-mps-full-v1/grpo-adapter"
    output = ROOT / "checkpoints/local-mps-full-v1/merged-fp32"
    report_path = ROOT / "results/local-mps-full-v1/merge-report.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite an existing export or report")
    summary = json.loads((ROOT / "results/local-mps-full-v1/grpo-summary.json").read_text())
    assert summary["completed_steps"] == 100 and summary["reload_predictions_exact"]
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, adapter).eval()
    validation = read_jsonl(ROOT / "data/demonstrations/validation.jsonl")[:1]
    inputs = tokenizer.apply_chat_template(
        validation[0]["messages"][:-1],
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    with torch.inference_mode():
        before = model(**inputs, use_cache=False).logits[:, -1, :].clone()
    print("Unmerged FP32 reference logits saved", flush=True)
    model = model.merge_and_unload(safe_merge=True).eval()
    assert not any("lora_" in name for name, _ in model.named_parameters())
    with torch.inference_mode():
        after = model(**inputs, use_cache=False).logits[:, -1, :]
    difference = (before - after).abs()
    same_token = bool(torch.equal(before.argmax(-1), after.argmax(-1)))
    assert same_token and torch.allclose(before, after, atol=5e-4, rtol=1e-4)
    report = {
        "base_model": MODEL,
        "base_revision": REVISION,
        "adapter": str(adapter.relative_to(ROOT)),
        "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
        "output": str(output.relative_to(ROOT)),
        "dtype": "float32",
        "device": "cpu",
        "check": "One validation prompt; final-position logits before/after FP32 merge",
        "example_id": validation[0]["id"],
        "max_abs_logit_difference": float(difference.max()),
        "mean_abs_logit_difference": float(difference.mean()),
        "next_token_argmax_equal": same_token,
        "atol": 5e-4,
        "rtol": 1e-4,
        "limits": (
            "Checks FP32 adapter-versus-merged arithmetic, not BF16 backend parity, "
            "autoregressive behavior, or full-game quality. Serving casts need evaluation."
        ),
    }
    model.save_pretrained(output, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output)
    del model, base
    gc.collect()
    reloaded = AutoModelForCausalLM.from_pretrained(
        output, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    with torch.inference_mode():
        saved_logits = reloaded(**inputs, use_cache=False).logits[:, -1, :]
    assert torch.equal(after, saved_logits)
    report["saved_checkpoint_logits_exact"] = True
    report["elapsed_seconds"] = time.monotonic() - started
    report["files"] = {p.name: p.stat().st_size for p in sorted(output.iterdir()) if p.is_file()}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
