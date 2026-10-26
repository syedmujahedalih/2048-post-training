"""Export the pinned original instruct checkpoint as verified BF16, offline."""

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
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
OUTPUT = ROOT / "checkpoints/local-mps-full-v1/base-serving-bf16"
REPORT = ROOT / "results/local-mps-full-v1/base-serving-bf16-export.json"


def manifest(directory):
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        result[path.name] = {"bytes": path.stat().st_size, "sha256": h.hexdigest()}
    return result


def main():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("Refusing to overwrite an existing original-model export or report")
    started = time.monotonic()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    assert not any("lora_" in name for name, _ in model.named_parameters())
    model.save_pretrained(OUTPUT, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(OUTPUT)
    expected = model.state_dict()
    expected_names = set(expected)
    if model.config.tie_word_embeddings:
        assert (
            model.get_input_embeddings().weight.data_ptr()
            == model.get_output_embeddings().weight.data_ptr()
        )
        expected_names.remove("lm_head.weight")
    saved_names = set()
    total_bytes = 0
    for path in sorted(OUTPUT.glob("*.safetensors")):
        tensors = load_file(path, device="cpu")
        with safe_open(path, framework="pt", device="cpu") as reader:
            assert all(reader.get_slice(key).get_dtype() == "BF16" for key in reader.keys())
        for key, tensor in tensors.items():
            assert tensor.dtype == torch.bfloat16 and torch.equal(tensor, expected[key])
            assert key not in saved_names
            saved_names.add(key)
            total_bytes += tensor.numel() * tensor.element_size()
        del tensors
        gc.collect()
    assert saved_names == expected_names
    output_manifest = manifest(OUTPUT)
    reference = json.loads(
        (ROOT / "results/local-mps-full-v1/serving-bf16-export.json").read_text()
    )
    for name in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
        assert output_manifest[name]["sha256"] == reference["output_manifest"][name]["sha256"]
    report = {
        "base_model": MODEL,
        "base_revision": REVISION,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_dtype": "bfloat16",
        "export_device": "cpu",
        "conversion": "Original pinned instruct weights; no adapter or additional training",
        "tensor_count": len(saved_names),
        "total_tensor_bytes": total_bytes,
        "all_saved_tensors_match_original_bf16_exactly": True,
        "all_saved_tensor_headers_bf16": True,
        "tokenizer_and_chat_template_match_grpo": True,
        "output_manifest": output_manifest,
        "torch_version": torch.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "limits": "Exact BF16 serialization does not establish equality across inference backends.",
    }
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
