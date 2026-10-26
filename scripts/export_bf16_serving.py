"""Create a separate BF16 serving copy of the merged FP32 master, entirely offline."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "checkpoints/local-mps-full-v1/merged-fp32"
OUTPUT = ROOT / "checkpoints/local-mps-full-v1/serving-bf16"
REPORT = ROOT / "results/local-mps-full-v1/serving-bf16-export.json"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def manifest(directory: Path) -> dict:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("Refusing to overwrite an existing serving export or report")
    started = time.monotonic()
    torch.set_num_threads(4)
    source_before = manifest(SOURCE)
    index_name = "model.safetensors.index.json"
    index = json.loads((SOURCE / index_name).read_text())
    shards = sorted(set(index["weight_map"].values()))
    OUTPUT.mkdir()
    for path in SOURCE.iterdir():
        if path.is_file() and path.name not in {*shards, index_name, "config.json"}:
            shutil.copy2(path, OUTPUT / path.name)
    config = json.loads((SOURCE / "config.json").read_text())
    config["dtype"] = "bfloat16"
    if "torch_dtype" in config:
        config["torch_dtype"] = "bfloat16"
    (OUTPUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    total_size = 0
    tensor_count = 0
    for name in shards:
        source_tensors = load_file(SOURCE / name, device="cpu")
        if any(tensor.dtype != torch.float32 for tensor in source_tensors.values()):
            raise ValueError(f"Expected only FP32 master tensors in {name}")
        converted = {key: tensor.to(torch.bfloat16) for key, tensor in source_tensors.items()}
        save_file(converted, OUTPUT / name, metadata={"format": "pt"})
        saved = load_file(OUTPUT / name, device="cpu")
        assert saved.keys() == source_tensors.keys()
        for key, tensor in saved.items():
            assert tensor.dtype == torch.bfloat16
            assert tensor.shape == source_tensors[key].shape
            assert torch.equal(tensor, converted[key])
            total_size += tensor.numel() * tensor.element_size()
            tensor_count += 1
        with safe_open(OUTPUT / name, framework="pt", device="cpu") as reader:
            assert all(reader.get_slice(key).get_dtype() == "BF16" for key in reader.keys())
        print(f"Converted and verified {name}", flush=True)
        del source_tensors, converted, saved
        gc.collect()

    index["metadata"]["total_size"] = total_size
    (OUTPUT / index_name).write_text(json.dumps(index, indent=2) + "\n")
    source_after = manifest(SOURCE)
    assert source_before == source_after, "Original master checkpoint changed"
    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "source_dtype": "float32",
        "output_dtype": "bfloat16",
        "conversion_device": "cpu",
        "conversion": "FP32 to BF16 cast; no quantization, retraining, or adapter merge",
        "tensor_count": tensor_count,
        "total_tensor_bytes": total_size,
        "all_saved_tensors_match_cast_exactly": True,
        "all_saved_tensor_headers_bf16": True,
        "source_files_unchanged_sha256": source_before == source_after,
        "source_manifest": source_before,
        "output_manifest": manifest(OUTPUT),
        "torch_version": torch.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "limits": (
            "BF16 rounding changes the FP32 master. Exact tensor casting and source "
            "preservation do not establish MPS/MLX numerical or gameplay parity."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
