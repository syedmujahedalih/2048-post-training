"""Bounded TRL/PEFT compatibility checks on MPS, with CPU operator fallback disabled.

Defaults to a tiny random Qwen2 architecture; --student uses the real checkpoint.
SFT uses actual 2048 demonstrations; GRPO uses a synthetic token reward. This is
a device/trainer check, not a trained 2048 student or an RL performance result.
"""

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import time
import traceback
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
    TrainerCallback,
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def tiny_model(tokenizer):
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=1024,
            tie_word_embeddings=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attention_dropout=0.0,
            attn_implementation="eager",
        )
    )


class CheckGradients(TrainerCallback):
    def __init__(self):
        self.observations = []

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.observations.append(
            {
                "devices": sorted({str(g.device) for g in gradients}),
                "finite": all(torch.isfinite(g).all().item() for g in gradients),
                "nonzero_tensors": sum(torch.count_nonzero(g).item() > 0 for g in gradients),
            }
        )


def execute(stage, tokenizer, rows, folder, student_revision=None):
    set_seed(42)
    model = (
        AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            revision=student_revision,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        if student_revision
        else tiny_model(tokenizer)
    )
    peft = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    callback = CheckGradients()
    common = {
        "output_dir": str(folder / stage),
        "max_steps": 1,
        "learning_rate": 1e-3,
        "gradient_accumulation_steps": 1,
        "bf16": bool(student_revision),
        "fp16": False,
        "gradient_checkpointing": False,
        "optim": "adamw_torch",
        "save_strategy": "no",
        "report_to": "none",
        "logging_steps": 1,
        "disable_tqdm": True,
        "dataloader_num_workers": 0,
        "seed": 42,
    }
    rewards_seen = []
    if stage == "sft":
        samples = []
        for row in rows:
            prompt_ids = tokenizer.apply_chat_template(
                row["messages"][:-1],
                tools=row["tools"],
                add_generation_prompt=True,
                return_dict=False,
            )
            full_ids = tokenizer.apply_chat_template(
                row["messages"], tools=row["tools"], return_dict=False
            )
            assert full_ids[: len(prompt_ids)] == prompt_ids
            samples.append(
                {
                    "input_ids": full_ids,
                    "completion_mask": [0] * len(prompt_ids)
                    + [1] * (len(full_ids) - len(prompt_ids)),
                }
            )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            peft_config=peft,
            args=SFTConfig(
                **common, per_device_train_batch_size=1, max_length=512, completion_only_loss=True
            ),
            train_dataset=Dataset.from_list(samples),
            callbacks=[callback],
        )
        batch = next(iter(trainer.get_train_dataloader()))
        labels = batch["labels"]
        mask = {
            "ignored_tokens": int((labels == -100).sum()),
            "target_tokens": int((labels != -100).sum()),
        }
        assert mask["ignored_tokens"] > 0 and mask["target_tokens"] > 0
        assert mask["target_tokens"] == 20, "Expected exact audited move-call target tokens"
    else:
        prompts = [
            tokenizer.apply_chat_template(
                row["messages"][:-1], tools=row["tools"], tokenize=False, add_generation_prompt=True
            )
            for row in rows[:2]
        ]

        def synthetic_reward(completion_ids, **kwargs):
            values = [float(sum(ids) % 13) / 12 for ids in completion_ids]
            rewards_seen.append(values)
            return values

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            peft_config=peft,
            args=GRPOConfig(
                **common,
                per_device_train_batch_size=4,
                num_generations=4,
                max_completion_length=8,
                beta=0.0,
                loss_type="grpo",
                use_vllm=False,
                mask_truncated_completions=False,
            ),
            reward_funcs=synthetic_reward,
            train_dataset=Dataset.from_dict({"prompt": prompts}),
            callbacks=[callback],
        )
        mask = None
    assert trainer.accelerator.device.type == "mps", trainer.accelerator.device
    before = {
        name: p.detach().cpu().clone()
        for name, p in trainer.model.named_parameters()
        if p.requires_grad
    }
    start = time.perf_counter()
    result = trainer.train()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    after = {
        name: p.detach().cpu().clone()
        for name, p in trainer.model.named_parameters()
        if p.requires_grad
    }
    delta = max((after[name] - value).abs().max().item() for name, value in before.items())
    assert math.isfinite(result.training_loss)
    assert delta > 0, "No trainable weights changed"
    assert callback.observations, "No optimizer callback observed"
    for observed in callback.observations:
        assert (
            observed["devices"] == ["mps:0"]
            and observed["finite"]
            and observed["nonzero_tensors"] > 0
        )
    if rewards_seen:
        assert any(len(set(group)) > 1 for group in rewards_seen), "No group reward variation"
    checkpoint = folder / f"{stage}-adapter"
    trainer.model.save_pretrained(checkpoint)
    reloaded_weights = load_file(checkpoint / "adapter_model.safetensors")
    expected = get_peft_model_state_dict(trainer.model)
    assert expected.keys() == reloaded_weights.keys()
    assert all(
        torch.equal(value.detach().cpu(), reloaded_weights[name])
        for name, value in expected.items()
    )
    return {
        "status": "passed",
        "device": str(trainer.accelerator.device),
        "training_seconds": elapsed,
        "training_loss": result.training_loss,
        "max_adapter_weight_change": delta,
        "gradients": callback.observations,
        "sft_loss_mask": mask,
        "synthetic_grpo_rewards": rewards_seen,
        "adapter_roundtrip_exact": True,
        "model_parameter_count": sum(p.numel() for p in trainer.model.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        ),
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/mps-trl-check.json")
    parser.add_argument(
        "--student",
        action="store_true",
        help="Use the real pretrained student (weights must be cached)",
    )
    args = parser.parse_args()
    assert torch.backends.mps.is_available(), "MPS is unavailable"
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets")
        },
        "cpu_operator_fallback": False,
        "model": "Qwen/Qwen2.5-1.5B-Instruct"
        if args.student
        else "tiny random Qwen2: 1 layer, width 32; not the pretrained student",
        "stages": {},
    }
    tokenizer_revision = json.loads((ROOT / "results/tokenizer-check.json").read_text())[
        "tokenizer_revision"
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", revision=tokenizer_revision, local_files_only=True
    )
    revision = tokenizer_revision if args.student else None
    report["tokenizer_revision"] = tokenizer_revision
    report["student_revision"] = revision
    rows = [
        json.loads(line)
        for line in (ROOT / "data/demonstrations/train.jsonl").read_text().splitlines()[:20]
    ]
    with tempfile.TemporaryDirectory(prefix="trl-mps-check-") as path:
        for stage in ("sft", "grpo"):
            print(f"Starting {stage} on MPS", flush=True)
            try:
                report["stages"][stage] = execute(
                    stage, tokenizer, rows, Path(path), student_revision=revision
                )
            except Exception as error:
                report["stages"][stage] = {
                    "status": "failed",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                traceback.print_exc()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["stages"][stage]), flush=True)
            gc.collect()  # Reclaim unreachable Python objects, including reference cycles.
            torch.mps.empty_cache()  # Release unused MPS cache blocks, not live tensors.
    print(f"Report: {args.output}", flush=True)
    if any(stage["status"] != "passed" for stage in report["stages"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
