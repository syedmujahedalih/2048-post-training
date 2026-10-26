"""Bounded SFT warmup and stateful 2048 GRPO through TRL on Apple MPS.

Run sft --steps 80, then grpo --steps 10 --horizon 16. Weights stay in
ignored checkpoints/; measurements and trajectories are saved under results/.
"""

import argparse
import gc
import importlib.metadata
import itertools
import json
import math
import os
import random
import re
import time
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.0"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.9"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed  # noqa: E402
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer  # noqa: E402

from posttrain2048.data import SYSTEM, TOOLS, read_jsonl  # noqa: E402
from posttrain2048.game import ACTIONS  # noqa: E402
from posttrain2048.rl_env import Game2048Env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def append_json(path, data):
    with path.open("a") as stream:
        stream.write(json.dumps(data, allow_nan=False) + "\n")


def load_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def parse_move(text):
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S)
    if len(matches) == 1:
        try:
            call = json.loads(matches[0])
            action = call["arguments"]["direction"]
            if call["name"] == "move" and action in ACTIONS:
                return action
        except (KeyError, TypeError, ValueError):
            pass
    return None


def evaluate(model, tokenizer, rows):
    model.eval()
    predictions = []
    padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    for start in range(0, len(rows), 4):
        batch = rows[start : start + 4]
        prompts = [
            tokenizer.apply_chat_template(
                row["messages"][:-1], tools=TOOLS, tokenize=False, add_generation_prompt=True
            )
            for row in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("mps")
        with torch.inference_mode(), torch.autocast("mps", dtype=torch.bfloat16):
            generated = model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        texts = tokenizer.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        for row, text in zip(batch, texts, strict=True):
            action = parse_move(text)
            predictions.append(
                {
                    "id": row["id"],
                    "action": action,
                    "teacher_move": row["teacher_move"],
                    "legal": action in row["legal_moves"],
                    "teacher_match": action == row["teacher_move"],
                    "text": text,
                }
            )
    tokenizer.padding_side = padding
    model.train()
    return {
        "examples": len(predictions),
        "legal_rate": sum(row["legal"] for row in predictions) / len(predictions),
        "teacher_agreement": sum(row["teacher_match"] for row in predictions) / len(predictions),
        "predictions": predictions,
    }


class RunLog(TrainerCallback):
    def __init__(self, directory, phase, max_seconds):
        self.directory, self.phase, self.max_seconds = directory, phase, max_seconds
        self.gradients = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.started = time.perf_counter()
        self.before = {
            name: p.detach().cpu().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(
            torch.isfinite(g).all().item() and g.device.type == "mps" for g in gradients
        )
        self.gradients.append(
            {
                "step": state.global_step + 1,
                "finite": True,
                "nonzero_tensors": sum(torch.count_nonzero(g).item() > 0 for g in gradients),
            }
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        values = {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in (logs or {}).items()
        }
        append_json(
            self.directory / f"{self.phase}-metrics.jsonl", {"step": state.global_step, **values}
        )

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.perf_counter() - self.started
        allocated = torch.mps.driver_allocated_memory()
        append_json(
            self.directory / f"{self.phase}-resources.jsonl",
            {"step": state.global_step, "elapsed_seconds": elapsed, "mps_driver_bytes": allocated},
        )
        print(
            f"{self.phase}: update {state.global_step}/{state.max_steps}; {elapsed:.1f}s; MPS driver {allocated / 2**30:.2f} GiB",
            flush=True,
        )
        if self.max_seconds > 0 and elapsed > self.max_seconds:
            control.should_training_stop = True

    def weight_change(self, model):
        delta = max(
            (p.detach().cpu() - self.before[name]).abs().max().item()
            for name, p in model.named_parameters()
            if p.requires_grad
        )
        assert delta > 0
        return delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["sft", "grpo"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--examples", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", default="local-mps-pilot")
    parser.add_argument("--sft-run-name", default="local-mps-pilot")
    parser.add_argument("--max-seconds", type=int, default=0)
    args = parser.parse_args()
    steps = args.steps or (80 if args.phase == "sft" else 10)
    run = ROOT / "results" / args.run_name
    checkpoints = ROOT / "checkpoints" / args.run_name
    sft_adapter = ROOT / "checkpoints" / args.sft_run_name / "sft-adapter"
    run.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    if (run / f"{args.phase}-metrics.jsonl").exists() and not args.resume:
        raise FileExistsError("Refusing to mix runs; choose a new --run-name")
    if args.phase == "grpo" and not sft_adapter.exists():
        raise FileNotFoundError("Run the SFT warmup first")
    assert torch.backends.mps.is_available()
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    training = read_jsonl(ROOT / "data/demonstrations/train.jsonl")
    validation = random.Random(42).sample(
        read_jsonl(ROOT / "data/demonstrations/validation.jsonl"), 12
    )
    callback = RunLog(run, args.phase, args.max_seconds)
    common = {
        "output_dir": str(checkpoints / f"{args.phase}-trainer"),
        "max_steps": steps,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_torch",
        "report_to": "none",
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": 100 if args.phase == "sft" else 5,
        "save_total_limit": 2,
        "disable_tqdm": True,
        "seed": 42,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
    }
    peft = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = load_model()
    report = {
        "model": MODEL,
        "revision": REVISION,
        "phase": args.phase,
        "requested_steps": steps,
        "horizon": args.horizon if args.phase == "grpo" else None,
        "versions": {
            key: importlib.metadata.version(key) for key in ("torch", "trl", "transformers", "peft")
        },
        "cpu_operator_fallback": False,
        "lora_rank": 4,
        "batch_size": args.batch_size if args.phase == "sft" else 1,
        "configured_examples": args.examples if args.phase == "sft" else 32,
    }
    if args.phase == "sft":
        examples = random.Random(42).sample(training, args.examples)
        tokenized = []
        for row in examples:
            prompt = tokenizer.apply_chat_template(
                row["messages"][:-1], tools=TOOLS, add_generation_prompt=True, return_dict=False
            )
            full = tokenizer.apply_chat_template(row["messages"], tools=TOOLS, return_dict=False)
            assert full[: len(prompt)] == prompt
            tokenized.append(
                {
                    "input_ids": full,
                    "completion_mask": [0] * len(prompt) + [1] * (len(full) - len(prompt)),
                }
            )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            peft_config=peft,
            args=SFTConfig(
                **common,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                max_length=512,
                completion_only_loss=True,
            ),
            train_dataset=Dataset.from_list(tokenized),
            callbacks=[callback],
        )
        report["training_ids"] = [row["id"] for row in examples]
        report["before_validation"] = evaluate(trainer.model, tokenizer, validation)
        report["before_training_boards"] = evaluate(trainer.model, tokenizer, examples[:20])
        write_json(run / "sft-before.json", report)
    else:
        ids = itertools.count()
        current_episodes = []
        state_context = {}

        def factory():
            return Game2048Env(instance_id=next(ids), horizon=args.horizon)

        def game_reward(completions, environments, **kwargs):
            current_episodes.clear()
            for completion, environment in zip(completions, environments, strict=True):
                episode = environment._finish(completion)
                episode["update_before_generation"] = state_context["trainer"].state.global_step
                current_episodes.append(episode)
                append_json(run / "grpo-episodes.jsonl", episode)
            print(
                "rollout group: "
                + str(
                    [
                        (e["score_gain"], len(e["attempts"]), round(e["reward"], 3))
                        for e in current_episodes
                    ]
                ),
                flush=True,
            )
            return [episode["reward"] for episode in current_episodes]

        class AuditedGRPO(GRPOTrainer):
            def _generate_and_score_completions(self, inputs):
                output = super()._generate_and_score_completions(inputs)
                masks = output.get("tool_mask")
                assert masks is not None, "Native tool observation masks are missing"
                audit = []
                for index, episode in enumerate(current_episodes):
                    mask = masks[index]
                    observed_tokens = int(
                        ((mask == 0) & output["completion_mask"][index].bool()).sum().item()
                    )
                    assert not episode["attempts"] or observed_tokens > 0
                    ignored_tail = 0
                    if episode["unexecuted_tail_calls"] > 0:
                        zeros = torch.nonzero(mask == 0).flatten()
                        tail_start = int(zeros[-1].item()) + 1 if len(zeros) else 0
                        ignored_tail = int(
                            (mask[tail_start:] * output["completion_mask"][index][tail_start:])
                            .sum()
                            .item()
                        )
                        mask[tail_start:] = 0
                    audit.append(
                        {
                            "example_id": episode["example_id"],
                            "environment_tokens_masked": observed_tokens,
                            "unexecuted_tail_tokens_masked": ignored_tail,
                            "policy_tokens": int(
                                (mask * output["completion_mask"][index]).sum().item()
                            ),
                            "advantage": float(output["advantages"][index].item()),
                        }
                    )
                append_json(
                    run / "grpo-mask-audit.jsonl",
                    {"step": self.state.global_step, "rollouts": audit},
                )
                return output

        examples = random.Random(42).sample(
            [row for row in training if row["stage"] == "early"], 32
        )
        dataset = Dataset.from_list(
            [
                {
                    "prompt": [{"role": "system", "content": SYSTEM}, row["messages"][1]],
                    "initial_board": row["board"],
                    "example_id": row["id"],
                    "episode_seed": 900000 + index,
                }
                for index, row in enumerate(examples)
            ]
        )
        trainer = AuditedGRPO(
            model=model,
            processing_class=tokenizer,
            peft_config=peft,
            environment_factory=factory,
            reward_funcs=game_reward,
            train_dataset=dataset,
            callbacks=[callback],
            args=GRPOConfig(
                **common,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_generations=4,
                learning_rate=2e-5,
                beta=0.0,
                loss_type="grpo",
                use_vllm=False,
                max_completion_length=3072,
                max_tool_calling_iterations=args.horizon,
                generation_kwargs={"max_new_tokens": 48},
                mask_truncated_completions=False,
                remove_unused_columns=False,
            ),
        )
        set_peft_model_state_dict(
            trainer.model, load_file(sft_adapter / "adapter_model.safetensors")
        )
        state_context["trainer"] = trainer
        report["training_ids"] = [row["id"] for row in examples]
    assert trainer.accelerator.device.type == "mps"
    start = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    report.update(
        {
            "completed_steps": trainer.state.global_step,
            "training_seconds": time.perf_counter() - start,
            "training_loss": result.training_loss,
            "max_adapter_weight_change": callback.weight_change(trainer.model),
            "gradients": callback.gradients,
        }
    )
    assert math.isfinite(result.training_loss)
    report["after_validation"] = evaluate(trainer.model, tokenizer, validation)
    if args.phase == "sft":
        report["after_training_boards"] = evaluate(trainer.model, tokenizer, examples[:20])
    adapter = checkpoints / f"{args.phase}-adapter"
    trainer.model.save_pretrained(adapter)
    saved = load_file(adapter / "adapter_model.safetensors")
    assert all(
        torch.equal(value.detach().cpu(), saved[name])
        for name, value in get_peft_model_state_dict(trainer.model).items()
    )
    report["adapter_tensor_roundtrip_exact"] = True
    report["adapter_path"] = str(adapter.relative_to(ROOT))
    write_json(run / f"{args.phase}-summary.json", report)
    if args.phase == "grpo":
        state_context.clear()
    del trainer, model
    gc.collect()
    torch.mps.empty_cache()
    reloaded = PeftModel.from_pretrained(load_model(), adapter).to("mps")
    after_reload = evaluate(reloaded, tokenizer, validation)
    report["reload_predictions_exact"] = (
        after_reload["predictions"] == report["after_validation"]["predictions"]
    )
    assert report["reload_predictions_exact"]
    write_json(run / f"{args.phase}-summary.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "phase",
                    "completed_steps",
                    "training_seconds",
                    "max_adapter_weight_change",
                    "reload_predictions_exact",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
