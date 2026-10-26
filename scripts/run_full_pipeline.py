"""Keep the authorized local training sequence running independently of chat turns."""

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "results/local-training-status.json"
PYTHON = str(ROOT / ".venv-trl-mps/bin/python")


def status(stage, state, **extra):
    STATUS.write_text(
        json.dumps(
            {
                "stage": stage,
                "state": state,
                "pid": os.getpid(),
                "updated_unix": time.time(),
                **extra,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    jobs = [
        (
            "grpo-start-now",
            [
                "grpo",
                "--steps",
                "100",
                "--horizon",
                "16",
                "--run-name",
                "local-mps-grpo-now",
                "--sft-run-name",
                "local-mps-pilot",
            ],
        ),
        (
            "sft-all-8000",
            [
                "sft",
                "--steps",
                "2000",
                "--examples",
                "8000",
                "--batch-size",
                "4",
                "--run-name",
                "local-mps-full-v1",
            ],
        ),
        (
            "grpo-after-full-sft",
            [
                "grpo",
                "--steps",
                "100",
                "--horizon",
                "16",
                "--run-name",
                "local-mps-full-v1",
                "--sft-run-name",
                "local-mps-full-v1",
            ],
        ),
    ]
    for stage, arguments in jobs:
        phase = arguments[0]
        name = arguments[arguments.index("--run-name") + 1]
        summary_path = ROOT / "results" / name / f"{phase}-summary.json"
        target = int(arguments[arguments.index("--steps") + 1])
        if summary_path.exists():
            previous = json.loads(summary_path.read_text())
            if previous.get("completed_steps") == target and previous.get(
                "reload_predictions_exact"
            ):
                continue
        checkpoint_dir = ROOT / "checkpoints" / name / f"{phase}-trainer"
        if list(checkpoint_dir.glob("checkpoint-*")):
            arguments = [*arguments, "--resume"]
        status(stage, "running", command=arguments)
        with (ROOT / "logs" / f"{stage}.log").open("a") as log:
            result = subprocess.run(
                [PYTHON, "-u", "scripts/train_local.py", *arguments],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            status(stage, "failed", exit_code=result.returncode)
            raise SystemExit(result.returncode)
        summary = json.loads((ROOT / "results" / name / f"{phase}-summary.json").read_text())
        if summary["completed_steps"] != int(arguments[arguments.index("--steps") + 1]):
            status(stage, "incomplete", completed_steps=summary["completed_steps"])
            raise SystemExit(1)
    status("all", "complete")
    print("Local GRPO, all-example SFT, and post-SFT GRPO are complete.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("pipeline", "failed", error=str(error))
        raise
