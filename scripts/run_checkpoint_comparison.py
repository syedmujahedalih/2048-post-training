"""Prepare and evaluate SFT/original controls locally, then restore GRPO serving.

All endpoint configuration, process IDs and listener records remain ignored.
Only one model server is active at a time. Existing exports/runs are resumed,
not overwritten, and frozen GRPO/baseline measurements are checked unchanged.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".local/checkpoint-comparison-v1"
STATUS = PRIVATE / "status.json"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def progress(stage, **fields):
    write_json(STATUS, {"stage": stage, **fields})
    print(json.dumps({"stage": stage, **fields}), flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def process_table():
    result = subprocess.check_output(["ps", "-axo", "pid=,ppid=,stat=,command="]).decode()
    return {
        int(p): (int(parent), state, command)
        for p, parent, state, command in (line.split(None, 3) for line in result.splitlines())
    }


def descendants(server_pid):
    table = process_table()
    owned = {server_pid}
    while True:
        more = {p for p, (parent, _, _) in table.items() if parent in owned}
        if more <= owned:
            return owned
        owned |= more


def stop_server():
    pid_file = ROOT / ".local/serving-api.pid"
    if not pid_file.exists():
        return
    server_pid = int(pid_file.read_text())
    table = process_table()
    if server_pid not in table or table[server_pid][1].startswith("Z"):
        return
    command = table[server_pid][2]
    if "vllm" not in command or str(ROOT / "checkpoints") not in command:
        raise RuntimeError("Saved process is not this project's model server; refusing to stop it")
    owned = descendants(server_pid)
    original_commands = {p: table[p][2] for p in owned if p in table}
    os.kill(server_pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        table = process_table()
        live = [p for p in owned if p in table and not table[p][1].startswith("Z")]
        if not live:
            return
        time.sleep(0.25)
    table = process_table()
    for child in owned:
        if child in table and table[child][2] == original_commands.get(child):
            os.kill(child, signal.SIGTERM)
    time.sleep(1)
    table = process_table()
    for child in owned:
        if child in table and not table[child][1].startswith("Z"):
            if table[child][2] != original_commands.get(child):
                raise RuntimeError("Owned process identity changed during shutdown")
            os.kill(child, signal.SIGKILL)
    time.sleep(0.5)


def audit_listeners(server_pid, label):
    import re

    owned = descendants(server_pid)
    result = subprocess.run(
        ["/usr/sbin/lsof", "-nP", "-a", "-p", ",".join(map(str, owned)), "-iTCP", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
    )
    listeners = [line for line in result.stdout.splitlines()[1:] if "(LISTEN)" in line]
    all_loopback = bool(listeners) and all(
        re.search(r"TCP (?:127\.0\.0\.1:|\[::1\]:)", line) for line in listeners
    )
    write_json(
        PRIVATE / f"{label}-network.json",
        {"processes": sorted(owned), "listeners": listeners, "all_loopback": all_loopback},
    )
    if not all_loopback:
        stop_server()
        raise RuntimeError("A model listener was not loopback-only; server stopped")
    return len(listeners)


def start_server(label, model_dir, port):
    env = dict(os.environ, MODEL_DIR=str(model_dir), SERVED_MODEL_NAME=f"2048-{label}")
    log_path = ROOT / "logs" / f"vllm-metal-{label}-comparison.log"
    with log_path.open("a") as stream:
        server = subprocess.Popen(
            ["sh", str(ROOT / "scripts/serve_local_vllm.sh")],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (ROOT / ".local/serving-api.pid").write_text(str(server.pid) + "\n")
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"{label} server exited; inspect ignored startup log")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=2
            ) as response:
                models = json.load(response)["data"]
            model = next(m for m in models if m["id"] == f"2048-{label}")
            assert Path(model["root"]).resolve() == model_dir.resolve()
            count = audit_listeners(server.pid, label)
            progress("server_ready", policy=label, loopback_listener_count=count)
            return
        except (urllib.error.URLError, TimeoutError, StopIteration):
            time.sleep(1)
    stop_server()
    raise TimeoutError(f"{label} server did not become ready")


def run_command(command, log_name):
    with (ROOT / "logs" / log_name).open("a") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def verify_export(report):
    record = json.loads(report.read_text())
    directory = ROOT / record["output"]
    for name, expected in record["output_manifest"].items():
        path = directory / name
        assert path.stat().st_size == expected["bytes"] and digest(path) == expected["sha256"]
    return directory


def main():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    lock = (PRIVATE / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    probe = (ROOT / ".local/served-evaluation.lock").open("a")
    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(probe, fcntl.LOCK_UN)
    probe.close()
    values = dict(
        line.split("=", 1)
        for line in (ROOT / ".local/serving.env").read_text().splitlines()
        if line and not line.startswith("#")
    )
    port = int(values["PORT"])
    reference = ROOT / "results/served-grpo-v1"
    assert json.loads((reference / "summary.json").read_text())["games"]["grpo"]["games"] == 30
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in reference.rglob("*") if p.is_file()}
    torch_python = str(ROOT / ".venv-trl-mps/bin/python")
    eval_python = str(ROOT / ".venv/bin/python")
    directory = ROOT / "results/local-mps-full-v1"
    completed = False
    try:
        progress("preparing_exports")
        stop_server()
        if not (directory / "sft-merge-report.json").exists():
            run_command(
                [torch_python, "scripts/merge_local_adapter.py", "--phase", "sft"],
                "comparison-sft-merge.log",
            )
        if not (directory / "sft-serving-bf16-export.json").exists():
            run_command(
                [
                    torch_python,
                    "scripts/export_bf16_serving.py",
                    "--source",
                    "checkpoints/local-mps-full-v1/sft-merged-fp32",
                    "--output",
                    "checkpoints/local-mps-full-v1/sft-serving-bf16",
                    "--report",
                    "results/local-mps-full-v1/sft-serving-bf16-export.json",
                ],
                "comparison-sft-bf16.log",
            )
        if not (directory / "base-serving-bf16-export.json").exists():
            run_command(
                [torch_python, "scripts/export_base_serving.py"], "comparison-base-bf16.log"
            )
        for policy in ("sft", "base"):
            report = directory / f"{policy}-serving-bf16-export.json"
            model_dir = verify_export(report)
            run = f"served-{policy}-v1"
            progress("starting_server", policy=policy)
            start_server(policy, model_dir, port)
            progress("evaluating", policy=policy, run=run)
            command = [
                eval_python,
                "scripts/evaluate_served_model.py",
                "--port",
                str(port),
                "--policy",
                policy,
                "--model",
                f"2048-{policy}",
                "--run",
                run,
                "--export-report",
                str(report),
                "--reuse-baselines",
                str(reference),
            ]
            if (ROOT / "results" / run).exists():
                command.append("--resume")
            run_command(command, f"{run}.log")
            progress("evaluation_complete", policy=policy, run=run)
            stop_server()
        for name, expected in frozen.items():
            assert digest(ROOT / name) == expected, "A frozen GRPO reference file changed"
        completed = True
    except BaseException as error:
        progress("failed", error_type=type(error).__name__)
        raise
    finally:
        progress("restoring_grpo", evaluations_complete=completed)
        stop_server()
        start_server("grpo", ROOT / "checkpoints/local-mps-full-v1/serving-bf16", port)
        progress("evaluations_complete" if completed else "failed_grpo_restored")


if __name__ == "__main__":
    main()
