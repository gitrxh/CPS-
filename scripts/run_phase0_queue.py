"""Finite preregistered pilot queue; refuses to train a method on identity-only data."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PY = str(ROOT / ".venv/bin/python")
MODEL = "/home/jsy/zhangranxu-adrs-week-20260929/models/Qwen2.5-3B-Instruct"
QUEUE = ROOT / "outputs/phase0_queue_v1"
QUEUE.mkdir(parents=True, exist_ok=True)
START = time.monotonic()
DEADLINE = START + 8 * 3600
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1")
STATE = {"version": 1, "started_utc": datetime.now(timezone.utc).isoformat(), "stages": []}
LOCKED_HASHES = {}


def verify_locked_files():
    for name, expected in LOCKED_HASHES.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Locked experiment file changed: " + name)


def task_identities(path):
    result = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["gamefile"], row["seed"])
            if key in result:
                raise RuntimeError("Duplicate task identity in " + str(path))
            result[key] = row.get("selection", {}).get("pool_sha256")
    return result


def update(stage, **kwargs):
    STATE.update(current_stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **kwargs)
    (QUEUE / "status.json").write_text(json.dumps(STATE, indent=2))
    print(json.dumps({"stage": stage, **kwargs}), flush=True)


def wait_file(path):
    update("waiting_for_" + str(path))
    while not Path(path).exists():
        if time.monotonic() > DEADLINE:
            raise TimeoutError("Queue deadline exceeded waiting for " + str(path))
        time.sleep(20)


def wait_gpu(index):
    update("waiting_for_gpu", physical_gpu=index)
    while time.monotonic() < DEADLINE:
        from launch_gpu import resolve_devices
        import csv
        import io
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used", "--format=csv,noheader,nounits"], text=True)
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"], text=True)
        try:
            resolve_devices([index], list(csv.reader(io.StringIO(rows))), active_uuids=set(active.splitlines()))
            return
        except RuntimeError:
            time.sleep(30)
    raise TimeoutError("No idle authorized GPU before queue deadline")


def run(name, args, gpu=None):
    if gpu is not None:
        wait_gpu(gpu)
    update(name)
    verify_locked_files()
    command = [PY] + args
    if gpu is not None:
        command = [PY, "scripts/launch_gpu.py", "--gpus", str(gpu), "--receipt", str(QUEUE / (name + "_gpu.json")), "--"] + command
    start = time.monotonic()
    with (QUEUE / (name + ".stdout.log")).open("wb") as out, (QUEUE / (name + ".stderr.log")).open("wb") as err:
        process = subprocess.Popen(command, env=ENV, stdout=out, stderr=err, start_new_session=True)
        try:
            rc = process.wait(timeout=max(1, DEADLINE-time.monotonic()))
        except subprocess.TimeoutExpired:
            # Only this stage's own newly-created process group is affected.
            if process.poll() is None and os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise TimeoutError("Stage deadline exceeded; own process group terminated")
    STATE["stages"].append({"name": name, "command": command, "returncode": rc, "elapsed_seconds": time.monotonic()-start})
    update(name, last_returncode=rc)
    if rc:
        raise RuntimeError(f"Stage {name} failed ({rc}); inspect logs")


def train_eval(arm, data):
    out = "checkpoints/" + arm + "_seed42_v1"
    run("train_" + arm, ["-m", "gnrl.train_sft", "--model", MODEL,
        "--data", data, "--output", out, "--max-steps", "100", "--batch-size", "1",
        "--gradient-accumulation", "8", "--learning-rate", "0.0002", "--seed", "42",
        "--max-context-tokens", "2048", "--max-wall-seconds", "7200"], gpu=3)
    summary = json.loads(Path(out, "summary.json").read_text())
    if not summary["matched_step_budget_complete"]:
        raise RuntimeError("Incomplete optimizer budget: comparison must not proceed")
    run("eval_" + arm, ["-m", "gnrl.evaluate", "--model", MODEL, "--adapter", out + "/final",
        "--games-json", "data/manifests/dev24.json", "--output", "outputs/eval_" + arm + "_dev24_v1",
        "--selection", "generate", "--max-steps", "100", "--seed", "20261012",
        "--max-context-tokens", "2048", "--max-wall-seconds", "7200"], gpu=3)
    evaluation = json.loads(Path("outputs/eval_" + arm + "_dev24_v1/summary.json").read_text())
    manifest_games = json.loads(Path("data/manifests/dev24.json").read_text())["games"]
    if (not evaluation["complete"] or evaluation["technical_errors"] != 0 or
        evaluation["planned_episodes"] != len(manifest_games) or evaluation["attempted_episodes"] != len(manifest_games)):
        raise RuntimeError("Incomplete or technically failed development evaluation; review required")
    actual_games = [json.loads(line)["gamefile"] for line in Path("outputs/eval_" + arm + "_dev24_v1/episodes.jsonl").read_text().splitlines() if line.strip()]
    if actual_games != manifest_games:
        raise RuntimeError("Evaluation task identities/order do not match the frozen manifest")


def main():
    # Preserve a source receipt for these exact jobs; no secrets are in these paths.
    source = {}
    for folder in ("gnrl", "scripts", "configs"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json"):
                source[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((ROOT / "data/manifests").glob("*.json")):
        source[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    LOCKED_HASHES.update(source)
    (QUEUE / "source_sha256.json").write_text(json.dumps(source, indent=2))
    wait_file("data/expert_train120_raw.summary.json")
    raw_summary = json.loads(Path("data/expert_train120_raw.summary.json").read_text())
    if raw_summary["counts"].get("task_errors", 0):
        raise RuntimeError("Expert collection has technical/replay errors; review before training")
    run("curate", ["scripts/curate_experts.py", "--input", "data/expert_train120_raw.jsonl", "--output", "data/expert_train120_verified.jsonl"])
    # Baseline independently establishes usable agent learning; it does not claim BRP benefit.
    train_eval("expert", "data/expert_train120_verified.jsonl")
    run("macro_search", ["-m", "gnrl.macro_project", "--input", "data/expert_train120_verified.jsonl",
        "--output", "outputs/macro_train120_v1", "--search-only", "--horizons", "2", "4", "8",
        "--max-games", "120", "--max-candidates", "32", "--max-env-steps", "100000", "--max-search-seconds", "3600"])
    macro = json.loads(Path("outputs/macro_train120_v1/search_summary.json").read_text())
    if macro.get("invalid_sources_or_replay_errors", 0):
        raise RuntimeError("Macro replay failure requires Astra review")
    expected_games = sum(1 for line in Path("data/expert_train120_verified.jsonl").read_text().splitlines() if line.strip())
    if macro.get("games_completed", 0) != expected_games:
        update("needs_astra_method_review", reason="Macro search did not cover the full fixed training set; no unmatched arm training", macro_summary=macro, expected_games=expected_games, queue_finished=True)
        return
    if macro.get("verified_nontrivial_candidates", 0) < 20 or macro.get("games_with_nontrivial_candidates", 0) < 10:
        update("needs_astra_method_review", reason="Insufficient nontrivial BRP support under registered search budget", macro_summary=macro, queue_finished=True)
        return
    run("macro_select", ["-m", "gnrl.macro_project", "--pools", "outputs/macro_train120_v1/candidate_pools.jsonl",
        "--model", MODEL, "--output", "outputs/selection_train120_v1", "--max-score-seconds", "3600"], gpu=3)
    selection = json.loads(Path("outputs/selection_train120_v1/selection_summary.json").read_text())
    if selection["stop_reason"] != "complete" or selection.get("paired_trajectories_written") != macro.get("games_completed"):
        raise RuntimeError("Incomplete selection budgets; do not compare unmatched training sets")
    expert_tasks = task_identities("data/expert_train120_verified.jsonl")
    uniform_tasks = task_identities("outputs/selection_train120_v1/uniform_valid.jsonl")
    brp_tasks = task_identities("outputs/selection_train120_v1/learner_selected.jsonl")
    if set(expert_tasks) != set(uniform_tasks) or uniform_tasks != brp_tasks or any(value is None for value in brp_tasks.values()):
        raise RuntimeError("Training task identities or paired candidate pool hashes do not match")
    if selection.get("learner_changed_trajectories", 0) < 10:
        update("needs_astra_method_review", reason="Selector changed fewer than10 training tasks", selection_summary=selection, queue_finished=True)
        return
    train_eval("uniform", "outputs/selection_train120_v1/uniform_valid.jsonl")
    train_eval("brp", "outputs/selection_train120_v1/learner_selected.jsonl")
    update("needs_astra_result_review", queue_finished=True, note="Pilot only. No unseen evaluation or automatic positive-effect claim.")


try:
    main()
except Exception as error:
    update("failed_needs_review", queue_finished=True, error=f"{type(error).__name__}: {error}")
    raise
