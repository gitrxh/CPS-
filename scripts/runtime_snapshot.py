"""Read-only process and artifact snapshot for the active chemistry experiment."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    p = ROOT / path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"being_written": True}


def process(receipt):
    pid = receipt["pid"]
    p = Path("/proc") / str(pid) / "stat"
    if not p.exists():
        return {"pid": pid, "identity_live": False, "state": "absent"}
    fields = p.read_text().split(") ", 1)[1].split()
    matches = fields[19] == str(receipt["start_ticks"])
    return {"pid": pid, "identity_live": matches and fields[0] != "Z",
            "start_identity_matches": matches, "state": fields[0]}


def latest_json(path):
    p = ROOT / path
    if not p.exists():
        return None
    with p.open("rb") as f:
        f.seek(max(0, p.stat().st_size - 12000))
        lines = f.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def main():
    snapshot = {"utc": datetime.now(timezone.utc).isoformat(),
                "remote_workspace": str(ROOT), "authorized_physical_gpus": [0, 1, 2, 3],
                "goal_status": "active", "jobs": {}, "phase1": {},
                "known_completed_results": {"base_dev24": [1,24], "handcoded_sft_dev24": [6,24]},
                "interpretation": "Ordinary SFT benefit is exploratory. Sampling efficacy requires the complete paired three-arm comparison; partial successes are not used to alter the queue."}
    for job in ["phase0_queue_v1", "phase1_queue_v2", "collect_planner_train120_v1", "h1_pilot24_v1"]:
        folder = "outputs/" + job
        receipt = read(folder + "/job.json")
        if not receipt or receipt.get("being_written"):
            continue
        item = process(receipt)
        state = read(folder + "/status.json")
        if state and not state.get("being_written"):
            item.update({k: state[k] for k in ["current_stage", "queue_finished", "error", "updated_utc"] if k in state})
            if state.get("stages"):
                last = state["stages"][-1]
                item["latest_stage"] = {k:last[k] for k in ["name", "pid", "returncode", "elapsed_seconds"] if k in last}
                pid = last.get("pid")
                if pid:
                    path = Path("/proc") / str(pid) / "stat"
                    item["latest_stage"]["proc_state"] = path.read_text().split(") ",1)[1].split()[0] if path.exists() else "absent"
        else:
            item["latest_progress"] = latest_json(folder + "/stdout.log")
        snapshot["jobs"][job] = item
    all_complete = True
    outcomes = {}
    for arm in ["canonical", "uniform", "brp"]:
        checkpoint = "checkpoints/phase1_" + arm + "_seed42_v1"
        evaluation = "outputs/eval_phase1_" + arm + "_dev24_v1"
        item = {}
        training = read(checkpoint + "/summary.json")
        if training:
            item["training"] = training
        else:
            item["training_progress"] = latest_json(checkpoint + "/metrics.jsonl")
        complete = read(evaluation + "/summary.json")
        p = ROOT / evaluation / "episodes.jsonl"
        item["evaluation_episodes_written"] = sum(1 for line in p.open() if line.strip()) if p.exists() else 0
        is_complete = bool(complete and complete.get("complete") and complete.get("technical_errors") == 0)
        item["evaluation_complete_without_errors"] = is_complete
        all_complete = all_complete and is_complete
        if is_complete:
            outcomes[arm] = complete
        snapshot["phase1"][arm] = item
    snapshot["phase1_all_arms_complete"] = all_complete
    if all_complete:
        snapshot["phase1_complete_outcomes"] = outcomes
    snapshot["gpus_0_to_3"] = subprocess.check_output(
        ["nvidia-smi", "-i", "0,1,2,3", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=30).strip().splitlines()
    output = ROOT / "outputs/runtime_snapshot.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(snapshot, indent=2))
    temporary.replace(output)
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
