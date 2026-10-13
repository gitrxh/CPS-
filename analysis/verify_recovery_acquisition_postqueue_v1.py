"""Run as stdin in the remote project after full acquisition exit.

Read-only checks of processes and frozen bytes, followed by one exclusive-create
integrity receipt in the owned project. No records/events are parsed for outcomes.
"""
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
from scripts import run_successful_path_learning_v1 as old
from scripts.collect_recovery_projection_v1 import source_inventory

ADOPTION = "configs/recovery_acquisition_adoption_v1.json"
EXPECTED_ADOPTION = "1d1f1dae4a3c99649160ef6343f3e66d297aaf57c92573437b2d8e7ca74ddd55"
JOB = "outputs/recovery_projection_acquisition_job_v1/job.json"
OUTPUT = "outputs/recovery_acquisition_postqueue_integrity_v1.json"


def exited(pid, ticks, session=None):
    actual = old.process_record(pid)
    old.require(actual is None or actual["start_ticks"] == ticks, "PID reused; cannot establish original ownership")
    old.require(actual is None or actual["state"] == "Z", "Owned process is still live")
    if session is not None:
        old.require(session == pid and not old.owned_members(pid, ticks), "Owned worker session still live")


def measured(mapping, absolute=False):
    result = {}
    for name, expected in mapping.items():
        path = Path(name) if absolute else old.safe_relative(name)
        old.require(not absolute or path.is_absolute(), "Expected absolute original/runtime path")
        actual = old.sha256(path)
        old.require(actual == expected, "Frozen bytes differ: " + name)
        result[name] = actual
    return result


def main():
    old.require(sys.platform.startswith("linux") and ROOT == Path("/home/jsy/GNRL_20261012").resolve(), "Wrong host/project")
    old.require(not Path(OUTPUT).exists(), "Receipt already exists; do not overwrite")
    old.require(old.sha256(ADOPTION) == EXPECTED_ADOPTION, "Wrong original adoption bytes")
    adoption = old.read_json(ADOPTION)
    queue = Path(adoption["output_paths"]["queue"])
    status_path, locks_path = queue / "status.json", queue / "locked_sha256.json"
    status_hash, locks_hash = old.sha256(status_path), old.sha256(locks_path)
    status, locks = old.read_json(status_path), old.read_json(locks_path)
    old.require(status.get("complete") is True and status.get("current_stage") == "needs_astra_acquisition_review"
                and status["adoption_sha256"] == EXPECTED_ADOPTION and not status.get("error"), "Acquisition incomplete")
    old.require(status["global_deadline_unix"] > time.time(), "Original study deadline expired")
    old.require([s["name"] for s in status["stages"]] == ["collect", "oracle", "replay"], "Incomplete stage sequence")
    job = old.read_json(JOB)
    old.require(job["pid"] == 455646 and int(job["start_ticks"]) == 30336263, "Wrong recorded controller identity")
    exited(job["pid"], int(job["start_ticks"]), job["pid"])
    stages = []
    for stage in status["stages"]:
        old.require(stage["returncode"] == 0 and stage["owned_cleanup_complete"] is True, "Incomplete worker cleanup")
        exited(stage["pid"], stage["start_ticks"], stage["owned_session"])
        stages.append({k: stage[k] for k in ("name", "pid", "start_ticks", "owned_session")} | {"exited": True})
    old.require(source_inventory() == adoption["source_inventory"], "Source inventory changed before postqueue check")
    old.require(all(locks.get(name) == h for name, h in adoption["file_sha256"].items()), "Lost adopted input bindings")
    verified = measured(locks)
    originals = measured(adoption["original_file_sha256"], absolute=True)
    runtimes = measured(adoption["runtime_file_sha256"], absolute=True)
    inventory = old.read_json(adoption["input_inventory"])
    model_inventory = old.read_json(old.MODEL_INVENTORY)
    model_root = Path(model_inventory["model_path"]).resolve()
    expected_model = {r["name"]: r["sha256"] for r in model_inventory["files"]}
    old.require(len(expected_model) == 28 and {p.relative_to(model_root).as_posix() for p in model_root.rglob("*")
                if p.is_file()} == set(expected_model), "Base model inventory differs")
    model_hashes = {}
    for name, expected in expected_model.items():
        actual = old.sha256(model_root / name)
        old.require(actual == expected, "Base model bytes differ")
        model_hashes[name] = actual
    adapter_root = ROOT / "checkpoints/successful_path_canonical_seed42_v1/final"
    old.require({p.relative_to(ROOT).as_posix() for p in adapter_root.rglob("*") if p.is_file()}
                == set(inventory["initial_adapter_sha256"]), "Initial adapter inventory differs")
    adapter_hashes = measured(inventory["initial_adapter_sha256"])
    old.require(old.sha256(status_path) == status_hash and old.sha256(locks_path) == locks_hash
                and old.sha256(ADOPTION) == EXPECTED_ADOPTION, "Queue/adoption changed during verification")
    old.require(source_inventory() == adoption["source_inventory"], "Source inventory changed during verification")
    exited(job["pid"], int(job["start_ticks"]), job["pid"])
    for stage in stages:
        exited(stage["pid"], stage["start_ticks"], stage["owned_session"])
    receipt = {"kind": "recovery_acquisition_postqueue_integrity", "complete": True, "errors": [],
        "created_utc": old.utc(), "adoption_sha256": EXPECTED_ADOPTION, "queue_status_sha256": status_hash,
        "locked_sha256_file_sha256": locks_hash, "verified_file_sha256": verified,
        "verified_original_file_sha256": originals, "verified_runtime_file_sha256": runtimes,
        "source_inventory": source_inventory(), "base_inventory_sha256": old.sha256(old.MODEL_INVENTORY),
        "base_files_verified": len(model_hashes), "base_measured_sha256": model_hashes,
        "initial_adapter_sha256": adapter_hashes,
        "controller": {"pid": job["pid"], "start_ticks": int(job["start_ticks"]), "exited": True},
        "owned_workers_gone": True, "stages": stages}
    with Path(OUTPUT).open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({"complete": True, "receipt": OUTPUT, "sha256": old.sha256(OUTPUT),
                     "verified_workspace_files": len(verified), "original_files": len(originals),
                     "runtime_files": len(runtimes), "base_files": len(model_hashes)}))


if __name__ == "__main__":
    signal.alarm(300)
    main()
