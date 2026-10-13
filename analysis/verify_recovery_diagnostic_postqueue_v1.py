"""Read-only score/Q postqueue integrity checks, then one exclusive receipt.

Run from /home/jsy/GNRL_20261012 after both stages and controller exit. Accepts
explicit adoption/job paths, expected adoption SHA, PID and Linux start ticks;
there are no guessed identities. Can run as a saved script or through stdin.
No records/events/diagnostic outcomes are parsed, no worker is launched and no
other process is signalled. Every verified map contains measured byte hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
from scripts import run_successful_path_learning_v1 as old
from scripts.collect_recovery_projection_v1 import source_inventory

STAGES = ("score", "diagnostic")
ADAPTER = "checkpoints/successful_path_canonical_seed42_v1/final"
MODEL = "/home/jsy/zhangranxu-adrs-week-20260929/models/Qwen2.5-3B-Instruct"


def relative(name):
    old.require(isinstance(name, str), "Expected normalized relative workspace path")
    path = Path(name)
    old.require(isinstance(name, str) and not path.is_absolute() and path.as_posix() == name
                and ".." not in path.parts, "Expected normalized relative workspace path")
    return old.safe_relative(name)


def exited(pid, ticks, session):
    old.require(type(pid) is int and pid > 0 and type(ticks) is int and ticks > 0 and session == pid,
                "Missing exact positive process/session identity")
    actual = old.process_record(pid)
    old.require(actual is None or actual["start_ticks"] == ticks, "PID reused; cannot verify original identity")
    old.require(actual is None or actual["state"] == "Z", "Recorded process is still live")
    old.require(not old.owned_members(pid, ticks), "Recorded session still has live owned processes")


def measure_one(path, check_deadline):
    check_deadline()
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            check_deadline()
            digest.update(chunk)
    after = path.stat()
    old.require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "File changed during hash measurement")
    check_deadline()
    return digest.hexdigest()


def measured(mapping, check_deadline, *, absolute=False):
    result = {}
    for name, expected in mapping.items():
        path = Path(name) if absolute else relative(name)
        old.require(not absolute or path.is_absolute(), "Original/runtime path must be absolute")
        actual = measure_one(path, check_deadline)
        old.require(actual == expected, "Frozen bytes differ: " + name)
        result[name] = actual
    return result


def inherited_deadline(status, acquisition_status, candidate, now=None):
    now = time.time() if now is None else now
    start, end = status["main_study_started_unix"], status["global_deadline_unix"]
    old.require(type(start) in (int, float) and type(end) in (int, float)
                and math.isfinite(start) and math.isfinite(end)
                and start == acquisition_status["main_study_started_unix"]
                and end == acquisition_status["global_deadline_unix"]
                and end - start == 171840
                and candidate["version"] == "v2"
                and candidate["finite_budgets"]["global_wall_seconds_including_waits_verification_cleanup"] == 171840,
                "Study deadline was reset/changed")
    old.require(30 < end - now <= 171840, "Inherited study deadline exhausted/invalid")
    return end


def verify(args):
    old.require(sys.platform.startswith("linux") and ROOT == Path("/home/jsy/GNRL_20261012").resolve(), "Wrong host/project")
    old.require(type(args.expected_pid) is int and args.expected_pid > 0
                and type(args.expected_start_ticks) is int and args.expected_start_ticks > 0,
                "Expected controller PID/start ticks must be explicit positive integers")
    old.require(len(args.expected_adoption_sha256) == 64 and all(c in "0123456789abcdef" for c in args.expected_adoption_sha256),
                "Expected adoption SHA must be lowercase SHA256")
    adoption_path, job_path, output = relative(args.adoption_json), relative(args.job_json), relative(args.output)
    old.require(args.output.startswith("outputs/") and not output.exists() and output.parent.is_dir(),
                "Receipt requires an existing owned output directory and a new filename")
    old.require(old.sha256(adoption_path) == args.expected_adoption_sha256, "Wrong explicit adoption bytes")
    adoption = old.read_json(adoption_path)
    old.require(adoption.get("kind") == "recovery_diagnostic_parent_adoption" and adoption.get("adopted") is True
                and adoption.get("adopted_by") == "root_after_independent_GPT_6_Astra_review"
                and adoption["authorized_stages"] == list(STAGES) and adoption["physical_gpu"] == 2,
                "Wrong diagnostic phase adoption")
    paths = adoption["output_paths"]
    queue = relative(paths["queue"])
    status_path, locks_path = queue / "status.json", queue / "locked_sha256.json"
    status_hash, locks_hash, job_hash = old.sha256(status_path), old.sha256(locks_path), old.sha256(job_path)
    status, locks, job = old.read_json(status_path), old.read_json(locks_path), old.read_json(job_path)
    old.require(status.get("complete") is True and status.get("current_stage") == "needs_astra_diagnostic_review"
                and status.get("gate_evaluable") is True and type(status.get("training_gate_passed")) is bool
                and not status.get("error") and not status.get("errors")
                and status["adoption_sha256"] == args.expected_adoption_sha256,
                "Diagnostic queue is not fully complete")
    old.require([s["name"] for s in status["stages"]] == list(STAGES), "Missing/extra phase stage")
    required = {adoption[k] for k in ("candidate", "input_inventory", "acquisition_status", "acquisition_locks",
                                    "acquisition_parent_adoption", "acquisition_result_review", "controller_review")}
    required.update((old.MODEL_INVENTORY, old.PROVENANCE))
    old.require(required <= adoption["file_sha256"].keys(), "Missing adopted metadata locks")
    old.require(job["pid"] == args.expected_pid and int(job["start_ticks"]) == args.expected_start_ticks
                and Path(job["cwd"]).resolve() == ROOT, "Recorded controller identity/workspace differs")
    exited(args.expected_pid, args.expected_start_ticks, args.expected_pid)
    stages = []
    for stage in status["stages"]:
        old.require(stage.get("returncode") == 0 and stage.get("owned_cleanup_complete") is True, "Stage cleanup/completion failed")
        exited(stage["pid"], stage["start_ticks"], stage["owned_session"])
        folder = relative(paths[stage["name"]])
        required_outputs = {paths[stage["name"]] + "/" + f for f in
                            ("summary.json", "cost_progress.json", "config.json", "records.jsonl", "events.jsonl")}
        old.require(required_outputs <= locks.keys(), "Missing final stage artifact locks")
        summary, costs = old.read_json(folder / "summary.json"), old.read_json(folder / "cost_progress.json")
        old.require(summary.get("complete") is True and summary.get("stage") == stage["name"]
                    and not summary.get("technical_incomplete") and not summary.get("error")
                    and costs.get("active_call") is None and costs.get("unknown_generation_after_error") is False
                    and summary["costs"]["totals"] == costs["totals"], "Stage metadata/costs are incomplete")
        old.require(summary["records_sha256"] == locks[paths[stage["name"]] + "/records.jsonl"],
                    "Stage summary and locked records differ")
        old.require(all(costs["totals"][k + "_started"] == costs["totals"][k + "_completed"]
                        for k in ("environment_steps", "resets", "policy_calls")), "Unfinished stage call")
        if stage["name"] == "score":
            old.require(summary.get("record_count") == 180 and summary.get("labels_locked_before_Q") is True,
                        "Incomplete180 score/label barrier")
        else:
            old.require(summary.get("gate_evaluable") is True and type(summary.get("training_gate_passed")) is bool
                        and summary.get("outcome_unblinded") is True, "Incomplete diagnostic barrier")
        stages.append({k: stage[k] for k in ("name", "pid", "start_ticks", "owned_session")} | {"exited": True})
    candidate = old.read_json(relative(adoption["candidate"]))
    acquisition_status = old.read_json(relative(adoption["acquisition_status"]))
    end = inherited_deadline(status, acquisition_status, candidate)
    # A bounded read-only verification still uses the original study clock.
    verification_deadline = min(time.monotonic() + 300, time.monotonic() + end - time.time() - 30)
    def check():
        old.require(time.monotonic() < verification_deadline and time.time() < end - 30,
                    "Postqueue verification exceeded inherited deadline/reserve")
    old.require(source_inventory() == adoption["source_inventory"]
                and set(adoption["source_inventory"]) <= adoption["file_sha256"].keys()
                and all(locks.get(name) == h for name, h in adoption["file_sha256"].items()), "Source inventory/original locks differ")
    old.require(len(adoption["original_file_sha256"]) == 336 and len(adoption["runtime_file_sha256"]) == 9,
                "Original336/runtime9 bindings missing")
    verified = measured(locks, check)
    originals = measured(adoption["original_file_sha256"], check, absolute=True)
    runtimes = measured(adoption["runtime_file_sha256"], check, absolute=True)
    inventory = old.read_json(relative(adoption["input_inventory"]))
    old.require(inventory.get("complete") is True and inventory["original_file_sha256"] == originals
                and inventory["base_full_inventory"] == old.MODEL_INVENTORY
                and inventory["base_full_inventory_sha256"] == verified[old.MODEL_INVENTORY], "Input inventory provenance differs")
    model_inventory = old.read_json(relative(old.MODEL_INVENTORY))
    model_root = Path(model_inventory["model_path"]).resolve()
    old.require(model_root == Path(MODEL).resolve() and candidate["frozen_start_policy"]["base"] == MODEL
                and candidate["frozen_start_policy"]["adapter"] == ADAPTER, "Base/adapter selection changed")
    expected_model = {f["name"]: f["sha256"] for f in model_inventory["files"]}
    old.require(len(expected_model) == len(model_inventory["files"]) == 28
                and {f.relative_to(model_root).as_posix() for f in model_root.rglob("*") if f.is_file()} == set(expected_model),
                "Full base28 inventory differs")
    model_hashes = {}
    for name, expected in expected_model.items():
        target = (model_root / name).resolve()
        old.require(target.is_relative_to(model_root), "Model file escapes original root")
        actual = measure_one(target, check)
        old.require(actual == expected, "Base model bytes changed")
        model_hashes[name] = actual
    adapter_root = relative(ADAPTER)
    old.require(len(inventory["initial_adapter_sha256"]) == 11
                and {f.relative_to(ROOT).as_posix() for f in adapter_root.rglob("*") if f.is_file()}
                == set(inventory["initial_adapter_sha256"]), "Full adapter11 inventory differs")
    adapter_hashes = measured(inventory["initial_adapter_sha256"], check)
    old.require(all(verified.get(name) == h for name, h in adapter_hashes.items()), "Adapter not included in final locks")
    check()
    old.require(old.sha256(status_path) == status_hash and old.sha256(locks_path) == locks_hash
                and old.sha256(job_path) == job_hash and old.sha256(adoption_path) == args.expected_adoption_sha256,
                "Queue/job/adoption changed during verification")
    old.require(source_inventory() == adoption["source_inventory"], "Source inventory changed during verification")
    exited(args.expected_pid, args.expected_start_ticks, args.expected_pid)
    for stage in stages:
        exited(stage["pid"], stage["start_ticks"], stage["owned_session"])
    check()
    receipt = {"kind": "recovery_diagnostic_postqueue_integrity", "complete": True, "errors": [],
        "created_utc": old.utc(), "adoption_sha256": args.expected_adoption_sha256,
        "queue_status_sha256": status_hash, "locked_sha256_file_sha256": locks_hash,
        "job_receipt_sha256": job_hash, "verified_file_sha256": verified,
        "verified_original_file_sha256": originals, "verified_runtime_file_sha256": runtimes,
        "source_inventory": source_inventory(), "base_inventory_sha256": verified[old.MODEL_INVENTORY],
        "base_files_verified": len(model_hashes), "base_measured_sha256": model_hashes,
        "initial_adapter_sha256": adapter_hashes,
        "controller": {"pid": args.expected_pid, "start_ticks": args.expected_start_ticks, "exited": True},
        "owned_workers_gone": True, "stages": stages,
        "main_study_started_unix": status["main_study_started_unix"], "global_deadline_unix": end,
        "verification_finished_before_deadline": True, "outcome_records_parsed": False}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return {"complete": True, "receipt": args.output, "sha256": old.sha256(output),
            "verified_workspace_files": len(verified), "original_files": len(originals),
            "runtime_files": len(runtimes), "base_files": len(model_hashes), "adapter_files": len(adapter_hashes)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption-json", required=True)
    parser.add_argument("--expected-adoption-sha256", required=True)
    parser.add_argument("--job-json", required=True)
    parser.add_argument("--expected-pid", type=int, required=True)
    parser.add_argument("--expected-start-ticks", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    old.require(sys.platform.startswith("linux"), "Remote read-only verifier requires Linux")
    def timeout(signum, frame):
        raise TimeoutError("Postqueue verifier wall cap")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        print(json.dumps(verify(args)))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
