"""Read-only learning postqueue integrity; no episode/event/result parsing.

Run from the chemistry project root, saved or via stdin, after controller exit.
Requires explicit adoption SHA and recorded controller PID/start ticks. Reuses
the byte-hashed diagnostic verifier helpers; deploy both analysis files together.
Every returned inventory is measured. The sole write exclusively creates the
requested receipt. This script never launches or signals another process.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
from analysis import verify_recovery_diagnostic_postqueue_v1 as v
from scripts import run_recovery_diagnostic_queue_v1 as prior
from gnrl import recovery_evaluate_v1 as evaluator

old = v.old
HELPER = "analysis/verify_recovery_diagnostic_postqueue_v1.py"
HELPER_SHA256 = "efa7ce6466394f410848cc61bc55626573035a0d42cceb7b2b84ac270eee24e6"
TRAIN = ("expert_recovery", "uniform_viable_recovery", "reference_conditional_recovery")
ARMS = tuple(evaluator.ARMS)
STAGES = ("joint_token_audit", *("train_" + arm for arm in TRAIN), *("eval_" + arm for arm in ARMS))
OUTCOME_OR_TRACE_FILES = {"episodes.jsonl", "events.jsonl", "result.json", "records.jsonl", "diagnostic.json",
                          "metrics.jsonl", "example_exposures.jsonl"}


def read_metadata(name):
    """Reject a mistaken artifact path before parsing any outcome/trace bytes."""
    old.require(Path(name).name not in OUTCOME_OR_TRACE_FILES, "Outcome/trace parsing forbidden in postqueue verifier")
    path = v.relative(name)
    old.require(path.name not in OUTCOME_OR_TRACE_FILES, "Resolved path aliases an outcome/trace file")
    return old.read_json(path)


def metadata_barrier(adoption, status, locks, read, exited):
    """Only technical metadata, with all recorded process exits checked first."""
    old.require(status.get("complete") is True and status.get("current_stage") == "needs_astra_learning_result_review"
                and status.get("outcome_unblinded") is True
                and type(status.get("exploratory_practical_screen_passed")) is bool
                and not status.get("error") and not status.get("errors"), "Learning queue incomplete")
    old.require([row["name"] for row in status["stages"]] == list(STAGES), "Missing/extra learning stage")
    stages = []
    for row in status["stages"]:
        old.require(row.get("returncode") == 0 and row.get("owned_cleanup_complete") is True,
                    "Learning stage exit/cleanup incomplete")
        exited(row["pid"], row["start_ticks"], row["owned_session"])
        stages.append({key: row[key] for key in ("name", "pid", "start_ticks", "owned_session")} | {"exited": True})
    paths = adoption["output_paths"]
    old.require(paths["joint_token_audit"] in locks, "Missing token audit lock")
    audit = read(paths["joint_token_audit"])
    old.require(audit.get("kind") == "recovery_joint_training_token_audit_v1" and audit.get("complete") is True
                and audit.get("score_sft_token_context_mismatches") == 0
                and set(audit["arms"]) == set(TRAIN), "Incomplete joint token audit")
    for arm in TRAIN:
        folder = paths["train_" + arm]
        names = {folder + "/" + name for name in ("summary.json", "config.json", "metrics.jsonl",
                  "example_exposures.jsonl", "final_adapter_inventory.json", "final/training_state.json")}
        old.require(names <= locks.keys(), "Missing completed training artifact lock")
        summary = read(folder + "/summary.json")
        state = read(folder + "/final/training_state.json")
        expected = audit["arms"][arm]["totals"]["target_tokens"]
        old.require(summary.get("complete") is True and summary.get("matched_step_budget_complete") is True
                    and summary.get("stop_reason") == "max_steps" and summary["technical_errors"] == []
                    and summary["optimizer_steps"] == summary["requested_steps"] == state["optimizer_steps"] == 100
                    and 0 < expected <= 2000000
                    and summary["target_tokens"] == summary["actual_backward_target_tokens"]
                    == summary["expected_target_tokens"] == state["target_tokens"] == expected,
                    "Training completion/count metadata differs")
        for item in (summary, state):
            old.require(item["token_audit_sha256"] == locks[paths["joint_token_audit"]]
                        and item["schedule_sha256"] == audit["schedule_sha256"]
                        and item["sampled_examples"] == 800 and item["component_exposures"] == {"bc":400,"recovery":400},
                        "Training audit/exposure metadata differs")
    previous_parent = read(adoption["diagnostic_parent_adoption"])
    previous = read(previous_parent["output_paths"]["diagnostic"] + "/cost_progress.json")["totals"]
    total_episodes = 0
    for arm in ARMS:
        folder = paths["eval_" + arm]
        names = {folder + "/" + name for name in ("summary.json", "config.json", "cost_progress.json", "episodes.jsonl", "events.jsonl")}
        old.require(names <= locks.keys(), "Missing evaluation artifact lock")
        summary, costs = read(folder + "/summary.json"), read(folder + "/cost_progress.json")
        old.require(summary.get("complete") is True and summary.get("stage") == "evaluation" and summary.get("arm") == arm
                    and summary["planned_episodes"] == summary["attempted_episodes"] == summary["completed_episodes"] == 48
                    and summary["technical_errors"] == 0 and summary["partial_performance_summary"] is None
                    and not summary.get("error") and summary["episodes_sha256"] == locks[folder + "/episodes.jsonl"],
                    "Complete192 technical barrier failed")
        old.require(costs["stage_limits"] == evaluator.STAGE_LIMITS
                    and costs["stage_counts"]["resets_started"] == 48, "Evaluation cost metadata differs")
        previous = prior.validate_cost_lineage(previous, costs, summary)
        total_episodes += summary["completed_episodes"]
    old.require(total_episodes == 192 and status["cumulative_costs"] == previous,
                "Final episode/count or cost lineage differs")
    return stages


def measure_adapters(barrier, paths, locks, initial, check):
    old.require(barrier.get("complete") is True and barrier.get("arms") == list(ARMS)
                and set(barrier["adapters"]) == set(ARMS), "Incomplete four-model barrier")
    directories, result = [], {}
    for arm in ARMS:
        check()
        item = barrier["adapters"][arm]
        expected_path = v.ADAPTER if arm == ARMS[0] else paths["train_" + arm] + "/final"
        old.require(item["path"] == expected_path, "Wrong arm adapter directory")
        directory = v.relative(item["path"])
        directories.append(directory)
        names = {p.relative_to(ROOT).as_posix() for p in directory.rglob("*") if p.is_file()}
        required = {item["path"] + "/adapter_config.json", item["path"] + "/training_state.json"}
        old.require(names == set(item["file_sha256"]) and required <= names
                    and any(name.endswith(("/adapter_model.safetensors", "/adapter_model.bin")) for name in names),
                    "Adapter file inventory changed/incomplete")
        old.require(all(locks.get(name) == h for name, h in item["file_sha256"].items()), "Adapter not bound in final locks")
        result[arm] = v.measured(item["file_sha256"], check)
    old.require(len(set(directories)) == 4 and result[ARMS[0]] == initial, "Aliased models or canonical initialization differs")
    # Another writer can add a file after the first enumeration. Such additions
    # are not visited by measured(mapping), so explicitly reject them at the end.
    for arm, directory in zip(ARMS, directories):
        check()
        actual = {p.relative_to(ROOT).as_posix() for p in directory.rglob("*") if p.is_file()}
        old.require(actual == set(result[arm]), "Adapter inventory changed during measurement")
    return result


def measure_base(model_root, inventory, check):
    """Measure the full28 manifest and reject additions while hashing."""
    check()
    expected = {item["name"]: item["sha256"] for item in inventory["files"]}
    old.require(len(expected) == len(inventory["files"]) == 28
                and {p.relative_to(model_root).as_posix() for p in model_root.rglob("*") if p.is_file()} == set(expected),
                "Full base28 inventory changed")
    result = {}
    for item in inventory["files"]:
        name, h = item["name"], item["sha256"]
        path = (model_root / name).resolve()
        old.require(path.is_relative_to(model_root), "Model file escapes root")
        old.require(type(item["bytes"]) is int and item["bytes"] >= 0 and path.stat().st_size == item["bytes"],
                    "Base declared file size differs")
        result[name] = v.measure_one(path, check)
        old.require(result[name] == h, "Base bytes changed")
    check()
    old.require({p.relative_to(model_root).as_posix() for p in model_root.rglob("*") if p.is_file()} == set(result),
                "Base inventory changed during measurement")
    return result


def write_receipt(path, receipt):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)


def verify(args):
    old.require(sys.platform.startswith("linux") and ROOT == Path("/home/jsy/GNRL_20261012").resolve(), "Wrong host/project")
    old.require(old.sha256(ROOT / HELPER) == HELPER_SHA256, "Postqueue helper changed")
    old.require(type(args.expected_pid) is int and args.expected_pid > 0
                and type(args.expected_start_ticks) is int and args.expected_start_ticks > 0, "Missing explicit controller identity")
    old.require(len(args.expected_adoption_sha256) == 64 and all(c in "0123456789abcdef" for c in args.expected_adoption_sha256),
                "Expected adoption SHA must be lowercase SHA256")
    adoption_path, job_path, output = map(v.relative, (args.adoption_json, args.job_json, args.output))
    old.require(args.output.startswith("outputs/") and not output.exists() and output.parent.is_dir(), "Receipt path must be new")
    old.require(old.sha256(adoption_path) == args.expected_adoption_sha256, "Wrong explicit adoption bytes")
    adoption = read_metadata(args.adoption_json)
    old.require(adoption.get("kind") == "recovery_learning_parent_adoption" and adoption.get("adopted") is True
                and adoption.get("adopted_by") == "root_after_independent_GPT_6_Astra_review"
                and adoption["authorized_stages"] == list(STAGES) and adoption["physical_gpu"] == 2, "Wrong learning adoption")
    paths = adoption["output_paths"]
    status_path, locks_path = (v.relative(paths["queue"] + "/" + name) for name in ("status.json", "locked_sha256.json"))
    status_hash, locks_hash, job_hash = map(old.sha256, (status_path, locks_path, job_path))
    status, locks, job = (read_metadata(name) for name in
                        (paths["queue"] + "/status.json", paths["queue"] + "/locked_sha256.json", args.job_json))
    old.require(status["adoption_sha256"] == args.expected_adoption_sha256 and job["pid"] == args.expected_pid
                and int(job["start_ticks"]) == args.expected_start_ticks and Path(job["cwd"]).resolve() == ROOT,
                "Controller identity/adoption mismatch")
    v.exited(args.expected_pid, args.expected_start_ticks, args.expected_pid)
    read = read_metadata
    stages = metadata_barrier(adoption, status, locks, read, v.exited)
    required = {adoption[key] for key in ("candidate", "input_inventory", "diagnostic_parent_adoption", "diagnostic_status",
                "diagnostic_locks", "diagnostic_result_review", "controller_review", "training_review", "evaluator_review")}
    required.update((old.PROVENANCE, old.MODEL_INVENTORY))
    old.require(required <= adoption["file_sha256"].keys(), "Missing adopted metadata locks")
    old.require(v.source_inventory() == adoption["source_inventory"]
                and set(adoption["source_inventory"]) <= adoption["file_sha256"].keys()
                and all(locks.get(name) == h for name, h in adoption["file_sha256"].items()), "Frozen source/input inventory differs")
    candidate, previous_status = read(adoption["candidate"]), read(adoption["diagnostic_status"])
    end = v.inherited_deadline(status, previous_status, candidate)
    deadline = min(time.monotonic() + 300, time.monotonic() + end - time.time() - 30)
    def check():
        old.require(time.monotonic() < deadline and time.time() < end - 30, "Postqueue verification exhausted inherited deadline")
    old.require(len(adoption["original_file_sha256"]) == 336 and len(adoption["runtime_file_sha256"]) == 9,
                "Original336/runtime9 bindings missing")
    verified = v.measured(locks, check)
    originals = v.measured(adoption["original_file_sha256"], check, absolute=True)
    runtimes = v.measured(adoption["runtime_file_sha256"], check, absolute=True)
    inventory = read(adoption["input_inventory"])
    old.require(inventory.get("complete") is True and inventory["original_file_sha256"] == originals
                and inventory["base_full_inventory"] == old.MODEL_INVENTORY
                and inventory["base_full_inventory_sha256"] == verified[old.MODEL_INVENTORY], "Input provenance differs")
    model_inventory = read(old.MODEL_INVENTORY)
    model_root = Path(model_inventory["model_path"]).resolve()
    old.require(model_root == Path(v.MODEL).resolve() and candidate["frozen_start_policy"]["base"] == v.MODEL
                and candidate["frozen_start_policy"]["adapter"] == v.ADAPTER, "Base/initial policy changed")
    base = measure_base(model_root, model_inventory, check)
    initial = inventory["initial_adapter_sha256"]
    old.require(len(initial) == 11, "Starting adapter11 inventory missing")
    barrier_name = paths["queue"] + "/all_models_locked.json"
    old.require(barrier_name in verified and status["all_models_locked_sha256"] == verified[barrier_name]
                and paths["queue"] + "/result.json" in verified, "Missing final model/result locks")
    barrier = read(barrier_name)
    old.require(barrier["joint_token_audit_sha256"] == verified[paths["joint_token_audit"]], "Model barrier audit differs")
    all_adapters = measure_adapters(barrier, paths, verified, initial, check)
    check()
    old.require(old.sha256(status_path) == status_hash and old.sha256(locks_path) == locks_hash
                and old.sha256(job_path) == job_hash and old.sha256(adoption_path) == args.expected_adoption_sha256
                and old.sha256(ROOT / HELPER) == HELPER_SHA256, "Metadata/helper changed during verification")
    old.require(v.source_inventory() == adoption["source_inventory"], "Source inventory changed during verification")
    v.exited(args.expected_pid, args.expected_start_ticks, args.expected_pid)
    for row in stages:
        v.exited(row["pid"], row["start_ticks"], row["owned_session"])
    check()
    receipt = {"kind": "recovery_learning_postqueue_integrity", "complete": True, "errors": [], "created_utc": old.utc(),
        "adoption_sha256": args.expected_adoption_sha256, "queue_status_sha256": status_hash,
        "locked_sha256_file_sha256": locks_hash, "job_receipt_sha256": job_hash, "verified_file_sha256": verified,
        "verified_original_file_sha256": originals, "verified_runtime_file_sha256": runtimes,
        "source_inventory": v.source_inventory(), "base_inventory_sha256": verified[old.MODEL_INVENTORY],
        "base_files_verified": len(base), "base_measured_sha256": base, "initial_adapter_sha256": all_adapters[ARMS[0]],
        "all_models_adapter_sha256": all_adapters, "all_models_locked_sha256": verified[barrier_name],
        "controller": {"pid": args.expected_pid, "start_ticks": args.expected_start_ticks, "exited": True},
        "owned_workers_gone": True, "stages": stages, "technical_completed_evaluation_episodes": 192,
        "main_study_started_unix": status["main_study_started_unix"], "global_deadline_unix": end,
        "verification_finished_before_deadline": True, "outcome_records_parsed": False,
        "postqueue_helper_sha256": HELPER_SHA256}
    write_receipt(output, receipt)
    return {"complete": True, "receipt": args.output, "sha256": old.sha256(output),
            "verified_workspace_files": len(verified), "original_files": len(originals), "runtime_files": len(runtimes),
            "base_files": len(base), "adapter_file_counts": {arm: len(values) for arm, values in all_adapters.items()}}


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
        raise TimeoutError("Postqueue verification wall cap")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        print(json.dumps(verify(args)))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
