"""Explicitly adopted score -> actual-Q phase, inheriting acquisition deadline.

This controller never trains or evaluates held-out tasks. See the local schema
description analysis/recovery_diagnostic_queue_adoption_schema_v1.json. Imports
are inert; system Python provides pidfd supervision, project Python runs workers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_successful_path_learning_v1 as old
from scripts import run_recovery_acquisition_v1 as acquisition
from scripts import score_recovery_diagnostic_v1 as worker
from gnrl import recovery_projection_v1 as p
from gnrl import recovery_diagnostic_v1 as d

STAGES = ("score", "diagnostic")
CONTROLLER = "scripts/run_recovery_diagnostic_queue_v1.py"
TEST = "tests/test_recovery_diagnostic_queue_v1.py"


def inherited_deadline(status, candidate):
    deadline = status["global_deadline_unix"]
    duration = candidate["finite_budgets"]["global_wall_seconds_including_waits_verification_cleanup"]
    old.require(candidate["version"] == "v2" and duration == 171840 and math.isfinite(deadline)
                and deadline - status["main_study_started_unix"] == duration,
                "Inherited study deadline changed")
    remaining = deadline - time.time()
    old.require(30 < remaining <= duration, "Inherited study deadline exhausted/invalid")
    return deadline, time.monotonic() + remaining


def assert_exited(identity, *, session=False):
    """Read-only liveness check; never signal a prior stage or another process."""
    pid, ticks = identity["pid"], identity["start_ticks"]
    old.require(type(pid) is int and pid > 0 and type(ticks) is int and ticks > 0,
                "Missing prior process identity")
    leader = old.process_record(pid)
    old.require(leader is None or leader["start_ticks"] == ticks, "Prior PID reused; ownership cannot be verified")
    old.require(leader is None or leader["state"] == "Z", "Prior controller/worker is still running")
    if session:
        old.require(identity["owned_session"] == pid and not old.owned_members(pid, ticks),
                    "Prior owned worker session has live members")


def validate_cost_lineage(previous, costs, summary):
    old.require(costs.get("kind") == "recovery_cost_ledger" and costs.get("active_call") is None
                and costs.get("unknown_generation_after_error") is False
                and costs.get("global_limits") == p.GLOBAL_LIMITS, "Incomplete/wrong shared cost ledger")
    totals, counts = costs["totals"], costs["stage_counts"]
    old.require(set(previous) == set(totals) == set(counts) == set(p.COUNTERS), "Missing cumulative counters")
    for key in p.COUNTERS:
        old.require(type(totals[key]) is int and type(counts[key]) is int and counts[key] >= 0
                    and totals[key] == previous[key] + counts[key], "Cost reset/double-count: " + key)
    for key, limit in p.GLOBAL_LIMITS.items():
        old.require(totals[key] <= limit and counts[key] <= costs["stage_limits"][key], "Cost ceiling exceeded")
    for prefix in ("environment_steps", "resets", "policy_calls"):
        old.require(totals[prefix + "_started"] == totals[prefix + "_completed"], "Unfinished prior call")
    old.require(summary["costs"]["totals"] == totals and summary["costs"]["stage_counts"] == counts,
                "Summary/shared ledger costs differ")
    return totals


def verify_score_cost(selections, scoring, costs):
    """Recompute both-reference target and padded forward work from all rows."""
    target = forward = calls = 0
    for selected in selections:
        if not selected["eligible"]:
            continue
        actions = selected["actions"]
        for role in ("reference_scores", "base_scores"):
            scores = selected[role]
            for offset in range(0, len(actions), 2):
                batch = [scores[a] for a in actions[offset:offset + 2]]
                old.require(len({row["prompt_tokens"] for row in batch}) == 1, "Batch prompt lengths differ")
                target += sum(row["action_tokens_including_eos"] for row in batch)
                forward += len(batch) * (batch[0]["prompt_tokens"] + max(row["action_tokens_including_eos"] for row in batch))
                calls += 1
    old.require(scoring["target_limit"] == 500000 and scoring["forward_limit"] == 7000000
                and target <= scoring["target_limit"] and forward <= scoring["forward_limit"],
                "Scoring budget differs/exceeded")
    old.require(scoring["active_call"] is None and scoring["calls_started"] == scoring["calls_completed"] == calls
                and scoring["scored_target_tokens"] == target and scoring["forward_tokens"] == forward,
                "Incomplete/double-counted/missing base or reference score work")
    counts = costs["stage_counts"]
    old.require(counts["forward_tokens"] == forward and all(counts[key] == 0 for key in p.COUNTERS if key != "forward_tokens"),
                "Score stage shared costs differ from both-reference score ledger")


def preserve_worker_unblinding(folder, state):
    """A failed worker may have crossed its complete-result read barrier.

    Read only that durable technical flag after cleanup; never inspect partial
    branch outcomes to decide whether or how to retry.
    """
    summary = folder / "summary.json"
    if summary.exists():
        receipt = old.read_json(summary)
        if receipt.get("outcome_unblinded") is True:
            state.update(outcome_unblinded=True,
                         worker_outcome_unblinded_unix=receipt.get("outcome_unblinded_unix"))


def mark_incomplete(queue, error):
    # Also runs if owned cleanup itself raised before its ordinary final update.
    try:
        preserve_worker_unblinding(ROOT / queue.a["output_paths"]["diagnostic"], queue.state)
    except Exception as receipt_error:
        if queue.state.get("outcome_unblinded") is not True:
            queue.state["outcome_unblinded"] = None
        queue.state["unblinding_receipt_error"] = f"{type(receipt_error).__name__}: {receipt_error}"
    queue.update("technical_incomplete", complete=False, gate_evaluable=False, training_gate_passed=None,
                 error=f"{type(error).__name__}: {error}")


def validate_parent(path):
    a = old.read_json(path)
    for key, value in {"kind": "recovery_diagnostic_parent_adoption", "version": 1, "adopted": True,
            "adopted_by": "root_after_independent_GPT_6_Astra_review", "physical_gpu": 2}.items():
        old.require(type(a.get(key)) is type(value) and a[key] == value, "Parent adoption refused: " + key)
    old.require(a["authorized_stages"] == list(STAGES), "Only score and diagnostic authorized")
    old.require(a["source_inventory"] == worker.source_inventory(), "New phase source inventory differs")
    required = set(a["source_inventory"]) | {CONTROLLER, TEST, a["candidate"], a["implementation_review"],
        a["controller_review"], a["acquisition_parent_adoption"], a["acquisition_status"], a["acquisition_locks"],
        a["acquisition_result_review"], a["input_inventory"], old.PROVENANCE, old.MODEL_INVENTORY}
    old.require(required <= a["file_sha256"].keys(), "Missing parent input/review locks")
    old.verify_hashes(a["file_sha256"])
    component = old.read_json(ROOT / a["implementation_review"])
    old.require(component.get("recovery_diagnostic_review_passed") is True
                and component["module_sha256"] == a["file_sha256"][worker.MODULE]
                and component["runner_sha256"] == a["file_sha256"][worker.RUNNER], "Exact worker review missing")
    review = old.read_json(ROOT / a["controller_review"])
    old.require(review.get("recovery_diagnostic_controller_review_passed") is True
                and review["controller_sha256"] == old.sha256(__file__)
                and review["test_sha256"] == a["file_sha256"][TEST], "Exact controller review missing")
    prior = old.read_json(ROOT / a["acquisition_parent_adoption"])
    old.require(prior.get("kind") == "recovery_acquisition_parent_adoption" and prior.get("adopted") is True
                and prior["candidate"] == a["candidate"] and prior["input_inventory"] == a["input_inventory"],
                "Wrong prior acquisition adoption")
    prior_paths = prior["output_paths"]
    old.require(a["acquisition_status"] == prior_paths["queue"] + "/status.json"
                and a["acquisition_locks"] == prior_paths["queue"] + "/locked_sha256.json", "Acquisition evidence path differs")
    status = old.read_json(ROOT / a["acquisition_status"])
    old.require(status.get("complete") is True and status.get("current_stage") == "needs_astra_acquisition_review"
                and status["adoption_sha256"] == a["file_sha256"][a["acquisition_parent_adoption"]]
                and [s["name"] for s in status["stages"]] == ["collect", "oracle", "replay"], "Acquisition is not complete")
    assert_exited(a["acquisition_controller_identity"])
    for stage in status["stages"]:
        old.require(stage.get("owned_cleanup_complete") is True and stage.get("returncode") == 0,
                    "Prior worker cleanup/completion not certified")
        assert_exited(stage, session=True)
    prior_locks = old.read_json(ROOT / a["acquisition_locks"])
    old.require(all(a["file_sha256"].get(name) == h for name, h in prior_locks.items()),
                "New phase must preserve every prior frozen input/output byte")
    for name, value in prior["file_sha256"].items():
        old.require(prior_locks.get(name) == value, "Acquisition lost an original lock")
    old.require(a["original_file_sha256"] == prior["original_file_sha256"]
                and a["runtime_file_sha256"] == prior["runtime_file_sha256"]
                and len(a["original_file_sha256"]) == 336 and len(a["runtime_file_sha256"]) == 9,
                "Original336/runtime9 inventories must be inherited unchanged")
    old.require(str(Path(sys.executable).resolve()) in a["runtime_file_sha256"], "Controller executable not runtime-bound")
    inventory = old.read_json(ROOT / a["input_inventory"])
    old.require(inventory.get("complete") is True and inventory["original_file_sha256"] == a["original_file_sha256"],
                "Original inventory evidence differs")
    for name, h in inventory["initial_adapter_sha256"].items():
        old.require(a["file_sha256"].get(name) == h, "Canonical adapter lock differs")
    replay_cost = prior_paths["replay"] + "/cost_progress.json"
    result_review = old.read_json(ROOT / a["acquisition_result_review"])
    old.require(result_review.get("recovery_acquisition_result_review_passed") is True
                and result_review["acquisition_status_sha256"] == a["file_sha256"][a["acquisition_status"]]
                and result_review["acquisition_locks_sha256"] == a["file_sha256"][a["acquisition_locks"]]
                and result_review["replay_cost_sha256"] == a["file_sha256"][replay_cost],
                "Independent complete acquisition result review missing")
    previous = dict.fromkeys(p.COUNTERS, 0)
    for name, count in (("collect", 60), ("oracle", 180), ("replay", 180)):
        folder = prior_paths[name]
        summary = old.read_json(ROOT / folder / "summary.json")
        costs = old.read_json(ROOT / folder / "cost_progress.json")
        old.require(summary.get("complete") is True and summary["stage"] == name
                    and summary["record_count"] == summary["planned_records"] == count
                    and summary["records_sha256"] == a["file_sha256"][folder + "/records.jsonl"], "Incomplete acquisition stage")
        previous = validate_cost_lineage(previous, costs, summary)
    old.require(status["cumulative_costs"] == previous, "Acquisition final cumulative costs differ")
    candidate = old.read_json(ROOT / a["candidate"])
    inherited_deadline(status, candidate)
    paths = a["output_paths"]
    old.require(set(paths) == {"queue", *STAGES} and len(set(paths.values())) == 3, "Wrong diagnostic output paths")
    resolved = []
    prior_resolved = [old.safe_relative(name) for name in prior_paths.values()]
    for name in paths.values():
        target = old.safe_relative(name)
        old.require(name.startswith("outputs/recovery_projection_") and not target.exists(), "Output exists/wrong namespace")
        resolved.append(target)
    all_paths = resolved + prior_resolved
    old.require(not any(x != y and (x.is_relative_to(y) or y.is_relative_to(x)) for x in resolved for y in all_paths),
                "Output namespaces overlap")
    return a, prior_paths, status, previous, candidate


class DiagnosticQueue(acquisition.Acquisition):
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.adoption_hash = old.sha256(path)
        self.a, self.prior_paths, prior_status, self.previous_totals, self.candidate = validate_parent(path)
        self.deadline_unix, self.deadline = inherited_deadline(prior_status, self.candidate)
        self.locks = dict(self.a["file_sha256"])
        self.folder = ROOT / self.a["output_paths"]["queue"]
        self.folder.mkdir(parents=True, exist_ok=False)
        self.state = {"complete": False, "phase": "score_and_actual_Q_only", "stages": [],
            "adoption_sha256": self.adoption_hash, "physical_gpu": 2,
            "main_study_started_unix": prior_status["main_study_started_unix"],
            "global_deadline_unix": self.deadline_unix, "cumulative_costs": self.previous_totals,
            "gate_evaluable": False, "training_gate_passed": None, "outcome_unblinded": False}
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1",
            HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        self.env.pop("GNRL_GPU_GUARD", None)
        self.python = str(ROOT / ".venv/bin/python")
        self.update("preflight")

    def request(self, stage):
        old.require(stage in STAGES, "Unapproved stage")
        paths = self.a["output_paths"]
        previous = self.prior_paths["replay"] if stage == "score" else paths["score"]
        request = {"schema_version": 1, "kind": "recovery_diagnostic_stage_adoption", "adopted": True,
            "adopted_by": "root_after_independent_GPT_6_Astra_review", "stage": stage, "physical_gpu": 2,
            "parent_adoption_sha256": self.adoption_hash,
            "candidate": self.a["candidate"], "candidate_sha256": self.locks[self.a["candidate"]],
            "source_inventory": self.a["source_inventory"], "file_sha256": dict(self.locks),
            "original_file_sha256": self.a["original_file_sha256"], "runtime_file_sha256": self.a["runtime_file_sha256"],
            "implementation_review": self.a["implementation_review"], "output": paths[stage],
            "pool_records": self.prior_paths["oracle"] + "/records.jsonl", "pool_summary": self.prior_paths["oracle"] + "/summary.json",
            "replay_records": self.prior_paths["replay"] + "/records.jsonl", "replay_summary": self.prior_paths["replay"] + "/summary.json",
            "previous_cost_json": previous + "/cost_progress.json", "previous_summary": previous + "/summary.json",
            "global_deadline_unix": self.deadline_unix,
            "max_wall_seconds": self.candidate["finite_budgets"]["stage_wall_seconds"]["scoring" if stage == "score" else "diagnostic_continuation"],
            "gpu_receipt": (self.folder / (stage + "_gpu.json")).relative_to(ROOT).as_posix()}
        if stage == "diagnostic":
            request.update(selection_records=paths["score"] + "/records.jsonl", selection_summary=paths["score"] + "/summary.json",
                           label_files={arm: paths["score"] + "/" + arm + ".jsonl" for arm in d.ARMS})
        target = self.folder / (stage + "_request.json")
        with target.open("x", encoding="utf-8") as stream:
            json.dump(request, stream, indent=2)
        self.lock([target])
        return target

    def owned_run(self, stage, request):
        self.verify()
        self.wait_gpu()
        self.verify()
        command = [self.python, "scripts/launch_gpu.py", "--gpus", "2", "--receipt",
            str(self.folder / (stage + "_gpu.json")), "--", self.python, worker.RUNNER, "--adoption-json", str(request)]
        record = {"name": stage, "started_utc": old.utc(), "command": command, "owned_cleanup_complete": False}
        self.state["stages"].append(record)
        self.update(stage)
        process, ticks, code = None, None, None
        began = time.monotonic()
        signals = {signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM}
        with (self.folder / (stage + ".stdout.log")).open("xb") as stdout, (self.folder / (stage + ".stderr.log")).open("xb") as stderr:
            try:
                mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL,
                        stdout=stdout, stderr=stderr, start_new_session=True)
                    ident = old.process_record(process.pid)
                    old.require(ident and ident["session"] == process.pid, "Cannot establish own process identity")
                    ticks = ident["start_ticks"]
                    record.update(pid=process.pid, start_ticks=ticks, owned_session=process.pid)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                self.update(stage)
                ceiling = old.read_json(request)["max_wall_seconds"]
                old.require(self.deadline - time.monotonic() > 30, "No cleanup reserve before wait")
                code = process.wait(timeout=min(ceiling, self.deadline - time.monotonic() - 30))
            finally:
                if process is not None:
                    mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
                    try:
                        old.cleanup_owned(process, ticks)
                        record["owned_cleanup_complete"] = True
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                record.update(returncode=process.returncode if process else None, elapsed_seconds=time.monotonic()-began)
                if stage == "diagnostic":
                    preserve_worker_unblinding(ROOT / self.a["output_paths"][stage], self.state)
                self.update(stage)
        old.require(code == 0, "Technical incomplete stage: " + stage)
        self.verify()
        return self.accept(stage, request)

    def accept(self, stage, request):
        folder = ROOT / self.a["output_paths"][stage]
        summary = old.read_json(folder / "summary.json")
        costs = old.read_json(folder / "cost_progress.json")
        old.require(summary.get("complete") is True and summary["stage"] == stage
                    and summary["request_sha256"] == old.sha256(request)
                    and summary["records_sha256"] == old.sha256(folder / "records.jsonl"), "Incomplete/different worker output")
        totals = validate_cost_lineage(self.previous_totals, costs, summary)
        req = old.read_json(request)
        if stage == "score":
            old.require(summary["record_count"] == 180 and summary.get("labels_locked_before_Q") is True,
                        "Three labels were not fully locked before Q")
            diagnostic_req = dict(req, stage="diagnostic", selection_records=req["output"] + "/records.jsonl",
                selection_summary=req["output"] + "/summary.json",
                label_files={arm: req["output"] + "/" + arm + ".jsonl" for arm in d.ARMS})
            diagnostic_req["file_sha256"] = dict(req["file_sha256"])
            for file in folder.rglob("*"):
                if file.is_file():
                    diagnostic_req["file_sha256"][file.relative_to(ROOT).as_posix()] = old.sha256(file)
            _, selected = worker.inputs(diagnostic_req, self.candidate)
            scoring = old.read_json(folder / "score_ledger.json")
            old.require(summary["scoring"] == scoring, "Score summary differs from durable ledger")
            verify_score_cost(selected, scoring, costs)
        else:
            old.require(summary.get("gate_evaluable") is True and type(summary.get("training_gate_passed")) is bool,
                        "Technical diagnostic gate is not evaluable")
            self.update("validating_complete_diagnostic", outcome_unblinded=True, outcome_unblinded_utc=old.utc())
            pools, selected = worker.inputs(req, self.candidate)
            episodes = worker.rows(folder / "records.jsonl")
            old.require(summary["record_count"] == len(episodes), "Actual-Q record count differs")
            result = d.diagnostic_summary(pools, selected, self.candidate["train_only_diagnostic_gate"]["frozen_anchors"], episodes)
            stored = old.read_json(folder / "diagnostic.json")
            old.require(stored == result and summary["diagnostic_sha256"] == old.sha256(folder / "diagnostic.json")
                        and summary["training_gate_passed"] is result["training_gate_passed"], "Recomputed diagnostic gate differs")
            counts = costs["stage_counts"]
            old.require(counts["resets_started"] == len(episodes)
                        and counts["environment_steps_started"] == sum(r["environment_steps_including_prefix"] for r in episodes)
                        and counts["generated_tokens"] == sum(r["generated_tokens"] for r in episodes)
                        and counts["policy_calls_started"] == sum(len(r["transitions"])-1 for r in episodes),
                        "Actual-Q traces and cumulative environment/generation work differ")
            self.state.update(gate_evaluable=True, training_gate_passed=result["training_gate_passed"])
        self.lock(sorted(file for file in folder.rglob("*") if file.is_file()))
        self.lock([self.folder / (stage + "_gpu.json")])
        self.previous_totals = totals
        self.update("completed_" + stage, cumulative_costs=totals)

    def main(self):
        for stage in STAGES:
            request = self.request(stage)
            self.owned_run(stage, request)
        self.verify()
        self.update("needs_astra_diagnostic_review", complete=True,
            note="Score and actual-Q complete only. No training, held-out or unseen jobs authorized.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption-json", required=True)
    args = parser.parse_args()
    old.require(sys.platform.startswith("linux") and hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"),
                "Use system Python with pidfd support")
    old.require(ROOT.resolve() == Path("/home/jsy/GNRL_20261012").resolve(), "Wrong chemistry workspace")
    # Arm the inherited absolute deadline before expensive adoption verification.
    provisional = old.read_json(args.adoption_json)
    status = old.read_json(old.safe_relative(provisional["acquisition_status"]))
    remaining = status["global_deadline_unix"] - time.time()
    old.require(math.isfinite(remaining) and 30 < remaining <= 171840, "Inherited deadline exhausted/invalid")
    def stop(signum, frame):
        raise InterruptedError("Owned diagnostic controller interrupted")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    queue = None
    try:
        queue = DiagnosticQueue(args.adoption_json)
        queue.main()
    except BaseException as error:
        if queue is not None:
            mark_incomplete(queue, error)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
