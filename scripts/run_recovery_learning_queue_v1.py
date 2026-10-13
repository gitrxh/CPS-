"""Conditionally adopted three recovery trainings and complete four-model seen48.

Inert imports. No retry, unseen evaluation or automatic experiment continuation.
All subprocesses are serial GPU2 workers; system Python owns their pidfd cleanup.
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
from scripts import run_recovery_diagnostic_queue_v1 as prior
from scripts import run_recovery_acquisition_v1 as acquisition
from scripts import run_successful_path_learning_v1 as old
from gnrl import train_recovery_v1 as trainer
from gnrl import recovery_evaluate_v1 as evaluator
from gnrl import recovery_projection_v1 as shared

CONTROLLER = "scripts/run_recovery_learning_queue_v1.py"
TEST = "tests/test_recovery_learning_queue_v1.py"
TRAIN = tuple(trainer.ARMS)
EVAL = tuple(evaluator.ARMS)
STAGES = ("joint_token_audit", *("train_" + a for a in TRAIN), *("eval_" + a for a in EVAL))


def require_passed_diagnostic(status, result, review, hashes):
    old.require(status.get("complete") is True and status.get("current_stage") == "needs_astra_diagnostic_review"
                and status.get("gate_evaluable") is True and status.get("training_gate_passed") is True,
                "Diagnostic did not completely pass; training forbidden")
    old.require(result.get("complete") is True and result.get("training_gate_passed") is True,
                "Actual-Q result did not pass")
    old.require(review.get("kind") == "recovery_complete_diagnostic_result_review"
                and review.get("recovery_diagnostic_result_review_passed") is True
                and review.get("training_gate_passed") is True
                and review.get("complete") is True and review.get("gate_evaluable") is True
                and review.get("errors") == [], "Independent passed-gate review missing")
    for key, h in hashes.items():
        old.require(review.get(key) == h, "Diagnostic review binding differs: " + key)


def validate_parent(path):
    a = old.read_json(path)
    for key, value in {"kind": "recovery_learning_parent_adoption", "version": 1, "adopted": True,
                       "adopted_by": "root_after_independent_GPT_6_Astra_review", "physical_gpu": 2}.items():
        old.require(type(a.get(key)) is type(value) and a[key] == value, "Learning adoption refused: " + key)
    old.require(a["authorized_stages"] == list(STAGES), "Stage order differs")
    old.require(a["source_inventory"] == prior.worker.source_inventory(), "Source inventory differs")
    required = set(a["source_inventory"]) | {CONTROLLER, TEST, a["candidate"], a["controller_review"],
        a["training_review"], a["evaluator_review"], a["diagnostic_result_review"], a["diagnostic_parent_adoption"],
        a["diagnostic_status"], a["diagnostic_locks"], a["input_inventory"], a["bc_data"],
        evaluator.MANIFEST, old.PROVENANCE, old.MODEL_INVENTORY}
    old.require(required <= a["file_sha256"].keys(), "Missing source/input/review locks")
    old.verify_hashes(a["file_sha256"])
    reviews = (("controller_review", "recovery_learning_controller_review_passed", "controller_sha256", CONTROLLER),
               ("training_review", "recovery_training_review_passed", "module_sha256", "gnrl/train_recovery_v1.py"),
               ("evaluator_review", "recovery_evaluator_review_passed", "evaluator_sha256", "gnrl/recovery_evaluate_v1.py"))
    for field, flag, hashkey, source in reviews:
        r = old.read_json(ROOT / a[field])
        old.require(r.get(flag) is True and r[hashkey] == a["file_sha256"][source]
                    and r["candidate_sha256"] == a["file_sha256"][a["candidate"]], "Exact implementation review missing: " + field)
    old.require(old.read_json(ROOT / a["controller_review"])["test_sha256"] == a["file_sha256"][TEST], "Controller tests unreviewed")
    adopted = old.read_json(ROOT / a["diagnostic_parent_adoption"])
    old.require(adopted.get("kind") == "recovery_diagnostic_parent_adoption" and adopted.get("adopted") is True
                and adopted["candidate"] == a["candidate"] and adopted["input_inventory"] == a["input_inventory"],
                "Wrong previous phase")
    paths = adopted["output_paths"]
    old.require(a["diagnostic_status"] == paths["queue"] + "/status.json"
                and a["diagnostic_locks"] == paths["queue"] + "/locked_sha256.json", "Wrong diagnostic paths")
    locks = old.read_json(ROOT / a["diagnostic_locks"])
    old.require(all(a["file_sha256"].get(n) == h for n, h in locks.items())
                and all(locks.get(n) == h for n, h in adopted["file_sha256"].items()), "Prior frozen bytes lost")
    for field, count in (("original_file_sha256", 336), ("runtime_file_sha256", 9)):
        old.require(a[field] == adopted[field] and len(a[field]) == count, "Prior inventory changed: " + field)
    old.require(str(Path(sys.executable).resolve()) in a["runtime_file_sha256"], "Controller executable unbound")
    old.require(a["file_sha256"][a["bc_data"]] == trainer.BC_SHA256
                and a["file_sha256"][evaluator.MANIFEST] == evaluator.MANIFEST_SHA256, "BC/seen48 freeze changed")
    status = old.read_json(ROOT / a["diagnostic_status"])
    result_path = paths["diagnostic"] + "/diagnostic.json"
    cost_path = paths["diagnostic"] + "/cost_progress.json"
    bindings = {"diagnostic_status_sha256": a["file_sha256"][a["diagnostic_status"]],
        "adoption_sha256": a["file_sha256"][a["diagnostic_parent_adoption"]],
        "diagnostic_locks_sha256": a["file_sha256"][a["diagnostic_locks"]],
        "diagnostic_result_sha256": a["file_sha256"][result_path],
        "diagnostic_cost_sha256": a["file_sha256"][cost_path],
        "score_records_sha256": a["file_sha256"][paths["score"] + "/records.jsonl"]}
    require_passed_diagnostic(status, old.read_json(ROOT / result_path),
                             old.read_json(ROOT / a["diagnostic_result_review"]), bindings)
    old.require(status["adoption_sha256"] == a["file_sha256"][a["diagnostic_parent_adoption"]]
                and [s["name"] for s in status["stages"]] == ["score", "diagnostic"], "Wrong diagnostic run")
    prior.assert_exited(a["diagnostic_controller_identity"])
    for stage in status["stages"]:
        old.require(stage.get("owned_cleanup_complete") is True and stage.get("returncode") == 0, "Prior cleanup incomplete")
        prior.assert_exited(stage, session=True)
    # Reconcile the last stage to its actual predecessor, preserving both-model scoring once.
    previous = old.read_json(ROOT / paths["score"] / "cost_progress.json")["totals"]
    costs = old.read_json(ROOT / cost_path)
    summary = old.read_json(ROOT / paths["diagnostic"] / "summary.json")
    totals = prior.validate_cost_lineage(previous, costs, summary)
    old.require(summary.get("complete") is True and summary.get("gate_evaluable") is True
                and summary.get("training_gate_passed") is True and status["cumulative_costs"] == totals,
                "Diagnostic ledger/gate incomplete")
    candidate = old.read_json(ROOT / a["candidate"])
    prior.inherited_deadline(status, candidate)
    old.require(candidate["conditional_training"]["order"] == list(TRAIN)
                and candidate["new_research_evaluation"]["fixed_order"] == list(EVAL), "Candidate order differs")
    output = a["output_paths"]
    old.require(set(output) == {"queue", "joint_token_audit", *("train_" + x for x in TRAIN), *("eval_" + x for x in EVAL)}, "Output keys differ")
    resolved = []
    for key, name in output.items():
        target = old.safe_relative(name)
        prefix = "checkpoints/recovery_projection_" if key.startswith("train_") else "outputs/recovery_projection_"
        old.require(not target.exists() and name.startswith(prefix),
                    "Output exists or outside owned namespace")
        resolved.append(target)
    old.require(len(set(resolved)) == len(resolved), "Duplicate output paths")
    for x in resolved:
        for y in resolved + [old.safe_relative(n) for n in paths.values()]:
            old.require(x == y or not (x.is_relative_to(y) or y.is_relative_to(x)), "Output namespaces overlap")
    return a, paths, status, totals, candidate


def validate_training(folder, arm, audit, audit_hash):
    """Check final counts and the real ordered forwards/backwards, not the NLL."""
    summary = old.read_json(folder / "summary.json")
    state = old.read_json(folder / "final/training_state.json")
    config = old.read_json(folder / "config.json")
    expected = audit["arms"][arm]
    tokens = expected["totals"]["target_tokens"]
    old.require(0 < tokens <= 2000000 and summary.get("complete") is True
                and summary.get("matched_step_budget_complete") is True and summary["stop_reason"] == "max_steps"
                and not summary["technical_errors"] and summary["optimizer_steps"] == summary["requested_steps"] == 100,
                "Incomplete training")
    old.require(config["arm"] == arm and config["settings"] == trainer.SETTINGS
                and config["protocol"] == state["protocol"] == "alfworld-causal-v1", "Wrong training protocol/arm")
    for record in (state, summary):
        old.require(record["optimizer_steps"] == 100 and record["target_tokens"] == tokens
                    and record["sampled_examples"] == 800 and record["component_exposures"] == {"bc": 400, "recovery": 400}
                    and record["token_audit_sha256"] == audit_hash and record["schedule_sha256"] == audit["schedule_sha256"],
                    "Final training state/audit mismatch")
    old.require(summary["actual_backward_target_tokens"] == summary["expected_target_tokens"] == tokens,
                "Backward token total differs")
    exposures = trainer.read_rows(folder / "example_exposures.jsonl")
    slots = expected["scheduled_examples"]
    old.require(len(slots) == 800 and len(exposures) == 1600, "Missing training exposure records")
    for index, slot in enumerate(slots):
        for offset, event in enumerate(("forward_started", "backward_completed")):
            row = exposures[index * 2 + offset]
            old.require(row["event"] == event and all(row.get(k) == v for k, v in slot.items()), "Training exposure order differs")
            if event == "backward_completed":
                old.require(math.isfinite(row["action_token_nll"]), "Nonfinite backward loss")
    metrics = trainer.read_rows(folder / "metrics.jsonl")
    old.require(len(metrics) == 101 and metrics[-1] == {"summary": summary}, "Training metrics incomplete")
    cumulative = 0
    for step, metric in enumerate(metrics[:-1], 1):
        window = slots[(step-1)*8:step*8]
        window_tokens = sum(slot["target_tokens"] for slot in window)
        cumulative += window_tokens
        weighted = sum(exposures[2*i+1]["action_token_nll"]*slots[i]["target_tokens"]
                       for i in range((step-1)*8,step*8))/window_tokens
        old.require(metric["optimizer_step"] == step and metric["window_target_tokens"] == window_tokens
                    and metric["target_tokens"] == cumulative and metric["sampled_examples"] == step*8
                    and metric["component_exposures"] == {"bc":step*4,"recovery":step*4}
                    and math.isfinite(metric["action_token_nll"])
                    and math.isclose(metric["action_token_nll"],weighted,rel_tol=1e-10,abs_tol=1e-10),
                    "Training step/exposure evidence differs")
    inventory = trainer.inventory_tree(folder / "final")
    old.require(old.read_json(folder / "final_adapter_inventory.json") == inventory
                and summary["final_adapter_files_sha256"] == inventory["files_sha256"]
                and summary["final_adapter_inventory_sha256"] == old.sha256(folder / "final_adapter_inventory.json"),
                "Final adapter inventory differs")
    return {"target_tokens": tokens, "input_tokens": expected["totals"]["input_tokens"],
            "prompt_tokens": expected["totals"]["prompt_tokens"], "forward_calls": 800,
            "backward_calls": 800, "optimizer_updates": 100, "exposures": {"bc": 400, "recovery": 400}}


def complete_table(games, episodes, metadata):
    old.require(len(games) == len(set(games)) == 48 and set(episodes) == set(EVAL), "Missing complete192 barrier")
    table = []
    for arm in EVAL:
        old.require(len(episodes[arm]) == 48, "Partial held-out outcomes forbidden")
        for row, game in zip(episodes[arm], games):
            evaluator.validate_episode(row, game)
    for i, game in enumerate(games):
        table.append({"gamefile": game, **metadata[game], "success": {a: episodes[a][i]["success"] for a in EVAL}})
    counts = {a: sum(r["success"][a] for r in table) for a in EVAL}
    ref = "reference_conditional_recovery"
    contrasts = {}
    for arm in EVAL:
        if arm == ref:
            continue
        gains = sum(r["success"][ref] and not r["success"][arm] for r in table)
        losses = sum(r["success"][arm] and not r["success"][ref] for r in table)
        contrasts[arm] = {"denominator": 48, "gains": gains, "losses": losses, "difference": gains-losses,
                          "success_rate_difference": (gains-losses)/48}
    groups = {}
    for key in ("family", "scene"):
        groups[key] = {value: {"denominator": sum(r[key] == value for r in table),
            "successes": {a: sum(r[key] == value and r["success"][a] for r in table) for a in EVAL}}
            for value in sorted({r[key] for r in table})}
    old.require(len(groups["family"]) == 6 and all(x["denominator"] == 8 for x in groups["family"].values()), "Family denominator differs")
    passed = all(counts[ref]-counts[a] >= 3 for a in TRAIN if a != ref) and counts[ref] > counts[EVAL[0]]
    return {"complete": True, "episodes": 192, "task_denominator": 48, "table": table, "successes": counts,
        "reference_minus_comparator": contrasts, "groups": groups, "exploratory_practical_screen_passed": passed,
        "interpretation": "One-seed correlated seen-task exploratory differences only; no independent-episode significance or automatic continuation."}


def validate_events(episodes, events):
    expected = []
    for row in episodes:
        expected.append({"kind":"evaluation_initial", "gamefile":row["gamefile"], "state":row["initial"]})
        expected.extend({"kind":"evaluation_transition", "gamefile":row["gamefile"], "transition":t}
                        for t in row["transitions"])
    old.require(events == expected, "Evaluation event stream/actual episode trace differs")


class LearningQueue(acquisition.Acquisition):
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.adoption_hash = old.sha256(path)
        self.a, self.prior_paths, status, self.previous_totals, self.candidate = validate_parent(path)
        self.deadline_unix, self.deadline = prior.inherited_deadline(status, self.candidate)
        self.locks = dict(self.a["file_sha256"])
        self.folder = ROOT / self.a["output_paths"]["queue"]
        self.folder.mkdir(parents=True, exist_ok=False)
        self.state = {"complete": False, "phase": "conditional_recovery_learning_seen48", "stages": [],
            "adoption_sha256": self.adoption_hash, "physical_gpu": 2,
            "main_study_started_unix": status["main_study_started_unix"], "global_deadline_unix": self.deadline_unix,
            "cumulative_costs": self.previous_totals, "training_costs": {}, "outcome_unblinded": False}
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                        HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        self.env.pop("GNRL_GPU_GUARD", None)
        self.python = str(ROOT / ".venv/bin/python")
        self.games = old.read_json(ROOT / evaluator.MANIFEST)["games"]
        self.barrier = None
        self.trained_inventories = {}
        self.update("preflight")

    def verify(self):
        super().verify()
        for directory, expected in self.trained_inventories.items():
            actual = {f.relative_to(ROOT).as_posix() for f in (ROOT / directory).rglob("*") if f.is_file()}
            old.require(actual == expected, "Completed adapter file inventory changed")
        if self.barrier is not None:
            evaluator.validate_model_barrier(self.barrier_request(), self.barrier, self.audit)
        old.require(time.monotonic() < self.deadline - 30, "Deadline exhausted during verification")

    def trainer_args(self, arm, audit=False):
        selection = self.prior_paths["score"]
        args = ["-m", "gnrl.train_recovery_v1", "--model", trainer.BASE_MODEL,
            "--initial-adapter", evaluator.CANONICAL, "--bc-data", self.a["bc_data"],
            "--recovery-data", selection + "/" + arm + ".jsonl", "--arm", arm,
            "--token-audit", self.a["output_paths"]["joint_token_audit"], "--selection-dir", selection]
        if audit:
            args += ["--audit-only"]
            for peer in TRAIN:
                if peer != arm:
                    args += ["--peer-recovery-data", peer + "=" + selection + "/" + peer + ".jsonl"]
        else:
            args += ["--output", self.a["output_paths"]["train_" + arm]]
        return args

    def owned_run(self, stage, args, ceiling=7200):
        self.verify()
        self.wait_gpu()
        self.verify()
        command = [self.python, "scripts/launch_gpu.py", "--gpus", "2", "--receipt",
            str(self.folder / (stage + "_gpu.json")), "--", self.python, *args]
        record = {"name": stage, "command": command, "started_utc": old.utc(), "owned_cleanup_complete": False}
        self.state["stages"].append(record)
        self.update(stage)
        process, ticks, code = None, None, None
        began = time.monotonic()
        signals = {signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM}
        with (self.folder / (stage + ".stdout.log")).open("xb") as out, (self.folder / (stage + ".stderr.log")).open("xb") as err:
            try:
                mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL,
                        stdout=out, stderr=err, start_new_session=True)
                    identity = old.process_record(process.pid)
                    old.require(identity and identity["session"] == process.pid, "Cannot establish own process identity")
                    ticks = identity["start_ticks"]
                    record.update(pid=process.pid, start_ticks=ticks, owned_session=process.pid)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                self.update(stage)
                old.require(self.deadline - time.monotonic() > 30, "No cleanup reserve")
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
                self.update(stage)
        old.require(code == 0, "Technical incomplete stage: " + stage)
        self.verify()
        self.lock([self.folder / (stage + "_gpu.json")])

    def barrier_request(self):
        return {"file_sha256": self.locks, "joint_token_audit": self.a["output_paths"]["joint_token_audit"],
                "adapter": evaluator.CANONICAL, "arm": EVAL[0]}

    def lock_models(self):
        adapters = {}
        for arm in EVAL:
            name = evaluator.CANONICAL if arm == EVAL[0] else self.a["output_paths"]["train_" + arm] + "/final"
            files = sorted(f for f in (ROOT / name).rglob("*") if f.is_file())
            self.lock(files)
            adapters[arm] = {"path": name, "file_sha256": {f.relative_to(ROOT).as_posix(): old.sha256(f) for f in files}}
        barrier = {"complete": True, "arms": list(EVAL), "adapters": adapters,
                   "joint_token_audit_sha256": self.locks[self.a["output_paths"]["joint_token_audit"]]}
        evaluator.validate_model_barrier(self.barrier_request(), barrier, self.audit)
        target = self.folder / "all_models_locked.json"
        old.atomic_json(target, barrier)
        self.lock([target])
        self.barrier = barrier
        self.update("all_four_models_locked", all_models_locked_sha256=old.sha256(target))

    def evaluation_request(self, arm):
        old.require(self.barrier is not None, "All four models must be locked before evaluation")
        index = EVAL.index(arm)
        previous = self.prior_paths["diagnostic"] if index == 0 else self.a["output_paths"]["eval_" + EVAL[index-1]]
        r = {"kind": "recovery_evaluation_stage_adoption", "schema_version": 1, "adopted": True,
             "adopted_by": "root_after_independent_GPT_6_Astra_review", "physical_gpu": 2, "arm": arm,
             "parent_adoption_sha256": self.adoption_hash, "adapter": self.barrier["adapters"][arm]["path"],
             "manifest": evaluator.MANIFEST, "candidate": self.a["candidate"], "implementation_review": self.a["evaluator_review"],
             "file_sha256": dict(self.locks), "source_inventory": self.a["source_inventory"],
             "original_file_sha256": self.a["original_file_sha256"], "runtime_file_sha256": self.a["runtime_file_sha256"],
             "all_models_locked_receipt": (self.folder / "all_models_locked.json").relative_to(ROOT).as_posix(),
             "joint_token_audit": self.a["output_paths"]["joint_token_audit"],
             "previous_cost_json": previous + "/cost_progress.json", "previous_summary": previous + "/summary.json",
             "global_deadline_unix": self.deadline_unix, "max_wall_seconds": 7200,
             "output": self.a["output_paths"]["eval_" + arm],
             "gpu_receipt": (self.folder / ("eval_" + arm + "_gpu.json")).relative_to(ROOT).as_posix()}
        target = self.folder / ("eval_" + arm + "_request.json")
        with target.open("x", encoding="utf-8") as stream:
            json.dump(r, stream, indent=2)
        self.lock([target])
        return target

    def accept_evaluation(self, arm, request):
        folder = ROOT / self.a["output_paths"]["eval_" + arm]
        summary, cost = old.read_json(folder / "summary.json"), old.read_json(folder / "cost_progress.json")
        old.require(summary.get("complete") is True and summary["stage"] == "evaluation" and summary["arm"] == arm
                    and summary["planned_episodes"] == summary["attempted_episodes"] == summary["completed_episodes"] == 48
                    and summary["technical_errors"] == 0 and summary["partial_performance_summary"] is None
                    and summary["request_sha256"] == old.sha256(request)
                    and summary["episodes_sha256"] == old.sha256(folder / "episodes.jsonl"), "Incomplete evaluation")
        old.require(cost["stage_limits"] == evaluator.STAGE_LIMITS, "Evaluation cost limits changed")
        self.previous_totals = prior.validate_cost_lineage(self.previous_totals, cost, summary)
        old.require(cost["stage_counts"]["resets_started"] == 48, "Evaluation reset count differs")
        self.lock(sorted(f for f in folder.rglob("*") if f.is_file()))
        self.update("completed_eval_" + arm, cumulative_costs=self.previous_totals)

    def finalize(self):
        old.require([r["name"] for r in self.state["stages"]] == list(STAGES)
                    and all(r.get("returncode") == 0 and r.get("owned_cleanup_complete") is True for r in self.state["stages"]),
                    "Full192 process barrier incomplete")
        self.verify()
        # Durable marker precedes the first outcome read, including later validation failures.
        self.update("validating_complete192", outcome_unblinded=True, outcome_unblinded_unix=time.time())
        episodes, metadata = {}, {}
        for game in self.games:
            raw = old.read_json(Path(game).with_name("traj_data.json"))
            metadata[game] = {"family": raw["task_type"], "scene": str(raw["scene"]["floor_plan"])}
        for arm in EVAL:
            folder = ROOT / self.a["output_paths"]["eval_" + arm]
            rows = trainer.read_rows(folder / "episodes.jsonl")
            validate_events(rows, trainer.read_rows(folder / "events.jsonl"))
            episodes[arm] = rows
            counts = old.read_json(folder / "cost_progress.json")["stage_counts"]
            old.require(counts["environment_steps_started"] == sum(r["environment_steps"] for r in rows)
                        and counts["generated_tokens"] == sum(r["generated_tokens"] for r in rows)
                        and counts["policy_calls_started"] == sum(len(r["transitions"]) for r in rows),
                        "Evaluation real traces/costs differ")
        report = complete_table(self.games, episodes, metadata)
        self.verify()
        report.update(outcome_unblinded=True, cumulative_costs=self.previous_totals, training_costs=self.state["training_costs"])
        old.atomic_json(self.folder / "result.json", report)
        self.lock([self.folder / "result.json"])
        self.verify()
        self.update("needs_astra_learning_result_review", complete=True,
                    exploratory_practical_screen_passed=report["exploratory_practical_screen_passed"])

    def incomplete(self, error):
        # Always overwrite any provisional result if a post-unblind check fails.
        latest_cost = None
        cost_error = None
        try:
            stages = self.state["stages"]
            if stages and stages[-1]["name"].startswith("eval_"):
                cost_path = ROOT / self.a["output_paths"][stages[-1]["name"]] / "cost_progress.json"
                if cost_path.exists():
                    latest_cost = old.read_json(cost_path)
                    old.require(set(latest_cost["totals"]) == set(shared.COUNTERS)
                                and all(type(v) is int and v >= self.previous_totals[k]
                                        for k, v in latest_cost["totals"].items()), "Failure cost ledger regressed")
        except Exception as exc:
            cost_error = f"{type(exc).__name__}: {exc}"
        report = {"complete": False, "outcome_unblinded": self.state["outcome_unblinded"], "contrasts": None,
            "exploratory_practical_screen_passed": None, "error": f"{type(error).__name__}: {error}",
            "cells": [{"arm": a, "gamefile": g, "outcome": None, "status": "technical_incomplete",
                       "outcome_access": "unblinding_started" if self.state["outcome_unblinded"] else "not_unblinded"}
                      for a in EVAL for g in self.games], "last_completed_cumulative_costs": self.previous_totals,
            "latest_worker_cost_receipt": latest_cost, "failure_cost_receipt_error": cost_error,
            "training_costs_completed": self.state["training_costs"],
            "incomplete_cost_note": "Interrupted calls can have unknown work. Preserve durable started-call/forward logs; do not interpret completed-stage totals as total incurred work."}
        result_name = (self.folder / "result.json").relative_to(ROOT).as_posix()
        invalidated = self.locks.pop(result_name, None)
        if invalidated:
            report["invalidated_provisional_result_sha256"] = invalidated
            old.atomic_json(self.folder / "locked_sha256.json", self.locks)
        old.atomic_json(self.folder / "result.json", report)
        self.update("technical_incomplete", complete=False, exploratory_practical_screen_passed=None, error=report["error"])

    def main(self):
        self.owned_run("joint_token_audit", self.trainer_args(TRAIN[0], audit=True), ceiling=1800)
        audit_path = ROOT / self.a["output_paths"]["joint_token_audit"]
        self.audit = old.read_json(audit_path)
        old.require(self.audit.get("kind") == "recovery_joint_training_token_audit_v1"
                    and self.audit.get("complete") is True and self.audit["settings"] == trainer.SETTINGS
                    and self.audit["score_sft_token_context_mismatches"] == 0
                    and self.audit["bc_examples"] == 349 and self.audit["anchor_count"] > 0
                    and self.audit["teacher_fields_used"] is False and self.audit["same_prompt_all_arms"] is True
                    and self.audit["score_sft_selected_labels_checked"] == 3*self.audit["anchor_count"]
                    and set(self.audit["arms"]) == set(TRAIN), "Joint score/SFT token audit incomplete")
        self.lock([audit_path])
        for arm in TRAIN:
            self.owned_run("train_" + arm, self.trainer_args(arm))
            folder = ROOT / self.a["output_paths"]["train_" + arm]
            self.state["training_costs"][arm] = validate_training(folder, arm, self.audit, old.sha256(audit_path))
            self.lock(sorted(f for f in folder.rglob("*") if f.is_file()))
            self.trained_inventories[(folder / "final").relative_to(ROOT).as_posix()] = {
                f.relative_to(ROOT).as_posix() for f in (folder / "final").rglob("*") if f.is_file()}
            self.update("completed_train_" + arm)
        self.lock_models()
        for arm in EVAL:
            request = self.evaluation_request(arm)
            self.owned_run("eval_" + arm, ["-m", "gnrl.recovery_evaluate_v1", "--adoption-json", str(request)])
            self.accept_evaluation(arm, request)
        self.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption-json", required=True)
    args = parser.parse_args()
    old.require(sys.platform.startswith("linux") and hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"),
                "Use system Python with pidfd support")
    old.require(ROOT.resolve() == Path("/home/jsy/GNRL_20261012").resolve(), "Wrong chemistry workspace")
    provisional = old.read_json(args.adoption_json)
    status = old.read_json(old.safe_relative(provisional["diagnostic_status"]))
    remaining = status["global_deadline_unix"] - time.time()
    old.require(math.isfinite(remaining) and 30 < remaining <= 171840, "Inherited deadline exhausted/invalid")
    def stop(signum, frame):
        raise InterruptedError("Owned learning controller interrupted")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    queue = None
    try:
        queue = LearningQueue(args.adoption_json)
        queue.main()
    except BaseException as error:
        if queue is not None:
            queue.incomplete(error)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
