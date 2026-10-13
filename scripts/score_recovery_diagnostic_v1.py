"""Adoption-gated recovery score/diagnostic workers; never a controller.

Stage request uses kind=recovery_diagnostic_stage_adoption, schema_version=1,
adopted=true, adopted_by=root_after_independent_GPT_6_Astra_review,
stage=score|diagnostic, physical_gpu=2, candidate/candidate_sha256,
source_inventory (all gnrl/scripts/tests Python), file_sha256 (all local inputs),
original_file_sha256 (all60 games/metadata), runtime_file_sha256 (absolute files),
implementation_review (recovery_diagnostic_review_passed and module/runner hashes),
pool_records/pool_summary, replay_records/replay_summary, previous_cost_json,
previous_summary, output, global_deadline_unix, max_wall_seconds, gpu_receipt.
diagnostic additionally needs selection_records/selection_summary/label_files
(arm->path). All inputs except freshly written launch_gpu receipt are hash-bound.
Full base model inventories and every file in the canonical adapter are verified.
Parent controller owns process cleanup, absolute deadline and ledger lineage.

Outputs: records.jsonl (180 selections or <=288 actual Q episodes), events.jsonl,
config.json, cost_progress.json, summary.json. Score also writes three arm JSONL
files and score_ledger.json. Diagnostic writes a complete36 table only after all
actual branches and byte verification; technical failures never receive zeros.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gnrl import recovery_diagnostic_v1 as d
from gnrl import recovery_projection_v1 as p
from scripts.collect_recovery_projection_v1 import (sha, safe_path, atomic_json, read_json, source_inventory)

MODEL = "/home/jsy/zhangranxu-adrs-week-20260929/models/Qwen2.5-3B-Instruct"
ADAPTER = "checkpoints/successful_path_canonical_seed42_v1/final"
MODULE = "gnrl/recovery_diagnostic_v1.py"
RUNNER = "scripts/score_recovery_diagnostic_v1.py"
TEST = "tests/test_recovery_diagnostic_v1.py"
PROVENANCE = "outputs/reference_model_provenance_v1.json"
MODEL_INVENTORY = "outputs/reference_model_inventory_v2.json"


def rows(path):
    raw = Path(path).read_bytes()
    p.require(not raw or raw.endswith(b"\n"), "Truncated JSONL input")
    return [json.loads(line) for line in raw.splitlines()]


def check_bytes(request):
    p.require(request["source_inventory"] == source_inventory(), "Source inventory changed")
    for name, expected in request["file_sha256"].items():
        p.require(sha(safe_path(name)) == expected, "Bound file changed: " + name)
    for mapping in (request["original_file_sha256"], request["runtime_file_sha256"]):
        for name, expected in mapping.items():
            p.require(Path(name).is_absolute() and sha(name) == expected, "Original/runtime bytes changed")
    adapter = safe_path(ADAPTER)
    actual = {f.relative_to(ROOT).as_posix() for f in adapter.rglob("*") if f.is_file()}
    bound = {name for name in request["file_sha256"] if name.startswith(ADAPTER + "/")}
    p.require(actual == bound and {ADAPTER + "/adapter_config.json", ADAPTER + "/training_state.json"} <= actual
              and any(name.endswith(("adapter_model.safetensors", "adapter_model.bin")) for name in actual),
              "Canonical adapter inventory differs")
    from gnrl.successful_path_selection_v1 import verify_model
    verify_model(safe_path(PROVENANCE), MODEL, safe_path(MODEL_INVENTORY))


def verify_gpu_receipt(request):
    from gnrl.policy import require_gpu_guard
    require_gpu_guard()
    receipt = read_json(safe_path(request["gpu_receipt"]))
    devices = receipt["devices"]
    p.require(len(devices) == 1 and devices[0]["physical_index"] == 2
              and os.environ["CUDA_VISIBLE_DEVICES"] == devices[0]["uuid"], "Physical GPU2 receipt differs")


def validate_request(path):
    request = read_json(path)
    for key, expected in {"kind": "recovery_diagnostic_stage_adoption", "schema_version": 1,
            "adopted": True, "adopted_by": "root_after_independent_GPT_6_Astra_review", "physical_gpu": 2}.items():
        p.require(type(request.get(key)) is type(expected) and request[key] == expected, "Stage adoption refused: " + key)
    stage = request["stage"]
    p.require(stage in ("score", "diagnostic"), "Unknown diagnostic worker stage")
    bound = request["file_sha256"]
    required = set(request["source_inventory"]) | {MODULE, RUNNER, TEST, PROVENANCE, MODEL_INVENTORY,
        request["candidate"], request["implementation_review"], request["pool_records"], request["pool_summary"],
        request["replay_records"], request["replay_summary"], request["previous_cost_json"], request["previous_summary"]}
    if stage == "diagnostic":
        p.require(set(request["label_files"]) == set(d.ARMS), "Missing frozen three-arm labels")
        required |= {request["selection_records"], request["selection_summary"], *request["label_files"].values()}
    p.require(required <= bound.keys(), "Missing adoption input bindings")
    check_bytes(request)
    p.require(request["candidate_sha256"] == bound[request["candidate"]], "Candidate hash differs")
    candidate = read_json(safe_path(request["candidate"]))
    review = read_json(safe_path(request["implementation_review"]))
    p.require(review.get("recovery_diagnostic_review_passed") is True and review.get("module_sha256") == bound[MODULE]
              and review.get("runner_sha256") == bound[RUNNER], "Exact independent implementation review required")
    budgets = candidate["finite_budgets"]
    p.require(p.GLOBAL_LIMITS == {"environment_steps_started": budgets["environment_step_upper"]["hard_global_step_calls_started"],
        "resets_started": budgets["max_total_reset_calls"], "generated_tokens": budgets["max_total_generated_tokens"]},
        "Shared ledger and adopted main budgets differ")
    p.require(candidate["frozen_start_policy"]["base"] == MODEL and candidate["frozen_start_policy"]["adapter"] == ADAPTER
              and candidate["train_only_diagnostic_gate"]["fixed_anchor_count"] == 36
              and budgets["physical_gpu"] == 2, "Candidate model/diagnostic/GPU protocol differs")
    frozen_seen = candidate["new_research_evaluation"]["frozen_metadata_artifacts"]
    for key in ("manifest", "report"):
        p.require(bound.get(frozen_seen[key]) == frozen_seen[key + "_sha256"], "Research48 was not bound before acquisition")
    games = candidate["training_cohort"]["frozen_gamefiles"]
    p.require(len(games) == len(set(games)) == 60, "Wrong complete acquisition cohort")
    for game in games:
        for original in (game, str(Path(game).with_name("traj_data.json"))):
            p.require(original in request["original_file_sha256"], "Unbound original game/metadata")
    p.require(str(Path(sys.executable).resolve()) in request["runtime_file_sha256"], "Worker executable not bound")
    ceiling = budgets["stage_wall_seconds"]["scoring" if stage == "score" else "diagnostic_continuation"]
    wall = request["max_wall_seconds"]
    remaining = request["global_deadline_unix"] - time.time()
    p.require(type(wall) in (int, float) and math.isfinite(wall) and 0 < wall <= ceiling and
              math.isfinite(remaining) and 30 < remaining <= budgets["global_wall_seconds_including_waits_verification_cleanup"],
              "Stage/global deadline differs")
    previous = read_json(safe_path(request["previous_cost_json"]))
    summary = read_json(safe_path(request["previous_summary"]))
    p.require(request["previous_summary"] == request["replay_summary" if stage == "score" else "selection_summary"],
              "Cost lineage does not identify immediate input stage")
    p.require(previous["kind"] == "recovery_cost_ledger" and previous["active_call"] is None
              and previous["unknown_generation_after_error"] is False and summary.get("complete") is True
              and summary["stage"] == ("replay" if stage == "score" else "score")
              and previous["totals"] == summary["costs"]["totals"], "Previous stage/cost lineage incomplete")
    initial = previous["totals"]
    p.require(all(initial[a] == initial[b] for a, b in (("resets_started", "resets_completed"),
        ("environment_steps_started", "environment_steps_completed"), ("policy_calls_started", "policy_calls_completed"))),
        "Previous work contains unfinished calls")
    output = safe_path(request["output"])
    p.require(request["output"].startswith("outputs/recovery_projection_") and not output.exists(), "Output already exists/wrong namespace")
    return request, candidate, output, initial


def inputs(request, candidate):
    pools = rows(safe_path(request["pool_records"]))
    replay = rows(safe_path(request["replay_records"]))
    expected = [(g, k) for g in candidate["training_cohort"]["frozen_gamefiles"] for k in p.ANCHOR_STEPS]
    p.require(len(pools) == len(replay) == 180 and [(r["anchor"]["gamefile"], r["anchor"]["after_actions"]) for r in pools] == expected,
              "Incomplete/wrong180 acquisition slots")
    for key, stage in (("pool", "oracle"), ("replay", "replay")):
        summary = read_json(safe_path(request[key + "_summary"]))
        p.require(summary.get("complete") is True and summary.get("stage") == stage
                  and summary["records_sha256"] == request["file_sha256"][request[key + "_records"]],
                  "Acquisition/replay summary mismatch")
    for pool, check in zip(pools, replay):
        d.pool_actions(pool)
        p.require(check["complete"] is True and check["anchor_id"] == pool["anchor_id"]
                  and check["pool_sha256"] == pool["artifact_sha256"] and check["candidates"] == [
                      {"action": c["action"], "certificate_sha256": c["certificate_sha256"],
                       "replay_verified": True, "actual_transition": c["first_transition"]} for c in pool["retained_candidates"]],
                  "Missing/different independent candidate first-action replay")
    selected = None
    if request["stage"] == "diagnostic":
        summary = read_json(safe_path(request["selection_summary"]))
        p.require(summary.get("complete") is True and summary.get("stage") == "score"
                  and summary["records_sha256"] == request["file_sha256"][request["selection_records"]]
                  and summary.get("labels_locked_before_Q") is True, "Selection not complete/locked before Q")
        selected = rows(safe_path(request["selection_records"]))
        p.require(len(selected) == len(pools), "Missing selection slots")
        labels = {arm: [] for arm in d.ARMS}
        for pool, row in zip(pools, selected):
            for arm, label in d.label_rows(pool, row).items():
                labels[arm].append(label)
        for arm in d.ARMS:
            p.require(rows(safe_path(request["label_files"][arm])) == labels[arm]
                      and summary["label_sha256"][arm] == request["file_sha256"][request["label_files"][arm]],
                      "Frozen arm label bytes/content differ")
    return pools, selected


def run_stage(path):
    request_hash = sha(path)
    request, candidate, output, initial = validate_request(path)
    pools, selected = inputs(request, candidate)
    verify_gpu_receipt(request)
    stage = request["stage"]
    budgets = candidate["finite_budgets"]
    deadline = time.monotonic() + min(request["max_wall_seconds"], request["global_deadline_unix"] - time.time() - 30)
    limits = {"environment_steps_started": 0 if stage == "score" else budgets["environment_step_upper"]["diagnostic_including_prefix"],
              "resets_started": 0 if stage == "score" else budgets["max_diagnostic_learner_branches"],
              "generated_tokens": 0 if stage == "score" else budgets["max_diagnostic_learner_branches"] * 100 * 64}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "config.json", {"request": request, "request_sha256": request_hash,
                "worker_executable": sys.executable, "protocol": "alfworld-causal-v1"})
    global_limits = {"environment_steps_started": budgets["environment_step_upper"]["hard_global_step_calls_started"],
                     "resets_started": budgets["max_total_reset_calls"], "generated_tokens": budgets["max_total_generated_tokens"]}
    ledger = p.BudgetLedger(stage_limits=limits, global_limits=global_limits, deadline=deadline, initial_totals=initial,
                           persist=lambda row: atomic_json(output / "cost_progress.json", row))
    ledger.flush()
    summary = {"kind": "recovery_diagnostic_stage_summary", "stage": stage, "complete": False,
               "gate_evaluable": False, "training_gate_passed": None, "outcome_unblinded": False,
               "request_sha256": request_hash, "retry_authorized": False}
    count = 0
    with (output / "records.jsonl").open("x", encoding="utf-8", buffering=1) as records, \
         (output / "events.jsonl").open("x", encoding="utf-8", buffering=1) as events:
        def write(stream, row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        def emit(row):
            write(events, row)

        def factory(gamefile, expert_type):
            from gnrl.alfworld_env import AlfworldReplayEnv
            return AlfworldReplayEnv(gamefile, seed=p.SEED, max_steps=100, expert_type=expert_type)

        try:
            from gnrl.policy import HFCausalPolicy
            ledger.check()
            if stage == "score":
                def save_scores(row):
                    atomic_json(output / "score_ledger.json", row)
                    ledger.counts["forward_tokens"] = row["forward_tokens"]
                    ledger.flush()

                score_ledger = d.ScoreLedger(target_limit=budgets["max_scored_target_tokens"],
                    forward_limit=budgets["max_scoring_forward_tokens"], deadline=deadline,
                    persist=save_scores)
                score_ledger.persist(score_ledger.snapshot())
                by_role = {}
                for role, adapter in (("reference", str(safe_path(ADAPTER))), ("base", None)):
                    policy = HFCausalPolicy(MODEL, adapter, max_context_tokens=2048, max_action_tokens=64, score_batch_size=2)
                    by_role[role] = []
                    for pool in pools:
                        ledger.check()
                        scores = score_ledger.score(policy, pool, role, emit) if pool["eligible"] else {}
                        by_role[role].append(scores)
                        print(json.dumps({"stage": stage, "role": role, "completed_slots": len(by_role[role])}), flush=True)
                    del policy
                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()
                labels = {arm: [] for arm in d.ARMS}
                for index, pool in enumerate(pools):
                    row = d.selection(pool, by_role["reference"][index], by_role["base"][index])
                    write(records, row)
                    for arm, label in d.label_rows(pool, row).items():
                        labels[arm].append(label)
                    count += 1
                for arm in d.ARMS:
                    with (output / (arm + ".jsonl")).open("x", encoding="utf-8") as stream:
                        for row in labels[arm]:
                            write(stream, row)
                summary.update(labels_locked_before_Q=True, eligible_anchors=len(labels[d.ARMS[0]]),
                    label_sha256={arm: sha(output / (arm + ".jsonl")) for arm in d.ARMS}, scoring=score_ledger.snapshot())
            else:
                policy = HFCausalPolicy(MODEL, str(safe_path(ADAPTER)), max_context_tokens=2048, max_action_tokens=64)
                planned = candidate["train_only_diagnostic_gate"]["frozen_anchors"]
                indexed = {(r["anchor"]["gamefile"], r["anchor"]["after_actions"]): r for r in pools}
                episodes = []
                for anchor in planned:
                    ledger.check()
                    pool = indexed[(anchor["gamefile"], anchor["after_actions"])]
                    if pool["eligible"]:
                        for action in d.pool_actions(pool):
                            c = next(c for c in pool["retained_candidates"] if c["action"] == action)
                            row = d.learner_branch(factory, policy, pool, c, ledger, emit)
                            write(records, row)
                            episodes.append(row)
                            count += 1
                            print(json.dumps({"stage": stage, "completed_branches": count}), flush=True)
                # No scientific gate/comparison is evaluated before the complete barrier.
                ledger.check()
                check_bytes(request)
                p.require(sha(path) == request_hash, "Adoption changed during diagnostic")
                summary.update(outcome_unblinded=True, outcome_unblinded_unix=time.time())
                atomic_json(output / "summary.json", summary)
                result = d.diagnostic_summary(pools, selected, planned, episodes)
                atomic_json(output / "diagnostic.json", result)
                summary.update(gate_evaluable=True, training_gate_passed=result["training_gate_passed"],
                               diagnostic_sha256=sha(output / "diagnostic.json"))
            ledger.check()
            check_bytes(request)
            p.require(sha(path) == request_hash, "Adoption changed during stage")
            summary.update(complete=True, records_sha256=sha(output / "records.jsonl"))
        except BaseException as error:
            summary.update(complete=False, gate_evaluable=False, training_gate_passed=None,
                           error=f"{type(error).__name__}: {error}", technical_incomplete=True)
            if stage == "diagnostic":
                incomplete = {"complete": False, "gate_evaluable": False,
                    "outcome_unblinded": summary["outcome_unblinded"], "comparisons": None,
                    "cells": [{**a, "Q": None, "difference_q_uniform": None, "difference_q_expert": None,
                               "status": "technical_incomplete"}
                              for a in candidate["train_only_diagnostic_gate"]["frozen_anchors"]]}
                atomic_json(output / "diagnostic_incomplete.json", incomplete)
                if (output / "diagnostic.json").exists():
                    atomic_json(output / "diagnostic.json", incomplete)
                summary.pop("diagnostic_sha256", None)
            raise
        finally:
            summary.update(record_count=count, costs=ledger.snapshot())
            ledger.flush()
            atomic_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption-json", required=True, type=Path)
    args = parser.parse_args(argv)
    p.require(sys.platform.startswith("linux") and ROOT.resolve() == Path("/home/jsy/GNRL_20261012").resolve(),
              "Workers run only in adopted chemistry workspace")
    os.chdir(ROOT)
    run_stage(args.adoption_json)


if __name__ == "__main__":
    main()
