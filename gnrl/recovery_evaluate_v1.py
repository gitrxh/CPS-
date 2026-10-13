"""Metered seen48 evaluator with the existing unfiltered greedy policy protocol.

Adds neutral raw-state evidence and shared cost accounting. It does not alter
prompts, repair commands, request expert information or inspect another model's
outcomes. Imports do not construct a model or environment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from . import recovery_projection_v1 as shared
from .policy import HFCausalPolicy, PROTOCOL_VERSION, require_gpu_guard

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/home/jsy/zhangranxu-adrs-week-20260929/models/Qwen2.5-3B-Instruct"
ARMS = ("frozen_canonical_BC", "uniform_viable_recovery", "expert_recovery", "reference_conditional_recovery")
MANIFEST = "outputs/recovery_seen48_freeze_v1/manifest.json"
MANIFEST_SHA256 = "a718d0f77cccba4c839a0612baeec72d48d444d05232c9edd6ad64c021a9548a"
CANONICAL = "checkpoints/successful_path_canonical_seed42_v1/final"
STAGE_LIMITS = {"environment_steps_started": 4800, "resets_started": 48, "generated_tokens": 350000}


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_path(name):
    p = Path(name)
    shared.require(isinstance(name, str) and not p.is_absolute() and ".." not in p.parts
                   and p.as_posix() == name, "Expected normalized workspace-relative path")
    p = (ROOT / p).resolve()
    shared.require(p.is_relative_to(ROOT.resolve()), "Path escapes workspace")
    return p


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def evaluate_task(env, policy, gamefile, ledger, emit=lambda row: None):
    shared.require("/json_2.1.1/valid_seen/" in gamefile, "Only frozen seen research tasks")
    shared.check_env(env, gamefile, None)
    current = ledger.reset(env, "heldout")
    initial = shared.state_record(current)
    shared.require(initial["canonical_state"]["elapsed_steps"] == 0, "Reset count differs")
    emit({"kind": "evaluation_initial", "gamefile": gamefile, "state": initial})
    trace, task = [], initial["observation"]
    while not current.done and len(trace) < 100:
        if not current.admissible_commands:
            break
        prompt = shared.context(task, trace, shared.state_record(current))
        action, details = ledger.generate(policy, prompt, label="heldout")
        after = ledger.step(env, action, "heldout")
        t = shared.transition(current, action, after, len(trace), details)
        t["context_sha256"] = shared.digest(prompt)
        trace.append(t)
        emit({"kind": "evaluation_transition", "gamefile": gamefile, "transition": t})
        current = after
    shared.require(current.done or not current.admissible_commands, "Evaluation incomplete at100-step boundary")
    row = {"schema_version": 1, "kind": "recovery_seen_episode", "complete": True,
           "gamefile": gamefile, "seed": 20261012, "task": task, "initial": initial,
           "transitions": trace, "environment_steps": len(trace),
           "generated_tokens": sum(t["policy"]["generated_tokens"] for t in trace),
           "success": bool(initial["won"] or any(t["after"]["won"] for t in trace)),
           "stop_reason": "environment_done" if current.done else "no_admissible_commands",
           "final_state": shared.state_record(current)}
    row["artifact_sha256"] = shared.digest(row)
    return row


def validate_state(state, elapsed):
    canonical = state["canonical_state"]
    shared.require(isinstance(state["observation"], str) and isinstance(canonical["facts"], list)
                   and type(canonical["elapsed_steps"]) is int and canonical["elapsed_steps"] == elapsed
                   and canonical["max_steps"] == 100 and type(canonical["moves"]) is int
                   and 0 <= canonical["moves"] <= elapsed, "Neutral raw-state counts differ")
    commands = state["admissible_commands"]
    shared.require(isinstance(commands, list) and all(isinstance(a, str) for a in commands)
                   and commands == sorted(set(commands)) and commands == canonical["admissible_commands"],
                   "Neutral admissible-command evidence differs")
    shared.require(all(type(canonical[k]) is bool for k in ("won", "lost", "terminated", "truncated", "done"))
                   and type(state["won"]) is bool and type(state["done"]) is bool
                   and state["won"] is canonical["won"] and state["done"] is canonical["done"]
                   and canonical["done"] == (canonical["terminated"] or canonical["truncated"])
                   and canonical["truncated"] == (elapsed >= 100 and not canonical["terminated"])
                   and (not (canonical["won"] or canonical["lost"]) or canonical["terminated"]),
                   "Neutral outcome/termination flags contradict raw state")
    shared.require(type(state["reward"]) in (int, float) and math.isfinite(state["reward"])
                   and state["reward"] == canonical["reward"] and type(canonical["score"]) in (int, float)
                   and math.isfinite(canonical["score"]), "Neutral reward evidence differs")


def validate_episode(row, gamefile):
    body = dict(row)
    h = body.pop("artifact_sha256")
    shared.require(shared.digest(body) == h and row.get("schema_version") == 1 and row.get("complete") is True
                   and row.get("kind") == "recovery_seen_episode" and row["gamefile"] == gamefile
                   and row["seed"] == 20261012 and "/valid_seen/" in gamefile, "Episode schema/identity differs")
    previous, trace = row["initial"], row["transitions"]
    validate_state(previous, 0)
    shared.require(row["task"] == previous["observation"] and previous["canonical_state"]["elapsed_steps"] == 0
                   and row["environment_steps"] == len(trace) <= 100, "Task/initial count differs")
    for i, t in enumerate(trace):
        shared.require(t["step"] == i and t["before"] == previous and not previous["done"]
                       and bool(previous["admissible_commands"])
                       and t["after"]["canonical_state"]["elapsed_steps"] == i + 1, "Broken actual episode chain")
        details = t["policy"]
        shared.require(t["context_sha256"] == shared.digest(shared.context(row["task"], trace[:i], previous))
                       and details.get("selection") == "generate"
                       and details["raw_completion"].strip() == t["action"]
                       and details["is_admissible"] is (t["action"] in previous["admissible_commands"])
                       and type(details["generated_tokens"]) is int and 0 <= details["generated_tokens"] <= 64,
                       "Generated action was repaired or has invalid cost evidence")
        previous = t["after"]
        validate_state(previous, i + 1)
    shared.require(row["final_state"] == previous and (previous["done"] or not previous["admissible_commands"])
                   and row["stop_reason"] == ("environment_done" if previous["done"] else "no_admissible_commands")
                   and row["success"] is bool(row["initial"]["won"] or any(t["after"]["won"] for t in trace)),
                   "Episode final outcome/termination differs")
    shared.require(type(row["generated_tokens"]) is int
                   and row["generated_tokens"] == sum(t["policy"]["generated_tokens"] for t in trace),
                   "Episode generated-token total differs")


def validate_model_barrier(request, barrier, audit):
    """All four actual directories plus the same audited three training runs."""
    from .train_recovery_v1 import SETTINGS
    shared.require(barrier.get("complete") is True and barrier.get("arms") == list(ARMS)
                   and set(barrier["adapters"]) == set(ARMS), "Missing all-four-model barrier")
    paths = [item["path"] for item in barrier["adapters"].values()]
    shared.require(len(set(paths)) == 4 and barrier["adapters"][ARMS[0]]["path"] == CANONICAL,
                   "All four arm checkpoint paths must be distinct with fixed canonical role")
    audit_hash = request["file_sha256"][request["joint_token_audit"]]
    shared.require(audit.get("kind") == "recovery_joint_training_token_audit_v1" and audit.get("complete") is True
                   and audit["settings"] == SETTINGS and set(audit["arms"]) == set(ARMS[1:])
                   and audit["score_sft_token_context_mismatches"] == 0
                   and barrier["joint_token_audit_sha256"] == audit_hash, "Complete common token audit not bound to barrier")
    initial_inventory = audit["provenance"]["initial_adapter"]
    shared.require(shared.digest(initial_inventory["files"]) == initial_inventory["files_sha256"],
                   "Initial canonical audit inventory differs")
    for arm, item in barrier["adapters"].items():
        directory = safe_path(item["path"])
        actual = {path.relative_to(ROOT).as_posix() for path in directory.rglob("*") if path.is_file()}
        required = {item["path"] + "/training_state.json", item["path"] + "/adapter_config.json"}
        shared.require(actual == set(item["file_sha256"]) and required <= actual
                       and any(name.endswith(("/adapter_model.safetensors", "/adapter_model.bin")) for name in actual),
                       "Locked adapter inventory missing/changed: " + arm)
        for name, h in item["file_sha256"].items():
            shared.require(request["file_sha256"].get(name) == h, "Model barrier not bound in request")
        state = read(directory / "training_state.json")
        shared.require(state["optimizer_steps"] == 100 and state["protocol"] == PROTOCOL_VERSION,
                       "Wrong final checkpoint step/protocol")
        if arm == ARMS[0]:
            expected = {CANONICAL + "/" + f["name"]: f["sha256"] for f in initial_inventory["files"]}
            shared.require(item["file_sha256"] == expected and state == audit["provenance"]["initial_training_state"],
                           "Canonical barrier does not match common training initialization")
            continue
        expected = audit["arms"][arm]
        shared.require(expected["component_exposures"] == {"bc": 400, "recovery": 400}
                       and state["target_tokens"] == expected["totals"]["target_tokens"]
                       and state["sampled_examples"] == 800 and state["component_exposures"] == expected["component_exposures"]
                       and state["initial_adapter_optimizer_steps"] == 100 and state["cumulative_optimizer_steps"] == 200
                       and state["optimizer_state_resumed"] is False and state["token_audit_sha256"] == audit_hash
                       and state["schedule_sha256"] == audit["schedule_sha256"]
                       and state["initial_adapter_inventory_sha256"] == initial_inventory["files_sha256"],
                       "Trained final checkpoint differs from common audit")
        train_folder = directory.parent
        for name in ("config.json", "summary.json"):
            shared.require((train_folder / name).relative_to(ROOT).as_posix() in request["file_sha256"],
                           "Missing bound training config/summary")
        config, summary = read(train_folder / "config.json"), read(train_folder / "summary.json")
        shared.require(config["arm"] == arm and config["protocol"] == PROTOCOL_VERSION
                       and config["settings"] == SETTINGS and config["token_audit_sha256"] == audit_hash
                       and config["schedule_sha256"] == audit["schedule_sha256"], "Training arm/config binding differs")
        shared.require(summary.get("complete") is True and summary.get("matched_step_budget_complete") is True
                       and summary["optimizer_steps"] == summary["requested_steps"] == 100
                       and summary["stop_reason"] == "max_steps" and not summary["technical_errors"]
                       and summary["target_tokens"] == summary["actual_backward_target_tokens"]
                       == summary["expected_target_tokens"] == state["target_tokens"]
                       and summary["component_exposures"] == expected["component_exposures"]
                       and summary["sampled_examples"] == 800 and summary["token_audit_sha256"] == audit_hash
                       and summary["schedule_sha256"] == audit["schedule_sha256"], "Training completion/audit evidence differs")
    shared.require(request["adapter"] == barrier["adapters"][request["arm"]]["path"], "Wrong arm adapter")


def verify_bytes(request):
    from scripts.collect_recovery_projection_v1 import source_inventory
    from scripts.run_successful_path_learning_v1 import verify_model, PROVENANCE, MODEL_INVENTORY
    shared.require(source_inventory() == request["source_inventory"], "Source inventory changed")
    for name, h in request["file_sha256"].items():
        shared.require(sha(safe_path(name)) == h, "Bound input changed: " + name)
    for field in ("original_file_sha256", "runtime_file_sha256"):
        for name, h in request[field].items():
            shared.require(Path(name).is_absolute() and sha(name) == h, "Original/runtime changed")
    verify_model(read(ROOT / PROVENANCE), full_inventory=read(ROOT / MODEL_INVENTORY))
    validate_model_barrier(request, read(safe_path(request["all_models_locked_receipt"])),
                           read(safe_path(request["joint_token_audit"])))


def validate_request(path):
    r = read(path)
    for key, value in {"kind": "recovery_evaluation_stage_adoption", "schema_version": 1,
                       "adopted": True, "adopted_by": "root_after_independent_GPT_6_Astra_review", "physical_gpu": 2}.items():
        shared.require(type(r.get(key)) is type(value) and r[key] == value, "Evaluation adoption rejected: " + key)
    shared.require(r["arm"] in ARMS and r["manifest"] == MANIFEST, "Wrong arm/manifest")
    required = set(r["source_inventory"]) | {r["candidate"], r["implementation_review"], r["manifest"],
                 r["all_models_locked_receipt"], r["previous_cost_json"], r["previous_summary"],
                 r["joint_token_audit"],
                 "outputs/reference_model_provenance_v1.json", "outputs/reference_model_inventory_v2.json"}
    shared.require(required <= r["file_sha256"].keys() and r["file_sha256"][MANIFEST] == MANIFEST_SHA256,
                   "Incomplete evaluation input bindings")
    verify_bytes(r)
    review = read(safe_path(r["implementation_review"]))
    shared.require(review.get("recovery_evaluator_review_passed") is True
                   and review["evaluator_sha256"] == sha(__file__)
                   and review["candidate_sha256"] == r["file_sha256"][r["candidate"]], "Exact evaluator/candidate review missing")
    candidate = read(safe_path(r["candidate"]))
    shared.require(candidate["new_research_evaluation"]["fixed_order"] == list(ARMS)
                   and candidate["new_research_evaluation"]["episodes"] == 192, "Evaluation protocol changed")
    budgets = candidate["finite_budgets"]
    shared.require(shared.GLOBAL_LIMITS == {"environment_steps_started": budgets["environment_step_upper"]["hard_global_step_calls_started"],
        "resets_started": budgets["max_total_reset_calls"], "generated_tokens": budgets["max_total_generated_tokens"]},
        "Candidate cumulative budgets differ")
    shared.require(str(Path(sys.executable).resolve()) in r["runtime_file_sha256"], "Worker executable not runtime-bound")
    games = read(safe_path(MANIFEST))["games"]
    shared.require(len(games) == len(set(games)) == 48, "Wrong frozen48 identities")
    for game in games:
        shared.require("/json_2.1.1/valid_seen/" in game, "Non-seen task rejected")
        shared.require(game in r["original_file_sha256"] and str(Path(game).with_name("traj_data.json"))
                       in r["original_file_sha256"], "Unbound evaluation task")
    remaining = r["global_deadline_unix"] - time.time()
    shared.require(math.isfinite(remaining) and 30 < remaining <= 171840
                   and r["max_wall_seconds"] == 7200, "Invalid evaluation deadline")
    previous, summary = read(safe_path(r["previous_cost_json"])), read(safe_path(r["previous_summary"]))
    shared.require(previous["kind"] == "recovery_cost_ledger" and previous["active_call"] is None
                   and previous["unknown_generation_after_error"] is False and summary.get("complete") is True
                   and previous["global_limits"] == shared.GLOBAL_LIMITS
                   and summary["costs"]["totals"] == previous["totals"], "Previous cost lineage incomplete")
    if r["arm"] == ARMS[0]:
        shared.require(summary.get("stage") == "diagnostic" and summary.get("gate_evaluable") is True
                       and summary.get("training_gate_passed") is True,
                       "First evaluation must inherit passed diagnostic gate and environment costs")
    else:
        shared.require(summary.get("stage") == "evaluation" and summary.get("arm") == ARMS[ARMS.index(r["arm"]) - 1],
                       "Evaluation cost order differs")
    for key in ("resets", "environment_steps", "policy_calls"):
        shared.require(previous["totals"][key + "_started"] == previous["totals"][key + "_completed"], "Unfinished prior call")
    output = safe_path(r["output"])
    shared.require(r["output"].startswith("outputs/recovery_projection_") and not output.exists(), "Output exists/wrong namespace")
    require_gpu_guard()
    receipt = read(safe_path(r["gpu_receipt"]))
    shared.require(len(receipt["devices"]) == 1 and receipt["devices"][0]["physical_index"] == 2
                   and receipt["devices"][0]["uuid"] == os.environ["CUDA_VISIBLE_DEVICES"], "Wrong physicalGPU receipt")
    return r, games, output, previous["totals"]


def run(path):
    request_hash = sha(path)
    r, games, output, initial = validate_request(path)
    output.mkdir(parents=True, exist_ok=False)
    atomic(output / "config.json", {"request": r, "request_sha256": request_hash,
           "protocol": "alfworld-causal-v1", "model": MODEL, "adapter": r["adapter"],
           "selection": "generate", "do_sample": False, "max_context_tokens": 2048,
           "max_action_tokens": 64, "max_steps": 100, "seed": 20261012, "expert_type": None,
           "max_generated_tokens": 350000, "max_scored_tokens": 10000000, "max_wall_seconds": 7200})
    deadline = time.monotonic() + min(7200, r["global_deadline_unix"] - time.time() - 30)
    ledger = shared.BudgetLedger(stage_limits=STAGE_LIMITS, deadline=deadline, initial_totals=initial,
                               persist=lambda value: atomic(output / "cost_progress.json", value))
    ledger.flush()
    summary = {"schema_version": 1, "stage": "evaluation", "arm": r["arm"], "complete": False,
               "planned_episodes": 48, "attempted_episodes": 0, "completed_episodes": 0,
               "technical_errors": 0, "request_sha256": request_hash,
               "partial_performance_summary": None}
    try:
        ledger.check()
        policy = HFCausalPolicy(MODEL, str(safe_path(r["adapter"])), max_context_tokens=2048, max_action_tokens=64)
        ledger.check()
        from .alfworld_env import AlfworldReplayEnv
        with (output / "episodes.jsonl").open("x", encoding="utf-8", buffering=1) as episodes, \
             (output / "events.jsonl").open("x", encoding="utf-8", buffering=1) as events:
            def emit(row):
                events.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            for game in games:
                ledger.check()
                summary["attempted_episodes"] += 1
                env = None
                try:
                    env = AlfworldReplayEnv(game, seed=20261012, max_steps=100, expert_type=None)
                    row = evaluate_task(env, policy, game, ledger, emit)
                    validate_episode(row, game)
                except BaseException as error:
                    episodes.write(json.dumps({"gamefile": game, "seed": 20261012, "complete": False,
                        "success": None, "technical_error": f"{type(error).__name__}: {error}"}) + "\n")
                    raise
                finally:
                    if env is not None:
                        env.close()
                episodes.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                summary["completed_episodes"] += 1
                print(json.dumps({"arm": r["arm"], "completed_episodes": summary["completed_episodes"]}), flush=True)
        shared.require(summary["completed_episodes"] == 48, "Incomplete48 evaluation")
        ledger.check()
        verify_bytes(r)
        ledger.check()
        shared.require(sha(path) == request_hash, "Evaluation request changed")
        summary.update(complete=True, episodes_sha256=sha(output / "episodes.jsonl"))
    except BaseException as error:
        summary.update(complete=False, technical_errors=1, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["costs"] = ledger.snapshot()
        ledger.flush()
        atomic(output / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adoption-json", required=True)
    a = p.parse_args()
    shared.require(sys.platform.startswith("linux") and ROOT.resolve() == Path("/home/jsy/GNRL_20261012").resolve(),
                   "Only chemistry project execution supported")
    os.chdir(ROOT)
    run(a.adoption_json)


if __name__ == "__main__":
    main()
