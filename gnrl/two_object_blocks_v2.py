"""Same phase-1 method with cached planner seeds and three planner-free passes.

Requires complete, successful, replay-verified planner acquisition for the
entire frozen manifest. It does not regenerate or change canonical actions.
Linux CLI re-execs under GNU timeout, including a 10-second forced-kill grace.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .alfworld_env import AlfworldReplayEnv, ReplayMismatch, StepResult
from .macro_project import BudgetExceeded, SearchBudget, candidate_id, dump_line, validate_pool
from .two_object_blocks import (IneligibleBlock, canonical_row, compile_delivery_order,
                                inspect_deliveries, task_entities, validate_useful_plan, verify_trace)


def initial_from_source(row):
    first = row["transitions"][0]
    state = first["state_before"]
    return StepResult(row["initial_observation"], first["admissible_commands"], None,
                      state["reward"], state["done"], state["won"], state)


def load_frozen_sources(path, games_json):
    path = Path(path)
    summary_path = path.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    frozen = json.loads(Path(games_json).read_text())
    expected = frozen["games"] if isinstance(frozen, dict) else frozen
    if len(expected) != 120 or len(set(expected)) != len(expected):
        raise ValueError("This phase requires the complete 120-task frozen manifest without duplicates")
    if summary.get("expert_type") != "planner" or summary.get("split") != "train":
        raise ValueError("Input must be official training native-planner acquisition")
    if summary.get("games") != expected:
        raise ValueError("Planner acquisition manifest/order differs from frozen training manifest")
    counts = summary["counts"]
    if counts.get("task_errors", 0) or counts.get("expert_replays_verified") != len(expected) or counts.get("expert_successes") != len(expected):
        raise ValueError("Wait for complete, successful, error-free native planner acquisition")
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("kind") == "trajectory"]
    if [r["gamefile"] for r in rows] != expected:
        raise ValueError("Source trajectory identities/order differ from the frozen manifest")
    for row in rows:
        if not row.get("success") or not row.get("replay_verified") or not row.get("reset_replay_verified"):
            raise ValueError("Unverified or failed planner source")
        if row.get("split") != "train" or row.get("seed") != summary["seed"]:
            raise ValueError("Source split/seed differs from acquisition summary")
    selected = [r for r in rows if Path(r["gamefile"]).parent.parent.name.startswith("pick_two_obj_and_place-")]
    if len(selected) != 20:
        raise ValueError("This phase requires exactly the frozen 20 two-object training tasks")
    return selected, summary, hashlib.sha256(raw).hexdigest()


def build_pool_from_seed(row, metadata, env, budget, max_plan_steps=24):
    if not row.get("success") or not row.get("replay_verified"):
        raise ValueError("Canonical seed must be verified successful")
    canonical = row["transitions"]
    if not canonical or len(canonical) > max_plan_steps:
        raise IneligibleBlock("Canonical native plan exceeds registered plan-step limit")
    initial = initial_from_source(row)
    validate_useful_plan(canonical)
    # Pass 1: full original actions in expert_type=None environment. Verify
    # exact source states/observations, omitting only unused expert annotations.
    canonical = verify_trace(env, canonical, budget, initial.observation)
    original_actions = [t["action"] for t in canonical]
    identity = candidate_id(original_actions)
    pool = {"kind": "macro_candidate_pool", "schema_version": 1,
            "phase": "phase1_two_object_block_reversal", "execution_version": 2,
            "gamefile": row["gamefile"], "seed": row["seed"], "split": "train",
            "task": initial.observation, "initial_observation": initial.observation,
            "success": True, "original_candidate_id": identity,
            "candidates": [{"candidate_id": identity, "origin": "original", "transitions": canonical,
                            "evidence": {"full_replay_verified": True, "terminal_success": True,
                                         "strict_endpoint_match": True, "canonical_source": "cached_native_planner",
                                         "raw_handcoded_trace_is_not_this_baseline": True}}],
            "equivalence": "entire_adapter_canonical_state_plus_exact_action_multiset",
            "proposal_uses_privileged_expert_actions": True,
            "search_counts": {"canonical_plan_verified": 1}, "search_stop": "complete",
            "performance_change_only": "Reuse verified native-planner actions; all execution uses expert_type=None."}
    objects, targets = task_entities(initial, metadata)
    try:
        order, target = inspect_deliveries(canonical, objects, targets)
        pool["teacher_search"] = {"native_object_order_ids": order, "target_receptacle_id": target,
                                   "used_only_for_offline_proposal": True}
        # Pass 2: compile and execute the reversed complete blocks.
        reverse = compile_delivery_order(env, budget, list(reversed(order)), target, max_plan_steps, initial)
        actions = [t["action"] for t in reverse]
        if Counter(actions) != Counter(original_actions):
            raise IneligibleBlock("Reverse plan changes action multiset or action count")
        if reverse[-1]["state_after"] != canonical[-1]["state_after"]:
            raise IneligibleBlock("Reverse plan has different strict terminal state")
        if actions == original_actions:
            raise IneligibleBlock("Identical reversed plan")
        # Pass 3: independent fresh replay of the complete reverse plan.
        reverse = verify_trace(env, reverse, budget, initial.observation)
        pool["candidates"].append({"candidate_id": candidate_id(actions),
            "origin": "reverse_complete_delivery_blocks", "transitions": reverse,
            "evidence": {"full_replay_verified": True, "terminal_success": True,
                         "strict_endpoint_match": True, "same_exact_action_multiset": True,
                         "equal_action_count": True, "additive_operator_cost_preserved_by_multiset": True,
                         "changed_delivery_order": True, "same_two_object_ids": True,
                         "same_target_receptacle_id": True, "physical_noops": 0, "physical_loops": 0}})
        pool["search_counts"]["verified_nontrivial_candidates"] = 1
    except IneligibleBlock as error:
        pool["reverse_rejected_reason"] = str(error)
        pool["search_counts"]["reverse_rejected"] = 1
    validate_pool(pool)
    return pool


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--games-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-plan-steps", type=int, default=24)
    p.add_argument("--max-env-steps", type=int, default=20000)
    p.add_argument("--max-wall-seconds", type=int, default=1800)
    p.add_argument("--minimum-covered-tasks", type=int, default=10)
    p.add_argument("--under-gnu-timeout", action="store_true", help=argparse.SUPPRESS)
    return p


def main():
    args = parser().parse_args()
    if min(args.max_plan_steps, args.max_env_steps, args.max_wall_seconds, args.minimum_covered_tasks) <= 0:
        raise ValueError("All budgets must be positive")
    if not args.under_gnu_timeout:
        executable = shutil.which("timeout")
        if not sys.platform.startswith("linux") or not executable:
            raise RuntimeError("Actual simulator CLI requires Linux GNU timeout for an external hard deadline")
        # GNU timeout controls only this newly started command's process group.
        os.execvp(executable, [executable, "--signal=TERM", "--kill-after=10s",
                              str(args.max_wall_seconds) + "s", sys.executable,
                              "-m", "gnrl.two_object_blocks_v2", *sys.argv[1:], "--under-gnu-timeout"])
    rows, source_summary, source_hash = load_frozen_sources(args.input, args.games_json)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(args) | {
        "source_sha256": source_hash,
        "source_summary_sha256": hashlib.sha256(Path(args.input).with_suffix(".summary.json").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(Path(args.games_json).read_bytes()).hexdigest(),
        "task_identities": [[r["gamefile"], r["seed"]] for r in rows],
        "canonical_actions_unchanged": True, "planner_wrapper_enabled": False,
        "outer_deadline": "GNU timeout TERM at budget; KILL after 10-second grace"}, indent=2), encoding="utf-8")
    budget = SearchBudget(args.max_env_steps, args.max_wall_seconds)
    counts = Counter()
    stop = "complete"
    with (out / "candidate_pools.jsonl").open("w", encoding="utf-8") as pools, \
         (out / "canonical_baseline.jsonl").open("w", encoding="utf-8") as canonical, \
         (out / "events.jsonl").open("w", encoding="utf-8") as events:
        for row in rows:
            env = None
            start_steps = budget.steps
            event = {"gamefile": row["gamefile"]}
            try:
                budget.check()
                actual_hash = hashlib.sha256(Path(row["gamefile"]).read_bytes()).hexdigest()
                if actual_hash != source_summary["game_sha256"][row["gamefile"]]:
                    raise ReplayMismatch("Game file changed since native-planner acquisition")
                metadata = json.loads(Path(row["gamefile"]).with_name("traj_data.json").read_text())
                env = AlfworldReplayEnv(row["gamefile"], max_steps=row["transitions"][0]["state_before"]["max_steps"],
                                       expert_type=None, seed=row["seed"])
                pool = build_pool_from_seed(row, metadata, env, budget, args.max_plan_steps)
                pool["acquisition_cost"] = {"environment_steps_including_replay": budget.steps - start_steps}
                pool["gamefile_sha256"] = actual_hash
                digest = hashlib.sha256(json.dumps(pool, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                dump_line(pools, pool)
                dump_line(canonical, canonical_row(pool, digest))
                counts["canonical_tasks"] += 1
                counts["tasks_with_nontrivial_candidates"] += int(len(pool["candidates"]) > 1)
                event.update(pool_size=len(pool["candidates"]), canonical_actions=len(pool["candidates"][0]["transitions"]),
                             reverse_rejected_reason=pool.get("reverse_rejected_reason"))
            except BudgetExceeded as error:
                stop = str(error)
                event.update(omitted=True, error=stop)
            except IneligibleBlock as error:
                counts["planner_or_task_ineligible"] += 1
                event.update(omitted=True, error=str(error))
            except Exception as error:
                counts["technical_errors"] += 1
                event.update(omitted=True, error=f"{type(error).__name__}: {error}")
                stop = "technical_error"
            finally:
                if env is not None:
                    env.close()
            counts["tasks_attempted"] += 1
            event["environment_steps_including_replay"] = budget.steps - start_steps
            dump_line(events, event)
            print(json.dumps(event), flush=True)
            if stop != "complete":
                break
    full = counts["canonical_tasks"] == len(rows) and not counts["technical_errors"] and stop == "complete"
    summary = {**dict(counts), "execution_version": 2, "planned_tasks": len(rows), "stop_reason": stop,
               "full_canonical_cohort_complete": full,
               "coverage_gate_passed": full and counts["tasks_with_nontrivial_candidates"] >= args.minimum_covered_tasks,
               "minimum_covered_tasks": args.minimum_covered_tasks,
               "environment_steps_including_all_replay": budget.steps, "resets": budget.resets,
               "wall_seconds": time.monotonic() - budget.started,
               "interpretation": "Cached source is unchanged. Exact-state block reversal is the same method as v1; no learning outcome measured."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
