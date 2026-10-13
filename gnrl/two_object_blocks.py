"""Phase-1 CPU pilot: native-planner seeds and reversed two-object deliveries.

The original phase-0 code/config remains unchanged. Successful native planner
traces are the NEW canonical baseline, not the noisy handcoded baseline. Each
alternative moves the same two named objects into the same named receptacle,
in reverse order, using real simulator observations at every step. Retained
alternatives have the same exact command multiset and strict terminal state.

python -m gnrl.two_object_blocks --input data/expert_train120_raw.jsonl --output outputs/two_blocks_v1
Then existing gnrl.macro_project --pools .../candidate_pools.jsonl can score the
same pools and emit paired uniform/learner-selected trajectories.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from .alfworld_env import AlfworldReplayEnv, ReplayMismatch, information_only_action, transition_record
from .macro_project import (BudgetExceeded, SearchBudget, candidate_id, dump_line,
                            physical_key, read_rows, replay_checked, validate_pool)


class IneligibleBlock(ValueError):
    pass


def predicates(state):
    result = {}
    for name, args in state["facts"]:
        result.setdefault(str(name).lower(), set()).add(tuple(str(v[0]) for v in args))
    return result


def entity_names(env):
    """Backend names are teacher-side grounding metadata, never prompt fields."""
    native = env.env
    infos = getattr(native, "state", {}).get("_entity_infos")
    if infos is None:
        infos = getattr(native, "_entity_infos", None)
    if not infos:
        raise RuntimeError("Backend did not expose entity ID/name map")
    names = {str(identity): str(info.name) for identity, info in infos.items() if info.name}
    return names


def task_entities(initial, metadata):
    if metadata.get("task_type") != "pick_two_obj_and_place":
        raise IneligibleBlock("Not a two-object task")
    params = metadata["pddl_params"]
    if params.get("object_sliced") or params.get("mrecep_target"):
        raise IneligibleBlock("Sliced/movable-receptacle task unsupported")
    facts = predicates(initial.canonical_state)
    object_type = (params["object_target"] + "Type").lower()
    receptacle_type = (params["parent_target"] + "Type").lower()
    objects = {obj for obj, kind in facts.get("objecttype", ()) if kind.lower() == object_type}
    targets = {rec for rec, kind in facts.get("receptacletype", ()) if kind.lower() == receptacle_type}
    if len(objects) < 2 or not targets:
        raise IneligibleBlock("Initial symbolic metadata lacks two goal objects or a target")
    return objects, targets


def inspect_deliveries(transitions, objects, targets):
    """Read grounded pickup/place effects, not ambiguous type-only strings."""
    deliveries = []
    for transition in transitions:
        before = predicates(transition["state_before"])
        after = predicates(transition["state_after"])
        released = {o for _, o in before.get("holds", set()) - after.get("holds", set())}
        additions = after.get("inreceptacle", set()) - before.get("inreceptacle", set())
        for obj, rec in sorted(additions):
            if obj in released and obj in objects and rec in targets:
                deliveries.append((obj, rec))
    if len(deliveries) != 2 or deliveries[0][0] == deliveries[1][0]:
        raise IneligibleBlock("Native plan does not contain exactly two distinct goal deliveries")
    if deliveries[0][1] != deliveries[1][1]:
        raise IneligibleBlock("Objects were not delivered to the same target instance")
    return [deliveries[0][0], deliveries[1][0]], deliveries[0][1]


def validate_useful_plan(trace):
    seen = {physical_key(trace[0]["state_before"])} if trace else set()
    for transition in trace:
        if information_only_action(transition["action"]):
            raise IneligibleBlock("Information-only action in canonical plan")
        key = physical_key(transition["state_after"])
        if key in seen:
            raise IneligibleBlock("Physical no-op or loop in canonical plan")
        seen.add(key)


def collect_planner(env, budget, max_plan_steps):
    current = budget.reset(env)
    initial = current
    trace = []
    for index in range(max_plan_steps):
        if current.done:
            break
        action = current.expert_action
        if not action or action not in current.admissible_commands:
            raise IneligibleBlock("Native planner returned no executable action")
        after = budget.step(env, action)
        trace.append(transition_record(current, action, after, index))
        current = after
    if not trace or not current.won or not current.done:
        raise IneligibleBlock("Native planner did not reach success within plan-step budget")
    validate_useful_plan(trace)
    return initial, trace


def compile_delivery_order(env, budget, order, target, max_plan_steps, expected_initial):
    """Execute whole blocks with live preconditions, preserving one-slot inventory.

    A block is navigate-to-source, open-if-required, take, navigate-to-target,
    open-if-required, place. No path or observation is fabricated.
    """
    current = budget.reset(env)
    if current.canonical_state != expected_initial.canonical_state or current.observation != expected_initial.observation:
        raise ReplayMismatch("Recipe reset differs from planner initial state")
    names = entity_names(env)
    trace = []

    def act(command):
        nonlocal current
        if len(trace) >= max_plan_steps:
            raise IneligibleBlock("Recipe exceeded plan-step budget")
        if current.done or command not in current.admissible_commands:
            raise IneligibleBlock("Recipe action inadmissible or after termination: " + command)
        after = budget.step(env, command)
        trace.append(transition_record(current, command, after, len(trace)))
        current = after

    def navigate(receptacle):
        facts = predicates(current.canonical_state)
        locations = {loc for rec, loc in facts.get("receptacleatlocation", ()) if rec == receptacle}
        agent_locations = {loc for _, loc in facts.get("atlocation", ())}
        if len(locations) != 1 or len(agent_locations) != 1:
            raise IneligibleBlock("Missing or ambiguous navigation locations")
        if locations != agent_locations:
            act("go to " + names[receptacle])

    def open_if_needed(receptacle):
        facts = predicates(current.canonical_state)
        if (receptacle,) in facts.get("openable", set()) and (receptacle,) not in facts.get("opened", set()):
            act("open " + names[receptacle])

    for obj in order:
        facts = predicates(current.canonical_state)
        if facts.get("holds") or facts.get("holdsany"):
            raise IneligibleBlock("Delivery must start with empty inventory")
        sources = {rec for item, rec in facts.get("inreceptacle", ()) if item == obj}
        if len(sources) != 1:
            raise IneligibleBlock("Missing or ambiguous object source")
        source = next(iter(sources))
        if source == target:
            raise IneligibleBlock("Object already in goal receptacle; not an independent delivery")
        navigate(source)
        open_if_needed(source)
        act("take " + names[obj] + " from " + names[source])
        navigate(target)
        open_if_needed(target)
        act("move " + names[obj] + " to " + names[target])
    if not current.won or not current.done:
        raise IneligibleBlock("Two compiled deliveries did not satisfy actual environment goal")
    validate_useful_plan(trace)
    return trace


def verify_trace(env, trace, budget, initial_observation):
    fresh_initial, fresh = replay_checked(env, [t["action"] for t in trace], budget)
    if fresh_initial != initial_observation:
        raise ReplayMismatch("Fresh initial observation differs")
    for old, new in zip(trace, fresh):
        for key in ("observation", "next_observation", "state_before", "state_after"):
            if old[key] != new[key]:
                raise ReplayMismatch("Independent full replay differed: " + key)
    if not fresh[-1]["won"] or not fresh[-1]["done"]:
        raise ReplayMismatch("Independent full replay is unsuccessful")
    return fresh


def build_pool(row, metadata, env, budget, max_plan_steps=24):
    initial, canonical = collect_planner(env, budget, max_plan_steps)
    if row.get("initial_observation") != initial.observation:
        raise ReplayMismatch("Planner initial observation differs from frozen collector task")
    if row["transitions"][0]["state_before"] != initial.canonical_state:
        raise ReplayMismatch("Planner initial state differs from frozen collector task")
    canonical = verify_trace(env, canonical, budget, initial.observation)
    original_actions = [t["action"] for t in canonical]
    baseline_id = candidate_id(original_actions)
    original = {"candidate_id": baseline_id, "origin": "original",
                "transitions": canonical,
                "evidence": {"full_replay_verified": True, "terminal_success": True,
                             "strict_endpoint_match": True, "canonical_source": "native_planner",
                             "raw_handcoded_trace_is_not_this_baseline": True}}
    pool = {"kind": "macro_candidate_pool", "schema_version": 1,
            "phase": "phase1_two_object_block_reversal", "gamefile": row["gamefile"],
            "seed": row.get("seed", 0), "split": "train", "task": initial.observation,
            "initial_observation": initial.observation, "success": True,
            "original_candidate_id": baseline_id, "candidates": [original],
            "equivalence": "entire_adapter_canonical_state_plus_exact_action_multiset",
            "proposal_uses_privileged_expert_actions": True,
            "search_counts": {"canonical_plan_verified": 1}, "search_stop": "complete"}
    objects, targets = task_entities(initial, metadata)
    try:
        order, target = inspect_deliveries(canonical, objects, targets)
        pool["teacher_search"] = {"native_object_order_ids": order, "target_receptacle_id": target,
                                   "used_only_for_offline_proposal": True}
        reverse = compile_delivery_order(env, budget, list(reversed(order)), target, max_plan_steps, initial)
        reverse_actions = [t["action"] for t in reverse]
        if Counter(reverse_actions) != Counter(original_actions):
            raise IneligibleBlock("Reverse plan changes action multiset or action count")
        if reverse[-1]["state_after"] != canonical[-1]["state_after"]:
            raise IneligibleBlock("Reverse plan has different strict terminal state")
        if reverse_actions == original_actions:
            raise IneligibleBlock("Identical reversed plan")
        reverse = verify_trace(env, reverse, budget, initial.observation)
        pool["candidates"].append({"candidate_id": candidate_id(reverse_actions),
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


def canonical_row(pool, pool_hash):
    base = pool["candidates"][0]
    return {"kind": "trajectory", "gamefile": pool["gamefile"], "seed": pool["seed"],
            "split": "train", "task": pool["task"], "initial_observation": pool["initial_observation"],
            "transitions": base["transitions"], "success": True,
            "replay_verified": True, "reset_replay_verified": True,
            "selection": {"arm": "canonical_native_planner", "candidate_id": base["candidate_id"],
                          "pool_sha256": pool_hash, "pool_size": len(pool["candidates"]),
                          "method": "original native-planner delivery order; no learner-aware selection"},
            "replay_evidence": base["evidence"]}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Frozen raw train120 trajectory JSONL, including unsuccessful rows")
    p.add_argument("--output", required=True)
    p.add_argument("--max-games", type=int, default=20)
    p.add_argument("--max-plan-steps", type=int, default=24)
    p.add_argument("--max-env-steps", type=int, default=20000)
    p.add_argument("--max-wall-seconds", type=float, default=1800)
    p.add_argument("--minimum-covered-tasks", type=int, default=10)
    return p


def main():
    args = parser().parse_args()
    if min(args.max_games, args.max_plan_steps, args.max_env_steps, args.max_wall_seconds,
           args.minimum_covered_tasks) <= 0:
        raise ValueError("All budgets must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(args.input)
    frozen_bytes = source.read_bytes()
    rows = []
    identities = set()
    for row in (json.loads(line) for line in frozen_bytes.decode("utf-8-sig").splitlines() if line.strip()):
        if row.get("kind") != "trajectory" or "pick_two_obj_and_place-" not in Path(row["gamefile"]).parent.parent.name:
            continue
        if row.get("split", "train") != "train":
            raise ValueError("Only frozen training tasks are permitted")
        identity = (row["gamefile"], row.get("seed", 0))
        if identity in identities:
            raise ValueError("Duplicate frozen task identity")
        identities.add(identity)
        rows.append(row)
    rows = rows[:args.max_games]
    if not rows:
        raise ValueError("No two-object training tasks in input")
    (out / "config.json").write_text(json.dumps(vars(args) | {
        "source_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "task_identities": [[r["gamefile"], r.get("seed", 0)] for r in rows],
        "method": "native planner canonical baseline plus reversed complete delivery blocks",
        "raw_expert_cleanup_is_separate_from_projection": True}, indent=2), encoding="utf-8")
    budget = SearchBudget(args.max_env_steps, args.max_wall_seconds)
    counts = Counter()
    stop = "complete"
    with (out / "candidate_pools.jsonl").open("w", encoding="utf-8") as pools, \
         (out / "canonical_baseline.jsonl").open("w", encoding="utf-8") as canonical, \
         (out / "events.jsonl").open("w", encoding="utf-8") as events:
        for row in rows:
            start_steps = budget.steps
            env = None
            event = {"gamefile": row["gamefile"], "source_handcoded_success": row.get("success", False)}
            try:
                budget.check()
                metadata = json.loads(Path(row["gamefile"]).with_name("traj_data.json").read_text())
                max_steps = row["transitions"][0]["state_before"]["max_steps"]
                env = AlfworldReplayEnv(row["gamefile"], max_steps=max_steps, expert_type="planner", seed=row.get("seed", 0))
                pool = build_pool(row, metadata, env, budget, args.max_plan_steps)
                pool["acquisition_cost"] = {"environment_steps_including_replay": budget.steps - start_steps}
                pool["gamefile_sha256"] = hashlib.sha256(Path(row["gamefile"]).read_bytes()).hexdigest()
                digest = hashlib.sha256(json.dumps(pool, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                dump_line(pools, pool)
                dump_line(canonical, canonical_row(pool, digest))
                counts["canonical_tasks"] += 1
                counts["tasks_with_nontrivial_candidates"] += int(len(pool["candidates"]) > 1)
                counts["repaired_handcoded_failures"] += int(not row.get("success"))
                counts["canonical_actions"] += len(pool["candidates"][0]["transitions"])
                counts["raw_actions_on_same_tasks"] += len(row["transitions"])
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
    full_cohort = counts["canonical_tasks"] == len(rows) and not counts["technical_errors"]
    gate = full_cohort and counts["tasks_with_nontrivial_candidates"] >= args.minimum_covered_tasks
    summary = {**dict(counts), "planned_tasks": len(rows), "stop_reason": stop,
               "full_canonical_cohort_complete": full_cohort, "coverage_gate_passed": gate,
               "minimum_covered_tasks": args.minimum_covered_tasks,
               "environment_steps_including_all_replay": budget.steps, "resets": budget.resets,
               "wall_seconds": time.monotonic() - budget.started,
               "interpretation": "Canonical data quality and reference-aware projection are separate comparisons. No learning outcome measured."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
