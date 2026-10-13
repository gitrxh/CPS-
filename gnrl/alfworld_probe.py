"""Finite real-simulator expert collection and horizon-one equivalence audit.

python -m gnrl.alfworld_probe --data-root /data/json_2.1.1 --split train \
    --limit 20 --max-steps 100 --output artifacts/alfworld_probe.jsonl

Horizon one compares exact canonical successors including time budget and
reward. Identical commands and actions with unchanged world facts are excluded.
This diagnostic does not claim BRP gains when no nontrivial alternatives exist.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
from collections import Counter
from pathlib import Path

from .alfworld_env import (AlfworldReplayEnv, ReplayMismatch, StepResult, list_games,
                          replay_actions, state_digest, transition_record, information_only_action)


def is_world_noop(before: dict, after: dict) -> bool:
    """Ignore bookkeeping only when detecting noops, never for equivalence."""
    return all(before[key] == after[key] for key in ("facts", "score", "won", "lost"))


def reclassify_h1_probe(probe: dict) -> dict:
    """Apply the information-action filter to existing audit evidence, offline.

    This never changes strict successor matches, executes no simulator actions,
    and preserves the old classification for an explicit correction audit.
    Previously saved branch trajectories must also be filtered by their actions.
    """
    result = copy.deepcopy(probe)
    result["expert_information_only"] = information_only_action(result["expert_action"])
    result["information_action_filter_version"] = 1
    for item in result["candidates"]:
        item.setdefault("eligible_nontrivial_before_information_filter", item.get("eligible_nontrivial", False))
        item["information_only"] = information_only_action(item["action"])
        item["eligible_nontrivial"] = bool(
            item.get("exact_successor_match") and not item.get("world_noop")
            and not result["expert_world_noop"] and not result["expert_information_only"]
            and not item["information_only"] and item["action"] != result["expert_action"]
            and "error" not in item)
    result["nontrivial_alternatives"] = sum(bool(x["eligible_nontrivial"]) for x in result["candidates"])
    return result


def audit_recorded_h1(path: str | Path) -> dict:
    """Read-only correction summary for a previously collected JSONL file."""
    counts = Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "horizon_one_probe":
            counts["legacy_reported_nontrivial_h1_matches"] += row.get("nontrivial_alternatives", 0)
            revised = reclassify_h1_probe(row)
            counts["states_probed"] += 1
            counts["information_only_expert_states"] += int(revised["expert_information_only"])
            counts["nontrivial_h1_matches"] += revised["nontrivial_alternatives"]
            counts["exhaustive_states_probed"] += int(revised["exhaustive"])
            counts["exhaustive_states_without_nontrivial_alternatives"] += int(
                revised["exhaustive"] and not revised["nontrivial_alternatives"])
            for candidate in revised["candidates"]:
                counts["candidate_commands_recorded"] += 1
                counts["information_only_candidate_commands_excluded"] += int(candidate["information_only"])
                counts["noop_commands_excluded"] += int(candidate.get("world_noop", False))
                counts["branch_errors"] += int("error" in candidate)
                counts["exact_successor_matches_before_exclusions"] += int(candidate.get("exact_successor_match", False))
        elif row.get("kind") == "branch_trajectory":
            excluded = information_only_action(row["replaced_action"]) or information_only_action(row["replacement_action"])
            counts["legacy_branch_trajectories_excluded_information_only"] += int(excluded)
            counts["branch_trajectories_after_information_filter"] += int(not excluded)
    return {"source": str(Path(path).resolve()),
            "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "information_action_filter_version": 1, "simulator_rerun": False,
            "strict_successor_equality_changed": False, "counts": dict(counts),
            "count_overlap": "Information-only and world-noop counts can overlap; they are independent exclusion reasons."}


def audit_horizon_one(env, trajectory: list[dict], max_branch_states: int = 8,
                      max_candidates: int = 64) -> tuple[list[dict], list[dict], Counter]:
    probes, branches, counts = [], [], Counter()
    actions = [t["action"] for t in trajectory]
    # Even coverage across an episode, fixed before seeing equivalence results.
    n = min(len(trajectory), max_branch_states)
    indices = sorted({int(i * len(trajectory) / n) for i in range(n)}) if n else []
    for index in indices:
        t = trajectory[index]
        commands = [c for c in sorted(set(t["admissible_commands"])) if c != t["action"]]
        counts["identical_commands_excluded"] += int(t["action"] in t["admissible_commands"])
        candidates = commands[:max_candidates]
        counts["candidate_commands_skipped_by_cap"] += max(0, len(commands) - len(candidates))
        probe = {
            "kind": "horizon_one_probe", "step": index, "expert_action": t["action"],
            "expert_world_noop": is_world_noop(t["state_before"], t["state_after"]),
            "expert_information_only": information_only_action(t["action"]),
            "information_action_filter_version": 1,
            "exhaustive": len(candidates) == len(commands),
            "available_nonidentical_commands": len(commands), "candidates": [],
        }
        counts["information_only_expert_states"] += int(probe["expert_information_only"])
        for command in candidates:
            item = {"action": command, "information_only": information_only_action(command)}
            counts["information_only_candidate_commands_excluded"] += int(item["information_only"])
            try:
                before = env.restore(actions[:index], expected=t["state_before"])
                if before.observation != t["observation"]:
                    raise ReplayMismatch("Restored branch observation differs from recorded prefix")
                counts["prefix_restores_verified"] += 1
                after = env.step(command)
                noop = is_world_noop(before.canonical_state, after.canonical_state)
                matches = after.canonical_state == t["state_after"]
                item.update(world_noop=noop, exact_successor_match=matches,
                            state_after_hash=state_digest(after.canonical_state),
                            fresh_observation=after.observation)
                counts["candidate_commands_executed"] += 1
                counts["noop_commands_excluded"] += int(noop)
                nontrivial = (matches and not noop and not probe["expert_world_noop"]
                              and not item["information_only"] and not probe["expert_information_only"])
                item["eligible_nontrivial"] = nontrivial
                if nontrivial:
                    counts["nontrivial_h1_matches"] += 1
                    changed = list(actions)
                    changed[index] = command
                    # Record only actually executed observations, including suffix.
                    fresh = replay_actions(env, changed)
                    suffix_ok = len(fresh) == len(trajectory) and all(
                        fresh[j]["state_after"] == trajectory[j]["state_after"]
                        for j in range(index, len(fresh)))
                    item["suffix_state_replay_verified"] = suffix_ok
                    if not suffix_ok:
                        raise ReplayMismatch("Equivalent successor did not preserve suffix states")
                    branches.append({"kind": "branch_trajectory", "branch_step": index,
                                     "replaced_action": t["action"], "replacement_action": command,
                                     "initial_observation": fresh[0]["observation"],
                                     "transitions": fresh, "success": bool(fresh[-1]["won"]),
                                     "suffix_state_replay_verified": True})
                    counts["branch_trajectories_verified"] += 1
            except Exception as error:
                item["error"] = f"{type(error).__name__}: {error}"
                counts["branch_errors"] += 1
                probe["exhaustive"] = False
            probe["candidates"].append(item)
        probe["nontrivial_alternatives"] = sum(
            bool(x.get("eligible_nontrivial")) and "error" not in x for x in probe["candidates"])
        counts["states_probed"] += 1
        counts["exhaustive_states_probed"] += int(probe["exhaustive"])
        counts["exhaustive_states_without_nontrivial_alternatives"] += int(
            probe["exhaustive"] and not probe["nontrivial_alternatives"])
        probes.append(probe)
    return probes, branches, counts


def collect_expert(env) -> dict:
    result = env.reset()
    initial_observation = result.observation
    transitions = []
    while not result.done:
        action = result.expert_action
        if action is None:
            raise RuntimeError("Expert supplied no next action before termination")
        before = result
        result = env.step(action)
        transitions.append(transition_record(before, action, result, len(transitions)))
    return {"kind": "trajectory", "initial_observation": initial_observation,
            "task": initial_observation, "transitions": transitions,
            "success": result.won, "truncated": result.canonical_state["truncated"]}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--games-json", type=Path, help="Frozen JSON list of game paths, or object with games list; paths must belong to the requested official split")
    p.add_argument("--split", choices=["train", "valid_seen", "valid_unseen"], default="train")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expert-type", choices=["handcoded", "planner"], default="handcoded")
    p.add_argument("--skip-branches", action="store_true", help="Collect expert trajectories and verify replay only")
    p.add_argument("--max-branch-states", type=int, default=8, help="At most this many evenly spaced states per task")
    p.add_argument("--max-candidates", type=int, default=64, help="At most this many nonidentical admissible commands per state")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if min(args.limit, args.max_steps, args.max_candidates) < 1 or min(args.offset, args.max_branch_states) < 0:
        raise SystemExit("Limits must be positive; offset and max-branch-states must be nonnegative")
    games = list_games(args.data_root, args.split)
    if args.games_json:
        requested = json.loads(args.games_json.read_text(encoding="utf-8-sig"))
        requested = requested["games"] if isinstance(requested, dict) else requested
        if not isinstance(requested, list) or not all(isinstance(p, str) for p in requested):
            raise SystemExit("games-json must contain a JSON list of game path strings")
        requested = [Path(p).expanduser().resolve() for p in requested]
        if len(set(requested)) != len(requested):
            raise SystemExit("games-json contains duplicate game paths")
        invalid = set(requested) - set(games)
        if invalid:
            raise SystemExit(f"games-json contains {len(invalid)} paths outside the eligible official {args.split} split")
        games = requested
    else:
        random.Random(args.seed).shuffle(games)
    games = games[args.offset:args.offset + args.limit]
    if not games:
        raise SystemExit("No supported solvable games selected in the official split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("alfworld", "textworld", "fast-downward"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    manifest = {"schema_version": 1, "split": args.split, "seed": args.seed,
                "games": [str(p) for p in games], "versions": versions,
                "game_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in games},
                "selection_manifest": str(args.games_json.resolve()) if args.games_json else None,
                "max_steps": args.max_steps, "expert_type": args.expert_type,
                "restore_mode": AlfworldReplayEnv.restore_mode,
                "max_branch_states": 0 if args.skip_branches else args.max_branch_states,
                "max_candidates": args.max_candidates,
                "equivalence": "raw_facts+elapsed_steps+moves+score+reward+terminal_flags+admissibles",
                "information_action_filter_version": 1,
                "information_actions_excluded": ["look", "inventory", "help", "examine <object-or-receptacle>"],
                "reward_semantics": "TextWorld cumulative score, not reward delta"}
    manifest["manifest_hash"] = state_digest(manifest)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    totals = Counter()
    with args.output.open("w", encoding="utf-8") as stream:
        def write(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
        for number, game in enumerate(games):
            context = {"gamefile": str(game), "split": args.split, "seed": args.seed,
                       "manifest_hash": manifest["manifest_hash"]}
            env = None
            totals["tasks_attempted"] += 1
            try:
                env = AlfworldReplayEnv(game, max_steps=args.max_steps,
                                       expert_type=args.expert_type, seed=args.seed)
                row = collect_expert(env)
                row.update(context)
                replay = replay_actions(env, [t["action"] for t in row["transitions"]])
                if replay != row["transitions"]:
                    raise ReplayMismatch("Expert trajectory failed full reset replay verification")
                row["reset_replay_verified"] = True
                row["replay_verified"] = True
                write(row)
                totals["expert_replays_verified"] += 1
                totals["expert_successes"] += int(row["success"])
                totals["expert_steps"] += len(row["transitions"])
                if not args.skip_branches and not row["success"]:
                    totals["h1_tasks_skipped_unsuccessful_expert"] += 1
                if not args.skip_branches and row["success"]:
                    probes, branches, counts = audit_horizon_one(
                        env, row["transitions"], args.max_branch_states, args.max_candidates)
                    totals.update(counts)
                    for item in probes + branches:
                        write({**context, **item})
            except Exception as error:
                totals["task_errors"] += 1
                write({**context, "kind": "task_error", "error": f"{type(error).__name__}: {error}"})
            finally:
                if env is not None:
                    totals["actual_simulator_steps"] += env.interactions
                    totals["actual_simulator_resets"] += env.resets
                    env.close()
            print(json.dumps({"completed": number + 1, "total": len(games), "counts": dict(totals)}), flush=True)
    summary = {**manifest, "counts": dict(totals),
               "interpretation": "No policy improvement claim follows from this diagnostic.",
               "next_if_no_nontrivial_h1": "Preregister horizon 2/3 macros with equal action budgets and fresh suffix replay; do not relabel identity/noop as nontrivial."}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if totals["expert_replays_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
