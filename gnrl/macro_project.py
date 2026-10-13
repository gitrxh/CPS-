"""Bounded equal-length branch-and-rejoin search, followed by paired selection.

CPU: python -m gnrl.macro_project --input expert.jsonl --output pools --search-only
GPU: python -m gnrl.macro_project --pools pools/candidate_pools.jsonl --model PATH --output selected
Launch model scoring through scripts/launch_gpu.py. This is a finite candidate
search plus greedy likelihood heuristic, NOT Metropolis-Hastings or exact
path-space projection. Both training arms use exactly the same verified pools.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time

from .alfworld_env import (AlfworldReplayEnv, ReplayMismatch, information_only_action,
                          state_digest, transition_record)


class BudgetExceeded(RuntimeError):
    pass


class SearchBudget:
    def __init__(self, max_steps=100000, max_seconds=3600):
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.steps = 0
        self.resets = 0

    def check(self):
        if time.monotonic() - self.started >= self.max_seconds:
            raise BudgetExceeded("search_wall_time")

    def reset(self, env):
        self.check()
        self.resets += 1
        return env.reset()

    def step(self, env, action):
        self.check()
        if self.steps >= self.max_steps:
            raise BudgetExceeded("environment_steps")
        self.steps += 1
        return env.step(action)


def physical_key(state):
    """Only for no-op/loop exclusion; strict rejoin compares the ENTIRE state."""
    facts = [f for f in state["facts"] if str(f[0]).lower() not in {"checked", "examined"}]
    return json.dumps(facts, sort_keys=True, separators=(",", ":"))


def is_manipulation(action):
    return action.split(" ", 1)[0] in {"open", "close", "take", "move", "put", "use", "heat", "cool", "clean", "slice"}


def useful_change(original, candidate):
    """Conservative operational definition: manipulation order/content changes.

    Reordered navigation alone does not establish a task-useful alternative.
    Actual physical changes and absence of loops are checked during execution.
    """
    left = [a for a in original if is_manipulation(a)]
    right = [a for a in candidate if is_manipulation(a)]
    return bool(right) and left != right


def permutation_proposals(actions, limit, rng):
    """Bounded unique permutations; no factorial materialization for long macros."""
    original = tuple(actions)
    seen = {original}
    proposed = []

    def add(value):
        value = tuple(value)
        if value not in seen:
            seen.add(value)
            proposed.append(value)

    # Prioritize swaps of useful operations, then any adjacent swap.
    indices = [i for i, action in enumerate(actions) if is_manipulation(action)]
    pairs = list(itertools.combinations(indices, 2))
    rng.shuffle(pairs)
    pairs += [(i, i + 1) for i in range(len(actions) - 1)]
    for i, j in pairs:
        candidate = list(actions)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        add(candidate)
        if len(proposed) >= limit:
            return proposed
    if len(actions) <= 4:
        for candidate in itertools.permutations(actions):
            add(candidate)
            if len(proposed) >= limit:
                return proposed
    else:
        for _ in range(limit * 20):
            candidate = list(actions)
            rng.shuffle(candidate)
            add(candidate)
            if len(proposed) >= limit:
                break
    return proposed


def replay_checked(env, actions, budget):
    current = budget.reset(env)
    initial = current.observation
    transitions = []
    for index, action in enumerate(actions):
        if current.done or action not in current.admissible_commands:
            raise ReplayMismatch("Invalid or post-terminal action during full replay")
        after = budget.step(env, action)
        transitions.append(transition_record(current, action, after, index))
        current = after
    return initial, transitions


def restore_prefix(env, original, anchor, budget):
    current = budget.reset(env)
    for transition in original[:anchor]:
        if current.canonical_state != transition["state_before"] or current.observation != transition["observation"]:
            raise ReplayMismatch("Prefix replay differs from verified original")
        current = budget.step(env, transition["action"])
    if current.canonical_state != original[anchor]["state_before"] or current.observation != original[anchor]["observation"]:
        raise ReplayMismatch("Branch start differs from verified original")
    return current


def execute_branch(env, original, anchor, horizon, budget, rng, proposal):
    """Execute an exact expert-action multiset permutation.

    Preserving the multiset also preserves additive PDDL operator cost, which
    the current adapter's symbolic facts do not expose. Future expert actions
    are proposal information only, never student prompt information.
    """
    current = restore_prefix(env, original, anchor, budget)
    old = [t["action"] for t in original[anchor:anchor + horizon]]
    if len(proposal) != horizon or Counter(proposal) != Counter(old):
        return None, "action_multiset_mismatch"
    if any(information_only_action(a) or not (is_manipulation(a) or a.startswith("go to ")) for a in proposal):
        return None, "information_action"
    seen_states = {physical_key(current.canonical_state)}
    branch = []
    for offset in range(horizon):
        if current.done:
            return None, "early_terminal"
        action = proposal[offset]
        if action not in current.admissible_commands:
            return None, "inadmissible"
        after = budget.step(env, action)
        key = physical_key(after.canonical_state)
        if key == physical_key(current.canonical_state):
            return None, "physical_noop"
        if key in seen_states:
            return None, "physical_loop"
        seen_states.add(key)
        branch.append(transition_record(current, action, after, anchor + offset))
        current = after
    actions = [t["action"] for t in branch]
    if not useful_change(old, actions):
        return None, "no_manipulation_change"
    if current.canonical_state != original[anchor + horizon - 1]["state_after"]:
        return None, "endpoint_mismatch"
    return branch, "joined"


def candidate_id(actions):
    return hashlib.sha256(json.dumps(actions, ensure_ascii=False).encode()).hexdigest()


def make_pool(row, env, args, budget):
    if not row.get("success") or not row.get("transitions"):
        raise ValueError("Only successful expert trajectories may be projected")
    source = row["transitions"]
    actions = [t["action"] for t in source]
    initial, original = replay_checked(env, actions, budget)
    if not original[-1]["won"] or not original[-1]["done"]:
        raise ReplayMismatch("Original did not replay to terminal success")
    for expected, actual in zip(source, original):
        for key in ("observation", "next_observation", "state_before", "state_after"):
            if expected.get(key) != actual[key]:
                raise ReplayMismatch("Original trajectory field mismatch: " + key)
    if row.get("initial_observation", initial) != initial:
        raise ReplayMismatch("Initial observation mismatch")
    task = row.get("task") or initial
    if task != initial:
        # Task may be a separately extracted instruction but must not be invented
        # from privileged future state. Preserve the collector's declared task.
        task = str(task)
    seed_material = f"{args.seed}:{row['gamefile']}:{row.get('seed', 0)}"
    rng = random.Random(int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16))
    baseline_id = candidate_id(actions)
    candidates = [{"candidate_id": baseline_id, "origin": "original", "transitions": original,
                   "evidence": {"full_replay_verified": True, "strict_endpoint_match": True,
                                "terminal_success": True}}]
    seen = {baseline_id}
    counts = Counter(original_replay_verified=1)
    # Interleave horizons so a short-horizon failure does not consume all attempts.
    by_horizon = []
    for horizon in args.horizons:
        anchors = [i for i in range(len(original) - horizon + 1)
                   if sum(is_manipulation(t["action"]) for t in original[i:i + horizon]) >= 2
                   and all(is_manipulation(t["action"]) or t["action"].startswith("go to ")
                           for t in original[i:i + horizon])]
        rng.shuffle(anchors)
        by_horizon.append([(i, horizon) for i in anchors[:args.max_anchors]])
    anchors = [pair for group in itertools.zip_longest(*by_horizon) for pair in group if pair is not None]
    stop = "complete"
    try:
        for anchor, horizon in anchors:
            if counts["candidate_attempts"] >= args.max_candidates:
                stop = "candidate_budget"
                break
            counts["anchors_audited"] += 1
            macro = actions[anchor:anchor + horizon]
            proposals = [p for p in permutation_proposals(macro, args.per_anchor, rng) if useful_change(macro, p)]
            for proposal in proposals:
                if counts["candidate_attempts"] >= args.max_candidates:
                    stop = "candidate_budget"
                    break
                counts["candidate_attempts"] += 1
                counts[f"horizon_{horizon}_attempts"] += 1
                branch, reason = execute_branch(env, original, anchor, horizon, budget, rng, proposal)
                counts[reason] += 1
                if branch is None:
                    continue
                branch_actions = [t["action"] for t in branch]
                full_actions = actions[:anchor] + branch_actions + actions[anchor + horizon:]
                identity = candidate_id(full_actions)
                if identity in seen:
                    counts["duplicate_full_trajectory"] += 1
                    continue
                # Independently replay the complete changed trace, including suffix.
                replay_initial, replay = replay_checked(env, full_actions, budget)
                if replay_initial != initial or not replay[-1]["won"] or not replay[-1]["done"]:
                    raise ReplayMismatch("Joined candidate failed full successful replay")
                if replay[-1]["state_after"] != original[-1]["state_after"]:
                    raise ReplayMismatch("Full suffix ended in different strict terminal state")
                for offset, transition in enumerate(branch):
                    fresh = replay[anchor + offset]
                    for key in ("observation", "next_observation", "state_before", "state_after"):
                        if fresh[key] != transition[key]:
                            raise ReplayMismatch("Independent replay of branch differed")
                seen.add(identity)
                candidates.append({"candidate_id": identity,
                                   "origin": "expert_permutation",
                                   "transitions": replay,
                                   "evidence": {"anchor": anchor, "horizon": horizon,
                                                "original_macro": macro, "candidate_macro": branch_actions,
                                                "strict_endpoint_match": True,
                                                "endpoint_hash": state_digest(replay[anchor + horizon - 1]["state_after"]),
                                                "full_replay_verified": True, "terminal_success": True,
                                                "equal_action_count": len(replay) == len(original),
                                                "same_exact_action_multiset": True,
                                                "additive_operator_cost_preserved_by_multiset": True,
                                                "physical_noops": 0, "physical_loops": 0,
                                                "manipulation_order_or_content_changed": True}})
                counts["verified_nontrivial_candidates"] += 1
    except BudgetExceeded as error:
        stop = str(error)
    return {"kind": "macro_candidate_pool", "schema_version": 1, "gamefile": row["gamefile"],
            "seed": row.get("seed", 0), "split": row.get("split", "train"),
            "task": task, "initial_observation": initial,
            "success": True, "original_candidate_id": baseline_id,
            "candidates": candidates, "search_counts": dict(counts), "search_stop": stop,
            "equivalence": "entire_adapter_canonical_state_plus_exact_action_multiset", "proposal_uses_privileged_expert_actions": True,
            "exclusions": "Each branch action changes physical facts; no repeated physical branch state; manipulation sequence must differ."}


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def dump_line(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()


def search(args, output):
    budget = SearchBudget(args.max_env_steps, args.max_search_seconds)
    aggregate = Counter()
    pools_path = output / "candidate_pools.jsonl"
    with pools_path.open("w", encoding="utf-8") as pools, (output / "search_events.jsonl").open("w", encoding="utf-8") as events:
        for row in read_rows(args.input):
            if row.get("kind") not in (None, "trajectory"):
                continue
            if aggregate["games_considered"] >= args.max_games:
                break
            aggregate["games_considered"] += 1
            if row.get("split", "train") != "train":
                raise ValueError("Projection search accepts only training split")
            if not row.get("success"):
                aggregate["unsuccessful_sources_skipped"] += 1
                continue
            before_steps, before_resets = budget.steps, budget.resets
            env = None
            try:
                limit = row["transitions"][0]["state_before"]["max_steps"]
                env = AlfworldReplayEnv(row["gamefile"], max_steps=limit, expert_type=None, seed=row.get("seed", 0))
                pool = make_pool(row, env, args, budget)
                pool["acquisition_cost"] = {"environment_steps_including_replay": budget.steps - before_steps,
                                            "resets": budget.resets - before_resets}
                dump_line(pools, pool)
                aggregate.update(pool["search_counts"])
                aggregate["games_completed"] += 1
                aggregate["games_with_nontrivial_candidates"] += int(len(pool["candidates"]) > 1)
                event = {"gamefile": row["gamefile"], "pool_size": len(pool["candidates"]),
                         "search_stop": pool["search_stop"], "counts": pool["search_counts"], **pool["acquisition_cost"]}
                dump_line(events, event)
                print(json.dumps(event), flush=True)
                if pool["search_stop"] in {"search_wall_time", "environment_steps"}:
                    break
            except BudgetExceeded as error:
                aggregate["incomplete_original_due_to_budget"] += 1
                dump_line(events, {"gamefile": row["gamefile"], "error": str(error), "retained": False})
                break
            except (ReplayMismatch, ValueError) as error:
                aggregate["invalid_sources_or_replay_errors"] += 1
                dump_line(events, {"gamefile": row["gamefile"], "error": str(error), "retained": False})
            finally:
                if env is not None:
                    env.close()
    summary = {**dict(aggregate), "environment_steps_including_all_replay": budget.steps,
               "resets": budget.resets, "wall_seconds": time.monotonic() - budget.started,
               "interpretation": "Failure of bounded macro search is not proof that the equivalence class is singleton."}
    (output / "search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return pools_path


class TrajectoryScorer:
    def __init__(self, policy, max_score_tokens, max_forward_tokens, max_seconds):
        self.policy = policy
        self.max_score_tokens = max_score_tokens
        self.max_forward_tokens = max_forward_tokens
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.cache = {}
        self.cache_hits = 0

    def score(self, task, transitions):
        from .policy import encode_action
        history = []
        scores = []
        for transition in transitions:
            if time.monotonic() - self.started >= self.max_seconds:
                raise BudgetExceeded("score_wall_time")
            action = transition["action"]
            prompt = self.policy.prompt_ids(task, history, transition["observation"], transition["admissible_commands"])
            key = (tuple(prompt), action)
            if key not in self.cache:
                target = encode_action(self.policy.tokenizer, action, self.policy.max_action_tokens)
                if self.policy.scored_tokens + len(target) > self.max_score_tokens:
                    raise BudgetExceeded("score_token_budget")
                if self.policy.forward_tokens + len(prompt) + len(target) > self.max_forward_tokens:
                    raise BudgetExceeded("forward_token_budget")
                result = self.policy.score_encoded_actions(prompt, [action])[0]
                if not math.isfinite(result["logprob"]):
                    raise ValueError("Nonfinite reference log probability")
                self.cache[key] = result
            else:
                self.cache_hits += 1
            scores.append(self.cache[key])
            history.append({"observation": transition["observation"], "action": action})
        return {"logprob": sum(s["logprob"] for s in scores),
                "action_tokens_including_eos": sum(s["token_count"] for s in scores),
                "per_action_logprobs": [s["logprob"] for s in scores]}


def validate_pool(pool):
    candidates = pool["candidates"]
    if pool.get("split", "train") != "train" or not candidates:
        raise ValueError("Invalid training candidate pool")
    identities = [c["candidate_id"] for c in candidates]
    if len(set(identities)) != len(identities) or pool["original_candidate_id"] not in identities:
        raise ValueError("Duplicate candidates or missing original fallback")
    original = next(c for c in candidates if c["candidate_id"] == pool["original_candidate_id"])
    length = len(original["transitions"])
    for candidate in candidates:
        trace = candidate["transitions"]
        if len(trace) != length or not trace or not trace[-1]["won"] or not trace[-1]["done"]:
            raise ValueError("Candidate is not equal-length successful trajectory")
        if not candidate["evidence"].get("full_replay_verified"):
            raise ValueError("Missing full-replay evidence")
        if candidate_id([t["action"] for t in trace]) != candidate["candidate_id"]:
            raise ValueError("Candidate action digest mismatch")
        if Counter(t["action"] for t in trace) != Counter(t["action"] for t in original["transitions"]):
            raise ValueError("Candidate must preserve exact action multiset and additive operator cost")
        for i, transition in enumerate(trace):
            if transition["action"] not in transition["admissible_commands"]:
                raise ValueError("Inadmissible stored training action")
            if i and (trace[i-1]["next_observation"] != transition["observation"] or
                      trace[i-1]["state_after"] != transition["state_before"]):
                raise ValueError("Discontinuous candidate trace")
        if trace[-1]["state_after"] != original["transitions"][-1]["state_after"]:
            raise ValueError("Candidate terminal state differs")
    return original


def selected_row(pool, candidate, arm, pool_hash, scores):
    return {"kind": "trajectory", "gamefile": pool["gamefile"], "seed": pool["seed"],
            "split": pool["split"], "task": pool["task"], "initial_observation": pool["initial_observation"],
            "transitions": candidate["transitions"], "success": True, "reset_replay_verified": True,
            "replay_verified": True,
            "selection": {"arm": arm, "candidate_id": candidate["candidate_id"],
                          "original_candidate_id": pool["original_candidate_id"],
                          "pool_sha256": pool_hash, "pool_size": len(pool["candidates"]),
                          "reference_score": scores.get(candidate["candidate_id"]),
                          "method": "greedy full-trajectory sum log p(action tokens + EOS), not MH" if arm == "learner_selected" else "uniform over the identical verified candidate pool"},
            "replay_evidence": candidate["evidence"]}


def select(args, pools_path, output, policy=None):
    from .policy import HFCausalPolicy, PROTOCOL_VERSION
    scorer = None
    aggregate = Counter()
    stop = "complete"
    with (output / "uniform_valid.jsonl").open("w", encoding="utf-8") as uniform, \
         (output / "learner_selected.jsonl").open("w", encoding="utf-8") as selected, \
         (output / "selection_events.jsonl").open("w", encoding="utf-8") as events:
        for pool in read_rows(pools_path):
            if aggregate["pools_considered"] >= args.max_games:
                stop = "max_games"
                break
            original = validate_pool(pool)
            aggregate["pools_considered"] += 1
            scores = {}
            if len(pool["candidates"]) > 1:
                if policy is None:
                    if not args.model:
                        raise ValueError("A local --model is required to score nontrivial pools")
                    policy = HFCausalPolicy(args.model, adapter_path=args.adapter, device=args.device,
                                            max_context_tokens=args.max_context_tokens,
                                            max_action_tokens=args.max_action_tokens,
                                            score_batch_size=1)
                if scorer is None:
                    scorer = TrajectoryScorer(policy, args.max_score_tokens, args.max_forward_tokens, args.max_score_seconds)
                try:
                    for candidate in pool["candidates"]:
                        scores[candidate["candidate_id"]] = scorer.score(pool["task"], candidate["transitions"])
                except BudgetExceeded as error:
                    stop = str(error)
                    # Never compare different partial pools. Both arms omit this
                    # task and all later tasks; only completed paired pools emit.
                    dump_line(events, {"gamefile": pool["gamefile"], "omitted_from_both_arms": True, "reason": stop})
                    break
                best = max(pool["candidates"], key=lambda c: scores[c["candidate_id"]]["logprob"])
            else:
                best = original
                aggregate["identity_only_pools"] += 1
            pool_hash = hashlib.sha256(json.dumps(pool, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            rng = random.Random(f"{args.seed}:{pool_hash}")
            random_candidate = rng.choice(pool["candidates"])
            dump_line(uniform, selected_row(pool, random_candidate, "uniform_valid", pool_hash, scores))
            dump_line(selected, selected_row(pool, best, "learner_selected", pool_hash, scores))
            aggregate["paired_trajectories_written"] += 1
            aggregate["learner_changed_trajectories"] += int(best["candidate_id"] != pool["original_candidate_id"])
            aggregate["uniform_changed_trajectories"] += int(random_candidate["candidate_id"] != pool["original_candidate_id"])
            delta = None if not scores else scores[best["candidate_id"]]["logprob"] - scores[original["candidate_id"]]["logprob"]
            dump_line(events, {"gamefile": pool["gamefile"], "pool_sha256": pool_hash,
                               "pool_size": len(pool["candidates"]), "scores": scores,
                               "learner_candidate": best["candidate_id"], "uniform_candidate": random_candidate["candidate_id"],
                               "learner_minus_original_logprob": delta})
            # Bound cache lifetime by task; no cross-task reuse is needed.
            if scorer:
                scorer.cache.clear()
    summary = {**dict(aggregate), "stop_reason": stop, "protocol": PROTOCOL_VERSION,
               "score_tokens": getattr(policy, "scored_tokens", 0),
               "forward_tokens": getattr(policy, "forward_tokens", 0),
               "generated_tokens": getattr(policy, "generated_tokens", 0),
               "cache_hits": scorer.cache_hits if scorer else 0,
               "scoring_seconds": time.monotonic() - scorer.started if scorer else 0,
               "candidate_pools_sha256": hashlib.sha256(Path(pools_path).read_bytes()).hexdigest(),
               "interpretation": "Selection-score increase is built into the heuristic; held-out task success remains unmeasured."}
    (output / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Verified expert trajectory JSONL for CPU candidate search")
    source.add_argument("--pools", help="Previously produced candidate_pools.jsonl for scoring")
    p.add_argument("--output", required=True)
    p.add_argument("--search-only", action="store_true")
    p.add_argument("--horizons", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--max-games", type=int, default=120)
    p.add_argument("--max-anchors", type=int, default=12, help="Per horizon, per task")
    p.add_argument("--max-candidates", type=int, default=32, help="Total executed branch proposals per task")
    p.add_argument("--per-anchor", type=int, default=8)
    p.add_argument("--max-env-steps", type=int, default=100000)
    p.add_argument("--max-search-seconds", type=float, default=3600)
    p.add_argument("--model")
    p.add_argument("--adapter")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-context-tokens", type=int, default=2048)
    p.add_argument("--max-action-tokens", type=int, default=64)
    p.add_argument("--max-score-tokens", type=int, default=1000000)
    p.add_argument("--max-forward-tokens", type=int, default=20000000)
    p.add_argument("--max-score-seconds", type=float, default=3600)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = parser().parse_args()
    positive = ("max_games", "max_anchors", "max_candidates", "per_anchor", "max_env_steps",
                "max_search_seconds", "max_context_tokens", "max_action_tokens", "max_score_tokens",
                "max_forward_tokens", "max_score_seconds")
    if any(getattr(args, key) <= 0 for key in positive):
        raise ValueError("Budgets must be positive")
    if not args.horizons or any(h < 2 or h > 32 for h in args.horizons):
        raise ValueError("Macro horizons must lie in [2, 32]")
    args.horizons = sorted(set(args.horizons))
    if args.search_only and args.pools:
        raise ValueError("--search-only requires --input")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.input or args.pools)
    (output / "config.json").write_text(json.dumps(vars(args) | {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "algorithm": "equal-length bounded BRP with replay; greedy reference selection is not MH",
        "candidate_information_regime": "offline privileged expert sequence and admissible commands",
        "student_information_regime": "causal history plus current admissible commands"}, indent=2), encoding="utf-8")
    pools_path = Path(args.pools) if args.pools else search(args, output)
    if not args.search_only:
        select(args, pools_path, output)


if __name__ == "__main__":
    main()
