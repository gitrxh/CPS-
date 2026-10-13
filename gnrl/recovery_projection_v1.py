"""Recovery collection/pool contracts. Import is CPU-only and performs no work.

This module does not choose labels, train, evaluate held-out tasks or authorize
jobs. Every environment operation goes through BudgetLedger before invocation.
Factories have signature factory(gamefile, expert_type), returning an existing
AlfworldReplayEnv-compatible object with seed20261012 and max_steps100.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import Counter

from .alfworld_env import information_only_action

VERSION = "recovery-projection-v1"
SEED = 20261012
ANCHOR_STEPS = (4, 8, 16)
MAX_STEPS = 100
STRUCTURAL_STATUSES = frozenset({"terminated_before_anchor", "terminated_at_anchor",
    "no_admissible_before_anchor", "no_admissible_at_anchor"})
GLOBAL_LIMITS = {"environment_steps_started": 224000, "resets_started": 3600,
                 "generated_tokens": 4000000}
COUNTERS = tuple(GLOBAL_LIMITS) + ("environment_steps_completed", "resets_completed",
    "policy_calls_started", "policy_calls_completed", "forward_tokens")


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def relative_game(gamefile):
    marker = "/json_2.1.1/"
    require(isinstance(gamefile, str) and gamefile.count(marker) == 1,
            "Game identity lacks unique json_2.1.1 root")
    relative = gamefile.split(marker, 1)[1]
    require(relative.startswith("train/") and ".." not in relative.split("/"),
            "Recovery collection only accepts frozen training identities")
    return relative


class BudgetLedger:
    """Durable callback fires before calls, so errors/kill cannot erase starts.

    Limits are both cumulative and stage-local. This object is intentionally
    serial; parallel processes require controller-owned disjoint allocations.
    Unknown generated work after an exception is explicit, never claimed zero.
    """
    def __init__(self, *, stage_limits, deadline, initial_totals=None, persist=None, global_limits=None):
        self.global_limits = dict(GLOBAL_LIMITS if global_limits is None else global_limits)
        require(set(self.global_limits) == set(GLOBAL_LIMITS) and
                all(type(v) is int and 0 <= v <= GLOBAL_LIMITS[k] for k, v in self.global_limits.items()),
                "Invalid cumulative limits")
        require(set(stage_limits) == set(GLOBAL_LIMITS), "Wrong stage-limit keys")
        require(all(type(v) is int and 0 <= v <= GLOBAL_LIMITS[k]
                    for k, v in stage_limits.items()), "Invalid stage limits")
        self.limits = dict(stage_limits)
        self.initial = dict.fromkeys(COUNTERS, 0)
        if initial_totals is not None:
            require(set(initial_totals) == set(COUNTERS), "Incomplete inherited cost counters")
            require(all(type(v) is int and v >= 0 for v in initial_totals.values()), "Invalid inherited costs")
            self.initial.update(initial_totals)
        require(all(self.initial[k] <= v for k, v in self.global_limits.items()), "Inherited budget exhausted")
        self.counts = dict.fromkeys(COUNTERS, 0)
        self.deadline, self.persist = deadline, persist
        self.started = time.monotonic()
        self.active_call = None
        self.unknown_generation_after_error = False
        self.by_label = Counter()

    @property
    def totals(self):
        return {k: self.initial[k] + self.counts[k] for k in COUNTERS}

    def snapshot(self):
        return {"schema_version": 1, "kind": "recovery_cost_ledger", "totals": self.totals,
                "stage_counts": dict(self.counts), "stage_limits": self.limits,
                "global_limits": self.global_limits, "calls_by_label": dict(self.by_label),
                "active_call": self.active_call, "unknown_generation_after_error": self.unknown_generation_after_error,
                "elapsed_seconds": time.monotonic() - self.started}

    def flush(self):
        if self.persist:
            self.persist(self.snapshot())

    def check(self):
        require(time.monotonic() < self.deadline, "Recovery stage/global wall budget exhausted")

    def reserve(self, key, n=1):
        self.check()
        require(self.counts[key] + n <= self.limits[key] and
                self.totals[key] + n <= self.global_limits[key], "Recovery budget exhausted: " + key)

    def _environment(self, env, operation, label, *args):
        key = "resets" if operation == "reset" else "environment_steps"
        self.reserve(key + "_started")
        self.counts[key + "_started"] += 1
        self.by_label[label + ":" + operation] += 1
        self.active_call = {"operation": operation, "label": label}
        self.flush()
        result = getattr(env, operation)(*args)
        self.counts[key + "_completed"] += 1
        self.active_call = None
        self.flush()
        self.check()
        return result

    def reset(self, env, label):
        return self._environment(env, "reset", label)

    def step(self, env, action, label):
        return self._environment(env, "step", label, action)

    def generate(self, policy, context, *, label="learner_collection"):
        self.reserve("generated_tokens", 64)
        self.counts["policy_calls_started"] += 1
        self.active_call = {"operation": "generate", "label": label}
        self.by_label[label + ":generate"] += 1
        self.flush()
        generated_before = policy.generated_tokens
        forward_before = policy.forward_tokens
        try:
            action, details = policy.choose_action(context["task"], copy.deepcopy(context["history"]),
                context["observation"], list(context["admissible_commands"]), "generate")
        except BaseException:
            self.unknown_generation_after_error = True
            # Counters may include completed emissions even when later decoding
            # failed. Preserve this known lower bound and flag the remainder.
            for name, before in (("generated_tokens", generated_before), ("forward_tokens", forward_before)):
                delta = getattr(policy, name, before) - before
                if type(delta) is int and delta >= 0:
                    self.counts[name] += delta
            self.flush()
            raise
        generated = policy.generated_tokens - generated_before
        forward = policy.forward_tokens - forward_before
        require(type(generated) is int and 0 <= generated <= 64 and
                type(forward) is int and forward >= 0, "Policy token accounting differs")
        self.counts["generated_tokens"] += generated
        self.counts["forward_tokens"] += forward
        self.counts["policy_calls_completed"] += 1
        self.active_call = None
        self.flush()
        require(details.get("selection") == "generate" and details.get("generated_tokens") == generated
                and isinstance(details.get("raw_completion"), str) and details["raw_completion"].strip() == action
                and details.get("is_admissible") is (action in context["admissible_commands"]),
                "Raw learner generation contract differs")
        self.check()
        return action, details


def state_record(state):
    """Neutral evidence: exclude expert_action so planner/no-planner can compare."""
    result = {"observation": state.observation, "admissible_commands": list(state.admissible_commands),
              "reward": state.reward, "done": state.done, "won": state.won,
              "canonical_state": copy.deepcopy(state.canonical_state)}
    require(isinstance(result["observation"], str) and
            all(isinstance(a, str) for a in result["admissible_commands"]), "Malformed environment text")
    require(type(state.done) is bool and type(state.won) is bool and
            state.canonical_state["done"] is state.done and state.canonical_state["won"] is state.won,
            "Environment flag/canonical-state disagreement")
    require(state.canonical_state["max_steps"] == MAX_STEPS and
            state.canonical_state["admissible_commands"] == result["admissible_commands"], "Environment protocol differs")
    return result


def transition(before, action, after, index, policy=None):
    result = {"step": index, "before": state_record(before), "action": action,
              "after": state_record(after)}
    require(result["before"]["canonical_state"]["elapsed_steps"] == index and
            result["after"]["canonical_state"]["elapsed_steps"] == index + 1,
            "Environment action index differs")
    if policy is not None:
        result["policy"] = copy.deepcopy(policy)
    return result


def context(task, prefix, state):
    """Only this explicit allowlist is passed to the student/scorer/trainer."""
    return {"task": task, "history": [{"observation": t["before"]["observation"], "action": t["action"]}
            for t in prefix], "observation": state["observation"],
            "admissible_commands": list(state["admissible_commands"])}


def check_env(env, gamefile, expert_type):
    require(env.gamefile == gamefile and env.seed == SEED and env.max_steps == MAX_STEPS
            and env.expert_type == expert_type, "Environment identity/seed/expert mode differs")


def collect_learner_task(env, policy, gamefile, ledger, emit=lambda event: None):
    """One complete frozen greedy rollout; returns exactly three anchor slots."""
    relative_game(gamefile)
    check_env(env, gamefile, None)
    current = ledger.reset(env, "collection")
    initial = state_record(current)
    require(initial["canonical_state"]["elapsed_steps"] == 0, "Reset did not reset interaction count")
    task, trace = initial["observation"], []
    emit({"kind": "collection_initial", "gamefile": gamefile, "state": initial})
    reason = "environment_done" if current.done else "max_steps"
    while len(trace) < MAX_STEPS and not current.done:
        if not current.admissible_commands:
            reason = "no_admissible_commands"
            break
        prompt = context(task, trace, state_record(current))
        action, policy_row = ledger.generate(policy, prompt)
        after = ledger.step(env, action, "collection")
        row = transition(current, action, after, len(trace), policy_row)
        trace.append(row)
        emit({"kind": "collection_transition", "gamefile": gamefile, "transition": row})
        current = after
        if current.done:
            reason = "environment_done"
    require(current.done or not current.admissible_commands, "Environment failed its100-action bound")
    slots = []
    for k in ANCHOR_STEPS:
        anchor = {"schema_version": 1, "kind": "recovery_anchor", "gamefile": gamefile,
                  "seed": SEED, "after_actions": k, "anchor_id": digest([gamefile, SEED, k])}
        if k > len(trace):
            anchor.update(status="terminated_before_anchor" if current.done else "no_admissible_before_anchor")
        else:
            at = trace[k - 1]["after"]
            if at["done"]:
                anchor.update(status="terminated_at_anchor")
            elif not at["admissible_commands"]:
                anchor.update(status="no_admissible_at_anchor")
            else:
                require(k < len(trace), "Active anchor lacks recorded learner next action")
                anchor.update(status="available", initial=copy.deepcopy(initial),
                    prefix_transitions=copy.deepcopy(trace[:k]), anchor_state=copy.deepcopy(at),
                    student_context=context(task, trace[:k], at), recorded_learner_next_action=trace[k]["action"])
                validate_anchor(anchor)
        slots.append(anchor)
    result = {"schema_version": 1, "kind": "recovery_learner_episode", "complete": True,
              "gamefile": gamefile, "seed": SEED, "initial": initial, "transitions": trace,
              "stop_reason": reason, "success": any(t["after"]["won"] for t in trace) or initial["won"],
              "environment_steps": len(trace), "anchors": slots}
    result["artifact_sha256"] = digest(result)
    return result


def validate_anchor(anchor):
    require(anchor["status"] == "available" and anchor["seed"] == SEED and
            anchor["after_actions"] in ANCHOR_STEPS, "Wrong available anchor")
    relative_game(anchor["gamefile"])
    k = anchor["after_actions"]
    trace = anchor["prefix_transitions"]
    require(len(trace) == k and anchor["anchor_id"] == digest([anchor["gamefile"], SEED, k]), "Anchor identity/prefix differs")
    previous = anchor["initial"]
    require(previous["canonical_state"]["elapsed_steps"] == 0, "Anchor initial count differs")
    for i, t in enumerate(trace):
        require(t["step"] == i and t["before"] == previous and not previous["done"]
                and t["after"]["canonical_state"]["elapsed_steps"] == i + 1,
                "Broken/terminal anchor prefix")
        previous = t["after"]
    require(previous == anchor["anchor_state"] and not previous["done"] and previous["admissible_commands"],
            "Anchor state differs or is unavailable")
    require(anchor["student_context"] == context(anchor["initial"]["observation"], trace, previous),
            "Student context is not exact actual causal prefix")
    require(isinstance(anchor["recorded_learner_next_action"], str), "Missing recorded learner action")


def validate_anchor_identity(anchor):
    require(anchor.get("schema_version") == 1 and anchor.get("kind") == "recovery_anchor"
            and anchor.get("seed") == SEED and anchor.get("after_actions") in ANCHOR_STEPS,
            "Invalid anchor schema/identity")
    relative_game(anchor["gamefile"])
    require(anchor["anchor_id"] == digest([anchor["gamefile"], SEED, anchor["after_actions"]]),
            "Anchor identity differs")
    require(anchor["status"] in STRUCTURAL_STATUSES | {"available"},
            "Unknown/technical anchor status cannot be structural")


def validate_episode(row):
    body = copy.deepcopy(row)
    expected = body.pop("artifact_sha256")
    require(digest(body) == expected and row.get("complete") is True
            and row.get("kind") == "recovery_learner_episode" and row.get("schema_version") == 1
            and row.get("seed") == SEED, "Incomplete/changed learner episode")
    relative_game(row["gamefile"])
    trace, previous = row["transitions"], row["initial"]
    require(previous["canonical_state"]["elapsed_steps"] == 0 and len(trace) <= MAX_STEPS
            and row["environment_steps"] == len(trace), "Learner episode interaction count differs")
    for i, t in enumerate(trace):
        require(t["step"] == i and t["before"] == previous and not previous["done"]
                and previous["admissible_commands"] and isinstance(t["action"], str)
                and t["after"]["canonical_state"]["elapsed_steps"] == i + 1,
                "Broken learner episode prefix")
        policy = t["policy"]
        require(policy.get("selection") == "generate" and
                policy["raw_completion"].strip() == t["action"] and
                policy.get("is_admissible") is (t["action"] in previous["admissible_commands"]),
                "Learner episode generation differs")
        previous = t["after"]
    require(previous["done"] or not previous["admissible_commands"], "Partial learner episode")
    require(row["stop_reason"] == ("environment_done" if previous["done"] else "no_admissible_commands")
            and row["success"] is bool(previous["won"]), "Learner final outcome differs")
    require([a["after_actions"] for a in row["anchors"]] == list(ANCHOR_STEPS), "Missing anchor slots")
    for anchor in row["anchors"]:
        validate_anchor_identity(anchor)
        require(anchor["gamefile"] == row["gamefile"], "Anchor game differs from episode")
        k = anchor["after_actions"]
        if k > len(trace):
            status = "terminated_before_anchor" if previous["done"] else "no_admissible_before_anchor"
        else:
            at = trace[k - 1]["after"]
            status = "terminated_at_anchor" if at["done"] else "no_admissible_at_anchor" if not at["admissible_commands"] else "available"
        require(anchor["status"] == status, "Structural anchor status contradicts actual episode")
        if status == "available":
            validate_anchor(anchor)
            require(k < len(trace) and anchor["initial"] == row["initial"] and
                    anchor["prefix_transitions"] == trace[:k] and
                    anchor["recorded_learner_next_action"] == trace[k]["action"],
                    "Anchor is not actual collected prefix/action")


def restore_anchor(env, anchor, ledger, label):
    validate_anchor(anchor)
    current = ledger.reset(env, label)
    require(state_record(current) == anchor["initial"], "Reset observation/raw state differs")
    for t in anchor["prefix_transitions"]:
        require(state_record(current) == t["before"], "Replay pre-state differs")
        current = ledger.step(env, t["action"], label)
        require(state_record(current) == t["after"], "Replay successor differs")
    require(state_record(current) == anchor["anchor_state"], "Restored anchor differs")
    return current


def propose_actions(anchor, planner_first):
    validate_anchor(anchor)
    commands = anchor["anchor_state"]["admissible_commands"]
    actions = []
    if planner_first is not None:
        require(isinstance(planner_first, str) and planner_first in commands, "Native planner first action is inadmissible")
        actions.append(planner_first)
    learner = anchor["recorded_learner_next_action"]
    if learner in commands and learner not in actions and not information_only_action(learner):
        actions.append(learner)
    salt = "recovery_projection_v1_action_pool|" + relative_game(anchor["gamefile"]) + "|" + str(anchor["after_actions"]) + "|"
    remaining = [a for a in set(commands) if a not in actions and not information_only_action(a)]
    remaining.sort(key=lambda a: (hashlib.sha256((salt + a).encode("utf-8")).hexdigest(), a.encode("utf-8")))
    return (actions + remaining)[:8]


def certify_anchor(env_factory, anchor, ledger, emit=lambda event: None):
    """One planner mode, full fixed proposals. Failure does not expand the pool."""
    validate_anchor_identity(anchor)
    result = {"schema_version": 1, "kind": "recovery_oracle_pool", "anchor": copy.deepcopy(anchor),
              "anchor_id": anchor["anchor_id"], "status": anchor["status"], "eligible": False,
              "expert_action": None, "proposed_actions": [], "branches": [], "retained_candidates": []}
    if anchor["status"] != "available":
        result["complete"] = True
        result["artifact_sha256"] = digest(result)
        return result
    validate_anchor(anchor)
    env = env_factory(anchor["gamefile"], "planner")
    try:
        check_env(env, anchor["gamefile"], "planner")
        # This restore is the first candidate's restore, not an unbudgeted probe.
        current = restore_anchor(env, anchor, ledger, "oracle_prefix")
        expert = current.expert_action
        actions = propose_actions(anchor, expert)
        result.update(expert_action=expert, proposed_actions=actions)
        for index, action in enumerate(actions):
            if index:
                current = restore_anchor(env, anchor, ledger, "oracle_prefix")
            trace, reason = [], "no_plan"
            next_action = action
            while not current.done:
                if next_action is None:
                    reason = "no_plan"
                    break
                require(next_action in current.admissible_commands, "Planner emitted inadmissible continuation")
                after = ledger.step(env, next_action, "oracle_branch")
                row = transition(current, next_action, after, anchor["after_actions"] + len(trace))
                trace.append(row)
                emit({"kind": "oracle_transition", "anchor_id": anchor["anchor_id"],
                      "candidate_index": index, "transition": row})
                current = after
                if current.done:
                    reason = "won" if current.won else "episode_limit" if current.canonical_state["truncated"] else "environment_done"
                    break
                require(anchor["after_actions"] + len(trace) < MAX_STEPS, "Oracle environment exceeded episode budget")
                next_action = current.expert_action
            branch = {"candidate_index": index, "action": action, "success": bool(current.won and current.done),
                      "stop_reason": reason, "transitions": trace, "final_state": state_record(current)}
            require(trace and trace[0]["action"] == action, "Missing actual candidate transition")
            branch["certificate_sha256"] = digest(branch)
            result["branches"].append(branch)
            if branch["success"]:
                result["retained_candidates"].append({"action": action, "candidate_index": index,
                    "certificate_sha256": branch["certificate_sha256"], "first_transition": copy.deepcopy(trace[0])})
            emit({"kind": "oracle_branch_complete", "anchor_id": anchor["anchor_id"], "branch": branch})
        result["eligible"] = expert is not None and any(c["action"] == expert for c in result["retained_candidates"])
        result["status"] = "eligible" if result["eligible"] else "no_certified_expert_action"
        result["complete"] = True
        result["artifact_sha256"] = digest(result)
        return result
    finally:
        env.close()


def validate_pool(pool):
    value = copy.deepcopy(pool)
    expected = value.pop("artifact_sha256")
    require(digest(value) == expected and pool["complete"] is True, "Pool bytes/complete flag differ")
    anchor = pool["anchor"]
    validate_anchor_identity(anchor)
    require(pool.get("schema_version") == 1 and pool.get("kind") == "recovery_oracle_pool"
            and pool["anchor_id"] == anchor["anchor_id"], "Pool identity/schema differs")
    if anchor["status"] != "available":
        require(pool["status"] == anchor["status"] and pool["eligible"] is False
                and pool["expert_action"] is None and not pool["proposed_actions"]
                and not pool["branches"] and not pool["retained_candidates"], "Unavailable anchor has candidates")
        return
    validate_anchor(anchor)
    require(pool["proposed_actions"] == propose_actions(anchor, pool["expert_action"]), "Fixed proposal changed")
    require(len(pool["branches"]) == len(pool["proposed_actions"]), "Missing proposed branch")
    retained = []
    for i, b in enumerate(pool["branches"]):
        body = copy.deepcopy(b)
        cert = body.pop("certificate_sha256")
        require(digest(body) == cert and b["candidate_index"] == i and b["action"] == pool["proposed_actions"][i], "Branch certificate differs")
        previous = anchor["anchor_state"]
        for j, t in enumerate(b["transitions"]):
            require(t["before"] == previous and not previous["done"] and t["action"] in previous["admissible_commands"]
                    and t["step"] == anchor["after_actions"] + j
                    and t["after"]["canonical_state"]["elapsed_steps"] == t["step"] + 1, "Broken oracle branch")
            previous = t["after"]
        require(b["transitions"] and b["transitions"][0]["action"] == b["action"] and previous == b["final_state"]
                and type(b["success"]) is bool and b["success"] == bool(previous["won"] and previous["done"]), "Oracle success evidence differs")
        require(anchor["after_actions"] + len(b["transitions"]) <= MAX_STEPS and
                b["stop_reason"] == ("won" if previous["done"] and previous["won"] else
                "episode_limit" if previous["done"] and previous["canonical_state"]["truncated"] else
                "environment_done" if previous["done"] else "no_plan"), "Oracle stopping contract differs")
        if b["success"]:
            retained.append({"action": b["action"], "candidate_index": i, "certificate_sha256": cert,
                             "first_transition": b["transitions"][0]})
    require(retained == pool["retained_candidates"] and pool["eligible"] is
            (pool["expert_action"] is not None and any(c["action"] == pool["expert_action"] for c in retained)), "Pool eligibility differs")
    require(pool["status"] == ("eligible" if pool["eligible"] else "no_certified_expert_action"),
            "Pool eligibility status differs")


def replay_certificates(env_factory, pool, ledger, emit=lambda event: None):
    validate_pool(pool)
    anchor = pool["anchor"]
    checked = []
    if pool["retained_candidates"]:
        env = env_factory(anchor["gamefile"], None)
        try:
            check_env(env, anchor["gamefile"], None)
            for candidate in pool["retained_candidates"]:
                current = restore_anchor(env, anchor, ledger, "candidate_replay_prefix")
                after = ledger.step(env, candidate["action"], "candidate_replay_action")
                actual = transition(current, candidate["action"], after, anchor["after_actions"])
                require(actual == candidate["first_transition"], "Planner/no-planner candidate successor differs")
                row = {"action": candidate["action"], "certificate_sha256": candidate["certificate_sha256"],
                       "replay_verified": True, "actual_transition": actual}
                checked.append(row)
                emit({"kind": "candidate_replay_verified", "anchor_id": anchor["anchor_id"], **row})
        finally:
            env.close()
    return {"schema_version": 1, "kind": "recovery_candidate_replay", "complete": True,
            "anchor_id": anchor["anchor_id"], "pool_sha256": pool["artifact_sha256"], "candidates": checked}
