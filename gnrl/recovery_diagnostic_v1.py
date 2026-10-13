"""Finite viable-action scoring and actual learner continuation, with inert imports.

No stage here authorizes execution. Factories/policies are injected; real workers
require an independently reviewed parent adoption. Teacher success never supplies Q.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time

from . import recovery_projection_v1 as p
from .policy import encode_action

ARMS = ("expert_recovery", "uniform_viable_recovery", "reference_conditional_recovery")
UNAVAILABLE = frozenset(("terminated_before_anchor", "terminated_at_anchor",
    "no_admissible_before_anchor", "no_admissible_at_anchor", "no_certified_expert_action"))


def seal(row):
    row = copy.deepcopy(row)
    row["artifact_sha256"] = p.digest(row)
    return row


def verify_seal(row):
    body = dict(row)
    expected = body.pop("artifact_sha256")
    p.require(p.digest(body) == expected, "Diagnostic artifact digest differs")


def pool_actions(pool):
    p.validate_pool(pool)
    if not pool["eligible"]:
        p.require(pool["status"] in UNAVAILABLE, "Unknown/technical unavailability")
        return []
    p.require(pool["status"] == "eligible", "Eligible pool status differs")
    actions = [c["action"] for c in pool["retained_candidates"]]
    p.require(1 <= len(actions) <= 8 and len(set(actions)) == len(actions)
              and pool["expert_action"] in actions, "Invalid certified support")
    return sorted(actions, key=lambda a: a.encode("utf-8"))


def probabilities(actions, scores):
    p.require(actions and len(actions) == len(set(actions)) and set(scores) == set(actions),
              "Incomplete probability support")
    p.require(all(type(scores[a]) in (int, float) and math.isfinite(scores[a]) for a in actions),
              "Nonfinite/incomplete action score")
    top = max(scores.values())
    weights = [math.exp(scores[a] - top) for a in actions]
    denominator = math.fsum(weights)
    return {a: w / denominator for a, w in zip(actions, weights)}


def draw(actions, masses, salt, gamefile, elapsed):
    """One deterministic SHA64 draw; stable UTF-8 ordering and no redraw."""
    ordered = sorted(actions, key=lambda a: a.encode("utf-8"))
    p.require(ordered and len(set(ordered)) == len(ordered) and set(masses) == set(ordered),
              "Draw support differs")
    p.require(all(math.isfinite(masses[a]) and masses[a] >= 0 for a in ordered)
              and abs(math.fsum(masses.values()) - 1) <= 1e-12, "Invalid CDF masses")
    material = f"{salt}|{p.relative_game(gamefile)}|{elapsed}"
    integer = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    # Keep exact integer recorded; nextafter avoids the rare float rounding to1.
    u = min(integer / 2**64, math.nextafter(1.0, 0.0))
    cdf, chosen, boundaries = 0.0, None, []
    for index, action in enumerate(ordered):
        cdf = math.fsum(masses[a] for a in ordered[:index + 1])
        boundaries.append(cdf)
        if chosen is None and u < cdf:
            chosen = action
    if chosen is None:
        chosen = next(a for a in reversed(ordered) if masses[a] > 0)
    return {"action": chosen, "salt": salt, "sha64_integer": integer,
            "u": u, "ordered_actions": ordered, "cdf": boundaries}


class ScoreLedger:
    """Pre-reserved padded-token work shared by both scorers; no score cache."""
    def __init__(self, *, target_limit, forward_limit, deadline, persist=lambda row: None):
        p.require(0 < target_limit <= 500000 and 0 < forward_limit <= 7000000,
                  "Invalid score limits")
        self.target_limit, self.forward_limit = target_limit, forward_limit
        self.deadline, self.persist = deadline, persist
        self.target_tokens = self.forward_tokens = self.calls_started = self.calls_completed = 0
        self.active_call = None

    def snapshot(self):
        return {"kind": "recovery_score_ledger", "scored_target_tokens": self.target_tokens,
                "forward_tokens": self.forward_tokens, "calls_started": self.calls_started,
                "calls_completed": self.calls_completed, "active_call": self.active_call,
                "target_limit": self.target_limit, "forward_limit": self.forward_limit}

    def check(self):
        p.require(time.monotonic() < self.deadline, "Score/global deadline exhausted")

    def score(self, policy, pool, role, emit=lambda row: None):
        actions = pool_actions(pool)
        p.require(actions and role in ("reference", "base"), "Scoring unsupported role/pool")
        anchor = pool["anchor"]
        p.validate_anchor(anchor)
        ctx = p.context(anchor["initial"]["observation"], anchor["prefix_transitions"], anchor["anchor_state"])
        prompt = policy.prompt_ids(ctx["task"], ctx["history"], ctx["observation"], ctx["admissible_commands"])
        targets = {a: encode_action(policy.tokenizer, a, 64) for a in actions}
        results = {}
        for offset in range(0, len(actions), 2):
            batch = actions[offset:offset + 2]
            target_count = sum(len(targets[a]) for a in batch)
            forward_count = len(batch) * (len(prompt) + max(len(targets[a]) for a in batch))
            self.check()
            p.require(len(prompt) + max(len(targets[a]) for a in batch) <= 2048,
                      "Scoring context exceeds frozen protocol")
            p.require(self.target_tokens + target_count <= self.target_limit and
                      self.forward_tokens + forward_count <= self.forward_limit, "Score token budget exhausted")
            self.calls_started += 1
            self.active_call = {"anchor_id": pool["anchor_id"], "role": role,
                                "actions": batch, "reserved_target_tokens": target_count,
                                "reserved_forward_tokens": forward_count}
            self.persist(self.snapshot())
            before = (policy.scored_tokens, policy.forward_tokens, policy.generated_tokens)
            rows = policy.score_encoded_actions(prompt, batch)
            p.require([r["action"] for r in rows] == batch and
                      policy.scored_tokens - before[0] == target_count and
                      policy.forward_tokens - before[1] == forward_count and
                      policy.generated_tokens == before[2], "Score work accounting differs")
            self.target_tokens += target_count
            self.forward_tokens += forward_count
            self.calls_completed += 1
            self.active_call = None
            self.persist(self.snapshot())
            for action, row in zip(batch, rows):
                p.require(row["token_count"] == len(targets[action]) and math.isfinite(row["logprob"]),
                          "Invalid full command+EOS score")
                results[action] = {"logprob": row["logprob"], "action_tokens_including_eos": len(targets[action]),
                    "prompt_tokens": len(prompt), "prompt_ids_sha256": p.digest(prompt),
                    "action_ids_sha256": p.digest(targets[action]), "context_sha256": p.digest(ctx)}
                emit({"kind": "recovery_action_score", "anchor_id": pool["anchor_id"],
                      "pool_sha256": pool["artifact_sha256"], "role": role, "action": action, **results[action]})
            self.check()
        return results


def selection(pool, reference_scores, base_scores):
    actions = pool_actions(pool)
    result = {"kind": "recovery_selection", "schema_version": 1, "complete": True,
              "anchor_id": pool["anchor_id"], "pool_sha256": pool["artifact_sha256"],
              "eligible": pool["eligible"], "status": pool["status"]}
    if not actions:
        p.require(not reference_scores and not base_scores, "Unavailable pool received scores")
        return seal(result)
    p.require(set(reference_scores) == set(base_scores) == set(actions), "Missing complete two-reference scores")
    anchor = pool["anchor"]
    expected_context = p.digest(p.context(anchor["initial"]["observation"],
        anchor["prefix_transitions"], anchor["anchor_state"]))
    for a in actions:
        for score in (reference_scores[a], base_scores[a]):
            p.require(score["context_sha256"] == expected_context
                      and type(score["prompt_tokens"]) is int and score["prompt_tokens"] > 0
                      and type(score["action_tokens_including_eos"]) is int
                      and 1 <= score["action_tokens_including_eos"] <= 64
                      and score["prompt_tokens"] + score["action_tokens_including_eos"] <= 2048,
                      "Score context/token protocol differs")
        p.require(reference_scores[a]["prompt_ids_sha256"] == base_scores[a]["prompt_ids_sha256"]
                  and reference_scores[a]["action_ids_sha256"] == base_scores[a]["action_ids_sha256"]
                  and reference_scores[a]["prompt_tokens"] == base_scores[a]["prompt_tokens"]
                  and reference_scores[a]["action_tokens_including_eos"] == base_scores[a]["action_tokens_including_eos"],
                  "Base/reference tokenization differs")
    q = probabilities(actions, {a: reference_scores[a]["logprob"] for a in actions})
    qb = probabilities(actions, {a: base_scores[a]["logprob"] for a in actions})
    uniform = draw(actions, dict.fromkeys(actions, 1 / len(actions)), "recovery_projection_v1_uniform",
                   anchor["gamefile"], anchor["after_actions"])
    conditional = draw(actions, q, "recovery_projection_v1_conditional", anchor["gamefile"], anchor["after_actions"])
    result.update(actions=actions, reference_scores=reference_scores, base_scores=base_scores,
                  q=q, q_base=qb, uniform_draw=uniform, conditional_draw=conditional,
                  labels=dict(zip(ARMS, (pool["expert_action"], uniform["action"], conditional["action"]))))
    return seal(result)


def validate_selection(pool, selected):
    verify_seal(selected)
    rebuilt = selection(pool, selected.get("reference_scores", {}), selected.get("base_scores", {}))
    p.require(selected == rebuilt, "Selection probabilities/draws/labels differ")


def label_rows(pool, selected):
    validate_selection(pool, selected)
    if not selected["eligible"]:
        return {}
    anchor = pool["anchor"]
    ctx = p.context(anchor["initial"]["observation"], anchor["prefix_transitions"], anchor["anchor_state"])
    return {arm: {"kind": "recovery_supervised_label", "schema_version": 1,
        "anchor_id": pool["anchor_id"], "gamefile": anchor["gamefile"], "seed": p.SEED,
        "after_actions": anchor["after_actions"], **copy.deepcopy(ctx), "action": selected["labels"][arm],
        "arm": arm, "pool_sha256": pool["artifact_sha256"], "selection_sha256": selected["artifact_sha256"]}
        for arm in ARMS}


def learner_branch(env_factory, policy, pool, candidate, ledger, emit=lambda row: None):
    """Actual reset+prefix+forced action, then untouched greedy learner to100."""
    actions = pool_actions(pool)
    p.require(candidate in pool["retained_candidates"] and candidate["action"] in actions,
              "Diagnostic candidate is not certified/eligible")
    anchor = pool["anchor"]
    env = env_factory(anchor["gamefile"], None)
    try:
        p.check_env(env, anchor["gamefile"], None)
        current = p.restore_anchor(env, anchor, ledger, "diagnostic_prefix")
        history = copy.deepcopy(anchor["prefix_transitions"])
        trace = []
        after = ledger.step(env, candidate["action"], "diagnostic_forced_action")
        first = p.transition(current, candidate["action"], after, anchor["after_actions"])
        p.require(first == candidate["first_transition"], "Actual forced successor differs from certificate")
        trace.append(first)
        history.append(first)
        emit({"kind": "diagnostic_transition", "anchor_id": pool["anchor_id"], "candidate_action": candidate["action"], "transition": first})
        current = after
        reason = "environment_done" if current.done else "max_steps"
        while not current.done:
            if not current.admissible_commands:
                reason = "no_admissible_commands"
                break
            p.require(len(history) < p.MAX_STEPS, "Diagnostic horizon exceeded")
            ctx = p.context(anchor["initial"]["observation"], history, p.state_record(current))
            action, details = ledger.generate(policy, ctx, label="diagnostic_continuation")
            after = ledger.step(env, action, "diagnostic_learner_action")
            row = p.transition(current, action, after, len(history), details)
            row["context_sha256"] = p.digest(ctx)
            trace.append(row)
            history.append(row)
            emit({"kind": "diagnostic_transition", "anchor_id": pool["anchor_id"], "candidate_action": candidate["action"], "transition": row})
            current = after
            if current.done:
                reason = "environment_done"
        result = seal({"kind": "recovery_actual_Q", "schema_version": 1, "complete": True,
            "anchor_id": pool["anchor_id"], "pool_sha256": pool["artifact_sha256"],
            "action": candidate["action"], "certificate_sha256": candidate["certificate_sha256"],
            "gamefile": anchor["gamefile"], "seed": p.SEED, "initial": copy.deepcopy(anchor["initial"]),
            "verified_prefix_transitions": copy.deepcopy(anchor["prefix_transitions"]),
            "transitions": trace, "final_state": p.state_record(current), "stop_reason": reason,
            "Q": bool(current.won), "environment_steps_including_prefix": len(history),
            "generated_tokens": sum(t.get("policy", {}).get("generated_tokens", 0) for t in trace)})
        validate_branch(pool, candidate, result)
        return result
    finally:
        env.close()


def validate_branch(pool, candidate, row):
    verify_seal(row)
    anchor = pool["anchor"]
    p.require(row["complete"] is True and row["anchor_id"] == pool["anchor_id"]
              and row["pool_sha256"] == pool["artifact_sha256"] and row["action"] == candidate["action"]
              and row["certificate_sha256"] == candidate["certificate_sha256"]
              and row["gamefile"] == anchor["gamefile"] and row["seed"] == p.SEED
              and row["initial"] == anchor["initial"]
              and row["verified_prefix_transitions"] == anchor["prefix_transitions"], "Q identity/prefix differs")
    trace = row["transitions"]
    p.require(trace and trace[0] == candidate["first_transition"], "Q lacks actual certified first action")
    previous = anchor["anchor_state"]
    history = copy.deepcopy(anchor["prefix_transitions"])
    tokens = 0
    for i, t in enumerate(trace):
        p.require(t["before"] == previous and not previous["done"] and t["step"] == len(history)
                  and t["after"]["canonical_state"]["elapsed_steps"] == len(history) + 1,
                  "Broken actual continuation")
        if i:
            ctx = p.context(anchor["initial"]["observation"], history, previous)
            details = t["policy"]
            p.require(t["context_sha256"] == p.digest(ctx) and details["selection"] == "generate"
                      and details["raw_completion"].strip() == t["action"]
                      and details["is_admissible"] is (t["action"] in previous["admissible_commands"])
                      and type(details["generated_tokens"]) is int and 0 <= details["generated_tokens"] <= 64,
                      "Continuation policy/history/token evidence differs")
            tokens += details["generated_tokens"]
        previous = t["after"]
        p.require(previous["done"] is previous["canonical_state"]["done"] and
                  previous["won"] is previous["canonical_state"]["won"], "State flags differ")
        history.append(t)
    p.require(previous == row["final_state"] and type(row["Q"]) is bool and row["Q"] is previous["won"]
              and row["Q"] == any(t["after"]["won"] for t in trace)
              and len(history) == row["environment_steps_including_prefix"] <= 100
              and tokens == row["generated_tokens"], "Actual Q/terminal/tokens differ")
    p.require((row["stop_reason"] == "environment_done" and previous["done"]) or
              (row["stop_reason"] == "no_admissible_commands" and not previous["done"]
               and not previous["admissible_commands"]), "Incomplete/technical continuation stop")


def diagnostic_summary(pools, selected_rows, planned, episodes):
    """Called only after all planned branches finish. Missing Q is never zero."""
    p.require(len(planned) == 36 and len({(a["gamefile"], a["after_actions"]) for a in planned}) == 36,
              "Diagnostic denominator is not fixed36")
    p.require(len(pools) == len(selected_rows) and len({r["anchor_id"] for r in pools}) == len(pools),
              "Duplicate/incomplete pool selections")
    indexed = {}
    label_differences = dict.fromkeys(ARMS[:2], 0)
    for pool, selected in zip(pools, selected_rows):
        validate_selection(pool, selected)
        key = (pool["anchor"]["gamefile"], pool["anchor"]["after_actions"])
        indexed[key] = (pool, selected)
        if selected["eligible"]:
            for arm in ARMS[:2]:
                label_differences[arm] += selected["labels"][arm] != selected["labels"][ARMS[2]]
    expected = set()
    for anchor in planned:
        key = (anchor["gamefile"], anchor["after_actions"])
        p.require(key in indexed, "Missing fixed diagnostic pool")
        pool, _ = indexed[key]
        if pool["eligible"]:
            expected.update((pool["anchor_id"], a) for a in pool_actions(pool))
    actual = {(r["anchor_id"], r["action"]): r for r in episodes}
    p.require(len(actual) == len(episodes) and set(actual) == expected, "Technical missing/extra/duplicate actual Q")
    cells, multi_families, heterogeneous = [], set(), 0
    multi = 0
    for anchor in planned:
        pool, selected = indexed[(anchor["gamefile"], anchor["after_actions"])]
        cell = {**anchor, "anchor_id": pool["anchor_id"], "status": pool["status"],
                "Q": None, "Vq": None, "Vu": None, "Ve": None, "Vbase": None,
                "difference_q_uniform": 0.0, "difference_q_expert": 0.0}
        if pool["eligible"]:
            values = {}
            for candidate in pool["retained_candidates"]:
                row = actual[(pool["anchor_id"], candidate["action"])]
                validate_branch(pool, candidate, row)
                values[candidate["action"]] = int(row["Q"])
            vq = math.fsum(selected["q"][a] * values[a] for a in values)
            vu = sum(values.values()) / len(values)
            ve = values[pool["expert_action"]]
            cell.update(Q=values, Vq=vq, Vu=vu, Ve=ve,
                        Vbase=math.fsum(selected["q_base"][a] * values[a] for a in values),
                        difference_q_uniform=vq-vu, difference_q_expert=vq-ve)
            if len(values) >= 2:
                multi += 1
                multi_families.add(anchor["family"])
            heterogeneous += len(set(values.values())) == 2
        else:
            p.require(pool["status"] in UNAVAILABLE, "Technical missingness cannot become zero difference")
        cells.append(cell)
    du = math.fsum(c["difference_q_uniform"] for c in cells) / 36
    de = math.fsum(c["difference_q_expert"] for c in cells) / 36
    gates = {"multiaction_coverage": multi >= 12 and len(multi_families) >= 4,
             "actual_Q_variation": heterogeneous >= 8, "q_over_uniform": du >= 2/36,
             "q_over_expert": de >= 0,
             "label_nondegeneracy": all(n >= 12 for n in label_differences.values())}
    return {"kind": "recovery_diagnostic_summary", "complete": True, "gate_evaluable": True,
            "training_gate_passed": all(gates.values()), "gate_components": gates,
            "denominator": 36, "cells": cells, "multiaction_anchors": multi,
            "multiaction_families": sorted(multi_families), "heterogeneous_actual_Q_anchors": heterogeneous,
            "label_differences": label_differences, "mean_q_minus_uniform": du, "mean_q_minus_expert": de,
            "actual_learner_branches": len(episodes), "unavailable_Q_imputed": False}
