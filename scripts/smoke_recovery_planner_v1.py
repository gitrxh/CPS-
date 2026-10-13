"""Frozen six-task CPU native-planner recovery smoke; preparation is not execution.

Default --plan and --self-test never instantiate an environment. --execute is a
separate, explicit operation requiring a new output directory and CPU-only env.
All simulator reset/step calls, including inherited restore(), are accounted by
the metered wrapper. No model is used and no existing data/source is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gnrl.alfworld_env import (AlfworldReplayEnv, ReplayMismatch, StepResult,
                              TASK_TYPES, information_only_action, transition_record)
from gnrl.two_object_blocks import entity_names, predicates

MANIFEST = ROOT / "data/manifests/train120.json"
MANIFEST_SHA256 = "724c18f5711bfb3d9ef002005dda34c40471617eb44f921c78f8fa7f42759af0"
SEED, MAX_STEPS = 42, 100
LIMITS = {"steps": 6000, "resets": 200, "wall_seconds": 900}
INVALID = "gnrl_nonexistent_command_recovery_smoke_v1"
CASES = ("invalid", "legal_deviation", "wrong_object")
COMPARE_FIELDS = ("step", "observation", "admissible_commands", "action",
                  "next_observation", "reward", "done", "won", "state_before", "state_after")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()


def frozen_tasks(manifest):
    if manifest.get("split") != "train" or len(manifest["games"]) != 120:
        raise ValueError("Expected frozen train120 manifest")
    first = {}
    for game in manifest["games"]:
        # Folder identifies cohort strata only. Object types come from raw facts.
        family = Path(game).parent.parent.name.split("-", 1)[0]
        if family not in TASK_TYPES:
            raise ValueError("Unknown task family in manifest")
        first.setdefault(family, game)
    if set(first) != TASK_TYPES:
        raise ValueError("Missing task family")
    return [{"family": family, "gamefile": game} for family, game in first.items()]


def physical_facts(state):
    """Ignore observation/cache bookkeeping, never moves or rendered text as proxy."""
    return sorted([f for f in state["facts"] if f[0].lower() not in {"checked", "examined"}
                   and not f[0].lower().startswith("not_")], key=lambda f: json.dumps(f))


def legal_candidates(trace):
    return [{"anchor": i, "action": action} for i, t in enumerate(trace)
            for action in sorted(set(t["admissible_commands"]))
            if action != t["expert_action"] and not information_only_action(action)]


def wrong_object_candidates(trace, names, target):
    if not isinstance(target, str) or not target:
        raise ValueError("Missing original metadata pddl_params.object_target")
    expected = (target + "Type").lower()
    result = []
    for i, t in enumerate(trace):
        facts = predicates(t["state_before"])
        object_types = {}
        for obj, typ in facts.get("objecttype", ()):
            object_types.setdefault(obj, set()).add(typ.lower())
        for action in sorted(set(t["admissible_commands"])):
            if not action.startswith("take "):
                continue
            # Match the exact grounded command, including receptacle name. Never
            # infer a type from its English display name or command substrings.
            matches = [(obj, rec) for obj, rec in facts.get("inreceptacle", ())
                       if obj in names and rec in names
                       and action == f"take {names[obj]} from {names[rec]}"]
            if len(matches) != 1:
                raise ValueError("Ambiguous/unmapped admissible take command: " + action)
            obj, rec = matches[0]
            types = object_types.get(obj, set())
            if len(types) != 1:
                raise ValueError("Missing/ambiguous grounded object type")
            typ = next(iter(types))
            if typ != expected:
                result.append({"anchor": i, "action": action, "object_id": obj,
                               "receptacle_id": rec, "object_type": typ,
                               "goal_object_type": expected})
    return result


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, stream, clock=time.monotonic):
        self.stream, self.clock = stream, clock
        self.started = clock()
        self.started_calls = Counter()
        self.completed_calls = Counter()
        self.by_phase = Counter()
        self.phase = "initialization"
        self.errors = []

    def check(self):
        if self.clock() - self.started >= LIMITS["wall_seconds"]:
            raise BudgetExceeded("wall_seconds")

    def begin(self, kind, gamefile, action=None):
        self.check()
        if self.started_calls[kind] >= LIMITS[kind]:
            raise BudgetExceeded(kind)
        self.started_calls[kind] += 1
        self.by_phase[(self.phase, kind)] += 1
        ident = sum(self.started_calls.values())
        emit(self.stream, {"event": "call_started", "id": ident, "kind": kind,
                           "phase": self.phase, "gamefile": gamefile, "action": action})
        return ident

    def complete(self, ident, kind, payload):
        self.completed_calls[kind] += 1
        emit(self.stream, {"event": "call_completed", "id": ident, "kind": kind,
                           "phase": self.phase, **payload})
        self.check()

    def error(self, ident, kind, error):
        value = {"event": "call_error", "id": ident, "kind": kind,
                 "phase": self.phase, "error": f"{type(error).__name__}: {error}"}
        self.errors.append(value)
        emit(self.stream, value)

    def receipt(self):
        return {"limits": LIMITS, "started": dict(self.started_calls),
                "completed": dict(self.completed_calls), "call_errors": self.errors,
                "by_phase": [{"phase": p, "kind": k, "count": n}
                             for (p, k), n in sorted(self.by_phase.items())],
                "wall_seconds": self.clock() - self.started}


class MeteredEnv(AlfworldReplayEnv):
    """Inherited restore uses these overrides, so its internal calls are charged."""
    def __init__(self, gamefile, budget, expert_type="planner"):
        self.budget = budget
        super().__init__(gamefile, max_steps=MAX_STEPS, expert_type=expert_type, seed=SEED)

    def reset(self):
        ident = self.budget.begin("resets", self.gamefile)
        try:
            result = super().reset()
        except BaseException as error:
            self.budget.error(ident, "resets", error)
            raise
        self.budget.complete(ident, "resets", {"initial": result.to_dict()})
        return result

    def step(self, action):
        before = self.current
        ident = self.budget.begin("steps", self.gamefile, action)
        try:
            result = super().step(action)
        except BaseException as error:
            self.budget.error(ident, "steps", error)
            raise
        self.budget.complete(ident, "steps", {
            "transition": transition_record(before, action, result, len(self.actions) - 1)})
        return result


def record_step(env, current, action, trace):
    after = env.step(action)
    trace.append(transition_record(current, action, after, len(trace)))
    return after


def finish_with_planner(env, current, trace):
    while not current.done:
        action = current.expert_action
        if not action:
            return current, "planner_no_action"
        if action not in current.admissible_commands:
            return current, "planner_action_not_admissible"
        current = record_step(env, current, action, trace)
    return current, "original_goal_won" if current.won else "episode_failed_or_100_step_limit"


def replay_verify(env, trace, expected_initial):
    current = env.reset()
    if (current.canonical_state != expected_initial.canonical_state
            or current.observation != expected_initial.observation):
        raise ReplayMismatch("Full replay initial state/observation differs")
    for i, old in enumerate(trace):
        if current.done:
            raise ReplayMismatch("Full replay terminated before trace end")
        after = env.step(old["action"])
        fresh = transition_record(current, old["action"], after, i)
        for field in COMPARE_FIELDS:
            if old[field] != fresh[field]:
                raise ReplayMismatch(f"Full replay mismatch at {i}: {field}")
        current = after
    return {"verified": True, "trace_sha256": digest(trace), "won": current.won,
            "done": current.done, "steps": len(trace),
            "terminal_state_sha256": digest(current.canonical_state)}


def restore_anchor(env, canonical, initial, anchor):
    if not 0 <= anchor < len(canonical):
        raise ValueError("Anchor not on original canonical trajectory")
    target = canonical[anchor]
    actions = [t["action"] for t in canonical[:anchor]]
    current = env.restore(actions, expected=target["state_before"])
    if current.observation != target["observation"] or current.admissible_commands != target["admissible_commands"]:
        raise ReplayMismatch("Restored anchor causal observation/admissibles differ")
    # restore() already executed every prefix action and checked the exact state.
    # Verify the full prefix a second time only by the actual ledger's records,
    # not by another environment call: metered_trace captures this restore below.
    if anchor == 0 and current.canonical_state != initial.canonical_state:
        raise ReplayMismatch("Initial anchor mismatch")
    return current


def restored_prefix(env, canonical, initial, anchor):
    """Capture and compare each actual restore transition, not a copied old prefix."""
    original_step = env.step
    actual = []

    def capture(action):
        before = env.current
        after = original_step(action)
        fresh = transition_record(before, action, after, len(actual))
        old = canonical[len(actual)]
        if any(fresh[k] != old[k] for k in COMPARE_FIELDS):
            raise ReplayMismatch("Restore prefix causal/state mismatch")
        actual.append(fresh)
        return after

    env.step = capture
    try:
        current = restore_anchor(env, canonical, initial, anchor)
    finally:
        env.step = original_step
    return current, actual


def run_case(env, replay_env, budget, kind, canonical, initial, names, metadata, record):
    if kind == "invalid":
        candidates = [{"anchor": 0, "action": INVALID}]
        if INVALID in canonical[0]["admissible_commands"]:
            raise ValueError("Fixed invalid command unexpectedly admissible")
    elif kind == "legal_deviation":
        candidates = legal_candidates(canonical)
    else:
        candidates = wrong_object_candidates(canonical, names, metadata["pddl_params"]["object_target"])
    record.update(candidate_order=candidates, candidate_order_sha256=digest(candidates), probes=[])
    if not candidates:
        record.update(status="not_applicable", reason="no_eligible_action_on_fixed_original_trajectory")
        return
    # Selection is frozen before execution, and never depends on recovery success.
    for candidate in candidates:
        budget.phase = kind + ":restore_probe"
        current, trace = restored_prefix(env, canonical, initial, candidate["anchor"])
        before = current
        if kind != "invalid" and candidate["action"] not in current.admissible_commands:
            raise ReplayMismatch("Frozen candidate is no longer admissible")
        record["active_probe_trace"] = trace  # Preserve partial traces on any error.
        current = record_step(env, current, candidate["action"], trace)
        changed = physical_facts(before.canonical_state) != physical_facts(current.canonical_state)
        record["probes"].append({**candidate, "physical_change": changed, "trace": trace[:]})
        if kind == "legal_deviation" and not changed:
            continue
        record.update(selected=candidate, trace=trace, physical_change=changed)
        if kind == "invalid":
            old, new = before.canonical_state, current.canonical_state
            if old["facts"] != new["facts"] or new["elapsed_steps"] != old["elapsed_steps"] + 1 or new["moves"] != old["moves"]:
                raise ReplayMismatch("Fixed invalid command changed facts/moves or failed to consume one wrapper step")
            record["invalid_transition_verified"] = True
        if kind == "wrong_object":
            held = {obj for _, obj in predicates(current.canonical_state).get("holds", ())}
            if candidate["object_id"] not in held or not changed:
                raise ReplayMismatch("Wrong-object take did not hold intended grounded object")
        budget.phase = kind + ":recovery"
        current, stop = finish_with_planner(env, current, trace)
        record.update(stop_reason=stop, won=current.won, done=current.done,
                      recovery_actions=len(trace) - candidate["anchor"] - 1,
                      history_before_deviation=[{"observation": t["observation"], "action": t["action"]}
                                                for t in trace[:candidate["anchor"]]])
        budget.phase = kind + ":full_replay"
        record["replay"] = replay_verify(replay_env, trace, initial)
        record["replay"]["expert_type"] = None
        record.update(status="success" if current.won else "scientific_failure")
        record.pop("active_probe_trace", None)
        return
    record.pop("active_probe_trace", None)
    record.update(status="not_applicable", reason="no_physically_changed_legal_candidate")


def plan():
    if file_hash(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("Frozen manifest bytes changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"kind": "native_planner_recovery_smoke_v1", "manifest_sha256": MANIFEST_SHA256,
            "tasks": frozen_tasks(manifest), "seed": SEED, "max_steps": MAX_STEPS,
            "expert_type": "planner", "limits": LIMITS, "cases": list(CASES),
            "invalid_action": INVALID, "invalid_anchor": 0, "gpu_jobs": 0,
            "canonical_source": "one fresh native-planner trajectory per fixed task at seed42",
            "selection": "earliest original anchor then sorted admissible command; first physical change for legal; first grounded wrong-type take",
            "failure_policy": "no retries, no task replacement, no recovery-result-dependent candidate replacement",
            "budget_scope": "all reset and step calls: canonical, restore/prefix, probes, recovery, and complete replay",
            "episode_budget_scope": "100 actions including original prefix and deviation",
            "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in
                              [Path(__file__), ROOT / "gnrl/alfworld_env.py", ROOT / "gnrl/two_object_blocks.py"]}}


def execute(output):
    prepared = plan()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Require CUDA_VISIBLE_DEVICES='' for CPU-only execution")
    if not hasattr(signal, "SIGALRM"):
        raise ValueError("Live execution requires Linux SIGALRM for the 900-second deadline")
    output = output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("All outputs must remain inside the GNRL workspace")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "plan.json").open("x", encoding="utf-8") as stream:
        json.dump(prepared, stream, indent=2)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records, technical, canonical_records = [], [], []
    stopped = None
    old_handler = signal.getsignal(signal.SIGALRM)

    def deadline(signum, frame):
        raise BudgetExceeded("wall_seconds_hard_deadline")

    with (output / "environment_ledger.jsonl").open("x", encoding="utf-8") as ledger, \
         (output / "cases.jsonl").open("x", encoding="utf-8") as cases, \
         (output / "canonical.jsonl").open("x", encoding="utf-8") as originals:
        budget = Budget(ledger)
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        try:
            for task in prepared["tasks"]:
                env = replay_env = None
                baseline = {**task, "trace": [], "status": "in_progress"}
                try:
                    budget.check()
                    game = Path(task["gamefile"])
                    if file_hash(game) != manifest["game_sha256"][str(game)]:
                        raise ValueError("Frozen game bytes changed")
                    meta_path = game.with_name("traj_data.json")
                    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                    if metadata["task_type"] != task["family"]:
                        raise ValueError("Metadata family differs from frozen task")
                    game_data = json.loads(game.read_text(encoding="utf-8"))
                    baseline.update(game_sha256=file_hash(game), metadata_sha256=file_hash(meta_path),
                                    original_pddl_problem_sha256=digest(game_data["pddl_problem"]))
                    env = MeteredEnv(game, budget)
                    replay_env = MeteredEnv(game, budget, expert_type=None)
                    budget.phase = "canonical"
                    initial = env.reset()
                    baseline["initial"] = initial.to_dict()
                    names = entity_names(env)
                    baseline["entity_names"] = names
                    current, stop = finish_with_planner(env, initial, baseline["trace"])
                    baseline.update(won=current.won, stop_reason=stop)
                    budget.phase = "canonical:full_replay"
                    baseline["replay"] = replay_verify(replay_env, baseline["trace"], initial)
                    baseline["replay"]["expert_type"] = None
                    baseline["status"] = "success" if current.won else "scientific_failure"
                    # A failed canonical baseline is retained, never replaced. Its
                    # cases are explicitly not_run, not misreported as inapplicable.
                    for kind in CASES:
                        record = {**task, "case": kind, "seed": SEED, "status": "in_progress"}
                        if not current.won:
                            record.update(status="not_run", reason="canonical_planner_failed")
                        else:
                            try:
                                run_case(env, replay_env, budget, kind, baseline["trace"], initial, names, metadata, record)
                            except Exception as error:
                                record.update(status="technical_error", error=f"{type(error).__name__}: {error}")
                                technical.append({**task, "case": kind, "error": record["error"]})
                                if isinstance(error, BudgetExceeded):
                                    stopped = str(error)
                            records.append(record)
                            emit(cases, record)
                            if record["status"] == "technical_error":
                                raise RuntimeError("Case technical error; smoke stops without retries")
                            continue
                        records.append(record)
                        emit(cases, record)
                except Exception as error:
                    technical.append({**task, "error": f"{type(error).__name__}: {error}"})
                    if baseline["status"] == "in_progress":
                        baseline.update(status="technical_error", error=f"{type(error).__name__}: {error}")
                    stopped = stopped or str(error)
                finally:
                    for opened_env in (env, replay_env):
                        if opened_env is None:
                            continue
                        try:
                            opened_env.close()
                        except Exception as error:
                            technical.append({**task, "close_error": repr(error)})
                            stopped = stopped or "close_error"
                    canonical_records.append(baseline)
                    emit(originals, baseline)
                if stopped:
                    break
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        summary = {"kind": prepared["kind"], "complete": len(records) == 18 and not technical,
                   "planned_tasks": 6, "planned_cases": 18, "cases_recorded": len(records),
                   "cases_not_run_after_stop": 18 - len(records),
                   "case_status_counts": dict(Counter(r["status"] for r in records)),
                   "canonical_status_counts": dict(Counter(r["status"] for r in canonical_records)),
                   "technical_errors": technical, "stop_reason": stopped or "complete",
                   "budget": budget.receipt(), "gpu_jobs": 0, "environment_retries": 0,
                   "interpretation": "Fixed-task teacher recovery smoke only; no student learning or unseen evaluation evidence."}
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary), flush=True)
    return 0 if summary["complete"] else 1


def self_test():
    """Pure data tests: never instantiate an environment or call reset/step."""
    import io
    states = {"facts": [["objecttype", [["o1", "object"], ["cupType", "otype"]]],
                        ["objecttype", [["o2", "object"], ["bookType", "otype"]]],
                        ["inreceptacle", [["o1", "object"], ["r1", "receptacle"]]],
                        ["inreceptacle", [["o2", "object"], ["r1", "receptacle"]]]]}
    trace = [{"state_before": states, "expert_action": "take misleading book from desk 1",
              "admissible_commands": ["look", "examine desk 1", "take misleading cup from desk 1",
                                      "take misleading book from desk 1", "go to shelf 1"]}]
    names = {"o1": "misleading book", "o2": "misleading cup", "r1": "desk 1"}
    wrong = wrong_object_candidates(trace, names, "Cup")
    assert len(wrong) == 1 and wrong[0]["object_id"] == "o2"
    target_only = [{**trace[0], "admissible_commands": ["take misleading book from desk 1"]}]
    assert wrong_object_candidates(target_only, names, "Cup") == []
    assert legal_candidates(trace) == [{"anchor": 0, "action": "go to shelf 1"},
                                       {"anchor": 0, "action": "take misleading cup from desk 1"}]
    augmented = {"facts": states["facts"] + [["checked", [["r1", "receptacle"]]]]}
    assert physical_facts(states) == physical_facts(augmented)
    changed = {"facts": states["facts"] + [["opened", [["r1", "receptacle"]]]]}
    assert physical_facts(states) != physical_facts(changed)
    b = Budget(io.StringIO(), clock=lambda: 0)
    b.started_calls["steps"] = LIMITS["steps"]
    try:
        b.begin("steps", "fake")
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("Step cap did not block")
    b.started_calls["resets"] = LIMITS["resets"]
    try:
        b.begin("resets", "fake")
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("Reset cap did not block")
    b.clock = lambda: LIMITS["wall_seconds"]
    try:
        b.check()
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("Wall cap did not block")
    return {"passed": True, "real_environment_calls": 0, "model_calls": 0,
            "checks": ["grounded_objecttype_not_display_text", "stable_candidate_order",
                       "wrong_object_not_applicable", "checked_only_excluded", "physical_change",
                       "step_reset_wall_caps", "frozen_manifest_six_tasks"],
            "plan": plan()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.execute:
        if args.output is None:
            parser.error("--execute requires --output (new directory)")
        return execute(args.output)
    if args.output is not None:
        parser.error("--output only applies to explicit --execute")
    print(json.dumps(self_test() if args.self_test else plan(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
