"""Post-hoc descriptive audit of existing phase1 logs; stdlib CPU only."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("canonical", "uniform", "brp")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def short(t):
    return {"step_1based": t["step"] + 1, "action": t["action"],
            "admissible": t["policy"]["is_admissible"],
            "next_observation": t["next_observation"]}


def features(r):
    ts = r["transitions"]
    invalid = [t["action"] not in t["admissible_commands"] for t in ts]
    seen = set()
    repeats = []
    for i, t in enumerate(ts):
        assert i == t["step"]
        assert invalid[i] == (not t["policy"]["is_admissible"])
        assert t["action"] == t["policy"]["raw_completion"].strip()
        if i:
            assert t["observation"] == ts[i - 1]["next_observation"]
        key = (t["observation"], tuple(sorted(t["admissible_commands"])),
               t["action"], t["next_observation"])
        repeats.append(key in seen)
        seen.add(key)
    first = next((i + 1 for i, v in enumerate(invalid) if v), None)
    suffix = 0
    for v in reversed(invalid):
        if not v:
            break
        suffix += 1
    counts = Counter((t["action"], not bad) for t, bad in zip(ts, invalid))
    return {"task": Path(r["gamefile"]).parents[1].name,
            "success": r["success"], "steps": len(ts), "stop_reason": r["stop_reason"],
            "invalid": sum(invalid), "first_invalid_step_1based": first,
            "invalid_first10": sum(invalid[:10]), "invalid_steps11_50": sum(invalid[10:50]),
            "invalid_steps51_100": sum(invalid[50:100]),
            "invalid_repeated_observable_transition": sum(b and rep for b, rep in zip(invalid, repeats)),
            "valid_repeated_observable_transition": sum(not b and rep for b, rep in zip(invalid, repeats)),
            "invalid_nothing_happens": sum(b and t["next_observation"].strip() == "Nothing happens." for b, t in zip(invalid, ts)),
            "all_invalid_suffix_length": suffix,
            "unique_actions": len(set(t["action"] for t in ts)),
            "top_actions": [{"action": a, "admissible": good, "count": n} for (a, good), n in counts.most_common(3)]}


def main():
    output = ROOT / "outputs/phase1_failure_audit_v1.json"
    assert not output.exists(), "Versioned audit is write-once"
    locked = json.loads((ROOT / "outputs/phase1_queue_v2/locked_sha256.json").read_text())
    assert all(sha(ROOT / p) == h for p, h in locked.items())
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Post-hoc descriptive audit of existing dev24 single-training-seed42 online rollout logs. No model calls, simulator steps, training/design changes or new evaluation.",
              "definitions": {
                  "invalid": "Generated action absent from that transition's recorded admissible commands; checked against policy.is_admissible.",
                  "repeated_observable_transition": "Same exact (observation, sorted admissible commands, action, next_observation) occurred earlier in this episode. Not equality of hidden simulator state or full prompt/history.",
                  "step_numbering": "All reported steps are 1-based.",
                  "all_invalid_suffix": "Consecutive inadmissible transitions reaching the end of the recorded episode; action may vary.",
                  "interpretation": "These are post-hoc observables, not registered endpoints or causal diagnoses; valid does not mean task-correct and inadmissible does not uniquely imply syntactic error."},
              "checks": {"queue_locked_hashes_match": len(locked)}, "source_sha256": {}, "arms": {}}
    evidence = {}
    for arm in ARMS:
        p = ROOT / f"outputs/eval_phase1_{arm}_dev24_v1/episodes.jsonl"
        report["source_sha256"][str(p.relative_to(ROOT))] = sha(p)
        records = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        assert len(records) == 24 and len({r["gamefile"] for r in records}) == 24
        evidence[arm] = {r["gamefile"]: r for r in records}
        fs = [features(r) for r in records]
        failed = [f for f in fs if not f["success"]]
        totals = {k: sum(f[k] for f in failed) for k in
                  ("steps", "invalid", "invalid_first10", "invalid_steps11_50", "invalid_steps51_100",
                   "invalid_repeated_observable_transition", "valid_repeated_observable_transition",
                   "invalid_nothing_happens", "all_invalid_suffix_length")}
        firsts = [f["first_invalid_step_1based"] for f in failed if f["first_invalid_step_1based"]]
        totals.update(episodes=len(failed), first_action_invalid=sum(f["first_invalid_step_1based"] == 1 for f in failed),
                      first_invalid_by_step10=sum(v <= 10 for v in firsts),
                      first_invalid_step_median=statistics.median(firsts),
                      invalid_at_least80_episodes=sum(f["invalid"] >= 80 for f in failed),
                      majority_valid_episodes=sum(f["invalid"] < f["steps"] / 2 for f in failed))
        report["arms"][arm] = {"successes": 24 - len(failed), "all_episode_invalid": sum(f["invalid"] for f in fs),
                              "successful_episode_invalid": sum(f["invalid"] for f in fs if f["success"]),
                              "failure_totals": totals, "failures": failed}
    games = list(evidence["canonical"])
    assert all(list(evidence[a]) == games for a in ARMS)
    assert all(evidence[a][g]["transitions"][0][k] == evidence["canonical"][g]["transitions"][0][k]
               for a in ARMS for g in games for k in ("observation", "admissible_commands"))
    report["checks"]["initial_observations_and_commands_paired"] = True
    report["checks"]["action_flags_and_observation_continuity"] = True
    report["checks"]["raw_initial_state"] = "Not recorded by evaluator; identical raw initial states cannot be verified from these logs."
    divergences = []
    for g in games:
        rs = {a: evidence[a][g] for a in ARMS}
        if len({r["success"] for r in rs.values()}) < 2:
            continue
        ts = {a: r["transitions"] for a, r in rs.items()}
        first = next(i for i in range(min(map(len, ts.values()))) if len({ts[a][i]["action"] for a in ARMS}) > 1)
        divergences.append({"task": Path(g).parents[1].name,
                            "success": {a: r["success"] for a, r in rs.items()},
                            "first_action_divergence_step_1based": first + 1,
                            "first_actions": {a: [short(t) for t in v[:min(first + 4, 14)]] for a, v in ts.items()}})
    report["outcome_discordant_tasks"] = divergences
    # Execute only the pure prompt formatter AST: no policy/tokenizer/model import.
    source = (ROOT / "gnrl/policy.py").read_text()
    tree = ast.parse(source)
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "messages") or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SYSTEM_PROMPT" for t in n.targets))]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "prompt_allowlist_only", "exec"), namespace)
    sentinel = "PRIVILEGED_SENTINEL_DO_NOT_SHOW"
    history = [{"observation": "past_observation", "action": "past_action", "expert_action": sentinel,
                "future_observation": sentinel, "state": sentinel}]
    rendered = json.dumps(namespace["messages"]("task", history, "current_observation", ["current_command"]))
    assert sentinel not in rendered
    report["prompt_boundary"] = {
        "policy_source_sha256": sha(ROOT / "gnrl/policy.py"),
        "evaluate_source_sha256": sha(ROOT / "gnrl/evaluate.py"),
        "pure_formatter_extra_history_fields_excluded": True,
        "source_observations": [
            "Prompt formatter reads task, past observation/action pairs, current observation, current admissible commands only; explicit extra-history-field sentinel excluded.",
            "Evaluator obtains every next_observation from env.step(action) and passes no expert trajectory to rollout.",
            "Prompt encoder drops oldest whole history turns until 2048-token context budget including 64-token action reserve fits; current task/observation/commands are retained.",
            "Greedy generation is free-form, action equals stripped raw completion, with no admissibility repair/retry; all recorded flags match command-list membership."],
        "limits": "Rendered prompt/token IDs and number of retained history turns are not logged. This source/formatter audit finds no explicit future/expert input path; it does not prove arbitrary text fields lack privileged content or diagnose why a model ignores a command list."}
    report["interpretation_limits"] = [
        "Do not infer hidden reasoning, memory failure, search intention, or causal training effects from action traces.",
        "Repeated visible transitions may mask hidden-state differences; prompt histories differ even when visible transitions repeat.",
        "Task-discordant valid actions are observable (e.g. holding laptop for keychain task), but correctness of every exploratory move is not identifiable.",
        "Registered phase1 BRP > both baseline success-count gate remains false (13/16/15); this audit does not authorize unseen evaluation or change comparison settings."]
    assert all(sha(ROOT / p) == h for p, h in locked.items())
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"output": str(output), "arms": {a: {k: v for k, v in r.items() if k != "failures"} for a, r in report["arms"].items()},
                      "discordant_tasks": [{k: v for k, v in r.items() if k != "first_actions"} for r in divergences]}, indent=2))


if __name__ == "__main__":
    main()
