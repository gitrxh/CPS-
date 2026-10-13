"""Local, complete-only stored-artifact review. Never launches a model or env.

Postqueue receipt (created by a separate read-only remote check) must contain:
 kind=recovery_acquisition_postqueue_integrity, complete=true, errors=[],
 adoption_sha256, queue_status_sha256, locked_sha256_file_sha256,
 verified_file_sha256=<exact final queue lock map>,
 verified_original_file_sha256=<exact adoption originals>,
 verified_runtime_file_sha256=<exact adoption runtimes>,
 source_inventory=<exact adopted source list>,
 base_inventory_sha256, base_files_verified=28,
 initial_adapter_sha256=<exact input inventory adapter map>,
 controller={pid,start_ticks,exited:true}, owned_workers_gone=true,
 stages=[{name,pid,start_ticks,owned_session,exited:true}] in stage order.

The caller supplies its independently recorded receipt SHA256. This receipt is
remote verification evidence; this local program does not claim to rehash remote
model/runtime/game bytes itself. No records/events are opened until the complete
metadata and exited-process barrier succeeds. Output is acquisition descriptives,
never a learner-Q gate or task-success improvement conclusion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gnrl import recovery_projection_v1 as p
from gnrl.alfworld_env import TASK_TYPES, information_only_action

STAGES = ("collect", "oracle", "replay")
EXPECTED = dict(zip(STAGES, (60, 180, 180)))
LIMITS = {"collect": (6000, 60, 384000), "oracle": (144000, 1440, 0),
          "replay": (24480, 1440, 0)}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def local(root, name):
    require(isinstance(name, str), "Non-string artifact path")
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name,
            "Non-relative artifact path")
    result = (root / path).resolve()
    require(result.is_relative_to(root.resolve()), "Artifact path escapes root")
    return result


def family(game):
    value = p.relative_game(game).split("/")[1].split("-", 1)[0]
    require(value in TASK_TYPES, "Unknown training family")
    return value


def metadata_barrier(root, adoption_path, receipt_path, receipt_sha256):
    """This function deliberately opens no records.jsonl or events.jsonl."""
    root = Path(root).resolve()
    adoption = read(adoption_path)
    require(adoption.get("kind") == "recovery_acquisition_parent_adoption"
            and adoption.get("adopted") is True and adoption["authorized_stages"] == list(STAGES)
            and adoption["physical_gpu"] == 2, "Invalid parent acquisition adoption")
    paths = adoption["output_paths"]
    status_path = local(root, paths["queue"] + "/status.json")
    locks_path = local(root, paths["queue"] + "/locked_sha256.json")
    status = read(status_path)
    require(status.get("complete") is True and status.get("current_stage") == "needs_astra_acquisition_review"
            and not status.get("error") and not status.get("errors")
            and status["adoption_sha256"] == sha(adoption_path), "Queue incomplete or erroneous; outcomes remain closed")
    stages = status["stages"]
    require([s["name"] for s in stages] == list(STAGES) and all(s.get("returncode") == 0
            and s.get("owned_cleanup_complete") is True for s in stages), "Stage processes incomplete")
    summaries, costs, requests = {}, {}, {}
    for stage in STAGES:
        directory = local(root, paths[stage])
        summaries[stage] = read(directory / "summary.json")
        costs[stage] = read(directory / "cost_progress.json")
        s = summaries[stage]
        require(s.get("complete") is True and s.get("stage") == stage
                and s.get("record_count") == s.get("planned_records") == EXPECTED[stage]
                and not s.get("error") and not s.get("technical_incomplete"),
                "Stage incomplete; outcomes remain closed: " + stage)
        c = costs[stage]
        require(c.get("kind") == "recovery_cost_ledger" and c.get("active_call") is None
                and c.get("unknown_generation_after_error") is False
                and c["totals"] == s["costs"]["totals"], "Incomplete cost ledger")
        require(all(c["totals"][k + "_started"] == c["totals"][k + "_completed"]
                    for k in ("environment_steps", "resets", "policy_calls")), "Unfinished recorded call")
        require(set(c["totals"]) == set(p.COUNTERS) and all(type(v) is int and v >= 0 for v in c["totals"].values())
                and all(c["totals"][k] <= ceiling for k, ceiling in p.GLOBAL_LIMITS.items()),
                "Malformed/exceeded complete-stage cumulative ledger")
    require(sha(receipt_path) == receipt_sha256, "Postqueue receipt bytes differ")
    receipt = read(receipt_path)
    locks = read(locks_path)
    require(receipt.get("kind") == "recovery_acquisition_postqueue_integrity"
            and receipt.get("complete") is True and receipt.get("errors") == []
            and receipt["adoption_sha256"] == sha(adoption_path)
            and receipt["queue_status_sha256"] == sha(status_path)
            and receipt["locked_sha256_file_sha256"] == sha(locks_path), "Invalid postqueue integrity barrier")
    controller = receipt["controller"]
    require(type(controller.get("pid")) is int and controller["pid"] > 0
            and type(controller.get("start_ticks")) is int and controller["start_ticks"] > 0
            and controller.get("exited") is True and receipt.get("owned_workers_gone") is True,
            "Controller/owned processes have not exited")
    require(len(receipt["stages"]) == 3, "Missing exited stage identities")
    for actual, evidence in zip(stages, receipt["stages"]):
        require(all(actual[k] == evidence[k] for k in ("name", "pid", "start_ticks", "owned_session"))
                and evidence.get("exited") is True, "Exited stage identity differs")
    require(receipt["verified_file_sha256"] == locks
            and receipt["verified_original_file_sha256"] == adoption["original_file_sha256"]
            and receipt["verified_runtime_file_sha256"] == adoption["runtime_file_sha256"]
            and receipt["source_inventory"] == adoption["source_inventory"], "Postqueue hash inventory differs")
    require(all(locks.get(k) == h for k, h in adoption["file_sha256"].items()), "Adopted input locks changed")
    # Only metadata and code are checked here; output content remains unopened.
    names = [adoption["candidate"], adoption["input_inventory"], "data/manifests/train120.json",
             "configs/successful_path_selection_candidate_v1.json", "gnrl/recovery_projection_v1.py",
             "gnrl/alfworld_env.py"]
    for name in names:
        require(sha(local(root, name)) == locks[name], "Local review input differs: " + name)
    inventory = read(local(root, adoption["input_inventory"]))
    require(receipt["base_inventory_sha256"] == inventory["base_full_inventory_sha256"]
            == locks[inventory["base_full_inventory"]]
            and receipt["base_files_verified"] == inventory["base_file_count_verified"] == 28
            and receipt["initial_adapter_sha256"] == inventory["initial_adapter_sha256"]
            and all(locks.get(k) == h for k, h in inventory["initial_adapter_sha256"].items())
            and inventory["original_file_sha256"] == adoption["original_file_sha256"], "Model/game binding differs")
    for i, stage in enumerate(STAGES):
        request_name = paths["queue"] + "/" + stage + "_request.json"
        request_path = local(root, request_name)
        require(sha(request_path) == locks[request_name] == summaries[stage]["request_sha256"], "Stage request hash differs")
        request = requests[stage] = read(request_path)
        require(request["kind"] == "recovery_projection_stage_adoption" and request["schema_version"] == 1
                and request["adopted"] is True and request["physical_gpu"] == 2
                and request["stage"] == stage and request["output"] == paths[stage]
                and request["parent_adoption_sha256"] == sha(adoption_path)
                and request["candidate"] == adoption["candidate"]
                and request["candidate_sha256"] == locks[adoption["candidate"]]
                and request["source_inventory"] == adoption["source_inventory"]
                and request["original_file_sha256"] == adoption["original_file_sha256"]
                and request["global_deadline_unix"] == status["global_deadline_unix"]
                and all(locks.get(k) == h for k, h in request["file_sha256"].items()), "Derived stage binding differs")
        require(all(request["file_sha256"].get(k) == h for k, h in adoption["file_sha256"].items()),
                "Derived request dropped original binding")
        for key, suffix in (("previous_cost_json", "cost_progress.json"),
                            ("input_records", "records.jsonl"), ("input_summary", "summary.json")):
            prior = paths[STAGES[i - 1]] + "/" + suffix if i else None
            require(request[key] == prior and (prior is None or request["file_sha256"].get(prior) == locks[prior]),
                    "Stage cumulative input lineage differs")
        for filename in ("summary.json", "cost_progress.json", "config.json"):
            name = paths[stage] + "/" + filename
            require(sha(local(root, name)) == locks[name], "Stage metadata bytes differ")
        config = read(local(root, paths[stage] + "/config.json"))
        require(config["request_sha256"] == locks[request_name] and config["request"] == request,
                "Worker config request differs")
    return {"adoption": adoption, "status": status, "locks": locks, "costs": costs,
            "summaries": summaries, "receipt_sha256": receipt_sha256}


def jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            require(line.endswith("\n") and line.strip(), "Truncated/blank JSONL")
            yield json.loads(line)


def event_equal(iterator, expected):
    require(next(iterator, None) == expected, "Missing/different actual event")


def neutral(state):
    require(set(state) == {"observation", "admissible_commands", "reward", "done", "won", "canonical_state"},
            "Neutral state schema differs")
    c = state["canonical_state"]
    require(type(state["done"]) is bool and type(state["won"]) is bool
            and state["done"] is c["done"] and state["won"] is c["won"]
            and c["max_steps"] == 100 and c["admissible_commands"] == state["admissible_commands"]
            and c["reward"] == state["reward"] and c["done"] == bool(c["terminated"] or c["truncated"]),
            "Neutral/canonical state or terminal flags differ")
    require(type(c["elapsed_steps"]) is int and 0 <= c["elapsed_steps"] <= 100, "State clock differs")
    require(not state["won"] or state["done"], "Won without terminal original-goal state")


def independent_episode(row):
    p.validate_episode(row)
    previous = row["initial"]
    neutral(previous)
    generated = invalid = 0
    for i, transition in enumerate(row["transitions"]):
        neutral(transition["before"])
        neutral(transition["after"])
        details = transition["policy"]
        require(transition["before"] == previous and transition["step"] == i
                and details["raw_completion"].strip() == transition["action"]
                and details["is_admissible"] is (transition["action"] in previous["admissible_commands"]),
                "Actual raw-generation chain differs")
        n = details["generated_tokens"]
        require(type(n) is int and 0 <= n <= 64, "Generation token evidence differs")
        generated += n
        invalid += not details["is_admissible"]
        previous = transition["after"]
    require(row["success"] is previous["won"], "Collection success differs from actual final state")
    for anchor in row["anchors"]:
        k = anchor["after_actions"]
        if k > len(row["transitions"]):
            expected_status = "terminated_before_anchor" if previous["done"] else "no_admissible_before_anchor"
        else:
            at = row["transitions"][k - 1]["after"]
            expected_status = "terminated_at_anchor" if at["done"] else "no_admissible_at_anchor" if not at["admissible_commands"] else "available"
        require(anchor["status"] == expected_status, "Structural status not supported by whole rollout")
        if anchor["status"] != "available":
            continue
        prefix = row["transitions"][:anchor["after_actions"]]
        expected = {"task": row["initial"]["observation"],
            "history": [{"observation": t["before"]["observation"], "action": t["action"]} for t in prefix],
            "observation": prefix[-1]["after"]["observation"],
            "admissible_commands": prefix[-1]["after"]["admissible_commands"]}
        require(anchor["student_context"] == expected, "Student context includes noncausal/future data")
    return generated, invalid


def independent_proposals(anchor, expert):
    commands = anchor["anchor_state"]["admissible_commands"]
    first = [] if expert is None else [expert]
    require(expert is None or expert in commands, "Planner action outside admissibles")
    learner = anchor["recorded_learner_next_action"]
    if learner in commands and learner not in first and not information_only_action(learner):
        first.append(learner)
    salt = "recovery_projection_v1_action_pool|" + p.relative_game(anchor["gamefile"]) + "|" + str(anchor["after_actions"]) + "|"
    rest = sorted({a for a in commands if a not in first and not information_only_action(a)},
                  key=lambda a: (hashlib.sha256((salt + a).encode()).hexdigest(), a.encode()))
    return (first + rest)[:8]


def review_rows(games, planned, collect, pools, replays, events):
    anchors, families, diagnostic, counts = {}, {}, [], {s: Counter() for s in STAGES}
    for f in sorted(TASK_TYPES):
        families[f] = Counter()
    collection = iter(collect)
    for game in games:
        row = next(collection, None)
        require(row is not None and row["gamefile"] == game, "Collection task count/order differs")
        generated, invalid = independent_episode(row)
        stat = families[family(game)]
        stat.update(tasks=1, learner_episodes_won=int(row["success"]), invalid_learner_actions=invalid)
        event_equal(events["collect"], {"kind": "collection_initial", "gamefile": game, "state": row["initial"]})
        for transition in row["transitions"]:
            event_equal(events["collect"], {"kind": "collection_transition", "gamefile": game, "transition": transition})
        counts["collect"].update(resets=1, steps=len(row["transitions"]), policy_calls=len(row["transitions"]), tokens=generated)
        for anchor in row["anchors"]:
            anchors[anchor["anchor_id"]] = anchor
    require(next(collection, None) is None, "Extra collection task")
    pools, replays = iter(pools), iter(replays)
    pool_summaries = {}
    for game in games:
        for k in (4, 8, 16):
            pool, replay = next(pools, None), next(replays, None)
            identity = p.digest([game, p.SEED, k])
            require(pool is not None and replay is not None and pool["anchor_id"] == identity
                    and pool["anchor"] == anchors[identity], "Pool missing/order/prefix differs from collection")
            p.validate_pool(pool)
            anchor, stat = pool["anchor"], families[family(game)]
            stat.update(slots=1, available_anchors=int(anchor["status"] == "available"), eligible_anchors=int(pool["eligible"]),
                        proposed_branches=len(pool["branches"]), certified_actions=len(pool["retained_candidates"]),
                        eligible_multiaction_anchors=int(pool["eligible"] and len(pool["retained_candidates"]) >= 2))
            stat["status:" + pool["status"]] += 1
            if anchor["status"] == "available":
                require(pool["proposed_actions"] == independent_proposals(anchor, pool["expert_action"]), "Proposal changed")
                resets = max(1, len(pool["branches"]))
                counts["oracle"].update(resets=resets, prefix_steps=k * resets)
            for branch in pool["branches"]:
                previous = anchor["anchor_state"]
                for transition in branch["transitions"]:
                    neutral(transition["before"])
                    neutral(transition["after"])
                    require(transition["before"] == previous and not previous["done"]
                            and transition["action"] in previous["admissible_commands"], "Broken actual oracle branch")
                    previous = transition["after"]
                    event_equal(events["oracle"], {"kind": "oracle_transition", "anchor_id": identity,
                        "candidate_index": branch["candidate_index"], "transition": transition})
                require(branch["success"] is bool(previous["done"] and previous["won"])
                        and previous["canonical_state"]["elapsed_steps"] <= 100, "Unverified goal-success certificate")
                event_equal(events["oracle"], {"kind": "oracle_branch_complete", "anchor_id": identity, "branch": branch})
                counts["oracle"]["branch_steps"] += len(branch["transitions"])
            actual_certified = [b["action"] for b in pool["branches"] if
                                b["transitions"][-1]["after"]["done"] and b["transitions"][-1]["after"]["won"]]
            require(actual_certified == [c["action"] for c in pool["retained_candidates"]]
                    and pool["eligible"] is (pool["expert_action"] is not None and pool["expert_action"] in actual_certified),
                    "Certificate retention/eligibility differs from actual original-goal success")
            expected_replay = {"schema_version": 1, "kind": "recovery_candidate_replay", "complete": True,
                "anchor_id": identity, "pool_sha256": pool["artifact_sha256"], "candidates": []}
            for candidate in pool["retained_candidates"]:
                checked = {"action": candidate["action"], "certificate_sha256": candidate["certificate_sha256"],
                           "replay_verified": True, "actual_transition": candidate["first_transition"]}
                expected_replay["candidates"].append(checked)
                event_equal(events["replay"], {"kind": "candidate_replay_verified", "anchor_id": identity, **checked})
                counts["replay"].update(resets=1, prefix_steps=k, branch_steps=1)
            require(replay == expected_replay, "Retained first-action replay differs or missing")
            pool_summaries[(game, k)] = {"status": pool["status"], "eligible": pool["eligible"],
                                       "certified_actions": len(pool["retained_candidates"])}
    require(next(pools, None) is None and next(replays, None) is None, "Extra pool/replay slot")
    require(all(next(e, None) is None for e in events.values()), "Extra/unmatched environment event")
    for anchor in planned:
        require(family(anchor["gamefile"]) == anchor["family"], "Diagnostic family differs")
        diagnostic.append({**anchor, **pool_summaries[(anchor["gamefile"], anchor["after_actions"])]})
    return {"families": {f: dict(v) for f, v in families.items()}, "diagnostic36_availability": diagnostic,
            "recomputed_workload": {s: dict(c) for s, c in counts.items()}}


def review_costs(costs, workload):
    previous, result = dict.fromkeys(p.COUNTERS, 0), {}
    for stage in STAGES:
        cost, work = costs[stage], Counter(workload[stage])
        steps = work["steps"] if stage == "collect" else work["prefix_steps"] + work["branch_steps"]
        expected = dict.fromkeys(p.COUNTERS, 0)
        for key, value in (("environment_steps", steps), ("resets", work["resets"]), ("policy_calls", work["policy_calls"])):
            expected[key + "_started"] = expected[key + "_completed"] = value
        expected["generated_tokens"] = work["tokens"]
        # Generation records do not store input token IDs/counts. Do not invent a
        # forward-token recount from raw strings: retain the durable ledger value.
        expected["forward_tokens"] = cost["stage_counts"]["forward_tokens"] if stage == "collect" else 0
        require(all(type(v) is int and v >= 0 for v in expected.values()), "Invalid durable counters")
        require(cost["stage_counts"] == expected and cost["totals"] == {k: previous[k] + expected[k] for k in p.COUNTERS},
                "Stage counts or cumulative cost lineage differs")
        limit = dict(zip(p.GLOBAL_LIMITS, LIMITS[stage]))
        require(cost["stage_limits"] == limit and cost["global_limits"] == p.GLOBAL_LIMITS
                and all(expected[k] <= v for k, v in limit.items())
                and all(cost["totals"][k] <= v for k, v in p.GLOBAL_LIMITS.items()), "Budget ceiling differs/exceeded")
        labels = ({"collection:reset": work["resets"], "collection:step": steps,
                   "learner_collection:generate": work["policy_calls"]} if stage == "collect" else
                  {"oracle_prefix:reset": work["resets"], "oracle_prefix:step": work["prefix_steps"],
                   "oracle_branch:step": work["branch_steps"]} if stage == "oracle" else
                  {"candidate_replay_prefix:reset": work["resets"], "candidate_replay_prefix:step": work["prefix_steps"],
                   "candidate_replay_action:step": work["branch_steps"]})
        require(cost["calls_by_label"] == {k: v for k, v in labels.items() if v}, "Per-operation cost labels differ")
        result[stage] = expected
        previous = cost["totals"]
    return {"stage_counts": result, "cumulative_totals": previous,
            "event_independent": ["collection reset completions and actual transitions", "oracle branch transitions", "retained candidate first-action replay completions", "collection generated-token counts from per-action policy records"],
            "ledger_dependent": ["Oracle/replay resets and prefix steps have no individual events: expected counts are reconstructed from complete branch/candidate contracts and checked against durable started/completed counters.",
                                 "The no-proposal oracle anchor still incurs one reset and its prefix replay; this is reconstructed from the frozen implementation and checked against its ledger.",
                                 "Generation forward_tokens cannot be independently recomputed without tokenizer/input-token evidence; use the durable policy ledger and do not infer token counts from strings.",
                                 "Started-call durability and absence of hidden simulator calls rely on the reviewed frozen worker, complete equal started/completed counters, and parent process/integrity receipt."]}


def run(root, adoption_path, receipt_path, receipt_sha256):
    root = Path(root).resolve()
    barrier = metadata_barrier(root, adoption_path, receipt_path, receipt_sha256)
    adoption, locks = barrier["adoption"], barrier["locks"]
    candidate = read(local(root, adoption["candidate"]))
    games = candidate["training_cohort"]["frozen_gamefiles"]
    train = read(local(root, "data/manifests/train120.json"))["games"]
    old = read(local(root, "configs/successful_path_selection_candidate_v1.json"))["fixed_training_cohort"]["frozen_gamefiles"]
    require(games == [g for g in train if g not in set(old)] and len(games) == len(set(games)) == 60
            and Counter(map(family, games)) == Counter({f: 10 for f in TASK_TYPES}), "Frozen complementary60 differs")
    planned = candidate["train_only_diagnostic_gate"]["frozen_anchors"]
    require(len(planned) == len({(a["gamefile"], a["after_actions"]) for a in planned}) == 36,
            "Frozen diagnostic36 differs")
    independently_fixed = []
    for f in sorted(TASK_TYPES):
        ranked = sorted((g for g in games if family(g) == f), key=lambda g: hashlib.sha256(
            ("recovery_projection_v1_diagnostic36|" + p.relative_game(g)).encode()).hexdigest())
        independently_fixed.extend({"gamefile": g, "family": f, "after_actions": (4, 8, 16)[i // 2]}
                                   for i, g in enumerate(ranked[:6]))
    require(planned == independently_fixed, "Diagnostic36 no longer equals preregistered identity-only draw")
    directories = {s: local(root, adoption["output_paths"][s]) for s in STAGES}
    # Barrier succeeded: it is now permissible to hash and parse outcome files.
    for stage, directory in directories.items():
        for name in ("records.jsonl", "events.jsonl"):
            relative = adoption["output_paths"][stage] + "/" + name
            require(sha(directory / name) == locks[relative], "Locked outcome bytes differ")
        require(locks[adoption["output_paths"][stage] + "/records.jsonl"] == barrier["summaries"][stage]["records_sha256"],
                "Outcome summary hash differs")
    result = review_rows(games, planned, *(jsonl(directories[s] / "records.jsonl") for s in STAGES),
                         {s: jsonl(directories[s] / "events.jsonl") for s in STAGES})
    result["cost_review"] = review_costs(barrier["costs"], result["recomputed_workload"])
    require(barrier["status"]["cumulative_costs"] == result["cost_review"]["cumulative_totals"], "Queue final costs differ")
    # A local file may not change between its pre-read hash and final decision.
    require(sha(receipt_path) == receipt_sha256, "Postqueue receipt changed during review")
    for stage, directory in directories.items():
        for name in ("records.jsonl", "events.jsonl", "summary.json", "cost_progress.json", "config.json"):
            require(sha(directory / name) == locks[adoption["output_paths"][stage] + "/" + name],
                    "Stage artifact changed during review")
    return {"kind": "recovery_acquisition_complete_result_review", "schema_version": 1,
        "recovery_acquisition_result_review_passed": True, "complete": True, "errors": [],
        "adoption_sha256": sha(adoption_path), "postqueue_receipt_sha256": receipt_sha256,
        "acquisition_status_sha256": sha(local(root, adoption["output_paths"]["queue"] + "/status.json")),
        "acquisition_locks_sha256": sha(local(root, adoption["output_paths"]["queue"] + "/locked_sha256.json")),
        "replay_cost_sha256": locks[adoption["output_paths"]["replay"] + "/cost_progress.json"],
        "candidate_sha256": locks[adoption["candidate"]], "review_program_sha256": sha(__file__),
        "complete_record_counts": EXPECTED, **result,
        "training_gate_passed": None, "execution_authorized": False,
        "interpretation": "Acquisition integrity and descriptive finite-pool availability only. Oracle certificates are not actual learner Q, and coverage is not selection utility or held-out task-success improvement.",
        "remote_verification_scope": "Remote source/model/adapter/game/runtime byte checks and exited-process identities rely on the supplied hash-bound postqueue receipt; local output and review-input bytes are checked directly."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption-json", required=True, type=Path)
    parser.add_argument("--postqueue-receipt", required=True, type=Path)
    parser.add_argument("--postqueue-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Review output already exists")
    try:
        result = run(ROOT, args.adoption_json, args.postqueue_receipt, args.postqueue_sha256)
    except Exception as error:
        result = {"kind": "recovery_acquisition_complete_result_review", "schema_version": 1,
            "recovery_acquisition_result_review_passed": False, "complete": False,
            "errors": [f"{type(error).__name__}: {error}"], "training_gate_passed": None,
            "execution_authorized": False}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("recovery_acquisition_result_review_passed", "complete", "errors")}))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
