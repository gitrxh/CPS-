"""Complete-only score/actual-Q artifact review; no model or environment calls.

Run locally or from project/analysis on the remote host after all jobs exit.
Receipt schema matches the acquisition postqueue receipt, except kind is
recovery_diagnostic_postqueue_integrity and stages are score,diagnostic. Every
verified mapping must describe the final diagnostic queue locks/adoption. The
caller independently supplies the receipt SHA256. No outcome file is parsed
until both stage summaries, final queue, locks and exited-process receipt pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import review_recovery_acquisition_complete_v1 as a
from gnrl import recovery_projection_v1 as p
from gnrl import recovery_diagnostic_v1 as d

STAGES = ("score", "diagnostic")
ACQUISITION_REVIEW_SHA256 = "42fe7a2f7c840d61514b58926edc935bfc2518dc10fa6eadbbe0ea55dc2390cd"
require, sha, read, local, jsonl = a.require, a.sha, a.read, a.local, a.jsonl


def metadata_barrier(root, adoption_path, receipt_path, receipt_hash):
    adoption = read(adoption_path)
    require(adoption.get("kind") == "recovery_diagnostic_parent_adoption" and adoption.get("adopted") is True
            and adoption["authorized_stages"] == list(STAGES) and adoption["physical_gpu"] == 2, "Wrong diagnostic parent adoption")
    paths = adoption["output_paths"]
    status_path = local(root, paths["queue"] + "/status.json")
    locks_path = local(root, paths["queue"] + "/locked_sha256.json")
    status = read(status_path)
    require(status.get("complete") is True and status.get("current_stage") == "needs_astra_diagnostic_review"
            and status.get("gate_evaluable") is True and type(status.get("training_gate_passed")) is bool
            and not status.get("error") and not status.get("errors")
            and status["adoption_sha256"] == sha(adoption_path), "Diagnostic queue incomplete; outcome streams remain closed")
    require([s["name"] for s in status["stages"]] == list(STAGES)
            and all(s.get("returncode") == 0 and s.get("owned_cleanup_complete") is True for s in status["stages"]),
            "Diagnostic workers incomplete")
    summaries, costs, requests = {}, {}, {}
    for stage in STAGES:
        folder = local(root, paths[stage])
        s = summaries[stage] = read(folder / "summary.json")
        c = costs[stage] = read(folder / "cost_progress.json")
        require(s.get("complete") is True and s.get("stage") == stage
                and not s.get("technical_incomplete") and not s.get("error"), "Incomplete stage; outcome streams remain closed")
        require(c.get("active_call") is None and c.get("unknown_generation_after_error") is False
                and c["totals"] == s["costs"]["totals"] and c["stage_counts"] == s["costs"]["stage_counts"]
                and set(c["totals"]) == set(p.COUNTERS)
                and all(type(v) is int and v >= 0 for v in c["totals"].values()), "Incomplete shared cost evidence")
        require(all(c["totals"][k + "_started"] == c["totals"][k + "_completed"]
                    for k in ("environment_steps", "resets", "policy_calls"))
                and all(c["totals"][k] <= v for k, v in p.GLOBAL_LIMITS.items()), "Unfinished/excess recorded work")
    require(summaries["score"].get("record_count") == 180 and summaries["score"].get("labels_locked_before_Q") is True
            and summaries["diagnostic"].get("gate_evaluable") is True
            and type(summaries["diagnostic"].get("training_gate_passed")) is bool
            and summaries["diagnostic"].get("outcome_unblinded") is True, "Incomplete score/Q barrier")
    require(sha(receipt_path) == receipt_hash, "Postqueue receipt bytes differ")
    receipt, locks = read(receipt_path), read(locks_path)
    require(receipt.get("kind") == "recovery_diagnostic_postqueue_integrity" and receipt.get("complete") is True
            and receipt.get("errors") == [] and receipt["adoption_sha256"] == sha(adoption_path)
            and receipt["queue_status_sha256"] == sha(status_path)
            and receipt["locked_sha256_file_sha256"] == sha(locks_path), "Wrong postqueue receipt")
    require(receipt["controller"].get("exited") is True and receipt.get("owned_workers_gone") is True
            and type(receipt["controller"].get("pid")) is int and receipt["controller"]["pid"] > 0
            and type(receipt["controller"].get("start_ticks")) is int and receipt["controller"]["start_ticks"] > 0,
            "Diagnostic controller/owned workers have not exited")
    require(len(receipt["stages"]) == 2, "Missing exited-stage identities")
    for recorded, exited in zip(status["stages"], receipt["stages"]):
        require(all(recorded[k] == exited[k] for k in ("name", "pid", "start_ticks", "owned_session"))
                and exited.get("exited") is True, "Wrong exited worker identity")
    require(receipt["verified_file_sha256"] == locks and receipt["source_inventory"] == adoption["source_inventory"]
            and receipt["verified_original_file_sha256"] == adoption["original_file_sha256"]
            and receipt["verified_runtime_file_sha256"] == adoption["runtime_file_sha256"]
            and all(locks.get(k) == v for k, v in adoption["file_sha256"].items()), "Final frozen inventories differ")
    metadata = [adoption["candidate"], adoption["input_inventory"], adoption["acquisition_result_review"],
        adoption["acquisition_status"], adoption["acquisition_locks"], adoption["acquisition_parent_adoption"], "gnrl/recovery_projection_v1.py",
        "gnrl/recovery_diagnostic_v1.py", "gnrl/alfworld_env.py"]
    for name in metadata:
        require(sha(local(root, name)) == locks[name], "Changed local review input: " + name)
    inventory = read(local(root, adoption["input_inventory"]))
    require(receipt["base_inventory_sha256"] == inventory["base_full_inventory_sha256"] == locks[inventory["base_full_inventory"]]
            and receipt["base_files_verified"] == 28 and receipt["initial_adapter_sha256"] == inventory["initial_adapter_sha256"]
            and all(locks.get(k) == v for k, v in inventory["initial_adapter_sha256"].items()), "Base/adapter evidence differs")
    acquisition_review = read(local(root, adoption["acquisition_result_review"]))
    require(acquisition_review.get("recovery_acquisition_result_review_passed") is True
            and acquisition_review["acquisition_status_sha256"] == locks[adoption["acquisition_status"]]
            and acquisition_review["acquisition_locks_sha256"] == locks[adoption["acquisition_locks"]]
            and acquisition_review["candidate_sha256"] == locks[adoption["candidate"]], "Missing bound acquisition result review")
    old_status = read(local(root, adoption["acquisition_status"]))
    prior_adoption = read(local(root, adoption["acquisition_parent_adoption"]))
    prior_locks = read(local(root, adoption["acquisition_locks"]))
    require(all(locks.get(k) == h for k, h in prior_locks.items())
            and prior_adoption["candidate"] == adoption["candidate"]
            and old_status["adoption_sha256"] == locks[adoption["acquisition_parent_adoption"]],
            "Prior acquisition input/output locks not preserved")
    require(status["global_deadline_unix"] == old_status["global_deadline_unix"]
            and status["main_study_started_unix"] == old_status["main_study_started_unix"], "Study deadline reset")
    for i, stage in enumerate(STAGES):
        request_name = paths["queue"] + "/" + stage + "_request.json"
        require(sha(local(root, request_name)) == locks[request_name] == summaries[stage]["request_sha256"], "Request hash differs")
        request = requests[stage] = read(local(root, request_name))
        require(request["kind"] == "recovery_diagnostic_stage_adoption" and request["adopted"] is True
                and request["physical_gpu"] == 2 and request["stage"] == stage
                and request["parent_adoption_sha256"] == sha(adoption_path) and request["output"] == paths[stage]
                and request["candidate"] == adoption["candidate"] and request["candidate_sha256"] == locks[adoption["candidate"]]
                and request["global_deadline_unix"] == status["global_deadline_unix"]
                and all(locks.get(k) == v for k, v in request["file_sha256"].items())
                and all(request["file_sha256"].get(k) == v for k, v in adoption["file_sha256"].items()), "Stage adoption lineage differs")
        for filename in ("summary.json", "cost_progress.json", "config.json"):
            name = paths[stage] + "/" + filename
            require(sha(local(root, name)) == locks[name], "Stage metadata changed")
        config = read(local(root, paths[stage] + "/config.json"))
        require(config["request"] == request and config["request_sha256"] == locks[request_name], "Worker config differs")
    for key in ("pool_records", "pool_summary", "replay_records", "replay_summary"):
        require(requests["diagnostic"][key] == requests["score"][key], "Score/Q acquisition inputs differ")
        phase = "oracle" if key.startswith("pool") else "replay"
        filename = "records.jsonl" if key.endswith("records") else "summary.json"
        require(requests["score"][key] == prior_adoption["output_paths"][phase] + "/" + filename,
                "Score/Q does not reference reviewed acquisition outputs")
    require(requests["score"]["previous_summary"] == requests["score"]["replay_summary"]
            and requests["score"]["previous_cost_json"] == str(Path(requests["score"]["replay_summary"]).with_name("cost_progress.json")).replace("\\", "/")
            and locks[requests["score"]["previous_cost_json"]] == acquisition_review["replay_cost_sha256"], "Score initial cost lineage differs")
    for key, suffix in (("previous_cost_json", "cost_progress.json"), ("previous_summary", "summary.json"),
                        ("selection_records", "records.jsonl"), ("selection_summary", "summary.json")):
        expected = paths["score"] + "/" + suffix
        require(requests["diagnostic"][key] == expected
                and requests["diagnostic"]["file_sha256"].get(expected) == locks[expected], "Q request lacks frozen complete scores")
    return {"adoption": adoption, "status": status, "locks": locks, "costs": costs,
            "summaries": summaries, "requests": requests, "status_sha256": sha(status_path), "locks_sha256": sha(locks_path)}


def validate_scores(games, planned, pools, selections, replays, score_events, label_rows):
    planned_keys = {(x["gamefile"], x["after_actions"]) for x in planned}
    diagnostic_pools, selected_all = {}, []
    expected_labels = {arm: [] for arm in d.ARMS}
    difference = dict.fromkeys(d.ARMS[:2], 0)
    pools, selections, replays = iter(pools), iter(selections), iter(replays)
    score_work = Counter()
    for game in games:
        for k in (4, 8, 16):
            pool, selected, replay = next(pools, None), next(selections, None), next(replays, None)
            require(pool is not None and selected is not None and replay is not None
                    and pool["anchor"]["gamefile"] == game and pool["anchor"]["after_actions"] == k,
                    "Incomplete/misordered180 pool/selection/replay slots")
            d.validate_selection(pool, selected)
            expected_replay = [{"action": c["action"], "certificate_sha256": c["certificate_sha256"],
                "replay_verified": True, "actual_transition": c["first_transition"]} for c in pool["retained_candidates"]]
            require(replay.get("complete") is True and replay["anchor_id"] == pool["anchor_id"]
                    and replay["pool_sha256"] == pool["artifact_sha256"] and replay["candidates"] == expected_replay,
                    "Missing/different acquired first-action replay")
            selected_all.append(selected)
            if (game, k) in planned_keys:
                diagnostic_pools[(game, k)] = pool
            if selected["eligible"]:
                for arm, label in d.label_rows(pool, selected).items():
                    expected_labels[arm].append(label)
                for arm in d.ARMS[:2]:
                    difference[arm] += selected["labels"][arm] != selected["labels"][d.ARMS[2]]
    require(all(next(it, None) is None for it in (pools, selections, replays)), "Extra acquisition/selection slot")
    for role in ("reference", "base"):
        for selected in selected_all:
            if not selected["eligible"]:
                continue
            actions, scores = selected["actions"], selected[role + "_scores"]
            peak = max(s["logprob"] for s in scores.values())
            weights = {x: math.exp(scores[x]["logprob"] - peak) for x in actions}
            total = math.fsum(weights.values())
            require(selected["q" if role == "reference" else "q_base"] == {x: weights[x] / total for x in actions},
                    "Conditional full-command probabilities differ")
            for action in actions:
                a.event_equal(score_events, {"kind": "recovery_action_score", "anchor_id": selected["anchor_id"],
                    "pool_sha256": selected["pool_sha256"], "role": role, "action": action, **scores[action]})
            for start in range(0, len(actions), 2):
                batch = [scores[x] for x in actions[start:start + 2]]
                require(len({x["prompt_tokens"] for x in batch}) == 1, "Padded score batch prompt differs")
                score_work.update(calls=1, target_tokens=sum(x["action_tokens_including_eos"] for x in batch),
                    forward_tokens=len(batch) * (batch[0]["prompt_tokens"] + max(x["action_tokens_including_eos"] for x in batch)))
    require(next(score_events, None) is None, "Extra score event")
    for arm in d.ARMS:
        require(list(label_rows[arm]) == expected_labels[arm], "Three frozen label files differ from complete selection")
    return diagnostic_pools, {s["anchor_id"]: s for s in selected_all}, difference, score_work


def validate_true_q(planned, pools, rows, events):
    rows, values, workload = iter(rows), {}, Counter()
    for anchor in planned:
        pool = pools[(anchor["gamefile"], anchor["after_actions"])]
        if not pool["eligible"]:
            require(pool["status"] in d.UNAVAILABLE, "Technical missing cannot become structural zero")
            continue
        by_action = {}
        for action in d.pool_actions(pool):
            candidate = next(c for c in pool["retained_candidates"] if c["action"] == action)
            row = next(rows, None)
            require(row is not None, "Missing actual learner branch")
            d.validate_branch(pool, candidate, row)
            history = list(anchor_row for anchor_row in pool["anchor"]["prefix_transitions"])
            for i, transition in enumerate(row["transitions"]):
                a.neutral(transition["before"])
                a.neutral(transition["after"])
                if i:
                    ctx = {"task": pool["anchor"]["initial"]["observation"],
                        "history": [{"observation": t["before"]["observation"], "action": t["action"]} for t in history],
                        "observation": transition["before"]["observation"],
                        "admissible_commands": transition["before"]["admissible_commands"]}
                    require(transition["context_sha256"] == p.digest(ctx)
                            and transition["policy"]["raw_completion"].strip() == transition["action"], "Noncausal actual continuation")
                history.append(transition)
                a.event_equal(events, {"kind": "diagnostic_transition", "anchor_id": pool["anchor_id"],
                    "candidate_action": action, "transition": transition})
            final = row["transitions"][-1]["after"]
            require(row["Q"] is bool(final["won"] and final["done"]), "Actual Q differs from original-goal completion")
            by_action[action] = int(final["won"])
            workload.update(resets=1, prefix_steps=anchor["after_actions"], forced_steps=1,
                learner_steps=len(row["transitions"]) - 1,
                generated_tokens=sum(t.get("policy", {}).get("generated_tokens", 0) for t in row["transitions"]))
        values[pool["anchor_id"]] = by_action
    require(next(rows, None) is None and next(events, None) is None, "Extra actual-Q branch/event")
    return values, workload


def independent_gate(planned, pools, selected, values, label_differences):
    require(len(planned) == len({(x["gamefile"], x["after_actions"]) for x in planned}) == 36,
            "Diagnostic denominator must be36")
    require(set(label_differences) == set(d.ARMS[:2]) and all(type(v) is int and 0 <= v <= 180 for v in label_differences.values()),
            "Incomplete full-training-label difference counts")
    cells, families, multi, varied = [], set(), 0, 0
    branches = 0
    for anchor in planned:
        pool = pools[(anchor["gamefile"], anchor["after_actions"])]
        selection = selected[pool["anchor_id"]]
        cell = {**anchor, "anchor_id": pool["anchor_id"], "status": pool["status"], "Q": None,
            "Vq": None, "Vu": None, "Ve": None, "Vbase": None,
            "difference_q_uniform": 0.0, "difference_q_expert": 0.0}
        if pool["eligible"]:
            q_values = values[pool["anchor_id"]]
            require(set(q_values) == set(selection["actions"]), "Partial actual Q cannot form gate")
            # Follow certificate order only for presentation; math.fsum is used.
            observed = {c["action"]: q_values[c["action"]] for c in pool["retained_candidates"]}
            branches += len(observed)
            vq = math.fsum(selection["q"][x] * observed[x] for x in observed)
            vu = sum(observed.values()) / len(observed)
            ve = observed[pool["expert_action"]]
            cell.update(Q=observed, Vq=vq, Vu=vu, Ve=ve,
                Vbase=math.fsum(selection["q_base"][x] * observed[x] for x in observed),
                difference_q_uniform=vq-vu, difference_q_expert=vq-ve)
            if len(observed) >= 2:
                multi += 1
                families.add(anchor["family"])
            varied += len(set(observed.values())) == 2
        else:
            require(pool["status"] in d.UNAVAILABLE and pool["anchor_id"] not in values,
                    "Technical/extra Q cannot be structural zero")
        cells.append(cell)
    du = math.fsum(c["difference_q_uniform"] for c in cells) / 36
    de = math.fsum(c["difference_q_expert"] for c in cells) / 36
    gates = {"multiaction_coverage": multi >= 12 and len(families) >= 4,
        "actual_Q_variation": varied >= 8, "q_over_uniform": du >= 2/36,
        "q_over_expert": de >= 0, "label_nondegeneracy": all(v >= 12 for v in label_differences.values())}
    return {"kind": "recovery_diagnostic_summary", "complete": True, "gate_evaluable": True,
        "training_gate_passed": all(gates.values()), "gate_components": gates, "denominator": 36, "cells": cells,
        "multiaction_anchors": multi, "multiaction_families": sorted(families), "heterogeneous_actual_Q_anchors": varied,
        "label_differences": label_differences, "mean_q_minus_uniform": du, "mean_q_minus_expert": de,
        "actual_learner_branches": branches, "unavailable_Q_imputed": False}


def validate_costs(previous, costs, score_ledger, score_work, q_work):
    require(score_ledger["kind"] == "recovery_score_ledger" and score_ledger["active_call"] is None
            and score_ledger["target_limit"] == 500000 and score_ledger["forward_limit"] == 7000000
            and score_ledger["calls_started"] == score_ledger["calls_completed"] == score_work["calls"]
            and score_ledger["scored_target_tokens"] == score_work["target_tokens"] <= 500000
            and score_ledger["forward_tokens"] == score_work["forward_tokens"] <= 7000000, "Scoring ledger/caps differ")
    for stage in STAGES:
        expected = dict.fromkeys(p.COUNTERS, 0)
        if stage == "score":
            expected["forward_tokens"] = score_work["forward_tokens"]
            limits = dict.fromkeys(p.GLOBAL_LIMITS, 0)
            labels = {}
        else:
            steps = q_work["prefix_steps"] + q_work["forced_steps"] + q_work["learner_steps"]
            for prefix, n in (("resets", q_work["resets"]), ("environment_steps", steps), ("policy_calls", q_work["learner_steps"])):
                expected[prefix + "_started"] = expected[prefix + "_completed"] = n
            expected["generated_tokens"] = q_work["generated_tokens"]
            expected["forward_tokens"] = costs[stage]["stage_counts"]["forward_tokens"]
            limits = dict(zip(p.GLOBAL_LIMITS, (28800, 288, 1843200)))
            labels = {"diagnostic_prefix:reset": q_work["resets"], "diagnostic_prefix:step": q_work["prefix_steps"],
                "diagnostic_forced_action:step": q_work["forced_steps"], "diagnostic_learner_action:step": q_work["learner_steps"],
                "diagnostic_continuation:generate": q_work["learner_steps"]}
        cost = costs[stage]
        require(cost["stage_counts"] == expected and cost["totals"] == {k: previous[k] + expected[k] for k in p.COUNTERS}
                and cost["stage_limits"] == limits and cost["global_limits"] == p.GLOBAL_LIMITS
                and cost["calls_by_label"] == {k: v for k, v in labels.items() if v}
                and all(type(n) is int and n >= 0 for n in expected.values())
                and all(expected[k] <= v for k, v in limits.items())
                and all(cost["totals"][k] <= v for k, v in p.GLOBAL_LIMITS.items()), "Score/Q shared cost lineage differs")
        previous = cost["totals"]
    return {"cumulative_totals": previous, "score_event_recomputed": dict(score_work), "actual_Q_workload": dict(q_work),
        "limitations": ["Prefix steps/resets are reconstructed from each complete verified-prefix record and matched to durable starts/completions; they lack individual event rows.",
                        "Diagnostic generation forward_tokens lack prompt token IDs/counts in records and remain durable-ledger evidence, not an independently tokenized recount.",
                        "Scoring target and padded forward work is independently recomputed from all reference/base events. These counts do not imply utility."]}


def run(root, adoption_path, receipt_path, receipt_hash):
    root = Path(root).resolve()
    require(sha(a.__file__) == ACQUISITION_REVIEW_SHA256, "Review helper source changed")
    barrier = metadata_barrier(root, adoption_path, receipt_path, receipt_hash)
    adoption, locks, requests = barrier["adoption"], barrier["locks"], barrier["requests"]
    paths = adoption["output_paths"]
    candidate = read(local(root, adoption["candidate"]))
    games = candidate["training_cohort"]["frozen_gamefiles"]
    planned = candidate["train_only_diagnostic_gate"]["frozen_anchors"]
    require(len(games) == len(set(games)) == 60 and len(planned) == 36, "Frozen cohort/count differs")
    require(candidate["version"] == "v2" and candidate["finite_budgets"]["global_wall_seconds_including_waits_verification_cleanup"] == 171840,
            "Fixed candidate version/deadline differs")
    require(set(requests["diagnostic"]["label_files"]) == set(d.ARMS), "Missing/extra frozen arm label files")
    names = {requests["score"][key] for key in ("pool_records", "pool_summary", "replay_records", "replay_summary", "previous_cost_json")}
    for stage in STAGES:
        names.update(paths[stage] + "/" + f for f in ("records.jsonl", "events.jsonl", "summary.json", "cost_progress.json"))
    names.update(paths["score"] + "/" + arm + ".jsonl" for arm in d.ARMS)
    names.update((paths["score"] + "/score_ledger.json", paths["diagnostic"] + "/diagnostic.json"))
    for name in names:
        require(sha(local(root, name)) == locks[name], "Complete artifact bytes differ: " + name)
    for key, stage in (("pool", "oracle"), ("replay", "replay")):
        summary = read(local(root, requests["score"][key + "_summary"]))
        require(summary.get("complete") is True and summary["stage"] == stage
                and summary["records_sha256"] == locks[requests["score"][key + "_records"]], "Acquisition artifact summary differs")
    score_summary, q_summary = barrier["summaries"]["score"], barrier["summaries"]["diagnostic"]
    for stage in STAGES:
        require(barrier["summaries"][stage]["records_sha256"] == locks[paths[stage] + "/records.jsonl"], "Stage record hash differs")
    for arm in d.ARMS:
        name = paths["score"] + "/" + arm + ".jsonl"
        require(requests["diagnostic"]["label_files"][arm] == name
                and requests["diagnostic"]["file_sha256"].get(name) == locks[name] == score_summary["label_sha256"][arm],
                "Frozen before-Q label binding differs")
    pools, selected, difference, score_work = validate_scores(games, planned,
        jsonl(local(root, requests["score"]["pool_records"])), jsonl(local(root, paths["score"] + "/records.jsonl")),
        jsonl(local(root, requests["score"]["replay_records"])), jsonl(local(root, paths["score"] + "/events.jsonl")),
        {arm: jsonl(local(root, paths["score"] + "/" + arm + ".jsonl")) for arm in d.ARMS})
    require(score_summary["eligible_anchors"] == sum(s["eligible"] for s in selected.values()),
            "Eligible score-summary count differs")
    values, q_work = validate_true_q(planned, pools, jsonl(local(root, paths["diagnostic"] + "/records.jsonl")),
                                    jsonl(local(root, paths["diagnostic"] + "/events.jsonl")))
    result = independent_gate(planned, pools, selected, values, difference)
    require(result == read(local(root, paths["diagnostic"] + "/diagnostic.json"))
            and q_summary["diagnostic_sha256"] == locks[paths["diagnostic"] + "/diagnostic.json"]
            and q_summary["record_count"] == result["actual_learner_branches"]
            and q_summary["training_gate_passed"] is barrier["status"]["training_gate_passed"] is result["training_gate_passed"],
            "Independent complete gate differs from stored result")
    score_ledger = read(local(root, paths["score"] + "/score_ledger.json"))
    require(score_ledger == score_summary["scoring"], "Score summary ledger differs")
    previous = read(local(root, requests["score"]["previous_cost_json"]))["totals"]
    costs = validate_costs(previous, barrier["costs"], score_ledger, score_work, q_work)
    require(costs["cumulative_totals"] == barrier["status"]["cumulative_costs"], "Final queue cumulative costs differ")
    for name in names:
        require(sha(local(root, name)) == locks[name], "Artifact changed during review")
    require(sha(receipt_path) == receipt_hash, "Receipt changed during review")
    require(sha(local(root, paths["queue"] + "/status.json")) == barrier["status_sha256"]
            and sha(local(root, paths["queue"] + "/locked_sha256.json")) == barrier["locks_sha256"], "Queue metadata changed during review")
    return {"kind": "recovery_complete_diagnostic_result_review", "schema_version": 1,
        "recovery_diagnostic_result_review_passed": True, "complete": True, "errors": [],
        "training_gate_passed": result["training_gate_passed"], "gate_evaluable": True,
        "adoption_sha256": sha(adoption_path), "postqueue_receipt_sha256": receipt_hash,
        "diagnostic_status_sha256": barrier["status_sha256"], "diagnostic_locks_sha256": barrier["locks_sha256"],
        "diagnostic_result_sha256": locks[paths["diagnostic"] + "/diagnostic.json"],
        "diagnostic_cost_sha256": locks[paths["diagnostic"] + "/cost_progress.json"],
        "score_records_sha256": locks[paths["score"] + "/records.jsonl"], "review_program_sha256": sha(__file__),
        "review_helper_sha256": ACQUISITION_REVIEW_SHA256,
        "gate_recomputation": result, "cost_review": costs, "execution_authorized": False,
        "decision": "Eligible for separately adopted fixed training only" if result["training_gate_passed"] else "STOP candidate v2 before training and held-out evaluation",
        "interpretation": "This is a complete training-only mechanistic gate, not held-out efficacy. Oracle certificate success, score/NLL and availability are never substituted for actual learner Q.",
        "remote_verification_scope": "Source/model/adapter/game/runtime hashes and exited process identities rely on the exact postqueue receipt. Stored artifacts and loaded validator sources are hashed directly on the host running this review."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption-json", type=Path, required=True)
    parser.add_argument("--postqueue-receipt", type=Path, required=True)
    parser.add_argument("--postqueue-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Review output exists")
    try:
        result = run(ROOT, args.adoption_json, args.postqueue_receipt, args.postqueue_sha256)
    except Exception as error:
        result = {"kind": "recovery_complete_diagnostic_result_review", "schema_version": 1,
            "recovery_diagnostic_result_review_passed": False, "complete": False,
            "training_gate_passed": None, "gate_evaluable": False, "execution_authorized": False,
            "errors": [f"{type(error).__name__}: {error}"]}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("complete", "recovery_diagnostic_result_review_passed", "training_gate_passed", "errors")}))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
