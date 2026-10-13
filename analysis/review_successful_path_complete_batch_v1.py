"""LOCAL complete-batch review; never imports model code, runs SSH or evaluates.

Read all three evaluation summaries and refuse incomplete/error batches BEFORE
opening any episode file. --root is a local copy of the artifact tree. Only new
local report files are written. No raw-initial-state equality claim is made.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import statistics

ARMS = ("canonical", "uniform", "reference")
TARGET_TOKENS = {"canonical": 6510, "uniform": 6512, "reference": 6515}
MODEL = "/home/jsy/zhangranxu-adrs-week-20260929/models/Qwen2.5-3B-Instruct"
PROTOCOL = "alfworld-causal-v1"
MANIFEST = "data/manifests/dev24.json"
TRAIN = {"model": MODEL, "max_steps": 100, "batch_size": 1, "gradient_accumulation": 8,
         "learning_rate": .0002, "lora_rank": 16, "lora_alpha": 32, "lora_dropout": .05,
         "seed": 42, "max_context_tokens": 2048, "max_action_tokens": 64, "max_target_tokens": 2000000}
EVAL = {"model": MODEL, "selection": "generate", "max_steps": 100, "seed": 20261012,
        "max_context_tokens": 2048, "max_action_tokens": 64, "max_generated_tokens": 100000,
        "max_scored_tokens": 10000000, "score_batch_size": 2}
DATA_FILES = {"canonical": "canonical_baseline.jsonl", "uniform": "uniform_valid.jsonl", "reference": "learner_selected.jsonl"}
FAMILIES = {"look_at_obj_in_light", "pick_and_place_simple", "pick_clean_then_place_in_recep",
            "pick_cool_then_place_in_recep", "pick_heat_then_place_in_recep", "pick_two_obj_and_place"}


class ReviewError(RuntimeError): pass


def require(condition, reason):
    if not condition: raise ReviewError(reason)


def sha(raw): return hashlib.sha256(raw).hexdigest()


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.consumed = {}; self.locks = None

    def path(self, name):
        name = Path(name); path = (name if name.is_absolute() else self.root / name).resolve()
        require(path.is_relative_to(self.root), "Artifact outside local snapshot")
        return path

    def raw(self, name, *, locked=True):
        path = self.path(name); key = path.relative_to(self.root).as_posix()
        raw = path.read_bytes(); value = sha(raw)
        if self.locks is not None and locked:
            require(self.locks.get(key) == value, "Missing lock/changed artifact: " + key)
        require(key not in self.consumed or self.consumed[key] == value, "Artifact changed during review: " + key)
        self.consumed[key] = value
        return raw

    def json(self, name, *, locked=True): return json.loads(self.raw(name, locked=locked))
    def jsonl(self, name): return [json.loads(line) for line in self.raw(name).splitlines() if line.strip()]

    def verify_consumed(self):
        for key, value in self.consumed.items():
            require(sha(self.path(key).read_bytes()) == value, "Consumed artifact changed: " + key)


def complete_summary_gate(reader, paths):
    """No episodes/configs/queue report may be opened before this barrier."""
    summaries = {}; unavailable = []
    for arm in ARMS:
        try: summaries[arm] = reader.json(paths["evaluations"][arm] + "/summary.json", locked=False)
        except (FileNotFoundError, json.JSONDecodeError): unavailable.append(arm)
    incomplete = [a for a, s in summaries.items() if not (s.get("complete") is True
        and s.get("technical_errors") == 0 and s.get("planned_episodes") == s.get("attempted_episodes") == 24)]
    require(not unavailable and not incomplete, "Three complete error-free 24-task summaries required; episode reads refused. Missing/incomplete: "
            + ", ".join(sorted(set(unavailable + incomplete))))
    return summaries


def sample_indices(size):
    require(type(size) is int and size > 0, "Invalid example count")
    rng = random.Random(42); order = list(range(size)); rng.shuffle(order)
    cursor = 0; result = []
    for _ in range(800):
        if cursor == len(order): rng.shuffle(order); cursor = 0
        result.append(order[cursor]); cursor += 1
    return result


def validate_episode(ep, game):
    require(ep.get("gamefile") == game and ep.get("seed") == 20261012 and "error" not in ep
            and type(ep.get("success")) is bool, "Episode identity/error/success schema differs")
    require(ep.get("stop_reason") in {"environment_done", "max_steps", "no_admissible_commands"}, "Technical/budget stop in episode")
    trace = ep.get("transitions")
    require(isinstance(trace, list) and 0 < len(trace) <= 100 and ep.get("environment_steps") == len(trace), "Missing/invalid transitions")
    invalid = generated = 0
    for i, t in enumerate(trace):
        require(t.get("step") == i and isinstance(t.get("observation"), str)
                and isinstance(t.get("admissible_commands"), list) and isinstance(t.get("action"), str)
                and type(t.get("won")) is bool and type(t.get("done")) is bool, "Malformed transition")
        if i: require(trace[i - 1]["next_observation"] == t["observation"] and not trace[i - 1]["done"], "Broken/post-terminal trajectory")
        p = t["policy"]; admissible = t["action"] in t["admissible_commands"]
        require(p.get("selection") == "generate" and isinstance(p.get("raw_completion"), str)
                and p["raw_completion"].strip() == t["action"] and p.get("is_admissible") is admissible,
                "Raw generation/admissibility evidence differs")
        require(type(p.get("generated_tokens")) is int and p["generated_tokens"] >= 0, "Invalid token count")
        invalid += not admissible; generated += p["generated_tokens"]
    require(ep["success"] == any(t["won"] for t in trace), "Success differs from actual won transitions")
    require(ep.get("terminal") is trace[-1]["done"] and ep.get("final_observation") == trace[-1]["next_observation"], "Terminal evidence differs")
    require((ep["stop_reason"] == "environment_done") == trace[-1]["done"], "Stop/done mismatch")
    return {"success": ep["success"], "steps": len(trace), "invalid_actions": invalid, "generated_tokens": generated,
            "stop_reason": ep["stop_reason"], "terminal": ep["terminal"]}


def review(root, adoption_path):
    r = Reader(root); adoption = r.json(adoption_path, locked=False)
    require(adoption.get("kind") == "successful_path_learning_adoption" and adoption.get("adopted") is True
            and adoption.get("scope") == "frozen60_single_seed_three_arm_exploratory_learning", "Wrong adoption")
    paths = adoption["output_paths"]
    summaries = complete_summary_gate(r, paths)  # FIRST result reads; no episodes before full barrier.
    queue = paths["queue"]; status = r.json(queue + "/status.json", locked=False)
    require(status.get("complete") is True and status.get("trained_arms") == status.get("evaluated_arms") == list(ARMS), "Queue has not completed")
    require(status.get("adoption_sha256") == sha(r.raw(adoption_path, locked=False)), "Queue adoption differs")
    locks = r.json(queue + "/locked_sha256.json", locked=False); r.locks = locks
    require(all(locks.get(k) == v for k, v in adoption["file_sha256"].items()), "Locks differ from adopted inputs")
    for arm in ARMS: r.json(paths["evaluations"][arm] + "/summary.json")
    stages = status["stages"]
    require([s["name"] for s in stages] == ["select", "token_exposure_audit"] + ["train_" + a for a in ARMS] + ["eval_" + a for a in ARMS]
            and all(s.get("returncode") == 0 and s.get("owned_cleanup_complete") is True for s in stages), "Stage order/completion differs")
    checkpoint_receipt = r.json(queue + "/all_training_complete.json")
    require(checkpoint_receipt["arms"] == list(ARMS)
            and all(locks.get(k) == v for k, v in checkpoint_receipt["checkpoint_sha256"].items()), "All-training hash receipt differs")
    manifest = r.json(MANIFEST); games = manifest["games"]
    require(manifest["split"] == "valid_seen" and len(games) == len(set(games)) == 24, "Wrong dev24 cohort")
    original = r.json(queue + "/original_game_locks.json")
    require(all(original[g] == manifest["game_sha256"][g] == adoption["original_file_sha256"][g] for g in games), "Original game hash identities differ")
    audit = r.json(queue + "/token_exposure_audit.json")
    require(audit.get("complete") is True and audit.get("score_sft_token_mismatches") == 0
            and audit.get("teacher_fields_used") is False, "Token audit incomplete")
    sampled = sample_indices(audit["examples_per_arm"])
    require(audit["sampled_indices"] == sampled and len(audit["ordered_example_keys"]) == audit["examples_per_arm"], "Training schedule differs")
    training = {}; results = {}; episodes = {}; weight_receipts = {}
    for arm in ARMS:
        checkpoint = paths["checkpoints"][arm]; evaluation = paths["evaluations"][arm]
        ts = r.json(checkpoint + "/summary.json"); tc = r.json(checkpoint + "/config.json")
        require(all(tc.get(k) == v for k, v in TRAIN.items()) and tc.get("protocol") == PROTOCOL, "Training protocol differs: " + arm)
        require(ts.get("matched_step_budget_complete") is True and ts.get("optimizer_steps") == ts.get("requested_steps") == 100
                and ts.get("stop_reason") == "max_steps", "Training incomplete: " + arm)
        data = paths["selection"] + "/" + DATA_FILES[arm]
        require(tc.get("output") == checkpoint and tc.get("data") == data
                and tc.get("data_sha256") == locks[data] == audit["dataset_bytes_sha256"][arm], "Training data binding differs")
        examples = audit["examples"][arm]
        require(len(examples) == ts["training_examples"] == audit["examples_per_arm"], "Training example count differs")
        targets = sum(examples[i]["target_tokens"] for i in sampled)
        inputs = sum(examples[i]["input_tokens"] for i in sampled)
        require(targets == audit["exposures"][arm]["sampled_target_tokens"] == ts["target_tokens"] == TARGET_TOKENS[arm], "Exact target exposure differs: " + arm)
        require(inputs == audit["exposures"][arm]["sampled_input_tokens"], "Input exposure differs")
        state = r.json(checkpoint + "/final/training_state.json")
        require(state["optimizer_steps"] == 100 and state["target_tokens"] == targets and state["protocol"] == PROTOCOL, "Final checkpoint step/token mismatch")
        adapter = r.json(checkpoint + "/final/adapter_config.json")
        require(adapter["base_model_name_or_path"] == MODEL and adapter["r"] == 16 and adapter["lora_alpha"] == 32, "Adapter base/config mismatch")
        weight_keys = [k for k in checkpoint_receipt["checkpoint_sha256"] if k in {
            checkpoint + "/final/adapter_model.safetensors", checkpoint + "/final/adapter_model.bin"}]
        require(len(weight_keys) == 1, "Missing/ambiguous adapter weight receipt")
        weight_receipts[arm] = {k: locks[k] for k in weight_keys}
        metrics = r.jsonl(checkpoint + "/metrics.jsonl")
        require(metrics[-1].get("summary") == ts, "Metrics final summary differs")
        updates = [x for x in metrics if "optimizer_step" in x]
        require([x["optimizer_step"] for x in updates] == list(range(1, 101)), "Missing optimizer steps")
        require(all(x["target_tokens"] == sum(examples[i]["target_tokens"] for i in sampled[:8 * n])
                    and x["examples_per_step"] == 8 and math.isfinite(x["action_token_nll"])
                    for n, x in enumerate(updates, 1)), "Per-update training evidence differs")
        training[arm] = {"optimizer_steps": 100, "target_tokens": targets, "input_tokens": inputs,
                        "prompt_tokens": inputs - targets, "sampled_examples": 800, "training_examples": len(examples),
                        "elapsed_seconds": ts["elapsed_seconds"]}
        ec = r.json(evaluation + "/config.json")
        require(all(ec.get(k) == v for k, v in EVAL.items()) and ec.get("protocol") == PROTOCOL
                and ec.get("manifest_sha256") == locks[MANIFEST] and ec.get("adapter") == checkpoint + "/final"
                and ec.get("games_json") == MANIFEST and ec.get("output") == evaluation, "Evaluation protocol/checkpoint differs")
        episodes[arm] = r.jsonl(evaluation + "/episodes.jsonl")
        require(len(episodes[arm]) == 24 and [e["gamefile"] for e in episodes[arm]] == games, "Incomplete/reordered/duplicate episode identities")
        results[arm] = [validate_episode(e, g) for e, g in zip(episodes[arm], games)]
        require(sum(e["success"] for e in results[arm]) == summaries[arm]["successes"], "Success summary differs")
        require(sum(e["steps"] for e in results[arm]) == summaries[arm]["environment_steps"]
                and sum(e["generated_tokens"] for e in results[arm]) == summaries[arm]["generated_tokens"]
                and summaries[arm]["scored_tokens"] == 0, "Evaluation token/step counts differ")
        require(all(math.isclose(summaries[arm][k], summaries[arm]["successes"] / 24, abs_tol=1e-12)
                    for k in ("success_rate_attempted", "success_rate_planned_lower_bound")), "Failure denominator changed")
    matched = []; families = {}
    for i, game in enumerate(games):
        initial = [(episodes[a][i]["transitions"][0]["observation"], episodes[a][i]["transitions"][0]["admissible_commands"]) for a in ARMS]
        require(initial[0] == initial[1] == initial[2], "Initial observation/admissibles differ: " + game)
        family = PurePosixPath(game).parent.parent.name.split("-")[0]
        row = {"index": i, "gamefile": game, "family": family, "seed": 20261012, "arms": {a: results[a][i] for a in ARMS}}
        matched.append(row)
        group = families.setdefault(family, {"denominator": 0, **{a: {"successes": 0, "failures": 0, "steps": 0, "invalid_actions": 0, "generated_tokens": 0} for a in ARMS}})
        group["denominator"] += 1
        for a in ARMS:
            group[a]["successes"] += results[a][i]["success"]; group[a]["failures"] += not results[a][i]["success"]
            for k in ("steps", "invalid_actions", "generated_tokens"): group[a][k] += results[a][i][k]
    require(set(families) == FAMILIES and all(v["denominator"] == 4 for v in families.values()), "Wrong six-family denominators")
    totals = {}; comparisons = {}
    for arm, vals in results.items():
        lengths = [x["steps"] for x in vals]; successes = sum(x["success"] for x in vals); invalid = sum(x["invalid_actions"] for x in vals)
        totals[arm] = {"denominator": 24, "successes": successes, "failures": 24 - successes, "success_rate": successes / 24,
            "environment_steps": sum(lengths), "invalid_actions": invalid, "invalid_action_rate": invalid / sum(lengths),
            "episode_lengths_including_failures": lengths, "mean_steps_all24": statistics.mean(lengths), "median_steps_all24": statistics.median(lengths),
            "unsuccessful_100_step_episodes": sum(not x["success"] and x["steps"] == 100 for x in vals),
            "stop_reasons_including_failures": dict(Counter(x["stop_reason"] for x in vals)),
            "generated_tokens": summaries[arm]["generated_tokens"], "reported_forward_tokens": summaries[arm]["forward_tokens"],
            "evaluation_elapsed_seconds": summaries[arm]["elapsed_seconds"], "training": training[arm]}
    for other in ("canonical", "uniform"):
        groups = {"gains": [], "losses": [], "ties_success": [], "ties_failure": []}
        for row in matched:
            ref, control = row["arms"]["reference"]["success"], row["arms"][other]["success"]
            label = "gains" if ref and not control else "losses" if control and not ref else "ties_success" if ref else "ties_failure"
            groups[label].append(row["gamefile"])
        comparisons[other] = {**{k: len(v) for k, v in groups.items()}, "ties": len(groups["ties_success"]) + len(groups["ties_failure"]),
            "paired_task_ids": groups, "reference_minus_control_successes": totals["reference"]["successes"] - totals[other]["successes"],
            "rate_difference": (totals["reference"]["successes"] - totals[other]["successes"]) / 24}
    qr = r.json(queue + "/complete_batch_report.json")
    require(qr.get("complete") is True and qr["successes_out_of_24"] == {a: totals[a]["successes"] for a in ARMS}
            and qr.get("raw_initial_state_comparison_available") is False, "Queue report differs")
    for other, counts in comparisons.items():
        require(all(qr["reference_comparisons"][other][k] == counts[k] for k in ("gains", "losses", "ties", "rate_difference")), "Queue pair counts differ")
    scoring = r.json(paths["selection"] + "/selection_summary.json")
    require(scoring["complete"] is True and scoring["training_gate_passed"] is True and not scoring["technical_errors"], "Incomplete selection")
    cpu = r.json(adoption["accepted_cpu_output"] + "/summary.json")
    failed = r.json("outputs/successful_paths_cpu_v1/summary.json")
    require(cpu["complete"] is True and cpu["coverage_gate_passed"] is True and failed["complete"] is False, "Acquisition/failure receipts differ")
    r.verify_consumed()
    return {"kind": "independent_complete_successful_path_result_review", "complete": True,
        "all_three_summary_barrier_passed_before_episode_reads": True, "denominator_per_arm": 24,
        "all_scientific_failures_retained": True, "arms": totals, "families": families,
        "reference_comparisons": comparisons, "matched_tasks": matched,
        "reference_strictly_exceeds_both_controls": all(totals["reference"]["successes"] > totals[a]["successes"] for a in ("canonical", "uniform")),
        "initial_observation_and_admissible_equality": True, "raw_initial_state_comparison_available": False,
        "verified_file_sha256": r.consumed, "checkpoint_weight_hash_receipts_not_reread": weight_receipts,
        "costs": {"scoring": {k: scoring[k] for k in ("scored_action_tokens", "forward_tokens", "cache_hits", "model_calls", "wall_seconds")},
            "acquisition": {k: cpu.get(k) for k in ("environment_step_calls_started", "reset_calls_started", "wall_seconds", "historical_costs", "cumulative_environment_step_calls_started", "cumulative_reset_calls_started")},
            "failed_v1_acquisition_steps": failed["environment_step_calls_started"],
            "queue_stage_wall_seconds": {s["name"]: s["elapsed_seconds"] for s in stages}},
        "scope": "Single seed, reused exploratory dev24. No confirmatory claim or new-run authorization.",
        "verification_limits": ["First observations/admissibles compared directly; raw initial states are not recorded.",
            "Game and weight hashes cross-checked through locked receipts; original games and large adapter weights are not reread locally.",
            "Generated tokens/invalid actions recomputed from transitions; forward-token totals reported from locked evaluator summaries."]}


def write_outputs(result, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    (directory / "review.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    keys = ("success", "steps", "invalid_actions", "generated_tokens", "stop_reason")
    with (directory / "paired_tasks.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["index", "gamefile", "family", "seed"] + [a + "_" + k for a in ARMS for k in keys]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in result["matched_tasks"]:
            writer.writerow({**{k: row[k] for k in fields[:4]}, **{a + "_" + k: row["arms"][a][k] for a in ARMS for k in keys}})
    with (directory / "families.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["family", "denominator"] + [a + "_successes" for a in ARMS]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for family, row in sorted(result["families"].items()):
            writer.writerow({"family": family, "denominator": row["denominator"], **{a + "_successes": row[a]["successes"] for a in ARMS}})
    with (directory / "arms.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        keys = ["successes", "failures", "denominator", "success_rate", "environment_steps", "invalid_actions",
                "invalid_action_rate", "generated_tokens", "reported_forward_tokens", "mean_steps_all24",
                "median_steps_all24", "unsuccessful_100_step_episodes", "evaluation_elapsed_seconds"]
        fields = ["arm"] + keys + ["training_target_tokens", "training_input_tokens", "training_elapsed_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for arm in ARMS:
            row = result["arms"][arm]
            writer.writerow({"arm": arm, **{k: row[k] for k in keys},
                **{"training_" + k: row["training"][k] for k in ("target_tokens", "input_tokens", "elapsed_seconds")}})
    with (directory / "reference_comparisons.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        keys = ["gains", "losses", "ties", "ties_success", "ties_failure", "reference_minus_control_successes", "rate_difference"]
        writer = csv.DictWriter(stream, fieldnames=["control"] + keys); writer.writeheader()
        for other, row in result["reference_comparisons"].items():
            writer.writerow({"control": other, **{k: row[k] for k in keys}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path, help="Local artifact tree; never SSH")
    p.add_argument("--adoption-json", required=True, type=Path, help="Path relative to --root")
    p.add_argument("--output", required=True, type=Path, help="New local report directory")
    args = p.parse_args(); require(not args.output.exists(), "Output already exists")
    try: result = review(args.root, args.adoption_json)
    except (ReviewError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"complete": False, "reason": str(error), "partial_outcomes_reported": False})); return 1
    write_outputs(result, args.output)
    print(json.dumps({"complete": True, "arms": {a: result["arms"][a]["successes"] for a in ARMS}, "report": str(args.output / "review.json")}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
