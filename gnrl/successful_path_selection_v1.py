"""Adoption-gated offline successful-path selection; no environment or dev reads.

Prototype preparation does not authorize model execution. Live CLI additionally
requires the GPU launcher, a complete CPU60 gate and a hash-bound learning
adoption. All three arms retain the same60 tasks and the identical finite pools.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .policy import HFCausalPolicy, PROTOCOL_VERSION, encode_action, require_gpu_guard
from .successful_path_pools_v2 import digest, summarize_acquisition, validate_successful_pool

ROOT = Path(__file__).resolve().parents[1]
SEARCH_CONFIG = "configs/successful_path_selection_candidate_v1.json"
LEARNING_CONFIG = "configs/successful_path_learning_candidate_v1.json"
NATIVE = "data/planner_train120_raw.jsonl"
TRAIN_MANIFEST = "data/manifests/train120.json"
MODEL = "/home/jsy/zhangranxu-adrs-week-20260929/models/Qwen2.5-3B-Instruct"
MODEL_INVENTORY = "outputs/reference_model_inventory_v2.json"
SOURCES = (SEARCH_CONFIG, LEARNING_CONFIG, "gnrl/successful_path_selection_v1.py",
           "gnrl/successful_path_pools_v2.py", "gnrl/policy.py", "gnrl/alfworld_env.py", "gnrl/two_object_blocks.py")
CPU_FILES = ("summary.json", "candidate_pools.jsonl", "canonical_baseline.jsonl", "canonical_source_index.json", "controller.json")
LIMITS = {"max_score_seconds": 3600, "max_scored_action_tokens": 1000000,
          "max_forward_tokens": 32000000, "max_context_tokens": 2048,
          "max_action_tokens_including_eos": 64, "score_batch_size": 2}
ARM_FILES = {"canonical": "canonical_baseline.jsonl", "uniform": "uniform_valid.jsonl", "reference": "learner_selected.jsonl"}


def require(condition, reason):
    if not condition: raise ValueError(reason)


def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): result.update(chunk)
    return result.hexdigest()


def rows(path): return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def write_json(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def dump_line(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()


def token_digest(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def validate_acquisition(directory):
    """Fail closed before model loading; recompute the entire original CPU gate."""
    directory = Path(directory)
    config = read_json(ROOT / SEARCH_CONFIG)
    expected = config["fixed_training_cohort"]["frozen_gamefiles"]
    summary = read_json(directory / "summary.json")
    require(summary.get("complete") is True and summary.get("coverage_gate_passed") is True
            and summary.get("stop_reason") == "complete" and not summary.get("technical_errors"), "CPU acquisition incomplete or gate failed")
    require(0 <= summary["environment_step_calls_started"] <= 100000
            and summary["environment_step_calls_completed"] == summary["environment_step_calls_started"]
            and summary["reset_calls_completed"] == summary["reset_calls_started"]
            and summary["wall_seconds"] <= 3600 and summary.get("gpu_jobs") == 0, "CPU budget/call accounting invalid")
    pools = rows(directory / "candidate_pools.jsonl")
    verified = summarize_acquisition(pools, expected)
    require(verified["complete"] and verified["coverage_gate_passed"], "Recomputed CPU gate failed")
    require(sum(len(p["attempts"]) for p in pools) <= 420, "CPU proposal cap exceeded")
    native = {json.loads(line)["gamefile"]: line for line in (ROOT / NATIVE).read_bytes().splitlines(keepends=True) if line.strip()}
    manifest = read_json(ROOT / TRAIN_MANIFEST)
    originals = []
    for pool in pools:
        validate_successful_pool(pool)
        require(pool.get("verifier_version") == 2, "Expected reviewed v2 verifier")
        raw = pool["canonical_source_line_utf8"].encode()
        require(raw == native[pool["gamefile"]] and raw.endswith(b"\n"), "Canonical source bytes changed")
        require(pool["source_hashes"]["game_sha256"] == manifest["game_sha256"][pool["gamefile"]], "Original game hash differs")
        originals.append(raw)
    require((directory / "canonical_baseline.jsonl").read_bytes() == b"".join(originals), "CPU canonical60 bytes differ")
    index = read_json(directory / "canonical_source_index.json")["records"]
    require(index == [{"gamefile": p["gamefile"], "seed": p["seed"], "source_line_sha256": hashlib.sha256(raw).hexdigest()}
                      for p, raw in zip(pools, originals)], "Canonical source index differs")
    cpu_hashes = summary["adoption"]["file_sha256"]
    for name in (SEARCH_CONFIG, NATIVE, TRAIN_MANIFEST, "gnrl/successful_path_pools_v2.py", "gnrl/policy.py"):
        require(sha_file(ROOT / name) == cpu_hashes[name], "Acquisition-bound input changed: " + name)
    return pools, {"acquisition_dir": str(directory.resolve()), "recomputed_cpu_gate": verified,
                   "file_sha256": {name: sha_file(directory / name) for name in CPU_FILES},
                   "native120_sha256": sha_file(ROOT / NATIVE), "ordered_pool_sha256": [digest(p) for p in pools]}


def required_adoption_paths(directory, provenance):
    return [ROOT / p for p in SOURCES] + [ROOT / NATIVE, ROOT / TRAIN_MANIFEST, ROOT / MODEL_INVENTORY, Path(provenance)] + [Path(directory) / n for n in CPU_FILES]


def validate_adoption(path, directory, provenance):
    path = Path(path)
    require(path.is_file(), "Explicit learning adoption required; CPU adoption does not authorize scoring")
    adoption = read_json(path)
    require(adoption.get("kind") == "successful_path_learning_adoption" and adoption.get("adopted") is True
            and adoption.get("scope") == "frozen60_single_seed_three_arm_exploratory_learning"
            and adoption.get("physical_gpu") == 2, "Wrong learning adoption scope")
    require(Path(adoption["accepted_cpu_output"]).resolve() == Path(directory).resolve(), "Adoption identifies another acquisition")
    hashes = adoption["file_sha256"]
    for file in required_adoption_paths(directory, provenance):
        require(file.resolve().is_relative_to(ROOT.resolve()), "Adopted inputs must reside in project")
        name = file.resolve().relative_to(ROOT.resolve()).as_posix()
        require(hashes.get(name) == sha_file(file), "Missing/changed learning-adopted input: " + name)
    config = read_json(ROOT / LEARNING_CONFIG)
    require(all(config["reference_scoring"].get(k) == v for k, v in LIMITS.items()), "Registered scoring limits changed")
    require(config["reference_scoring"]["model_path"] == MODEL and config["reference_scoring"]["adapter_path"] is None
            and config["same60_same_pool_data"]["uniform_seed"] == 42, "Reference/selection protocol changed")
    return {"adoption_sha256": sha_file(path), "adoption_path": str(path.resolve()),
            "file_sha256": {file.resolve().relative_to(ROOT.resolve()).as_posix(): sha_file(file)
                            for file in required_adoption_paths(directory, provenance)}}


def verify_model(provenance, model_path=MODEL, inventory_path=None):
    """Verify every frozen model/tokenizer byte; never instantiate a model here."""
    provenance = Path(provenance); data = read_json(provenance); root = Path(model_path).resolve()
    require(root == Path(data["model_path"]).resolve() == Path(data["resolved_path"]).resolve(), "Reference model path differs")
    core = data["files"]; names = [f["name"] for f in core]
    require(len(names) == len(set(names)) and {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"} <= set(names), "Incomplete model provenance")
    inventory_path = Path(inventory_path) if inventory_path is not None else ROOT / MODEL_INVENTORY
    inventory = read_json(inventory_path)
    require(Path(inventory["model_path"]).resolve() == root and inventory.get("algorithm") == "SHA256", "Wrong full model inventory")
    files = inventory["files"]; full_names = [f["name"] for f in files]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(len(full_names) == len(set(full_names)) and actual == set(full_names)
            and not (root / "adapter_config.json").exists(), "Unbound loader/model inventory files")
    indexed = {f["name"]: f for f in files}
    require(all(f["name"] in indexed and all(indexed[f["name"]][k] == f[k] for k in ("sha256", "bytes")) for f in core), "Core model provenance differs from full inventory")
    for file in files:
        path = (root / file["name"]).resolve()
        require(path.is_relative_to(root) and path.is_file() and path.stat().st_size == file["bytes"]
                and sha_file(path) == file["sha256"], "Reference model bytes differ: " + file["name"])
    index = read_json(root / "model.safetensors.index.json")
    require(set(index["weight_map"].values()) <= set(names), "Unbound reference weight shard")
    return {"provenance_sha256": sha_file(provenance), "full_inventory_sha256": sha_file(inventory_path), "model_path": str(root), "files": files,
            "model_files_digest": digest([{k: f[k] for k in ("name", "bytes", "sha256")} for f in files]), "adapter": None}


def uniform_choice(pool, seed=42):
    pool_hash = digest(pool)
    material = f"{seed}:{pool_hash}"
    seed_int = int(hashlib.sha256(material.encode()).hexdigest(), 16)
    index = random.Random(seed_int).randrange(len(pool["candidates"]))
    return pool["candidates"][index], {"seed": seed, "seed_material": material,
        "seed_integer_hex": format(seed_int, "064x"), "algorithm": "random.Random(seed_int).randrange(pool_size)",
        "candidate_order": [c["candidate_id"] for c in pool["candidates"]], "selected_index": index}


def reference_choice(pool, scores):
    require(set(scores) == {c["candidate_id"] for c in pool["candidates"]}
            and all(math.isfinite(v["logprob"]) for v in scores.values()), "Missing/nonfinite complete candidate scores")
    maximum = max(v["logprob"] for v in scores.values())
    ties = sorted(cid for cid, value in scores.items() if value["logprob"] == maximum)
    cid = pool["canonical_candidate_id"] if pool["canonical_candidate_id"] in ties else ties[0]
    return next(c for c in pool["candidates"] if c["candidate_id"] == cid), ties


class ScoringBudgetExceeded(RuntimeError): pass


class Scorer:
    def __init__(self, policy, *, limits=None, started=None):
        self.policy = policy; self.limits = dict(LIMITS if limits is None else limits)
        self.started = time.monotonic() if started is None else started
        self.cache = {}; self.cache_hits = 0; self.calls = 0
        require(policy.scored_tokens == policy.forward_tokens == policy.generated_tokens == 0, "Reference scorer must start fresh")

    def check(self):
        if time.monotonic() - self.started >= self.limits["max_score_seconds"]: raise ScoringBudgetExceeded("score_wall_seconds")

    def score(self, pool, candidate, ledger):
        history = []; per_action = []; target_total = 0; prompt_total = 0
        pool_hash = digest(pool)
        for step, t in enumerate(candidate["transitions"]):
            self.check()
            prompt = self.policy.prompt_ids(pool["task"], history, t["observation"], t["admissible_commands"])
            target = encode_action(self.policy.tokenizer, t["action"], self.policy.max_action_tokens)
            key = (tuple(prompt), tuple(target)); cache_hit = key in self.cache
            if not cache_hit:
                if self.policy.scored_tokens + len(target) > self.limits["max_scored_action_tokens"]: raise ScoringBudgetExceeded("action_token_budget")
                if self.policy.forward_tokens + len(prompt) + len(target) > self.limits["max_forward_tokens"]: raise ScoringBudgetExceeded("forward_token_budget")
                before_tokens, before_forward = self.policy.scored_tokens, self.policy.forward_tokens
                result = self.policy.score_encoded_actions(prompt, [t["action"]])
                require(len(result) == 1 and result[0]["action"] == t["action"]
                        and result[0]["token_count"] == len(target) and math.isfinite(result[0]["logprob"]), "Invalid action score")
                require(self.policy.scored_tokens - before_tokens == len(target)
                        and self.policy.forward_tokens - before_forward == len(prompt) + len(target), "Unexpected model token accounting")
                self.cache[key] = float(result[0]["logprob"]); self.calls += 1
            else: self.cache_hits += 1
            self.check()
            score = self.cache[key]; per_action.append(score); target_total += len(target); prompt_total += len(prompt)
            dump_line(ledger, {"gamefile": pool["gamefile"], "pool_sha256": pool_hash,
                "candidate_id": candidate["candidate_id"], "step": step, "prompt_ids_sha256": token_digest(prompt),
                "action_ids_sha256": token_digest(target), "prompt_tokens": len(prompt),
                "action_tokens_including_eos": len(target), "logprob": score, "cache_hit": cache_hit})
            history.append({"observation": t["observation"], "action": t["action"]})
        return {"logprob": sum(per_action), "per_action_logprobs": per_action,
                "action_tokens_including_eos": target_total, "prompt_tokens": prompt_total}


def selected_row(pool, candidate, arm, scores):
    return {"kind": "trajectory", "gamefile": pool["gamefile"], "seed": pool["seed"], "split": "train",
        "task": pool["task"], "initial_observation": pool["initial_observation"], "success": True,
        "replay_verified": True, "reset_replay_verified": True, "transitions": copy.deepcopy(candidate["transitions"]),
        "replay_evidence": candidate["evidence"], "selection": {"arm": arm, "pool_sha256": digest(pool),
        "candidate_id": candidate["candidate_id"], "canonical_candidate_id": pool["canonical_candidate_id"],
        "pool_size": len(pool["candidates"]), "reference_score": scores[candidate["candidate_id"]],
        "method": "outcome_constrained_successful_path_selection", "terminal_semantics_audit": candidate["terminal_semantics_audit"]}}


def pairing_record(pool, uniform, reference):
    candidates = {"canonical": pool["candidates"][0], "uniform": uniform, "reference": reference}
    return {"gamefile": pool["gamefile"], "seed": pool["seed"], "pool_sha256": digest(pool),
        "canonical_source_line_sha256": pool["canonical_source_line_sha256"],
        "candidates_by_arm": {arm: {"candidate_id": c["candidate_id"],
            "action_digest": digest([t["action"] for t in c["transitions"]]),
            "trajectory_digest": digest(c["transitions"])} for arm, c in candidates.items()}}


def export_selection(pools, output, policy, *, input_receipt, model_receipt, adoption_receipt, limits=None, started=None):
    """Internal prepared API; live CLI verifies CPU/adoption/model before calling."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    scorer = Scorer(policy, limits=limits, started=started)
    records = []; errors = []; complete = False
    try:
        with (output / "score_ledger.jsonl").open("x", encoding="utf-8") as ledger, \
             (output / "selection_events.jsonl").open("x", encoding="utf-8") as events:
            for pool in pools:
                validate_successful_pool(pool)
                scores = {c["candidate_id"]: scorer.score(pool, c, ledger) for c in pool["candidates"]}
                selected, ties = reference_choice(pool, scores); uniform, rng = uniform_choice(pool)
                records.append((pool, uniform, selected, scores))
                dump_line(events, {"gamefile": pool["gamefile"], "pool_sha256": digest(pool), "scores": scores,
                    "reference_candidate_id": selected["candidate_id"], "uniform_candidate_id": uniform["candidate_id"],
                    "reference_ties": ties, "uniform": rng})
                scorer.cache.clear()
        scorer.check()
        require(len(pools) == 60 and len(records) == 60, "Entire60 scoring required")
        # No arm data is emitted before every candidate in every pool is scored.
        with (output / ARM_FILES["canonical"]).open("xb") as canonical, \
             (output / ARM_FILES["uniform"]).open("x", encoding="utf-8") as uniform_file, \
             (output / ARM_FILES["reference"]).open("x", encoding="utf-8") as reference_file:
            for pool, uniform, selected, scores in records:
                canonical.write(pool["canonical_source_line_utf8"].encode())
                dump_line(uniform_file, selected_row(pool, uniform, "uniform", scores))
                dump_line(reference_file, selected_row(pool, selected, "reference", scores))
        pairing = {"complete": True, "records": [pairing_record(p, u, r) for p, u, r, _ in records]}
        write_json(output / "pairing_receipt.json", pairing)
        complete = True
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    changed_canonical = sum(r["candidate_id"] != p["canonical_candidate_id"] for p, _, r, _ in records)
    changed_uniform = sum(r["candidate_id"] != u["candidate_id"] for _, u, r, _ in records)
    summary = {"complete": complete, "training_gate_passed": complete and changed_canonical > 0 and changed_uniform > 0,
        "paired_tasks": len(records) if complete else 0, "fully_scored_pools": len(records), "technical_errors": errors,
        "reference_changed_from_canonical": changed_canonical, "reference_changed_from_uniform": changed_uniform,
        "stop_reason": "complete" if complete else "incomplete_scoring", "limits": scorer.limits, "protocol": PROTOCOL_VERSION,
        "scored_action_tokens": policy.scored_tokens, "forward_tokens": policy.forward_tokens, "generated_tokens": policy.generated_tokens,
        "cache_hits": scorer.cache_hits, "model_calls": scorer.calls, "wall_seconds": time.monotonic() - scorer.started,
        "input_receipt": input_receipt, "model_receipt": model_receipt, "adoption_receipt": adoption_receipt,
        "output_sha256": {p.name: sha_file(p) for p in output.iterdir() if p.is_file()},
        "interpretation": "Raw full-trajectory likelihood argmax over successful paths; different terminal states are allowed. Score gain is not learning evidence."}
    write_json(output / "selection_summary.json", summary)
    return summary


def validate_selection_artifacts(pools, output):
    output = Path(output); summary = read_json(output / "selection_summary.json")
    require(summary.get("complete") is True and summary.get("training_gate_passed") is True
            and summary.get("paired_tasks") == 60 and not summary.get("technical_errors"), "Selection incomplete/degenerate")
    require(len({p["gamefile"] for p in pools}) == 60 and summary.get("fully_scored_pools") == 60,
            "Duplicate/missing frozen cohort")
    require(set(summary["output_sha256"]) == set(ARM_FILES.values()) | {"score_ledger.jsonl", "selection_events.jsonl", "pairing_receipt.json"}, "Incomplete output hash manifest")
    for name, value in summary["output_sha256"].items(): require(sha_file(output / name) == value, "Selection output changed: " + name)
    pairing = read_json(output / "pairing_receipt.json")
    arms = {arm: rows(output / name) for arm, name in ARM_FILES.items()}
    require((output / ARM_FILES["canonical"]).read_bytes() == b"".join(p["canonical_source_line_utf8"].encode() for p in pools), "Canonical bytes changed")
    events = rows(output / "selection_events.jsonl"); ledger = rows(output / "score_ledger.jsonl")
    expected_keys = [(p["gamefile"], c["candidate_id"], i) for p in pools for c in p["candidates"] for i in range(len(c["transitions"]))]
    require([(r["gamefile"], r["candidate_id"], r["step"]) for r in ledger] == expected_keys, "Incomplete/reordered score ledger")
    require(len(pools) == len(pairing["records"]) == len(events) == 60 and all(len(r) == 60 for r in arms.values()), "Missing paired cohort")
    lookup = {(r["gamefile"], r["candidate_id"], r["step"]): r for r in ledger}
    misses = [r for r in ledger if not r["cache_hit"]]
    require(summary["scored_action_tokens"] == sum(r["action_tokens_including_eos"] for r in misses) <= LIMITS["max_scored_action_tokens"]
            and summary["forward_tokens"] == sum(r["prompt_tokens"] + r["action_tokens_including_eos"] for r in misses) <= LIMITS["max_forward_tokens"]
            and summary["cache_hits"] == len(ledger) - len(misses) and summary["model_calls"] == len(misses)
            and summary["generated_tokens"] == 0 and summary["wall_seconds"] <= LIMITS["max_score_seconds"], "Scoring cost accounting differs")
    require(all(type(r["cache_hit"]) is bool and type(r["prompt_tokens"]) is int and r["prompt_tokens"] > 0
            and type(r["action_tokens_including_eos"]) is int and 0 < r["action_tokens_including_eos"] <= 64
            and all(len(r[k]) == 64 and set(r[k]) <= set("0123456789abcdef") for k in ("prompt_ids_sha256", "action_ids_sha256")) for r in ledger), "Malformed token ledger")
    changed_canonical = changed_uniform = 0
    for index, pool in enumerate(pools):
        validate_successful_pool(pool); event = events[index]
        require(event["gamefile"] == pool["gamefile"] and event["pool_sha256"] == digest(pool), "Score event pool differs")
        scores = event["scores"]
        for c in pool["candidates"]:
            parts = [lookup[(pool["gamefile"], c["candidate_id"], i)] for i in range(len(c["transitions"]))]
            require(all(r["pool_sha256"] == digest(pool) and math.isfinite(r["logprob"]) and r["action_tokens_including_eos"] > 0 for r in parts), "Invalid score ledger row")
            require(scores[c["candidate_id"]] == {"logprob": sum(r["logprob"] for r in parts),
                "per_action_logprobs": [r["logprob"] for r in parts], "action_tokens_including_eos": sum(r["action_tokens_including_eos"] for r in parts),
                "prompt_tokens": sum(r["prompt_tokens"] for r in parts)}, "Candidate sum differs from complete action ledger")
        reference, ties = reference_choice(pool, scores); uniform, rng = uniform_choice(pool)
        require(event["uniform"] == rng and event["reference_ties"] == ties
                and event["reference_candidate_id"] == reference["candidate_id"] and event["uniform_candidate_id"] == uniform["candidate_id"], "Selection rule differs")
        require(pairing["records"][index] == pairing_record(pool, uniform, reference), "Immutable pairing receipt differs")
        for arm, candidate in (("uniform", uniform), ("reference", reference)):
            require(arms[arm][index] == selected_row(pool, candidate, arm, scores), "Selected training row differs")
        changed_canonical += reference["candidate_id"] != pool["canonical_candidate_id"]
        changed_uniform += reference["candidate_id"] != uniform["candidate_id"]
    require(changed_canonical > 0 and changed_uniform > 0, "Selector comparison degenerate")
    require(changed_canonical == summary["reference_changed_from_canonical"]
            and changed_uniform == summary["reference_changed_from_uniform"], "Change-count summary differs")
    return pairing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adoption-json", required=True, type=Path)
    parser.add_argument("--model-provenance", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); started = time.monotonic()
    require(not args.output.exists() and args.output.resolve().is_relative_to((ROOT / "outputs").resolve()), "Selection output must be a new project outputs directory")
    require_gpu_guard()
    adoption = validate_adoption(args.adoption_json, args.acquisition_dir, args.model_provenance)
    require(args.output.resolve() == Path(read_json(args.adoption_json)["output_paths"]["selection"]).resolve(), "Unadopted selection output")
    pools, receipt = validate_acquisition(args.acquisition_dir)
    model = verify_model(args.model_provenance)
    policy = HFCausalPolicy(MODEL, adapter_path=None, device=args.device,
        max_context_tokens=2048, max_action_tokens=64, score_batch_size=2)
    summary = export_selection(pools, args.output, policy, input_receipt=receipt, model_receipt=model,
                               adoption_receipt=adoption, started=started)
    try:
        require(validate_adoption(args.adoption_json, args.acquisition_dir, args.model_provenance) == adoption, "Adoption inputs changed during scoring")
        require(verify_model(args.model_provenance) == model, "Reference changed during scoring")
        require(time.monotonic() - started < LIMITS["max_score_seconds"], "Selection wall-time budget exceeded during verification")
        if summary["training_gate_passed"]: validate_selection_artifacts(pools, args.output)
    except Exception as error:
        summary.update(complete=False, training_gate_passed=False, stop_reason="final_verification_failed")
        summary["technical_errors"].append(str(error)); write_json(args.output / "selection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["complete"] else 1


if __name__ == "__main__": raise SystemExit(main())
