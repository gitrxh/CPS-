"""Fixed 4+4 recovery SFT from the common canonical LoRA, with a joint token audit.

Importing this module loads no tokenizer, torch, model, or simulator. Audit-only
requires the launcher guard and loads only the tokenizer; training requires the
complete joint audit. Existing adapter weights are loaded once, trainable, onto
the original base; a new AdamW is constructed and no optimizer is resumed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from .policy import PROTOCOL_VERSION, require_gpu_guard, supervised_example
from .train_sft import collate_examples, load_examples

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("expert_recovery", "uniform_viable_recovery", "reference_conditional_recovery")
BASE_MODEL = "/home/jsy/zhangranxu-adrs-week-20260929/models/Qwen2.5-3B-Instruct"
INITIAL_ADAPTER = ROOT / "checkpoints/successful_path_canonical_seed42_v1/final"
BC_SHA256 = "0cc074e4f07ce999a889b84ba9a3fe2ae44fec5e5ac81c2c0623764859e7fe70"
MODEL_INVENTORY = ROOT / "outputs/reference_model_inventory_v2.json"
MODEL_INVENTORY_SHA256 = "6100d348e2454c113412e4c47228adaa66e2d6c2cf88f280dda93f117a3b8c8f"
INITIAL_LOCKS = ROOT / "outputs/successful_path_learning_queue_v1/locked_sha256.json"
TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
SETTINGS = {"optimizer_steps": 100, "seed": 42, "batch_size": 1,
            "gradient_accumulation": 8, "bc_per_step": 4, "recovery_per_step": 4,
            "bc_schedule_seed": 613300, "anchor_schedule_seed": 613301,
            "learning_rate": 2e-4, "lora_rank": 16, "lora_alpha": 32,
            "lora_dropout": 0.05, "weight_decay": 0.0, "gradient_clip_norm": 1.0,
            "max_context_tokens": 2048, "max_action_tokens": 64,
            "max_target_tokens": 2000000, "max_wall_seconds": 7200,
            "dtype": "bfloat16", "loss_scope": "action_tokens_and_eos_only",
            "accumulation_weighting": "target_tokens_in_entire_8_example_window",
            "optimizer": "fresh_AdamW", "optimizer_state_resumed": False,
            "adapter_stacking": False, "checkpoint": "final_step100_only"}
IDENTITY_FIELDS = ("anchor_id", "gamefile", "seed", "after_actions", "pool_sha256", "selection_sha256")
CONTEXT_FIELDS = ("task", "history", "observation", "admissible_commands")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def shuffled_indices(size, count, seed):
    """Shuffle the existing permutation on every wrap, using its private RNG."""
    require(size > 0 and count >= 0, "Invalid schedule size/count")
    rng = random.Random(seed)
    order = list(range(size))
    rng.shuffle(order)
    indices, cursor = [], 0
    for _ in range(count):
        if cursor == size:
            rng.shuffle(order)
            cursor = 0
        indices.append(order[cursor])
        cursor += 1
    return indices


def exact_schedule(bc_size, recovery_size):
    bc = shuffled_indices(bc_size, 400, SETTINGS["bc_schedule_seed"])
    recovery = shuffled_indices(recovery_size, 400, SETTINGS["anchor_schedule_seed"])
    return [{"optimizer_step": step + 1, "microstep": 2 * slot + offset,
             "source": source, "index": indices[4 * step + slot]}
            for step in range(100) for slot in range(4)
            for offset, source, indices in ((0, "bc", bc), (1, "recovery", recovery))]


def causal_row(row):
    """Explicit allowlist; identity, oracle evidence and scores are never model inputs."""
    result = {field: row[field] for field in CONTEXT_FIELDS}
    result["history"] = [{"observation": h["observation"], "action": h["action"]}
                         for h in row["history"]]
    result["action"] = row["action"]
    require(result["action"] in result["admissible_commands"], "Recovery label is inadmissible")
    return result


def validate_labels(datasets):
    require(set(datasets) == set(ARMS), "Joint audit needs exactly all three arms")
    reference = datasets[ARMS[0]]
    require(0 < len(reference) <= 180, "Empty/oversized eligible recovery set")
    require(len({r["anchor_id"] for r in reference}) == len(reference), "Duplicate anchor id")
    for arm in ARMS:
        require(len(datasets[arm]) == len(reference), "Cross-arm anchor count differs")
        for first, row in zip(reference, datasets[arm]):
            require(row.get("kind") == "recovery_supervised_label" and row.get("schema_version") == 1
                    and row.get("arm") == arm, "Unexpected recovery label schema/arm")
            require(row["seed"] == 20261012 and row["after_actions"] in (4, 8, 16), "Unexpected anchor seed/time")
            require(len(row["history"]) == row["after_actions"], "Incomplete learner causal history")
            require(all(row[k] == first[k] for k in IDENTITY_FIELDS + CONTEXT_FIELDS),
                    "Cross-arm ordered identity/context mismatch")
            causal_row(row)
    return [r["anchor_id"] for r in reference]


def validate_adapter_config(config):
    require(config.get("peft_type") == "LORA" and config.get("task_type") == "CAUSAL_LM",
            "Initial adapter is not causal-LM LoRA")
    require(config.get("r") == 16 and config.get("lora_alpha") == 32
            and config.get("lora_dropout") == 0.05 and config.get("bias") == "none",
            "Initial adapter hyperparameters differ")
    require(set(config.get("target_modules", [])) == TARGET_MODULES, "Initial LoRA target modules differ")
    require(not config.get("rank_pattern") and not config.get("alpha_pattern")
            and not config.get("use_dora") and not config.get("use_rslora")
            and not config.get("modules_to_save"), "Unexpected nonstandard initial adapter")


def inventory_tree(directory):
    root = Path(directory).resolve()
    require(root.is_dir(), "Inventory root missing")
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        require(path.resolve().is_relative_to(root), "Inventory file escapes root")
        before = path.stat()
        sha = sha_file(path)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "File changed while hashing")
        files.append({"name": path.relative_to(root).as_posix(), "bytes": after.st_size, "sha256": sha})
    require(files, "Empty inventory")
    return {"root": str(root), "files": files, "files_sha256": digest(files)}


def input_provenance(args, recovery_paths):
    require(Path(args.model).resolve() == Path(BASE_MODEL).resolve(), "Original frozen base required")
    require(Path(args.initial_adapter).resolve() == INITIAL_ADAPTER.resolve(), "Canonical final initial adapter required")
    require(sha_file(args.bc_data) == BC_SHA256, "Common original canonical60 BC bytes changed")
    initial = inventory_tree(args.initial_adapter)
    locks = read_json(INITIAL_LOCKS)
    prefix = INITIAL_ADAPTER.relative_to(ROOT).as_posix() + "/"
    expected = {key[len(prefix):]: value for key, value in locks.items() if key.startswith(prefix)}
    actual = {f["name"]: f["sha256"] for f in initial["files"]}
    require(actual == expected, "Canonical final inventory differs from completed learning locks")
    config = read_json(Path(args.initial_adapter) / "adapter_config.json")
    validate_adapter_config(config)
    require(Path(config["base_model_name_or_path"]).resolve() == Path(args.model).resolve(), "Adapter base differs")
    initial_state = read_json(Path(args.initial_adapter) / "training_state.json")
    require(initial_state["optimizer_steps"] == 100, "Initial adapter is not canonical final step100")
    require(sha_file(MODEL_INVENTORY) == MODEL_INVENTORY_SHA256, "Frozen base inventory changed")
    model_inventory = read_json(MODEL_INVENTORY)
    base = inventory_tree(args.model)
    expected_model = [{k: f[k] for k in ("name", "bytes", "sha256")} for f in model_inventory["files"]]
    require(base["files"] == sorted(expected_model, key=lambda f: f["name"]), "Original base bytes differ")
    require(not (Path(args.model) / "adapter_config.json").exists(), "Base must not already include an adapter")
    return {"base": base, "initial_adapter": initial, "initial_training_state": initial_state,
            "initial_locks_sha256": sha_file(INITIAL_LOCKS), "base_inventory_sha256": MODEL_INVENTORY_SHA256,
            "bc_data": {"path": str(Path(args.bc_data).resolve()), "sha256": BC_SHA256},
            "recovery_data": {arm: {"path": str(Path(path).resolve()), "sha256": sha_file(path)}
                              for arm, path in recovery_paths.items()},
            "source_sha256": {p: sha_file(ROOT / p) for p in
                              ("gnrl/train_recovery_v1.py", "gnrl/train_sft.py", "gnrl/policy.py")},
            "runtime": {"python": sys.version, "executable": sys.executable,
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ("torch", "transformers", "peft", "tokenizers", "safetensors")},
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
            "protocol": PROTOCOL_VERSION}


def prepare_raw(args, recovery_paths):
    datasets = {arm: read_rows(path) for arm, path in recovery_paths.items()}
    anchors = validate_labels(datasets)
    bc = load_examples(args.bc_data)
    require(len(bc) == 349, "Expected exactly 349 common BC examples")
    return bc, datasets, anchors


def selection_artifacts(directory, recovery_paths, datasets):
    """Bind complete score-stage artifacts; this performs no model/environment work."""
    directory = Path(directory).resolve()
    required = {"summary.json", "records.jsonl", "events.jsonl", "score_ledger.json",
                "config.json", "cost_progress.json", *(arm + ".jsonl" for arm in ARMS)}
    inventory = inventory_tree(directory)
    require(required <= {f["name"] for f in inventory["files"]}, "Incomplete selection directory")
    summary = read_json(directory / "summary.json")
    require(summary.get("complete") is True and summary.get("stage") == "score"
            and summary.get("labels_locked_before_Q") is True
            and not summary.get("technical_incomplete") and not summary.get("error")
            and summary.get("record_count") == 180, "Complete locked score stage required")
    require(sha_file(directory / "records.jsonl") == summary["records_sha256"], "Selection records hash differs")
    selected = read_rows(directory / "records.jsonl")
    require(len(selected) == 180 and len({s["anchor_id"] for s in selected}) == 180,
            "Missing/duplicate complete selection slots")
    for row in selected:
        body = dict(row)
        artifact = body.pop("artifact_sha256")
        require(digest(body) == artifact and row.get("kind") == "recovery_selection"
                and row.get("schema_version") == 1 and row.get("complete") is True,
                "Incomplete/unsealed selection record")
    eligible = [s for s in selected if s["eligible"]]
    require(summary.get("eligible_anchors") == len(eligible), "Selection eligible count differs")
    expected_events = {}
    scored_targets = scored_forward = expected_calls = 0
    for row in selected:
        if not row["eligible"]:
            require(not row.get("reference_scores") and not row.get("base_scores") and not row.get("labels"),
                    "Unavailable selection contains labels/scores")
            continue
        actions = row["actions"]
        require(row["status"] == "eligible" and 1 <= len(actions) <= 8
                and len(set(actions)) == len(actions) and set(row["labels"]) == set(ARMS),
                "Invalid eligible selection support")
        for role in ("reference", "base"):
            scores = row[role + "_scores"]
            require(set(scores) == set(actions), "Incomplete complete-action scores")
            for action in actions:
                score = scores[action]
                require(type(score["logprob"]) in (int, float) and math.isfinite(score["logprob"]),
                        "Nonfinite selected-stage score")
                require(type(score["action_tokens_including_eos"]) is int
                        and 0 < score["action_tokens_including_eos"] <= 64
                        and type(score["prompt_tokens"]) is int and 0 < score["prompt_tokens"] <= 1984,
                        "Invalid score token counts")
                key = (role, row["anchor_id"], action)
                expected_events[key] = {"kind": "recovery_action_score", "anchor_id": row["anchor_id"],
                    "pool_sha256": row["pool_sha256"], "role": role, "action": action, **score}
                scored_targets += score["action_tokens_including_eos"]
            for start in range(0, len(actions), 2):
                batch = [scores[a] for a in actions[start:start + 2]]
                require(len({s["prompt_ids_sha256"] for s in batch}) == 1
                        and len({s["prompt_tokens"] for s in batch}) == 1, "Score batch prompts differ")
                scored_forward += len(batch) * (batch[0]["prompt_tokens"]
                                               + max(s["action_tokens_including_eos"] for s in batch))
                expected_calls += 1
    events = read_rows(directory / "events.jsonl")
    require(len(events) == len(expected_events), "Scoring events missing/duplicated")
    seen = set()
    for event in events:
        key = (event.get("role"), event.get("anchor_id"), event.get("action"))
        require(key not in seen and event == expected_events.get(key), "Scoring event differs from selection scores")
        seen.add(key)
    ledger = read_json(directory / "score_ledger.json")
    require(ledger == summary["scoring"] and ledger.get("kind") == "recovery_score_ledger"
            and ledger.get("active_call") is None
            and ledger["calls_started"] == ledger["calls_completed"] == expected_calls
            and ledger["scored_target_tokens"] == scored_targets
            and ledger["forward_tokens"] == scored_forward, "Incomplete/inconsistent score ledger")
    for arm in ARMS:
        require(sha_file(recovery_paths[arm]) == sha_file(directory / (arm + ".jsonl"))
                == summary["label_sha256"][arm], "Recovery label bytes differ from complete selection")
        require([r["anchor_id"] for r in datasets[arm]] == [r["anchor_id"] for r in eligible],
                "Recovery labels differ from eligible selection order")
        for label, row in zip(datasets[arm], eligible):
            require(label["pool_sha256"] == row["pool_sha256"]
                    and label["selection_sha256"] == row["artifact_sha256"]
                    and label["action"] == row["labels"][arm]
                    and label["anchor_id"] == digest([label["gamefile"], label["seed"], label["after_actions"]]),
                    "Recovery label identity/target differs from selection")
    # Detect concurrent additions/changes before returning the binding. Training
    # repeats this entire function, and joint-audit equality checks this inventory.
    require(inventory_tree(directory) == inventory, "Selection artifacts changed during audit")
    return {s["anchor_id"]: s for s in eligible}, inventory


def verify_score_sft(label, selected, encoded):
    """Compare actual SFT encoding with the frozen adapted-reference action score."""
    require(selected.get("eligible") is True and selected.get("complete") is True
            and selected["anchor_id"] == label["anchor_id"]
            and selected["pool_sha256"] == label["pool_sha256"]
            and selected["artifact_sha256"] == label["selection_sha256"]
            and selected["labels"][label["arm"]] == label["action"], "Score/SFT selection identity differs")
    score = selected["reference_scores"][label["action"]]
    actual = {"prompt_ids_sha256": encoded["prompt_ids_sha256"],
              "action_ids_sha256": encoded["target_ids_sha256"],
              "prompt_tokens": encoded["prompt_tokens"],
              "action_tokens_including_eos": encoded["target_tokens"],
              "context_sha256": digest({k: causal_row(label)[k] for k in CONTEXT_FIELDS})}
    require(all(score.get(k) == v for k, v in actual.items()), "Score/SFT token/context mismatch")


def encoded_record(tokenizer, row):
    encoded = supervised_example(tokenizer, row, 2048, 64)
    n = encoded["target_tokens"]
    prompt, target = encoded["input_ids"][:-n], encoded["input_ids"][-n:]
    require(encoded["labels"] == [-100] * len(prompt) + target, "Incorrect action-only label masking")
    return encoded, {"prompt_tokens": len(prompt), "target_tokens": n,
                     "input_tokens": len(encoded["input_ids"]), "prompt_ids_sha256": digest(prompt),
                     "target_ids_sha256": digest(target), "input_ids_sha256": digest(encoded["input_ids"]),
                     "labels_sha256": digest(encoded["labels"])}


def token_audit(tokenizer, bc, recovery, anchors, provenance, selections):
    bc_pairs = [encoded_record(tokenizer, row) for row in bc]
    pairs = {arm: [encoded_record(tokenizer, causal_row(row)) for row in recovery[arm]] for arm in ARMS}
    for arm in ARMS:
        for label, (_, encoded) in zip(recovery[arm], pairs[arm]):
            verify_score_sft(label, selections[label["anchor_id"]], encoded)
    schedule = exact_schedule(len(bc), len(anchors))
    for i in range(len(anchors)):
        require(len({pairs[arm][i][1]["prompt_ids_sha256"] for arm in ARMS}) == 1,
                "Cross-arm encoded prompt mismatch")
    arms = {}
    for arm in ARMS:
        records = {"bc": [v[1] for v in bc_pairs], "recovery": [v[1] for v in pairs[arm]]}
        exposures = [{**slot, "anchor_id": anchors[slot["index"]] if slot["source"] == "recovery" else None,
                      **records[slot["source"]][slot["index"]]} for slot in schedule]
        totals = {field: sum(e[field] for e in exposures)
                  for field in ("target_tokens", "prompt_tokens", "input_tokens")}
        require(totals["target_tokens"] <= SETTINGS["max_target_tokens"], "Planned target-token budget exceeded")
        arms[arm] = {"examples": records, "scheduled_examples": exposures, "totals": totals,
                     "component_exposures": dict(Counter(e["source"] for e in exposures)),
                     "step_target_tokens": [sum(e["target_tokens"] for e in exposures[i:i + 8])
                                            for i in range(0, 800, 8)]}
    result = {"kind": "recovery_joint_training_token_audit_v1", "complete": True,
              "settings": SETTINGS, "provenance": provenance, "bc_examples": len(bc),
              "ordered_anchor_ids": anchors, "anchor_count": len(anchors),
              "ordered_context_sha256": [digest({k: row[k] for k in CONTEXT_FIELDS})
                                         for row in recovery[ARMS[0]]],
              "schedule": schedule, "schedule_sha256": digest(schedule), "arms": arms,
              "teacher_fields_used": False, "same_prompt_all_arms": True,
              "score_sft_token_context_mismatches": 0,
              "score_sft_selected_labels_checked": 3 * len(anchors),
              "interpretation": "Same updates and 400+400 exposures; targets/token counts may differ. No padding or resampling."}
    encoded = {arm: {"bc": [v[0] for v in bc_pairs], "recovery": [v[0] for v in pairs[arm]]} for arm in ARMS}
    return result, encoded


def resolve_recovery_paths(args, saved=None):
    paths = {args.arm: str(Path(args.recovery_data).resolve())}
    if saved is not None:
        for arm, info in saved["provenance"]["recovery_data"].items():
            if arm != args.arm:
                paths[arm] = info["path"]
        require(not args.peer_recovery_data, "Training gets peer paths from frozen token audit")
    else:
        for item in args.peer_recovery_data:
            arm, separator, path = item.partition("=")
            require(separator and arm in ARMS and arm not in paths, "Invalid/duplicate peer ARM=PATH")
            paths[arm] = str(Path(path).resolve())
    require(set(paths) == set(ARMS), "Audit-only requires both --peer-recovery-data ARM=PATH peers")
    return {arm: paths[arm] for arm in ARMS}


def owned_output(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()) and not path.is_relative_to(INITIAL_ADAPTER.resolve()),
            "Output must be in project and outside initial adapter")
    require(not path.exists(), "Exclusive-create output already exists")
    return path


def run_training(args, tokenizer, joint, encoded, started):
    # Only called after the guard, initial byte verification, and complete audit match.
    import torch
    from transformers import AutoModelForCausalLM, set_seed
    from peft import PeftModel
    output = owned_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    completed = optimized_tokens = consumed_tokens = 0
    consumed_examples = 0
    by_component = Counter()
    stop_reason, error = "incomplete", None
    final_inventory_hash = final_files_hash = None
    audit_hash = sha_file(args.token_audit)
    state = {"settings": SETTINGS, "arm": args.arm, "protocol": PROTOCOL_VERSION,
             "model": str(Path(args.model).resolve()), "initial_adapter": str(Path(args.initial_adapter).resolve()),
             "token_audit_path": str(Path(args.token_audit).resolve()), "token_audit_sha256": audit_hash,
             "schedule_sha256": joint["schedule_sha256"], "provenance": joint["provenance"]}
    write_new(output / "config.json", state)
    write_new(output / "schedule_audit.json", {"schedule_sha256": joint["schedule_sha256"],
                                               **joint["arms"][args.arm]})

    def check_time():
        if time.monotonic() - started >= SETTINGS["max_wall_seconds"]:
            raise TimeoutError("training_stage_wall_budget")

    with (output / "metrics.jsonl").open("x", encoding="utf-8", buffering=1) as metrics, \
         (output / "example_exposures.jsonl").open("x", encoding="utf-8", buffering=1) as exposures:
        try:
            check_time()
            set_seed(42)
            base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                         local_files_only=True, attn_implementation="sdpa").to(args.device)
            model = PeftModel.from_pretrained(base, args.initial_adapter, is_trainable=True)
            require(set(model.peft_config) == {"default"}, "Expected one continued adapter, no stack")
            model.config.use_cache = False
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            require(trainable and all("lora_" in name for name, _ in trainable), "Unexpected trainable base/non-LoRA weights")
            write_new(output / "trainable_parameters.json", {"count": sum(p.numel() for _, p in trainable),
                       "parameters": [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype)} for name, p in trainable]})
            optimizer = torch.optim.AdamW((p for _, p in trainable), lr=2e-4, weight_decay=0.0)
            schedule = joint["arms"][args.arm]["scheduled_examples"]
            for step in range(1, 101):
                check_time()
                window = schedule[(step - 1) * 8:step * 8]
                denominator = sum(s["target_tokens"] for s in window)
                require(optimized_tokens + denominator <= SETTINGS["max_target_tokens"], "Target budget exceeded")
                optimizer.zero_grad(set_to_none=True)
                window_nll = 0.0
                for slot in window:
                    check_time()
                    ex = encoded[args.arm][slot["source"]][slot["index"]]
                    require(ex["target_tokens"] == slot["target_tokens"]
                            and digest(ex["input_ids"]) == slot["input_ids_sha256"], "Scheduled example changed")
                    # A started forward is separately visible if a call fails/OOMs.
                    exposures.write(json.dumps({"event": "forward_started", **slot}) + "\n")
                    batch = {k: v.to(args.device) for k, v in collate_examples([ex], tokenizer.pad_token_id, torch).items()}
                    loss = model(**batch).loss
                    require(bool(torch.isfinite(loss)), "Nonfinite training loss")
                    (loss * (ex["target_tokens"] / denominator)).backward()
                    nll = float(loss.detach().item())
                    window_nll += nll * ex["target_tokens"]
                    consumed_examples += 1
                    consumed_tokens += ex["target_tokens"]
                    by_component[slot["source"]] += 1
                    exposures.write(json.dumps({"event": "backward_completed", **slot, "action_token_nll": nll}) + "\n")
                check_time()
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
                require(bool(torch.isfinite(norm)), "Nonfinite gradient norm")
                optimizer.step()
                completed = step
                optimized_tokens += denominator
                metric = {"optimizer_step": step, "action_token_nll": window_nll / denominator,
                          "window_target_tokens": denominator, "target_tokens": optimized_tokens,
                          "sampled_examples": consumed_examples, "component_exposures": dict(by_component),
                          "elapsed_seconds": time.monotonic() - started}
                metrics.write(json.dumps(metric) + "\n")
                print(json.dumps(metric), flush=True)
            require(completed == 100 and dict(by_component) == {"bc": 400, "recovery": 400}
                    and optimized_tokens == joint["arms"][args.arm]["totals"]["target_tokens"], "Final audit exposure mismatch")
            check_time()
            final = output / "final"
            final.mkdir(exist_ok=False)
            model.save_pretrained(final)
            tokenizer.save_pretrained(final)
            write_new(final / "training_state.json", {"optimizer_steps": completed,
                      "initial_adapter_optimizer_steps": 100, "cumulative_optimizer_steps": 200,
                      "target_tokens": optimized_tokens, "sampled_examples": consumed_examples,
                      "component_exposures": dict(by_component), "protocol": PROTOCOL_VERSION,
                      "token_audit_sha256": audit_hash, "schedule_sha256": joint["schedule_sha256"],
                      "initial_adapter_inventory_sha256": joint["provenance"]["initial_adapter"]["files_sha256"],
                      "optimizer_state_resumed": False, "elapsed_seconds": time.monotonic() - started})
            check_time()
            final_inventory = inventory_tree(final)
            write_new(output / "final_adapter_inventory.json", final_inventory)
            final_inventory_hash = sha_file(output / "final_adapter_inventory.json")
            final_files_hash = final_inventory["files_sha256"]
            check_time()
            stop_reason = "max_steps"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            stop_reason = "technical_or_budget_failure"
        summary = {"optimizer_steps": completed, "requested_steps": 100,
                   "matched_step_budget_complete": completed == 100 and error is None,
                   "complete": completed == 100 and error is None, "stop_reason": stop_reason,
                   "target_tokens": optimized_tokens, "actual_backward_target_tokens": consumed_tokens,
                   "sampled_examples": consumed_examples, "component_exposures": dict(by_component),
                   "expected_target_tokens": joint["arms"][args.arm]["totals"]["target_tokens"],
                   "training_examples": {"bc": joint["bc_examples"], "recovery": joint["anchor_count"]},
                   "token_audit_sha256": audit_hash, "schedule_sha256": joint["schedule_sha256"],
                   "final_adapter_inventory_sha256": final_inventory_hash,
                   "final_adapter_files_sha256": final_files_hash,
                   "technical_errors": [] if error is None else [error], "elapsed_seconds": time.monotonic() - started,
                   "note": "Only a complete final step100 is eligible for later locking. Training NLL is not task success."}
        metrics.write(json.dumps({"summary": summary}) + "\n")
    write_new(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["complete"] else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--initial-adapter", required=True)
    p.add_argument("--bc-data", required=True)
    p.add_argument("--recovery-data", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--peer-recovery-data", action="append", default=[], metavar="ARM=PATH")
    p.add_argument("--token-audit", required=True)
    p.add_argument("--selection-dir", required=True,
                   help="Complete frozen score-stage directory; required for both audit and training")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--output")
    p.add_argument("--device", default="cuda:0")
    return p


def main():
    args = parser().parse_args()
    started = time.monotonic()
    require_gpu_guard()
    require(args.device == "cuda:0", "Use the launcher's sole logical cuda:0")
    selection_root = Path(args.selection_dir).resolve()
    require(not Path(args.token_audit).resolve().is_relative_to(selection_root)
            and (args.output is None or not Path(args.output).resolve().is_relative_to(selection_root)),
            "Token audit/training outputs must be outside the immutable selection directory")
    if args.audit_only:
        require(args.output is None, "Audit-only writes only --token-audit")
        audit_path = owned_output(args.token_audit)
        saved = None
    else:
        require(args.output is not None, "Training requires exclusive --output")
        owned_output(args.output)
        saved = read_json(args.token_audit)
        require(saved.get("kind") == "recovery_joint_training_token_audit_v1" and saved.get("complete") is True,
                "Complete joint token audit required before training")
    paths = resolve_recovery_paths(args, saved)
    bc, recovery, anchors = prepare_raw(args, paths)
    selections, selection_inventory = selection_artifacts(args.selection_dir, paths, recovery)
    provenance = input_provenance(args, paths)
    provenance["selection_artifacts"] = selection_inventory
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    joint, encoded = token_audit(tokenizer, bc, recovery, anchors, provenance, selections)
    require(time.monotonic() - started < SETTINGS["max_wall_seconds"], "Audit/preflight wall budget")
    if args.audit_only:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        write_new(audit_path, joint)
        print(json.dumps({"complete": True, "token_audit_sha256": sha_file(audit_path),
                          "schedule_sha256": joint["schedule_sha256"], "model_calls": 0,
                          "totals": {arm: joint["arms"][arm]["totals"] for arm in ARMS}}), flush=True)
        return 0
    require(joint == saved, "Recomputed data/adapter/source/token/schedule audit differs")
    return run_training(args, tokenizer, joint, encoded, started)


if __name__ == "__main__":
    raise SystemExit(main())
