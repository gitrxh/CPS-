"""Offline, tokenizer-only audit of the frozen successful-path training schedule.

No model weights, simulator, optimization, or GPU access. The queue calls audit()
only after complete selection and before the first training stage.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .policy import encode_action, encode_prompt, messages, supervised_example

ARMS = ("canonical", "uniform", "reference")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def sample_indices(size, steps=100, accumulation=8, seed=42):
    if size < 1:
        raise ValueError("Empty training examples")
    rng = random.Random(seed)
    order = list(range(size))
    rng.shuffle(order)
    result, cursor = [], 0
    for _ in range(steps * accumulation):
        if cursor == len(order):
            rng.shuffle(order)
            cursor = 0
        result.append(order[cursor])
        cursor += 1
    return result


def audit(tokenizer, datasets, selected_ids, scored_tokens):
    """scored_tokens maps (gamefile,candidate_id,step) to encoded-token hashes.

    Dataset rows are original canonical or selected real trajectories. Token
    audit uses exactly the trainer's causal allowlist; teacher fields never
    cross into the tokenizer. Audit errors abort before any optimizer step.
    """
    examples, tasks = {}, {}
    keys = None
    for arm in ARMS:
        current, task_rows, arm_keys = [], [], []
        for trajectory in datasets[arm]:
            game = trajectory["gamefile"]
            cid = selected_ids[arm][game]
            history, records = [], []
            for step, transition in enumerate(trajectory["transitions"]):
                row = {"task": trajectory.get("task") or trajectory["initial_observation"],
                       "history": list(history), "observation": transition["observation"],
                       "admissible_commands": transition["admissible_commands"], "action": transition["action"]}
                if row["action"] not in row["admissible_commands"]:
                    raise ValueError("Audit target is inadmissible")
                prompt = encode_prompt(tokenizer, row["task"], history, row["observation"], row["admissible_commands"], 2048, 64)
                action = encode_action(tokenizer, row["action"], 64)
                encoded = supervised_example(tokenizer, row, 2048, 64)
                if encoded != {"input_ids": prompt + action, "labels": [-100] * len(prompt) + action,
                               "target_tokens": len(action)}:
                    raise ValueError("SFT and scoring encodings differ")
                hashes = {"prompt_ids_sha256": digest(prompt), "action_ids_sha256": digest(action)}
                scored = scored_tokens[(game, cid, step)]
                if any(scored.get(k) != v for k, v in hashes.items()):
                    raise ValueError("Selected scoring/SFT token hashes differ")
                if scored.get("prompt_tokens") != len(prompt) or scored.get("action_tokens_including_eos") != len(action):
                    raise ValueError("Selected scoring/SFT token counts differ")
                dropped = 0
                while dropped <= len(history):
                    candidate = tokenizer.apply_chat_template(messages(row["task"], history[dropped:], row["observation"], row["admissible_commands"]), tokenize=True, add_generation_prompt=True)
                    if len(candidate) + 64 <= 2048:
                        if list(candidate) != prompt:
                            raise ValueError("Whole-turn truncation differs")
                        break
                    dropped += 1
                record = {"gamefile": game, "step": step, "candidate_id": cid, **hashes,
                          "prompt_tokens": len(prompt), "target_tokens": len(action),
                          "input_tokens": len(prompt) + len(action), "history_turns_dropped": dropped,
                          "causal_fields_sha256": digest(row), "target_action_sha256": digest(row["action"])}
                current.append(record)
                records.append(record)
                arm_keys.append([game, step])
                history.append({"observation": row["observation"], "action": row["action"]})
            task_rows.append({"gamefile": game, "candidate_id": cid, "action_count": len(records),
                              "target_tokens": sum(x["target_tokens"] for x in records),
                              "prompt_tokens": sum(x["prompt_tokens"] for x in records),
                              "unchanged_source_data_sha256": digest(trajectory)})
        if keys is not None and arm_keys != keys:
            raise ValueError("Arms do not have identical ordered gamefile/step example keys")
        keys, examples[arm], tasks[arm] = arm_keys, current, task_rows
    sampled = sample_indices(len(keys))
    totals = {arm: {"sampled_target_tokens": sum(examples[arm][i]["target_tokens"] for i in sampled),
                    "sampled_input_tokens": sum(examples[arm][i]["input_tokens"] for i in sampled),
                    "sampled_prompt_tokens": sum(examples[arm][i]["prompt_tokens"] for i in sampled)} for arm in ARMS}
    comparisons = {}
    for other in ("canonical", "uniform"):
        changed = [i for i, (a, b) in enumerate(zip(examples["reference"], examples[other])) if a["prompt_ids_sha256"] != b["prompt_ids_sha256"]]
        comparisons["reference_vs_" + other] = {
            "changed_causal_inputs": len(changed),
            "changed_sampled_inputs": sum(i in set(changed) for i in sampled),
            "changed_target_actions": sum(a["target_action_sha256"] != b["target_action_sha256"] for a, b in zip(examples["reference"], examples[other])),
            "changed_trajectories": sum(selected_ids["reference"][g] != selected_ids[other][g] for g in selected_ids["reference"])}
    return {"kind": "successful_path_token_exposure_audit", "complete": True, "examples_per_arm": len(keys),
            "sampled_indices": sampled, "ordered_example_keys": keys, "exposures": totals,
            "tasks": tasks, "examples": examples, "comparisons": comparisons,
            "settings": {"seed": 42, "optimizer_steps": 100, "batch_size": 1, "gradient_accumulation": 8,
                         "max_context_tokens": 2048, "max_action_tokens": 64},
            "score_sft_token_mismatches": 0, "teacher_fields_used": False,
            "interpretation": "Same updates and index schedule; measured token totals are not forced equal."}


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", required=True, type=Path)
    parser.add_argument("--pools", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Tokenizer audit requires GPUs disabled")
    from transformers import AutoTokenizer
    from .successful_path_selection_v1 import validate_selection_artifacts
    def load_rows(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pools = load_rows(args.pools)
    receipt = validate_selection_artifacts(pools, args.selection_dir)
    records = receipt["records"]
    selected = {arm: {r["gamefile"]: r["candidates_by_arm"][arm]["candidate_id"] for r in records} for arm in ARMS}
    ledger = load_rows(args.selection_dir / "score_ledger.jsonl")
    scores = {(r["gamefile"], r["candidate_id"], r["step"]): r for r in ledger}
    if len(scores) != len(ledger):
        raise ValueError("Duplicate scoring token ledger entries")
    files = {"canonical": "canonical_baseline.jsonl", "uniform": "uniform_valid.jsonl", "reference": "learner_selected.jsonl"}
    datasets = {arm: load_rows(args.selection_dir / name) for arm, name in files.items()}
    result = audit(AutoTokenizer.from_pretrained(args.model, local_files_only=True), datasets, selected, scores)
    for arm in ARMS:
        for task, pool in zip(result["tasks"][arm], pools):
            if task["action_count"] != pool["canonical_length_cost"][0]:
                raise ValueError("Audit action count differs from canonical")
            task["audited_operator_cost"] = pool["canonical_length_cost"][1]
        if result["exposures"][arm]["sampled_target_tokens"] > 2000000:
            raise ValueError("Predicted training target-token budget exceeded")
    result["dataset_bytes_sha256"] = {arm: hashlib.sha256((args.selection_dir / name).read_bytes()).hexdigest() for arm, name in files.items()}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
