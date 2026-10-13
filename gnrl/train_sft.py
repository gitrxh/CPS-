"""Action-only LoRA SFT with an explicit optimizer-step budget."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from .policy import PROTOCOL_VERSION, require_gpu_guard, supervised_example


def load_examples(path: str) -> list[dict]:
    """Accept causal action rows or simulator-collected trajectory JSONL."""
    examples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "transitions" not in row:
            required = {"task", "observation", "admissible_commands", "action"}
            if not required <= row.keys():
                raise ValueError("Unexpected diagnostic/nontraining row in SFT data: expected causal action or trajectory")
            examples.append({k: row[k] for k in ("task", "history", "observation",
                                                 "admissible_commands", "action") if k in row})
            continue
        if not row.get("success", False):
            raise ValueError("SFT trajectory is not verified successful")
        if row.get("replay_verified") is not True:
            raise ValueError("SFT trajectory requires replay_verified:true from actual simulator replay")
        history = []
        task = row.get("task") or row["initial_observation"]
        for transition in row["transitions"]:
            examples.append({"task": task, "history": list(history),
                             "observation": transition["observation"],
                             "admissible_commands": transition["admissible_commands"],
                             "action": transition["action"]})
            history.append({"observation": transition["observation"], "action": transition["action"]})
    if not examples:
        raise ValueError("Training dataset contains no examples")
    for row in examples:
        if row["action"] not in row["admissible_commands"]:
            raise ValueError("Training target is not admissible")
    return examples


def collate_examples(examples, pad_token_id, torch):
    width = max(len(x["input_ids"]) for x in examples)
    ids, labels, masks = [], [], []
    for ex in examples:
        n = len(ex["input_ids"])
        ids.append(ex["input_ids"] + [pad_token_id] * (width - n))
        labels.append(ex["labels"] + [-100] * (width - n))
        masks.append([1] * n + [0] * (width - n))
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-context-tokens", type=int, default=2048)
    p.add_argument("--max-action-tokens", type=int, default=64)
    p.add_argument("--max-target-tokens", type=int, default=2000000)
    p.add_argument("--max-wall-seconds", type=float, default=7200)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    return p


def main():
    args = parser().parse_args()
    for name in ("max_steps", "batch_size", "gradient_accumulation", "max_target_tokens",
                 "max_wall_seconds", "save_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    require_gpu_guard()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import LoraConfig, get_peft_model
    started = time.monotonic()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {"protocol": PROTOCOL_VERSION,
                           "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                           "loss_scope": "action_tokens_and_eos_only"}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = load_examples(args.data)
    dataset = [supervised_example(tokenizer, row, args.max_context_tokens,
                                  args.max_action_tokens) for row in raw]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to(args.device)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=0.0)
    rng = random.Random(args.seed)
    order = list(range(len(dataset)))
    rng.shuffle(order)
    cursor = 0
    completed = 0
    seen_tokens = 0
    stop_reason = "max_steps"
    log = (out / "metrics.jsonl").open("w", encoding="utf-8", buffering=1)

    def save_checkpoint(name):
        checkpoint = out / name
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        (checkpoint / "training_state.json").write_text(json.dumps(
            {"optimizer_steps": completed, "target_tokens": seen_tokens,
             "elapsed_seconds": time.monotonic() - started, "protocol": PROTOCOL_VERSION}, indent=2))

    for step in range(1, args.max_steps + 1):
        if time.monotonic() - started >= args.max_wall_seconds:
            stop_reason = "wall_time_budget"
            break
        # Construct the whole accumulation window so token weighting is exact,
        # including unequal action lengths and a small dataset crossing epochs.
        microbatches = []
        for _ in range(args.gradient_accumulation):
            batch = []
            for _ in range(args.batch_size):
                if cursor == len(order):
                    rng.shuffle(order)
                    cursor = 0
                batch.append(dataset[order[cursor]])
                cursor += 1
            microbatches.append(batch)
        window_tokens = sum(ex["target_tokens"] for batch in microbatches for ex in batch)
        if seen_tokens + window_tokens > args.max_target_tokens:
            stop_reason = "target_token_budget"
            break
        optimizer.zero_grad(set_to_none=True)
        window_nll = 0.0
        timed_out = False
        for examples in microbatches:
            if time.monotonic() - started >= args.max_wall_seconds:
                timed_out = True
                break
            n = sum(ex["target_tokens"] for ex in examples)
            batch = {k: v.to(args.device) for k, v in
                     collate_examples(examples, tokenizer.pad_token_id, torch).items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            (loss * (n / window_tokens)).backward()
            window_nll += float(loss.detach().item()) * n
        if timed_out:
            optimizer.zero_grad(set_to_none=True)
            stop_reason = "wall_time_budget_partial_window_discarded"
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed = step
        seen_tokens += window_tokens
        metric = {"optimizer_step": step, "action_token_nll": window_nll / window_tokens,
                  "target_tokens": seen_tokens, "examples_per_step": args.batch_size * args.gradient_accumulation,
                  "elapsed_seconds": time.monotonic() - started}
        log.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)
        if step % args.save_every == 0:
            save_checkpoint(f"checkpoint-{step}")
    save_checkpoint("final")
    summary = {"optimizer_steps": completed, "requested_steps": args.max_steps,
               "matched_step_budget_complete": completed == args.max_steps,
               "stop_reason": stop_reason, "target_tokens": seen_tokens,
               "training_examples": len(dataset), "elapsed_seconds": time.monotonic() - started,
               "note": "Training NLL is not held-out task success; run gnrl.evaluate."}
    log.write(json.dumps({"summary": summary}) + "\n")
    log.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
