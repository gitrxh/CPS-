"""Shared causal prompt and HF policy. Importing this module does not import torch."""
from __future__ import annotations

import os
from typing import Any

PROTOCOL_VERSION = "alfworld-causal-v1"
SYSTEM_PROMPT = (
    "You are controlling an ALFWorld household agent. Complete the task using the "
    "observations and previous actions. Respond with exactly one command from the "
    "current admissible commands, with no explanation."
)


def require_gpu_guard() -> None:
    """Must run before importing torch/transformers, including for CPU experiments."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if os.environ.get("GNRL_GPU_GUARD") != "1" or not visible:
        raise RuntimeError("Launch model work through scripts/launch_gpu.py (physical GPUs 0–3 only)")
    if not all(part.startswith("GPU-") for part in visible.split(",")):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain GPU UUIDs set by launch_gpu.py")


def messages(task: str, history: list[dict], observation: str,
             admissible_commands: list[str]) -> list[dict[str, str]]:
    # Explicit allowlist: expert_action, states, and future observations cannot leak.
    lines = ["Task:\n" + task, "Previous interactions:"]
    for item in history:
        lines.append("Observation: " + str(item["observation"]))
        lines.append("Action: " + str(item["action"]))
    if not history:
        lines.append("(none)")
    lines.extend(["Current observation:\n" + observation,
                  "Current admissible commands:\n" + "\n".join(admissible_commands)])
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(lines)}]


def encode_prompt(tokenizer: Any, task: str, history: list[dict], observation: str,
                  admissible_commands: list[str], max_context_tokens: int = 2048,
                  max_action_tokens: int = 64) -> list[int]:
    """Drop oldest complete turns only; reserve identical action space for every arm."""
    recent = list(history)
    while True:
        ids = tokenizer.apply_chat_template(
            messages(task, recent, observation, admissible_commands),
            tokenize=True, add_generation_prompt=True,
        )
        if len(ids) + max_action_tokens <= max_context_tokens:
            return list(ids)
        if not recent:
            raise ValueError("Task/current observation/commands exceed context budget; increase it consistently")
        recent.pop(0)


def encode_action(tokenizer: Any, action: str, max_action_tokens: int = 64) -> list[int]:
    ids = tokenizer.encode(action, add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    ids = list(ids) + [tokenizer.eos_token_id]
    if len(ids) > max_action_tokens:
        raise ValueError(f"Action exceeds {max_action_tokens} tokens (including EOS): {action!r}")
    return ids


def supervised_example(tokenizer: Any, row: dict, max_context_tokens: int = 2048,
                       max_action_tokens: int = 64) -> dict:
    prompt = encode_prompt(tokenizer, row["task"], row.get("history", []),
                           row["observation"], row["admissible_commands"],
                           max_context_tokens, max_action_tokens)
    target = encode_action(tokenizer, row["action"], max_action_tokens)
    return {"input_ids": prompt + target, "labels": [-100] * len(prompt) + target,
            "target_tokens": len(target)}


class HFCausalPolicy:
    def __init__(self, model_path: str, adapter_path: str | None = None,
                 device: str = "cuda:0", dtype: str = "bfloat16",
                 max_context_tokens: int = 2048, max_action_tokens: int = 64,
                 score_batch_size: int = 2):
        require_gpu_guard()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.max_context_tokens = max_context_tokens
        self.max_action_tokens = max_action_tokens
        self.score_batch_size = score_batch_size
        if score_batch_size < 1:
            raise ValueError("score_batch_size must be positive")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype), local_files_only=True,
            attn_implementation="sdpa",
        ).to(device)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
        self.model.eval()
        self.scored_tokens = 0
        self.forward_tokens = 0
        self.generated_tokens = 0

    def prompt_ids(self, task, history, observation, admissible_commands):
        return encode_prompt(self.tokenizer, task, history, observation, admissible_commands,
                             self.max_context_tokens, self.max_action_tokens)

    def score_actions(self, task: str, history: list[dict], observation: str,
                      admissible_commands: list[str], actions: list[str]) -> list[dict]:
        """Exact sum log p(command tokens, EOS | causal prefix); no implicit normalization."""
        prompt = self.prompt_ids(task, history, observation, admissible_commands)
        return self.score_encoded_actions(prompt, actions)

    def score_encoded_actions(self, prompt: list[int], actions: list[str]) -> list[dict]:
        torch = self.torch
        result = []
        for offset in range(0, len(actions), self.score_batch_size):
            batch_actions = actions[offset:offset + self.score_batch_size]
            targets = [encode_action(self.tokenizer, a, self.max_action_tokens) for a in batch_actions]
            sequences = [prompt + t for t in targets]
            width = max(map(len, sequences))
            ids = torch.full((len(sequences), width), self.tokenizer.pad_token_id,
                             dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            for i, seq in enumerate(sequences):
                ids[i, :len(seq)] = torch.tensor(seq, device=self.device)
                mask[i, :len(seq)] = 1
            with torch.inference_mode():
                output = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
                for i, (action, target) in enumerate(zip(batch_actions, targets)):
                    logits = output.logits[i, len(prompt)-1:len(prompt)+len(target)-1].float()
                    token_logprobs = torch.log_softmax(logits, dim=-1).gather(
                        1, torch.tensor(target, device=self.device).unsqueeze(1)).squeeze(1)
                    total = float(token_logprobs.sum().item())
                    result.append({"action": action, "logprob": total,
                                   "token_count": len(target), "mean_logprob": total / len(target)})
            self.scored_tokens += sum(map(len, targets))
            self.forward_tokens += len(sequences) * width
        return result

    def choose_action(self, task: str, history: list[dict], observation: str,
                      admissible_commands: list[str], selection: str = "generate") -> tuple[str, dict]:
        if not admissible_commands:
            raise ValueError("No admissible commands")
        if selection == "generate":
            prompt = self.prompt_ids(task, history, observation, admissible_commands)
            ids = self.torch.tensor([prompt], dtype=self.torch.long, device=self.device)
            with self.torch.inference_mode():
                output = self.model.generate(
                    input_ids=ids, attention_mask=self.torch.ones_like(ids),
                    do_sample=False, max_new_tokens=self.max_action_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id, use_cache=True,
                )
            generated = output[0, len(prompt):]
            raw = self.tokenizer.decode(generated, skip_special_tokens=True)
            # No admissibility filtering, retries, extraction, or command repair.
            action = raw.strip()
            self.generated_tokens += len(generated)
            self.forward_tokens += len(prompt) + len(generated)
            return action, {"selection": selection, "raw_completion": raw,
                            "generated_tokens": len(generated),
                            "is_admissible": action in admissible_commands}
        scores = self.score_actions(task, history, observation, admissible_commands,
                                   sorted(set(admissible_commands)))
        field = {"sum_logprob": "logprob", "mean_logprob": "mean_logprob"}[selection]
        best = max(scores, key=lambda row: row[field])
        return best["action"], {"selection": selection, "chosen_score": best,
                                "candidate_count": len(scores)}
