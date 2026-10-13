"""Evaluate held-out ALFWorld success through actual online simulator rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .policy import PROTOCOL_VERSION, HFCausalPolicy


def read_games(path: str) -> list[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        obj = obj["games"]
    games = [str(x["gamefile"] if isinstance(x, dict) else x) for x in obj]
    if not games or len(games) != len(set(games)):
        raise ValueError("Evaluation manifest must have unique, nonempty games")
    return games


def rollout(env, policy, max_steps=100, selection="generate", deadline=None,
            max_scored_tokens=10000000, task_override=None, max_generated_tokens=100000):
    """No expert trajectory is accepted here; every successor comes from env.step."""
    state = env.reset()
    task = task_override or getattr(env, "task", None) or state.observation
    history = []
    transitions = []
    success = bool(state.won)
    stop_reason = "environment_done" if state.done else "max_steps"
    for step in range(max_steps):
        if state.done:
            break
        if deadline is not None and time.monotonic() >= deadline:
            stop_reason = "wall_time_budget"
            break
        if policy.scored_tokens >= max_scored_tokens:
            stop_reason = "scored_token_budget"
            break
        if getattr(policy, "generated_tokens", 0) >= max_generated_tokens:
            stop_reason = "generated_token_budget"
            break
        if not state.admissible_commands:
            stop_reason = "no_admissible_commands"
            break
        action, details = policy.choose_action(task, history, state.observation,
                                               state.admissible_commands, selection)
        before_observation = state.observation
        before_commands = list(state.admissible_commands)
        state = env.step(action)
        transitions.append({"step": step, "observation": before_observation,
                            "admissible_commands": before_commands, "action": action,
                            "next_observation": state.observation, "reward": state.reward,
                            "done": state.done, "won": state.won, "policy": details})
        history.append({"observation": before_observation, "action": action})
        success = success or bool(state.won)
        if state.done:
            stop_reason = "environment_done"
    return {"success": success, "environment_steps": len(transitions),
            "stop_reason": stop_reason, "transitions": transitions,
            "terminal": bool(state.done), "final_observation": state.observation}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--games-json", required=True, help="Frozen held-out manifest: list or {games: list}")
    p.add_argument("--output", required=True)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-wall-seconds", type=float, default=7200)
    p.add_argument("--max-scored-tokens", type=int, default=10000000)
    p.add_argument("--max-generated-tokens", type=int, default=100000)
    p.add_argument("--max-context-tokens", type=int, default=2048)
    p.add_argument("--max-action-tokens", type=int, default=64)
    p.add_argument("--score-batch-size", type=int, default=2)
    p.add_argument("--selection", choices=["generate", "sum_logprob", "mean_logprob"], default="generate",
                   help="Greedy free-form generation by default; command-list likelihood selection is an ablation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    return p


def main():
    args = parser().parse_args()
    if min(args.max_steps, args.max_wall_seconds, args.max_scored_tokens, args.max_generated_tokens) <= 0:
        raise ValueError("Budgets must be positive")
    started = time.monotonic()
    games = read_games(args.games_json)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {"protocol": PROTOCOL_VERSION,
                           "manifest_sha256": hashlib.sha256(Path(args.games_json).read_bytes()).hexdigest(),
                           "evaluation_type": "online_environment_rollout",
                           "candidate_scoring": "command_and_eos_conditional_logprob"}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    policy = HFCausalPolicy(args.model, args.adapter, device=args.device,
                            max_context_tokens=args.max_context_tokens,
                            max_action_tokens=args.max_action_tokens,
                            score_batch_size=args.score_batch_size)
    from .alfworld_env import AlfworldReplayEnv
    records = []
    with (out / "episodes.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for gamefile in games:
            if (time.monotonic() - started >= args.max_wall_seconds or
                    policy.scored_tokens >= args.max_scored_tokens or
                    policy.generated_tokens >= args.max_generated_tokens):
                break
            env = None
            try:
                env = AlfworldReplayEnv(gamefile, max_steps=args.max_steps, expert_type=None, seed=args.seed)
                episode = rollout(env, policy, args.max_steps, args.selection,
                                  started + args.max_wall_seconds, args.max_scored_tokens,
                                  max_generated_tokens=args.max_generated_tokens)
            except Exception as exc:
                # Technical errors remain visible and in the attempted denominator.
                episode = {"success": False, "error": repr(exc), "stop_reason": "error",
                           "environment_steps": getattr(env, "interactions", 0)}
            finally:
                if env is not None and hasattr(env, "close"):
                    env.close()
            episode.update(gamefile=gamefile, seed=args.seed,
                           elapsed_seconds=time.monotonic() - started)
            records.append(episode)
            log.write(json.dumps(episode, ensure_ascii=False) + "\n")
            print(json.dumps({k: v for k, v in episode.items() if k != "transitions"}), flush=True)
    successes = sum(x["success"] for x in records)
    errors = sum("error" in x for x in records)
    summary = {"planned_episodes": len(games), "attempted_episodes": len(records),
               "successes": successes, "technical_errors": errors,
               "success_rate_attempted": successes / len(records) if records else None,
               "success_rate_planned_lower_bound": successes / len(games),
               "complete": len(records) == len(games) and errors == 0 and
                           not any(x["stop_reason"] in ("wall_time_budget", "scored_token_budget",
                                                       "generated_token_budget") for x in records),
               "environment_steps": sum(x.get("environment_steps") or 0 for x in records),
               "scored_tokens": policy.scored_tokens, "forward_tokens": policy.forward_tokens,
               "generated_tokens": policy.generated_tokens,
               "elapsed_seconds": time.monotonic() - started, "protocol": PROTOCOL_VERSION}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
