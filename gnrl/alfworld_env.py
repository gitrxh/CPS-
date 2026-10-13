"""Real ALFWorld/TextWorld adapter. Imports simulator dependencies lazily.

PddlEnv owns native Fast Downward state: Python deepcopy is not a validated
checkpoint. Restoration therefore replays the exact prefix from a seeded reset.
Canonical state uses raw entity IDs, never observations as a proxy for state.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

TASK_TYPES = {
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
}


def information_only_action(action: str) -> bool:
    """Official alfred.twl2 observation commands, independent of fact changes.

    ALFWorld PDDL may add `checked` for these commands, so unchanged facts are
    not a sufficient test. `use` is deliberately excluded: it toggles objects.
    Only official templates are classified; unknown/invalid verbs are not aliases.
    """
    normalized = " ".join(str(action).strip().lower().split())
    return normalized in {"look", "inventory", "help"} or normalized.startswith("examine ")


def list_games(data_root: str | Path, split: str = "train") -> list[Path]:
    """Use official split directories and ALFWorld's supported/solvable filters."""
    if split not in {"train", "valid_seen", "valid_unseen"}:
        raise ValueError(f"Unsupported official split: {split}")
    root = Path(data_root).expanduser().resolve()
    root = root if root.name == split else root / split
    if not root.is_dir():
        raise FileNotFoundError(f"Official split directory missing: {root}")
    games = []
    for game in sorted(root.rglob("game.tw-pddl")):
        if "movable" in str(game.parent) or "Sliced" in str(game.parent):
            continue
        trajectory = game.parent / "traj_data.json"
        if not trajectory.is_file():
            continue
        metadata = json.loads(trajectory.read_text(encoding="utf-8"))
        if metadata.get("task_type") not in TASK_TYPES:
            continue
        if json.loads(game.read_text(encoding="utf-8")).get("solvable") is True:
            games.append(game)
    return games


def canonical_facts(facts: Iterable[Any]) -> list[list[Any]]:
    """Keep predicate, argument ID, and type; do not merge numbered objects."""
    return sorted([
        [str(f.name), [[str(v.name), str(v.type)] for v in f.arguments]]
        for f in facts
    ], key=lambda x: json.dumps(x, sort_keys=True))


def state_digest(state: dict) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class StepResult:
    observation: str
    admissible_commands: list[str]
    expert_action: str | None
    reward: float
    done: bool
    won: bool
    canonical_state: dict

    def to_dict(self) -> dict:
        return asdict(self)


class ReplayMismatch(RuntimeError):
    pass


class AlfworldReplayEnv:
    """One CPU TextWorld game; rewards are the backend's cumulative scores.

    `done` includes our explicit interaction budget. `terminated` and `truncated`
    are separately available in canonical_state. No future/expert information
    should be passed to a student prompt.
    """
    restore_mode = "seeded_reset_prefix_replay"

    def __init__(self, gamefile: str | Path, max_steps: int = 100,
                 expert_type: str | None = "handcoded", seed: int = 0):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos, AlfredExpert
        self.gamefile = str(Path(gamefile).resolve())
        self.max_steps, self.seed = max_steps, seed
        self.expert_type = expert_type
        self._rng = random.Random(seed)
        self.interactions = 0
        self.resets = 0
        wrappers = [lambda env: AlfredDemangler(env, shuffle=False), AlfredInfos]
        if expert_type is not None:
            wrappers.append(lambda env: AlfredExpert(env=env, expert_type=expert_type))
        infos = textworld.EnvInfos(facts=True, won=True, lost=True, moves=True,
                                  admissible_commands=True, extras=["gamefile", "expert_plan"])
        self.env = textworld.start(self.gamefile, request_infos=infos, wrappers=wrappers)
        self.current: StepResult | None = None
        self.actions: list[str] = []

    def _call(self, method, *args):
        # ALFWorld handcoded expert and grammar use Python's global random module.
        previous = random.getstate()
        random.setstate(self._rng.getstate())
        try:
            return method(*args)
        finally:
            self._rng.setstate(random.getstate())
            random.setstate(previous)

    def _result(self, state, reward: float, terminated: bool) -> StepResult:
        # _facts is PddlEnv's true symbol state. Human-readable facts can alias IDs.
        if "_facts" not in state or "moves" not in state:
            raise RuntimeError("TextWorld backend did not expose raw _facts and moves; refusing observation equivalence")
        truncated = len(self.actions) >= self.max_steps and not terminated
        commands = sorted(set(state.get("admissible_commands", [])))
        canonical = {
            "facts": canonical_facts(state["_facts"]),
            "moves": int(state["moves"]), "elapsed_steps": len(self.actions),
            "max_steps": self.max_steps,
            "score": float(state.get("score") or 0), "reward": float(reward),
            "won": bool(state.get("won", False)), "lost": bool(state.get("lost", False)),
            "terminated": bool(terminated), "truncated": truncated,
            "done": bool(terminated or truncated), "admissible_commands": commands,
        }
        plan = state.get("extra.expert_plan") or []
        result = StepResult(str(state.get("feedback", "")), commands,
                            str(plan[0]) if plan else None, float(reward),
                            canonical["done"], canonical["won"], canonical)
        self.current = result
        return result

    def reset(self) -> StepResult:
        self._rng.seed(self.seed)
        self.actions = []
        self.resets += 1
        state = self._call(self.env.reset)
        return self._result(state, 0.0, bool(state.get("won") or state.get("lost")))

    def step(self, action: str) -> StepResult:
        if self.current is None:
            raise RuntimeError("Call reset before step")
        if self.current.done:
            raise RuntimeError("Cannot step a completed episode")
        self.interactions += 1
        state, reward, done = self._call(self.env.step, str(action))
        self.actions.append(str(action))
        return self._result(state, reward, done)

    def snapshot(self) -> dict:
        if self.current is None:
            raise RuntimeError("Call reset before snapshot")
        return json.loads(json.dumps(self.current.canonical_state))

    def restore(self, prefix: Iterable[str], expected: StepResult | dict | None = None) -> StepResult:
        result = self.reset()
        for action in list(prefix):
            result = self.step(action)
        if expected is not None:
            expected_state = expected.canonical_state if isinstance(expected, StepResult) else expected
            if result.canonical_state != expected_state:
                raise ReplayMismatch("Reset+prefix produced a different canonical state")
            if isinstance(expected, StepResult) and result.observation != expected.observation:
                raise ReplayMismatch("Reset+prefix produced a different observation")
        return result

    def close(self):
        self.env.close()


def transition_record(before: StepResult, action: str, after: StepResult, index: int) -> dict:
    return {
        "step": index, "observation": before.observation,
        "admissible_commands": before.admissible_commands,
        "expert_action": before.expert_action, "action": action,
        "next_observation": after.observation, "reward": after.reward,
        "done": after.done, "won": after.won,
        "state_before": before.canonical_state, "state_after": after.canonical_state,
    }


def replay_actions(env: AlfworldReplayEnv, actions: Iterable[str]) -> list[dict]:
    """Execute the entire candidate trajectory; never reuse stored observations."""
    result = env.reset()
    transitions = []
    for index, action in enumerate(actions):
        if result.done:
            raise ReplayMismatch("Candidate terminated before its requested action suffix ended")
        before = result
        result = env.step(action)
        transitions.append(transition_record(before, action, result, index))
    return transitions
