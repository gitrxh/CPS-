"""CPS v2 trajectory-level projection training scaffold.

This keeps the original recovery SFT entry point unchanged and provides a
separate path for trajectory-projected data. The module is intentionally small:
the data builder is responsible for producing simulator-verified projected
trajectories, while training consumes only causal context and action targets.

No oracle state, future observation, reward, or planner trace is accepted as a
model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .trajectory_projection import ProjectedTrajectory


@dataclass
class TrajectoryProjectionConfig:
    max_suffix_steps: int = 20
    require_success: bool = True
    temperature: float = 1.0
    loss_scope: str = "action_tokens_and_eos_only"



def build_projection_example(row: dict) -> dict:
    """Convert a projected simulator trajectory into an SFT example.

    Expected row fields:
      task, history, observation, admissible_commands, projected_actions

    Projection metadata stays outside the language-model input.
    """
    required = {
        "task", "history", "observation",
        "admissible_commands", "projected_actions"
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"Missing causal fields: {sorted(missing)}")

    return {
        "task": row["task"],
        "history": [
            {"observation": x["observation"], "action": x["action"]}
            for x in row["history"]
        ],
        "observation": row["observation"],
        "admissible_commands": row["admissible_commands"],
        "action": row["projected_actions"][0],
        "trajectory_length": len(row["projected_actions"]),
    }


def save_projected_dataset(rows: list[dict], output: str | Path):
    """Write causal training records only."""
    with Path(output).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(build_projection_example(row), ensure_ascii=False) + "\n")


def validate_projected_trajectory(item: ProjectedTrajectory):
    if not item.actions:
        raise ValueError("Empty projected trajectory")
    if not item.success:
        raise ValueError("Projection training requires executable successful trajectories")
