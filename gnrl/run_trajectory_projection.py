"""End-to-end CPS v2 experiment runner skeleton.

This runner connects the new trajectory-level components without changing the
legacy action-level CPS experiments. The intended pipeline is:

learner state -> viable candidate actions -> simulator rollout -> trajectory
projection -> trajectory SFT dataset.

The environment remains the source of truth for every successor observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .trajectory_projection import CausalTrajectoryProjector, ProjectedTrajectory


@dataclass
class ProjectionConfig:
    temperature: float = 1.0
    require_success: bool = True
    max_suffix_length: int = 20


class TrajectoryProjectionRunner:
    """Glue layer between ALFWorld rollout collection and projection.

    Candidate generation and environment execution are intentionally injected.
    This prevents accidentally introducing planner state or future observation
    leakage into the projection stage.
    """

    def __init__(self, config: ProjectionConfig | None = None):
        self.config = config or ProjectionConfig()
        self.projector = CausalTrajectoryProjector(self.config.temperature)

    def project_rollouts(self, trajectories: Iterable[ProjectedTrajectory]):
        return self.projector.project(
            trajectories,
            require_success=self.config.require_success,
        )

    def build_training_row(self, context: dict, trajectory: ProjectedTrajectory):
        """Create causal SFT data.

        Only model-visible context and the projected action sequence are kept.
        Oracle evidence and verification metadata stay outside training inputs.
        """
        return {
            "task": context["task"],
            "history": context.get("history", []),
            "observation": context["observation"],
            "admissible_commands": context["admissible_commands"],
            "trajectory": trajectory.actions,
        }
