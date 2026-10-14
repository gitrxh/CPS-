"""Evaluation utilities for action CPS versus trajectory CPS.

The evaluator keeps the original ALFWorld evaluation philosophy: task success is
measured by actual environment execution, not likelihood or offline scores.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationResult:
    method: str
    total_tasks: int
    successes: int

    @property
    def success_rate(self):
        return self.successes / self.total_tasks if self.total_tasks else 0.0


def summarize(method: str, outcomes: list[bool]) -> EvaluationResult:
    return EvaluationResult(
        method=method,
        total_tasks=len(outcomes),
        successes=sum(bool(x) for x in outcomes),
    )


def paired_gain(new: EvaluationResult, baseline: EvaluationResult):
    return new.successes - baseline.successes
