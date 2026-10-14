# CPS-v2: Causal Trajectory Projection Sampling

## Motivation

The original CPS selects actions from a finite viable action set using learner action probabilities. This document defines the trajectory-level extension.

## Relation to Finetuning with Sampling

Finetuning with Sampling projects a constrained sequence distribution toward a learner distribution:

- FtS object: token sequence
- CPS-v2 object: environment-executable agent trajectory
- FtS constraint: correctness
- CPS-v2 constraint: causal simulator execution and task success

The projection target changes from:

`q(a | h)`

into:

`q(tau | h)`

where `tau` is a complete executable suffix trajectory.

## Pipeline

```
learner state
    |
    v
candidate actions
    |
    v
real environment rollout
    |
    v
successful trajectory pool
    |
    v
trajectory likelihood scoring
    |
    v
causal projection
    |
    v
SFT dataset
```

## Design principles

1. Environment transitions are authoritative.
2. Stored observations are never reused after changed actions.
3. Oracle information is used only for candidate verification, not model input.
4. Action-level CPS remains as an ablation baseline.

## Main hypothesis

Trajectory-level projection should better align learner distribution with successful behaviors because long-horizon agent performance depends on complete action sequences rather than isolated actions.
