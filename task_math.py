"""Pure timing and reward helpers shared by JOSE G1 environments and tests."""

from __future__ import annotations

import numpy as np
import torch


# Unitree's G1 RL/deployment stack uses 200 Hz physics with a 50 Hz policy.
PHYSICS_DT = 1.0 / 200.0
CONTROL_DECIMATION = 4
POLICY_DT = PHYSICS_DT * CONTROL_DECIMATION
# AMP references are sampled as independent states rather than phase-tracked
# trajectories, so all motion tasks intentionally share a 20-second episode.
EPISODE_LENGTH_S = 20.0
AMP_HISTORY_STEPS = 4
WALK_TARGET_VELOCITY = 0.6
TWIST_ACTION_SCALE = 0.5


def twist_action_to_position(
    actions: torch.Tensor,
    default_positions: torch.Tensor,
    soft_joint_limits: torch.Tensor,
    action_scale: float = TWIST_ACTION_SCALE,
) -> torch.Tensor:
    """Map TWIST-style actions to default-centered targets and enforce soft limits.

    TWIST interprets each policy output as a radian offset scaled by ``0.5``:
    ``q_target = q_default + action_scale * action``. The policy action itself is
    deliberately not clipped to ``[-1, 1]``; only the resulting joint target is
    bounded here.
    """
    if action_scale <= 0.0:
        raise ValueError(f"Action scale must be positive, got {action_scale}")
    if soft_joint_limits.shape[-1] != 2:
        raise ValueError("Soft joint limits must end with lower/upper bounds")
    lower, upper = soft_joint_limits.unbind(dim=-1)
    if torch.any(upper < lower):
        raise ValueError("Soft joint upper limits must be greater than or equal to lower limits")
    targets = default_positions + action_scale * actions
    return torch.maximum(torch.minimum(targets, upper), lower)


def action_finite_difference_penalties(
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    previous_previous_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample first- and second-order squared action differences."""
    if actions.shape != previous_actions.shape or actions.shape != previous_previous_actions.shape:
        raise ValueError("Current and historical action tensors must have identical shapes")
    action_rate = (actions - previous_actions).square().mean(dim=-1)
    action_second_difference = (
        actions - 2.0 * previous_actions + previous_previous_actions
    ).square().mean(dim=-1)
    return action_rate, action_second_difference


def inject_observation_estimate(
    observations: torch.Tensor, estimate: torch.Tensor, indices: tuple[int, ...]
) -> torch.Tensor:
    """Replace selected policy-observation columns without mutating the source tensor."""
    if estimate.shape[:-1] != observations.shape[:-1] or estimate.shape[-1] != len(indices):
        raise ValueError("Estimator output does not match the selected observation columns")
    result = observations.clone()
    result[..., list(indices)] = estimate
    return result


def reference_history_times(
    current_times: np.ndarray,
    num_steps: int = AMP_HISTORY_STEPS,
    policy_dt: float = POLICY_DT,
) -> np.ndarray:
    """Return newest-to-oldest AMP sample times at the policy control interval."""
    return (
        np.expand_dims(current_times, axis=-1)
        - policy_dt * np.arange(num_steps, dtype=np.float64)
    ).reshape(-1)
