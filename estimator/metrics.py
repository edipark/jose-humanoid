"""Per-step rollout metrics, shared so every method reports the same quantity.

`collect_rollout`, the estimator's closed-loop evaluation and the distillation
student's evaluation all publish metrics under the same names into the same
comparison table, but they used to compute four of them differently. The worst
was `energy`: a sum over joints in one loop and a mean over joints in the other,
so the teacher read 4.03 and the students 0.14 -- a factor of 29, exactly the
joint count, in the table whose whole purpose is to compare them. Keeping one
implementation here is what makes those columns mean the same thing.
"""

from __future__ import annotations

import torch

from ..skrl_compat import amp_reward_components

# Fraction of the commanded range past which an action counts as saturated. The
# two loops disagreed (0.999 vs 0.99); the looser bound is kept because a
# near-rail action is already a rail action for these purposes.
ACTION_SATURATION_THRESHOLD = 0.99

AMP_COMPONENT_NAMES = ("raw_style", "scaled_task", "scaled_style", "effective_reward")


def joint_torque(core):
    """Torque actually applied to the joints, or None when the sim exposes neither field.

    `applied_torque` is the post-clamp value the articulation really used, so it
    is preferred; `computed_torque` is the pre-clamp command and only stands in
    when the former is unavailable.
    """
    data = core.robot.data
    for name in ("applied_torque", "computed_torque"):
        value = getattr(data, name, None)
        if value is not None:
            return value
    return None


def base_speeds(core, adapter) -> tuple[float, float]:
    """Root linear and angular speed, preferring the simulator's own body velocities.

    Falls back to the estimator target vector for envs that expose no reference
    body index (ppo_walk), where those columns carry the same quantity.
    """
    data = core.robot.data
    reference = getattr(core, "ref_body_index", None)
    if reference is not None and getattr(data, "body_lin_vel_w", None) is not None:
        return (
            float(data.body_lin_vel_w[:, reference].norm(dim=-1).mean()),
            float(data.body_ang_vel_w[:, reference].norm(dim=-1).mean()),
        )
    names = adapter.schema.estimator_target_names
    target = adapter.estimator_target()
    linear = [names.index(f"base_lin_vel_{axis}") for axis in "xyz"]
    angular = [names.index(f"base_ang_vel_{axis}") for axis in "xyz"]
    return (
        float(target[:, linear].norm(dim=-1).mean()),
        float(target[:, angular].norm(dim=-1).mean()),
    )


@torch.no_grad()
def step_metrics(
    core,
    adapter,
    teacher_agent,
    *,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    rewards: torch.Tensor,
    policy_action: torch.Tensor | None = None,
    teacher_action: torch.Tensor | None = None,
) -> dict[str, float]:
    """Metrics for one environment step, already reduced to python floats.

    `action` is what actually drove the robot. `policy_action` and
    `teacher_action` are compared into `teacher_action_mse` when both are given:
    the method's own action versus what the teacher would have done from the
    true state. For the estimator that is "same policy, estimated state vs true
    state"; for a distillation student it is "student vs teacher". Consumes no
    RNG, so it can be added to a loop without shifting the random stream.
    """
    metrics = {
        "action_smoothness": float((action - previous_action).square().mean()),
        "action_saturation": float((action.abs() >= ACTION_SATURATION_THRESHOLD).float().mean()),
        "raw_task_reward": float(rewards.mean()),
    }
    torque = joint_torque(core)
    if torque is not None:
        effort_limit = core.robot.data.joint_effort_limits.clamp_min(1.0e-6)
        metrics["torque_rms"] = float(torque.square().mean().sqrt())
        metrics["energy"] = float((torque * core.robot.data.joint_vel).abs().sum(dim=-1).mean() * core.step_dt)
        metrics["torque_saturation"] = float((torque.abs() >= effort_limit).float().mean())
    linear_speed, angular_speed = base_speeds(core, adapter)
    metrics["base_linear_speed"] = linear_speed
    metrics["base_angular_speed"] = angular_speed
    if policy_action is not None and teacher_action is not None:
        metrics["teacher_action_mse"] = float((policy_action - teacher_action).square().mean())
    amp_observations = getattr(core, "extras", {}).get("amp_obs")
    if amp_observations is not None:
        components = amp_reward_components(teacher_agent, amp_observations, rewards)
        for name in AMP_COMPONENT_NAMES:
            metrics[f"amp_{name}"] = float(components[name].mean())
        metrics["task_reward_scale"] = components["task_reward_scale"]
        metrics["style_reward_scale"] = components["style_reward_scale"]
    return metrics


class MetricAccumulator:
    """Running per-key mean of `step_metrics` dicts.

    Counts each key separately rather than dividing everything by the step
    count, because AMP components only appear on steps where the env published
    `amp_obs` -- dividing those by the total step count would silently scale
    them down on any run where they are intermittent.
    """

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def add(self, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            self._totals[name] = self._totals.get(name, 0.0) + value
            self._counts[name] = self._counts.get(name, 0) + 1

    def mean(self) -> dict[str, float]:
        return {name: total / max(self._counts[name], 1) for name, total in self._totals.items()}
