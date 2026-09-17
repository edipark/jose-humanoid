"""Command-grid evaluation for the history-distillation students.

``estimator/pipeline.py``'s :func:`evaluate_locomotion_grid` drives the *teacher*
-- optionally through an estimated observation -- so it cannot evaluate a
standalone distilled policy. The grid itself can: :meth:`CommandEvaluator.run`
takes any callable mapping an observation to actions, plus an ``on_step(dones)``
hook for callers holding state that must be cleared when an environment resets.

That is exactly what a history student is: a policy whose real input comes from
``get_distillation_sensor_state()`` rather than from the observation it is
handed, backed by an :class:`ObservationHistory` (and, for the IMU variant, a
:class:`SensorCorruptor`) that must be reset in step with the environment.

Both callers share this module so they cannot drift:

* ``train_history_student.py`` -- so every future distillation run reports
  command tracking natively.
* ``eval_distillation_grid.py`` -- so runs that already finished can be
  back-filled from their checkpoints without retraining.

Why this matters: on the locomotion task episode length and death rate pin to
their ceilings for every competent policy, so they separate nothing. Command
tracking is the column that does. Without it the distilled students land in a
different report table from the teacher and JOSE (``reporting.py`` dispatches on
``track_error_norm_mean``) and the four methods are never comparable in one row
set -- which reads as "the students survive best" when in fact they survive by
not following the command.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch

from .history import (
    COMMAND_DIM,
    build_imu_frame,
    build_joint_frame,
    checkpoint_command_conditioning,
    require_command_conditioning,
)


def sensor_state(core) -> dict[str, torch.Tensor]:
    """The deployable sensor dict, with the forbidden-feature guard.

    ``base_linear_velocity`` and ``linear_acceleration`` are the two quantities a
    joint/IMU student must never see: the first is the privileged state the whole
    paper is about, the second would let it integrate its way to that state.
    """
    if not hasattr(core, "get_distillation_sensor_state"):
        raise RuntimeError("Task does not implement the JOSE distillation sensor contract")
    state = core.get_distillation_sensor_state()
    forbidden = {"base_linear_velocity", "linear_acceleration"}.intersection(state)
    if forbidden:
        raise RuntimeError(f"Deploy student was exposed to forbidden features: {sorted(forbidden)}")
    return state


def build_frame_fn(
    core,
    method: str,
    imu_spec=None,
    corruptor=None,
    command_conditioned: bool = False,
) -> Callable[[bool], torch.Tensor]:
    """The per-step observation frame, identical for training and evaluation.

    Shared so that a run back-filled from a checkpoint feeds the network exactly
    what training fed it. ``joint_only`` is a pure function of the sensor dict;
    ``imu`` additionally passes attitude through :class:`IMUObservationSpec` and,
    when ``use_corruption`` is set, through the quaternion sign flip and the
    :class:`SensorCorruptor` -- because q and -q are the same attitude, and a
    deploy student has to be invariant to which one the LowState message carries.
    """
    if method not in ("joint_only", "imu"):
        raise ValueError(f"Unknown distillation method {method!r}; choose joint_only or imu")
    if method == "imu" and imu_spec is None:
        raise ValueError("The imu method needs an IMUObservationSpec")

    def command_of(state: dict) -> torch.Tensor | None:
        """The velocity command, or ``None`` on a task that has no command.

        Failing loudly here matters: a command-conditioned run that silently fell
        back to ``None`` would train the very baseline this guard exists to stop.
        """
        if not command_conditioned:
            return None
        if "velocity_command" not in state:
            raise RuntimeError(
                "command_conditioned=True but the task exposes no 'velocity_command'; "
                "the student would be imitating a command-conditioned teacher blind"
            )
        command = state["velocity_command"]
        if command.shape[-1] != COMMAND_DIM:
            raise RuntimeError(f"velocity_command must have {COMMAND_DIM} entries")
        return command

    def frame(use_corruption: bool = True) -> torch.Tensor:
        state = sensor_state(core)
        if method == "joint_only":
            return build_joint_frame(
                state["joint_position"], state["joint_velocity"], state["previous_action"],
                command_of(state),
            )
        quaternion = state["quaternion_wxyz"]
        if use_corruption:
            signs = torch.where(
                torch.rand(quaternion.shape[0], 1, device=quaternion.device) < 0.5,
                -torch.ones(1, device=quaternion.device),
                torch.ones(1, device=quaternion.device),
            )
            quaternion = quaternion * signs
        observation = imu_spec.observe(quaternion, state["angular_velocity"], timestamp_s=0.0)
        if not observation.valid:
            raise RuntimeError(f"Simulation IMU adapter fault: {observation.fault.value}")
        gyro, gravity = observation.angular_velocity, observation.projected_gravity
        if use_corruption and corruptor is not None:
            gyro, gravity = corruptor(gyro, gravity)
        return build_imu_frame(
            state["joint_position"], state["joint_velocity"], state["previous_action"], gyro, gravity,
            command_of(state),
        )

    return frame


def reset_ids(dones: torch.Tensor | None) -> torch.Tensor | None:
    """Index tensor for the environments that just reset.

    ``CommandEvaluator`` passes a boolean mask (and an all-ones mask right after
    its initial reset). ``ObservationHistory.reset`` tolerates either form, but
    ``SensorCorruptor.reset`` sizes its fresh bias/latency draws with
    ``len(ids)``, which is the environment count -- not the reset count -- for a
    boolean mask. Convert once, here.
    """
    if dones is None:
        return None
    if dones.dtype == torch.bool:
        return dones.nonzero(as_tuple=True)[0]
    return dones


def build_student_policy(
    student,
    history,
    observation_normalizer,
    action_normalizer,
    frame_fn: Callable[[], torch.Tensor],
    extra_resets: Iterable[Callable[[torch.Tensor], None]] = (),
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], None]]:
    """``(act, on_step)`` for :meth:`CommandEvaluator.run_all`.

    ``act`` ignores the observation it is given: a deploy student reads the
    sensor dict instead, which is the whole point of the baseline. The action
    path is identical to the training and evaluation loops in
    ``train_history_student.py`` -- normalize, forward, denormalize -- so the
    policy measured here is the policy that was trained.

    ``extra_resets`` receives the same index tensor as the history; pass
    ``corruptor.reset`` for the IMU variant.
    """
    resets = tuple(extra_resets)

    def act(_observations: torch.Tensor) -> torch.Tensor:
        flattened = history.push(frame_fn())
        return action_normalizer.denormalize(student(observation_normalizer.normalize(flattened)))

    def on_step(dones: torch.Tensor) -> None:
        ids = reset_ids(dones)
        if ids is None or ids.numel() == 0:
            return
        history.reset(ids)
        for reset in resets:
            reset(ids)

    return act, on_step


@torch.no_grad()
def evaluate_student_command_grid(
    env,
    adapter,
    act: Callable[[torch.Tensor], torch.Tensor],
    on_step: Callable[[torch.Tensor], None] | None = None,
    settle_s: float = 1.0,
    measure_s: float = 4.0,
    seed: int | None = None,
    command_seed_base: int | None = None,
    commands: Sequence[tuple[float, float, float]] | None = None,
) -> dict:
    """Drive ``act`` over the fixed command grid and summarise it.

    Mirrors the setup half of ``evaluate_locomotion_grid`` so the numbers land in
    the same metric names the teacher and JOSE already report: ``track_*``,
    ``grid_*`` and ``command_tracking``. The sampler is snapshotted and restored
    by ``run_all`` itself, so the environment is left as it was found.
    """
    import numpy as np

    from jose.estimator.locomotion import CommandEvaluator, resolve_feet_cfgs, summarize

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    unwrapped = adapter.core_env
    feet_sensor_cfg, feet_asset_cfg = resolve_feet_cfgs(unwrapped.scene)
    step_dt = unwrapped.step_dt
    evaluator = CommandEvaluator(
        env,
        feet_sensor_cfg,
        feet_asset_cfg,
        settle_steps=int(round(settle_s / step_dt)),
        measure_steps=int(round(measure_s / step_dt)),
    )
    return summarize(
        evaluator.run_all(
            act,
            commands=commands,
            on_step=on_step,
            command_seed_base=command_seed_base,
        )
    )
