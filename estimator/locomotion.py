"""Fixed-command locomotion evaluation, shared by the eval script and the ablation.

The AMP tasks are judged by how long the robot survives, but a velocity-tracking
walk policy that is any good survives the whole episode every time: episode
length pins to the 1000-step ceiling and death rate to zero, and every estimator
architecture ties. What separates them on this task is whether the robot still
follows the commanded ``(vx, vy, yaw)`` once the privileged state it consumes is
replaced by an estimate.

Measuring that needs the command held *fixed*. The environment resamples it every
10 s from a random range, so an episode sees two commands, each arm sees
different ones, and the per-command failures that matter -- creeping under a zero
command, losing yaw authority -- disappear into an average. :data:`EVAL_COMMANDS`
pins one command across every environment at a time instead, which makes the
numbers comparable across arms and keeps the per-command breakdown.

This module holds the measurement itself so that ``eval_ppo_walk.py`` (a trained
checkpoint driven directly) and ``estimator/pipeline.py`` (the same teacher driven
by an estimate) report the same quantity computed by the same code. Only the
action callable differs.
"""

from __future__ import annotations

import torch

# Isaac Lab is imported lazily, inside the functions that touch a live scene.
# `EVAL_COMMANDS`, `COMMAND_SCALES`, `summarize` and `run_sanity_checks` are pure
# and are exercised by the CPU test suite, which runs without the simulator.


#: Both feet. Resolves to ``left_ankle_roll_link`` and ``right_ankle_roll_link``.
FEET_BODIES = ".*ankle_roll.*"

# (vx [m/s], vy [m/s], yaw rate [rad/s]) -- the evaluation set.
#
# The task samples commands from vx 0..1, vy +-0.5, yaw +-1 (see
# ``ppo_walk/walk_env_cfg.py``). The first eight rows below are a low-speed
# interior subset that tops out at 60%/40%/30% of those ranges, which left the
# outer half of the command distribution unmeasured: a policy that loses
# authority only at high yaw rate or full forward speed scored the same as one
# that did not. The remaining rows pin each axis at its range limit and add two
# combined commands, so the reported tracking error covers the distribution the
# policy was actually trained on.
#
# The original eight are kept verbatim and first so the per-command breakdown
# stays comparable with runs logged before the set was widened.
#
# The zero command is first and is load-bearing: it is the only row that exposes
# a policy that marches in place or creeps when told to stand still.
#
# Cost is linear in the row count and small: each row runs
# ``--grid-settle-s`` + ``--grid-measure-s`` (1 s + 4 s) across all environments
# at once, so the full set is well under two minutes of simulated time.
EVAL_COMMANDS = [
    # -- interior, unchanged ------------------------------------------------
    (0.0, 0.0, 0.0),
    (0.3, 0.0, 0.0),
    (0.6, 0.0, 0.0),
    (0.0, 0.2, 0.0),
    (0.0, -0.2, 0.0),
    (0.0, 0.0, 0.3),
    (0.0, 0.0, -0.3),
    (0.4, 0.2, 0.2),
    # -- range limits, one axis at a time -----------------------------------
    (1.0, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, -0.5, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    # -- combined, where axes compete for foot placement --------------------
    (0.8, -0.3, 0.6),
    (1.0, 0.0, 1.0),
]

#: Per-axis normalisers for :func:`summarize`, one per command axis, taken from the
#: task's command ranges (``vx 0..1``, ``vy +-0.5``, ``yaw +-1``). Dividing each
#: axis by its own range is what lets the three be averaged at all: ``vx_rmse`` is
#: in m/s and ``yaw_rmse`` in rad/s, so their raw mean has no meaning.
COMMAND_SCALES = (1.0, 0.5, 1.0)

#: Largest normalised tracking error a command may show and still count as a
#: success, given the robot also survived the whole measurement window.
#: Expressed in the same units as ``track_error_norm``: the mean over axes of
#: each axis RMSE divided by that axis's amplitude, so 0.25 is "off by a quarter
#: of the command range". Fixed here from the structure of the command space
#: rather than tuned against any measured result, so that a rerun cannot move it.
TASK_SUCCESS_MAX_NORM_ERROR = 0.25


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _mean_present(values: list) -> float | None:
    """Mean over the entries that exist, or ``None`` when none do.

    A command where every robot fell before the measurement window reports
    ``None`` for its tracking fields. Averaging those in as NaN would take the
    whole sweep down with them, and averaging them in as zero would score a
    total failure as perfect tracking; dropping them and reporting how many were
    dropped is the only honest option. Callers that need the sweep to be
    complete should read ``commands_all_dead``.
    """
    present = [v for v in values if v is not None]
    return float(sum(present) / len(present)) if present else None


def resolve_feet_cfgs(scene):
    """Resolve the contact-sensor and asset configs for both feet.

    Raises:
        RuntimeError: If the pattern does not match exactly two bodies. The
            biped air-time metric is defined for two feet and silently averaging
            over some other number would not be an error anywhere downstream.
    """
    from isaaclab.managers import SceneEntityCfg

    sensor_cfg = SceneEntityCfg("contact_forces", body_names=FEET_BODIES)
    sensor_cfg.resolve(scene)
    asset_cfg = SceneEntityCfg("robot", body_names=FEET_BODIES)
    asset_cfg.resolve(scene)
    if len(sensor_cfg.body_ids) != 2:
        raise RuntimeError(
            f"'{FEET_BODIES}' resolved to {len(sensor_cfg.body_ids)} bodies "
            f"({sensor_cfg.body_names}); the biped air-time metric needs exactly 2."
        )
    return sensor_cfg, asset_cfg


class _EnvShim:
    """Uniform ``reset``/``step`` over the two vector-env wrappers in this repo.

    ``eval_ppo_walk.py`` hands in an ``RslRlVecEnvWrapper``, whose ``step``
    returns ``(obs, reward, dones, extras)``. The ablation path hands in
    ``JoseRslRlEnvAdapter``, which presents SKRL's five-value signature so the
    estimator pipeline can drive it. Both expose ``unwrapped``, so only the step
    arity has to be reconciled.
    """

    def __init__(self, env):
        self.env = env
        self.unwrapped = env.unwrapped

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, actions) -> tuple[object, torch.Tensor]:
        outputs = self.env.step(actions)
        if len(outputs) == 4:  # rsl-rl wrapper: obs, reward, dones, extras
            obs, _, dones, _ = outputs
            return obs, dones.bool().view(-1)
        # SKRL-shaped: obs, reward, terminated, truncated, extras
        obs, _, terminated, truncated, _ = outputs
        return obs, (terminated | truncated).bool().view(-1)


class CommandEvaluator:
    """Runs one fixed command across all environments and accumulates statistics."""

    def __init__(self, env, feet_sensor_cfg, feet_asset_cfg, settle_steps, measure_steps):
        self.env = _EnvShim(env)
        self.unwrapped = self.env.unwrapped
        self.feet_sensor_cfg = feet_sensor_cfg
        self.feet_asset_cfg = feet_asset_cfg
        self.settle_steps = settle_steps
        self.measure_steps = measure_steps
        self.dt = self.unwrapped.step_dt
        self.command_term = self.unwrapped.command_manager.get_term("base_velocity")

    def _saved_command_cfg(self) -> dict:
        """Snapshot the command settings this evaluator is about to overwrite."""
        ranges = self.command_term.cfg.ranges
        return {
            "lin_vel_x": tuple(ranges.lin_vel_x),
            "lin_vel_y": tuple(ranges.lin_vel_y),
            "ang_vel_z": tuple(ranges.ang_vel_z),
            "rel_standing_envs": self.command_term.cfg.rel_standing_envs,
        }

    def _restore_command_cfg(self, saved: dict) -> None:
        """Put the random sampler back.

        In ``eval_ppo_walk.py`` the process exits afterwards and this does not
        matter. In the ablation the same environment goes straight back into
        DAgger collection, which is supposed to see the *random* command
        distribution -- leaving the ranges pinned would collapse every
        subsequent round onto the last evaluated command.
        """
        ranges = self.command_term.cfg.ranges
        ranges.lin_vel_x = saved["lin_vel_x"]
        ranges.lin_vel_y = saved["lin_vel_y"]
        ranges.ang_vel_z = saved["ang_vel_z"]
        self.command_term.cfg.rel_standing_envs = saved["rel_standing_envs"]

    def _force_command(self, cmd: torch.Tensor):
        """Pin the command for every environment.

        ``ranges`` is narrowed so that any mid-episode resample reproduces the same
        value, ``is_standing_env`` is cleared so no environment is silently zeroed,
        and the buffer itself is overwritten because ``_update_command`` runs at the
        end of :meth:`step`, after the reward has already been computed.
        """
        ranges = self.command_term.cfg.ranges
        ranges.lin_vel_x = (float(cmd[0]), float(cmd[0]))
        ranges.lin_vel_y = (float(cmd[1]), float(cmd[1]))
        ranges.ang_vel_z = (float(cmd[2]), float(cmd[2]))
        self.command_term.cfg.rel_standing_envs = 0.0
        self.command_term.is_standing_env[:] = False
        self.command_term.vel_command_b[:] = cmd

    def run(self, policy, cmd_tuple: tuple[float, float, float], on_step=None) -> dict:
        """Measure one command.

        Args:
            policy: Any callable mapping the observation to actions. The teacher
                policy, the same teacher driven by an estimate, or a distillation
                student all satisfy this.
            cmd_tuple: The ``(vx, vy, yaw)`` to hold fixed.
            on_step: Optional ``fn(dones)`` invoked after every step, for callers
                carrying state that must be cleared when an environment resets
                (an estimator's history buffer, for instance).
        """
        env, unwrapped = self.env, self.unwrapped
        device = unwrapped.device
        num_envs = unwrapped.num_envs
        cmd = torch.tensor(cmd_tuple, device=device, dtype=torch.float32)

        # `no_grad`, not `inference_mode`. Tensors first created under inference
        # mode stay inference tensors forever, and the environment materialises
        # some of its own buffers during `reset` -- doing that here would make the
        # next `collect_rollout` reset, which runs under plain `no_grad`, raise.
        # The rest of the estimator pipeline uses `no_grad` for the same reason.
        with torch.no_grad():
            obs = env.reset()
            if on_step is not None:
                on_step(torch.ones(num_envs, dtype=torch.bool, device=device))
        self._force_command(cmd)

        robot = unwrapped.scene["robot"]
        contact_sensor = unwrapped.scene.sensors[self.feet_sensor_cfg.name]
        feet_sensor_ids = self.feet_sensor_cfg.body_ids

        # per-env accumulators over the measurement window
        alive = torch.ones(num_envs, dtype=torch.bool, device=device)
        fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Split so a caller can tell "never got going" from "lost it mid-measure".
        fell_settling = torch.zeros(num_envs, dtype=torch.bool, device=device)
        fell_measuring = torch.zeros(num_envs, dtype=torch.bool, device=device)
        n_samples = torch.zeros(num_envs, device=device)
        sum_v = torch.zeros(num_envs, 3, device=device)  # vx, vy (yaw frame), yaw rate (base frame)
        sum_sq_err = torch.zeros(num_envs, 3, device=device)
        sum_height = torch.zeros(num_envs, device=device)
        sum_air_time_rew = torch.zeros(num_envs, device=device)
        sum_slide_pen = torch.zeros(num_envs, device=device)
        lift_count = torch.zeros(num_envs, device=device)
        double_stance_steps = torch.zeros(num_envs, device=device)
        # violation counter for the "air time must be 0 in double stance" invariant
        double_stance_reward_violations = 0
        max_air_reward_in_double_stance = 0.0
        start_xy = None
        end_xy = torch.zeros(num_envs, 2, device=device)
        prev_contact = None

        # Imported here rather than at module scope: these pull in isaaclab and
        # the mdp package, which are only importable once the simulator is up.
        from isaaclab.utils.math import quat_apply_inverse, yaw_quat

        from ..ppo_walk import mdp as walk_mdp

        total_steps = self.settle_steps + self.measure_steps
        with torch.no_grad():
            for step in range(total_steps):
                actions = policy(obs)
                obs, dones = env.step(actions)
                if on_step is not None:
                    on_step(dones)
                self._force_command(cmd)

                measuring = step >= self.settle_steps
                # A fall is recorded wherever it happens. Counting only inside
                # the measurement window hid every settle-phase fall: the
                # environment was still marked dead by `alive &= ~dones` below
                # and silently stopped contributing samples, so a command that
                # killed every robot during settle reported `fall_count` 0.
                # `dones` here includes time-outs, but the window is short
                # enough that none occur.
                terminated = unwrapped.termination_manager.terminated
                newly_fell = terminated & alive
                fell |= newly_fell
                if measuring:
                    fell_measuring |= newly_fell
                else:
                    fell_settling |= newly_fell
                alive &= ~dones

                if not measuring:
                    prev_contact = contact_sensor.data.current_contact_time[:, feet_sensor_ids] > 0.0
                    continue

                if start_xy is None:
                    start_xy = robot.data.root_pos_w[:, :2].clone()

                mask = alive.float()

                # -- velocity, measured the same way the reward kernels measure it
                vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
                meas = torch.stack([vel_yaw[:, 0], vel_yaw[:, 1], robot.data.root_ang_vel_b[:, 2]], dim=1)
                sum_v += meas * mask.unsqueeze(1)
                sum_sq_err += torch.square(meas - cmd.unsqueeze(0)) * mask.unsqueeze(1)
                sum_height += robot.data.root_pos_w[:, 2] * mask
                n_samples += mask

                # -- gait statistics
                in_contact = contact_sensor.data.current_contact_time[:, feet_sensor_ids] > 0.0
                lifts = (prev_contact & ~in_contact).float().sum(dim=1)
                lift_count += lifts * mask
                double_stance = in_contact.all(dim=1)
                double_stance_steps += double_stance.float() * mask
                prev_contact = in_contact

                # -- raw (unweighted) reward-term values
                air_rew = walk_mdp.feet_air_time_positive_biped(
                    unwrapped,
                    command_name="base_velocity",
                    threshold=0.4,
                    sensor_cfg=self.feet_sensor_cfg,
                )
                slide_pen = walk_mdp.feet_slide(
                    unwrapped, sensor_cfg=self.feet_sensor_cfg, asset_cfg=self.feet_asset_cfg
                )
                sum_air_time_rew += air_rew * mask
                sum_slide_pen += slide_pen * mask

                # -- invariant: no air-time reward while both feet are down
                bad = double_stance & alive & (air_rew > 0.0)
                if bad.any():
                    double_stance_reward_violations += int(bad.sum().item())
                    max_air_reward_in_double_stance = max(
                        max_air_reward_in_double_stance, float(air_rew[bad].max().item())
                    )

                end_xy = torch.where(alive.unsqueeze(1), robot.data.root_pos_w[:, :2], end_xy)

        with torch.no_grad():
            # `valid` is "contributed at least one sample", which includes an
            # environment that fell after one step. Averaging over it alone is a
            # survivor statistic: the RMSE of a robot that managed two frames
            # carries the same weight as one that held the command for the whole
            # window. `complete` is the honest denominator, and the two are
            # reported side by side so a reader can see when they diverge.
            valid = n_samples > 0
            complete = n_samples >= float(self.measure_steps)
            n_valid = int(valid.sum())
            n = n_samples.clamp(min=1.0)

            scales = torch.tensor(COMMAND_SCALES, device=device, dtype=torch.float32)
            rmse_all = torch.sqrt(sum_sq_err / n.unsqueeze(1))
            norm_err_all = (rmse_all / scales).mean(dim=1)
            # Task success demands both halves of the job: stay up for the whole
            # measurement AND deliver the command. An environment that never
            # contributed a sample fails by construction, since `complete` is
            # false for it.
            success = complete & (norm_err_all <= TASK_SUCCESS_MAX_NORM_ERROR)

            mean_v = sum_v / n.unsqueeze(1)
            measure_s = self.measure_steps * self.dt
            drift_all = (
                torch.norm(end_xy - start_xy, dim=1)
                if start_xy is not None
                else torch.zeros(num_envs, device=device)
            )

            def survivor_mean(values: torch.Tensor) -> float | None:
                """Mean over contributing environments, or ``None`` if there are none.

                Returning ``None`` rather than NaN keeps the JSON readable and
                makes an all-dead command explicit downstream instead of
                poisoning every aggregate that touches it.
                """
                return float(values[valid].mean()) if n_valid else None

            return {
                "command": {"vx": cmd_tuple[0], "vy": cmd_tuple[1], "yaw": cmd_tuple[2]},
                "measured": {
                    "vx": survivor_mean(mean_v[:, 0]),
                    "vy": survivor_mean(mean_v[:, 1]),
                    "yaw": survivor_mean(mean_v[:, 2]),
                },
                "error": {
                    "vx_mean": survivor_mean(mean_v[:, 0] - cmd[0]),
                    "vy_mean": survivor_mean(mean_v[:, 1] - cmd[1]),
                    "yaw_mean": survivor_mean(mean_v[:, 2] - cmd[2]),
                    "vx_rmse": survivor_mean(rmse_all[:, 0]),
                    "vy_rmse": survivor_mean(rmse_all[:, 1]),
                    "yaw_rmse": survivor_mean(rmse_all[:, 2]),
                },
                "survival_rate": float(alive.float().mean()),
                "task_success_rate": float(success.float().mean()),
                "fall_count": int(fell.sum()),
                "settle_fall_count": int(fell_settling.sum()),
                "measure_fall_count": int(fell_measuring.sum()),
                "num_envs": int(num_envs),
                "num_contributing": n_valid,
                "num_complete": int(complete.sum()),
                "base_height_m": survivor_mean(sum_height / n),
                "feet_air_time_reward": survivor_mean(sum_air_time_rew / n),
                "feet_slide_penalty": survivor_mean(sum_slide_pen / n),
                "foot_lifts_per_s": survivor_mean(lift_count / measure_s),
                "double_stance_fraction": survivor_mean(double_stance_steps / n),
                "drift_m": survivor_mean(drift_all),
                "drift_speed_mps": survivor_mean(drift_all / measure_s),
                "double_stance_air_reward_violations": double_stance_reward_violations,
                "max_air_reward_in_double_stance": max_air_reward_in_double_stance,
            }

    def run_all(
        self,
        policy,
        commands=None,
        on_step=None,
        progress=None,
        command_seed_base: int | None = None,
    ) -> dict:
        """Measure every command in turn, restoring the sampler afterwards.

        ``command_seed_base`` makes resets comparable across policies.  A seed
        is installed immediately before each command's reset, so random draws
        made by a policy or a stateful sensor during an earlier command cannot
        change the next command's initial condition.  Method and checkpoint
        identities deliberately do not enter this seed.
        """
        import random

        import numpy as np

        saved = self._saved_command_cfg()
        results = {}
        try:
            selected = EVAL_COMMANDS if commands is None else commands
            for index, cmd in enumerate(selected):
                if command_seed_base is not None:
                    command_seed = int(command_seed_base) + index
                    random.seed(command_seed)
                    np.random.seed(command_seed)
                    torch.manual_seed(command_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(command_seed)
                if progress is not None:
                    progress(cmd)
                results[str(cmd)] = self.run(policy, cmd, on_step=on_step)
        finally:
            self._restore_command_cfg(saved)
        return results


def summarize(results: dict) -> dict:
    """Flatten per-command results into the scalar metrics the report aggregates.

    ``track_error_norm`` is the headline: each axis RMSE divided by its own
    command range (:data:`COMMAND_SCALES`) and averaged, so it reads as the
    fraction of the command range the policy fails to deliver. It is the ranking
    key for DAgger round selection on this task, where episode length is pinned
    to its ceiling and carries no signal.
    """
    rows = list(results.values())
    if not rows:
        raise ValueError("summarize() needs at least one evaluated command")

    axis_rmse = {
        axis: _mean_present([row["error"][f"{axis}_rmse"] for row in rows])
        for axis in ("vx", "vy", "yaw")
    }
    metrics = {f"track_{axis}_rmse": value for axis, value in axis_rmse.items()}
    metrics.update(
        {
            f"track_{axis}_bias": _mean_present([row["error"][f"{axis}_mean"] for row in rows])
            for axis in ("vx", "vy", "yaw")
        }
    )
    metrics["track_error_norm"] = _mean_present(
        [
            None if axis_rmse[axis] is None else axis_rmse[axis] / scale
            for axis, scale in zip(("vx", "vy", "yaw"), COMMAND_SCALES)
        ]
    )
    for name in ("foot_lifts_per_s", "double_stance_fraction", "feet_air_time_reward", "feet_slide_penalty"):
        metrics[name] = _mean_present([row[name] for row in rows])
    # Prefixed so they cannot collide with the episode-based evaluation's own
    # `success_rate`/`death_rate`, which are still reported alongside these.
    metrics["grid_survival_rate"] = _mean([row["survival_rate"] for row in rows])
    metrics["grid_task_success_rate"] = _mean([row["task_success_rate"] for row in rows])
    metrics["grid_drift_speed_mps"] = _mean_present([row["drift_speed_mps"] for row in rows])
    metrics["grid_fall_count"] = float(sum(row["fall_count"] for row in rows))
    metrics["grid_settle_fall_count"] = float(sum(row["settle_fall_count"] for row in rows))
    metrics["grid_measure_fall_count"] = float(sum(row["measure_fall_count"] for row in rows))
    # How much of the sweep the tracking aggregates above actually rest on. A
    # nonzero count means `track_*` describes only the commands that had a
    # survivor, and must not be read as sweep-wide performance.
    metrics["commands_all_dead"] = int(sum(1 for row in rows if row["num_contributing"] == 0))
    metrics["commands_total"] = len(rows)
    metrics["command_tracking"] = [
        {
            "command": row["command"],
            "measured": row["measured"],
            "error": row["error"],
            "survival_rate": row["survival_rate"],
            "task_success_rate": row["task_success_rate"],
            "fall_count": row["fall_count"],
            "settle_fall_count": row["settle_fall_count"],
            "measure_fall_count": row["measure_fall_count"],
            "num_contributing": row["num_contributing"],
            "num_complete": row["num_complete"],
            "foot_lifts_per_s": row["foot_lifts_per_s"],
            "double_stance_fraction": row["double_stance_fraction"],
            "drift_speed_mps": row["drift_speed_mps"],
            "base_height_m": row["base_height_m"],
        }
        for row in rows
    ]
    return metrics


def run_sanity_checks(results: dict) -> list[tuple[str, bool, str]]:
    """Return ``(name, passed, detail)`` for each required sanity check."""
    checks = []

    def r(cmd):
        return results[str(cmd)]

    zero = r((0.0, 0.0, 0.0))
    fwd_03 = r((0.3, 0.0, 0.0))
    fwd_06 = r((0.6, 0.0, 0.0))
    yaw_p = r((0.0, 0.0, 0.3))
    yaw_n = r((0.0, 0.0, -0.3))

    # 1. air-time reward must never fire while both feet are on the ground
    violations = sum(v["double_stance_air_reward_violations"] for v in results.values())
    checks.append((
        "feet_air_time is 0 in double stance",
        violations == 0,
        f"{violations} violating samples (max reward seen "
        f"{max(v['max_air_reward_in_double_stance'] for v in results.values()):.4f})",
    ))

    # 2. a nonzero command that produces no stepping must earn no air-time reward
    standing_under_command = [
        (k, v)
        for k, v in results.items()
        if (v["command"]["vx"], v["command"]["vy"]) != (0.0, 0.0) and v["double_stance_fraction"] > 0.99
    ]
    checks.append((
        "no air-time reward while standing under a nonzero command",
        all(v["feet_air_time_reward"] <= 1e-6 for _, v in standing_under_command),
        f"{len(standing_under_command)} commands stood still"
        + ("" if not standing_under_command else f": {[k for k, _ in standing_under_command]}"),
    ))

    # 3. zero command must not make the robot step in place
    checks.append((
        "zero command does not lift feet repeatedly",
        zero["foot_lifts_per_s"] < 0.5,
        f"{zero['foot_lifts_per_s']:.3f} lifts/s, air-time reward {zero['feet_air_time_reward']:.4f}",
    ))

    # 4. zero command must not drift
    checks.append((
        "zero command does not drift",
        zero["drift_speed_mps"] < 0.1,
        f"{zero['drift_speed_mps']:.3f} m/s over the window",
    ))

    # 5. measured vx must increase *meaningfully* with commanded vx. A bare
    #    monotonicity test passes on numerical noise (0.001 < 0.002), which would
    #    report a statue as responsive, so require a real margin per step.
    margin = 0.05
    checks.append((
        "measured vx increases with commanded vx",
        (fwd_03["measured"]["vx"] > zero["measured"]["vx"] + margin)
        and (fwd_06["measured"]["vx"] > fwd_03["measured"]["vx"] + margin),
        f"vx(0)={zero['measured']['vx']:.3f}, vx(0.3)={fwd_03['measured']['vx']:.3f},"
        f" vx(0.6)={fwd_06['measured']['vx']:.3f} (each step must gain > {margin})",
    ))

    # 6. yaw-only commands must produce yaw rate of the right sign
    checks.append((
        "yaw-only command produces angular velocity",
        yaw_p["measured"]["yaw"] > 0.1 and yaw_n["measured"]["yaw"] < -0.1,
        f"yaw(+0.3)={yaw_p['measured']['yaw']:.3f}, yaw(-0.3)={yaw_n['measured']['yaw']:.3f}",
    ))

    return checks
