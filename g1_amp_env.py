"""29-DOF Unitree G1 AMP environment with a TWIST-style action interface."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .g1_amp_env_cfg import G1AmpEnvCfg
from .motions.motion_loader import MotionLoader
from .schema import G1_JOINT_NAMES, G1_KEY_BODY_NAMES
from .task_math import (
    action_finite_difference_penalties,
    reference_history_times,
    twist_action_to_position,
)


class G1AmpEnv(DirectRLEnv):
    cfg: G1AmpEnvCfg

    def __init__(self, cfg: G1AmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_offset = self.robot.data.default_joint_pos[0].clone()
        self.action_scale = torch.full_like(self.action_offset, float(self.cfg.action_scale))
        self.action_soft_limits = self.robot.data.soft_joint_pos_limits[0].clone()
        self.actions = torch.zeros((self.num_envs, 29), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.previous_previous_actions = torch.zeros_like(self.actions)
        self._action_history_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self._motion_loader = MotionLoader(
            self.cfg.motion_file,
            self.device,
            expected_dof_names=G1_JOINT_NAMES,
        )
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in G1_KEY_BODY_NAMES]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(G1_KEY_BODY_NAMES)

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        self._vel_window_buf = torch.full(
            (self.num_envs, self.cfg.vel_window_steps), 1.0e3, device=self.device
        )
        self._vel_window_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._episode_stats_capacity = self.num_envs
        self._episode_stats_count = 0
        self._episode_stats_write_index = 0
        self._completed_episode_lengths = torch.zeros(self.num_envs, device=self.device)
        self._completed_episode_timeouts = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                )
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_previous_actions.copy_(self.previous_actions)
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clone()
        self._action_history_count.clamp_max_(2).add_(1)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.action_to_joint_position(self.actions))

    def action_to_joint_position(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert raw policy actions into safe TWIST-style position targets."""
        return twist_action_to_position(
            actions,
            self.action_offset,
            self.action_soft_limits,
            action_scale=float(self.cfg.action_scale),
        )

    def _current_amp_observation(self) -> torch.Tensor:
        return compute_amp_observation(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

    def get_estimator_target(self) -> torch.Tensor:
        obs = self._current_amp_observation()
        return obs[:, 58:]

    def get_estimator_joint_state(self) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
        return self.robot.data.joint_pos, self.robot.data.joint_vel, tuple(self.robot.data.joint_names)

    def get_distillation_sensor_state(self) -> dict[str, torch.Tensor]:
        """Expose deployable joint/IMU quantities without modifying AMP state."""
        quaternion = self.robot.data.body_quat_w[:, self.ref_body_index]
        angular_velocity = quat_apply_inverse(
            quaternion, self.robot.data.body_ang_vel_w[:, self.ref_body_index]
        )
        gravity_world = torch.zeros_like(angular_velocity)
        gravity_world[:, 2] = -1.0
        projected_gravity = quat_apply_inverse(quaternion, gravity_world)
        return {
            "joint_position": self.robot.data.joint_pos,
            "joint_velocity": self.robot.data.joint_vel,
            "previous_action": self.previous_actions,
            "quaternion_wxyz": quaternion,
            "angular_velocity": angular_velocity,
            "projected_gravity": projected_gravity,
        }

    def _get_observations(self) -> dict:
        obs = self._current_amp_observation()
        self.amp_observation_buffer[:, 1:] = self.amp_observation_buffer[:, :-1].clone()
        self.amp_observation_buffer[:, 0] = obs
        previous_log = self.extras.get("log", {}) if isinstance(self.extras, dict) else {}
        self.extras = {
            "amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size),
            "log": previous_log,
        }
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        height = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
        vx = self.robot.data.body_lin_vel_w[:, self.ref_body_index, 0]
        velocity_reward = torch.exp(
            -self.cfg.velocity_tracking_coeff * (vx - self.cfg.target_velocity).square()
        )
        action_rate, action_second_difference = action_finite_difference_penalties(
            self.actions, self.previous_actions, self.previous_previous_actions
        )
        action_second_difference = action_second_difference * (
            self._action_history_count >= 3
        ).to(action_second_difference.dtype)
        action_rate_penalty = self.cfg.action_rate_penalty_weight * action_rate
        action_second_difference_penalty = (
            self.cfg.action_second_difference_penalty_weight * action_second_difference
        )
        raw_task = (
            self.cfg.velocity_reward_weight * velocity_reward
            - action_rate_penalty
            - action_second_difference_penalty
        )
        effort_limits = self.robot.data.joint_effort_limits
        saturation_fraction = (
            self.robot.data.computed_torque.abs() >= effort_limits - 1.0e-5
        ).float().mean(dim=-1)
        previous_log = self.extras.get("log", {}) if isinstance(self.extras.get("log"), dict) else {}
        self.extras["log"] = {
            **previous_log,
            "reward/raw_task": raw_task.mean().detach(),
            "reward/velocity_tracking": velocity_reward.mean().detach(),
            "reward/velocity_tracking_weighted": (
                self.cfg.velocity_reward_weight * velocity_reward.mean()
            ).detach(),
            "reward/action_rate_penalty": action_rate.mean().detach(),
            "reward/action_rate_penalty_weighted": action_rate_penalty.mean().detach(),
            "reward/action_second_difference_penalty": action_second_difference.mean().detach(),
            "reward/action_second_difference_penalty_weighted": (
                action_second_difference_penalty.mean()
            ).detach(),
            "metric/action_rate": action_rate.mean().detach(),
            "metric/action_second_difference": action_second_difference.mean().detach(),
            "metric/base_vel_x": vx.mean().detach(),
            "metric/base_height": height.mean().detach(),
            "metric/joint_velocity_squared": self.robot.data.joint_vel.square().mean().detach(),
            "metric/torque_saturation_fraction": saturation_fraction.mean().detach(),
        }
        return raw_task

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        too_low = torch.zeros_like(time_out)
        too_slow_window = torch.zeros_like(time_out)
        if self.cfg.early_termination:
            too_low = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
            vx = self.robot.data.body_lin_vel_w[:, self.ref_body_index, 0]
            if self.cfg.vel_window_min_vx > 0.0:
                write_index = self.episode_length_buf % self.cfg.vel_window_steps
                env_index = torch.arange(self.num_envs, device=self.device)
                self._vel_window_buf[env_index, write_index] = vx
                self._vel_window_count = torch.clamp(self._vel_window_count + 1, max=self.cfg.vel_window_steps)
                window_full = self._vel_window_count >= self.cfg.vel_window_steps
                too_slow_window = window_full & (self._vel_window_buf.mean(dim=1) < self.cfg.vel_window_min_vx)
            died = too_low | too_slow_window
        else:
            died = torch.zeros_like(time_out)
        log = self.extras.setdefault("log", {})
        log["episode/deaths"] = died.sum().float().detach()
        log["episode/height_deaths"] = too_low.sum().float().detach()
        log["episode/slow_velocity_deaths"] = (
            too_slow_window & ~too_low
        ).sum().float().detach()
        log["episode/timeouts"] = (time_out & ~died).sum().float().detach()
        self._log_completed_episode_metrics(died, time_out, log)
        return died, time_out

    def _log_completed_episode_metrics(
        self, died: torch.Tensor, time_out: torch.Tensor, log: dict[str, torch.Tensor]
    ) -> None:
        completed_ids = (died | time_out).nonzero(as_tuple=False).squeeze(-1)
        num_completed = completed_ids.numel()
        if num_completed:
            lengths = self.episode_length_buf[completed_ids].float()
            timeouts = (time_out[completed_ids] & ~died[completed_ids]).float()
            capacity = self._episode_stats_capacity
            if num_completed >= capacity:
                self._completed_episode_lengths[:] = lengths[-capacity:]
                self._completed_episode_timeouts[:] = timeouts[-capacity:]
                self._episode_stats_write_index = 0
                self._episode_stats_count = capacity
            else:
                start = self._episode_stats_write_index
                first = min(num_completed, capacity - start)
                self._completed_episode_lengths[start : start + first] = lengths[:first]
                self._completed_episode_timeouts[start : start + first] = timeouts[:first]
                remaining = num_completed - first
                if remaining:
                    self._completed_episode_lengths[:remaining] = lengths[first:]
                    self._completed_episode_timeouts[:remaining] = timeouts[first:]
                self._episode_stats_write_index = (start + num_completed) % capacity
                self._episode_stats_count = min(capacity, self._episode_stats_count + num_completed)

        if not self._episode_stats_count:
            return
        values = slice(None) if self._episode_stats_count == self._episode_stats_capacity else slice(
            0, self._episode_stats_count
        )
        mean_length = self._completed_episode_lengths[values].mean()
        timeout_fraction = self._completed_episode_timeouts[values].mean()
        log["episode/mean_length"] = mean_length.detach()
        log["episode/mean_length_s"] = (mean_length * self.step_dt).detach()
        log["episode/timeout_fraction"] = timeout_fraction.detach()
        log["episode/death_fraction"] = (1.0 - timeout_fraction).detach()
        log["episode/stats_window_count"] = torch.tensor(
            self._episode_stats_count, dtype=torch.float32, device=self.device
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._vel_window_buf[env_ids] = 1.0e3
        self._vel_window_count[env_ids] = 0
        if self.cfg.reset_strategy == "default":
            root_state = self.robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += self.scene.env_origins[env_ids]
            joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
            joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        elif self.cfg.reset_strategy.startswith("random"):
            root_state, joint_pos, joint_vel = self._sample_motion_reset(
                env_ids, start="start" in self.cfg.reset_strategy
            )
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.previous_previous_actions[env_ids] = 0.0
        self._action_history_count[env_ids] = 0
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _sample_motion_reset(self, env_ids: torch.Tensor, start: bool = False):
        count = env_ids.shape[0]
        reset_duration = max(self._motion_loader.duration - self.cfg.reset_time_end_margin, 0.0)
        times = np.zeros(count) if start else self._motion_loader.sample_times(count, duration=reset_duration)
        dof_pos, dof_vel, body_pos, body_rot, body_lin, body_ang = self._motion_loader.sample(count, times)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = body_pos[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.05
        root_state[:, 3:7] = body_rot[:, self.motion_ref_body_index]
        root_state[:, 7:10] = body_lin[:, self.motion_ref_body_index]
        root_state[:, 10:13] = body_ang[:, self.motion_ref_body_index]
        amp = self.collect_reference_motions(count, times)
        self.amp_observation_buffer[env_ids] = amp.view(count, self.cfg.num_amp_observations, -1)
        return root_state, dof_pos[:, self.motion_dof_indexes], dof_vel[:, self.motion_dof_indexes]

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None):
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        policy_dt = self.cfg.sim.dt * self.cfg.decimation
        times = reference_history_times(current_times, self.cfg.num_amp_observations, policy_dt)
        dof_pos, dof_vel, body_pos, body_rot, body_lin, body_ang = self._motion_loader.sample(
            num_samples, times
        )
        obs = compute_amp_observation(
            dof_pos[:, self.motion_dof_indexes],
            dof_vel[:, self.motion_dof_indexes],
            body_pos[:, self.motion_ref_body_index],
            body_rot[:, self.motion_ref_body_index],
            body_lin[:, self.motion_ref_body_index],
            body_ang[:, self.motion_ref_body_index],
            body_pos[:, self.motion_key_body_indexes],
        )
        return obs.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    tangent = torch.zeros_like(q[..., :3])
    normal = torch.zeros_like(q[..., :3])
    tangent[..., 0] = 1.0
    normal[..., 2] = 1.0
    return torch.cat((quat_apply(q, tangent), quat_apply(q, normal)), dim=-1)


@torch.jit.script
def compute_amp_observation(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    key_body_pos: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            joint_pos,
            joint_vel,
            root_pos[:, 2:3],
            quaternion_to_tangent_and_normal(root_quat),
            root_lin_vel,
            root_ang_vel,
            (key_body_pos - root_pos.unsqueeze(1)).reshape(key_body_pos.shape[0], -1),
        ),
        dim=-1,
    )
