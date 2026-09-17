"""Manager-based walk environment exposing JOSE's state-estimator interface.

``estimator/pipeline.py`` and ``estimator/adapters.py`` were written against the
Direct-workflow JOSE environments. This subclass provides the same three hooks on
a :class:`ManagerBasedRLEnv` so the estimator, DAgger and ablation code can drive
the PPO walk teacher without any change to their methodology.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class G1WalkEstimatorEnv(ManagerBasedRLEnv):
    """Walk environment with the estimator input/target accessors attached."""

    @property
    def robot(self):
        """The articulation, under the attribute name the estimator pipeline expects."""
        return self.scene["robot"]

    def get_estimator_target(self) -> torch.Tensor:
        """Privileged base state the estimator is trained to reproduce.

        Layout is ``(base_lin_vel_b, base_ang_vel_b, projected_gravity_b)``, which
        matches the 9-D target of JOSE's existing PPO schema, so estimator models,
        training schedules and ablation axes are reused unchanged.
        """
        data = self.robot.data
        return torch.cat((data.root_lin_vel_b, data.root_ang_vel_b, data.projected_gravity_b), dim=-1)

    def get_estimator_joint_state(self) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
        """Simulator joint positions and velocities plus their canonical names."""
        data = self.robot.data
        return data.joint_pos, data.joint_vel, tuple(data.joint_names)

    def get_distillation_sensor_state(self) -> dict[str, torch.Tensor]:
        """Expose deployable joint/IMU quantities, mirroring ``G1AmpEnv``'s contract.

        ``root_ang_vel_b`` and ``projected_gravity_b`` are already body-frame
        quantities computed by :class:`ArticulationData`, unlike the Direct-workflow
        AMP env which derives them from ``body_ang_vel_w`` via ``quat_apply_inverse``
        -- same values, cheaper path here.
        """
        data = self.robot.data
        return {
            "joint_position": data.joint_pos,
            "joint_velocity": data.joint_vel,
            "previous_action": self.action_manager.action,
            "quaternion_wxyz": data.root_quat_w,
            "angular_velocity": data.root_ang_vel_b,
            "projected_gravity": data.projected_gravity_b,
            # Deployable, and load-bearing: this teacher's action depends on it,
            # so a student that cannot see it cannot imitate the teacher. An
            # operator supplies the same three numbers on hardware.
            "velocity_command": self.command_manager.get_command("base_velocity")[:, :3],
        }
