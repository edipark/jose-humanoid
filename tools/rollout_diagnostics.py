"""Shared rollout diagnostics for G1 play and video tools."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import torch


def unwrap_env_with_robot(env):
    current = env
    for _ in range(32):
        candidate = getattr(current, "unwrapped", current)
        if hasattr(candidate, "robot") and hasattr(candidate, "ref_body_index"):
            return candidate
        next_env = getattr(current, "_env", None)
        if next_env is None or next_env is current:
            next_env = getattr(current, "env", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    return None


class RolloutDiagnostics:
    def __init__(self, core_env, env_index: int = 0, max_steps: int = 6000):
        if not hasattr(core_env, "action_offset") or not hasattr(core_env, "action_scale"):
            raise ValueError("Environment does not expose an action-to-position mapping")
        self.core = core_env
        self.env_index = env_index
        self.max_steps = max_steps
        self.samples = {
            name: []
            for name in (
                "action_raw",
                "action_applied",
                "position_target",
                "joint_position",
                "computed_torque",
                "applied_torque",
                "effort_limit",
            )
        }

    def record(self, actions: torch.Tensor) -> None:
        if self.max_steps > 0 and len(self.samples["action_raw"]) >= self.max_steps:
            return
        index = min(max(self.env_index, 0), actions.shape[0] - 1)
        action = actions[index].detach().float()
        applied_action = action
        if hasattr(self.core, "action_to_joint_position"):
            target = self.core.action_to_joint_position(applied_action)
        else:
            target = self.core.action_offset + self.core.action_scale * applied_action
            if hasattr(self.core, "action_soft_limits"):
                lower, upper = self.core.action_soft_limits.unbind(dim=-1)
                target = torch.maximum(torch.minimum(target, upper), lower)
        data = self.core.robot.data
        values = {
            "action_raw": action,
            "action_applied": applied_action,
            "position_target": target,
            "joint_position": data.joint_pos[index],
            "computed_torque": data.computed_torque[index],
            "applied_torque": data.applied_torque[index],
            "effort_limit": data.joint_effort_limits[index],
        }
        for name, value in values.items():
            self.samples[name].append(value.detach().cpu().numpy().copy())

    def save(self, output_dir: str | Path, step_dt: float, stem: str = "rollout_diagnostics") -> dict:
        if not self.samples["action_raw"]:
            return {}
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        arrays = {name: np.asarray(values) for name, values in self.samples.items()}
        arrays["time_s"] = np.arange(len(arrays["action_raw"]), dtype=np.float64) * float(step_dt)
        joint_names = np.asarray(self.core.robot.data.joint_names)
        data_path = output / f"{stem}.npz"
        np.savez_compressed(data_path, joint_names=joint_names, **arrays)
        result = {"data": str(data_path)}
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return result

        num_joints = len(joint_names)
        figure, axes = plt.subplots(num_joints, 2, figsize=(18, max(12, 2.4 * num_joints)), sharex=True)
        axes = np.asarray(axes).reshape(num_joints, 2)
        time_s = arrays["time_s"]
        for joint_id, joint_name in enumerate(joint_names):
            position_axis, torque_axis = axes[joint_id]
            position_axis.plot(time_s, arrays["joint_position"][:, joint_id], label="actual q")
            position_axis.plot(time_s, arrays["position_target"][:, joint_id], "--", label="target q")
            action_axis = position_axis.twinx()
            action_axis.plot(
                time_s, arrays["action_applied"][:, joint_id], color="0.55", alpha=0.6, label="action"
            )
            position_axis.set_title(str(joint_name))
            position_axis.grid(alpha=0.25)
            torque_axis.plot(time_s, arrays["computed_torque"][:, joint_id], label="computed")
            torque_axis.plot(time_s, arrays["applied_torque"][:, joint_id], label="applied")
            torque_axis.plot(time_s, arrays["effort_limit"][:, joint_id], ":", color="0.3", label="limit")
            torque_axis.plot(time_s, -arrays["effort_limit"][:, joint_id], ":", color="0.3")
            torque_axis.grid(alpha=0.25)
            if joint_id == 0:
                position_axis.legend(loc="upper left")
                action_axis.legend(loc="upper right")
                torque_axis.legend(loc="upper right")
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        figure.tight_layout()
        plot_path = output / f"{stem}.png"
        figure.savefig(plot_path, dpi=150)
        plt.close(figure)
        result["plot"] = str(plot_path)

        joint_plot_dir = output / f"{stem}_joints"
        joint_plot_dir.mkdir(parents=True, exist_ok=True)
        joint_plots = []
        for joint_id, joint_name in enumerate(joint_names):
            joint_figure, (position_axis, torque_axis) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            position_axis.plot(time_s, arrays["joint_position"][:, joint_id], label="actual q")
            position_axis.plot(time_s, arrays["position_target"][:, joint_id], "--", label="target q")
            action_axis = position_axis.twinx()
            action_axis.plot(
                time_s, arrays["action_applied"][:, joint_id], color="0.55", alpha=0.6, label="action"
            )
            position_axis.set_title(str(joint_name))
            position_axis.set_ylabel("position [rad]")
            action_axis.set_ylabel("raw policy action")
            position_axis.grid(alpha=0.25)
            position_axis.legend(loc="upper left")
            action_axis.legend(loc="upper right")
            torque_axis.plot(time_s, arrays["computed_torque"][:, joint_id], label="computed")
            torque_axis.plot(time_s, arrays["applied_torque"][:, joint_id], label="applied")
            torque_axis.plot(time_s, arrays["effort_limit"][:, joint_id], ":", color="0.3", label="limit")
            torque_axis.plot(time_s, -arrays["effort_limit"][:, joint_id], ":", color="0.3")
            torque_axis.set_xlabel("time [s]")
            torque_axis.set_ylabel("torque [Nm]")
            torque_axis.grid(alpha=0.25)
            torque_axis.legend(loc="upper right")
            joint_figure.tight_layout()
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(joint_name))
            joint_path = joint_plot_dir / f"{joint_id:02d}_{safe_name}.png"
            joint_figure.savefig(joint_path, dpi=150)
            plt.close(joint_figure)
            joint_plots.append(str(joint_path))
        result["joint_plots"] = joint_plots
        return result
