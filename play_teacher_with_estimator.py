"""Play a G1 teacher with 43-D estimated privileged state or replay logged actions."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="JOSE G1 teacher + state estimator play")
parser.add_argument("--teacher-checkpoint", default=None)
parser.add_argument("--estimator-checkpoint", default=None)
parser.add_argument("--replay-action-log", default=None)
parser.add_argument("--task", default="Isaac-G1-AMP-Walk-JOSE-Direct-v0")
parser.add_argument("--agent", default="skrl_amp_cfg_entry_point")
parser.add_argument("--adapter", choices=("amp", "ppo_walk"), default="amp")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video-length", type=int, default=600)
parser.add_argument("--video-dir", default="logs/jose_g1/videos/teacher_estimator")
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--csv-output", default=None)
parser.add_argument("--action-log-output", default=None)
parser.add_argument("--log-env-id", type=int, default=0)
parser.add_argument("--diagnostic-output", default=None)
parser.add_argument(
    "--keep-velocity-termination",
    action="store_true",
    help="Keep the training-time low-forward-velocity termination during play.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.replay_action_log is None and (not args_cli.teacher_checkpoint or not args_cli.estimator_checkpoint):
    parser.error("teacher and estimator checkpoints are required unless --replay-action-log is used")
from jose.teacher_setup import resolve_agent_entry_point  # noqa: E402

# `--adapter ppo_walk` implies the rsl-rl runner config unless overridden.
args_cli.agent = resolve_agent_entry_point(args_cli.adapter, args_cli.agent)

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401

from jose.estimator.adapters import make_policy_adapter
from jose.estimator.pipeline import HistoryBuffer, load_estimator
from jose.skrl_compat import disable_velocity_termination_for_evaluation
from jose.teacher_setup import build_env_and_teacher, uses_rsl_rl
from jose.tools.rollout_diagnostics import RolloutDiagnostics, unwrap_env_with_robot


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if not args_cli.keep_velocity_termination:
        disable_velocity_termination_for_evaluation(env_cfg)

    def wrap_raw(raw_env):
        if not args_cli.video:
            return raw_env
        return gym.wrappers.RecordVideo(
            raw_env,
            video_folder=args_cli.video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env, teacher_agent = build_env_and_teacher(
        args_cli.task,
        args_cli.adapter,
        env_cfg,
        agent_cfg,
        args_cli.teacher_checkpoint,
        args_cli.device,
        render_mode="rgb_array" if args_cli.video else None,
        wrap_raw=wrap_raw,
    )
    core = unwrap_env_with_robot(env) or getattr(env, "unwrapped", None)
    if core is None or not hasattr(core, "robot"):
        raise RuntimeError("Unable to find the G1 environment exposing `robot`")
    estimator = payload = None
    history = None
    if args_cli.replay_action_log is None:
        teacher_agent.enable_training_mode(False, apply_to_models=True)
        estimator, payload = load_estimator(args_cli.estimator_checkpoint, args_cli.device)
        adapter = make_policy_adapter(args_cli.adapter, env, payload["joint_preset"])
        if payload["task"] != args_cli.task or payload["adapter"] != args_cli.adapter:
            raise ValueError("Estimator checkpoint task/adapter does not match the requested environment")
        if payload["model_config"]["output_dim"] != adapter.schema.estimator_target_dim:
            raise ValueError("Estimator checkpoint does not predict the complete privileged schema")
        if payload["observation_schema"] != adapter.schema.to_dict():
            raise ValueError("Estimator observation schema does not match the requested policy")
    else:
        adapter = make_policy_adapter(args_cli.adapter, env, "all")
    replay_actions = None
    if args_cli.replay_action_log:
        with np.load(args_cli.replay_action_log) as data:
            replay_actions = np.asarray(data["actions"], dtype=np.float32)
        if replay_actions.ndim != 2 or replay_actions.shape[1] != adapter.schema.action_dim:
            raise ValueError(f"Action log must have shape [T, {adapter.schema.action_dim}]")

    observations, _ = env.reset()
    if payload is not None:
        history = HistoryBuffer(observations.shape[0], payload["window"], adapter.input_dim, observations.device)
    log_index = min(max(args_cli.log_env_id, 0), observations.shape[0] - 1)
    csv_stream = None
    csv_writer = None
    if args_cli.csv_output:
        csv_path = Path(args_cli.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_stream = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_stream)
        csv_writer.writerow(
            ["step", *[f"action_{i}" for i in range(adapter.schema.action_dim)],
             *[f"estimate_{name}" for name in adapter.schema.estimator_target_names]]
        )
    action_log: list[np.ndarray] = []
    # RolloutDiagnostics reads the Direct envs' action-to-position mapping, which the
    # manager-based walk env expresses through its action manager instead.
    diagnostics = (
        None
        if uses_rsl_rl(args_cli.adapter)
        else RolloutDiagnostics(core, log_index, max_steps=max(args_cli.steps, args_cli.video_length))
    )
    step_limit = args_cli.video_length if args_cli.video else args_cli.steps
    try:
        for step in range(step_limit):
            if not simulation_app.is_running():
                break
            started = time.monotonic()
            with torch.inference_mode():
                estimate = None
                if replay_actions is not None:
                    if step >= len(replay_actions):
                        break
                    action = torch.as_tensor(replay_actions[step], device=observations.device).repeat(
                        observations.shape[0], 1
                    )
                else:
                    frame = adapter.estimator_input()
                    sequence = history.push(frame)
                    model_input = frame if payload["model_config"]["type"].upper() == "MLP" else sequence
                    estimate = estimator.predict(model_input)
                    action = adapter.action(teacher_agent, adapter.inject_estimate(observations, estimate))
                observations, _, terminated, truncated, _ = env.step(action)
                done = (terminated | truncated).flatten()
                if history is not None and done.any():
                    history.reset(done)
            if diagnostics is not None:
                diagnostics.record(action)
            action_log.append(action[log_index].detach().cpu().numpy().copy())
            if csv_writer is not None:
                estimate_values = [] if estimate is None else estimate[log_index].detach().cpu().tolist()
                csv_writer.writerow([step, *action[log_index].detach().cpu().tolist(), *estimate_values])
            if args_cli.real_time:
                remaining = float(core.step_dt) - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if csv_stream is not None:
            csv_stream.close()
        if args_cli.action_log_output:
            output = Path(args_cli.action_log_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                actions=np.asarray(action_log),
                joint_names=np.asarray(core.robot.data.joint_names),
                policy_dt=float(core.step_dt),
            )
        if diagnostics is not None:
            diagnostic_dir = args_cli.diagnostic_output or args_cli.video_dir
            artifacts = diagnostics.save(diagnostic_dir, float(core.step_dt), "teacher_estimator_diagnostics")
            if artifacts:
                print(f"Diagnostics: {artifacts}")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
