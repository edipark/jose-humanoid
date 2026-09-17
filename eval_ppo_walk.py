"""Fixed-command quantitative evaluation of a G1 PPO walk policy.

Playing a checkpoint in the viewer only tells you whether the robot looks alive.
This script answers the question the flat task actually poses: *does the policy
follow the commanded (vx, vy, yaw)?* It holds each command constant across every
environment for a fixed window and reports tracking error, survival, gait
statistics and the sanity checks that distinguish a walking policy from a
standing one.

Example
-------
    python eval_ppo_walk.py --task Isaac-G1-PPO-Walk-JOSE-v0 --headless \
        --num_envs 64 \
        --checkpoint logs/rsl_rl/isaac_g1_ppo_walk_jose_v0/<run>/model_499.pt \
        --output eval.json

A rising ``Train/mean_reward`` is not evidence of success: with ``std = 0.5``
kernels a motionless robot already collects ``exp(-|cmd|^2 / 0.25)`` every step.
Judge a run by this script and by ``Episode_Reward/feet_air_time``, which is zero
whenever the robot is not stepping.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

from jose.ppo_walk.utils import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Fixed-command evaluation of a G1 PPO walk policy.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments per command.")
parser.add_argument("--task", type=str, default="Isaac-G1-PPO-Walk-JOSE-v0", help="Name of the task.")
parser.add_argument("--settle_s", type=float, default=2.0, help="Seconds discarded after reset before measuring.")
parser.add_argument("--measure_s", type=float, default=8.0, help="Seconds of measurement per command.")
parser.add_argument("--output", type=str, default=None, help="Optional path to write the results as JSON.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import importlib.metadata as metadata
import json
import os

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path

import jose  # noqa: F401
from jose.estimator.locomotion import (
    EVAL_COMMANDS,
    CommandEvaluator,
    resolve_feet_cfgs,
    run_sanity_checks,
)
from jose.ppo_walk.utils.parser_cfg import parse_env_cfg

installed_version = metadata.version("rsl-rl-lib")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    # deterministic measurement
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    feet_sensor_cfg, feet_asset_cfg = resolve_feet_cfgs(env.unwrapped.scene)
    print(f"[INFO] feet bodies: {feet_sensor_cfg.body_names}")

    dt = env.unwrapped.step_dt
    evaluator = CommandEvaluator(
        env,
        feet_sensor_cfg,
        feet_asset_cfg,
        settle_steps=int(round(args_cli.settle_s / dt)),
        measure_steps=int(round(args_cli.measure_s / dt)),
    )

    results = evaluator.run_all(policy, progress=lambda cmd: print(f"[INFO] evaluating command {cmd} ..."))

    # ---- report ----
    header = (
        f"{'command (vx,vy,yaw)':>22} | {'measured (vx,vy,yaw)':>24} | {'RMSE (vx,vy,yaw)':>22} |"
        f" {'surv':>5} {'fall':>5} {'h[m]':>6} {'air':>6} {'slide':>7} {'lift/s':>7} {'2stance':>8} {'drift':>7}"
    )
    print("\n" + "=" * len(header))
    print(f"checkpoint: {resume_path}")
    print(f"task: {args_cli.task}   num_envs: {args_cli.num_envs}   "
          f"settle: {args_cli.settle_s}s   measure: {args_cli.measure_s}s")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for cmd in EVAL_COMMANDS:
        v = results[str(cmd)]
        c, m, e = v["command"], v["measured"], v["error"]
        print(
            f"{c['vx']:>7.2f}{c['vy']:>7.2f}{c['yaw']:>8.2f} |"
            f"{m['vx']:>8.3f}{m['vy']:>8.3f}{m['yaw']:>8.3f} |"
            f"{e['vx_rmse']:>7.3f}{e['vy_rmse']:>7.3f}{e['yaw_rmse']:>8.3f} |"
            f" {v['survival_rate']:>5.2f} {v['fall_count']:>5d} {v['base_height_m']:>6.3f}"
            f" {v['feet_air_time_reward']:>6.3f} {v['feet_slide_penalty']:>7.3f}"
            f" {v['foot_lifts_per_s']:>7.2f} {v['double_stance_fraction']:>8.2f} {v['drift_m']:>7.3f}"
        )
    print("=" * len(header))

    print("\nSanity checks")
    print("-" * 13)
    checks = run_sanity_checks(results)
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}\n         {detail}")
    n_pass = sum(1 for _, p, _ in checks if p)
    print(f"\n{n_pass}/{len(checks)} sanity checks passed.")

    if args_cli.output:
        payload = {
            "checkpoint": resume_path,
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "settle_s": args_cli.settle_s,
            "measure_s": args_cli.measure_s,
            "results": results,
            "sanity_checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        }
        with open(args_cli.output, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[INFO] wrote {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
