"""Gym registrations for the push-disturbed locomotion task.

Importing this module registers the ids; ``ppo_walk/__init__.py`` does so.
"""

import gymnasium as gym

#: Root package name, derived rather than hardcoded (see ``terrain_tasks``).
_PKG = __name__.rsplit(".", 2)[0]


gym.register(
    id="Isaac-G1-PPO-Walk-Push-JOSE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PKG}.ppo_walk.push_env_cfg:G1WalkPushEnvCfg",
        "play_env_cfg_entry_point": f"{_PKG}.ppo_walk.push_env_cfg:G1WalkPushPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PKG}.ppo_walk.agents.rsl_rl_ppo_cfg:G1WalkPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-PPO-Walk-Push-Estimator-JOSE-v0",
    entry_point=f"{_PKG}.ppo_walk.walk_estimator_env:G1WalkEstimatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PKG}.ppo_walk.push_env_cfg:G1WalkPushEstimatorEnvCfg",
        "play_env_cfg_entry_point": f"{_PKG}.ppo_walk.push_env_cfg:G1WalkPushEstimatorPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PKG}.ppo_walk.agents.rsl_rl_ppo_cfg:G1WalkEstimatorPPORunnerCfg",
    },
)
