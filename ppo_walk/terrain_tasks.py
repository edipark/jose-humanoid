"""Gym registrations for the friction-randomized and sloped locomotion tasks.

Importing this module registers the ids; ``ppo_walk/__init__.py`` does so.
"""

import gymnasium as gym

#: Root package name, derived rather than hardcoded so a checkout installed under
#: a different distribution name still resolves its own entry points.
_PKG = __name__.rsplit(".", 2)[0]


gym.register(
    id="Isaac-G1-PPO-Walk-Friction-JOSE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PKG}.ppo_walk.terrain_env_cfg:G1WalkFrictionEnvCfg",
        "play_env_cfg_entry_point": f"{_PKG}.ppo_walk.terrain_env_cfg:G1WalkFrictionPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PKG}.ppo_walk.agents.rsl_rl_ppo_cfg:G1WalkPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-PPO-Walk-Friction-Estimator-JOSE-v0",
    entry_point=f"{_PKG}.ppo_walk.walk_estimator_env:G1WalkEstimatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PKG}.ppo_walk.terrain_env_cfg:G1WalkFrictionEstimatorEnvCfg",
        "play_env_cfg_entry_point": f"{_PKG}.ppo_walk.terrain_env_cfg:G1WalkFrictionEstimatorPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PKG}.ppo_walk.agents.rsl_rl_ppo_cfg:G1WalkEstimatorPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-PPO-Walk-Slope-JOSE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PKG}.ppo_walk.terrain_env_cfg:G1WalkSlopeEnvCfg",
        "play_env_cfg_entry_point": f"{_PKG}.ppo_walk.terrain_env_cfg:G1WalkSlopePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PKG}.ppo_walk.agents.rsl_rl_ppo_cfg:G1WalkPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-PPO-Walk-Slope-Estimator-JOSE-v0",
    entry_point=f"{_PKG}.ppo_walk.walk_estimator_env:G1WalkEstimatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PKG}.ppo_walk.terrain_env_cfg:G1WalkSlopeEstimatorEnvCfg",
        "play_env_cfg_entry_point": f"{_PKG}.ppo_walk.terrain_env_cfg:G1WalkSlopeEstimatorPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PKG}.ppo_walk.agents.rsl_rl_ppo_cfg:G1WalkEstimatorPPORunnerCfg",
    },
)
