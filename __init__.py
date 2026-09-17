"""JOSE G1 AMP, PPO, and state-estimation environments."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-G1-AMP-Walk-JOSE-Direct-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_walk_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_walk_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-G1-AMP-Jump-JOSE-Direct-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpJumpEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_jump_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_jump_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-G1-AMP-Dance-JOSE-Direct-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_dance_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_dance_amp_cfg.yaml",
    },
)

# Flat-terrain velocity-tracking walk task. Runs on ManagerBasedRLEnv and trains
# with rsl-rl through `train_ppo_walk.py`; see `ppo_walk/walk_env_cfg.py` for why
# its reward set diverges from the original unitree_rl_lab port.
gym.register(
    id="Isaac-G1-PPO-Walk-JOSE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ppo_walk.walk_env_cfg:G1WalkEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.ppo_walk.walk_env_cfg:G1WalkPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_walk.agents.rsl_rl_ppo_cfg:G1WalkPPORunnerCfg",
    },
)

# Same recipe with `base_lin_vel` added to the policy observation group, giving
# the JOSE state estimator a slot to write into.
gym.register(
    id="Isaac-G1-PPO-Walk-Estimator-JOSE-v0",
    entry_point=f"{__name__}.ppo_walk.walk_estimator_env:G1WalkEstimatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ppo_walk.walk_estimator_env_cfg:G1WalkEstimatorEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.ppo_walk.walk_estimator_env_cfg:G1WalkEstimatorPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_walk.agents.rsl_rl_ppo_cfg:G1WalkEstimatorPPORunnerCfg",
    },
)

# The terrain and push variants register themselves when ``ppo_walk`` is
# imported. Those modules import gymnasium only, so this is safe before
# ``AppLauncher``.
from . import ppo_walk  # noqa: E402,F401
