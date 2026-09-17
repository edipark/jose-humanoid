"""MDP terms for the G1 walk task.

Re-exports Isaac Lab's locomotion terms and adds the JOSE-specific rewards and
observations on top. The upstream ``UniformLevelVelocityCommandCfg`` and the
``lin_vel_cmd_levels`` / ``terrain_levels_vel`` curricula were removed with the
terrain generator: the flat task samples its full command range from step 0.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .rewards import *  # noqa: F401, F403
