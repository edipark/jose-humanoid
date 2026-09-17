"""Terrain-relative MDP terms for the sloped locomotion variant.

Isaac Lab's :func:`~isaaclab.envs.mdp.terminations.root_height_below_minimum`
says so itself: *"This is currently only supported for flat terrains, i.e. the
minimum height is in the world frame."* On the sloped terrain built in
``ppo_walk/terrain_env_cfg.py`` the ground moves through roughly ``±1.2`` m --
a pyramid of slope ``0.4`` rises ``0.4 * (8 - 2) / 2`` m from platform to edge,
and the inverted pyramid falls by the same -- so an absolute threshold of
``0.2`` m would kill an upright robot standing in a valley and would never fire
for one lying on a summit. Survival would stop meaning "did not fall".

The upstream answer for rough terrain is to terminate on base contact instead
(``velocity_env_cfg.py:271``). Here survival keeps its flat-task definition, the
base dropping below a task-specific height, so sloped and flat results measure
the same event. Only the *reference* is made terrain-relative, using the same
ray-cast ground height that :func:`~isaaclab.envs.mdp.rewards.base_height_l2`
already accepts a sensor for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ground_height_below(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Mean ray-cast ground height under the scanner, one value per environment.

    Matches how ``base_height_l2`` reads the same sensor
    (``isaaclab/envs/mdp/rewards.py:118``), so the reward and the termination
    below are referenced to an identical ground estimate rather than to two
    slightly different ones.
    """
    sensor: RayCaster = env.scene[sensor_cfg.name]
    return torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)


def root_height_below_minimum_terrain(
    env: "ManagerBasedRLEnv",
    minimum_height: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base is ``minimum_height`` above the ground beneath it.

    Drop-in replacement for ``mdp.root_height_below_minimum`` that subtracts the
    local ground height, so the threshold means the same thing on a slope as on
    a plane. On flat ground the two agree exactly: the ray hits ``z = 0``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ground = ground_height_below(env, sensor_cfg)
    return (asset.data.root_pos_w[:, 2] - ground) < minimum_height
