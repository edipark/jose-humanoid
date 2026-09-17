"""Environment configurations for G1 AMP walk, dance, and jump."""

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .g1_cfg import G1_SOLO_CFG
from .task_math import (
    AMP_HISTORY_STEPS,
    CONTROL_DECIMATION,
    EPISODE_LENGTH_S,
    PHYSICS_DT,
    TWIST_ACTION_SCALE,
)


MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")
G1_AMP_ROBOT_CFG = G1_SOLO_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    episode_length_s = EPISODE_LENGTH_S
    decimation = CONTROL_DECIMATION
    observation_space = 101
    action_space = 29
    action_scale = TWIST_ACTION_SCALE
    state_space = 0
    num_amp_observations = AMP_HISTORY_STEPS
    amp_observation_space = 101

    early_termination = True
    termination_height = 0.55
    vel_window_min_vx = 0.0
    vel_window_steps = 10
    motion_file: str = MISSING
    reference_body = "pelvis"
    reset_strategy = "default"
    # Excludes the last `reset_time_end_margin` seconds of the clip from random
    # reset sampling. 0.0 preserves prior behavior everywhere except where a
    # subclass overrides it.
    reset_time_end_margin = 0.0

    # World +X velocity task reward. Disabled in the base/dance/jump task and
    # enabled by G1AmpWalkEnvCfg, matching SOLO's reward equation.
    target_velocity = 0.6
    velocity_tracking_coeff = 2.0
    velocity_reward_weight = 0.0
    # mean((action[t] - action[t - 1]) ** 2) across joints.
    action_rate_penalty_weight = 0.0
    # mean((action[t] - 2 * action[t - 1] + action[t - 2]) ** 2) across joints.
    action_second_difference_penalty_weight = 0.0

    sim: SimulationCfg = SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=decimation,
        physx=PhysxCfg(gpu_found_lost_pairs_capacity=2**23, gpu_total_aggregate_pairs_capacity=2**23),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    robot = G1_AMP_ROBOT_CFG


@configclass
class G1AmpWalkEnvCfg(G1AmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "G1_walk.npz")
    velocity_reward_weight = 0.5
    action_rate_penalty_weight = 0.0
    action_second_difference_penalty_weight = 0.1


@configclass
class G1AmpDanceEnvCfg(G1AmpEnvCfg):
    robot = G1_AMP_ROBOT_CFG.replace(soft_joint_pos_limit_factor=1.0)
    motion_file = os.path.join(MOTIONS_DIR, "G1_dance1.npz")
    reset_strategy = "random_start"
    # G1_dance1.npz is 10 s long; the inherited 20 s episode asked the policy
    # to sustain a non-periodic clip for 2x its own length, the same fixed-point
    # incentive that made the jump task collapse before its episode length was
    # matched to its motion. 10 s also doubles reset-phase diversity per env.
    episode_length_s = 10.0
    action_second_difference_penalty_weight = 0.05



@configclass
class G1AmpJumpEnvCfg(G1AmpEnvCfg):
    """SOLO-aligned AMP with the jump motion and a jump-safe pelvis threshold."""

    robot = G1_AMP_ROBOT_CFG.replace(soft_joint_pos_limit_factor=1.0)
    motion_file = os.path.join(MOTIONS_DIR, "G1_jump.npz")
    reset_strategy = "random"
    episode_length_s = 10.0
    # The reference pelvis dips to 0.241 m at the bottom of the countermovement
    # crouch that launches each jump. A threshold above that outlaws the crouch:
    # at 0.28 it cut 2.6% of reference frames, the measured height-death rate was
    # 2.9%, and the policy backed away to a 0.361 m minimum and could only manage
    # a 0.194 s hop against the reference's 0.708 s flight. Keep this below 0.241.
    termination_height = 0.2
    action_second_difference_penalty_weight = 0.0
    # G1_jump.npz cuts off mid-motion: at t=9.0s (the last frame) the pelvis is
    # still rising at +0.61 m/s with joint speeds matching mid-jump, not settled.
    # Resetting into that state handed the policy a combination of pose and
    # momentum it otherwise never sees, and both of the height-deaths measured
    # for agent_50000.pt at 4096-env scale were resets landing at 99.9%/100% of
    # the clip. 0.25 s excludes that tail without touching real jump phases.
    reset_time_end_margin = 0.25
    # A longer AMP window was tried here and did not work. The reasoning was that
    # the shared 4-step window spans only 60 ms, about 8% of one flight phase, so
    # the discriminator judges near-instantaneous poses and a short hop looks as
    # real as a long jump. `num_amp_observations = 12` (220 ms) was trained to
    # 200k timesteps and measured: it produced the deepest crouch of any run
    # (pelvis min 0.28 m, mean 0.45 m against the reference's 0.24 / 0.50) but
    # almost stopped jumping -- 6.2% airborne and 1.8 jumps/10 s, against 25.7%
    # and 7.1 for the same threshold at 4 steps. It was not undertrained: the
    # style reward rose from 0.45 to 0.55 and then flattened from 140k on, and the
    # 100k and 200k checkpoints measure identically. Widening the window appears
    # to strengthen the discriminator faster than the policy can follow, and the
    # policy settles for a safe crouch. Keep the shared 4-step window.
