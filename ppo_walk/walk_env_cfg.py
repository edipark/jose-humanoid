"""Flat-terrain velocity-tracking environment for the G1 29-DOF robot.

Originally ported from unitree_rl_lab (Apache License 2.0),
https://github.com/unitreerobotics/unitree_rl_lab

The scene, action space, terminations and PPO hyper-parameters are still the
upstream ones. The reward set, the command sampler and the terrain are not: the
upstream recipe converged to a policy that stands still and ignores the velocity
command, and this file is the configuration that was measured to actually walk.

What changed from the upstream port, and why
--------------------------------------------
``foot_clearance_reward`` is gone. It computes

    exp( -sum( (foot_z - target)^2 * tanh(k * |v_foot,xy|) ) / std )

and ``tanh(...)`` is zero whenever a foot is not moving, so the term returns
``exp(0) = 1`` -- its global maximum -- for a robot standing on two feet. In run
``2026-08-28_23-06-25`` it saturated at 0.787 of a 0.81 ceiling, 69% of the net
episode reward. Upstream tracks the same defect in unitree_rl_lab#80. It is
replaced by Isaac Lab's ``feet_air_time_positive_biped``, which pays nothing in
double stance and nothing at zero command.

Removing it alone is not enough -- that was measured too. Four further changes
were each required, and the remaining reward set follows the official Isaac Lab
G1 flat config (``isaaclab_tasks/manager_based/locomotion/velocity/config/g1/``):

1. ``termination_penalty`` (-200) added. ``foot_clearance_reward`` had been
   acting as an unconditional survival bonus worth up to +1.0/s; without it an
   untrained policy's reward rate is net negative and PPO learns to topple
   immediately. Measured: mean episode length peaked at 47 steps around
   iteration 5 and decayed to ~10 by iteration 45, ``bad_orientation`` at 100%
   of terminations. With the penalty it climbs 14 -> 382 by iteration 120.
2. ``joint_vel`` (``joint_vel_l2``, -0.001) removed. Absent from official G1,
   and the largest single penalty in every run at -0.13 to -0.17/s. It taxes
   exactly what a gait needs: fast leg swing.
3. ``alive`` (+0.15) removed. Absent from official G1, which relies on the
   termination penalty alone. A flat bonus per surviving step pays for standing.
4. Posture penalties brought to official weights: ``joint_deviation_legs`` and
   ``joint_deviation_waists`` -1.0 -> -0.1, ``flat_orientation_l2`` -5.0 -> -1.0,
   ``base_height`` -10.0 -> -1.0. Each opposed a motion a biped gait requires --
   hip roll weight shift, waist balance, torso lean, pelvis rise and fall.

The command ranges are the official G1 flat ones. This matters more than it
looks: the tracking kernels use ``std = 0.5``, so a *motionless* robot collects
``exp(-|cmd|^2 / 0.25)`` every step, and how much depends entirely on the command
distribution. Measured over 2e6 samples, the reward per second a statue earns
from the two tracking terms alone is 1.293 under the original curriculum, 1.246
with merely-widened ranges, and 0.795 with these -- the yaw range dominates,
because +-0.5 rad/s against a 0.5 kernel still hands a statue 75% of the angular
reward.

Verified behaviour at 500 iterations (``eval_ppo_walk.py``, 64 envs): commanded
vx 0.0 / 0.3 / 0.6 measures 0.13 / 0.27 / 0.38 m/s, yaw +-0.3 measures +-0.24
rad/s, ~11 foot lifts/s, zero falls. The known remaining defect is that it
marches in place and creeps at 0.13 m/s under a zero command; fix that by
fine-tuning a checkpoint that already walks, not by re-tuning from scratch.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .g1_asset import G1_29DOF_CFG as ROBOT_CFG

#: Both feet. Resolves to ``left_ankle_roll_link`` and ``right_ankle_roll_link``;
#: the biped air-time reward requires exactly two bodies.
FEET_BODIES = ".*ankle_roll.*"


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat ground, the G1, and foot contact sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Reset and randomisation events.

    This is the verification build: friction randomisation, base-mass
    randomisation and random pushes are all off so that command tracking can be
    measured without confounds. Re-enable them once tracking is confirmed --
    widen ``physics_material``'s ranges back to (0.3, 1.0), and restore
    ``add_base_mass`` and ``push_robot`` from the git history of this file.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class CommandsCfg:
    """Direct yaw-rate velocity command over the official Isaac Lab G1 flat ranges.

    No curriculum. The upstream ``UniformLevelVelocityCommandCfg`` started at
    +-0.1 m/s, well inside the ``std = 0.5`` bandwidth of the tracking kernel, so
    a standing robot already scored ``exp(-0.01/0.25) = 0.96`` of the tracking
    reward and had no gradient pushing it to walk.
    """

    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations.

        ``base_lin_vel`` is first on purpose. The term order defines the
        flattened layout, so keeping it in front lets a narrower actor be
        warm-started into this one (see ``ppo_walk/utils/checkpoint.py``) and
        matches the layout ``jose.schema.PPO_WALK_OBSERVATION_SCHEMA`` describes
        for the estimator variant.

        Deployment note: base linear velocity is not directly measurable on the
        real G1. For a deployable policy, train
        ``Isaac-G1-PPO-Walk-Estimator-JOSE-v0`` and feed the slot from
        ``train_state_estimator.py``, or drop the term and rely on the five-step
        history or the teacher/student path in ``distillation/``.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the critic."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP. See the module docstring for the derivation."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Replaces the degenerate ``foot_clearance_reward``: zero in double stance,
    # zero when |cmd_xy| < 0.1, so it can never pay for standing still.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
            "threshold": 0.4,
        },
    )

    # Without this, dropping ``foot_clearance_reward`` makes falling immediately
    # the highest-reward policy. See the module docstring.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.78})

    # -- feet
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class G1WalkEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # tick the contact sensor at the physics rate
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class G1WalkPlayEnvCfg(G1WalkEnvCfg):
    """Deterministic playback and evaluation build."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        # long episode so the fixed-command evaluator sees no time-outs
        self.episode_length_s = 60.0
