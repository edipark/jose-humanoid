"""Friction-randomized and sloped variants of the G1 locomotion environment.

JOSE reconstructs the privileged base state from a window of joint encoders,
which works because the stance foot ties the base to a fixed frame. The flat
task pins friction at ``(1.0, 1.0)`` on a plane, the condition under which that
tie is strongest. These variants loosen it:

* **Friction** varies whether the contact holds. A slipping foot is not a fixed
  frame, so the joint chain stops determining base *velocity*.
* **Slope** varies what the contact is fixed *to*. Joint angles give base
  attitude relative to the ground; the ``projected_gravity`` component of the
  estimator target is attitude relative to *gravity*. On a plane those coincide.
  On a slope they differ by the slope angle, which no instantaneous joint reading
  contains -- only the history can carry it.

Everything below subclasses :mod:`walk_env_cfg` and :mod:`walk_estimator_env_cfg`.
Base mass and external pushes are not varied here, so each variant changes the
terrain and nothing else; pushes have their own variant in :mod:`push_env_cfg`.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp import terrain_mdp
from .walk_env_cfg import (
    EventCfg,
    G1WalkEnvCfg,
    G1WalkPlayEnvCfg,
    RewardsCfg,
    RobotSceneCfg,
    TerminationsCfg,
)
from .walk_estimator_env_cfg import EstimatorObservationsCfg


# =============================================================================
# Friction
# =============================================================================


@configclass
class FrictionEventCfg(EventCfg):
    """The base events with friction randomisation switched back on.

    The range is the one ``EventCfg``'s own docstring names: "widen
    ``physics_material``'s ranges back to (0.3, 1.0)". Restitution stays at zero
    and the two reset terms are inherited untouched.

    The event randomises the *robot's* material while the ground stays at 1.0
    with ``friction_combine_mode="multiply"``, so the sampled value is the
    effective contact coefficient.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class G1WalkFrictionEnvCfg(G1WalkEnvCfg):
    """Locomotion on a plane, with per-environment ground friction in [0.3, 1.0]."""

    events: FrictionEventCfg = FrictionEventCfg()


@configclass
class G1WalkFrictionPlayEnvCfg(G1WalkPlayEnvCfg):
    events: FrictionEventCfg = FrictionEventCfg()


@configclass
class G1WalkFrictionEstimatorEnvCfg(G1WalkFrictionEnvCfg):
    observations: EstimatorObservationsCfg = EstimatorObservationsCfg()


@configclass
class G1WalkFrictionEstimatorPlayEnvCfg(G1WalkFrictionPlayEnvCfg):
    observations: EstimatorObservationsCfg = EstimatorObservationsCfg()


# =============================================================================
# Slope
# =============================================================================

#: Slopes only, with a flat share so the curriculum has somewhere to start.
#:
#: Derived from ``isaaclab.terrains.config.rough.ROUGH_TERRAINS_CFG`` by keeping
#: its two sloped sub-terrains and dropping stairs, boxes and the random height
#: field. Two reasons, and the second is the binding one:
#:
#: 1. ``step_height_range=(0.05, 0.23)`` is not blind-walkable for a G1 within
#:    this training budget.
#: 2. Discontinuous terrain needs a height-scan *observation*, and adding one
#:    changes the 495-D policy layout that ``schema.py`` hardcodes, and with it
#:    the estimator interface. A slope is locally flat, so a blind policy trained
#:    on the unchanged observation set handles it.
#:
#: The scanner declared in the scene below is used only by the reward and the
#: termination. It never enters an observation group, so the layout is untouched.
SLOPE_TERRAINS_CFG = TerrainGeneratorCfg(
    # Pinned, and load-bearing. Left at ``None`` the generator seeds itself from
    # ``np.random.get_state()[1][0]`` -- whatever the global NumPy stream happens
    # to hold when the scene is built (``terrain_generator.py:139-144``). Two
    # consequences, both bad for this study: the terrain would differ between
    # training seeds, so seed-to-seed spread would mix terrain variation into
    # training variation; and it would differ between *methods*, so survival would
    # not be measured on a common test set. Worse, any trainer that does not seed
    # NumPy would draw an irreproducible terrain -- which is what
    # ``train_history_student.py`` did before the fix in this same change.
    #
    # Fixing it makes the terrain a property of the task, exactly as the reference
    # clip is for the AMP teachers, and leaves the training seed to vary only what
    # it is supposed to vary.
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    # Safe *because* the seed above is pinned: Isaac Lab refuses to promise
    # reproducibility for a cached terrain with an unset seed and warns about
    # exactly that pairing (``terrain_generator.py:132-136``). With both set, the
    # cache is a pure speed-up -- generating these 200 tiles costs several minutes
    # of CPU at scene build, and this study builds the scene once per training run
    # and once per evaluation, roughly fifteen times per terrain variant.
    use_cache=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.4, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.4, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)


@configclass
class SlopeSceneCfg(RobotSceneCfg):
    """The base scene with a generated slope and a termination-only height scanner."""

    # Restated rather than derived: ``@configclass`` turns these into dataclass
    # fields, so ``RobotSceneCfg.terrain`` is not readable as a class attribute.
    # The physics material is byte-identical to the base scene's, which keeps the
    # sloped variant's friction pinned at 1.0 -- the one thing it must share with
    # the flat task for the two to differ in terrain alone.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SLOPE_TERRAINS_CFG,
        max_init_terrain_level=SLOPE_TERRAINS_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # Attached to ``torso_link``: the G1 has no ``base`` link, which is also why
    # the upstream rough config's ``illegal_contact`` termination cannot be reused
    # verbatim. Rays are cast from 20 m up against the ground mesh only, so the
    # robot cannot occlude its own reading.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(0.2, 0.2)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class SlopeRewardsCfg(RewardsCfg):
    """The base rewards with the height target referenced to the ground below.

    ``base_height_l2`` compares absolute world ``z`` to ``target_height`` unless
    handed a sensor. Left alone on this terrain it would charge the robot for the
    elevation of the ground it is standing on -- a squared error of up to
    ``1.2^2`` at weight ``-1.0``, which swamps the entire tracking reward and
    trains a policy that seeks the altitude of the flat patches instead of
    following the command.
    """

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.78, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )


@configclass
class SlopeTerminationsCfg(TerminationsCfg):
    """The base terminations with the fall threshold referenced to the ground.

    See ``ppo_walk/mdp/terrain_mdp.py`` for why the flat-terrain term cannot be
    reused and why the definition of a fall is kept rather than replaced with the
    upstream base-contact rule.
    """

    base_height = DoneTerm(
        func=terrain_mdp.root_height_below_minimum_terrain,
        params={
            "minimum_height": 0.2,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )


@configclass
class SlopeCurriculumCfg:
    """Terrain difficulty rises with the distance actually walked.

    ``terrain_levels_vel`` reads only root position against ``env_origins`` and
    the commanded velocity, so it needs the terrain to be a ``generator`` but
    needs no height scan of its own.
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class G1WalkSlopeEnvCfg(G1WalkEnvCfg):
    """Locomotion over pyramid slopes of up to 0.4, blind."""

    scene: SlopeSceneCfg = SlopeSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: SlopeRewardsCfg = SlopeRewardsCfg()
    terminations: SlopeTerminationsCfg = SlopeTerminationsCfg()
    curriculum: SlopeCurriculumCfg = SlopeCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True


@configclass
class G1WalkSlopePlayEnvCfg(G1WalkSlopeEnvCfg):
    """Deterministic playback and evaluation build.

    Mirrors ``G1WalkPlayEnvCfg``: it cannot subclass it and pick up the sloped
    scene at the same time, so the three play settings are repeated here. The
    terrain shrinks and its curriculum is switched off, so every evaluation
    episode is drawn from the same fixed spread of slopes rather than from
    whatever difficulty the last training run happened to reach.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.episode_length_s = 60.0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.max_init_terrain_level = None


@configclass
class G1WalkSlopeEstimatorEnvCfg(G1WalkSlopeEnvCfg):
    observations: EstimatorObservationsCfg = EstimatorObservationsCfg()


@configclass
class G1WalkSlopeEstimatorPlayEnvCfg(G1WalkSlopePlayEnvCfg):
    observations: EstimatorObservationsCfg = EstimatorObservationsCfg()
