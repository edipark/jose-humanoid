"""Estimator-compatible variant of the G1 29-DOF walk environment.

This is :mod:`walk_env_cfg` with two changes, both about observation noise.

First, the *policy* observation group carries a **noiseless** ``base_lin_vel``
term. The base walk task adds observation noise to it, which is right for a
policy that will be handed a learned estimate; here the term is the privileged
quantity the estimator is trained to reproduce, so it must be clean.

Second, ``enable_corruption`` is **off**. The same argument applies to the other
two estimated terms: ``base_ang_vel`` and ``projected_gravity`` are estimator
targets too, and ``estimator/adapters.py`` overwrites their entire history block
at injection time. JOSE therefore never reads their corrupted values while the
teacher baseline does -- an asymmetry that let the student outscore the teacher
it distils from. With corruption off, teacher and student read the same
observations and the only difference between them is true versus estimated base
state. It also matches the AMP tasks, whose observations carry no noise at all.
Sensor noise, if wanted, belongs in a dedicated evaluation that applies it
deliberately and measures the result. The per-term ``noise=``
arguments below are left in place: they are inert while corruption is off and
they document the noise model the teacher was trained under.

Rewards, actions, commands, events, terminations and every algorithm
hyperparameter are inherited unchanged, so this variant automatically tracks the
base task's reward set.

Do not reorder or resize this group. :data:`jose.schema.PPO_WALK_OBSERVATION_SCHEMA`
hardcodes the resulting 495-D layout --

    base_lin_vel(5x3) base_ang_vel(5x3) projected_gravity(5x3)
    velocity_commands(5x3) joint_pos_rel(5x29) joint_vel_rel(5x29) last_action(5x29)

-- and ``estimator/adapters.py`` injects estimates at fixed indices derived from
it. Changing the term order or the history length silently corrupts the
estimator's injection targets; ``test_jose.py`` guards the layout.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .walk_env_cfg import G1WalkEnvCfg, G1WalkPlayEnvCfg, ObservationsCfg


@configclass
class EstimatorPolicyCfg(ObsGroup):
    """Policy observations with the estimated base linear velocity in front.

    Term order defines the flattened layout, so ``base_lin_vel`` is declared
    first and every following term keeps its original relative order.
    """

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
    last_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.history_length = 5
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class EstimatorObservationsCfg(ObservationsCfg):
    """Same observation groups as the walk task, with the extended policy group."""

    policy: EstimatorPolicyCfg = EstimatorPolicyCfg()


@configclass
class G1WalkEstimatorEnvCfg(G1WalkEnvCfg):
    observations: EstimatorObservationsCfg = EstimatorObservationsCfg()


@configclass
class G1WalkEstimatorPlayEnvCfg(G1WalkPlayEnvCfg):
    observations: EstimatorObservationsCfg = EstimatorObservationsCfg()
