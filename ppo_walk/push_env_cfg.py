"""Push-disturbed variant of the G1 locomotion environment.

A shove to the base loads the stance-foot contact that JOSE's estimate relies
on, and a hard enough one breaks it (the foot slips, or the robot steps).
Friction varies whether the contact holds under the robot's own gait, slope what
the contact is fixed to; a push loads it with a force the policy did not choose.

Everything below subclasses :mod:`walk_env_cfg` and :mod:`walk_estimator_env_cfg`.

The push is a base-velocity change of up to 0.5 m/s in x and y, at an interval
sampled from (1.0, 4.0) s rather than a fixed 5.0 s. The command-grid evaluation
resets the environment before each command and then runs 1 s of settle plus 4 s
of measurement, so a timer that restarts at 5.0 s on reset would fire at the very
end of the window or after it: the grid would never see a push. The grid is what
picks JOSE's best DAgger round, so under a fixed 5.0 s interval the round kept
would be the one that tracks best *undisturbed*. The sampled interval puts at
least one push inside every grid command's measurement window, and roughly
eight into each 20 s training episode.

Base mass and friction stay pinned at the flat task's values: one variable.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from . import mdp
from .walk_env_cfg import EventCfg, G1WalkEnvCfg, G1WalkPlayEnvCfg
from .walk_estimator_env_cfg import EstimatorObservationsCfg

#: Seconds between pushes, resampled per environment after every push and reset.
PUSH_INTERVAL_S = (1.0, 4.0)
#: Base-velocity change per push (m/s), commit ``a002570``'s ``push_robot`` range.
PUSH_VELOCITY_RANGE = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}


@configclass
class PushEventCfg(EventCfg):
    """The base events plus ``push_robot``; every inherited term is untouched."""

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=PUSH_INTERVAL_S,
        params={"velocity_range": PUSH_VELOCITY_RANGE},
    )


@configclass
class G1WalkPushEnvCfg(G1WalkEnvCfg):
    """Locomotion on a plane with random base-velocity pushes."""

    events: PushEventCfg = PushEventCfg()


@configclass
class G1WalkPushPlayEnvCfg(G1WalkPlayEnvCfg):
    """Playback and teacher gate: pushes off, like the play build's other disturbances.

    The gate's checks (a zero command must not drift, measured vx must follow the
    command) are statements about undisturbed command following; a push would
    fail them by design.
    """

    events: PushEventCfg = PushEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None


@configclass
class G1WalkPushEstimatorEnvCfg(G1WalkPushEnvCfg):
    observations: EstimatorObservationsCfg = EstimatorObservationsCfg()


@configclass
class G1WalkPushEstimatorPlayEnvCfg(G1WalkPushPlayEnvCfg):
    observations: EstimatorObservationsCfg = EstimatorObservationsCfg()
