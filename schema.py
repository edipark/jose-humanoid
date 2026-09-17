"""Stable observation and joint schemas shared by JOSE G1 tasks and tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


SCHEMA_VERSION = 2

# Canonical order exposed by the 29-DOF G1 asset. Reference NPZ files are
# name-mapped because the published walk and dance files use different orders.
G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_LEG_JOINT_NAMES = tuple(
    name
    for name in G1_JOINT_NAMES
    if any(token in name for token in ("hip_", "knee_", "ankle_"))
)
G1_UPPER_JOINT_NAMES = tuple(name for name in G1_JOINT_NAMES if name not in G1_LEG_JOINT_NAMES)

JOINT_PRESETS: Mapping[str, tuple[str, ...]] = {
    "all": G1_JOINT_NAMES,
    "legs": G1_LEG_JOINT_NAMES,
    "upper": G1_UPPER_JOINT_NAMES,
}

G1_KEY_BODY_NAMES = (
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "right_hip_yaw_link",
    "left_hip_yaw_link",
    "right_rubber_hand",
    "left_rubber_hand",
    "right_ankle_roll_link",
    "left_ankle_roll_link",
)

AMP_PRIVILEGED_NAMES = (
    "base_height",
    "base_tangent_x",
    "base_tangent_y",
    "base_tangent_z",
    "base_normal_x",
    "base_normal_y",
    "base_normal_z",
    "base_lin_vel_x",
    "base_lin_vel_y",
    "base_lin_vel_z",
    "base_ang_vel_x",
    "base_ang_vel_y",
    "base_ang_vel_z",
    *tuple(f"{body}_{axis}" for body in G1_KEY_BODY_NAMES for axis in ("x", "y", "z")),
)


@dataclass(frozen=True)
class ObservationSchema:
    """Serializable policy/estimator interface contract."""

    name: str
    policy_dim: int
    action_dim: int = 29
    joint_position_start: int = 0
    joint_velocity_start: int = 29
    estimator_target_indices: tuple[int, ...] = ()
    estimator_target_names: tuple[str, ...] = (
        "base_lin_vel_x",
        "base_lin_vel_y",
        "base_lin_vel_z",
        "base_ang_vel_x",
        "base_ang_vel_y",
        "base_ang_vel_z",
        "projected_gravity_x",
        "projected_gravity_y",
        "projected_gravity_z",
    )
    velocity_source: str = "sim_joint_velocity"
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.estimator_target_indices) != len(self.estimator_target_names):
            raise ValueError("Estimator target indices and names must have equal length")
        if len(set(self.estimator_target_indices)) != len(self.estimator_target_indices):
            raise ValueError("Estimator target indices must be unique")
        if self.estimator_target_indices and max(self.estimator_target_indices) >= self.policy_dim:
            raise ValueError("Estimator target index is outside the policy observation")
        if self.velocity_source != "sim_joint_velocity":
            raise ValueError("JOSE G1 only supports simulator joint velocities")

    @property
    def estimator_target_dim(self) -> int:
        return len(self.estimator_target_indices)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "ObservationSchema":
        values = dict(data)
        values["estimator_target_indices"] = tuple(values.get("estimator_target_indices", ()))
        values["estimator_target_names"] = tuple(values.get("estimator_target_names", ()))
        return cls(**values)


# AMP layout: q(29), qd(29), height(1), tangent(3), normal/gravity(3),
# root linear velocity(3), root angular velocity(3), key bodies(30).
AMP_OBSERVATION_SCHEMA = ObservationSchema(
    name="g1_amp_101",
    policy_dim=101,
    estimator_target_indices=tuple(range(58, 101)),
    estimator_target_names=AMP_PRIVILEGED_NAMES,
)

def joint_indices(robot_joint_names: Sequence[str], preset: str = "all") -> tuple[int, ...]:
    """Resolve a preset by name, rejecting missing, duplicate, or ambiguous joints."""
    if preset not in JOINT_PRESETS:
        raise ValueError(f"Unknown joint preset {preset!r}; choose from {tuple(JOINT_PRESETS)}")
    names = tuple(robot_joint_names)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate robot joint names: {duplicates}")
    missing = [name for name in JOINT_PRESETS[preset] if name not in names]
    if missing:
        raise ValueError(f"Robot is missing {preset!r} estimator joints: {missing}")
    return tuple(names.index(name) for name in JOINT_PRESETS[preset])


def estimator_input_dim(preset: str = "all") -> int:
    if preset not in JOINT_PRESETS:
        raise ValueError(f"Unknown joint preset {preset!r}")
    return 2 * len(JOINT_PRESETS[preset])


# Manager-based PPO walk layout. The policy group keeps a 5-step history and the
# observation manager flattens each term's history separately (oldest first), so
# the group is a concatenation of per-term blocks rather than per-frame records:
#
#   base_lin_vel(5x3)  base_ang_vel(5x3)  projected_gravity(5x3)
#   velocity_commands(5x3)  joint_pos_rel(5x29)  joint_vel_rel(5x29)
#   last_action(5x29)                                        = 495
#
# The estimator target indices below address the newest frame of the three base
# state terms; the adapter widens them across the whole history at injection time.
PPO_WALK_HISTORY_LENGTH = 5
PPO_WALK_TERM_LAYOUT = (
    # (name, per-frame dim, block start, observation scale applied by the manager)
    ("base_lin_vel", 3, 0, 1.0),
    ("base_ang_vel", 3, 15, 0.2),
    ("projected_gravity", 3, 30, 1.0),
    ("velocity_commands", 3, 45, 1.0),
    ("joint_pos_rel", 29, 60, 1.0),
    ("joint_vel_rel", 29, 205, 0.05),
    ("last_action", 29, 350, 1.0),
)
PPO_WALK_ESTIMATOR_TERMS = ("base_lin_vel", "base_ang_vel", "projected_gravity")

PPO_WALK_OBSERVATION_SCHEMA = ObservationSchema(
    name="g1_ppo_walk_495",
    policy_dim=495,
    joint_position_start=176,  # newest frame of joint_pos_rel
    joint_velocity_start=321,  # newest frame of joint_vel_rel
    estimator_target_indices=(12, 13, 14, 27, 28, 29, 42, 43, 44),
)


def ppo_walk_history_target_indices() -> tuple[int, ...]:
    """Every history slot of the three base-state terms, newest frame last.

    Injecting only the newest frame would leave four frames of ground-truth
    privileged state in the policy input, so closed-loop evaluation replaces the
    complete history block of each estimated term.
    """
    layout = {name: (dim, start) for name, dim, start, _ in PPO_WALK_TERM_LAYOUT}
    indices: list[int] = []
    for frame in range(PPO_WALK_HISTORY_LENGTH):
        for name in PPO_WALK_ESTIMATOR_TERMS:
            dim, start = layout[name]
            offset = start + frame * dim
            indices.extend(range(offset, offset + dim))
    return tuple(indices)


def ppo_walk_target_scales() -> tuple[float, ...]:
    """Observation scale of each of the 9 estimator target columns.

    ``base_ang_vel`` is stored scaled by 0.2, so an estimate expressed in raw
    units must be scaled the same way before it is written into the observation.
    """
    layout = {name: scale for name, _, _, scale in PPO_WALK_TERM_LAYOUT}
    scales: list[float] = []
    for name in PPO_WALK_ESTIMATOR_TERMS:
        scales.extend([layout[name]] * 3)
    return tuple(scales)
