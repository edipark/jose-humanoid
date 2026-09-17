"""Self-contained Unitree G1 29-DOF articulation with TWIST-aligned PD gains."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd")

G1_SOLO_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ASSET_DIR, "g1_29dof_rev_1_0.usd"),
        # SOLO derives rewards from rigid-body state, not contact reports.
        # Disabling reports avoids reserving a large PhysX GPU buffer.
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.40,
            ".*_ankle_pitch_joint": -0.20,
            "left_shoulder_roll_joint": 0.40,
            "left_elbow_joint": 1.20,
            "right_shoulder_roll_joint": -0.40,
            "right_elbow_joint": 1.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # TWIST training/deployment PD gains, expressed as PhysX implicit drives.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee_joint"],
            effort_limit_sim={".*_hip_.*": 88.0, ".*_knee_joint": 139.0},
            velocity_limit_sim={".*_hip_.*": 32.0, ".*_knee_joint": 20.0},
            stiffness={".*_hip_.*": 100.0, ".*_knee_joint": 150.0},
            damping={".*_hip_.*": 2.0, ".*_knee_joint": 4.0},
            armature=0.03,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.03,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim={"waist_yaw_joint": 88.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0},
            velocity_limit_sim={"waist_yaw_joint": 32.0, "waist_roll_joint": 37.0, "waist_pitch_joint": 37.0},
            stiffness=150.0,
            damping=4.0,
            armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"],
            effort_limit_sim=300.0,
            velocity_limit_sim=100.0,
            stiffness={".*_shoulder_.*": 40.0, ".*_elbow_joint": 40.0, ".*_wrist_.*": 20.0},
            damping={".*_shoulder_.*": 5.0, ".*_elbow_joint": 5.0, ".*_wrist_.*": 1.0},
            armature=0.001,
        ),
    },
)

# Backwards-compatible JOSE names used by replay and task modules.
G1_CFG = G1_SOLO_CFG
G1_JOSE_CFG = G1_SOLO_CFG
