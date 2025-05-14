# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Hightorque robots.

The following configurations are available:

* :obj:`Pai_MINIMAL_CFG`: Pai biped robot with minimal collision bodies

Reference: https://github.com/hightorquerobotics/hightorque_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from legged_lab.assets import ISAAC_ASSET_DIR


Pai_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/hightorque/pai/pai.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3453),
        joint_pos={
            ".*_hip_pitch_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_thigh_joint": 0.0,
            ".*_calf_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_thigh_joint",
                ".*_calf_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 20.0,
                ".*_hip_roll_joint": 20.0,
                ".*_thigh_joint": 20.0,
                ".*_calf_joint": 20.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 21.0,
                ".*_hip_roll_joint": 21.0,
                ".*_thigh_joint": 21.0,
                ".*_calf_joint": 21.0,
            },
            stiffness={
                ".*_hip_pitch_joint": 20.0,
                ".*_hip_roll_joint": 20.0,
                ".*_thigh_joint": 20.0,
                ".*_calf_joint": 20.0,
            },
            damping={
                ".*_hip_pitch_joint": 0.5,
                ".*_hip_roll_joint": 0.5,
                ".*_thigh_joint": 0.5,
                ".*_calf_joint": 0.5,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 21.1,
                ".*_ankle_roll_joint": 21.1,
            },
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.5,
                ".*_ankle_roll_joint": 0.5,
            },
            armature=0.01,
        ),
    },
)
