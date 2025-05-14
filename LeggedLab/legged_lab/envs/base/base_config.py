from dataclasses import MISSING
import math
from isaaclab.utils import configclass
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


@configclass
class RewardCfg:
    pass


@configclass
class gaitCfg:
    num_gait_params = 4
    resampling_time = 5

    class ranges:
        frequencies = [1.0, 2.0]
        offsets = [0.5, 0.5]
        durations = [0.5, 0.5]
        swing_height = [0.05, 0.1]


@configclass
class HeightScannerCfg:
    enable_height_scan: bool = False
    prim_body_name: str = MISSING
    resolution: float = 0.1
    size: tuple = (1.6, 1.0)
    debug_vis: bool = False
    drift_range: tuple = (0.0, 0.0)


@configclass
class BaseSceneCfg:
    max_episode_length_s: float = 20.0
    num_envs: int = 4096
    env_spacing: float = 2.5
    robot: ArticulationCfg = MISSING
    terrain_type: str = MISSING
    terrain_generator: TerrainGeneratorCfg = None
    max_init_terrain_level: int = 5
    height_scanner: HeightScannerCfg = HeightScannerCfg()


@configclass
class RobotCfg:
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_scale: float = 0.25
    terminate_contacts_body_names: list = []
    feet_body_names: list = []


@configclass
class ObsScalesCfg:
    lin_vel: float = 1.0
    ang_vel: float = 1.0
    projected_gravity: float = 1.0
    commands: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actions: float = 1.0
    height_scan: float = 1.0


@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    height_scan_offset: float = 0.5


@configclass
class CommandRangesCfg:
    lin_vel_x: tuple = (-0.6, 1.0)
    lin_vel_y: tuple = (-0.5, 0.5)
    ang_vel_z: tuple = (-1.0, 1.0)
    heading: tuple = (-math.pi, math.pi)


@configclass
class CommandsCfg:
    resampling_time_range: tuple = (10.0, 10.0)
    rel_standing_envs: float = 0.2
    rel_heading_envs: float = 1.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    debug_vis: bool = True
    ranges: CommandRangesCfg = CommandRangesCfg()


@configclass
class NoiseScalesCfg:
    ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5
    height_scan: float = 0.1


@configclass
class NoiseCfg:
    add_noise: bool = True
    noise_level: float = 1.0
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class ResetRobotJointsCfg:
    params: dict = {"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)}


@configclass
class ResetRobotBaseCfg:
    params: dict = {
        "pose_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "yaw": (-3.14, 3.14),
        },
        "velocity_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        },
    }


@configclass
class RandomizeRobotFrictionCfg:
    enable: bool = True
    params: dict = {
        "static_friction_range": [0.6, 1.0],
        "dynamic_friction_range": [0.4, 0.8],
        "restitution_range": [0.0, 0.005],
        "num_buckets": 64,
    }


@configclass
class AddRigidBodyMassCfg:
    enable: bool = True
    params: dict = {
        "body_names": MISSING,
        "mass_distribution_params": (-5.0, 5.0),
        "operation": "add",
    }


@configclass
class PushRobotCfg:
    enable: bool = True
    push_interval_s: float = 15.0
    params: dict = {"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}}


@configclass
class ActionDelayCfg:
    enable: bool = False
    params: dict = {"max_delay": 5, "min_delay": 0}


@configclass
class DomainRandCfg:
    reset_robot_joints: ResetRobotJointsCfg = ResetRobotJointsCfg()
    reset_robot_base: ResetRobotBaseCfg = ResetRobotBaseCfg()
    randomize_robot_friction: RandomizeRobotFrictionCfg = RandomizeRobotFrictionCfg()
    add_rigid_body_mass: AddRigidBodyMassCfg = AddRigidBodyMassCfg()
    push_robot: PushRobotCfg = PushRobotCfg()
    action_delay: ActionDelayCfg = ActionDelayCfg()


@configclass
class PhysxCfg:
    gpu_max_rigid_patch_count: int = 10 * 2**15


@configclass
class SimCfg:
    dt: float = 0.005
    decimation: int = 4
    physx: PhysxCfg = PhysxCfg()
