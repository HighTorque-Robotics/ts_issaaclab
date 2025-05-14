from dataclasses import MISSING
import math
from isaaclab.utils import configclass
from .base_config import BaseSceneCfg, HeightScannerCfg, RobotCfg, RewardCfg, gaitCfg, \
    NormalizationCfg, ObsScalesCfg, CommandsCfg, CommandRangesCfg, NoiseCfg, NoiseScalesCfg, \
    DomainRandCfg, ResetRobotJointsCfg, ResetRobotBaseCfg, RandomizeRobotFrictionCfg, AddRigidBodyMassCfg, \
    PushRobotCfg, ActionDelayCfg, SimCfg, PhysxCfg

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlRndCfg, RslRlSymmetryCfg  # noqa:F401


@configclass
class BaseEnvCfg:
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=MISSING,
        terrain_type=MISSING,
        terrain_generator=None,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name=MISSING,
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(-0.3, 0.3)
        )
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=1,
        action_scale=0.25,
        terminate_contacts_body_names=MISSING,
        feet_body_names=MISSING,
    )
    reward = RewardCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=2.0,
            ang_vel=0.25,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=0.05,
            actions=1.0,
            height_scan=5.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.8, 0.8),
            heading=(-math.pi, math.pi)
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_level=1.0,
        noise_scales=NoiseScalesCfg(
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        )
    )
    gait: gaitCfg = gaitCfg()
    domain_rand: DomainRandCfg = DomainRandCfg(
        reset_robot_joints=ResetRobotJointsCfg(
            params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)}
        ),
        reset_robot_base=ResetRobotBaseCfg(
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    # "roll": (-0.8, 0.8),
                    # "pitch": (-0.8, 0.8),
                    "yaw": (-0.0, 0.0),
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
        ),
        randomize_robot_friction=RandomizeRobotFrictionCfg(
            enable=True,
            params={
                "static_friction_range": [0.6, 1.0],
                "dynamic_friction_range": [0.4, 0.8],
                "restitution_range": [0.0, 0.005],
                "num_buckets": 64,
            }
        ),
        add_rigid_body_mass=AddRigidBodyMassCfg(
            enable=True,
            params={
                "body_names": MISSING,
                "mass_distribution_params": (-5.0, 5.0),
                "operation": "add",
            }
        ),
        push_robot=PushRobotCfg(
            enable=True,
            push_interval_s=15.0,
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.3, 0.3)}}

        ),
        action_delay=ActionDelayCfg(
            enable=False,
            params={"max_delay": 5, "min_delay": 0}
        ),
    )
    sim: SimCfg = SimCfg(
        dt=0.005,
        decimation=4,
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=10 * 2**15
        )
    )

    def __post_init__(self):
        pass


@configclass
class BaseAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 10000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 50
    experiment_name = ""
    run_name = ""
    logger = "wandb"
    neptune_project = "leggedlab"
    wandb_project = "leggedlab"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    def __post_init__(self):
        pass
