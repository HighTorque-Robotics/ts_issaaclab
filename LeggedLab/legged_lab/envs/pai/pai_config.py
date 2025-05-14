from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseEnvCfg, BaseAgentCfg, BaseSceneCfg, RobotCfg, DomainRandCfg,
    RewardCfg, HeightScannerCfg, AddRigidBodyMassCfg, PhysxCfg, SimCfg
)
from legged_lab.assets.hightorque import Pai_CFG
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG, PlANE_TERRAINS_CFG
from isaaclab.managers import RewardTermCfg as RewTerm
import legged_lab.mdp as mdp
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


import torch


@configclass
class PaiRewardCfg(RewardCfg):
    track_lin_vel_x_exp = RewTerm(func=mdp.track_lin_vel_x_yaw_frame_exp, weight=1.0, params={"std": 0.25})
    track_lin_vel_y_exp = RewTerm(func=mdp.track_lin_vel_y_yaw_frame_exp, weight=2.0, params={"std": 0.45})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # action_smooth_l2 = RewTerm(func=mdp.action_smooth_l2, weight=-0.001)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-30.0)
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_stumble = RewTerm(
        func=mdp.feet_stumble, 
        weight=-1.0, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*")
        }
    )
    
    feet_regulation = RewTerm(
        func=mdp.feet_regulation, weight=-0.05, 
        params={
            "base_height_target": 0.3454, 
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*")
        }
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance, 
        weight=-100.0, 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), 
            "min_feet_distance": 0.115
        }
    )

    feet_landing_vel = RewTerm(
        func=mdp.feet_landing_vel, 
        weight=-1.0, 
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"]), 
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"), 
                "about_landing_threshold": 0.03
        }
    )
    feet_takeoff_vel = RewTerm(
        func=mdp.feet_takeoff_vel, 
        weight=-1.0, 
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"]), 
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"), 
                "about_takeoff_threshold": 0.03
        }
    )
    tracking_contacts_shaped_force = RewTerm(
        func=mdp.tracking_contacts_shaped_force, 
        weight=-2.0, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), 
            "gait_force_sigma": 25.0
        }
    )
    tracking_contacts_shaped_vel = RewTerm(
        func=mdp.tracking_contacts_shaped_vel, 
        weight=-2.0, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "gait_vel_sigma": 0.25
        }
    )
    # keep_balance = RewTerm(func=mdp.keep_balance, weight=1.0)

    tracking_feet_swing_height= RewTerm(
        func=mdp.tracking_feet_swing_height, 
        weight=5.0, 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "std": 0.03,
        }
    )
    # tracking_feet_orientation = RewTerm(
    #     func=mdp.tracking_feet_orientation, 
    #     weight=5.0, 
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
    #         "std": 0.5,
    #     }
    # )
    # stand_still_l1 = RewTerm(func=mdp.stand_still_l1, weight=-1.0)

    # default_ankle_roll_pos = RewTerm(
    #     func=mdp.default_ankle_roll_pos, 
    #     weight=2.0, 
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
    #         "sigma": 0.1,
    #     }
    # )



@configclass
class PaiFlatEnvCfg(BaseEnvCfg):

    reward = PaiRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.prim_body_name = "base_link"
        self.scene.robot = Pai_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = PlANE_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = ["base_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.add_rigid_body_mass.params["body_names"] = ["base_link"]
        self.domain_rand.add_rigid_body_mass.params["mass_distribution_params"] = (-1.0, 1.0)


@configclass
class SymmetryCfg:
    use_data_augmentation: bool = False  # this adds symmetric trajectories to the batch
    use_mirror_loss: bool = False  # this adds symmetry loss term to the loss function
    data_augmentation_func: str = "legged_lab.envs.pai.pai_config:data_augmentation_func_pai"
    mirror_loss_coeff:float = 1e-3


@configclass
class PaiFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "pai_flat"
    wandb_project: str = "pai_flat"
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
        symmetry_cfg = SymmetryCfg(),
        rnd_cfg=None,  # RslRlRndCfg()
    )


@configclass
class PaiRoughEnvCfg(PaiFlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        # self.reward.feet_air_time.weight = 0.25
        # self.reward.track_lin_vel_xy_exp_relax = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.7})
        # self.reward.track_lin_vel_xy_exp.weight = 0.5
        # self.reward.track_ang_vel_z_exp.weight = 1.5
        # self.reward.lin_vel_z_l2.weight = -0.25


@configclass
class PaiRoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "pai_rough"
    wandb_project: str = "pai_rough"

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"


def data_augmentation_func_pai(obs, actions, env, obs_type):
    if obs is None:
        output_obs = None
    else:
        if obs_type == "policy":
            flipped_obs = flip_pi_actor_obs(obs)
            output_obs = torch.cat((obs, flipped_obs), dim = 0)
        elif obs_type == "critic":
            flipped_obs = flip_pi_critic_obs(obs)
            output_obs = torch.cat((obs, flipped_obs), dim = 0)

    if actions is None:
        output_actions = None
    else:
        flipped_actions = flip_pi_actions(actions)
        output_actions = torch.cat((actions, flipped_actions), dim = 0)
    return output_obs, output_actions


def flip_pi_actor_obs(obs):
    if obs is None:
        return obs
    
    one_step_obs = obs[..., :50]
    latent = obs[..., 50:-3]
    command = obs[..., -3:]

    flipped_one_step_obs = torch.zeros_like(one_step_obs)
    flipped_command = torch.zeros_like(command)

    flipped_one_step_obs[..., :3] = one_step_obs[..., :3] # ang_vel
    flipped_one_step_obs[..., 0] = -flipped_one_step_obs[..., 0] # ang_vel_x
    flipped_one_step_obs[..., 2] = -flipped_one_step_obs[..., 2] # ang_vel_z
    flipped_one_step_obs[..., 3:6] = one_step_obs[..., 3:6] * torch.tensor(
        [1.0, -1.0, 1.0], 
        dtype=one_step_obs[..., 3:6].dtype, 
        device=one_step_obs[..., 3:6].device
    ) # projected_gravity
    flipped_one_step_obs[..., 6:18] = flip_dof(one_step_obs[..., 6:18]) # dof_pos
    flipped_one_step_obs[..., 18:30] = flip_dof(one_step_obs[..., 18:30]) # dof_vel
    flipped_one_step_obs[..., 30:42] = flip_dof(one_step_obs[..., 30:42]) # last_actions
    flipped_one_step_obs[..., 42:46] = one_step_obs[..., 42:46] # gait
    flipped_one_step_obs[..., 46] = one_step_obs[..., 47] # first_foot_swing_height_target
    flipped_one_step_obs[..., 47] = one_step_obs[..., 46] # second_foot_swing_height_target
    flipped_one_step_obs[..., 48] = one_step_obs[..., 48] # gait
    flipped_one_step_obs[..., 49] = one_step_obs[..., 49] # gait

    flipped_command[..., 0] = command[..., 0] # lin_vel_x
    flipped_command[..., 1] = -command[..., 1] # lin_vel_y
    flipped_command[..., 2] = -command[..., 2] # heading

    flipped_obs = torch.cat((flipped_one_step_obs, latent, flipped_command), dim = -1)

    return flipped_obs


def flip_pi_critic_obs(obs):
    if obs is None:
        return obs
    
    one_step_obs = obs[..., :55]
    command = obs[..., 55:58]
    latent = obs[..., 58:]

    flipped_one_step_obs = torch.zeros_like(one_step_obs)
    flipped_command = torch.zeros_like(command)
    
    flipped_one_step_obs[..., :3] = one_step_obs[..., :3] # lin_vel
    flipped_one_step_obs[..., 1] = -flipped_one_step_obs[..., 1] # lin_vel_y
    flipped_one_step_obs[..., 3] = one_step_obs[..., 4] # feet_contact
    flipped_one_step_obs[..., 4] = one_step_obs[..., 3] # feet_contact
    flipped_one_step_obs[..., 5:8] = one_step_obs[..., 5:8] # ang_vel
    flipped_one_step_obs[..., 5] = -flipped_one_step_obs[..., 5] # ang_vel_x
    flipped_one_step_obs[..., 7] = -flipped_one_step_obs[..., 7] # ang_vel_z
    flipped_one_step_obs[..., 8:11] = one_step_obs[..., 8:11] * torch.tensor(
        [1.0, -1.0, 1.0], 
        dtype=one_step_obs[..., 8:11].dtype, 
        device=one_step_obs[..., 8:11].device
    ) # projected_gravity
    flipped_one_step_obs[..., 11:23] = flip_dof(one_step_obs[..., 11:23]) # dof_pos
    flipped_one_step_obs[..., 23:35] = flip_dof(one_step_obs[..., 23:35]) # dof_vel
    flipped_one_step_obs[..., 35:47] = flip_dof(one_step_obs[..., 35:47]) # last_actions
    flipped_one_step_obs[..., 47:51] = one_step_obs[..., 47:51] # gait
    flipped_one_step_obs[..., 51] = one_step_obs[..., 52] # first_foot_swing_height_target
    flipped_one_step_obs[..., 52] = one_step_obs[..., 51] # second_foot_swing_height_target
    flipped_one_step_obs[..., 53] = one_step_obs[..., 53] # gait
    flipped_one_step_obs[..., 54] = one_step_obs[..., 54] # gait

    flipped_command[..., 0] = command[..., 0] # lin_vel_x
    flipped_command[..., 1] = -command[..., 1] # lin_vel_y
    flipped_command[..., 2] = -command[..., 2] # heading

    flipped_obs = torch.cat((flipped_one_step_obs, flipped_command, latent), dim = -1)

    return flipped_obs

def flip_pi_actions(actions):
    if actions is None:
        return None
    
    fliped_actions = flip_dof(actions)
    
    return fliped_actions


def flip_dof(dof):
    flipped_dof = torch.zeros_like(dof)
    flipped_dof[..., 0] = dof[..., 1] # hip pitch
    flipped_dof[..., 1] = dof[..., 0] # hip pitch
    flipped_dof[..., 2] = -dof[..., 3] # hip roll
    flipped_dof[..., 3] = -dof[..., 2] # hip roll
    flipped_dof[..., 4] = -dof[..., 5] # thigh
    flipped_dof[..., 5] = -dof[..., 4] # thigh
    flipped_dof[..., 6] = dof[..., 7] # calf
    flipped_dof[..., 7] = dof[..., 6] # calf
    flipped_dof[..., 8] = dof[..., 9] # ankle pitch
    flipped_dof[..., 9] = dof[..., 8] # ankle pitch
    flipped_dof[..., 10] = -dof[..., 11] # ankle roll
    flipped_dof[..., 11] = -dof[..., 10] # ankle roll
    return flipped_dof