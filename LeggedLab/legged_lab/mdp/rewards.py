from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils
import isaacsim.core.utils.torch as torch_utils
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv


def track_lin_vel_x_yaw_frame_exp(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])

    coeffi = (
        env.gait_generator.first_foot_swing_height_target + 
        env.gait_generator.second_foot_swing_height_target
    ) / env.gait_generator.gaits[:, 3] - 0.5
    command = env.command_generator.command[:, 0] * (1 + 0.4 * coeffi)
    lin_vel_error = torch.square(command - vel_yaw[:, 0])
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_y_yaw_frame_exp(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    # coeffi = (
    #     -env.gait_generator.first_foot_swing_height_target / env.gait_generator.gaits[:, 3] + 0.5 \
    #     + env.gait_generator.second_foot_swing_height_target / env.gait_generator.gaits[:, 3] - 0.5
    # )
    coeffi = 0
    command = env.command_generator.command[:, 1] * (1 + 0.4 * coeffi)
    lin_vel_error = torch.square(command - vel_yaw[:, 1])
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    coeffi = (
        env.gait_generator.first_foot_swing_height_target + 
        env.gait_generator.second_foot_swing_height_target
    ) / env.gait_generator.gaits[:, 3] - 0.5
    command = env.command_generator.command[:, 2] * (1 + 0.4 * coeffi)
    ang_vel_error = torch.square(command - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]), dim=1)


def action_smooth_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - \
            2 * env.action_buffer._circular_buffer.buffer[:, -2, :] + \
            env.action_buffer._circular_buffer.buffer[:, -3, :]
        ), dim=1
    )


def undesired_contacts(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def joint_symmetry(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids[0]] - asset.data.default_joint_pos[:, asset_cfg.joint_ids[0]] + asset.data.joint_pos[:, asset_cfg.joint_ids[1]] - asset.data.default_joint_pos[:, asset_cfg.joint_ids[1]])
    return torch.exp(-joint_error / std**2)


def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    # reward *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    return reward


def feet_slide(env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(
        env: BaseEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    asset: Articulation = env.scene[asset_cfg.name]
    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]

    for i in range(2):
        contact_force[:, i] = math_utils.quat_rotate_inverse(math_utils.yaw_quat(feet_quat[:, i]), contact_force[:, i])
    
    return torch.any(torch.norm(contact_force[:, :, :2], dim=2) > 5 * torch.abs(contact_force[:, :, 2]), dim=1)


def feet_distance(env: BaseEnv, 
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
                  min_feet_distance: float = 0.115
                  ) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    # Penalize base height away from target
    asset: Articulation = env.scene[asset_cfg.name]

    feet_pos_f_left = math_utils.quat_rotate_inverse(math_utils.yaw_quat(asset.data.root_quat_w), env.feet_pos_f[:, 0])
    feet_pos_f_right = math_utils.quat_rotate_inverse(math_utils.yaw_quat(asset.data.root_quat_w), env.feet_pos_f[:, 1])
    feet_pos_h_left = math_utils.quat_rotate_inverse(math_utils.yaw_quat(asset.data.root_quat_w), env.feet_pos_h[:, 0])
    feet_pos_h_right = math_utils.quat_rotate_inverse(math_utils.yaw_quat(asset.data.root_quat_w), env.feet_pos_h[:, 1])

    # feet_distance_f = torch.norm(feet_pos_f[:, 0, :2] - feet_pos_f[:, 1, :2], dim=-1)
    # feet_distance_h = torch.norm(feet_pos_h[:, 0, :2] - feet_pos_h[:, 1, :2], dim=-1)

    # feet_distance = torch.min(feet_distance_f, feet_distance_h)
    # if env.command_generator.cfg.ranges.ang_vel_z[1] == 0:
    #     return torch.clip(min_feet_distance - feet_distance, 0, 1)
    # else:
    #     return torch.clip(min_feet_distance - feet_distance, 0, 1) * (
    #         (env.command_generator.cfg.ranges.ang_vel_z[1] - env.command_generator.command[:, 2]) / 
    #         env.command_generator.cfg.ranges.ang_vel_z[1]
    #     )

    feet_distance_f = torch.abs(feet_pos_f_left[:, 1] - feet_pos_f_right[:, 1])
    feet_distance_h = torch.abs(feet_pos_h_left[:, 1] - feet_pos_h_right[:, 1])
    feet_distance = torch.min(feet_distance_f, feet_distance_h)
    return torch.clip(min_feet_distance - feet_distance, 0, 1)


def feet_regulation(
        env: BaseEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
        base_height_target: float = 0.3454, 
        ) -> torch.Tensor:

    feet_height_target = base_height_target * 0.001
    asset: Articulation = env.scene[asset_cfg.name]
    feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    feet_height = (env.feet_pos_f[:, :, 2] + env.feet_pos_h[:, :, 2]) / 2

    return torch.sum(
        torch.exp(-feet_height / feet_height_target)
        * torch.square(torch.norm(feet_vel, dim=-1)),
        dim=1,
    )


def feet_landing_vel(env: BaseEnv, 
                    about_landing_threshold: float,
                    sensor_cfg: SceneEntityCfg, 
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    z_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0

    asset: Articulation = env.scene[asset_cfg.name]
    feet_height = (env.feet_pos_f[:, :, 2] + env.feet_pos_h[:, :, 2]) / 2
    about_to_land = (feet_height < about_landing_threshold) & (~contacts) & (z_vels < 0.0)
    landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
    return torch.sum(torch.square(landing_z_vels), dim=1)


def feet_takeoff_vel(env: BaseEnv, 
                    about_takeoff_threshold: float,
                    sensor_cfg: SceneEntityCfg, 
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    z_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0

    asset: Articulation = env.scene[asset_cfg.name]
    feet_height = (env.feet_pos_f[:, :, 2] + env.feet_pos_h[:, :, 2]) / 2
    about_to_land = (feet_height < about_takeoff_threshold) & (~contacts) & (z_vels > 0.0)
    takeoff_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
    return torch.sum(torch.square(takeoff_z_vels), dim=1)


def tracking_contacts_shaped_force(
        env: BaseEnv, 
        gait_force_sigma: float,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim = -1)
    desired_swing_mask = env.gait_generator.desired_contact_states > 0.0

    reward = 0
    for i in range(2):
        reward += desired_swing_mask[:, i] * (1 - (torch.exp(-foot_forces[:, i] ** 2 / gait_force_sigma)))
    # no reward for zero command
    # reward *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    return reward


def tracking_contacts_shaped_vel(env: BaseEnv, 
                                 gait_vel_sigma: float,
                                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                 ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_vel = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim = -1)
    desired_contact_mask = env.gait_generator.desired_contact_states < 0.0
    reward = 0
    for i in range(2):
        reward += desired_contact_mask[:, i] * (1 - (torch.exp(-feet_vel[:, i] ** 2 / gait_vel_sigma)))
    # no reward for zero command
    # reward *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    return reward


def keep_balance(env: BaseEnv) -> torch.Tensor:
    return torch.ones(
        env.num_envs, dtype=torch.float, device=env.device, requires_grad=False
    )


def tracking_feet_swing_height(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    # Penalize base height away from target
    asset: Articulation = env.scene[asset_cfg.name]

    feet_height_f = env.feet_pos_f[:, :, 2]
    feet_height_h = env.feet_pos_h[:, :, 2]

    feet_height_target_left = env.gait_generator.first_foot_swing_height_target
    feet_height_target_right = env.gait_generator.second_foot_swing_height_target

    feet_height_error_left_f = torch.abs((feet_height_target_left - feet_height_f[:, 0]))
    feet_height_error_left_h = torch.abs((feet_height_target_left - feet_height_h[:, 0]))
    feet_height_error_right_f = torch.abs((feet_height_target_right - feet_height_f[:, 1]))
    feet_height_error_right_h = torch.abs((feet_height_target_right - feet_height_h[:, 1]))

    # feet_height_error_left = torch.square(torch.cat([feet_height_error_left_f.unsqueeze(-1), feet_height_error_left_h.unsqueeze(-1)], dim=-1)).mean(-1)
    # feet_height_error_right = torch.square(torch.cat([feet_height_error_right_f.unsqueeze(-1), feet_height_error_right_h.unsqueeze(-1)], dim=-1)).mean(-1)
    feet_height_error_left = torch.square(feet_height_error_left_h)
    feet_height_error_right = torch.square(feet_height_error_right_h)

    left_feet_swing_mask = env.gait_generator.desired_contact_states[:, 0] > 0
    right_feet_swing_mask = env.gait_generator.desired_contact_states[:, 1] > 0
    reward = (
        torch.exp(-feet_height_error_left / std**2)*left_feet_swing_mask + 
        torch.exp(-feet_height_error_right / std**2)*right_feet_swing_mask
    ) / 2
    # no reward for zero command
    # reward *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    return reward


def tracking_feet_orientation(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_ori = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    feet_ori_target = torch.zeros_like(feet_ori)
    feet_ori_target[:, :, 3] = 1.0
    feet_ori_error = torch.square(feet_ori - feet_ori_target).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-feet_ori_error / std**2)
    # no reward for zero command
    # reward *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    return reward


def stand_still_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1) * ((torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) <= 0.1)


def default_ankle_roll_pos(
        env: BaseEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
        sigma: float = 0.1
    ) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    roll_1, _, _ = torch_utils.get_euler_xyz(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :])
    roll_2, _, _ = torch_utils.get_euler_xyz(asset.data.body_quat_w[:, asset_cfg.body_ids[1], :])

    desired_contact_mask = env.gait_generator.desired_contact_states < 0.0

    return torch.exp(-(torch.abs(roll_1)) / sigma) * desired_contact_mask[:, 0] + torch.exp(-(torch.abs(roll_2)) / sigma) * desired_contact_mask[:, 1]
