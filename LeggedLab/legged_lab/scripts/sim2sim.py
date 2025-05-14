# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R

from legged_lab import LEGGED_LAB_ROOT_DIR

import torch

from pynput import keyboard
import onnxruntime as ort


class Reset:
    reset = False


def on_press(key):
    try:
        if key.char == 'w':
            cmd.vx += 0.1
        if key.char == 's':
            cmd.vx -= 0.1
        if key.char == 'a':
            cmd.vyaw += 0.1
        if key.char == 'd':
            cmd.vyaw -= 0.1
        cmd.vx = np.clip(cmd.vx, -0.5, 0.5)
        cmd.vy = np.clip(cmd.vy, -0.5, 0.5)
        cmd.vyaw = np.clip(cmd.vyaw, -0.8, 0.8)

        if key.char == 'j':
            gait.frequencies += 0.2
        if key.char == 'l':
            gait.frequencies -= 0.2
        if key.char == 'i':
            gait.swing_height += 0.1
        if key.char == 'k':
            gait.swing_height -= 0.1
        gait.frequencies = np.clip(gait.frequencies, 1.0, 2.0)
        gait.swing_height = np.clip(gait.swing_height, 0.05, 0.1)

        if key.char == '0':
            cmd.vx = 0.0
            cmd.vy = 0.0
            cmd.vyaw = 0.0
            gait.frequencies = 1.25
            gait.swing_height = 0.05
    except AttributeError:
      pass

def on_release(key):
    if key == keyboard.Key.esc:
        return False
    
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


class env:
    obs = 42 + 4 + 4 + 3
    num_single_obs = 42 + 4 + 4 + 3
    obs_his = (42 + 4 + 4 + 3) * 10


class gait:
    frequencies = 1.25 # [1.0, 2.5]
    offsets = 0.5 # [0, 1]
    durations = 0.5 # [0.5, 0.5]
    swing_height = 0.05 #  [0.05, 0.1]


class cmd:
    vx = 0.0
    vy = 0.0
    vyaw = 0.0


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def low_pass_action_filter(actions, last_actions):
    alpha = 0.2
    actons_filtered = last_actions * alpha + actions * (1 - alpha)
    return actons_filtered


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    # print("p:", (target_q - q) * kp )
    # print("d", (target_dq - dq) * kd)
    return (target_q - q) * kp + (target_dq - dq) * kd 


def plot(traj):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    for i in range(traj.shape[1]):
        plt.plot(traj[:, i], '-', label=f'{i}')
    
    plt.title('Six-dimensional data curves')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.legend()

    plt.grid(True)
    plt.show()


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros(12, dtype=np.double)
    last_action = np.zeros(12, dtype=np.double)
    action = np.zeros(12, dtype=np.double)
    hist_obs = np.zeros([1, env.obs_his])
    gait_indices = np.zeros(1, dtype=np.double)
    swing_height_indices = np.zeros(1, dtype=np.double)

    joint_index_mujoco = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
    joint_index_lab = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]

    dq_traj = np.zeros([2500, 8])
    counter = 0
    count_lowlevel = 0
    while True:
        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-12 :]
        dq = dq[-12 :]

        # print(v)
        print(dq.max())

        q_lab = np.zeros_like(q)
        dq_lab = np.zeros_like(dq)
        for i in range(12):
            q_lab[joint_index_lab[i]] = q[joint_index_mujoco[i]]
            dq_lab[joint_index_lab[i]] = dq[joint_index_mujoco[i]]    

        # 1000hz -> 50hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = np.zeros([1, env.obs])
            _q = quat
            _v = np.array([0.0, 0.0, -1.0])
            projected_gravity = quat_rotate_inverse(_q, _v)

            obs[0, 0:3] = omega * cfg.normalization.ang_vel
            obs[0, 3:6] = projected_gravity
            obs[0, 6:18] = q_lab * cfg.normalization.dof_pos
            obs[0, 18:30] = dq_lab * cfg.normalization.dof_vel
            obs[0, 30:42] = last_action

            obs[0, 42] = gait.frequencies
            obs[0, 43] = gait.offsets
            obs[0, 44] = gait.durations
            obs[0, 45] = gait.swing_height

            gait_indices = np.remainder(gait_indices + cfg.sim_config.dt * cfg.sim_config.decimation * gait.frequencies, 1.0)
            swing_height_indices = np.remainder(swing_height_indices + cfg.sim_config.dt * cfg.sim_config.decimation * gait.frequencies * 2, 1.0)

            desired_contact_states = np.remainder(
                np.hstack(
                    [
                        (gait_indices + gait.offsets + 1),
                        gait_indices
                    ]
                ),
                1.0,
            ) - gait.durations
            # stance_idxs = desired_contact_states < 0.0
            # swing_idxs = desired_contact_states > 0.0

            first_foot_swing_height_target = gait.swing_height * (np.sin(2 * np.pi * swing_height_indices - 1 / 2 * np.pi) / 2 + 0.5) * (desired_contact_states[0] > 0)
            second_foot_swing_height_target = gait.swing_height * (np.sin(2 * np.pi * swing_height_indices - 1 / 2 * np.pi) / 2 + 0.5) * (desired_contact_states[1] > 0)

            obs[0, 46] = first_foot_swing_height_target
            obs[0, 47] = second_foot_swing_height_target
            obs[0, 48] = gait_indices
            obs[0, 49] = swing_height_indices

            obs[0, 50] = cmd.vx
            obs[0, 51] = cmd.vy
            obs[0, 52] = cmd.vyaw

            hist_obs = np.concatenate((hist_obs[:, env.num_single_obs:], obs[:, :env.num_single_obs]), axis=-1).astype(np.float32)
            input = hist_obs.astype(np.float32)
            # input = np.concatenate((obs, hist_obs), axis=-1).astype(np.float32)
            # input = obs.astype(np.float32)
            action = policy.run(None, {'input': input})[0]
            if cfg.robot_config.use_filter:
                action = low_pass_action_filter(action, last_action)
            last_action = action

            action = np.clip(
                action,
                -cfg.normalization.clip_actions,
                cfg.normalization.clip_actions,
            )
            target_q = action * cfg.control.action_scale
            # target_q[:, [10, 11]] = 0

            target_q_mujoco = np.zeros_like(target_q)
            for i in range(12):
                target_q_mujoco[0][joint_index_mujoco[i]] = target_q[0][joint_index_lab[i]]

        target_dq = np.zeros((12), dtype=np.double)

        # Generate PD control
        tau = pd_control(
            target_q_mujoco, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds
        )  # Calc torques
        tau = np.clip(
            tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit
        )[0]  # Clamp torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        import time
        time.sleep(0.001)
        if count_lowlevel % cfg.sim_config.decimation == 0:
            viewer.cam.lookat[:] = data.qpos.astype(np.float32)[0:3]
            viewer.render()
            # print(tau.max())
        count_lowlevel += 1

    viewer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument(
        "--logdir",
        type=str,
        required=False,
        default=f"{LEGGED_LAB_ROOT_DIR}/legged_lab/logs/pai_flat/1/exported",
        help="Run to load from.",
    )
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    args = parser.parse_args()

    class Sim2simCfg:
        class sim_config:
            mujoco_model_path = f"{LEGGED_LAB_ROOT_DIR}/legged_lab/robot/pai/mjcf/pi_12dof_release_v1.xml"
            sim_duration = 60.0
            dt = 0.001
            decimation = 20

        class normalization:
            lin_vel=2.0
            ang_vel=0.25
            projected_gravity=1.0
            commands=1.0
            dof_pos=1.0
            dof_vel=0.05
            actions=1.0
            height_scan=5.0

            clip_observations=100.0
            clip_actions=100.0

        class control:
            action_scale = 0.25

        class robot_config:
            kps = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], dtype=np.double)
            kds = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.double)
            tau_limit = 10.0 * np.ones(12, dtype=np.double)
            use_filter = False

    policy = ort.InferenceSession(args.logdir + f"/policy.onnx")
    run_mujoco(policy, Sim2simCfg())