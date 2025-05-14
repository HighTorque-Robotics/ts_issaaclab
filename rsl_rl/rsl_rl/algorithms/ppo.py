# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.modules import ActorCritic, LatentEncoder, MLPEncoder
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable


class PPO:

    policy: ActorCritic
    """The actor critic module."""

    actor_critic: ActorCritic
    teacher_encoder: LatentEncoder
    student_encoder: LatentEncoder
    def __init__(
            self,
            critic_takes_latent,
            actor_critic,
            teacher_encoder,
            student_encoder,
            num_obs,
            latent_dim,
            command_dim,
            num_learning_epochs=1,
            num_mini_batches=1,
            clip_param=0.2,
            gamma=0.998,
            lam=0.95,
            value_loss_coef=1.0,
            entropy_coef=0.0,
            learning_rate=1e-3,
            student_encoder_learning_rate=1e-3,
            max_grad_norm=1.0,
            use_clipped_value_loss=True,
            schedule="fixed",
            desired_kl=0.01,
            device='cpu',
            normalize_advantage_per_mini_batch=False,
            # RND parameters
            rnd_cfg: dict | None = None,
            # Symmetry parameters
            symmetry_cfg: dict | None = None,
            # Distributed training parameters
            multi_gpu_cfg: dict | None = None,
        ):
        
        self.device = device
        self.num_obs = num_obs
        self.latent_dim = latent_dim
        self.command_dim = command_dim
        self.critic_takes_latent = critic_takes_latent

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.teacher_encoder = teacher_encoder
        self.teacher_encoder.to(self.device)
        self.student_encoder = student_encoder
        self.student_encoder.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(
            [
                {"params": self.actor_critic.parameters()},
                {"params": self.teacher_encoder.parameters()},
            ],
            lr=learning_rate,
        )
        self.student_encoder_optimizer = optim.Adam(self.student_encoder.parameters(), lr=student_encoder_learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, history_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, 
            num_transitions_per_env, 
            actor_obs_shape, 
            history_obs_shape,
            critic_obs_shape, 
            action_shape, 
            self.device
        )

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, history_obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        obs, command = obs[:, 0:-self.command_dim], obs[:, -self.command_dim:]
        latent = self.teacher_encoder(critic_obs[:, :-self.command_dim]).detach()
        actor_obs = torch.cat((obs, latent, command), dim=-1)
        if self.critic_takes_latent:
            critic_obs = torch.cat((critic_obs, latent), dim=-1)
        else:
            critic_obs = torch.cat((critic_obs), dim=-1)
        self.transition.actions = self.actor_critic.act(actor_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.actor_observations = actor_obs
        self.transition.history_observations = history_obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_history_obs, last_critic_obs):
        if self.critic_takes_latent:
            last_latent = self.teacher_encoder(last_critic_obs[:, :-self.command_dim]).detach()
            last_critic_obs = torch.cat((last_critic_obs, last_latent), dim=-1)
        else:
            last_critic_obs = torch.cat((last_critic_obs), dim=-1)
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_student_encoder_loss = 0
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for actor_obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
                
                # number of augmentations per sample
                # we start with 1 and increase it if we use symmetry augmentation
                num_aug = 1
                # original batch size
                original_batch_size = actor_obs_batch.shape[0]

                # Perform symmetric augmentation
                if self.symmetry and self.symmetry["use_data_augmentation"]:
                    # augmentation using symmetry
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    # returned shape: [batch_size * num_aug, ...]
                    actor_obs_batch, actions_batch = data_augmentation_func(
                        obs=actor_obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                    )
                    critic_obs_batch, _ = data_augmentation_func(
                        obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(actor_obs_batch.shape[0] / original_batch_size)
                    # repeat the rest of the batch
                    # -- actor
                    old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                    # -- critic
                    target_values_batch = target_values_batch.repeat(num_aug, 1)
                    advantages_batch = advantages_batch.repeat(num_aug, 1)
                    returns_batch = returns_batch.repeat(num_aug, 1)

                obs_batch = actor_obs_batch[:, :self.num_obs]
                if self.critic_takes_latent:
                    latent_batch = self.teacher_encoder(critic_obs_batch[:, :-(self.latent_dim+self.command_dim)])
                    latent_batch_detached = latent_batch.detach()
                    critic_obs_no_latent = critic_obs_batch[:, :-self.latent_dim]
                    critic_obs_batch = torch.cat((critic_obs_no_latent, latent_batch_detached), dim=-1)
                else:
                    latent_batch = self.teacher_encoder(critic_obs_batch[:, :-(self.command_dim)])
                command_batch = actor_obs_batch[:, -self.command_dim:]
                actor_obs_batch = torch.cat((obs_batch, latent_batch, command_batch), dim=-1)

                self.actor_critic.act(actor_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean[:original_batch_size]
                sigma_batch = self.actor_critic.action_std[:original_batch_size]
                entropy_batch = self.actor_critic.entropy[:original_batch_size]

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Symmetry loss
                if self.symmetry:
                    # obtain the symmetric actions
                    # if we did augmentation before then we don't need to augment again
                    if not self.symmetry["use_data_augmentation"]:
                        data_augmentation_func = self.symmetry["data_augmentation_func"]
                        actor_obs_batch, _ = data_augmentation_func(
                            obs=actor_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                        )
                        # compute number of augmentations per sample
                        num_aug = int(actor_obs_batch.shape[0] / original_batch_size)

                    # actions predicted by the actor for symmetrically-augmented observations
                    mean_actions_batch = self.actor_critic.act_inference(actor_obs_batch.detach().clone())

                    # compute the symmetrically augmented actions
                    # note: we are assuming the first augmentation is the original one.
                    #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                    #   However, the symmetry loss is computed using the mean of the distribution.
                    action_mean_orig = mean_actions_batch[:original_batch_size]
                    _, actions_mean_symm_batch = data_augmentation_func(
                        obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                    )

                    # compute the loss (we skip the first augmentation as it is the original one)
                    mse_loss = torch.nn.MSELoss()
                    symmetry_loss = mse_loss(
                        mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                    )
                    # add the loss to the total loss
                    if self.symmetry["use_mirror_loss"]:
                        loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                    else:
                        symmetry_loss = symmetry_loss.detach()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy_batch.mean().item()

                # -- Symmetry loss
                if mean_symmetry_loss is not None:
                    mean_symmetry_loss += symmetry_loss.item()

        generator = self.storage.student_encoder_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            history_obs_batch,
            critic_obs_batch,
        ) in generator:
            if self.critic_takes_latent:
                latent_teacher = self.teacher_encoder(critic_obs_batch[:, :-(self.latent_dim+self.command_dim)]).detach()
            else:
                latent_teacher = self.teacher_encoder(critic_obs_batch[:, :-(self.command_dim)]).detach()
            latent_student = self.student_encoder(history_obs_batch)
            student_encoder_loss = F.mse_loss(latent_teacher, latent_student)

            self.student_encoder_optimizer.zero_grad()
            student_encoder_loss.backward()
            nn.utils.clip_grad_norm_(self.student_encoder.parameters(), self.max_grad_norm)
            self.student_encoder_optimizer.step()

            mean_student_encoder_loss += student_encoder_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        mean_entropy /= num_updates
        mean_student_encoder_loss /= num_updates
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "mean_student_encoder_loss": mean_student_encoder_loss,
        }

        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
