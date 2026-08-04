import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

from agent.model import MLP, Critic
from agent.diffusion import Diffusion  # Make sure this class has been updated with qsm_loss() & p_losses()
from agent.vae import VAE
from agent.helpers import EMA


class DiPo(object):
    def __init__(self,
                 args,
                 state_dim,
                 action_space,
                 memory,
                 diffusion_memory,
                 device,
                 ):
        action_dim = np.prod(action_space.shape)
        self.policy_type = 'QSM'

        # Use the QSM version of Diffusion if requested.
        if self.policy_type == 'QSM':
            self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim,
                                   noise_ratio=args.noise_ratio,
                                   beta_schedule=args.beta_schedule,
                                   n_timesteps=args.n_timesteps).to(device)
            # Set the coefficient for scaling the critic gradient in the QSM loss.
            self.actor.q_grad_coeff = getattr(args, "q_grad_coeff", 10.0)
        elif self.policy_type == 'Diffusion':
            self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim,
                                   noise_ratio=args.noise_ratio,
                                   beta_schedule=args.beta_schedule,
                                   n_timesteps=args.n_timesteps).to(device)
        elif self.policy_type == 'VAE':
            self.actor = VAE(state_dim=state_dim, action_dim=action_dim, device=device).to(device)
        else:
            self.actor = MLP(state_dim=state_dim, action_dim=action_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.diffusion_lr, eps=1e-5)

        self.memory = memory
        self.diffusion_memory = diffusion_memory
        self.action_gradient_steps = args.action_gradient_steps
        self.action_grad_norm = action_dim * args.ratio
        self.ac_grad_norm = args.ac_grad_norm
        self.step = 0
        self.tau = args.tau
        self.gamma = 0.9
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.sim_w = 0.1
        self.averageR = 0.01
        self.beta = 0.1

        # For target networks, we always create a copy of the actor.
        self.actor_target = copy.deepcopy(self.actor)
        self.update_actor_target_every = args.update_actor_target_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        self.action_dim = action_dim
        self.action_lr = args.action_lr
        self.device = device

        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = (action_space.high - action_space.low) / 2.
            self.action_bias = (action_space.high + action_space.low) / 2.

    def append_memory(self, state, action, reward, next_state, mask):
        action = (action - self.action_bias) / self.action_scale
        self.memory.append(state, action, reward, next_state, mask)
        self.diffusion_memory.append(state, action)

    def append_memory_batch(self, state, action, reward, next_state, mask):
        action = (action - self.action_bias) / self.action_scale
        self.memory.append_batch(state, action, reward, next_state, mask)
        self.diffusion_memory.append_batch(state, action)

    def sample_action(self, state, eval=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # Use the actor's forward method; for diffusion-based models, this performs denoising.
        action = self.actor(state, eval).cpu().data.numpy().flatten()
        action = np.clip(action, -1, 1)
        action = action * self.action_scale + self.action_bias
        return action

    def compute_and_save_action_gradients(self, states, actions, save_path):
        """
        Compute action gradients and save values and gradients to .npy files.
        """
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32, requires_grad=True)
        actions_optim = torch.optim.Adam([actions_tensor], lr=self.action_lr, eps=1e-5)

        action_gradients = []
        q_values = []

        for step in range(self.action_gradient_steps):
            actions_tensor.requires_grad_(True)
            q1, q2 = self.critic(states_tensor, actions_tensor)
            q_min = -torch.min(q1, q2)
            actions_optim.zero_grad()
            q_min.backward(torch.ones_like(q_min))
            if self.action_grad_norm > 0:
                nn.utils.clip_grad_norm_([actions_tensor], max_norm=self.action_grad_norm, norm_type=2)
            action_gradients.append(actions_tensor.grad.detach().cpu().numpy().copy())
            q_values.append(q_min.detach().cpu().numpy().copy())
            actions_optim.step()
            actions_tensor.requires_grad_(False)
            actions_tensor.clamp_(-1., 1.)

        action_gradients = np.array(action_gradients)
        q_values = np.array(q_values)

        os.makedirs(save_path, exist_ok=True)
        np.save(f"{save_path}/action_gradients.npy", action_gradients)
        np.save(f"{save_path}/q_values.npy", q_values)
        print(f"Action gradients and Q-values saved to {save_path}")
        return action_gradients, q_values

    def action_gradient(self, batch_size, log_writer):
        states, best_actions, idxs = self.diffusion_memory.sample(batch_size)
        actions_optim = torch.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)

        for i in range(self.action_gradient_steps):
            best_actions.requires_grad_(True)
            q1, q2 = self.critic(states, best_actions)
            loss = -torch.min(q1, q2)
            actions_optim.zero_grad()
            loss.backward(torch.ones_like(loss))
            if self.action_grad_norm > 0:
                nn.utils.clip_grad_norm_([best_actions], max_norm=self.action_grad_norm, norm_type=2)
            actions_optim.step()
            best_actions.requires_grad_(False)
            best_actions.clamp_(-1., 1.)

        best_actions = best_actions.detach()
        self.diffusion_memory.append_batch(states.detach().cpu().numpy(), best_actions.cpu().numpy())
        return states, best_actions

    def train(self, iterations, batch_size=256, log_writer=None):
        for _ in range(iterations):
            # Sample replay batch
            states, actions, rewards, next_states, masks = self.memory.sample(batch_size)

            """ Critic Training (Double Q-learning) """
            current_q1, current_q2 = self.critic(states, actions)
            next_actions = self.actor_target(next_states, eval=False)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = (rewards + masks * target_q).detach()
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.ac_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.ac_grad_norm, norm_type=2)
            self.critic_optimizer.step()

            """ Policy Training """
            states, best_actions = self.action_gradient(batch_size, log_writer)
            # For QSM, use the modified loss that aligns predicted noise with the critic gradient.
            if self.policy_type == 'QSM':
                actor_loss = self.actor.qsm_loss(best_actions, states, 1.0, self.critic)
            else:
                actor_loss = self.actor.loss(best_actions, states)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.ac_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.ac_grad_norm, norm_type=2)
            self.actor_optimizer.step()

            """ Soft-update target networks """
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if self.step % self.update_actor_target_every == 0:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            log_writer.log({
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "train_step": self.step,
            })

            self.step += 1

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))
