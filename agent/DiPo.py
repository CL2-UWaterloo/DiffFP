import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from agent.model import MLP, Critic
from agent.diffusion import Diffusion
from agent.vae import VAE
from agent.helpers import EMA
import os


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

        self.policy_type = args.policy_type
        if self.policy_type == 'Diffusion':
            self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, noise_ratio=args.noise_ratio,
                                   beta_schedule=args.beta_schedule, n_timesteps=args.n_timesteps).to(device)
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

        action = self.actor(state, eval).cpu().data.numpy().flatten()
        action = action.clip(-1, 1)
        action = action * self.action_scale + self.action_bias
        return action
    
    def compute_and_save_action_gradients(self, states, actions, save_path):
        """
        Compute action gradients and save values and gradients to .npy files.

        Args:
            states (numpy.ndarray): Input states for gradient computation.
            actions (numpy.ndarray): Input actions for gradient computation.
            critic (callable): Critic model to evaluate the Q-values.
            action_lr (float): Learning rate for action optimization.
            action_grad_norm (float): Maximum gradient norm for clipping.
            action_gradient_steps (int): Number of gradient steps.
            save_path (str): Directory path to save .npy files.
        """
        # Convert states and actions to PyTorch tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32, requires_grad=True)

        # Optimizer for actions
        actions_optim = torch.optim.Adam([actions_tensor], lr=self.action_lr, eps=1e-5)

        # Log gradients and Q-values
        action_gradients = []
        q_values = []

        for step in range(self.action_gradient_steps):
            actions_tensor.requires_grad_(True)

            # Evaluate Q-values
            q1, q2 = self.critic(states_tensor, actions_tensor)
            q_min = -torch.min(q1, q2)

            # Compute gradients
            actions_optim.zero_grad()
            q_min.backward(torch.ones_like(q_min))  # Backpropagate

            if self.action_grad_norm > 0:
                nn.utils.clip_grad_norm_([actions_tensor], max_norm=self.action_grad_norm, norm_type=2)

            # Log gradients and Q-values
            action_gradients.append(actions_tensor.grad.detach().cpu().numpy().copy())
            q_values.append(q_min.detach().cpu().numpy().copy())

            # Update actions
            actions_optim.step()

            actions_tensor.requires_grad_(False)
            actions_tensor.clamp_(-1., 1.)

        # Convert logged data to numpy arrays
        action_gradients = np.array(action_gradients)
        q_values = np.array(q_values)

        # Save gradients and Q-values to .npy files
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
                actions_grad_norms = nn.utils.clip_grad_norm_([best_actions], max_norm=self.action_grad_norm, norm_type=2)

            actions_optim.step()

            best_actions.requires_grad_(False)
            best_actions.clamp_(-1., 1.)

        # if self.step % 10 == 0:
        #     log_writer.add_scalar('Action Grad Norm', actions_grad_norms.max().item(), self.step)

        best_actions = best_actions.detach()

        # self.diffusion_memory.replace(idxs, best_actions.cpu().numpy())
        self.diffusion_memory.append_batch(states.detach().cpu().numpy(), best_actions.cpu().numpy())

        return states, best_actions

    def train(self, iterations, batch_size=256, log_writer=None):
        for _ in range(iterations):
            # Sample replay buffer / batch
            states, actions, rewards, next_states, masks = self.memory.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(states, actions)

            next_actions = self.actor_target(next_states, eval=False)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            target_q = (rewards + masks * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.ac_grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.ac_grad_norm, norm_type=2)
                # if self.step % 10 == 0:
                #     log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            self.critic_optimizer.step()

            """ Policy Training """
            states, best_actions = self.action_gradient(batch_size, log_writer)

            actor_loss = self.actor.loss(best_actions, states)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.ac_grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.ac_grad_norm, norm_type=2)
                # if self.step % 10 == 0:
                #     log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
            self.actor_optimizer.step()

            """ Step Target network """
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

    # def train(self, iterations, batch_size=256, log_writer=None):
    #     for _ in range(iterations):
    #         # Sample replay buffer / batch
    #         states, actions, rewards, next_states, masks = self.memory.sample(batch_size)

    #         """ Q Training """
    #         current_q1, current_q2 = self.critic(states, actions)

    #         # Generate noisy next actions
    #         next_actions = self.actor_target(next_states, eval=False)
    #         noise = torch.clamp(
    #             torch.randn_like(next_actions) * self.policy_noise, -self.noise_clip, self.noise_clip
    #         )
    #         next_actions = torch.clamp(next_actions + noise, -1, 1)

    #         # Behavioral cloning penalty
    #         bc_penalty = torch.sum((next_actions - actions) ** 2, dim=-1)

    #         # Compute next Q values
    #         next_q1, next_q2 = self.critic_target(next_states, next_actions)
    #         next_q = torch.min(next_q1, next_q2) - self.beta * bc_penalty

    #         # Compute target Q
    #         target_q = rewards + masks * self.gamma * next_q.detach()

    #         def critic_loss_fn():
    #             # Compute current Q values
    #             q1, q2 = self.critic(states, actions)
    #             q_min = torch.min(q1, q2).mean()

    #             # Compute first-order consistency loss
    #             states.requires_grad_(True)
    #             actions.requires_grad_(True)

    #             q_values = self.critic(states, actions)
    #             q_value_mean = torch.mean(q_values[0])

    #             state_grad = torch.autograd.grad(q_value_mean, states, retain_graph=True, create_graph=True)[0]
    #             action_grad = torch.autograd.grad(q_value_mean, actions, retain_graph=True, create_graph=True)[0]

    #             state_diff = next_states - states
    #             cosine_similarity = F.cosine_similarity(state_grad, state_diff, dim=-1)

    #             random_state = states - state_diff * torch.rand_like(state_diff)
    #             random_q_values = self.critic(random_state, actions)
    #             random_q_value_mean = torch.mean(random_q_values[0])

    #             random_state_grad = torch.autograd.grad(random_q_value_mean, random_state, retain_graph=True, create_graph=True)[0]
    #             random_cosine_similarity = F.cosine_similarity(random_state_grad, state_diff, dim=-1)

    #             action_grad_norm = torch.sum(action_grad ** 2, dim=-1)
    #             action_grad_norm_clipped = torch.where(
    #                 action_grad_norm < 1.0,
    #                 torch.zeros_like(action_grad_norm),
    #                 action_grad_norm,
    #             )

    #             first_order_loss = torch.exp(rewards - self.averageR) * (-cosine_similarity - random_cosine_similarity + action_grad_norm_clipped)

    #             # Bellman update loss
    #             td_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

    #             # Combine losses
    #             total_loss = td_loss + self.sim_w * first_order_loss.mean()
    #             return total_loss, (q_min, td_loss, first_order_loss.mean())

    #         # Compute loss and gradients
    #         self.critic_optimizer.zero_grad()
    #         critic_loss, (q_min, td_loss, sim_loss) = critic_loss_fn()
    #         critic_loss.backward()

    #         if self.ac_grad_norm > 0:
    #             nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.ac_grad_norm, norm_type=2)

    #         self.critic_optimizer.step()

    #         """ Policy Training """
    #         states, best_actions = self.action_gradient(batch_size, log_writer)

    #         actor_loss = self.actor.loss(best_actions, states)

    #         self.actor_optimizer.zero_grad()
    #         actor_loss.backward()

    #         if self.ac_grad_norm > 0:
    #             nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.ac_grad_norm, norm_type=2)

    #         self.actor_optimizer.step()

    #         """ Step Target network """
    #         for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
    #             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    #         if self.step % self.update_actor_target_every == 0:
    #             for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
    #                 target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    #         # Log metrics
    #         if log_writer and self.step % 10 == 0:
    #             log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
    #             log_writer.add_scalar('TD Loss', td_loss.item(), self.step)
    #             log_writer.add_scalar('Sim Loss', sim_loss.item(), self.step)

    #         self.step += 1


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

