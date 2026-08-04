import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from agent.networks import soft_update, TanhDiagGaussianMLPPolicy
from agent.model import MLP, Critic

class SAC(object):
    def __init__(self, args, state_dim, action_space, memory, diffusion_memory, device):
        """
        Updated SAC constructor:
          - Accepts a diffusion memory buffer along with the main replay memory.
          - Stores args as self.args for use in later updates.
        """
        self.args = args
        action_dim = np.prod(action_space.shape)

        self.actor = TanhDiagGaussianMLPPolicy(state_dim=state_dim, act_dim=action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor) #if not args.no_tgt_actor else self.actor

        self.actor_optimizer = AdamW(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=args.critic_lr)

        self.memory = memory
        self.diffusion_memory = diffusion_memory  # extra buffer for diffusion policy
        self.device = device
        self.tau = args.tau
        self.gamma = args.gamma
        self.alpha = args.alpha if args.alpha is not None else nn.Parameter(torch.zeros(1, device=device))
        self.target_entropy = -action_dim
        self.alpha_optimizer = AdamW([self.alpha], lr=args.alpha_lr) if args.alpha is None else None

        # self.n_step_buffer = NStepReplay(state_dim, action_dim, args.num_envs, args.nstep, device=device)
        # self.intrinsic = IntrinsicM(state_dim, type=args.intrinsic_type, normalize=args.intrinsic_normalize,
        #                             pos_enc=args.intrinsic_pos_enc, L=args.intrinsic_L, device=device)

        self.action_scale = (action_space.high - action_space.low) / 2.
        self.action_bias = (action_space.high + action_space.low) / 2.

        self.step = 0

    def append_memory(self, state, action, reward, next_state, mask):
        """
        Append a single transition to both the primary replay buffer and, if available,
        the diffusion memory buffer. Note that the action is normalized to the [-1, 1]
        range (the same scale used during training).
        """
        # Normalize action from environment scale to [-1, 1]
        action_norm = (action - self.action_bias) / self.action_scale
        
        self.memory.append(state, action_norm, reward, next_state, mask)
        if self.diffusion_memory is not None:
            self.diffusion_memory.append(state, action_norm)

    def append_memory_batch(self, states, actions, rewards, next_states, masks):
        """
        Append a batch of transitions to both replay buffers.
        """
        actions_norm = (actions - self.action_bias) / self.action_scale
        
        self.memory.append_batch(states, actions_norm, rewards, next_states, masks)
        if self.diffusion_memory is not None:
            self.diffusion_memory.append_batch(states, actions_norm)

    def sample_action(self, state, eval=False):
        """
        Given a state, sample an action from the actor network.
        The sampled action is clipped to the [-1, 1] range and then scaled
        back to the original action space.
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state, eval).cpu().data.numpy().flatten()
        # Clip action and then scale + bias it back to the original range.
        action = np.clip(action, -1, 1) * self.action_scale + self.action_bias
        return action

    def train(self, iterations, batch_size=256, log_writer=None):
        """
        SAC training loop. The logic remains SAC only.
        """
        for _ in range(iterations):
            states, actions, rewards, next_states, masks = self.memory.sample(batch_size)
            # Optionally, one might add intrinsic rewards here:
            # rewards += self.intrinsic.compute_reward(states, next_states)
            
            self.update_critic(states, actions, rewards, next_states, masks)
            self.update_actor(states)
            
            soft_update(self.critic_target, self.critic, self.tau)
            # if not self.args.no_tgt_actor:
            #     soft_update(self.actor_target, self.actor, self.tau)
            
            if log_writer is not None:
                log_writer.log({
                    "train_step": self.step,
                })
            self.step += 1

    def update_critic(self, states, actions, rewards, next_states, masks):
        """
        Update the critic network using a target computed from the target networks.
        """
        with torch.no_grad():
            # next_actions = self.actor_target(next_states, eval=False)
            next_actions, _, log_prob = self.actor.get_actions_logprob(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            # Incorporate an entropy bonus (using log-probability) in the target Q-value.
            target_q = torch.min(target_q1, target_q2) - self.get_alpha() * log_prob
            target_q = rewards + masks * (self.gamma) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, states):
        """
        Update the actor network by minimizing the weighted (by alpha) log-probability
        of the actions minus the critic's estimated Q-values.
        """
        # Freeze critic gradients during actor update.
        self.critic.requires_grad_(False)
        actions, _, log_prob = self.actor.get_actions_logprob(states)
        q1, q2 = self.critic(states, actions)
        q_min = torch.min(q1, q2)
        actor_loss = (self.get_alpha() * log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        # Automatic entropy tuning
        if self.alpha_optimizer:
            alpha_loss = (self.get_alpha(False) * (-torch.log(torch.abs(actions) + 1e-6) - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

    def get_alpha(self, detach=True):
        """
        Returns the current entropy temperature (alpha). Optionally detaches
        the value from the computation graph.
        """
        if detach and isinstance(self.alpha, nn.Parameter):
            return self.alpha.exp().detach()
        return self.alpha

    def save_model(self, dir, id=None):
        """
        Save the actor and critic models.
        """
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        """
        Load the actor and critic models.
        """
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))
