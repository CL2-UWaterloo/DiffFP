import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from agent.networks import soft_update, TanhMLPPolicy, DistributionalDoubleQ, projection
# (If you use any intrinsic modules, import them here as needed)
# Here we assume that your models are defined in agent.model:
from agent.model import MLP, Critic

class TD3(object):
    def __init__(self, args, state_dim, action_space, memory, diffusion_memory, device):
        """
        TD3 agent that is adapted similarly to the DiPO SAC version.
        
        Args:
            args: hyper-parameters namespace. Must contain:
                - actor_lr, critic_lr, tau, gamma,
                - policy_noise (std for target noise),
                - noise_clip (max absolute noise added to target actions),
                - policy_delay (actor updated every N critic updates),
                - no_tgt_actor (optional: if True the target actor is not separate).
            state_dim: dimensionality (or shape) of the observation space.
            action_space: gym-like action space with attributes 'shape', 'high', 'low'.
            memory: main replay buffer.
            diffusion_memory: secondary replay buffer for the diffusion policy (can be None).
            device: torch device.
        """
        self.args = args
        action_dim = np.prod(action_space.shape)
        
        # Build actor and critic networks.
        self.actor = TanhMLPPolicy(state_dim=state_dim, act_dim=action_dim).to(device)
        self.critic = DistributionalDoubleQ(state_dim, action_dim, device=device).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        # Optionally use a separate target actor.
        self.actor_target = copy.deepcopy(self.actor) #if not args.no_tgt_actor else self.actor
        
        self.actor_optimizer = AdamW(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=args.critic_lr)
        
        self.memory = memory
        self.diffusion_memory = diffusion_memory  # extra buffer for the diffusion policy
        
        self.device = device
        self.tau = args.tau
        self.gamma = args.gamma
        
        # TD3 target smoothing noise and actor update delay.
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_delay = args.policy_delay
        
        # For environments where actions are not naturally in [-1,1],
        # we normalize them.
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0
        
        self.step = 0
        self.args.v_min = 0
        self.args.v_max = 5
        self.args.num_atoms = 51
        self.args.nstep = 1

    def get_tgt_policy_actions(self, obs, sample=True):
        with torch.no_grad():
            actions = self.actor_target(obs)
        if sample:
            actions += torch.randn_like(actions) * 0.1
        return actions

    def append_memory(self, state, action, reward, next_state, mask):
        """
        Append a single transition to both the primary replay buffer and,
        if available, the diffusion memory. The action is normalized to [-1,1].
        """
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
        Given a state, sample an action from the actor network. During training,
        a small Gaussian noise is added for exploration. The resulting action is
        clipped to [-1, 1] and then scaled to the original environment range.
        """
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()
        if not eval:
            noise = np.random.normal(0, self.policy_noise, size=action.shape)
            action = action + noise
        action = np.clip(action, -1, 1)
        action = np.float32(action)
        # Scale back to the environment action range.
        action = action #* self.action_scale + self.action_bias
        return action

    def train(self, iterations, batch_size=256, log_writer=None):
        """
        Run training for a number of iterations. At each iteration, a batch of
        transitions is sampled, the critic is updated and then (delayed) the actor
        is updated. Soft updates of the target networks are performed after the actor update.
        """
        for it in range(iterations):
            states, actions, rewards, next_states, masks = self.memory.sample(batch_size)
            self.update_critic(states, actions, rewards, next_states, masks)
            
            # Delayed actor update.
            if it % self.policy_delay == 0:
                self.update_actor(states)
                soft_update(self.critic_target, self.critic, self.tau)
                soft_update(self.actor_target, self.actor, self.tau)
                
            if log_writer is not None:
                log_writer.log({"train_step": self.step})
            self.step += 1

    def update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            next_actions = self.get_tgt_policy_actions(next_obs)
                
        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target.get_q1_q2(next_obs, next_actions)
            target_Q1_projected = projection(next_dist=target_Q1,
                                            reward=reward,
                                            done=done,
                                            gamma=self.args.gamma ** self.args.nstep,
                                            v_min=self.args.v_min,
                                            v_max=self.args.v_max,
                                            num_atoms=self.args.num_atoms,
                                            support=self.critic.z_atoms,
                                            device=self.device)
            target_Q2_projected = projection(next_dist=target_Q2,
                                            reward=reward,
                                            done=done,
                                            gamma=self.args.gamma ** self.args.nstep,
                                            v_min=self.args.v_min,
                                            v_max=self.args.v_max,
                                            num_atoms=self.args.num_atoms,
                                            support=self.critic.z_atoms,
                                            device=self.device)
            target_Q = torch.min(target_Q1_projected, target_Q2_projected)
       
        current_Q1, current_Q2 = self.critic.get_q1_q2(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    # def update_critic(self, states, actions, rewards, next_states, masks):
    #     """
    #     Update the critic networks with a TD error computed using the target networks.
    #     Noise is added to the target actions to smooth the target Q-values.
    #     """
    #     with torch.no_grad():
    #         # Compute target actions with added noise (and clip the noise).
    #         next_actions = self.actor_target(next_states)
    #         noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
    #         next_actions = (next_actions + noise).clamp(-1, 1)
            
    #         target_q1, target_q2 = self.critic_target(next_states, next_actions)
    #         target_q = torch.min(target_q1, target_q2)
    #         # If masks are 0 when done, then this computes:
    #         # target_q = rewards + gamma * (1-done)*target_q
    #         target_q = rewards + masks * self.gamma * target_q

    #     current_q1, current_q2 = self.critic(states, actions)
    #     critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()

    def update_actor(self, states):
        """
        Update the actor network by maximizing the (estimated) Q-value.
        We use only the first Q estimate (as in the original TD3 paper).
        """
        # Freeze critic parameters during the actor update.
        # for param in self.critic.parameters():
        #     param.requires_grad = False

        # actions_pred = self.actor(states)
        # # Assuming that calling self.critic returns (q1, q2),
        # # we use q1 for the actor update.
        # current_q1, _ = self.critic(states, actions_pred)
        # actor_loss = -current_q1.mean()

        self.critic.requires_grad_(False)
        action = self.actor(states)
        Q = self.critic.get_q_min(states, action)
        actor_loss = -Q.mean()


        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic parameters.
        for param in self.critic.parameters():
            param.requires_grad = True

    def save_model(self, directory, id=None):
        """
        Save the actor and critic models.
        """
        if id is not None:
            torch.save(self.actor.state_dict(), f'{directory}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{directory}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{directory}/actor.pth')
            torch.save(self.critic.state_dict(), f'{directory}/critic.pth')

    def load_model(self, directory, id=None):
        """
        Load the actor and critic models.
        """
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{directory}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{directory}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{directory}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{directory}/critic.pth'))
