"""DiffFP trainer for the MPE-Tag experiment.

Requires the DiffFP PettingZoo fork (github.com/Aku02/PettingZoo), whose
simple_tag_v3 uses a sparse zero-sum payoff (catch threshold 0.1).
Setup: 1 ego agent vs 3 adversaries, 2 obstacles, continuous actions.
"""
import argparse
import numpy as np
import torch
import os
import copy
import wandb
from agent.DiPo import DiPo
from agent.sac import SAC
from agent.td3 import TD3
from agent.qsm import DiPo as QSM
from agent.qvpo import QVPO
from agent.replay_memory import ReplayMemory, DiffusionMemory
from pettingzoo.mpe import simple_tag_v3
import imageio

# Best-response learners: all expose the same interface
# (sample_action / append_memory / train / .actor callable as actor(state, eval=...)).
ALGOS = {'dipo': DiPo, 'sac': SAC, 'td3': TD3, 'qsm': QSM, 'qvpo': QVPO}

def readParser():
    parser = argparse.ArgumentParser(description='Diffusion Policy')
    parser.add_argument('--env_name', default="simple_tag",
                        help='Mujoco Gym environment (default: Hopper-v3)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')

    parser.add_argument('--num_steps', type=int, default=200, metavar='N',
                        help='env timesteps (default: 1000000)')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--update_actor_target_every', type=int, default=1, metavar='N',
                        help='update actor target per iteration (default: 1)')

    parser.add_argument("--policy_type", type=str, default="Diffusion", metavar='S',
                        help="Diffusion, VAE or MLP")
    parser.add_argument("--beta_schedule", type=str, default="cosine", metavar='S',
                        help="linear, cosine or vp")
    parser.add_argument('--n_timesteps', type=int, default=10, metavar='N',
                        help='diffusion timesteps (default: 100)')
    parser.add_argument('--diffusion_lr', type=float, default=0.0003, metavar='G',
                        help='diffusion learning rate (default: 0.0003)')
    parser.add_argument('--critic_lr', type=float, default=0.0003, metavar='G',
                        help='critic learning rate (default: 0.0003)')
    parser.add_argument('--action_lr', type=float, default=0.03, metavar='G',
                        help='diffusion learning rate (default: 0.03)')
    parser.add_argument('--noise_ratio', type=float, default=1.0, metavar='G',
                        help='noise ratio in sample process (default: 1.0)')

    parser.add_argument('--action_gradient_steps', type=int, default=20, metavar='N',
                        help='action gradient steps (default: 20)')
    parser.add_argument('--ratio', type=float, default=0.1, metavar='G',
                        help='the ratio of action grad norm to action_dim (default: 0.1)')
    parser.add_argument('--ac_grad_norm', type=float, default=2.0, metavar='G',
                        help='actor and critic grad norm (default: 1.0)')

    parser.add_argument('--algo', type=str, default='dipo', choices=['dipo', 'sac', 'td3', 'qsm', 'qvpo'],
                        help='best-response learner (default: dipo)')
    # SAC-specific
    parser.add_argument('--actor_lr', type=float, default=0.0003, metavar='G',
                        help='SAC/TD3 actor learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                        help='SAC entropy temperature; set to fixed value (default: 0.1)')
    parser.add_argument('--alpha_lr', type=float, default=0.3, metavar='G',
                        help='SAC entropy temperature learning rate (used only if alpha is learned)')
    # TD3-specific
    parser.add_argument('--policy_noise', type=float, default=0.2, metavar='G',
                        help='TD3 exploration/target noise std (default: 0.2)')
    parser.add_argument('--noise_clip', type=float, default=0.5, metavar='G',
                        help='TD3 target noise clip (default: 0.5)')
    parser.add_argument('--policy_delay', type=int, default=2, metavar='N',
                        help='TD3 delayed actor update period (default: 2)')
    # QSM-specific
    parser.add_argument('--q_grad_coeff', type=float, default=10.0, metavar='G',
                        help='QSM critic-gradient scaling coefficient (default: 10.0)')
    # QVPO-specific (Q-weighted regression on the diffusion loss; enable with --weighted)
    parser.add_argument('--weighted', action="store_true",
                        help='QVPO: weight the diffusion loss by transformed Q-values')
    parser.add_argument('--q_transform', type=str, default='qadv', metavar='S',
                        help='QVPO: Q-weight transform (qrelu, qexpn, qcut, qcutn, qadv)')
    parser.add_argument('--aug', action="store_true",
                        help='QVPO: sample-augmentation path instead of the diffusion buffer')
    parser.add_argument('--gradient', action="store_true",
                        help='QVPO: use gradient-based augmentation')
    parser.add_argument('--beta', type=float, default=1.0, metavar='G',
                        help='QVPO: exp-weighting temperature (default: 1.0)')
    parser.add_argument('--alpha_mean', type=float, default=0.001, metavar='G',
                        help='QVPO: running Q-mean update rate (default: 0.001)')
    parser.add_argument('--alpha_std', type=float, default=0.001, metavar='G',
                        help='QVPO: running Q-std update rate (default: 0.001)')
    parser.add_argument('--chosen', type=int, default=1, metavar='N',
                        help='QVPO: candidates kept per state (default: 1)')
    parser.add_argument('--q_neg', type=float, default=0.0, metavar='G',
                        help='QVPO: Q offset for weighting (default: 0.0)')
    parser.add_argument('--cut', type=float, default=1.0, metavar='G',
                        help='QVPO: weight cutoff (default: 1.0)')
    parser.add_argument('--train_sample', type=int, default=64, metavar='N',
                        help='QVPO: training candidate samples (default: 64)')
    parser.add_argument('--behavior_sample', type=int, default=4, metavar='N',
                        help='QVPO: behavior-time action candidates ranked by Q (default: 4)')
    parser.add_argument('--target_sample', type=int, default=4, metavar='N',
                        help='QVPO: target-actor action candidates (default: 4)')
    parser.add_argument('--eval_sample', type=int, default=32, metavar='N',
                        help='QVPO: eval-time action candidates ranked by Q (default: 32)')
    parser.add_argument('--deterministic', action="store_true",
                        help='QVPO: noiseless sampling at eval')
    parser.add_argument('--policy_freq', type=int, default=1, metavar='N',
                        help='QVPO: actor update period (default: 1)')
    parser.add_argument('--epsilon', type=float, default=0.0, metavar='G',
                        help='QVPO: probability of plain (non-Q-ranked) sampling (default: 0.0)')
    parser.add_argument('--entropy_alpha', type=float, default=0.02, metavar='G',
                        help='QVPO: entropy regularization weight (default: 0.02)')

    parser.add_argument('--cuda', default='cuda:0',
                        help='run on CUDA (default: cuda:0)')

    return parser.parse_args()


def pad_obs(obs):
    """Helper function to pad observation if needed."""
    if obs.shape[0] == 14:
        return np.pad(obs, (0, 2), mode='constant')
    return obs


class FSPTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.cuda)
        self.env = self._initialize_environment()
        self.logger = self._initialize_logger()
        agent1, agent2 = self._initialize_agents()
        self.agent1 = agent1
        self.agent2 = agent2
        self.policy_avg_ego = [self.snapshot_policy(self.agent1)]
        self.policy_avg_opp = [self.snapshot_policy(self.agent2)]
        self.p_sample_dist_ego = [1.0]
        self.p_sample_dist_opp = [1.0]
        self.global_step = 0

    def snapshot_policy(self, agent):
        """Freeze the agent's current policy for the FP averaging pool.

        Every pool entry is callable as policy(state, eval=...). QVPO actors
        rank sampled action candidates by Q at generation time, so for QVPO the
        critic is snapshotted together with the actor."""
        actor_copy = copy.deepcopy(agent.actor)
        if self.args.algo == 'qvpo':
            critic_copy = copy.deepcopy(agent.critic)

            def policy(state, eval=False):
                return actor_copy(state, eval, q_func=critic_copy, normal=False)

            return policy
        return actor_copy

    def _initialize_environment(self):
        # Note: using the parallel API here.
        # return simple_adversary_v3.parallel_env(
        #     N=1, 
        #     continuous_actions=True, 
        #     render_mode="rgb_array", 
        #     max_cycles=100, 
        #     dynamic_rescaling=True
        # )
        return simple_tag_v3.parallel_env(num_good =1,num_adversaries =3, num_obstacles=2, render_mode="rgb_array", max_cycles=150, continuous_actions=True, dynamic_rescaling=True)
        # Alternatively, you could use simple_tag_v3.parallel_env(...) if desired.

    def _initialize_logger(self):
        return wandb.init(
            project="fsp_training_mpe_tag",
            config=self.args.__dict__,
            name="dipo_mpe_tag",
        )

    def _initialize_agents(self):
        # Using the unwrapped environment to get observation and action spaces
        obs_space = self.env.unwrapped.observation_spaces['adversary_0']
        act_space = self.env.unwrapped.action_spaces['agent_0']
        state_size = int(np.prod(obs_space.shape))
        action_size = int(np.prod(act_space.shape))
        memory_size = 1e6

        memory1 = ReplayMemory(state_size, action_size, memory_size, self.device)
        diffusion_memory1 = DiffusionMemory(state_size, action_size, memory_size, self.device)

        memory2 = ReplayMemory(state_size, action_size, memory_size, self.device)
        diffusion_memory2 = DiffusionMemory(state_size, action_size, memory_size, self.device)

        AgentCls = ALGOS[self.args.algo]
        agent1 = AgentCls(self.args, state_size, act_space, memory1, diffusion_memory1, self.device)
        agent2 = AgentCls(self.args, state_size, act_space, memory2, diffusion_memory2, self.device)

        return agent1, agent2

    def update_policy_distribution(self, policy_avg, p_sample_dist):
        num_policies = len(policy_avg)
        current_time = num_policies

        avg_policy_prob = (current_time - 1) / (current_time + 1)
        latest_policy_prob = 2 / (current_time + 1)

        scaled_latest_prob = (1 / avg_policy_prob) * latest_policy_prob
        new_sample_dist = p_sample_dist + [scaled_latest_prob]

        total_sum = sum(new_sample_dist)
        normalized_sample_dist = [p / total_sum for p in new_sample_dist]

        return normalized_sample_dist

    def rescale_action(self, action):
        return (action + 1) / 2

    def compute_exploitability(self):
        """
        Evaluate exploitability using the parallel API.
        For each player (ego and opponent) we run two evaluations:
         - One using the current (average) policy.
         - One using the best response.
        The difference gives a measure of exploitability.
        """
        exp_opp_list = []
        exp_ego_list = []

        # For each player, evaluate current and best-response returns
        for player in ['ego', 'opp']:
            # ---- Evaluate current policy ----
            observations, infos = self.env.reset(seed=100)
            episode_reward_current = 0.0

            # Loop until episode termination (agents are removed once done)
            while self.env.agents:
                actions = {}
                for agent, obs in observations.items():
                    obs = pad_obs(obs)
                    # Select action based on current (average) policies.
                    if agent in ('agent_0', 'agent_1'):
                        if player == 'ego':
                            chosen_policy = np.random.choice(self.policy_avg_ego, p=self.p_sample_dist_ego)
                        else:
                            chosen_policy = np.random.choice(self.policy_avg_opp, p=self.p_sample_dist_opp)
                    elif agent in ('adversary_0', 'adversary_1', 'adversary_2'):
                        if player == 'opp':
                            chosen_policy = np.random.choice(self.policy_avg_opp, p=self.p_sample_dist_opp)
                        else:
                            chosen_policy = np.random.choice(self.policy_avg_ego, p=self.p_sample_dist_ego)
                    else:
                        chosen_policy = None

                    if chosen_policy is not None:
                        action = chosen_policy(torch.FloatTensor(obs.flatten().reshape(1, -1)).to(self.device), eval=True)
                        action = action.cpu().data.numpy().flatten()
                    else:
                        action = None
                    actions[agent] = self.rescale_action(action) if action is not None else None

                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                # Accumulate rewards for the appropriate player.
                for agent, reward in rewards.items():
                    if player == 'ego' and agent in ('agent_0', 'agent_1'):
                        episode_reward_current += reward
                    elif player == 'opp' and agent in ('adversary_0', 'adversary_1', 'adversary_2'):
                        episode_reward_current += reward

            # ---- Evaluate best response ----
            observations, infos = self.env.reset(seed=100)
            episode_reward_best_response = 0.0

            while self.env.agents:
                actions = {}
                for agent, obs in observations.items():
                    obs = pad_obs(obs)
                    if agent in ('agent_0', 'agent_1'):
                        if player == 'ego':  # Ego best response
                            action = self.agent1.sample_action(obs.flatten(), eval=True)
                        else:
                            chosen_policy = np.random.choice(self.policy_avg_opp, p=self.p_sample_dist_opp)
                            action = chosen_policy(torch.FloatTensor(obs.flatten().reshape(1, -1)).to(self.device), eval=True)
                            action = action.cpu().data.numpy().flatten()
                    elif agent in ('adversary_0', 'adversary_1', 'adversary_2'):
                        if player == 'opp':  # Opponent best response
                            action = self.agent2.sample_action(obs.flatten(), eval=True)
                        else:
                            chosen_policy = np.random.choice(self.policy_avg_ego, p=self.p_sample_dist_ego)
                            action = chosen_policy(torch.FloatTensor(obs.flatten().reshape(1, -1)).to(self.device), eval=True)
                            action = action.cpu().data.numpy().flatten()
                    else:
                        action = None
                    actions[agent] = self.rescale_action(action) if action is not None else None

                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                for agent, reward in rewards.items():
                    if player == 'ego' and agent in ('agent_0', 'agent_1'):
                        episode_reward_best_response += reward
                    elif player == 'opp' and agent in ('adversary_0', 'adversary_1', 'adversary_2'):
                        episode_reward_best_response += reward

            print(f"Episode reward current_{player}: {episode_reward_current}, Episode reward best response_{player}: {episode_reward_best_response}")
            if player == 'ego':
                exp_opp = episode_reward_best_response - episode_reward_current
                exp_opp_list.append(exp_opp)
            else:
                exp_ego = episode_reward_current - episode_reward_best_response
                exp_ego_list.append(exp_ego)

        exp_opp = np.mean(exp_opp_list) if exp_opp_list else 0.0
        exp_ego = np.mean(exp_ego_list) if exp_ego_list else 0.0
        total_exploitability = exp_opp + exp_ego
        sum_exploitability = np.sum(exp_opp_list + exp_ego_list)
        print(f"Opponent exploitability: {exp_opp:.4f}, \nEgo exploitability: {exp_ego:.4f}")
        return exp_opp, exp_ego, total_exploitability, sum_exploitability

    def train_agent_against_average(self, agent_name):
        episode = 0
        steps = 0

        while steps <= self.args.num_steps:
            observations, infos = self.env.reset()
            steps_in_env = 0
            episode_reward = 0.0

            while self.env.agents:
                actions = {}
                for agent, obs in observations.items():
                    obs = pad_obs(obs)
                    if agent_name == 'ego' and agent in ('agent_0', 'agent_1'):
                        action = self.agent1.sample_action(obs.flatten(), eval=False)
                    elif agent_name == 'opp' and agent in ('adversary_0', 'adversary_1', 'adversary_2'):
                        action = self.agent2.sample_action(obs.flatten(), eval=False)
                    else:
                        if agent in ('agent_0', 'agent_1'):
                            opponent_policy = np.random.choice(self.policy_avg_opp, p=self.p_sample_dist_opp)
                        elif agent in ('adversary_0', 'adversary_1', 'adversary_2'):
                            opponent_policy = np.random.choice(self.policy_avg_ego, p=self.p_sample_dist_ego)
                        action = opponent_policy(torch.FloatTensor(obs.reshape(1, -1)).to(self.device), eval=False)
                        action = action.cpu().data.numpy().flatten()
                    action = np.clip(action, -1, 1)
                    actions[agent] = self.rescale_action(action) if action is not None else None

                new_obs, rewards, terminations, truncations, infos = self.env.step(actions)

                # Store transitions and accumulate rewards
                for agent in actions.keys():
                    done = terminations.get(agent, False) or truncations.get(agent, False)
                    if not done:
                        mask = 0.0 if done else self.args.gamma
                        next_obs = pad_obs(new_obs[agent])
                        observations[agent] = pad_obs(observations[agent])
                        if agent in ('agent_0', 'agent_1') and agent_name == 'ego':
                            self.agent1.append_memory(observations[agent].flatten(), actions[agent], rewards[agent],
                                                        next_obs.flatten(), mask)
                            episode_reward += rewards[agent]
                        elif agent in ('adversary_0', 'adversary_1', 'adversary_2') and agent_name == 'opp':
                            self.agent2.append_memory(observations[agent].flatten(), actions[agent], rewards[agent],
                                                        next_obs.flatten(), mask)
                            episode_reward += rewards[agent]

                # Train the appropriate agent
                trained_agent = self.agent1 if agent_name == 'ego' else self.agent2
                if self.args.algo == 'qvpo':
                    # QVPO's train() takes the global step first (for its schedules).
                    trained_agent.train(self.global_step, 8, batch_size=self.args.batch_size, log_writer=self.logger)
                else:
                    trained_agent.train(8, batch_size=self.args.batch_size, log_writer=self.logger)

                steps += 1  # One joint step across all agents
                self.global_step += 1
                steps_in_env += 1
                observations = new_obs

            episode += 1
            if episode % 10 == 0:
                print(f"Episode: {episode}, steps in env: {steps_in_env}, global step: {self.global_step}")

        # Update average policy distribution
        if agent_name == 'ego':
            new_policy = self.snapshot_policy(self.agent1)
            self.policy_avg_ego.append(new_policy)
            self.p_sample_dist_ego = self.update_policy_distribution(self.policy_avg_ego, self.p_sample_dist_ego)
        else:
            new_policy = self.snapshot_policy(self.agent2)
            self.policy_avg_opp.append(new_policy)
            self.p_sample_dist_opp = self.update_policy_distribution(self.policy_avg_opp, self.p_sample_dist_opp)

    def evaluate(self, iteration):
        """Evaluate the agents using the parallel API and record a video."""
        episodes = 1
        eval_env = self._initialize_environment()
        video_folder = "mpe_tag_landmark_1v3_original_latest"
        os.makedirs(video_folder, exist_ok=True)
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            observations, infos = eval_env.reset(seed=100 + i)
            episode_reward = 0.0
            video_file = os.path.join(video_folder, f"episode_{iteration}.mp4")
            writer = imageio.get_writer(video_file, fps=30)

            while eval_env.agents:
                actions = {}
                for agent, obs in observations.items():
                    obs = pad_obs(obs)
                    if agent in ('agent_0', 'agent_1'):
                        action = self.agent1.sample_action(obs.flatten(), eval=True)
                    elif agent in ('adversary_0', 'adversary_1', 'adversary_2'):
                        action = self.agent2.sample_action(obs.flatten(), eval=True)
                    else:
                        raise ValueError(f"Unexpected agent: {agent}")
                    actions[agent] = self.rescale_action(action) if action is not None else None

                observations, rewards, terminations, truncations, infos = eval_env.step(actions)
                # Sum rewards across agents.
                for reward in rewards.values():
                    episode_reward += reward

                # Render and record the frame.
                frame = eval_env.render()
                if frame is not None:
                    writer.append_data(np.array(frame))
            returns[i] = episode_reward
            writer.close()

        mean_return = np.mean(returns)
        self.logger.log({
            "reward/test": mean_return,
            "steps": self.global_step,
        })

        print('-' * 60)
        print(f'Num steps: {self.global_step:<5}  reward: {mean_return:<5.1f}')
        print('-' * 60)
        eval_env.close()

    def populate_buffer(self, seed=42, gamma=0.9, buffer_size=10000):
        """Populate replay buffer using the parallel API."""
        np.random.seed(seed)
        steps = 0

        while steps <= buffer_size:
            observations, infos = self.env.reset()
            while self.env.agents:
                actions = {}
                for agent, obs in observations.items():
                    obs = pad_obs(obs)
                    actions[agent] = self.env.action_space(agent).sample() #self.rescale_action()
                new_obs, rewards, terminations, truncations, infos = self.env.step(actions)

                for agent in actions.keys():
                    done = terminations.get(agent, False) or truncations.get(agent, False)
                    if not done:
                        mask = 0.0 if done else gamma
                        next_obs = pad_obs(new_obs[agent])
                        observations[agent] = pad_obs(observations[agent])
                        if agent in ('agent_0', 'agent_1'):
                            self.agent1.append_memory(observations[agent].flatten(), actions[agent], rewards[agent],
                                                        next_obs.flatten(), mask)
                        elif agent in ('adversary_0', 'adversary_1', 'adversary_2'):
                            self.agent2.append_memory(observations[agent].flatten(), actions[agent], rewards[agent],
                                                        next_obs.flatten(), mask)
                        steps += 1
                        if steps >= buffer_size:
                            print("Buffer populated.")
                            return
                observations = new_obs

    def main(self):
        print("Populating buffer...")
        self.populate_buffer()

        fsp_iterations = 100
        for fsp_iter in range(fsp_iterations):
            for agent_name in ['opp', 'ego']:
                self.train_agent_against_average(agent_name)

            exp_opp, exp_ego, exploitability, sum_exploitability = self.compute_exploitability()
            self.logger.log({
                "fsp_iteration": fsp_iter + 1,
                "exploitability/total": exploitability,
                "exploitability/mean": sum_exploitability,
                "exploitability/opp": exp_opp,
                "exploitability/ego": exp_ego,
            })

            print(f"Exploitability after FSP Iteration {fsp_iter + 1}: {exploitability:.4f}")
            print(f"Mean Exploitability: {sum_exploitability:.4f}")
            if fsp_iter % 5 == 0 and fsp_iter > 0:
                self.evaluate(fsp_iter)
                self.agent1.save_model(dir='mpe_tag_ckpts/dipo_mpe_save_model_2', id=fsp_iter)
                self.agent2.save_model(dir='mpe_tag_ckpts/dipo_mpe_save_model_2', id=fsp_iter)

        self.evaluate(fsp_iter)
        self.agent1.save_model(dir='mpe_tag_ckpts/dipo_mpe_save_model_2', id=fsp_iter)
        self.agent2.save_model(dir='mpe_tag_ckpts/dipo_mpe_save_model_2', id=fsp_iter)
        print("FSP training complete.")


if __name__ == "__main__":
    args = readParser()
    trainer = FSPTrainer(args)
    trainer.main()
