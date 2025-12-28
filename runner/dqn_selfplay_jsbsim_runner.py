"""
DQN Runner for JSBSim Environment with Self-play
"""
import time
import torch
import logging
import numpy as np
from typing import List
from .base_runner import Runner


def _t2n(x):
    """Convert torch tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class DQNSelfplayRunner(Runner):
    """
    Runner for training DQN with self-play in JSBSim environment.
    
    Features:
    - Self-play training with opponent pool
    - Experience replay with warmup
    - Epsilon-greedy exploration
    - Periodic target network updates
    """
    
    def load(self):
        """Initialize policies, trainers, and buffers."""
        self.use_selfplay = self.all_args.use_selfplay
        assert self.use_selfplay == True, "This runner requires self-play mode"
        
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.num_opponents = self.all_args.n_choose_opponents
        
        # DQN policy and trainer
        from algorithms.dqn.dqn_policy import DQNPolicy
        from algorithms.dqn.dqn_trainer import DQNTrainer
        
        self.policy = DQNPolicy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.trainer = DQNTrainer(self.all_args, self.policy, device=self.device)
        
        # Self-play setup
        from algorithms.utils.selfplay import get_algorithm
        self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)
        
        self.policy_pool = {}  # type: dict[str, float]
        self.opponent_policy = [
            DQNPolicy(self.all_args, self.obs_space, self.act_space, device=self.device)
            for _ in range(self.num_opponents)
        ]
        self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
        
        # Evaluation opponent
        if self.use_eval:
            self.eval_opponent_policy = DQNPolicy(self.all_args, self.obs_space, self.act_space, device=self.device)
        
        # Episode data storage (for collecting transitions)
        self.episode_obs = [[] for _ in range(self.n_rollout_threads)]
        self.episode_actions = [[] for _ in range(self.n_rollout_threads)]
        self.episode_rewards = [[] for _ in range(self.n_rollout_threads)]
        
        logging.info("\n DQN Selfplay Runner initialized: num_opponents {}.\n"
                    .format(self.num_opponents))
        
        if self.model_dir is not None:
            self.restore()
    
    def run(self):
        """Main training loop."""
        self.warmup()
        
        start = time.time()
        self.total_num_steps = 0
        episodes_trained = 0
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        
        while self.total_num_steps < self.num_env_steps:
            # Collect one step from all parallel environments
            obs, rewards, dones, infos = self.step_envs()
            
            # Check for episode completions and log
            for env_idx in range(self.n_rollout_threads):
                if dones[env_idx].all():
                    # Episode completed, compute returns
                    episode_reward = np.sum(self.episode_rewards[env_idx])
                    episode_length = len(self.episode_rewards[env_idx])
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    
                    # Reset episode buffers
                    self.episode_obs[env_idx] = []
                    self.episode_actions[env_idx] = []
                    self.episode_rewards[env_idx] = []
                    episodes_trained += 1
            
            # Train the network
            train_info = self.trainer.train()
            
            # Logging
            if self.total_num_steps % (self.log_interval * 100) == 0:
                end = time.time()
                fps = int(self.total_num_steps / (end - start))
                
                logging.info("\n Algo {} Exp {} timesteps {}/{}, FPS {}."
                           .format(self.algorithm_name,
                                   self.experiment_name,
                                   self.total_num_steps,
                                   self.num_env_steps,
                                   fps))
                
                if len(episode_rewards) > 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    avg_length = np.mean(episode_lengths[-100:])
                    logging.info("Episodes: {}, Avg Reward: {:.2f}, Avg Length: {:.1f}"
                               .format(episodes_trained, avg_reward, avg_length))
                
                if train_info:
                    logging.info("Loss: {:.4f}, Q_mean: {:.2f}, Q_max: {:.2f}, Epsilon: {:.3f}, Buffer: {}"
                               .format(train_info.get('loss', 0),
                                       train_info.get('q_mean', 0),
                                       train_info.get('q_max', 0),
                                       train_info.get('epsilon', 0),
                                       train_info.get('buffer_size', 0)))
                    
                    # Log to wandb if enabled
                    train_info['average_episode_rewards'] = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                    train_info['average_episode_length'] = np.mean(episode_lengths[-100:]) if episode_lengths else 0
                    self.log_info(train_info, self.total_num_steps)
            
            # Evaluation
            if self.total_num_steps % (self.eval_interval * 1000) == 0 and self.total_num_steps > 0 and self.use_eval:
                self.eval(self.total_num_steps)
            
            # Save model
            if self.total_num_steps % (self.save_interval * 1000) == 0 and self.total_num_steps > 0:
                self.save(self.total_num_steps)
            
            # Update opponents periodically
            if episodes_trained % 100 == 0 and episodes_trained > 0:
                self.update_opponent_pool()
    
    def warmup(self):
        """Reset environment and initialize states."""
        obs = self.envs.reset()
        # Split ego/opponent observations
        self.opponent_obs = obs[:, self.num_agents // 2:, ...]
        self.ego_obs = obs[:, :self.num_agents // 2, ...]
        
        # Initialize episode buffers
        for env_idx in range(self.n_rollout_threads):
            self.episode_obs[env_idx] = [self.ego_obs[env_idx, 0].copy()]
            self.episode_actions[env_idx] = []
            self.episode_rewards[env_idx] = []
    
    def step_envs(self):
        """
        Collect one step from all environments.
        
        Returns:
            obs, rewards, dones, infos
        """
        # Get epsilon for exploration
        epsilon = self.trainer.get_epsilon()
        
        # Get ego actions
        ego_obs_batch = np.concatenate([self.ego_obs[:, i] for i in range(self.ego_obs.shape[1])])
        ego_actions = self.policy.get_actions(ego_obs_batch, epsilon=epsilon)
        ego_actions = ego_actions.reshape(self.n_rollout_threads, self.num_agents // 2, -1)
        
        # Get opponent actions (deterministic)
        opponent_actions = np.zeros_like(ego_actions)
        for policy_idx, policy in enumerate(self.opponent_policy):
            env_idx = self.opponent_env_split[policy_idx]
            opp_obs_batch = np.concatenate([self.opponent_obs[env_idx, i] for i in range(self.opponent_obs.shape[1])])
            opp_actions = policy.get_actions(opp_obs_batch, epsilon=0.0)
            opponent_actions[env_idx] = opp_actions.reshape(len(env_idx), self.num_agents // 2, -1)
        
        # Combine ego and opponent actions
        all_actions = np.concatenate((ego_actions, opponent_actions), axis=1)
        
        # Step environment
        next_obs, rewards, dones, infos = self.envs.step(all_actions)
        
        # Split observations
        next_opponent_obs = next_obs[:, self.num_agents // 2:, ...]
        next_ego_obs = next_obs[:, :self.num_agents // 2, ...]
        
        # Store transitions in replay buffer
        ego_rewards = rewards[:, :self.num_agents // 2, ...]
        ego_dones = dones[:, :self.num_agents // 2, ...]
        
        for env_idx in range(self.n_rollout_threads):
            for agent_idx in range(self.num_agents // 2):
                # Current observation
                obs = self.ego_obs[env_idx, agent_idx]
                # Action taken
                action = ego_actions[env_idx, agent_idx]
                # Reward received
                reward = ego_rewards[env_idx, agent_idx, 0]
                # Next observation
                next_obs_single = next_ego_obs[env_idx, agent_idx]
                # Done flag
                done = ego_dones[env_idx, agent_idx, 0]
                
                # Store in replay buffer
                self.trainer.store_transition(obs, action, reward, next_obs_single, done)
                
                # Store in episode buffer for logging
                self.episode_obs[env_idx].append(obs)
                self.episode_actions[env_idx].append(action)
                self.episode_rewards[env_idx].append(reward)
        
        # Update observations
        self.ego_obs = next_ego_obs
        self.opponent_obs = next_opponent_obs
        
        # Update total steps
        self.total_num_steps += self.n_rollout_threads * (self.num_agents // 2)
        
        return next_obs, rewards, dones, infos
    
    def update_opponent_pool(self):
        """Update opponent pool with current policy."""
        import copy
        
        # Create a snapshot of current policy
        policy_name = f"policy_{len(self.policy_pool)}"
        self.policy_pool[policy_name] = 1500.0  # Initial Elo rating
        
        # Update opponent policies using self-play algorithm
        selected_opponents = self.selfplay_algo.choose_opponent_policy(self.policy_pool, self.num_opponents)
        
        # For now, use latest policy for all opponents
        for opp_policy in self.opponent_policy:
            opp_policy.q_network.load_state_dict(self.policy.q_network.state_dict())
            opp_policy.target_network.load_state_dict(self.policy.target_network.state_dict())
        
        logging.info(f"Updated opponent pool: {len(self.policy_pool)} policies")
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        """Evaluate current policy."""
        logging.info("\nStart evaluation...")
        
        eval_episode_rewards = []
        eval_episode_lengths = []
        total_episodes = 0
        
        # Reset eval environment
        eval_obs = self.eval_envs.reset()
        eval_opponent_obs = eval_obs[:, self.num_agents // 2:, ...]
        eval_ego_obs = eval_obs[:, :self.num_agents // 2, ...]
        
        # Load opponent policy
        self.eval_opponent_policy.q_network.load_state_dict(self.policy.q_network.state_dict())
        self.eval_opponent_policy.target_network.load_state_dict(self.policy.target_network.state_dict())
        
        eval_episode_reward = np.zeros((self.n_eval_rollout_threads, self.num_agents // 2))
        eval_episode_length = np.zeros(self.n_eval_rollout_threads)
        
        while total_episodes < self.eval_episodes:
            # Get actions (deterministic)
            ego_obs_batch = np.concatenate([eval_ego_obs[:, i] for i in range(eval_ego_obs.shape[1])])
            ego_actions = self.policy.get_actions(ego_obs_batch, epsilon=0.0)
            ego_actions = ego_actions.reshape(self.n_eval_rollout_threads, self.num_agents // 2, -1)
            
            opp_obs_batch = np.concatenate([eval_opponent_obs[:, i] for i in range(eval_opponent_obs.shape[1])])
            opp_actions = self.eval_opponent_policy.get_actions(opp_obs_batch, epsilon=0.0)
            opp_actions = opp_actions.reshape(self.n_eval_rollout_threads, self.num_agents // 2, -1)
            
            all_actions = np.concatenate((ego_actions, opp_actions), axis=1)
            
            # Step
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(all_actions)
            
            eval_opponent_obs = eval_obs[:, self.num_agents // 2:, ...]
            eval_ego_obs = eval_obs[:, :self.num_agents // 2, ...]
            
            eval_episode_reward += eval_rewards[:, :self.num_agents // 2, 0]
            eval_episode_length += 1
            
            # Check for done episodes
            for env_idx in range(self.n_eval_rollout_threads):
                if eval_dones[env_idx].all():
                    total_episodes += 1
                    eval_episode_rewards.append(eval_episode_reward[env_idx].mean())
                    eval_episode_lengths.append(eval_episode_length[env_idx])
                    eval_episode_reward[env_idx] = 0
                    eval_episode_length[env_idx] = 0
        
        avg_eval_reward = np.mean(eval_episode_rewards)
        avg_eval_length = np.mean(eval_episode_lengths)
        
        logging.info(f"Evaluation over {self.eval_episodes} episodes:")
        logging.info(f"Average reward: {avg_eval_reward:.2f}")
        logging.info(f"Average length: {avg_eval_length:.1f}")
        
        eval_info = {
            'eval_average_episode_rewards': avg_eval_reward,
            'eval_average_episode_length': avg_eval_length,
        }
        self.log_info(eval_info, total_num_steps)
    
    def save(self, step):
        """Save model checkpoint."""
        import os
        save_dir = self.save_dir + f"/step_{step}"
        os.makedirs(save_dir, exist_ok=True)
        
        policy_save_path = save_dir + "/policy.pt"
        self.policy.save(policy_save_path)
        
        logging.info(f"Model saved at step {step}: {policy_save_path}")
    
    def restore(self):
        """Restore model from checkpoint."""
        import os
        policy_model_path = os.path.join(self.model_dir, "policy.pt")
        
        if os.path.exists(policy_model_path):
            self.policy.load(policy_model_path)
            logging.info(f"Restored policy from {policy_model_path}")
        else:
            logging.warning(f"No checkpoint found at {policy_model_path}")
