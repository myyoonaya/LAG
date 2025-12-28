"""
DQN Trainer
Implements training logic for Dueling DQN with experience replay
"""
import torch
import torch.nn.functional as F
import numpy as np
from .replay_buffer import ReplayBuffer


class DQNTrainer:
    """
    Trainer for DQN with multi-head action decomposition.
    
    Features:
    - Experience replay with warmup
    - Target network for stable learning
    - Periodic target network updates
    - Per-head Q-value learning
    """
    
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.args = args
        self.policy = policy
        self.device = device
        
        # Training hyperparameters
        self.gamma = args.gamma
        self.batch_size = getattr(args, 'dqn_batch_size', 32)
        self.update_interval = getattr(args, 'dqn_update_interval', 4)
        self.target_update_interval = getattr(args, 'dqn_target_update_interval', 10000)
        self.warmup_steps = getattr(args, 'dqn_warmup_steps', 20000)
        self.buffer_size = getattr(args, 'dqn_buffer_size', 100000)
        
        # Epsilon greedy parameters
        self.epsilon_start = getattr(args, 'epsilon_start', 1.0)
        self.epsilon_end = getattr(args, 'epsilon_end', 0.05)
        self.epsilon_decay_steps = getattr(args, 'epsilon_decay_steps', 100000)
        
        # Replay buffer
        obs_shape = policy.obs_space.shape
        num_action_heads = len(policy.q_network.action_dims)
        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            obs_shape=obs_shape,
            num_action_heads=num_action_heads,
            device=device
        )
        
        # Training state
        self.total_env_steps = 0
        self.total_updates = 0
        
    def get_epsilon(self):
        """Get current epsilon value for epsilon-greedy exploration."""
        if self.total_env_steps < self.warmup_steps:
            return 1.0  # Full exploration during warmup
        
        progress = min(1.0, (self.total_env_steps - self.warmup_steps) / self.epsilon_decay_steps)
        epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
        return epsilon
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """
        Store a transition in replay buffer.
        
        Args:
            obs: observation
            action: action taken, shape (num_action_heads,)
            reward: reward received
            next_obs: next observation
            done: whether episode is done
        """
        self.replay_buffer.push(obs, action, reward, next_obs, done)
        self.total_env_steps += 1
    
    def should_update(self):
        """Check if it's time to update the network."""
        # Need warmup steps first
        if self.total_env_steps < self.warmup_steps:
            return False
        
        # Update every N steps
        return self.total_env_steps % self.update_interval == 0
    
    def should_update_target(self):
        """Check if it's time to update target network."""
        return self.total_env_steps % self.target_update_interval == 0
    
    def train(self):
        """
        Perform one training step.
        
        Returns:
            train_info: dict containing training statistics
        """
        if not self.should_update():
            return {}
        
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        obs = batch['obs']
        actions = batch['actions']  # (batch_size, num_heads)
        rewards = batch['rewards']  # (batch_size, 1)
        next_obs = batch['next_obs']
        dones = batch['dones']  # (batch_size, 1)
        
        # Compute current Q values for each head
        current_q_values = self.policy.q_network(obs)  # List of (batch_size, num_actions_i)
        
        # Gather Q values for taken actions
        current_q = []
        for i, q_vals in enumerate(current_q_values):
            action_idx = actions[:, i].unsqueeze(1)  # (batch_size, 1)
            q = q_vals.gather(1, action_idx)  # (batch_size, 1)
            current_q.append(q)
        current_q = torch.stack(current_q, dim=1).squeeze(-1)  # (batch_size, num_heads)
        
        # Compute target Q values using target network
        with torch.no_grad():
            next_q_values = self.policy.target_network(next_obs)  # List of (batch_size, num_actions_i)
            next_q_max = []
            for q_vals in next_q_values:
                q_max = q_vals.max(dim=1, keepdim=True)[0]  # (batch_size, 1)
                next_q_max.append(q_max)
            next_q_max = torch.stack(next_q_max, dim=1).squeeze(-1)  # (batch_size, num_heads)
            
            # Bellman target: r + gamma * max Q(s', a') * (1 - done)
            target_q = rewards + self.gamma * next_q_max * (1 - dones)  # (batch_size, num_heads)
        
        # Compute loss (MSE loss for each head)
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.policy.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if hasattr(self.args, 'max_grad_norm'):
            torch.nn.utils.clip_grad_norm_(
                self.policy.q_network.parameters(), 
                self.args.max_grad_norm
            )
        
        self.policy.optimizer.step()
        
        self.total_updates += 1
        
        # Update target network if needed
        if self.should_update_target():
            self.policy.sync_target_network()
        
        # Return training info
        train_info = {
            'loss': loss.item(),
            'q_mean': current_q.mean().item(),
            'q_max': current_q.max().item(),
            'epsilon': self.get_epsilon(),
            'buffer_size': len(self.replay_buffer),
            'total_updates': self.total_updates,
        }
        
        return train_info
