"""
Experience Replay Buffer for DQN
"""
import numpy as np
import torch


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    Stores transitions and supports sampling for training.
    """
    
    def __init__(self, capacity, obs_shape, num_action_heads, device=torch.device("cpu")):
        """
        Args:
            capacity: maximum number of transitions to store
            obs_shape: shape of observations (e.g., (21,))
            num_action_heads: number of action heads (e.g., 5 for [41,41,41,30,2])
            device: torch device
        """
        self.capacity = capacity
        self.device = device
        self.num_action_heads = num_action_heads
        
        # Preallocate memory
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, num_action_heads), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.position = 0
        self.size = 0
    
    def push(self, obs, action, reward, next_obs, done):
        """
        Add a transition to the buffer.
        
        Args:
            obs: observation, shape (obs_dim,)
            action: action, shape (num_action_heads,)
            reward: reward, scalar
            next_obs: next observation, shape (obs_dim,)
            done: whether episode is done, bool
        """
        self.obs[self.position] = obs
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_obs[self.position] = next_obs
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: number of transitions to sample
        
        Returns:
            batch: dict with keys 'obs', 'actions', 'rewards', 'next_obs', 'dones'
                   all values are torch tensors on self.device
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch = {
            'obs': torch.from_numpy(self.obs[indices]).to(self.device),
            'actions': torch.from_numpy(self.actions[indices]).to(self.device),
            'rewards': torch.from_numpy(self.rewards[indices]).to(self.device),
            'next_obs': torch.from_numpy(self.next_obs[indices]).to(self.device),
            'dones': torch.from_numpy(self.dones[indices]).to(self.device),
        }
        
        return batch
    
    def __len__(self):
        return self.size
    
    def is_ready(self, min_size):
        """Check if buffer has enough samples."""
        return self.size >= min_size
