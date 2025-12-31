"""
Dueling DQN Network with Multi-Head Action Decomposition
"""
import torch
import torch.nn as nn
import numpy as np
from ..utils.mlp import MLPBase
from ..utils.utils import check


class DuelingDQN(nn.Module):
    """
    Dueling DQN Network with multi-head decomposition for complex action spaces.
    
    For action space like Tuple([MultiDiscrete([41, 41, 41, 30]), Discrete(2)]):
    - Creates separate dueling heads for each action dimension
    - Each head has its own value stream and advantage stream
    """
    
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(DuelingDQN, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # Parse hidden size string to get dueling head dimension
        # If hidden_size is "256 256", use last dimension (256)
        if isinstance(self.hidden_size, str):
            hidden_dims = list(map(int, self.hidden_size.split(' ')))
            self.dueling_hidden_dim = hidden_dims[-1]
        else:
            self.dueling_hidden_dim = self.hidden_size
        
        # Parse action space
        self.act_space = act_space
        self.action_dims = self._parse_action_space(act_space)
        self.num_heads = len(self.action_dims)
        
        # Shared feature extraction
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, 
                           self.use_feature_normalization)
        
        # Create dueling heads for each action dimension
        self.value_heads = nn.ModuleList()
        self.advantage_heads = nn.ModuleList()
        
        for num_actions in self.action_dims:
            # Value stream: outputs single value
            value_head = nn.Sequential(
                nn.Linear(self.base.output_size, self.dueling_hidden_dim),
                self._get_activation(),
                nn.Linear(self.dueling_hidden_dim, 1)
            )
            self.value_heads.append(value_head)
            
            # Advantage stream: outputs advantage for each action
            advantage_head = nn.Sequential(
                nn.Linear(self.base.output_size, self.dueling_hidden_dim),
                self._get_activation(),
                nn.Linear(self.dueling_hidden_dim, num_actions)
            )
            self.advantage_heads.append(advantage_head)
        
        self.to(device)
    
    def _parse_action_space(self, act_space):
        """Parse action space to get dimensions for each head."""
        from gymnasium import spaces
        
        action_dims = []
        if isinstance(act_space, spaces.Tuple):
            for sub_space in act_space.spaces:
                if isinstance(sub_space, spaces.MultiDiscrete):
                    action_dims.extend(sub_space.nvec.tolist())
                elif isinstance(sub_space, spaces.Discrete):
                    action_dims.append(sub_space.n)
        elif isinstance(act_space, spaces.MultiDiscrete):
            action_dims = act_space.nvec.tolist()
        elif isinstance(act_space, spaces.Discrete):
            action_dims = [act_space.n]
        else:
            raise NotImplementedError(f"Unsupported action space: {type(act_space)}")
        
        return action_dims
    
    def _get_activation(self):
        """Get activation function."""
        if self.activation_id == 1:
            return nn.Tanh()
        elif self.activation_id == 2:
            return nn.ReLU()
        elif self.activation_id == 3:
            return nn.LeakyReLU()
        else:
            return nn.ReLU()
    
    def forward(self, obs):
        """
        Forward pass to compute Q-values for all action heads.
        
        Args:
            obs: observations, shape (batch_size, obs_dim)
        
        Returns:
            q_values: list of Q-values for each head, each with shape (batch_size, num_actions_i)
        """
        obs = check(obs).to(**self.tpdv)
        
        # Extract features
        features = self.base(obs)
        
        # Compute Q-values for each head using Dueling architecture
        q_values = []
        for i in range(self.num_heads):
            value = self.value_heads[i](features)  # (batch_size, 1)
            advantage = self.advantage_heads[i](features)  # (batch_size, num_actions_i)
            
            # Dueling DQN: Q = V + (A - mean(A))
            q = value + (advantage - advantage.mean(dim=1, keepdim=True))
            q_values.append(q)
        
        return q_values
    
    def get_action(self, obs, epsilon=0.0):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            obs: observations, shape (batch_size, obs_dim)
            epsilon: exploration rate
        
        Returns:
            actions: selected actions for each head, shape (batch_size, num_heads)
        """
        batch_size = obs.shape[0]
        actions = []
        
        if np.random.rand() < epsilon:
            # Random action
            for num_actions in self.action_dims:
                actions.append(np.random.randint(0, num_actions, size=batch_size))
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(obs)
                for q in q_values:
                    actions.append(q.argmax(dim=1).cpu().numpy())
        
        # Stack actions: (batch_size, num_heads)
        actions = np.stack(actions, axis=1)
        return actions
