"""
DQN Policy
"""
import torch
from .dqn_network import DuelingDQN


class DQNPolicy:
    """
    DQN Policy with target network for stable learning.
    """
    
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        
        self.obs_space = obs_space
        self.act_space = act_space
        
        # Q-network and target network
        self.q_network = DuelingDQN(args, obs_space, act_space, device)
        self.target_network = DuelingDQN(args, obs_space, act_space, device)
        
        # Initialize target network with same weights
        self.sync_target_network()
        
        # Optimizer
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
    
    def get_actions(self, obs, epsilon=0.0):
        """
        Select actions using epsilon-greedy policy.
        
        Args:
            obs: observations, numpy array or tensor
            epsilon: exploration rate
        
        Returns:
            actions: selected actions, numpy array
        """
        return self.q_network.get_action(obs, epsilon)
    
    def sync_target_network(self):
        """Synchronize target network with Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def prep_training(self):
        """Set networks to training mode."""
        self.q_network.train()
        self.target_network.eval()
    
    def prep_rollout(self):
        """Set networks to evaluation mode."""
        self.q_network.eval()
        self.target_network.eval()
    
    def save(self, save_path):
        """Save model parameters."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, save_path)
    
    def load(self, load_path):
        """Load model parameters."""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
