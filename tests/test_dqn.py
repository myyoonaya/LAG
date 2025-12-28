"""
Quick test script for DQN implementation
"""
import sys
import os
import torch
import numpy as np
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Mock args for testing
class Args:
    def __init__(self):
        # Network config
        self.hidden_size = [256, 256]
        self.activation_id = 1
        self.use_feature_normalization = False
        
        # DQN config
        self.dqn_batch_size = 32
        self.dqn_update_interval = 4
        self.dqn_target_update_interval = 10000
        self.dqn_warmup_steps = 20000
        self.dqn_buffer_size = 100000
        
        # Epsilon config
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 100000
        
        # Training config
        self.gamma = 0.99
        self.lr = 0.0001
        self.max_grad_norm = 2.0


def test_dqn_network():
    """Test DQN network forward pass."""
    print("=" * 50)
    print("Testing DQN Network...")
    print("=" * 50)
    
    from algorithms.dqn.dqn_network import DuelingDQN
    
    args = Args()
    obs_space = spaces.Box(low=-10, high=10, shape=(21,), dtype=np.float32)
    # Action space: [41, 41, 41, 30, 2] (飞控动作 + 射击)
    act_space = spaces.Tuple([
        spaces.MultiDiscrete([41, 41, 41, 30]),
        spaces.Discrete(2)
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create network
    network = DuelingDQN(args, obs_space, act_space, device)
    print(f"Network created successfully")
    print(f"Number of action heads: {network.num_heads}")
    print(f"Action dimensions: {network.action_dims}")
    
    # Test forward pass
    batch_size = 4
    obs = np.random.randn(batch_size, 21).astype(np.float32)
    q_values = network(obs)
    
    print(f"\nForward pass test:")
    print(f"Input shape: {obs.shape}")
    for i, q in enumerate(q_values):
        print(f"Q-values head {i} shape: {q.shape}")
    
    # Test action selection
    actions = network.get_action(obs, epsilon=0.1)
    print(f"\nAction selection test:")
    print(f"Actions shape: {actions.shape}")
    print(f"Actions: {actions}")
    
    print("\n✓ DQN Network test passed!")


def test_replay_buffer():
    """Test replay buffer."""
    print("\n" + "=" * 50)
    print("Testing Replay Buffer...")
    print("=" * 50)
    
    from algorithms.dqn.replay_buffer import ReplayBuffer
    
    device = torch.device("cpu")
    buffer = ReplayBuffer(
        capacity=1000,
        obs_shape=(21,),
        num_action_heads=5,
        device=device
    )
    
    # Add some transitions
    for i in range(100):
        obs = np.random.randn(21).astype(np.float32)
        action = np.random.randint(0, 10, size=5).astype(np.int64)
        reward = np.random.randn()
        next_obs = np.random.randn(21).astype(np.float32)
        done = i % 20 == 19
        
        buffer.push(obs, action, reward, next_obs, done)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Sample a batch
    batch = buffer.sample(32)
    print(f"\nSampled batch:")
    print(f"obs shape: {batch['obs'].shape}")
    print(f"actions shape: {batch['actions'].shape}")
    print(f"rewards shape: {batch['rewards'].shape}")
    print(f"next_obs shape: {batch['next_obs'].shape}")
    print(f"dones shape: {batch['dones'].shape}")
    
    print("\n✓ Replay Buffer test passed!")


def test_dqn_policy():
    """Test DQN policy."""
    print("\n" + "=" * 50)
    print("Testing DQN Policy...")
    print("=" * 50)
    
    from algorithms.dqn.dqn_policy import DQNPolicy
    
    args = Args()
    obs_space = spaces.Box(low=-10, high=10, shape=(21,), dtype=np.float32)
    act_space = spaces.Tuple([
        spaces.MultiDiscrete([41, 41, 41, 30]),
        spaces.Discrete(2)
    ])
    
    device = torch.device("cpu")
    policy = DQNPolicy(args, obs_space, act_space, device)
    
    print("Policy created successfully")
    
    # Test action selection
    obs = np.random.randn(4, 21).astype(np.float32)
    actions = policy.get_actions(obs, epsilon=0.5)
    print(f"\nActions shape: {actions.shape}")
    print(f"Actions: {actions}")
    
    # Test target network sync
    policy.sync_target_network()
    print("\nTarget network synchronized")
    
    print("\n✓ DQN Policy test passed!")


def test_dqn_trainer():
    """Test DQN trainer."""
    print("\n" + "=" * 50)
    print("Testing DQN Trainer...")
    print("=" * 50)
    
    from algorithms.dqn.dqn_policy import DQNPolicy
    from algorithms.dqn.dqn_trainer import DQNTrainer
    
    args = Args()
    obs_space = spaces.Box(low=-10, high=10, shape=(21,), dtype=np.float32)
    act_space = spaces.Tuple([
        spaces.MultiDiscrete([41, 41, 41, 30]),
        spaces.Discrete(2)
    ])
    
    device = torch.device("cpu")
    policy = DQNPolicy(args, obs_space, act_space, device)
    trainer = DQNTrainer(args, policy, device)
    
    print("Trainer created successfully")
    print(f"Warmup steps: {trainer.warmup_steps}")
    print(f"Update interval: {trainer.update_interval}")
    print(f"Target update interval: {trainer.target_update_interval}")
    
    # Store some transitions
    for i in range(100):
        obs = np.random.randn(21).astype(np.float32)
        action = np.random.randint(0, 10, size=5).astype(np.int64)
        reward = np.random.randn()
        next_obs = np.random.randn(21).astype(np.float32)
        done = False
        
        trainer.store_transition(obs, action, reward, next_obs, done)
    
    print(f"\nStored 100 transitions")
    print(f"Total env steps: {trainer.total_env_steps}")
    print(f"Buffer size: {len(trainer.replay_buffer)}")
    print(f"Current epsilon: {trainer.get_epsilon():.3f}")
    
    # Test training (won't actually train due to warmup)
    train_info = trainer.train()
    print(f"\nTrain info (during warmup): {train_info}")
    
    # Skip warmup for testing
    trainer.total_env_steps = 25000
    for i in range(1000):
        obs = np.random.randn(21).astype(np.float32)
        action = np.random.randint(0, 10, size=5).astype(np.int64)
        reward = np.random.randn()
        next_obs = np.random.randn(21).astype(np.float32)
        done = False
        trainer.store_transition(obs, action, reward, next_obs, done)
    
    # Now try training
    train_info = trainer.train()
    if train_info:
        print(f"\nTrain info (after warmup):")
        for k, v in train_info.items():
            print(f"  {k}: {v}")
    
    print("\n✓ DQN Trainer test passed!")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("DQN Implementation Test Suite")
    print("=" * 50)
    
    try:
        test_dqn_network()
        test_replay_buffer()
        test_dqn_policy()
        test_dqn_trainer()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
