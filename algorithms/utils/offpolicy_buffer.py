import numpy as np
from .utils import get_shape_from_space


class OffPolicyBuffer:
    def __init__(self, args, num_agents, obs_space, act_space):
        self.buffer_size = args.td3_buffer_size
        self.n_rollout_threads = args.n_rollout_threads
        self.num_agents = num_agents
        self.obs_shape = get_shape_from_space(obs_space)
        self.act_shape = get_shape_from_space(act_space)

        self.obs = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, *self.act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def insert(self, obs, actions, rewards, next_obs, dones):
        obs = np.asarray(obs).reshape(-1, *self.obs_shape)
        actions = np.asarray(actions).reshape(-1, *self.act_shape)
        rewards = np.asarray(rewards).reshape(-1, 1)
        next_obs = np.asarray(next_obs).reshape(-1, *self.obs_shape)
        dones = np.asarray(dones).reshape(-1, 1)

        batch_size = obs.shape[0]
        for i in range(batch_size):
            self.obs[self.ptr] = obs[i]
            self.actions[self.ptr] = actions[i]
            self.rewards[self.ptr] = rewards[i]
            self.next_obs[self.ptr] = next_obs[i]
            self.dones[self.ptr] = dones[i]

            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError("Not enough samples in buffer to draw a batch.")
        indices = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            dones=self.dones[indices],
        )
