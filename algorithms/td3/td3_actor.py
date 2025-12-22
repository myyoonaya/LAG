import gymnasium as gym
import torch
import torch.nn as nn

from ..utils.mlp import MLPBase, MLPLayer
from ..utils.utils import check


class TD3Actor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(TD3Actor, self).__init__()
        if not isinstance(act_space, gym.spaces.Box):
            raise NotImplementedError("TD3 only supports continuous (Box) action spaces.")

        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        input_size = self.base.output_size

        self._use_mlp = bool(self.act_hidden_size.strip())
        if self._use_mlp:
            self.mlp = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
            input_size = self.mlp.output_size

        action_dim = act_space.shape[0]
        self.action_out = nn.Linear(input_size, action_dim)

        action_low = torch.as_tensor(act_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(act_space.high, dtype=torch.float32)
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

        self.to(device)

    def forward(self, obs):
        obs = check(obs).to(**self.tpdv)
        x = self.base(obs)
        if self._use_mlp:
            x = self.mlp(x)
        action = torch.tanh(self.action_out(x))
        return action * self.action_scale + self.action_bias
