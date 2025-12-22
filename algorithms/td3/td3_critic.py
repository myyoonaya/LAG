import gymnasium as gym
import torch
import torch.nn as nn

from ..utils.mlp import MLPBase, MLPLayer
from ..utils.utils import check


class TD3Critic(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(TD3Critic, self).__init__()
        if not isinstance(act_space, gym.spaces.Box):
            raise NotImplementedError("TD3 only supports continuous (Box) action spaces.")

        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)

        action_dim = act_space.shape[0]

        self.base1 = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        self.base2 = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)

        self._use_mlp = bool(self.act_hidden_size.strip())
        input_dim1 = self.base1.output_size + action_dim
        input_dim2 = self.base2.output_size + action_dim
        if self._use_mlp:
            self.mlp1 = MLPLayer(input_dim1, self.act_hidden_size, self.activation_id)
            self.mlp2 = MLPLayer(input_dim2, self.act_hidden_size, self.activation_id)
            input_dim1 = self.mlp1.output_size
            input_dim2 = self.mlp2.output_size
        self.q1_out = nn.Linear(input_dim1, 1)
        self.q2_out = nn.Linear(input_dim2, 1)

        self.to(device)

    def forward(self, obs, actions):
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)

        x1 = self.base1(obs)
        x2 = self.base2(obs)
        x1 = torch.cat([x1, actions], dim=-1)
        x2 = torch.cat([x2, actions], dim=-1)
        if self._use_mlp:
            x1 = self.mlp1(x1)
            x2 = self.mlp2(x2)
        q1 = self.q1_out(x1)
        q2 = self.q2_out(x2)
        return q1, q2

    def q1(self, obs, actions):
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)

        x1 = self.base1(obs)
        x1 = torch.cat([x1, actions], dim=-1)
        if self._use_mlp:
            x1 = self.mlp1(x1)
        return self.q1_out(x1)
