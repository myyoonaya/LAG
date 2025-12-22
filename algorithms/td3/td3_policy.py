import torch

from .td3_actor import TD3Actor
from .td3_critic import TD3Critic


class TD3Policy:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device

        self.actor = TD3Actor(args, obs_space, act_space, device)
        self.actor_target = TD3Actor(args, obs_space, act_space, device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3Critic(args, obs_space, act_space, device)
        self.critic_target = TD3Critic(args, obs_space, act_space, device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

    def act(self, obs, deterministic=False, noise_std=0.0):
        actions = self.actor(obs)
        if not deterministic and noise_std > 0:
            noise = torch.randn_like(actions) * noise_std
            actions = actions + noise
        actions = torch.max(torch.min(actions, self.actor.action_high), self.actor.action_low)
        return actions

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def soft_update(self, tau):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
