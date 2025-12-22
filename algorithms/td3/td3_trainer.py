import torch
import torch.nn as nn


class TD3Trainer:
    def __init__(self, args, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_delay = args.policy_delay
        self.target_noise = args.target_noise
        self.noise_clip = args.noise_clip
        self.batch_size = args.batch_size
        self.updates_per_step = args.updates_per_step
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm

        self.total_it = 0

    def update(self, policy, buffer):
        self.total_it += 1
        batch = buffer.sample(self.batch_size)

        obs = torch.as_tensor(batch["obs"], **self.tpdv)
        actions = torch.as_tensor(batch["actions"], **self.tpdv)
        rewards = torch.as_tensor(batch["rewards"], **self.tpdv)
        next_obs = torch.as_tensor(batch["next_obs"], **self.tpdv)
        dones = torch.as_tensor(batch["dones"], **self.tpdv)

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.target_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = policy.actor_target(next_obs)
            next_actions = next_actions + noise
            next_actions = torch.max(torch.min(next_actions, policy.actor.action_high), policy.actor.action_low)

            target_q1, target_q2 = policy.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1, current_q2 = policy.critic(obs, actions)
        critic_loss = nn.MSELoss()(current_q1, target) + nn.MSELoss()(current_q2, target)

        policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_max_grad_norm:
            nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm)
        policy.critic_optimizer.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.total_it % self.policy_delay == 0:
            actor_loss = -policy.critic.q1(obs, policy.actor(obs)).mean()
            policy.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_max_grad_norm:
                nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm)
            policy.actor_optimizer.step()
            policy.soft_update(self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q1_mean": current_q1.mean().item(),
            "q2_mean": current_q2.mean().item(),
        }
