import time
import torch
import logging
import numpy as np
from typing import List
from .jsbsim_runner import JSBSimRunner
from algorithms.utils.buffer import ReplayBuffer
from algorithms.utils.offpolicy_buffer import OffPolicyBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class SelfplayJSBSimRunner(JSBSimRunner):

    def load(self):
        self.use_selfplay = self.all_args.use_selfplay 
        assert self.use_selfplay == True, "Only selfplay can use SelfplayRunner"
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.num_opponents = self.all_args.n_choose_opponents
        assert self.eval_episodes >= self.num_opponents, \
        f"Number of evaluation episodes:{self.eval_episodes} should be greater than number of opponents:{self.num_opponents}"
        self.init_elo = self.all_args.init_elo
        self.latest_elo = self.init_elo

        # policy & algorithm
        if self.algorithm_name == "ppo":
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
        elif self.algorithm_name == "td3":
            from algorithms.td3.td3_trainer import TD3Trainer as Trainer
            from algorithms.td3.td3_policy import TD3Policy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer
        if self.algorithm_name == "ppo":
            self.buffer = ReplayBuffer(self.all_args, self.num_agents // 2, self.obs_space, self.act_space)
        else:
            self.buffer = OffPolicyBuffer(self.all_args, self.num_agents // 2, self.obs_space, self.act_space)

        # [Selfplay] allocate memory for opponent policy/data in training
        from algorithms.utils.selfplay import get_algorithm
        self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)

        assert self.num_opponents <= self.n_rollout_threads, \
            "Number of different opponents({}) must less than or equal to number of training threads({})!" \
            .format(self.num_opponents, self.n_rollout_threads)
        self.policy_pool = {}  # type: dict[str, float]
        self.opponent_policy = [
            Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
            for _ in range(self.num_opponents)]
        self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
        if self.algorithm_name == "ppo":
            self.opponent_obs = np.zeros_like(self.buffer.obs[0])
            self.opponent_rnn_states = np.zeros_like(self.buffer.rnn_states_actor[0])
            self.opponent_masks = np.ones_like(self.buffer.masks[0])
        else:
            self.opponent_obs = None
            self.opponent_rnn_states = None
            self.opponent_masks = None

        if self.use_eval:
            self.eval_opponent_policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)

        logging.info("\n Load selfplay opponents: Algo {}, num_opponents {}.\n"
                        .format(self.all_args.selfplay_algorithm, self.num_opponents))

        if self.model_dir is not None:
            self.restore()

    def run(self):
        if self.algorithm_name == "td3":
            return self.run_td3_selfplay()
        return super().run()

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        # [Selfplay] divide ego/opponent of initial obs
        self.opponent_obs = obs[:, self.num_agents // 2:, ...]
        obs = obs[:, :self.num_agents // 2, ...]
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        # split parallel data [N*M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # [Selfplay] get actions of opponent policy
        opponent_actions = np.zeros_like(actions)
        for policy_idx, policy in enumerate(self.opponent_policy):
            env_idx = self.opponent_env_split[policy_idx]
            opponent_action, opponent_rnn_states \
                = policy.act(np.concatenate(self.opponent_obs[env_idx]),
                                np.concatenate(self.opponent_rnn_states[env_idx]),
                                np.concatenate(self.opponent_masks[env_idx]))
            opponent_actions[env_idx] = np.array(np.split(_t2n(opponent_action), len(env_idx)))
            self.opponent_rnn_states[env_idx] = np.array(np.split(_t2n(opponent_rnn_states), len(env_idx)))
        actions = np.concatenate((actions, opponent_actions), axis=1)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data: List[np.ndarray]):
        obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data

        dones_env = np.all(dones.squeeze(axis=-1), axis=-1)

        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # [Selfplay] divide ego/opponent of collecting data
        self.opponent_obs = obs[:, self.num_agents // 2:, ...]
        self.opponent_masks = masks[:, self.num_agents // 2:, ...]
        self.opponent_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        
        obs = obs[:, :self.num_agents // 2, ...]
        actions = actions[:, :self.num_agents // 2, ...]
        rewards = rewards[:, :self.num_agents // 2, ...]
        masks = masks[:, :self.num_agents // 2, ...]

        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)

    def run_td3_selfplay(self):
        obs = self.envs.reset()
        self.total_num_steps = 0
        start = time.time()
        episodes = 0
        ego_episode_rewards = np.zeros((self.n_rollout_threads, self.num_agents // 2, 1), dtype=np.float32)
        completed_rewards = []
        update_infos = []

        next_log = self.log_interval
        next_eval = self.eval_interval
        next_save = self.save_interval

        while self.total_num_steps < self.num_env_steps:
            ego_obs = obs[:, :self.num_agents // 2, ...]
            opponent_obs = obs[:, self.num_agents // 2:, ...]

            if self.total_num_steps < self.all_args.start_timesteps:
                ego_actions = np.array([self.act_space.sample() for _ in range(self.n_rollout_threads * (self.num_agents // 2))])
                ego_actions = ego_actions.reshape(self.n_rollout_threads, self.num_agents // 2, -1)
            else:
                self.policy.prep_rollout()
                flat_obs = np.concatenate(ego_obs)
                flat_actions = self.policy.act(flat_obs, deterministic=False, noise_std=self.all_args.explore_noise)
                ego_actions = np.array(np.split(_t2n(flat_actions), self.n_rollout_threads))

            opponent_actions = np.zeros_like(ego_actions)
            for policy_idx, policy in enumerate(self.opponent_policy):
                env_idx = self.opponent_env_split[policy_idx]
                flat_obs = np.concatenate(opponent_obs[env_idx])
                flat_actions = policy.act(flat_obs, deterministic=True)
                opponent_actions[env_idx] = np.array(np.split(_t2n(flat_actions), len(env_idx)))

            actions = np.concatenate((ego_actions, opponent_actions), axis=1)
            next_obs, rewards, dones, infos = self.envs.step(actions)

            ego_rewards = rewards[:, :self.num_agents // 2, ...]
            ego_next_obs = next_obs[:, :self.num_agents // 2, ...]
            ego_dones = dones[:, :self.num_agents // 2, ...]
            self.buffer.insert(ego_obs, ego_actions, ego_rewards, ego_next_obs, ego_dones)

            obs = next_obs
            self.total_num_steps += self.n_rollout_threads

            ego_episode_rewards += ego_rewards
            dones_env = np.all(dones.squeeze(axis=-1), axis=-1)
            if np.any(dones_env):
                completed_rewards.append(ego_episode_rewards[dones_env == True])
                ego_episode_rewards[dones_env == True] = 0
                episodes += dones_env.sum()

            if self.total_num_steps >= self.all_args.start_timesteps and self.buffer.size >= self.all_args.batch_size:
                for _ in range(self.all_args.updates_per_step):
                    update_info = self.trainer.update(self.policy, self.buffer)
                    update_infos.append(update_info)

            if episodes >= next_log and episodes > 0:
                end = time.time()
                logging.info("\n Scenario {} Algo {} Exp {} episodes {}, total num timesteps {}/{}, FPS {}.\n"
                             .format(self.all_args.scenario_name,
                                     self.algorithm_name,
                                     self.experiment_name,
                                     episodes,
                                     self.total_num_steps,
                                     self.num_env_steps,
                                     int(self.total_num_steps / (end - start))))

                train_infos = {}
                if update_infos:
                    for key in update_infos[0].keys():
                        train_infos[key] = float(np.mean([info[key] for info in update_infos]))
                    update_infos = []

                if completed_rewards:
                    rewards_arr = np.concatenate(completed_rewards, axis=0)
                    train_infos["average_episode_rewards"] = rewards_arr.mean()
                    completed_rewards = []
                    logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))

                self.log_info(train_infos, self.total_num_steps)
                next_log += self.log_interval

            if self.use_eval and episodes >= next_eval:
                if not self.policy_pool:
                    self.save(0)
                self.eval(self.total_num_steps)
                next_eval += self.eval_interval

            if episodes >= next_save:
                self.save(episodes)
                next_save += self.save_interval

    @torch.no_grad()
    def eval(self, total_num_steps):
        if self.algorithm_name == "td3":
            return self.eval_td3_selfplay(total_num_steps)
        logging.info("\nStart evaluation...")
        self.policy.prep_rollout()
        total_episodes = 0
        episode_rewards, opponent_episode_rewards = [], []
        cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)
        opponent_cumulative_rewards= np.zeros_like(cumulative_rewards)

        # [Selfplay] Choose opponent policy for evaluation
        eval_choose_opponents = [self.selfplay_algo.choose(self.policy_pool) for _ in range(self.num_opponents)]
        eval_each_episodes = self.eval_episodes // self.num_opponents
        logging.info(f" Choose opponents {eval_choose_opponents} for evaluation")

        eval_cur_opponent_idx = 0
        # use for tacview's timestamp
        self.timestamp = 0
        while total_episodes < self.eval_episodes:

            # [Selfplay] Load opponent policy
            if total_episodes >= eval_cur_opponent_idx * eval_each_episodes:
                policy_idx = eval_choose_opponents[eval_cur_opponent_idx]
                self.eval_opponent_policy.actor.load_state_dict(torch.load(str(self.save_dir) + f'/actor_{policy_idx}.pt', weights_only=True))
                self.eval_opponent_policy.prep_rollout()
                eval_cur_opponent_idx += 1
                logging.info(f" Load opponent {policy_idx} for evaluation ({total_episodes + 1}/{self.eval_episodes})")

                # reset obs/rnn/mask
                obs = self.eval_envs.reset()
                masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
                rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
                opponent_obs = obs[:, self.num_agents // 2:, ...]
                obs = obs[:, :self.num_agents // 2, ...]
                opponent_masks = np.ones_like(masks, dtype=np.float32)
                opponent_rnn_states = np.zeros_like(rnn_states, dtype=np.float32)

            # [Selfplay] get actions
            actions, rnn_states = self.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks), deterministic=True)
            actions = np.array(np.split(_t2n(actions), self.n_eval_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_eval_rollout_threads))

            opponent_actions, opponent_rnn_states \
                = self.eval_opponent_policy.act(np.concatenate(opponent_obs),
                                                np.concatenate(opponent_rnn_states),
                                                np.concatenate(opponent_masks), deterministic=True)
            opponent_rnn_states = np.array(np.split(_t2n(opponent_rnn_states), self.n_eval_rollout_threads))
            opponent_actions = np.array(np.split(_t2n(opponent_actions), self.n_eval_rollout_threads))
            actions = np.concatenate((actions, opponent_actions), axis=1)

            # Obser reward and next obs
            obs, eval_rewards, dones, eval_infos = self.eval_envs.step(actions)
            dones_env = np.all(dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(dones_env)

            # [Selfplay] Reset obs, masks, rnn_states
            opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            masks[dones_env == True] = np.zeros(((dones_env == True).sum(), *masks.shape[1:]), dtype=np.float32)
            rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states.shape[1:]), dtype=np.float32)

            opponent_masks[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), *opponent_masks.shape[1:]), dtype=np.float32)
            opponent_rnn_states[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), *opponent_rnn_states.shape[1:]), dtype=np.float32)

            # [Selfplay] Get rewards
            opponent_rewards = eval_rewards[:, self.num_agents//2:, ...]
            opponent_cumulative_rewards += opponent_rewards
            opponent_episode_rewards.append(opponent_cumulative_rewards[dones_env == True])
            opponent_cumulative_rewards[dones_env == True] = 0

            eval_rewards = eval_rewards[:, :self.num_agents // 2, ...]
            cumulative_rewards += eval_rewards
            episode_rewards.append(cumulative_rewards[dones_env == True])
            cumulative_rewards[dones_env == True] = 0
            
            # Tacview real time render
            if self.render_mode == "real_time":
                render_data = [f"#{self.timestamp:.2f}\n"]
                for sim in self.eval_envs.envs[0]._jsbsims.values():
                    log_msg = sim.log()
                    
                    if log_msg is not None:
                        render_data.append(log_msg + "\n")
                        
                render_data_str = "".join(render_data)
                self.tacview.send_data_to_client(render_data_str)
            self.timestamp += 0.2

        # Compute average episode rewards
        episode_rewards = np.concatenate(episode_rewards) # shape (self.eval_episodes, self.num_agents, 1)
        episode_rewards = episode_rewards.squeeze(-1).mean(axis=-1) # shape: (self.eval_episodes,)
        eval_average_episode_rewards = np.array(np.split(episode_rewards, self.num_opponents)).mean(axis=-1) # shape (self.num_opponents,)

        opponent_episode_rewards = np.concatenate(opponent_episode_rewards)
        opponent_episode_rewards = opponent_episode_rewards.squeeze(-1).mean(axis=-1)
        opponent_average_episode_rewards = np.array(np.split(opponent_episode_rewards, self.num_opponents)).mean(axis=-1)

        # Update elo
        '''
        Elo Rating Algorithm:
            Players with higher ELO ratings have a higher probability of winning a game than players with lower ELO ratings.
            After each game, the ELO rating of players is updated.
            If a player with a higher ELO rating wins, only a few points are transferred from the lower-rated player.
            However if the lower-rated player wins, then the transferred points from a higher-rated player are far greater.

        ego_elo / opponent_elo: Elo scores for each agent
        expected_score: Probability of winning of `ego` agent, i.e. when ego_elo >> opponent_elo, expected_score -> 1.
        actual_score: Outcome for current match, 1: ego win, 0: ego lose, 0.5: tie.

        Update Rule:
            Ra = Ra + K * (outcome - Pa)
            Rb = Rb + K * ((1 - outcome) - Pb) = Rb + K * ((1 - outcome) - (1 - Pa)) = Rb + K * (Pa - outcome)
        '''
        ego_elo = np.array([self.latest_elo for _ in range(self.n_eval_rollout_threads)])
        opponent_elo = np.array([self.policy_pool[key] for key in eval_choose_opponents])
        expected_score = 1 / (1 + 10 ** ((opponent_elo - ego_elo) / 400))

        actual_score = np.zeros_like(expected_score)
        diff = eval_average_episode_rewards - opponent_average_episode_rewards
        actual_score[diff > 100] = 1 # win
        actual_score[abs(diff) < 100] = 0.5 # tie
        actual_score[diff < -100] = 0 # lose

        K = 32
        update_opponent_elo = opponent_elo + K * (expected_score - actual_score)
        for i, key in enumerate(eval_choose_opponents):
            self.policy_pool[key] = update_opponent_elo[i]
        update_ego_elo = ego_elo + K * (actual_score - expected_score)
        self.latest_elo = update_ego_elo.mean()

        # Logging
        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = eval_average_episode_rewards.mean()
        eval_infos['eval_opponent_average_episode_rewards'] = opponent_average_episode_rewards.mean()
        eval_infos['eval_elo_gain'] = (K * (actual_score - expected_score)).mean()
        eval_infos['latest_elo'] = self.latest_elo
        logging.info(f" eval ego elo: {ego_elo.mean()} | opponent average elo: {opponent_elo.mean()}\n"
                     f" ego average episode rewards: {eval_infos['eval_average_episode_rewards']}\n"
                     f" opponent average episode rewards: {eval_infos['eval_opponent_average_episode_rewards']}\n"
                     f" elo gain: {(K * (actual_score - expected_score)).mean()} | latest elo score: {self.latest_elo}")
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")

        # [Selfplay] Reset opponent for the following training
        self.reset_opponent()

    @torch.no_grad()
    def eval_td3_selfplay(self, total_num_steps):
        logging.info("\nStart evaluation...")
        self.policy.prep_rollout()
        total_episodes = 0
        episode_rewards, opponent_episode_rewards = [], []
        cumulative_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents // 2, 1), dtype=np.float32)
        opponent_cumulative_rewards = np.zeros_like(cumulative_rewards)

        eval_choose_opponents = [self.selfplay_algo.choose(self.policy_pool) for _ in range(self.num_opponents)]
        eval_each_episodes = self.eval_episodes // self.num_opponents
        logging.info(f" Choose opponents {eval_choose_opponents} for evaluation")

        eval_cur_opponent_idx = 0
        self.timestamp = 0
        while total_episodes < self.eval_episodes:
            if total_episodes >= eval_cur_opponent_idx * eval_each_episodes:
                policy_idx = eval_choose_opponents[eval_cur_opponent_idx]
                self.eval_opponent_policy.actor.load_state_dict(torch.load(str(self.save_dir) + f'/actor_{policy_idx}.pt', weights_only=True))
                self.eval_opponent_policy.prep_rollout()
                eval_cur_opponent_idx += 1
                logging.info(f" Load opponent {policy_idx} for evaluation ({total_episodes + 1}/{self.eval_episodes})")

                obs = self.eval_envs.reset()
                opponent_obs = obs[:, self.num_agents // 2:, ...]
                obs = obs[:, :self.num_agents // 2, ...]

            actions = self.policy.act(np.concatenate(obs), deterministic=True)
            actions = np.array(np.split(_t2n(actions), self.n_eval_rollout_threads))

            opponent_actions = self.eval_opponent_policy.act(np.concatenate(opponent_obs), deterministic=True)
            opponent_actions = np.array(np.split(_t2n(opponent_actions), self.n_eval_rollout_threads))
            actions = np.concatenate((actions, opponent_actions), axis=1)

            obs, eval_rewards, dones, eval_infos = self.eval_envs.step(actions)
            dones_env = np.all(dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(dones_env)

            opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]

            opponent_rewards = eval_rewards[:, self.num_agents // 2:, ...]
            opponent_cumulative_rewards += opponent_rewards
            opponent_episode_rewards.append(opponent_cumulative_rewards[dones_env == True])
            opponent_cumulative_rewards[dones_env == True] = 0

            eval_rewards = eval_rewards[:, :self.num_agents // 2, ...]
            cumulative_rewards += eval_rewards
            episode_rewards.append(cumulative_rewards[dones_env == True])
            cumulative_rewards[dones_env == True] = 0

            if self.render_mode == "real_time":
                render_data = [f"#{self.timestamp:.2f}\n"]
                for sim in self.eval_envs.envs[0]._jsbsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        render_data.append(log_msg + "\n")
                render_data_str = "".join(render_data)
                self.tacview.send_data_to_client(render_data_str)
            self.timestamp += 0.2

        episode_rewards = np.concatenate(episode_rewards)
        episode_rewards = episode_rewards.squeeze(-1).mean(axis=-1)
        eval_average_episode_rewards = np.array(np.split(episode_rewards, self.num_opponents)).mean(axis=-1)

        opponent_episode_rewards = np.concatenate(opponent_episode_rewards)
        opponent_episode_rewards = opponent_episode_rewards.squeeze(-1).mean(axis=-1)
        opponent_average_episode_rewards = np.array(np.split(opponent_episode_rewards, self.num_opponents)).mean(axis=-1)

        ego_elo = np.array([self.latest_elo for _ in range(self.n_eval_rollout_threads)])
        opponent_elo = np.array([self.policy_pool[key] for key in eval_choose_opponents])
        expected_score = 1 / (1 + 10 ** ((opponent_elo - ego_elo) / 400))

        actual_score = np.zeros_like(expected_score)
        diff = eval_average_episode_rewards - opponent_average_episode_rewards
        actual_score[diff > 100] = 1
        actual_score[abs(diff) < 100] = 0.5
        actual_score[diff < -100] = 0

        K = 32
        update_opponent_elo = opponent_elo + K * (expected_score - actual_score)
        for i, key in enumerate(eval_choose_opponents):
            self.policy_pool[key] = update_opponent_elo[i]
        update_ego_elo = ego_elo + K * (actual_score - expected_score)
        self.latest_elo = update_ego_elo.mean()

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = eval_average_episode_rewards.mean()
        eval_infos['eval_opponent_average_episode_rewards'] = opponent_average_episode_rewards.mean()
        eval_infos['eval_elo_gain'] = (K * (actual_score - expected_score)).mean()
        eval_infos['latest_elo'] = self.latest_elo
        logging.info(f" eval ego elo: {ego_elo.mean()} | opponent average elo: {opponent_elo.mean()}\n"
                     f" ego average episode rewards: {eval_infos['eval_average_episode_rewards']}\n"
                     f" opponent average episode rewards: {eval_infos['eval_opponent_average_episode_rewards']}\n"
                     f" elo gain: {(K * (actual_score - expected_score)).mean()} | latest elo score: {self.latest_elo}")
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")

        self.reset_opponent()

    def save(self, episode):
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
        if self.algorithm_name == "ppo":
            policy_critic_state_dict = self.policy.critic.state_dict()
            torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
        else:
            torch.save(self.policy.critic.state_dict(), str(self.save_dir) + '/critic_latest.pt')
        torch.save(policy_actor_state_dict, str(self.save_dir) + f'/actor_{episode}.pt')
        self.policy_pool[str(episode)] = self.latest_elo

    def reset_opponent(self):
        choose_opponents = []
        for policy in self.opponent_policy:
            choose_idx = self.selfplay_algo.choose(self.policy_pool)
            choose_opponents.append(choose_idx)
            policy.actor.load_state_dict(torch.load(str(self.save_dir) + f'/actor_{choose_idx}.pt', weights_only=True))
            policy.prep_rollout()
        logging.info(f" Choose opponents {choose_opponents} for training")

        # clear buffer
        if self.algorithm_name == "ppo":
            self.buffer.clear()
            self.opponent_obs = np.zeros_like(self.opponent_obs)
            self.opponent_rnn_states = np.zeros_like(self.opponent_rnn_states)
            self.opponent_masks = np.ones_like(self.opponent_masks)
        else:
            self.buffer.clear()

        # reset env
        obs = self.envs.reset()
        if self.num_opponents > 0:
            if self.algorithm_name == "ppo":
                self.opponent_obs = obs[:, self.num_agents // 2:, ...]
                obs = obs[:, :self.num_agents // 2, ...]
        if self.algorithm_name == "ppo":
            self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def render(self):
        if self.algorithm_name == "td3":
            return self.render_td3_selfplay()
        idx = self.all_args.render_index
        opponent_idx = self.all_args.render_opponent_index
        dir_list = str(self.run_dir).split('/')
        file_path = '/'.join(dir_list[:dir_list.index('results')+1])
        self.policy.actor.load_state_dict(torch.load(str(self.model_dir)+ f'/actor_{idx}.pt'))
        self.policy.prep_rollout()
        self.eval_opponent_policy.actor.load_state_dict(torch.load(str(self.model_dir) + f'/actor_{opponent_idx}.pt',weights_only=True))
        self.eval_opponent_policy.prep_rollout()
        logging.info("\nStart render ...")
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        self.envs.render(mode='txt', filepath=f'{file_path}/{self.experiment_name}.txt.acmi')
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
        render_obs = render_obs[:, :self.num_agents // 2, ...]
        render_opponent_masks = np.ones_like(render_masks, dtype=np.float32)
        render_opponent_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
            render_opponent_actions, render_opponent_rnn_states \
                = self.eval_opponent_policy.act(np.concatenate(render_opponent_obs),
                                                np.concatenate(render_opponent_rnn_states),
                                                np.concatenate(render_opponent_masks),
                                                deterministic=True)
            render_opponent_actions = np.expand_dims(_t2n(render_opponent_actions), axis=0)
            render_opponent_rnn_states = np.expand_dims(_t2n(render_opponent_rnn_states), axis=0)
            render_actions = np.concatenate((render_actions, render_opponent_actions), axis=1)
            # Obser reward and next obs
            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            render_rewards = render_rewards[:, :self.num_agents // 2, ...]
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{file_path}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
            render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
            render_obs = render_obs[:, :self.num_agents // 2, ...]
        print(render_episode_rewards)

    @torch.no_grad()
    def render_td3_selfplay(self):
        idx = self.all_args.render_index
        opponent_idx = self.all_args.render_opponent_index
        dir_list = str(self.run_dir).split('/')
        file_path = '/'.join(dir_list[:dir_list.index('results')+1])
        self.policy.actor.load_state_dict(torch.load(str(self.model_dir) + f'/actor_{idx}.pt', weights_only=True))
        self.policy.prep_rollout()
        self.eval_opponent_policy.actor.load_state_dict(torch.load(str(self.model_dir) + f'/actor_{opponent_idx}.pt', weights_only=True))
        self.eval_opponent_policy.prep_rollout()
        logging.info("\nStart render ...")
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        self.envs.render(mode='txt', filepath=f'{file_path}/{self.experiment_name}.txt.acmi')
        render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
        render_obs = render_obs[:, :self.num_agents // 2, ...]
        while True:
            self.policy.prep_rollout()
            render_actions = self.policy.act(np.concatenate(render_obs), deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_opponent_actions = self.eval_opponent_policy.act(np.concatenate(render_opponent_obs), deterministic=True)
            render_opponent_actions = np.expand_dims(_t2n(render_opponent_actions), axis=0)
            render_actions = np.concatenate((render_actions, render_opponent_actions), axis=1)
            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            render_rewards = render_rewards[:, :self.num_agents // 2, ...]
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{file_path}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
            render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
            render_obs = render_obs[:, :self.num_agents // 2, ...]
        print(render_episode_rewards)
