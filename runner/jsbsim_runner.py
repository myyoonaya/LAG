import time
import torch
import logging
import numpy as np
from typing import List
from .base_runner import Runner
from algorithms.utils.buffer import ReplayBuffer
from algorithms.utils.offpolicy_buffer import OffPolicyBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class JSBSimRunner(Runner):

    def load(self):
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.use_selfplay = self.all_args.use_selfplay

        # policy & algorithm
        if self.algorithm_name == "ppo":
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
        elif self.algorithm_name == "td3":
            if self.use_selfplay:
                raise NotImplementedError("TD3 does not support selfplay in this runner.")
            if self.all_args.env_name != "SingleCombat":
                raise NotImplementedError("TD3 is restricted to SingleCombat in this setup.")
            from algorithms.td3.td3_trainer import TD3Trainer as Trainer
            from algorithms.td3.td3_policy import TD3Policy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer
        if self.algorithm_name == "ppo":
            self.buffer = ReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.act_space)
        else:
            self.buffer = OffPolicyBuffer(self.all_args, self.num_agents, self.obs_space, self.act_space)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        if self.algorithm_name == "ppo":
            self.warmup()
            return self.run_ppo()
        return self.run_td3()

    def run_ppo(self):
        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads

        for episode in range(episodes):

            heading_turns_list = []

            for step in range(self.buffer_size):
                # Sample actions
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)

                # Extra recorded information
                for info in infos:
                    if 'heading_turn_counts' in info:
                        heading_turns_list.append(info['heading_turn_counts'])

                data = obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                logging.info("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                             .format(self.all_args.scenario_name,
                                     self.algorithm_name,
                                     self.experiment_name,
                                     episode,
                                     episodes,
                                     self.total_num_steps,
                                     self.num_env_steps,
                                     int(self.total_num_steps / (end - start))))

                train_infos["average_episode_rewards"] = self.buffer.rewards.sum() / (self.buffer.masks == False).sum()
                logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))

                if len(heading_turns_list):
                    train_infos["average_heading_turns"] = np.mean(heading_turns_list)
                    logging.info("average heading turns is {}".format(train_infos["average_heading_turns"]))
                self.log_info(train_infos, self.total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and episode != 0 and self.use_eval:
                    self.eval(self.total_num_steps)

            # save model
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

    def run_td3(self):
        obs = self.envs.reset()
        self.total_num_steps = 0
        start = time.time()
        episodes = 0
        episode_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        completed_rewards = []
        update_infos = []

        next_log = self.log_interval
        next_eval = self.eval_interval
        next_save = self.save_interval

        while self.total_num_steps < self.num_env_steps:
            if self.total_num_steps < self.all_args.start_timesteps:
                actions = np.array([self.act_space.sample() for _ in range(self.n_rollout_threads * self.num_agents)])
                actions = actions.reshape(self.n_rollout_threads, self.num_agents, -1)
            else:
                self.policy.prep_rollout()
                flat_obs = np.concatenate(obs)
                flat_actions = self.policy.act(flat_obs, deterministic=False, noise_std=self.all_args.explore_noise)
                actions = np.array(np.split(_t2n(flat_actions), self.n_rollout_threads))

            next_obs, rewards, dones, infos = self.envs.step(actions)
            self.buffer.insert(obs, actions, rewards, next_obs, dones)
            obs = next_obs
            self.total_num_steps += self.n_rollout_threads

            episode_rewards += rewards
            dones_env = np.all(dones.squeeze(axis=-1), axis=-1)
            if np.any(dones_env):
                completed_rewards.append(episode_rewards[dones_env == True])
                episode_rewards[dones_env == True] = 0
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
                self.eval(self.total_num_steps)
                next_eval += self.eval_interval

            if episodes >= next_save:
                self.save(episodes)
                next_save += self.save_interval

    def warmup(self):
        # reset env
        obs = self.envs.reset()
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
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data: List[np.ndarray]):
        obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data

        dones_env = np.all(dones.squeeze(axis=-1), axis=-1)

        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)

    @torch.no_grad()
    def eval(self, total_num_steps):
        if self.algorithm_name == "td3":
            return self.eval_td3(total_num_steps)
        logging.info("\nStart evaluation...")
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        eval_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)

        self.timestamp = 0 # use for tacview real time render 
        if self.render_mode == "real_time" and self.tacview: #reconnect tacview to clear the telemetry
            print("reconnect tacview.....")
            self.tacview.reconnect()
        
        while total_episodes < self.eval_episodes:

            self.policy.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_obs),
                                                            np.concatenate(eval_rnn_states),
                                                            np.concatenate(eval_masks), deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)

            # real render with tacview
            if self.render_mode == "real_time" and self.tacview:
                render_data = [f"#{self.timestamp:.2f}\n"]
                for sim in self.eval_envs.envs[0]._jsbsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        render_data.append(log_msg + "\n")
                for sim in self.eval_envs.envs[0]._tempsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        render_data.append(log_msg + "\n")
                render_data_str = "".join(render_data)
                try:
                    self.tacview.send_data_to_client(render_data_str)
                except Exception as e:
                    logging.error(f"Tacview rendering error: {e}")
            self.timestamp += 0.2   # step 0.2s
            
            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_dones_env)
            eval_episode_rewards.append(eval_cumulative_rewards[eval_dones_env == True])
            eval_cumulative_rewards[eval_dones_env == True] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]), dtype=np.float32)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean(axis=1)  # shape: [num_agents, 1]
        logging.info(" eval average episode rewards: " + str(np.mean(eval_infos['eval_average_episode_rewards'])))
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")

    @torch.no_grad()
    def eval_td3(self, total_num_steps):
        logging.info("\nStart evaluation...")
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_obs = self.eval_envs.reset()

        while total_episodes < self.eval_episodes:
            self.policy.prep_rollout()
            flat_actions = self.policy.act(np.concatenate(eval_obs), deterministic=True)
            eval_actions = np.array(np.split(_t2n(flat_actions), self.n_eval_rollout_threads))

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_dones_env)
            eval_episode_rewards.append(eval_cumulative_rewards[eval_dones_env == True])
            eval_cumulative_rewards[eval_dones_env == True] = 0

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean(axis=1)
        logging.info(" eval average episode rewards: " + str(np.mean(eval_infos['eval_average_episode_rewards'])))
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")

    @torch.no_grad()
    def render(self):
        if self.algorithm_name == "td3":
            return self.render_td3()
        logging.info("\nStart render, render mode is {self.render_mode} ... ...")
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        self.envs.render(mode=self.render_mode, filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi',tacview=self.tacview)
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)

            # Obser reward and next obs
            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            if self.use_selfplay:
                render_rewards = render_rewards[:, :self.num_agents // 2, ...]
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))

    def save(self, episode):
        if self.algorithm_name == "ppo":
            policy_actor_state_dict = self.policy.actor.state_dict()
            torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
            policy_critic_state_dict = self.policy.critic.state_dict()
            torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
        else:
            torch.save(self.policy.actor.state_dict(), str(self.save_dir) + '/actor_latest.pt')
            torch.save(self.policy.critic.state_dict(), str(self.save_dir) + '/critic_latest.pt')

    def restore(self):
        if self.algorithm_name == "ppo":
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_latest.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_latest.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
        else:
            actor_state_dict = torch.load(str(self.model_dir) + '/actor_latest.pt')
            self.policy.actor.load_state_dict(actor_state_dict)
            critic_state_dict = torch.load(str(self.model_dir) + '/critic_latest.pt')
            self.policy.critic.load_state_dict(critic_state_dict)

    @torch.no_grad()
    def render_td3(self):
        logging.info("\nStart render, render mode is {self.render_mode} ... ...")
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        self.envs.render(mode=self.render_mode, filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi', tacview=self.tacview)
        while True:
            self.policy.prep_rollout()
            flat_actions = self.policy.act(np.concatenate(render_obs), deterministic=True)
            render_actions = np.expand_dims(_t2n(flat_actions), axis=0)

            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))
