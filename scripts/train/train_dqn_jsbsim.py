#!/usr/bin/env python
"""
Training script for DQN on JSBSim SingleCombat environment with self-play
"""
import sys
import os
import traceback
import wandb
import socket
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from runner.dqn_selfplay_jsbsim_runner import DQNSelfplayRunner
from envs.JSBSim.envs import SingleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_train_env(all_args):
    """Create training environments."""
    def get_env_fn(rank):
        def init_env():
            env = SingleCombatEnv(all_args.scenario_name)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    """Create evaluation environments."""
    def get_env_fn(rank):
        def init_env():
            env = SingleCombatEnv(all_args.scenario_name)
            env.seed(all_args.seed * 50000 + rank * 1000)
            return env
        return init_env
    
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    """Parse JSBSim specific arguments."""
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--scenario-name', type=str, 
                      default='1v1/ShootMissile/Selfplay',
                      help="Which scenario to run on")
    group.add_argument('--render-mode', type=str, default='txt',
                      help="txt or real_time")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    """Main training function."""
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # Set algorithm name
    all_args.algorithm_name = "dqn"
    
    # Ensure self-play is enabled
    all_args.use_selfplay = True
    
    # Set random seeds
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    
    # Setup directories first
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))) / "results" / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    # Setup logging EARLY so we can see all output
    log_file = run_dir / 'train.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    
    # Setup CUDA
    if all_args.cuda and torch.cuda.is_available():
        logging.info("Using GPU for training...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        logging.info("Using CPU for training...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    
    # Setup wandb
    wandb_run = None
    if all_args.use_wandb:
        try:
            wandb_run = wandb.init(
                config=all_args,
                project=all_args.env_name,
                entity=all_args.wandb_name,
                notes=socket.gethostname(),
                name=str(all_args.algorithm_name) + "_" +
                     str(all_args.experiment_name) +
                     "_seed" + str(all_args.seed),
                group=all_args.scenario_name,
                dir=str(run_dir),
                job_type="training",
                reinit=True
            )
            logging.info("WandB initialized successfully!")
        except Exception as e:
            logging.warning(f"Failed to initialize WandB: {e}")
            logging.warning("Continuing training without WandB logging...")
            all_args.use_wandb = False
    
    if not all_args.use_wandb:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                           str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
    
    # Setup process title
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name)
    )
    
    logging.info("=" * 50)
    logging.info("Training DQN on JSBSim Environment")
    logging.info("=" * 50)
    logging.info(f"Algorithm: {all_args.algorithm_name}")
    logging.info(f"Scenario: {all_args.scenario_name}")
    logging.info(f"Experiment: {all_args.experiment_name}")
    logging.info(f"Seed: {all_args.seed}")
    logging.info(f"Device: {device}")
    logging.info(f"Rollout threads: {all_args.n_rollout_threads}")
    logging.info(f"Total env steps: {int(all_args.num_env_steps)}")
    logging.info("")
    logging.info("DQN Hyperparameters:")
    logging.info(f"  Batch size: {all_args.dqn_batch_size}")
    logging.info(f"  Update interval: {all_args.dqn_update_interval}")
    logging.info(f"  Target update interval: {all_args.dqn_target_update_interval}")
    logging.info(f"  Warmup steps: {all_args.dqn_warmup_steps}")
    logging.info(f"  Buffer size: {all_args.dqn_buffer_size}")
    logging.info(f"  Epsilon: {all_args.epsilon_start} -> {all_args.epsilon_end}")
    logging.info(f"  Learning rate: {all_args.lr}")
    logging.info(f"  Gamma: {all_args.gamma}")
    logging.info("=" * 50)
    
    # Create environments
    logging.info("Creating training environments...")
    envs = make_train_env(all_args)
    logging.info(f"Training environments created: {all_args.n_rollout_threads} parallel threads")
    
    if all_args.use_eval:
        logging.info("Creating evaluation environments...")
        eval_envs = make_eval_env(all_args)
        logging.info(f"Evaluation environments created: {all_args.n_eval_rollout_threads} parallel threads")
    else:
        eval_envs = None
    
    # Create runner
    logging.info("Initializing DQN Selfplay Runner...")
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": envs.num_agents,
        "device": device,
        "run_dir": run_dir,
        "render_mode": all_args.render_mode
    }
    
    runner = DQNSelfplayRunner(config)
    logging.info("Runner initialized successfully!")
    
    # Start training
    logging.info("\n" + "=" * 50)
    logging.info("Starting training...")
    logging.info("=" * 50 + "\n")
    try:
        runner.run()
    except Exception as e:
        logging.error("Training failed with exception:")
        logging.error(traceback.format_exc())
        raise e
    finally:
        # Cleanup
        envs.close()
        if eval_envs is not None:
            eval_envs.close()
        if all_args.use_wandb and wandb_run is not None:
            wandb_run.finish()
        logging.info("Training completed!")


if __name__ == "__main__":
    main(sys.argv[1:])
