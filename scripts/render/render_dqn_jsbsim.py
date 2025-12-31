#!/usr/bin/env python
"""
Rendering script for DQN on JSBSim SingleCombat environment
Generates ACMI files for Tacview visualization
"""
import sys
import os
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
from envs.env_wrappers import DummyVecEnv


def make_render_env(all_args):
    """Create rendering environment (single thread)."""
    def get_env_fn(rank):
        def init_env():
            env = SingleCombatEnv(all_args.scenario_name)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    return DummyVecEnv([get_env_fn(0)])


def parse_args(args, parser):
    """Parse JSBSim specific arguments."""
    group = parser.add_argument_group("JSBSim Render parameters")
    group.add_argument('--scenario-name', type=str, 
                      default='1v1/ShootMissile/Selfplay',
                      help="Which scenario to run on")
    group.add_argument('--render-mode', type=str, default='txt',
                      help="txt or real_time")
    group.add_argument('--render-index', type=int, default=0,
                      help="Episode index of ego policy to render")
    group.add_argument('--render-opponent-index', type=int, default=0,
                      help="Episode index of opponent policy to render")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    """Main rendering function."""
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # Set algorithm name
    all_args.algorithm_name = "dqn"
    
    # Ensure model directory is specified
    if all_args.model_dir is None:
        logging.error("Please specify --model-dir pointing to the trained model directory!")
        logging.error("Example: --model-dir results/SingleCombat/1v1/ShootMissile/Selfplay/dqn/exp_name")
        sys.exit(1)
    
    # Set random seeds
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    
    # Setup directories
    model_dir = Path(all_args.model_dir)
    if not model_dir.exists():
        logging.error(f"Model directory does not exist: {model_dir}")
        sys.exit(1)
    
    # Infer experiment name and run_dir from model_dir if not specified
    if hasattr(all_args, 'experiment_name') and all_args.experiment_name:
        experiment_name = all_args.experiment_name
    else:
        experiment_name = model_dir.name
        all_args.experiment_name = experiment_name
    
    run_dir = model_dir / "render"
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(run_dir / 'render.log'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    # Setup CUDA
    if all_args.cuda and torch.cuda.is_available():
        logging.info("Using GPU for rendering...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
    else:
        logging.info("Using CPU for rendering...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    
    # Setup process title
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-render-" + str(all_args.env_name) + "@" + str(all_args.user_name)
    )
    
    logging.info("=" * 50)
    logging.info("Rendering DQN on JSBSim Environment")
    logging.info("=" * 50)
    logging.info(f"Algorithm: {all_args.algorithm_name}")
    logging.info(f"Scenario: {all_args.scenario_name}")
    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"Model directory: {model_dir}")
    logging.info(f"Ego policy index: {all_args.render_index}")
    logging.info(f"Opponent policy index: {all_args.render_opponent_index}")
    logging.info(f"Device: {device}")
    logging.info("=" * 50)
    
    # Create environment (single thread for rendering)
    logging.info("Creating rendering environment...")
    envs = make_render_env(all_args)
    logging.info("Rendering environment created")
    
    # Create runner
    logging.info("Initializing DQN Selfplay Runner...")
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": envs.num_agents,
        "device": device,
        "run_dir": run_dir,
        "render_mode": all_args.render_mode
    }
    
    runner = DQNSelfplayRunner(config)
    logging.info("Runner initialized successfully!")
    
    # Start rendering
    logging.info("\n" + "=" * 50)
    logging.info("Starting rendering...")
    logging.info("=" * 50 + "\n")
    try:
        runner.render()
    except Exception as e:
        logging.error("Rendering failed with exception:")
        logging.error(str(e))
        import traceback
        logging.error(traceback.format_exc())
        raise e
    finally:
        # Cleanup
        envs.close()
        logging.info("\nRendering process completed!")


if __name__ == "__main__":
    main(sys.argv[1:])
