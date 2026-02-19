"""
PPO training entry point for cartpole balancing.

Usage:
    python scripts/inverted_pendulum/train.py --num_envs 128
    python scripts/inverted_pendulum/train.py --num_envs 128 --config scripts/inverted_pendulum/configs/cartpole_balance.yaml
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PPO Training for Cartpole")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments.")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.")
parser.add_argument("--exp_name", type=str, default=None, help="Override exp_name from config.")
parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

### -----------------------------------------------------------

import os
import sys
import random

import numpy as np
import torch
import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

# Add IsaacLab root to path
isaaclab_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if isaaclab_root not in sys.path:
    sys.path.insert(0, isaaclab_root)

from scripts.inverted_pendulum.training.config import load_config
from scripts.inverted_pendulum.training.agent import Agent
from scripts.inverted_pendulum.training import ppo
from scripts.inverted_pendulum.envs.env_wrapper import EnvWrapper


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def make_env(num_envs: int, norm_cfg=None, output_type: str = "torch"):
    id_name = "double-pendulum-swingup-v0"
    gym.register(
        id=id_name,
        entry_point="scripts.inverted_pendulum.envs.double_pendulum_swingup_env:DoublePendulumSwingUpEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "scripts.inverted_pendulum.envs.double_pendulum_swingup_env:DoublePendulumSwingUpEnvCfg",
        },
    )

    env_cfg = parse_env_cfg(id_name, num_envs=num_envs)
    env = gym.make(id_name, cfg=env_cfg, render_mode="rgb_array")

    enable_norm_obs = True
    enable_norm_rewards = False
    if norm_cfg is not None:
        enable_norm_obs = norm_cfg.enable_obs
        enable_norm_rewards = norm_cfg.enable_rewards

    env = EnvWrapper(
        env,
        output_type=output_type,
        enable_normalization_obs=enable_norm_obs,
        enable_normalization_rewards=enable_norm_rewards,
    )

    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Load config ---
    cli_overrides = {}
    if args_cli.exp_name is not None:
        cli_overrides["exp_name"] = args_cli.exp_name
    cli_overrides["num_envs"] = args_cli.num_envs

    train_cfg, ppo_cfg, net_cfg, norm_cfg, log_cfg = load_config(
        yaml_path=args_cli.config,
        cli_overrides=cli_overrides,
    )

    # Override defaults for cartpole if no config file provided
    if args_cli.config is None:
        train_cfg.exp_name = train_cfg.exp_name if args_cli.exp_name else "cartpole_balance"
        log_cfg.checkpoint_dir = "scripts/inverted_pendulum/logs/checkpoints"
        log_cfg.wandb_project = "cartpole_rl"

    # --- Seed ---
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    # --- Print config ---
    print("=" * 60)
    print(f"Experiment: {train_cfg.exp_name}")
    print(f"Config file: {args_cli.config or 'defaults'}")
    print(f"  train: {train_cfg}")
    print(f"  ppo:   {ppo_cfg}")
    print(f"  net:   {net_cfg}")
    print(f"  norm:  {norm_cfg}")
    print(f"  log:   {log_cfg}")
    print("=" * 60)

    # --- Create env ---
    env = make_env(
        num_envs=train_cfg.num_envs,
        norm_cfg=norm_cfg,
        output_type="torch",
    )
    env.train()

    # --- Create agent ---
    actor_obs_dim = env.total_obs_space["policy"].shape[-1]
    critic_obs_dim = env.total_obs_space["critic"].shape[-1]
    action_dim = env.action_space.shape[-1]

    agent = Agent(
        actor_obs_dim=actor_obs_dim,
        critic_obs_dim=critic_obs_dim,
        action_dim=action_dim,
        cfg=net_cfg,
        actor_lr=ppo_cfg.learning_rate,
        critic_lr=ppo_cfg.learning_rate,
        norm_value=ppo_cfg.norm_value,
    ).to(device)

    print(f"Actor obs dim: {actor_obs_dim}, Critic obs dim: {critic_obs_dim}, Action dim: {action_dim}")
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # --- Train ---
    ppo.train(
        env=env,
        agent=agent,
        train_cfg=train_cfg,
        ppo_cfg=ppo_cfg,
        log_cfg=log_cfg,
        resume=args_cli.resume,
    )

    env.close()
    simulation_app.close()
