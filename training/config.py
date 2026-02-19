"""
Configuration dataclasses and YAML loader for PPO-LSTM training.
"""

from dataclasses import dataclass, field, fields
from typing import Optional, List
import yaml


@dataclass
class TrainConfig:
    exp_name: str = "default"
    seed: int = 42
    num_envs: int = 128
    device: str = "cuda"


@dataclass
class PPOConfig:
    # Core PPO
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2

    # Training schedule
    num_steps: int = 128
    num_updates: int = 300
    update_epochs: int = 4
    num_minibatches: int = 32
    max_grad_norm: float = 1.0

    # Loss coefficients
    vf_coef: float = 2.0
    ent_coef: float = 0.0001
    bound_loss_coef: float = 0.0001

    # Normalization toggles
    norm_adv: bool = True
    norm_value: bool = True
    clip_vloss: bool = True

    # LR scheduling
    anneal_lr: bool = True
    adaptive_scheduler: bool = True
    target_kl: float = 0.008


@dataclass
class NetworkConfig:
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    actor_mlp_layers: list = field(default_factory=lambda: [64, 32])
    critic_mlp_layers: list = field(default_factory=lambda: [64, 32])
    activation: str = "elu"
    use_layer_norm: bool = True
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    value_clip_range: float = 5.0


@dataclass
class NormalizationConfig:
    enable_obs: bool = True
    obs_clip_range: float = 8.0
    enable_rewards: bool = False
    reward_clip_range: float = 5.0


@dataclass
class LoggingConfig:
    wandb_enabled: bool = True
    wandb_project: str = "cartpole_rl"
    wandb_group: Optional[str] = None
    checkpoint_dir: str = "scripts/inverted_pendulum/logs/checkpoints"
    checkpoint_interval: int = 1
    snapshot_divisor: int = 10


def _update_dataclass(dc, overrides: dict):
    """Update a dataclass instance with values from a dict. Unknown keys are ignored."""
    if overrides is None:
        return dc
    for f in fields(dc):
        if f.name in overrides:
            setattr(dc, f.name, overrides[f.name])
    return dc


def load_config(yaml_path: Optional[str] = None, cli_overrides: Optional[dict] = None):
    """
    Load configuration from YAML file, with optional CLI overrides.

    Returns:
        Tuple of (TrainConfig, PPOConfig, NetworkConfig, NormalizationConfig, LoggingConfig)
    """
    yaml_data = {}
    if yaml_path is not None:
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}

    train_cfg = _update_dataclass(TrainConfig(), yaml_data.get("train", {}))
    ppo_cfg = _update_dataclass(PPOConfig(), yaml_data.get("ppo", {}))
    net_cfg = _update_dataclass(NetworkConfig(), yaml_data.get("network", {}))
    norm_cfg = _update_dataclass(NormalizationConfig(), yaml_data.get("normalization", {}))
    log_cfg = _update_dataclass(LoggingConfig(), yaml_data.get("logging", {}))

    # Apply CLI overrides (flat keys mapped to TrainConfig)
    if cli_overrides:
        _update_dataclass(train_cfg, cli_overrides)
        _update_dataclass(ppo_cfg, cli_overrides)

    return train_cfg, ppo_cfg, net_cfg, norm_cfg, log_cfg
