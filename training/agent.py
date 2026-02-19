"""
Configurable Actor-Critic agent with LSTM for PPO.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from .config import NetworkConfig


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_activation(name: str):
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations.keys())}")
    return activations[name]


def build_mlp(layer_sizes: list, activation: str):
    """Build a sequential MLP from a list of layer sizes."""
    act_cls = get_activation(activation)
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(layer_init(nn.Linear(layer_sizes[i], layer_sizes[i + 1])))
        layers.append(act_cls())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Running Normalizer (for value normalization)
# ---------------------------------------------------------------------------

class RunningNormalizer(nn.Module):
    """Torch-based running mean/var normalizer with denormalization support."""

    def __init__(self, insize: int, epsilon: float = 1e-5, clip_range: float = 5.0):
        super().__init__()
        self.epsilon = epsilon
        self.clip_range = clip_range

        self.register_buffer("running_mean", torch.zeros(insize, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(insize, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update_stats(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        return new_mean, new_var, tot_count

    def forward(self, input, denorm=False):
        if self.training:
            mean = input.mean(0)
            var = input.var(0)
            self.running_mean, self.running_var, self.count = self._update_stats(
                self.running_mean, self.running_var, self.count,
                mean, var, input.size()[0],
            )

        current_mean = self.running_mean
        current_var = self.running_var

        if denorm:
            y = torch.clamp(input, min=-self.clip_range, max=self.clip_range)
            y = torch.sqrt(current_var.float() + self.epsilon) * y + current_mean.float()
        else:
            y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
            y = torch.clamp(y, min=-self.clip_range, max=self.clip_range)

        return y


# ---------------------------------------------------------------------------
# LSTM with done-flag handling
# ---------------------------------------------------------------------------

class LSTMwithDones(nn.Module):
    """LSTM that resets hidden state on episode boundaries."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.hidden_size = hidden_size
        self.input_size = input_size

    def forward(self, inputs, lstm_state, done):
        new_hidden = []
        for x, d in zip(inputs, done):
            h, lstm_state = self.lstm(
                x.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden.append(h)

        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: NetworkConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.cfg = cfg

        self.lstm = LSTMwithDones(obs_dim, cfg.lstm_hidden_size, cfg.lstm_num_layers)

        if cfg.use_layer_norm:
            self.layer_norm = nn.LayerNorm(cfg.lstm_hidden_size)
        else:
            self.layer_norm = nn.Identity()

        mlp_sizes = [cfg.lstm_hidden_size] + cfg.actor_mlp_layers
        self.mlp = build_mlp(mlp_sizes, cfg.activation)

        final_dim = cfg.actor_mlp_layers[-1]
        self.actor_mean = layer_init(nn.Linear(final_dim, action_dim), std=0.01)
        self.actor_logstd = layer_init(nn.Linear(final_dim, action_dim), std=0.01)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

    def get_states(self, x, lstm_state, done):
        batch_size = lstm_state[0].shape[1]
        x = x.reshape((-1, batch_size, self.obs_dim))
        done = done.reshape((-1, batch_size))
        new_hidden, new_lstm_state = self.lstm(x, lstm_state, done)
        return new_hidden, new_lstm_state

    def get_action(self, x, lstm_state, done, action=None, deterministic=False):
        hidden, new_lstm_state = self.get_states(x, lstm_state, done)
        hidden = self.layer_norm(hidden)

        mlp_output = self.mlp(hidden)
        action_mean = self.actor_mean(mlp_output)
        action_logstd = self.actor_logstd(mlp_output)
        action_logstd = torch.clamp(action_logstd, self.cfg.log_std_min, self.cfg.log_std_max)
        std = action_logstd.exp()

        dist = Normal(action_mean, std)
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = dist.sample()

        a_mu = action_mean.detach().clone()
        a_std = std.detach().clone()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), new_lstm_state, a_mu, a_std


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    def __init__(self, obs_dim: int, cfg: NetworkConfig):
        super().__init__()
        self.obs_dim = obs_dim

        self.lstm = LSTMwithDones(obs_dim, cfg.lstm_hidden_size, cfg.lstm_num_layers)

        if cfg.use_layer_norm:
            self.layer_norm = nn.LayerNorm(cfg.lstm_hidden_size)
        else:
            self.layer_norm = nn.Identity()

        mlp_sizes = [cfg.lstm_hidden_size] + cfg.critic_mlp_layers
        self.mlp = build_mlp(mlp_sizes, cfg.activation)

        final_dim = cfg.critic_mlp_layers[-1]
        self.critic_value = layer_init(nn.Linear(final_dim, 1), std=1.0)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

    def get_states(self, x, lstm_state, done):
        batch_size = lstm_state[0].shape[1]
        x = x.reshape((-1, batch_size, self.obs_dim))
        done = done.reshape((-1, batch_size))
        new_hidden, new_lstm_state = self.lstm(x, lstm_state, done)
        return new_hidden, new_lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, new_lstm_state = self.get_states(x, lstm_state, done)
        hidden = self.layer_norm(hidden)
        mlp_output = self.mlp(hidden)
        value = self.critic_value(mlp_output)
        return value, new_lstm_state


# ---------------------------------------------------------------------------
# Agent (combines Actor + Critic)
# ---------------------------------------------------------------------------

class Agent(nn.Module):
    def __init__(
        self,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        cfg: NetworkConfig,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        norm_value: bool = True,
        eval_mode: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.lstm_hidden_size
        self.num_layers = cfg.lstm_num_layers

        self.actor = Actor(actor_obs_dim, action_dim, cfg)
        self.critic = Critic(critic_obs_dim, cfg)

        if not eval_mode:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        self.norm_value = norm_value
        if norm_value:
            self.critic_normalizer = RunningNormalizer(1, clip_range=cfg.value_clip_range)

    def get_action(self, x, lstm_state, done, action=None, deterministic=False):
        x = x["policy"]
        return self.actor.get_action(x, lstm_state, done, action, deterministic)

    def get_value(self, x, lstm_state, done, denorm=True, update_running_mean=False):
        x = x["critic"]
        value, new_lstm_state = self.critic.get_value(x, lstm_state, done)

        if denorm and self.norm_value:
            if update_running_mean:
                self.critic_normalizer.train()
                value = self.critic_normalizer(value, denorm=True)
                self.critic_normalizer.eval()
            else:
                self.critic_normalizer.eval()
                value = self.critic_normalizer(value, denorm=True)

        return value, new_lstm_state
