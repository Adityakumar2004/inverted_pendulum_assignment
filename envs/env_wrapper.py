"""
Environment wrapper for obs/reward normalization and output type conversion.
Simplified from peg_hole_2/utils_1.py — no camera/recording/reward-shaping logic.
"""

import numpy as np
import torch
import gymnasium as gym


class RunningNormalizer:
    def __init__(self, size, epsilon=1e-5, clip_range=5.0):
        self.size = size
        self.mean = np.zeros(size, dtype=np.float32)
        self.var = np.ones(size, dtype=np.float32)
        self.count = epsilon
        self.clip_range = clip_range

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray):
        std = np.sqrt(self.var) + 1e-8
        x_norm = (x - self.mean) / std
        return np.clip(x_norm, -self.clip_range, self.clip_range)


class EnvWrapper(gym.Wrapper):
    """
    Wraps an Isaac Lab DirectRLEnv to provide:
      - obs dict with "policy" and "critic" keys
      - optional running normalization of observations and rewards
      - output as torch tensors or numpy arrays
    """

    def __init__(
        self,
        env,
        output_type: str = "torch",
        enable_normalization_obs: bool = True,
        enable_normalization_rewards: bool = False,
    ):
        super().__init__(env)
        self.env = env
        self.total_obs_space = {
            "policy": env.unwrapped.observation_space,
            "critic": env.unwrapped.state_space,
        }
        self.output_type = output_type

        self.enable_normalization_rewards = enable_normalization_rewards
        self.enable_normalization_obs = enable_normalization_obs
        self.training = True

        self.normalizers = {}
        if enable_normalization_obs:
            self.normalizers["policy"] = RunningNormalizer(
                self.total_obs_space["policy"].shape[-1], clip_range=8.0
            )
            self.normalizers["critic"] = RunningNormalizer(
                self.total_obs_space["critic"].shape[-1], clip_range=8.0
            )

        if enable_normalization_rewards:
            self.normalizers["rewards"] = RunningNormalizer(1, clip_range=5.0)

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.env.unwrapped.device)

        actions = torch.clamp(actions, -1, 1)

        obs, rewards, terminations, truncations, info = self.env.step(actions)

        info_custom = {}
        info_custom["org_reward"] = rewards.cpu().numpy()

        if self.enable_normalization_obs:
            if self.training:
                self.normalizers["policy"].update(obs["policy"].cpu().numpy())
                self.normalizers["critic"].update(obs["critic"].cpu().numpy())

            with torch.no_grad():
                obs["policy"] = torch.tensor(
                    self.normalizers["policy"].normalize(obs["policy"].cpu().numpy()),
                    device=self.env.unwrapped.device,
                )
                obs["critic"] = torch.tensor(
                    self.normalizers["critic"].normalize(obs["critic"].cpu().numpy()),
                    device=self.env.unwrapped.device,
                )

        if self.enable_normalization_rewards:
            if self.training:
                self.normalizers["rewards"].update(rewards.cpu().numpy())
                rewards = torch.tensor(
                    self.normalizers["rewards"].normalize(rewards.cpu().numpy()),
                    device=self.env.unwrapped.device,
                )

        if self.output_type == "numpy":
            obs = {k: v.cpu().numpy() for k, v in obs.items()}
            rewards = rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards
            terminations = terminations.cpu().numpy()
            truncations = truncations.cpu().numpy()

        return obs, rewards, terminations, truncations, info_custom

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        if self.enable_normalization_obs:
            if self.training:
                self.normalizers["policy"].update(obs["policy"].cpu().numpy())
                self.normalizers["critic"].update(obs["critic"].cpu().numpy())

            with torch.no_grad():
                obs["policy"] = torch.tensor(
                    self.normalizers["policy"].normalize(obs["policy"].cpu().numpy()),
                    device=self.env.unwrapped.device,
                )
                obs["critic"] = torch.tensor(
                    self.normalizers["critic"].normalize(obs["critic"].cpu().numpy()),
                    device=self.env.unwrapped.device,
                )

        if self.output_type == "numpy":
            obs = {k: v.cpu().numpy() for k, v in obs.items()}

        return obs, info

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
