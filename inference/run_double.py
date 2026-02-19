"""
Inference script for double pendulum policy evaluation.

Logs per-step data:
- Observations (policy and critic)
- Actions (raw from policy)
- Q-values (critic values)
- Rewards (total and raw)
- arm1 cos/sin/vel, arm2 cos/sin/vel, cart pos/vel
- Cumulative return

Usage:
    python scripts/inverted_pendulum/inference/run_double.py \
        --num_envs 2 \
        --num_episodes 5 \
        --checkpoint_path scripts/inverted_pendulum/logs/checkpoints/double_pendulum_swingup/best.pt \
        --config scripts/inverted_pendulum/configs/double_pendulum_swingup.yaml
"""

import argparse
import sys
import os

ISAACLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ISAACLAB_ROOT not in sys.path:
    sys.path.insert(0, ISAACLAB_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Double pendulum policy inference with logging")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run per env.")
parser.add_argument("--test_name", type=str, default="double_pendulum_test", help="Name for this test run.")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint file.")
parser.add_argument("--config", type=str, default=None, help="Path to YAML config (for network architecture).")
parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions.")
parser.add_argument("--record_cameras", action="store_true", default=False, help="Enable camera recording.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

### -----------------------------------------------------------

import json
import math
import time
from typing import Dict, List, Any, Optional

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

from scripts.inverted_pendulum.training.config import load_config
from scripts.inverted_pendulum.training.agent import Agent
from scripts.inverted_pendulum.envs.env_wrapper import EnvWrapper


### -------------------- TEST LOGGER --------------------

class TestLogger:
    """Logs per-step data for double pendulum policy evaluation."""

    def __init__(self, log_dir: str, env_ids_to_log: List[int], test_name: str = "test_run"):
        self.log_dir = log_dir
        self.env_ids_to_log = env_ids_to_log
        self.test_name = test_name
        self.save_dir = log_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.data_buffers = {env_id: self._create_empty_buffer() for env_id in env_ids_to_log}
        self.episode_counts = {env_id: 0 for env_id in env_ids_to_log}
        self.step_count = 0

        self.metadata = {
            "test_name": test_name,
            "env_ids_logged": env_ids_to_log,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        print(f"[TestLogger] Initialized. Log dir: {self.save_dir}")

    def _create_empty_buffer(self) -> Dict[str, List]:
        return {
            "obs_policy": [],
            "obs_critic": [],
            "actions_raw": [],
            "q_values": [],
            "rewards_total": [],
            "rewards_raw": [],
            "cumulative_reward": [],
            "arm1_cos": [],
            "arm1_sin": [],
            "arm1_vel": [],
            "arm2_cos": [],
            "arm2_sin": [],
            "arm2_vel": [],
            "cart_pos": [],
            "cart_vel": [],
            "dones": [],
            "episode_starts": [],
            "step_indices": [],
            "frames": [],
        }

    def log_step(
        self,
        obs: Dict[str, torch.Tensor],
        actions_raw: torch.Tensor,
        q_values: torch.Tensor,
        rewards: torch.Tensor,
        rewards_raw: np.ndarray,
        dones: torch.Tensor,
        cumulative_rewards: torch.Tensor,
        frames: Optional[Dict[int, np.ndarray]] = None,
    ):
        self.step_count += 1

        for env_id in self.env_ids_to_log:
            buf = self.data_buffers[env_id]
            buf["step_indices"].append(self.step_count)

            # Camera frame
            if frames is not None and env_id in frames:
                buf["frames"].append(frames[env_id])

            # Observations
            policy_obs = obs["policy"][env_id].cpu().numpy()
            buf["obs_policy"].append(policy_obs)
            buf["obs_critic"].append(obs["critic"][env_id].cpu().numpy())

            # Decompose obs: [cos(arm1), sin(arm1), arm1_vel, cos(arm2), sin(arm2), arm2_vel, cart_pos, cart_vel]
            buf["arm1_cos"].append(float(policy_obs[0]))
            buf["arm1_sin"].append(float(policy_obs[1]))
            buf["arm1_vel"].append(float(policy_obs[2]))
            buf["arm2_cos"].append(float(policy_obs[3]))
            buf["arm2_sin"].append(float(policy_obs[4]))
            buf["arm2_vel"].append(float(policy_obs[5]))
            buf["cart_pos"].append(float(policy_obs[6]))
            buf["cart_vel"].append(float(policy_obs[7]))

            # Actions
            buf["actions_raw"].append(actions_raw[env_id].cpu().numpy())

            # Q-values
            q_val = q_values[env_id]
            if isinstance(q_val, torch.Tensor):
                q_val = q_val.cpu().numpy()
            buf["q_values"].append(float(q_val.flatten()[0]) if hasattr(q_val, 'flatten') else float(q_val))

            # Rewards
            r = rewards[env_id]
            buf["rewards_total"].append(float(r.cpu().numpy()) if isinstance(r, torch.Tensor) else float(r))
            buf["rewards_raw"].append(float(rewards_raw[env_id]))

            # Cumulative return
            cr = cumulative_rewards[env_id]
            buf["cumulative_reward"].append(float(cr.cpu().numpy()) if isinstance(cr, torch.Tensor) else float(cr))

            # Dones
            d = dones[env_id]
            buf["dones"].append(float(d.cpu().numpy()) if isinstance(d, torch.Tensor) else float(d))

    def on_episode_done(self, env_id: int):
        if env_id in self.env_ids_to_log:
            self.episode_counts[env_id] += 1
            current_step = len(self.data_buffers[env_id]["rewards_total"])
            self.data_buffers[env_id]["episode_starts"].append(current_step)
            print(f"[TestLogger] Env {env_id} completed episode {self.episode_counts[env_id]}")

    def save(self):
        self.metadata["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.metadata["total_steps"] = self.step_count
        self.metadata["episodes_per_env"] = dict(self.episode_counts)

        for env_id in self.env_ids_to_log:
            buf = self.data_buffers[env_id]
            save_data = {}
            for key, value in buf.items():
                if len(value) > 0:
                    save_data[key] = np.array(value)
                else:
                    save_data[key] = np.array([])

            save_path = os.path.join(self.save_dir, f"env_{env_id}_data.npz")
            np.savez_compressed(save_path, **save_data)
            print(f"[TestLogger] Saved env {env_id} data to {save_path}")
            print(f"[TestLogger]   Steps: {len(buf['rewards_total'])}, Episodes: {self.episode_counts[env_id]}, Frames: {len(buf['frames'])}")

        metadata_path = os.path.join(self.save_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"[TestLogger] All data saved to {self.save_dir}")


### -------------------- CAMERA HELPERS --------------------

def set_camera_pose(env, camera_id: int, env_id: int):
    """Set camera to look at the double pendulum from the side."""
    unwrapped = env.unwrapped
    cart_root = unwrapped.robot.data.default_root_state[env_id, :3].clone()
    camera_target = cart_root + unwrapped.scene.env_origins[env_id]
    eye_camera = camera_target + torch.tensor([0.0, 5.0, 1.0], device=unwrapped.device)
    
    unwrapped.cameras[camera_id].set_world_poses_from_view(
        eye_camera.unsqueeze(0),
        camera_target.unsqueeze(0),
    )


def get_camera_frames(env, env_ids: List[int]) -> Dict[int, np.ndarray]:
    """Capture camera frames for specific environment IDs."""
    frames = {}
    unwrapped = env.unwrapped

    for camera in unwrapped.cameras:
        camera.update(unwrapped.step_dt)

    for i, env_id in enumerate(env_ids):
        if i < len(unwrapped.cameras):
            cam_data = unwrapped.cameras[i].data.output["rgb"]
            if isinstance(cam_data, torch.Tensor):
                cam_data = cam_data.cpu().numpy()
            if cam_data.ndim == 4:
                cam_data = cam_data[0]
            frames[env_id] = cam_data.astype(np.uint8).copy()

    return frames


### -------------------- ENV CREATION --------------------

def make_env(num_envs: int, output_type: str = "torch", enable_cameras: bool = False):
    id_name = "double-pendulum-swingup-v0-inference"
    gym.register(
        id=id_name,
        entry_point="scripts.inverted_pendulum.envs.double_pendulum_swingup_env:DoublePendulumSwingUpEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "scripts.inverted_pendulum.envs.double_pendulum_swingup_env:DoublePendulumSwingUpEnvCfg",
        },
    )

    env_cfg = parse_env_cfg(id_name, num_envs=num_envs)
    env_cfg.enable_cameras = enable_cameras
    env = gym.make(id_name, cfg=env_cfg, render_mode="rgb_array")

    env = EnvWrapper(
        env,
        output_type=output_type,
        enable_normalization_obs=True,
        enable_normalization_rewards=False,
    )

    return env


### -------------------- INFERENCE --------------------

def run_inference(
    env,
    agent: Agent,
    device: torch.device,
    checkpoint_path: str,
    num_episodes: int,
    logger: TestLogger,
    deterministic: bool = True,
    enable_cameras: bool = False,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load normalizers
    if hasattr(env, 'normalizers') and 'normalizer_state' in checkpoint:
        for k, state in checkpoint['normalizer_state'].items():
            if k in env.normalizers:
                env.normalizers[k].mean = state['mean']
                env.normalizers[k].var = state['var']
                env.normalizers[k].count = state['count']
                env.normalizers[k].clip_range = state['clip_range']

    # Load agent weights
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()

    print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
    print(f"[INFO] Checkpoint update: {checkpoint.get('update', '?')}, step: {checkpoint.get('global_step', '?')}")

    logger.metadata["checkpoint_path"] = checkpoint_path
    logger.metadata["checkpoint_update"] = checkpoint.get("update", "unknown")
    logger.metadata["checkpoint_global_step"] = checkpoint.get("global_step", "unknown")

    num_envs = env.unwrapped.num_envs
    env_ids_to_log = logger.env_ids_to_log

    # Initialize
    env.eval()
    obs, _ = env.reset()

    # Set camera poses
    if enable_cameras:
        for i, env_id in enumerate(env_ids_to_log):
            if i < len(env.unwrapped.cameras):
                set_camera_pose(env, camera_id=i, env_id=env_id)
        print(f"[INFO] Cameras enabled, tracking envs {env_ids_to_log}")

    # LSTM states
    next_lstm_state_actor = (
        torch.zeros(agent.num_layers, num_envs, agent.hidden_size, device=device),
        torch.zeros(agent.num_layers, num_envs, agent.hidden_size, device=device),
    )
    next_lstm_state_critic = (
        torch.zeros(agent.num_layers, num_envs, agent.hidden_size, device=device),
        torch.zeros(agent.num_layers, num_envs, agent.hidden_size, device=device),
    )

    dones = torch.zeros(num_envs, device=device)
    cumulative_rewards = torch.zeros(num_envs, device=device)

    # Episode tracking
    episodes_completed = {env_id: 0 for env_id in range(num_envs)}
    total_episodes_needed = num_episodes * len(env_ids_to_log)
    all_returns = []
    all_lengths = []
    episode_lengths = np.zeros(num_envs, dtype=np.int32)

    print(f"[INFO] Running {num_episodes} episodes per env for envs {env_ids_to_log}")
    print(f"[INFO] Deterministic: {deterministic}")

    step = 0
    try:
        with torch.no_grad():
            while True:
                step += 1

                action_raw, _, _, next_lstm_state_actor, _, _ = agent.get_action(
                    obs, next_lstm_state_actor, dones, deterministic=deterministic
                )

                q_value, next_lstm_state_critic = agent.get_value(
                    obs, next_lstm_state_critic, dones, denorm=True, update_running_mean=False
                )

                next_obs, reward, terminated, truncated, info_custom = env.step(action_raw)
                next_done = (terminated | truncated).float()

                rewards_raw = info_custom.get('org_reward', np.zeros(num_envs))
                if isinstance(rewards_raw, torch.Tensor):
                    rewards_raw = rewards_raw.cpu().numpy()

                cumulative_rewards += reward
                episode_lengths += 1

                # Capture camera frames
                frames = None
                if enable_cameras:
                    frames = get_camera_frames(env, env_ids_to_log)

                # Log step
                logger.log_step(
                    obs=next_obs,
                    actions_raw=action_raw,
                    q_values=q_value.flatten(),
                    rewards=reward,
                    rewards_raw=rewards_raw,
                    dones=next_done,
                    cumulative_rewards=cumulative_rewards.clone(),
                    frames=frames,
                )

                # Check episode completions
                done_np = next_done.cpu().numpy()
                if np.any(done_np):
                    done_env_ids = np.where(done_np > 0)[0]

                    for env_id in done_env_ids:
                        episodes_completed[env_id] += 1
                        all_returns.append(cumulative_rewards[env_id].item())
                        all_lengths.append(episode_lengths[env_id])

                        if env_id in env_ids_to_log:
                            logger.on_episode_done(env_id)

                        print(
                            f"  Env {env_id} ep {episodes_completed[env_id]}: "
                            f"return={cumulative_rewards[env_id].item():.2f}, "
                            f"length={episode_lengths[env_id]}"
                        )

                        cumulative_rewards[env_id] = 0
                        episode_lengths[env_id] = 0

                    logged_episodes = sum(logger.episode_counts.values())
                    if logged_episodes >= total_episodes_needed:
                        print(f"[INFO] Completed {logged_episodes} episodes")
                        break

                obs = next_obs
                dones = next_done

                if step % 200 == 0:
                    logged_eps = sum(logger.episode_counts.values())
                    print(f"[INFO] Step {step}, Episodes logged: {logged_eps}/{total_episodes_needed}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Saving logged data...")

    logger.save()

    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {len(all_returns)}")
    if len(all_returns) > 0:
        print(f"Avg Return:  {np.mean(all_returns):.2f}  (std: {np.std(all_returns):.2f})")
        print(f"Avg Length:  {np.mean(all_lengths):.1f}")
        print(f"Min Return:  {np.min(all_returns):.2f}")
        print(f"Max Return:  {np.max(all_returns):.2f}")
    print("=" * 60)


### -------------------- MAIN --------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cfg, ppo_cfg, net_cfg, norm_cfg, log_cfg = load_config(yaml_path=args_cli.config)

    env = make_env(num_envs=args_cli.num_envs, output_type="torch", enable_cameras=args_cli.record_cameras)

    actor_obs_dim = env.total_obs_space["policy"].shape[-1]
    critic_obs_dim = env.total_obs_space["critic"].shape[-1]
    action_dim = env.action_space.shape[-1]

    agent = Agent(
        actor_obs_dim=actor_obs_dim,
        critic_obs_dim=critic_obs_dim,
        action_dim=action_dim,
        cfg=net_cfg,
        eval_mode=True,
    ).to(device)

    print(f"Actor obs dim: {actor_obs_dim}, Critic obs dim: {critic_obs_dim}, Action dim: {action_dim}")

    log_dir = os.path.join("scripts", "inverted_pendulum", "logs", "inference", args_cli.test_name)
    env_ids_to_log = list(range(min(args_cli.num_envs, 2)))

    logger = TestLogger(
        log_dir=log_dir,
        env_ids_to_log=env_ids_to_log,
        test_name=args_cli.test_name,
    )

    logger.metadata["num_envs"] = args_cli.num_envs
    logger.metadata["num_episodes"] = args_cli.num_episodes
    logger.metadata["deterministic"] = args_cli.deterministic
    logger.metadata["enable_cameras"] = args_cli.record_cameras

    run_inference(
        env=env,
        agent=agent,
        device=device,
        checkpoint_path=args_cli.checkpoint_path,
        num_episodes=args_cli.num_episodes,
        logger=logger,
        deterministic=args_cli.deterministic,
        enable_cameras=args_cli.record_cameras,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
