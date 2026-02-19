"""
Rerun-based visualization of logged cartpole inference data.

Visualizes synchronized:
- Pole angle (deg from upright)
- Cart position
- Actions (force)
- Q-values
- Rewards (total, raw, cumulative)
- Episode boundaries
- Camera frames (if captured with --enable_cameras during inference)

Usage:
    python scripts/inverted_pendulum/visualize_rerun.py \
        --test_name cartpole_test --env_id 0

    python scripts/inverted_pendulum/visualize_rerun.py \
        --data_path scripts/inverted_pendulum/logs/inference/cartpole_test/env_0_data.npz
"""

import argparse
import os
import sys
import numpy as np

try:
    import rerun as rr
except ImportError:
    print("Rerun not found. Install with: pip install rerun-sdk")
    sys.exit(1)


def load_data(data_path: str) -> dict:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def visualize_with_rerun(data: dict, title: str = "Cartpole Evaluation", env_id: int = 0):
    rr.init(title, spawn=True)

    # Extract data
    actions_raw = data.get('actions_raw', np.array([]))
    q_values = data.get('q_values', np.array([]))
    rewards_total = data.get('rewards_total', np.array([]))
    rewards_raw = data.get('rewards_raw', np.array([]))
    cumulative_reward = data.get('cumulative_reward', np.array([]))
    pole_angle = data.get('pole_angle', np.array([]))
    pole_vel = data.get('pole_vel', np.array([]))
    cart_pos = data.get('cart_pos', np.array([]))
    cart_vel = data.get('cart_vel', np.array([]))
    episode_starts = data.get('episode_starts', np.array([]))
    dones = data.get('dones', np.array([]))
    frames = data.get('frames', np.array([]))

    has_frames = len(frames) > 0 and frames.ndim == 4  # (N, H, W, C)
    if has_frames:
        print(f"  Camera frames: {frames.shape[0]} frames, {frames.shape[1]}x{frames.shape[2]}")
    else:
        print("  No camera frames in data.")

    num_steps = len(q_values)
    if num_steps == 0:
        print("No data to visualize!")
        return

    print(f"Visualizing {num_steps} steps...")

    for step in range(num_steps):
        rr.set_time("step", sequence=step)

        # Pole angle (convert to degrees from upright)
        if step < len(pole_angle):
            angle_deg = float(pole_angle[step]) * 180.0 / np.pi
            rr.log(f"env_{env_id}/state/pole_angle_deg", rr.Scalars(angle_deg))

        # Pole angular velocity
        if step < len(pole_vel):
            rr.log(f"env_{env_id}/state/pole_vel", rr.Scalars(float(pole_vel[step])))

        # Cart position
        if step < len(cart_pos):
            rr.log(f"env_{env_id}/state/cart_pos", rr.Scalars(float(cart_pos[step])))

        # Cart velocity
        if step < len(cart_vel):
            rr.log(f"env_{env_id}/state/cart_vel", rr.Scalars(float(cart_vel[step])))

        # Action (force)
        if step < len(actions_raw):
            rr.log(f"env_{env_id}/action/force", rr.Scalars(float(actions_raw[step][0])))

        # Q-value
        if step < len(q_values):
            rr.log(f"env_{env_id}/values/q_value", rr.Scalars(float(q_values[step])))

        # Rewards
        if step < len(rewards_total):
            rr.log(f"env_{env_id}/rewards/total", rr.Scalars(float(rewards_total[step])))

        if step < len(rewards_raw):
            rr.log(f"env_{env_id}/rewards/raw", rr.Scalars(float(rewards_raw[step])))

        if step < len(cumulative_reward):
            rr.log(f"env_{env_id}/rewards/cumulative", rr.Scalars(float(cumulative_reward[step])))

        # Done flag
        if step < len(dones):
            rr.log(f"env_{env_id}/done", rr.Scalars(float(dones[step])))

        # Camera frame
        if has_frames and step < len(frames):
            rr.log(f"env_{env_id}/camera", rr.Image(frames[step]))

        if step % 100 == 0:
            print(f"  Logged step {step}/{num_steps}")

    # Episode boundaries
    for i, ep_start in enumerate(episode_starts):
        if ep_start > 0:
            rr.set_time("step", sequence=int(ep_start))
            rr.log(f"env_{env_id}/episode_marker", rr.TextLog(f"Episode {i+1} start"))

    print("Done! Rerun viewer should be open.")


def main():
    parser = argparse.ArgumentParser(description="Visualize cartpole inference data with Rerun")
    parser.add_argument("--test_name", type=str, default=None, help="Test name to load")
    parser.add_argument("--env_id", type=int, default=0, help="Environment ID to visualize")
    parser.add_argument("--data_path", type=str, default=None, help="Direct path to .npz file")

    args = parser.parse_args()

    if args.data_path:
        data_path = args.data_path
    elif args.test_name:
        data_path = os.path.join(
            "scripts", "inverted_pendulum", "logs", "inference",
            args.test_name, f"env_{args.env_id}_data.npz"
        )
    else:
        print("Please provide either --test_name or --data_path")
        return

    print(f"Loading data from {data_path}")
    data = load_data(data_path)
    visualize_with_rerun(data=data, title=f"Cartpole - Env {args.env_id}", env_id=args.env_id)


if __name__ == "__main__":
    main()
