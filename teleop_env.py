"""
Keyboard teleop through the actual gym RL environment.

Tests the full env pipeline: obs, rewards, termination, reset.
Useful for verifying env behavior before training.

Controls:
    A / D         Push cart left / right
    UP / DOWN     Increase / decrease force scale
    R             Reset environment

Usage:
    python scripts/inverted_pendulum/teleop_env.py --task swingup
    python scripts/inverted_pendulum/teleop_env.py --task balance
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Env keyboard teleop")
parser.add_argument("--task", type=str, default="swingup", choices=["swingup", "balance", "double_swingup"],
                    help="Which env to test.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--force_scale", type=float, default=130.0, help="Initial force scale [N].")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.ui as ui

### -----------------------------------------------------------

import os
import sys
import math
import weakref
import torch
import numpy as np
import gymnasium as gym

import carb
import omni

from isaaclab_tasks.utils import parse_env_cfg

ISAACLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ISAACLAB_ROOT not in sys.path:
    sys.path.insert(0, ISAACLAB_ROOT)


### -------------------- KEYBOARD --------------------

class CartpoleKeyboard:
    def __init__(self, force_scale: float = 130.0):
        self.force_scale = force_scale
        self._force_dir = 0.0
        self.reset_flag = False

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def __del__(self):
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self):
        msg = "Env Keyboard Teleop\n"
        msg += "  A / D         : Push cart left / right\n"
        msg += "  UP / DOWN     : Increase / decrease force scale\n"
        msg += "  R             : Reset\n"
        msg += f"  Force scale   : {self.force_scale:.1f} N\n"
        return msg

    def advance(self) -> dict:
        force = self._force_dir * self.force_scale
        reset = self.reset_flag
        self.reset_flag = False
        return {"force": force, "reset": reset}

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in ("A", "LEFT"):
                self._force_dir = -1.0
            elif event.input.name in ("D", "RIGHT"):
                self._force_dir = 1.0
            elif event.input.name == "R":
                self.reset_flag = True
            elif event.input.name == "UP":
                self.force_scale = min(self.force_scale + 10.0, 500.0)
                print(f"  Force scale: {self.force_scale:.1f} N")
            elif event.input.name == "DOWN":
                self.force_scale = max(self.force_scale - 10.0, 10.0)
                print(f"  Force scale: {self.force_scale:.1f} N")

        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ("A", "D", "LEFT", "RIGHT"):
                self._force_dir = 0.0

        return True


### -------------------- GUI --------------------

class ResetWindow:
    def __init__(self, reset_fn, task_name: str):
        self._reset_fn = reset_fn
        self.window = ui.Window("Env Teleop", width=250, height=80,
                                flags=ui.WINDOW_FLAGS_NO_COLLAPSE)
        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label(f"Task: {task_name}")
                ui.Button("Reset", clicked_fn=self._on_reset)
        self.window.visible = True

    def _on_reset(self):
        print("[GUI] Reset triggered")
        self._reset_fn()

    def destroy(self):
        if self.window:
            self.window.destroy()
            self.window = None


### -------------------- ENV CREATION --------------------

ENV_REGISTRY = {
    "swingup": {
        "id": "cartpole-swingup-v0-teleop",
        "entry_point": "scripts.inverted_pendulum.envs.cartpole_swingup_env:CartpoleSwingUpEnv",
        "cfg_entry_point": "scripts.inverted_pendulum.envs.cartpole_swingup_env:CartpoleSwingUpEnvCfg",
    },
    "balance": {
        "id": "cartpole-balance-v0-teleop",
        "entry_point": "scripts.inverted_pendulum.envs.cartpole_env:CartpoleBalanceEnv",
        "cfg_entry_point": "scripts.inverted_pendulum.envs.cartpole_env:CartpoleBalanceEnvCfg",
    },
    "double_swingup": {
        "id": "double-pendulum-swingup-v0-teleop",
        "entry_point": "scripts.inverted_pendulum.envs.double_pendulum_swingup_env:DoublePendulumSwingUpEnv",
        "cfg_entry_point": "scripts.inverted_pendulum.envs.double_pendulum_swingup_env:DoublePendulumSwingUpEnvCfg",
    },
}


def make_env(task: str, num_envs: int):
    reg = ENV_REGISTRY[task]
    gym.register(
        id=reg["id"],
        entry_point=reg["entry_point"],
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": reg["cfg_entry_point"]},
    )
    env_cfg = parse_env_cfg(reg["id"], num_envs=num_envs)
    env = gym.make(reg["id"], cfg=env_cfg, render_mode="rgb_array")
    return env


### -------------------- MAIN --------------------

def main():
    task = args_cli.task
    env = make_env(task=task, num_envs=args_cli.num_envs)
    obs, _ = env.reset()

    keyboard = CartpoleKeyboard(force_scale=args_cli.force_scale)
    print(f"\n{keyboard}")

    action_scale = env.unwrapped.action_scale
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    print(f"Task: {task}")
    print(f"Action scale: {action_scale} N")
    print(f"Num envs: {num_envs}")
    print(f"Obs space: {env.unwrapped.observation_space}")
    print(f"Action space: {env.unwrapped.action_space}\n")

    def reset_fn():
        nonlocal obs
        obs, _ = env.reset()
        print("  [ENV RESET]")

    reset_window = ResetWindow(reset_fn, task)

    step = 0
    cumulative_reward = 0.0
    episode_steps = 0

    while simulation_app.is_running():
        kb = keyboard.advance()

        if kb["reset"]:
            reset_fn()
            cumulative_reward = 0.0
            episode_steps = 0
            step = 0

        # Normalize action by action_scale (env multiplies by action_scale internally)
        action = torch.tensor([[kb["force"] / action_scale]], dtype=torch.float32)
        actions = action.repeat(num_envs, 1).to(device)

        next_obs, reward, terminated, truncated, info = env.step(actions)

        r = reward[0].item() if isinstance(reward, torch.Tensor) else float(reward)
        cumulative_reward += r
        episode_steps += 1

        done = False
        if isinstance(terminated, torch.Tensor):
            done = (terminated[0] | truncated[0]).item()
        else:
            done = terminated or truncated

        # Print telemetry
        if step % 15 == 0 or done:
            obs_dict = next_obs if isinstance(next_obs, dict) else {"policy": next_obs}
            obs_np = obs_dict["policy"][0].cpu().numpy()
            obs_str = " ".join([f"{v:+7.3f}" for v in obs_np])

            status = ""
            if done:
                if isinstance(terminated, torch.Tensor) and terminated[0].item():
                    status = " [TERMINATED]"
                else:
                    status = " [TRUNCATED]"

            print(
                f"[{step:5d}] obs=[{obs_str}] | "
                f"rew={r:+.4f} cumR={cumulative_reward:+.2f} | "
                f"force={kb['force']:+7.1f}N | ep_steps={episode_steps}"
                f"{status}"
            )

        if done:
            cumulative_reward = 0.0
            episode_steps = 0

        obs = next_obs
        step += 1

    reset_window.destroy()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
