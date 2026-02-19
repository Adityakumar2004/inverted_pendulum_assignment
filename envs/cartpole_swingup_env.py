"""
Cartpole Swing-Up Environment for RL training.

The pole starts hanging DOWN (pi radians) and the agent must learn to
swing it up to the upright position and balance it there.

Key differences from the balance env:
  - Pole initial angle: pi (hanging down) with random perturbation
  - Initial cart velocity: random ±2.0 m/s (helps exploration)
  - Reward: encourages swing-up via cos(pole_angle), relaxed penalties
  - No early termination on pole angle (agent needs freedom to swing)
  - Longer episode: 15 seconds (swing-up takes time)
  - Action scale: 130N (tuned via keyboard testing)
  - Observation space: 4 (pole_angle, pole_vel, cart_pos, cart_vel)
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


@configclass
class CartpoleSwingUpEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 15.0  # longer episode for swing-up
    action_scale = 130.0  # [N] tuned via keyboard testing
    action_space = 1
    observation_space = 5   # cos(pole), sin(pole), pole_vel, cart_pos, cart_vel
    state_space = 5         # critic gets the same obs (symmetric)

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot — pole starts hanging DOWN
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # reset
    max_cart_pos = 3.0  # cart reset boundary [m]
    initial_pole_angle_range = [-0.25, 0.25]  # perturbation around pi [rad]
    initial_cart_vel_range = [-2.0, 2.0]      # random cart velocity on reset [m/s]
    initial_pole_vel_range = [-1.0, 1.0]      # random pole angular velocity on reset [rad/s]

    # reward scales
    rew_scale_upright = 1.0        # cos(pole_angle): +1 when upright, -1 when down
    rew_scale_cart_pos = -0.005    # penalize cart drifting from center (relaxed)
    rew_scale_cart_vel = -0.001    # penalize excessive cart velocity (relaxed)
    rew_scale_pole_vel = -0.0005   # small penalty on pole angular velocity (relaxed)
    rew_scale_terminated = -2.0    # penalty for going out of bounds
    rew_scale_energy = 0.0         # disabled — let agent use full force

    # camera (only used during inference with --record_cameras)
    enable_cameras = False


class CartpoleSwingUpEnv(DirectRLEnv):
    cfg: CartpoleSwingUpEnvCfg

    def __init__(self, cfg: CartpoleSwingUpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        # ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["cartpole"] = self.cartpole
        # lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # cameras (only created when enable_cameras is True, e.g. during inference)
        self.cameras = []
        if self.cfg.enable_cameras:
            camera_cfg = CameraCfg(
                prim_path="/World/Camera",
                update_period=0.1,
                height=480,
                width=640,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0,
                    horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, -5.0, 3.0),
                    rot=(0.5, -0.5, 0.5, -0.5),
                    convention="ros",
                ),
            )
            self.cameras.append(Camera(camera_cfg))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        pole_angle = self.joint_pos[:, self._pole_dof_idx[0]]
        obs = torch.cat(
            (
                torch.cos(pole_angle).unsqueeze(dim=1),
                torch.sin(pole_angle).unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return {"policy": obs, "critic": obs.clone()}

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_swingup_rewards(
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_cart_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_energy,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.actions[:, 0],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # only terminate if cart goes too far — let the pole swing freely
        out_of_bounds = torch.any(
            torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1
        )
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]
        joint_pos[:, self._pole_dof_idx] = math.pi + sample_uniform(torch.tensor([-math.pi/2-0.4], device=joint_pos.device), torch.tensor([math.pi/2 + 0.4], device=joint_pos.device),
                                                                    joint_pos[:, self._pole_dof_idx].shape, joint_pos.device)

        # # pole starts at pi (hanging down) + random perturbation
        # joint_pos[:, self._pole_dof_idx] = math.pi + sample_uniform(
        #     self.cfg.initial_pole_angle_range[0],
        #     self.cfg.initial_pole_angle_range[1],
        #     joint_pos[:, self._pole_dof_idx].shape,
        #     joint_pos.device,
        # )


        # # random initial cart velocity to help exploration
        # joint_vel[:, self._cart_dof_idx] = sample_uniform(
        #     self.cfg.initial_cart_vel_range[0],
        #     self.cfg.initial_cart_vel_range[1],
        #     joint_vel[:, self._cart_dof_idx].shape,
        #     joint_vel.device,
        # )
        # # random initial pole angular velocity
        # joint_vel[:, self._pole_dof_idx] = sample_uniform(
        #     self.cfg.initial_pole_vel_range[0],
        #     self.cfg.initial_pole_vel_range[1],
        #     joint_vel[:, self._pole_dof_idx].shape,
        #     joint_vel.device,
        # )

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_swingup_rewards(
    rew_scale_upright: float,
    rew_scale_cart_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    rew_scale_terminated: float,
    rew_scale_energy: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    # upright reward: cos(pole_angle) → +1 at 0 (up), -1 at pi (down)
    rew_upright = rew_scale_upright * torch.cos(pole_pos)

    # penalize cart drifting from center
    rew_cart_pos = rew_scale_cart_pos * torch.square(cart_pos)

    # penalize cart velocity
    rew_cart_vel = rew_scale_cart_vel * torch.abs(cart_vel)

    # penalize pole angular velocity (helps stabilize once upright)
    rew_pole_vel = rew_scale_pole_vel * torch.abs(pole_vel)

    # penalize large forces (energy efficiency)
    rew_energy = rew_scale_energy * torch.square(actions)

    # termination penalty
    rew_termination = rew_scale_terminated * reset_terminated.float()

    total_reward = rew_upright + rew_cart_pos + rew_cart_vel + rew_pole_vel + rew_energy + rew_termination
    return total_reward
