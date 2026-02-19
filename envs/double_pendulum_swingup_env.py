"""
Double Pendulum Swing-Up Environment for RL training.

Both arms start hanging DOWN and the agent must learn to swing them
up to the upright position and balance.

Observations (8D):
    cos(arm1), sin(arm1), arm1_vel, cos(arm2), sin(arm2), arm2_vel, cart_pos, cart_vel

Action: 1D force on the cart (only actuated joint)
"""

from __future__ import annotations

import os
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


# ---------- Robot config ----------

URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "DoublePendulumURDF", "robot.urdf")

DOUBLE_PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0, damping=0.0,
            ),
            target_type="none",
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
        joint_pos={"slide": 0.0, "revolute": 3.14159, "revolute2": 0.0},
    ),
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slide"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=20.0,
        ),
        "arm_actuator": ImplicitActuatorCfg(
            joint_names_expr=["revolute"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "arm2_actuator": ImplicitActuatorCfg(
            joint_names_expr=["revolute2"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    prim_path="/World/envs/env_.*/Robot",
)


# ---------- Env config ----------

@configclass
class DoublePendulumSwingUpEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 15.0
    action_scale = 130.0  # [N]
    action_space = 1
    observation_space = 8  # cos/sin(arm1), arm1_vel, cos/sin(arm2), arm2_vel, cart_pos, cart_vel
    state_space = 8

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = DOUBLE_PENDULUM_CFG
    cart_dof_name = "slide"
    arm1_dof_name = "revolute"
    arm2_dof_name = "revolute2"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=6.0, replicate_physics=True
    )

    # reset
    max_cart_pos = 3.0  # cart OOB boundary [m]
    initial_arm1_angle_range = [-math.pi / 2 - 0.4, math.pi / 2 + 0.4]  # perturbation around pi
    initial_arm2_angle_range = [-0.5, 0.5]  # small perturbation around 0
    initial_cart_vel_range = [-2.0, 2.0]
    initial_arm1_vel_range = [-1.0, 1.0]
    initial_arm2_vel_range = [-1.0, 1.0]

    # reward scales
    rew_scale_arm1_upright = 1.0    # cos(arm1_angle): +1 upright, -1 down
    rew_scale_arm2_upright = 1.0    # cos(arm2_angle): +1 aligned up, -1 down
    rew_scale_cart_pos = -0.005
    rew_scale_cart_vel = -0.001
    rew_scale_arm1_vel = -0.0005
    rew_scale_arm2_vel = -0.0005
    rew_scale_terminated = -4.0
    rew_scale_energy = 0.0

    # camera
    enable_cameras = False


# ---------- Env ----------

class DoublePendulumSwingUpEnv(DirectRLEnv):
    cfg: DoublePendulumSwingUpEnvCfg

    def __init__(self, cfg: DoublePendulumSwingUpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._arm1_dof_idx, _ = self.robot.find_joints(self.cfg.arm1_dof_name)
        self._arm2_dof_idx, _ = self.robot.find_joints(self.cfg.arm2_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cartpole = self.robot  # alias for compatibility with inference/camera scripts
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=True)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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
        self.robot.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        arm1_angle = self.joint_pos[:, self._arm1_dof_idx[0]]
        arm2_angle = self.joint_pos[:, self._arm2_dof_idx[0]]
        obs = torch.cat(
            (
                torch.cos(arm1_angle).unsqueeze(dim=1),
                torch.sin(arm1_angle).unsqueeze(dim=1),
                self.joint_vel[:, self._arm1_dof_idx[0]].unsqueeze(dim=1),
                torch.cos(arm2_angle).unsqueeze(dim=1),
                torch.sin(arm2_angle).unsqueeze(dim=1),
                self.joint_vel[:, self._arm2_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return {"policy": obs, "critic": obs.clone()}

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_double_pendulum_rewards(
            self.cfg.rew_scale_arm1_upright,
            self.cfg.rew_scale_arm2_upright,
            self.cfg.rew_scale_cart_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_arm1_vel,
            self.cfg.rew_scale_arm2_vel,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_energy,
            self.joint_pos[:, self._arm1_dof_idx[0]],
            self.joint_vel[:, self._arm1_dof_idx[0]],
            self.joint_pos[:, self._arm2_dof_idx[0]],
            self.joint_vel[:, self._arm2_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.actions[:, 0],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(
            torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1
        )
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # arm1 starts at pi (hanging down) + random perturbation
        joint_pos[:, self._arm1_dof_idx] = math.pi + sample_uniform(
            self.cfg.initial_arm1_angle_range[0],
            self.cfg.initial_arm1_angle_range[1],
            joint_pos[:, self._arm1_dof_idx].shape,
            joint_pos.device,
        )

        # arm2 starts near 0 (aligned with arm1) + small perturbation
        joint_pos[:, self._arm2_dof_idx] = sample_uniform(
            self.cfg.initial_arm2_angle_range[0],
            self.cfg.initial_arm2_angle_range[1],
            joint_pos[:, self._arm2_dof_idx].shape,
            joint_pos.device,
        )

        # random initial velocities
        joint_vel[:, self._cart_dof_idx] = sample_uniform(
            self.cfg.initial_cart_vel_range[0],
            self.cfg.initial_cart_vel_range[1],
            joint_vel[:, self._cart_dof_idx].shape,
            joint_vel.device,
        )
        joint_vel[:, self._arm1_dof_idx] = sample_uniform(
            self.cfg.initial_arm1_vel_range[0],
            self.cfg.initial_arm1_vel_range[1],
            joint_vel[:, self._arm1_dof_idx].shape,
            joint_vel.device,
        )
        joint_vel[:, self._arm2_dof_idx] = sample_uniform(
            self.cfg.initial_arm2_vel_range[0],
            self.cfg.initial_arm2_vel_range[1],
            joint_vel[:, self._arm2_dof_idx].shape,
            joint_vel.device,
        )

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_double_pendulum_rewards(
    rew_scale_arm1_upright: float,
    rew_scale_arm2_upright: float,
    rew_scale_cart_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_arm1_vel: float,
    rew_scale_arm2_vel: float,
    rew_scale_terminated: float,
    rew_scale_energy: float,
    arm1_pos: torch.Tensor,
    arm1_vel: torch.Tensor,
    arm2_pos: torch.Tensor,
    arm2_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    # upright rewards: cos(angle) -> +1 at 0 (up), -1 at pi (down)
    rew_arm1_upright = rew_scale_arm1_upright * torch.cos(arm1_pos)
    rew_arm2_upright = rew_scale_arm2_upright * torch.cos(arm2_pos)

    # penalties
    rew_cart_pos = rew_scale_cart_pos * torch.square(cart_pos)
    rew_cart_vel = rew_scale_cart_vel * torch.abs(cart_vel)
    rew_arm1_vel = rew_scale_arm1_vel * torch.abs(arm1_vel)
    rew_arm2_vel = rew_scale_arm2_vel * torch.abs(arm2_vel)
    rew_energy = rew_scale_energy * torch.square(actions)
    rew_termination = rew_scale_terminated * reset_terminated.float()

    total_reward = (
        rew_arm1_upright + rew_arm2_upright
        + rew_cart_pos + rew_cart_vel
        + rew_arm1_vel + rew_arm2_vel
        + rew_energy + rew_termination
    )
    return total_reward
