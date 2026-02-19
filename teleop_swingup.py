"""
Keyboard teleop for pendulum environments (single or double).

Controls:
    A / D         Push cart left / right
    UP / DOWN     Increase / decrease force scale
    R             Reset environment

Usage:
    python scripts/inverted_pendulum/teleop_swingup.py --task single
    python scripts/inverted_pendulum/teleop_swingup.py --task double
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pendulum keyboard teleop")
parser.add_argument("--task", type=str, default="single", choices=["single", "double"],
                    help="Which pendulum: single or double.")
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

import carb
import omni

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

ISAACLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ISAACLAB_ROOT not in sys.path:
    sys.path.insert(0, ISAACLAB_ROOT)


### -------------------- ROBOT CONFIGS --------------------

URDF_PATH = os.path.join(os.path.dirname(__file__), "DoublePendulumURDF", "robot.urdf")

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
    prim_path="/World/Robot",
)


### -------------------- KEYBOARD --------------------

class CartpoleKeyboard:
    """Simple keyboard for 1D cart force control."""

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
        msg = "Pendulum Keyboard Teleop\n"
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
    def __init__(self, reset_fn):
        self._reset_fn = reset_fn
        self.window = ui.Window("Pendulum Control", width=250, height=80,
                                flags=ui.WINDOW_FLAGS_NO_COLLAPSE)
        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label(f"Task: {args_cli.task} pendulum")
                ui.Button("Reset", clicked_fn=self._on_reset)
        self.window.visible = True

    def _on_reset(self):
        print("[GUI] Reset triggered")
        self._reset_fn()

    def destroy(self):
        if self.window:
            self.window.destroy()
            self.window = None


### -------------------- SINGLE PENDULUM --------------------

def run_single():
    sim_cfg = SimulationCfg(dt=1.0 / 120.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[8.0, 0.0, 5.0], target=[0.0, 0.0, 2.0])

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = CARTPOLE_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state.joint_pos = {"slider_to_cart": 0.0, "cart_to_pole": math.pi}
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.reset()

    cart_idx, _ = robot.find_joints("slider_to_cart")
    pole_idx, _ = robot.find_joints("cart_to_pole")

    keyboard = CartpoleKeyboard(force_scale=args_cli.force_scale)
    print(f"\n{keyboard}\n")

    def reset_fn():
        joint_pos = robot.data.default_joint_pos.clone()
        joint_pos[:, pole_idx] = math.pi
        joint_vel = torch.zeros_like(joint_pos)
        root_state = robot.data.default_root_state.clone()
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.reset()
        print("  [RESET]")

    reset_window = ResetWindow(reset_fn)

    decimation = 2
    step = 0

    while simulation_app.is_running():
        kb = keyboard.advance()
        if kb["reset"]:
            reset_fn()
            step = 0

        force_val = kb["force"]
        effort = torch.zeros(robot.num_instances, robot.num_joints, device=robot.device)
        effort[:, cart_idx] = force_val

        for _ in range(decimation):
            robot.set_joint_effort_target(effort)
            robot.write_data_to_sim()
            sim.step()

        robot.update(dt=sim.get_physics_dt() * decimation)

        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        pole_angle = joint_pos[0, pole_idx[0]].item()
        pole_vel = joint_vel[0, pole_idx[0]].item()
        cart_pos = joint_pos[0, cart_idx[0]].item()
        cart_vel = joint_vel[0, cart_idx[0]].item()
        angle_norm = ((pole_angle + math.pi) % (2 * math.pi)) - math.pi
        angle_deg = abs(angle_norm) * 180.0 / math.pi

        if step % 15 == 0:
            print(
                f"[{step:5d}] pole={angle_deg:5.1f}° vel={pole_vel:+6.2f} | "
                f"cart={cart_pos:+6.2f}m vel={cart_vel:+6.2f} | "
                f"force={force_val:+7.1f}N"
            )
        step += 1

    reset_window.destroy()


### -------------------- DOUBLE PENDULUM --------------------

def run_double():
    sim_cfg = SimulationCfg(dt=1.0 / 120.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[5.0, -5.0, 4.0], target=[0.0, 0.0, 1.5])

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(DOUBLE_PENDULUM_CFG)

    sim.reset()
    robot.reset()

    slide_idx, _ = robot.find_joints("slide")
    rev1_idx, _ = robot.find_joints("revolute")
    rev2_idx, _ = robot.find_joints("revolute2")

    keyboard = CartpoleKeyboard(force_scale=args_cli.force_scale)
    print(f"\n{keyboard}\n")
    print(f"Joints: {robot.joint_names}")

    def reset_fn():
        joint_pos = robot.data.default_joint_pos.clone()
        joint_pos[:, slide_idx] = 0.0
        joint_pos[:, rev1_idx] = math.pi
        joint_pos[:, rev2_idx] = 0.0
        joint_vel = torch.zeros_like(joint_pos)
        root_state = robot.data.default_root_state.clone()
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.reset()
        print("  [RESET]")

    reset_window = ResetWindow(reset_fn)

    decimation = 2
    step = 0

    while simulation_app.is_running():
        kb = keyboard.advance()
        if kb["reset"]:
            reset_fn()
            step = 0

        force_val = kb["force"]
        effort = torch.zeros(robot.num_instances, robot.num_joints, device=robot.device)
        effort[:, slide_idx] = force_val

        for _ in range(decimation):
            robot.set_joint_effort_target(effort)
            robot.write_data_to_sim()
            sim.step()

        robot.update(dt=sim.get_physics_dt() * decimation)

        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        cart_pos = joint_pos[0, slide_idx[0]].item()
        cart_vel = joint_vel[0, slide_idx[0]].item()
        arm1_angle = joint_pos[0, rev1_idx[0]].item()
        arm1_vel = joint_vel[0, rev1_idx[0]].item()
        arm2_angle = joint_pos[0, rev2_idx[0]].item()
        arm2_vel = joint_vel[0, rev2_idx[0]].item()

        arm1_deg = ((arm1_angle + math.pi) % (2 * math.pi) - math.pi) * 180.0 / math.pi
        arm2_deg = ((arm2_angle + math.pi) % (2 * math.pi) - math.pi) * 180.0 / math.pi

        if step % 15 == 0:
            print(
                f"[{step:5d}] arm1={arm1_deg:+7.1f}° vel={arm1_vel:+6.2f} | "
                f"arm2={arm2_deg:+7.1f}° vel={arm2_vel:+6.2f} | "
                f"cart={cart_pos:+6.2f}m vel={cart_vel:+6.2f} | "
                f"force={force_val:+7.1f}N"
            )
        step += 1

    reset_window.destroy()


### -------------------- MAIN --------------------

def main():
    if args_cli.task == "single":
        run_single()
    else:
        run_double()


if __name__ == "__main__":
    main()
    simulation_app.close()
