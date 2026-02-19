"""
Double-pendulum teleop with GUI sliders for joint inspection.

Features:
    - Sliders to set each revolute joint angle (physics paused while dragging)
    - Live readout of raw angles (rad) and wrapped angles (deg)
    - Keyboard A/D to push cart, R to reset
    - --no_gravity flag to disable gravity for inspecting joint zeros

Usage:
    python scripts/inverted_pendulum/teleop_double.py
    python scripts/inverted_pendulum/teleop_double.py --no_gravity
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Double pendulum teleop with GUI sliders")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--force_scale", type=float, default=130.0, help="Initial force scale [N].")
parser.add_argument("--no_gravity", action="store_true", help="Disable gravity.")

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

ISAACLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ISAACLAB_ROOT not in sys.path:
    sys.path.insert(0, ISAACLAB_ROOT)


### -------------------- ROBOT CONFIG --------------------

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
        msg = "Double Pendulum Keyboard Teleop\n"
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


### -------------------- GUI WITH SLIDERS --------------------

class DoublePendulumGUI:
    """GUI with sliders for revolute1, revolute2, and live angle readouts."""

    def __init__(self, robot, slide_idx, rev1_idx, rev2_idx, reset_fn):
        self._robot = robot
        self._slide_idx = slide_idx
        self._rev1_idx = rev1_idx
        self._rev2_idx = rev2_idx
        self._reset_fn = reset_fn
        self._slider_active = False  # True while user drags a slider

        _DEG_360 = 2 * math.pi  # ±360° in radians

        self.window = ui.Window("Double Pendulum Control", width=500, height=380,
                                flags=ui.WINDOW_FLAGS_NO_COLLAPSE)
        with self.window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Double Pendulum Teleop", style={"font_size": 16})
                ui.Separator()

                # --- Revolute 1 ---
                ui.Label("Revolute 1 (cart -> arm1)")
                with ui.HStack(spacing=5):
                    self._rev1_slider = ui.FloatSlider(min=-_DEG_360, max=_DEG_360, step=0.01, width=ui.Fraction(3))
                    self._rev1_slider.model.set_value(math.pi)
                    self._rev1_field = ui.FloatField(width=70)
                    self._rev1_field.model.set_value(math.pi)
                    self._rev1_label = ui.Label("  +180.0°", width=80)
                self._rev1_slider.model.add_value_changed_fn(self._on_rev1_slider)
                self._rev1_field.model.add_value_changed_fn(self._on_rev1_field)

                ui.Spacer(height=4)

                # --- Revolute 2 ---
                ui.Label("Revolute 2 (arm1 -> arm2)")
                with ui.HStack(spacing=5):
                    self._rev2_slider = ui.FloatSlider(min=-_DEG_360, max=_DEG_360, step=0.01, width=ui.Fraction(3))
                    self._rev2_slider.model.set_value(0.0)
                    self._rev2_field = ui.FloatField(width=70)
                    self._rev2_field.model.set_value(0.0)
                    self._rev2_label = ui.Label("    +0.0°", width=80)
                self._rev2_slider.model.add_value_changed_fn(self._on_rev2_slider)
                self._rev2_field.model.add_value_changed_fn(self._on_rev2_field)

                ui.Spacer(height=4)

                # --- Cart ---
                ui.Label("Cart position (slide)")
                with ui.HStack(spacing=5):
                    self._cart_slider = ui.FloatSlider(min=-3.0, max=3.0, step=0.01, width=ui.Fraction(3))
                    self._cart_slider.model.set_value(0.0)
                    self._cart_field = ui.FloatField(width=70)
                    self._cart_field.model.set_value(0.0)
                    self._cart_label = ui.Label("  +0.000 m", width=80)
                self._cart_slider.model.add_value_changed_fn(self._on_cart_slider)
                self._cart_field.model.add_value_changed_fn(self._on_cart_field)

                ui.Spacer(height=8)

                with ui.HStack(spacing=10):
                    ui.Button("Reset", clicked_fn=self._on_reset, width=80)
                    ui.Button("Read Joints", clicked_fn=self._on_read_joints, width=100)

        self._syncing = False  # prevent slider<->field infinite loop
        self.window.visible = True

    def _set_joint_and_zero_vel(self, joint_idx, value):
        """Write a single joint position and zero its velocity."""
        self._slider_active = True
        joint_pos = self._robot.data.joint_pos.clone()
        joint_vel = self._robot.data.joint_vel.clone()
        joint_pos[:, joint_idx] = value
        joint_vel[:, joint_idx] = 0.0
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self._robot.reset()

    def _apply_joint(self, joint_idx, val, slider, field, label, is_angle=True):
        """Apply a value to joint, sync slider<->field, update label."""
        if self._syncing:
            return
        self._syncing = True
        self._slider_active = True
        self._set_joint_and_zero_vel(joint_idx, val)
        slider.model.set_value(val)
        field.model.set_value(val)
        if is_angle:
            wrapped = ((val + math.pi) % (2 * math.pi) - math.pi)
            label.text = f"  {math.degrees(wrapped):+.1f}°"
        else:
            label.text = f"  {val:+.3f} m"
        self._syncing = False

    # --- Revolute 1 ---
    def _on_rev1_slider(self, model):
        self._apply_joint(self._rev1_idx, model.as_float,
                          self._rev1_slider, self._rev1_field, self._rev1_label)

    def _on_rev1_field(self, model):
        self._apply_joint(self._rev1_idx, model.as_float,
                          self._rev1_slider, self._rev1_field, self._rev1_label)

    # --- Revolute 2 ---
    def _on_rev2_slider(self, model):
        self._apply_joint(self._rev2_idx, model.as_float,
                          self._rev2_slider, self._rev2_field, self._rev2_label)

    def _on_rev2_field(self, model):
        self._apply_joint(self._rev2_idx, model.as_float,
                          self._rev2_slider, self._rev2_field, self._rev2_label)

    # --- Cart ---
    def _on_cart_slider(self, model):
        self._apply_joint(self._slide_idx, model.as_float,
                          self._cart_slider, self._cart_field, self._cart_label, is_angle=False)

    def _on_cart_field(self, model):
        self._apply_joint(self._slide_idx, model.as_float,
                          self._cart_slider, self._cart_field, self._cart_label, is_angle=False)

    def _on_reset(self):
        print("[GUI] Reset triggered")
        self._reset_fn()
        self._syncing = True
        self._rev1_slider.model.set_value(math.pi)
        self._rev1_field.model.set_value(math.pi)
        self._rev1_label.text = f"  +180.0°"
        self._rev2_slider.model.set_value(0.0)
        self._rev2_field.model.set_value(0.0)
        self._rev2_label.text = f"    +0.0°"
        self._cart_slider.model.set_value(0.0)
        self._cart_field.model.set_value(0.0)
        self._cart_label.text = f"  +0.000 m"
        self._syncing = False

    def _on_read_joints(self):
        """Read current joint positions from sim and update sliders + labels."""
        jp = self._robot.data.joint_pos[0]
        r1 = jp[self._rev1_idx[0]].item()
        r2 = jp[self._rev2_idx[0]].item()
        cp = jp[self._slide_idx[0]].item()

        self._syncing = True
        self._rev1_slider.model.set_value(r1)
        self._rev1_field.model.set_value(r1)
        self._rev2_slider.model.set_value(r2)
        self._rev2_field.model.set_value(r2)
        self._cart_slider.model.set_value(cp)
        self._cart_field.model.set_value(cp)
        self._syncing = False

        w1 = ((r1 + math.pi) % (2 * math.pi) - math.pi)
        w2 = ((r2 + math.pi) % (2 * math.pi) - math.pi)
        self._rev1_label.text = f"  {math.degrees(w1):+.1f}°"
        self._rev2_label.text = f"  {math.degrees(w2):+.1f}°"
        self._cart_label.text = f"  {cp:+.3f} m"

        print(f"[READ] rev1={r1:+.4f} rad ({math.degrees(w1):+.1f}°)  "
              f"rev2={r2:+.4f} rad ({math.degrees(w2):+.1f}°)  cart={cp:+.4f} m")

    def update_labels(self, r1_raw, r2_raw, cart_pos):
        """Update labels from sim loop (non-slider updates)."""
        if self._slider_active:
            self._slider_active = False
            return
        w1 = ((r1_raw + math.pi) % (2 * math.pi) - math.pi)
        w2 = ((r2_raw + math.pi) % (2 * math.pi) - math.pi)
        self._rev1_label.text = f"  {math.degrees(w1):+.1f}°"
        self._rev2_label.text = f"  {math.degrees(w2):+.1f}°"
        self._cart_label.text = f"  {cart_pos:+.3f} m"

    def destroy(self):
        if self.window:
            self.window.destroy()
            self.window = None


### -------------------- MAIN --------------------

def main():
    gravity = (0.0, 0.0, 0.0) if args_cli.no_gravity else (0.0, 0.0, -9.81)
    sim_cfg = SimulationCfg(dt=1.0 / 120.0, gravity=gravity)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[5.0, -5.0, 4.0], target=[0.0, 0.0, 1.5])

    if args_cli.no_gravity:
        print("\n*** GRAVITY DISABLED ***\n")

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
    print(f"\n{keyboard}")
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

    gui = DoublePendulumGUI(robot, slide_idx, rev1_idx, rev2_idx, reset_fn)

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
        arm1_raw = joint_pos[0, rev1_idx[0]].item()
        arm1_vel = joint_vel[0, rev1_idx[0]].item()
        arm2_raw = joint_pos[0, rev2_idx[0]].item()
        arm2_vel = joint_vel[0, rev2_idx[0]].item()

        arm1_wrapped = ((arm1_raw + math.pi) % (2 * math.pi) - math.pi)
        arm2_wrapped = ((arm2_raw + math.pi) % (2 * math.pi) - math.pi)

        # Update GUI labels
        gui.update_labels(arm1_raw, arm2_raw, cart_pos)

        if step % 15 == 0:
            print(
                f"[{step:5d}] "
                f"arm1: raw={arm1_raw:+7.3f} wrap={math.degrees(arm1_wrapped):+7.1f}° vel={arm1_vel:+6.2f} | "
                f"arm2: raw={arm2_raw:+7.3f} wrap={math.degrees(arm2_wrapped):+7.1f}° vel={arm2_vel:+6.2f} | "
                f"cart={cart_pos:+6.2f}m vel={cart_vel:+6.2f} | "
                f"force={force_val:+7.1f}N"
            )
        step += 1

    gui.destroy()


if __name__ == "__main__":
    main()
    simulation_app.close()
