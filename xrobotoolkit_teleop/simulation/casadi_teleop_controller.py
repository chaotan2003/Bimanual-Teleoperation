"""CasADi/IPOPT teleoperation controller for simulation (meshcat visualization only).

The base controller owns the XR input path and the SWGR retargeting, and writes a 4x4
world target per manipulator. This class supplies everything downstream of that
target: the IK solver and the visualizer. No physics backend -- the "robot" is the
kinematic model, so its state is whatever the IK returned.
"""

import gc
import os
import time
import types
import webbrowser
from typing import Sequence

import meshcat.geometry as mgeom
import meshcat.transformations as tf
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.simulation.casadi_arm_ik import (
    FILTER_WEIGHTS,
    REGULARIZATION_COST_WEIGHT,
    SMOOTH_COST_WEIGHT,
    DualArmCasadiIK,
)
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD

# Upstream xr_teleoperate runs its arm IK at ~30 Hz (teleop_hand_and_arm.py), but that
# samples a 72-90 Hz controller stream at 33 ms intervals: the input is stale by up to
# one frame and meshcat visibly steps. The cycle costs far less than the budget, so run
# faster. Measured with meshcat, stub XR, hand at 1200 mm/s: compute p95 3.2 ms against
# an 8.33 ms budget, 0 frames over budget at 120 Hz (and still 0 at 200 Hz).
# Override per-run with --frequency; lower it if the real SDK reads cost more than this.
DEFAULT_FREQUENCY = 120.0
# Seconds spent blending the target in from the current EE pose when grip is pressed.
DEFAULT_ENGAGE_RAMP_TIME = 0.3
TARGET_COLORS = {"left": 0x2196F3, "right": 0xFF5722}


def blend_se3(T_from: np.ndarray, T_to: np.ndarray, alpha: float) -> np.ndarray:
    """Geodesic SE(3) blend: linear position, shortest-arc rotation via exp3/log3."""
    T = np.eye(4)
    T[:3, 3] = (1.0 - alpha) * T_from[:3, 3] + alpha * T_to[:3, 3]
    w = pin.log3(T_from[:3, :3].T @ T_to[:3, :3])
    T[:3, :3] = T_from[:3, :3] @ pin.exp3(np.ascontiguousarray(alpha * w))
    return T


class _Task:
    """Per-manipulator target slot; the base controller writes the retargeted target here."""

    def __init__(self, T_world_frame: np.ndarray):
        self.T_world_frame = np.array(T_world_frame, dtype=float)
        self.target_world = self.T_world_frame[:3, 3].copy()


class _Kinematics:
    """The ``self.kinematics`` handle the base IK loop requires.

    Implements exactly the three members :meth:`BaseTeleopController._update_ik` and
    this controller touch: ``state.q``, ``update_kinematics()`` and ``frame(name)``.
    """

    def __init__(self, ik: DualArmCasadiIK):
        self._ik = ik
        self.state = types.SimpleNamespace(q=ik.q_init.copy())
        self._frames = ik.forward_kinematics(self.state.q)

    def update_kinematics(self) -> None:
        self._frames = self._ik.forward_kinematics(self.state.q)

    def frame(self, name: str) -> np.ndarray:
        return self._frames[name]


class CasadiTeleopController(BaseTeleopController):
    def __init__(
        self,
        robot_urdf_path: str,
        manipulator_config: dict,
        q_init,
        left_frame: str = "lt",
        right_frame: str = "rt",
        smooth_cost_weight: float = SMOOTH_COST_WEIGHT,
        reg_cost_weight: float = REGULARIZATION_COST_WEIGHT,
        filter_weights: Sequence[float] = FILTER_WEIGHTS,
        frequency: float = DEFAULT_FREQUENCY,
        engage_ramp_time: float = DEFAULT_ENGAGE_RAMP_TIME,
        R_headset_world=R_HEADSET_TO_WORLD,
        open_browser: bool = True,
    ):
        # Set before super().__init__(): it calls _solver_setup(), which needs these.
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.smooth_cost_weight = smooth_cost_weight
        self.reg_cost_weight = reg_cost_weight
        # The output FIR is the dominant lag source, not IPOPT: its group delay is
        # sum(k*w[k]) samples, so the tracking error of a hand moving at v is about
        # v*dt*delay. Measured (80 mm sine, 503 mm/s peak): the default 4-tap
        # (0.4,0.3,0.2,0.1) has delay 1.00 -> 2.94 mm at 120 Hz; (0.7,0.3) has delay
        # 0.30 -> 1.09 mm; no filter -> 0.49 mm. Shorter taps trade IK chatter
        # suppression for responsiveness, so tune it against the real robot.
        self.filter_weights = tuple(filter_weights)
        self.frequency = frequency
        self.engage_ramp_time = engage_ramp_time
        self.open_browser = open_browser
        self.q_cmd = np.asarray(q_init, dtype=float).ravel().copy()
        self._frame_to_manipulator = {
            config["link_name"]: name for name, config in manipulator_config.items()
        }
        for frame in (left_frame, right_frame):
            if frame not in self._frame_to_manipulator:
                raise ValueError(
                    f"frame {frame!r} is not any manipulator's link_name "
                    f"(have {sorted(self._frame_to_manipulator)})"
                )
        self._engaged: dict = {}
        self._ramp_from: dict = {}
        self._ramp_left: dict = {}
        self._last_targets: dict = {}
        # Per-cycle IPOPT wall time in ms, drained by _tracking_report every status tick.
        self._solve_ms: list = []

        super().__init__(robot_urdf_path, manipulator_config, R_headset_world, q_init, 1.0 / frequency)
        self._init_viz()

    # ------------------------------------------------------------- solver setup
    def _robot_setup(self):
        pass  # simulation only: no physics backend to connect to

    def _solver_setup(self):
        self.arm_ik = DualArmCasadiIK(
            urdf_path=self.robot_urdf_path,
            left_frame=self.left_frame,
            right_frame=self.right_frame,
            q_init=self.q_init,
            smooth_cost_weight=self.smooth_cost_weight,
            reg_cost_weight=self.reg_cost_weight,
            filter_weights=self.filter_weights,
        )
        self.kinematics = _Kinematics(self.arm_ik)
        print("Joint names in the CasADi/Pinocchio model:")
        for name in self.arm_ik.model.names[1:]:
            print(f"  {name}")

        for name, config in self.manipulator_config.items():
            self.effector_control_mode[name] = config.get("control_mode", "pose")
            self.effector_task[name] = _Task(self.kinematics.frame(config["link_name"]))
            print(f"Created {self.effector_control_mode[name]} task for {name} -> {config['link_name']}")

    # ------------------------------------------------------------------ state
    def _update_robot_state(self):
        pass  # state is whatever the last IK returned; _solve_ik publishes it

    def _get_link_pose(self, link_name):
        T = self.kinematics.frame(link_name)
        return T[:3, 3].copy(), tf.quaternion_from_matrix(T)

    def _solve_ik(self):
        targets = {}
        for frame in (self.left_frame, self.right_frame):
            name = self._frame_to_manipulator[frame]
            target = self._ramp_target(name, self._task_target(name))
            targets[frame] = target
            self._last_targets[frame] = target
        # Time the optimizer call only, so the status line reports IPOPT and not the
        # ramp/FK bookkeeping around it.
        t_solve = time.perf_counter()
        q, _ = self.arm_ik.solve_ik(
            targets[self.left_frame], targets[self.right_frame], q_meas=self.kinematics.state.q
        )
        self._solve_ms.append((time.perf_counter() - t_solve) * 1e3)
        self.q_cmd = q
        self.kinematics.state.q = q
        self.kinematics.update_kinematics()

    # ---------------------------------------------------------- target handling
    def _task_target(self, name: str) -> np.ndarray:
        """The 4x4 target the retargeting layer wrote, in both control modes."""
        task = self.effector_task[name]
        if self.effector_control_mode[name] == "position":
            T = self.kinematics.frame(self.manipulator_config[name]["link_name"]).copy()
            T[:3, 3] = task.target_world  # hold current orientation
            return T
        return np.asarray(task.T_world_frame, dtype=float)

    def _ramp_target(self, name: str, target: np.ndarray) -> np.ndarray:
        """Blend the target in from the current EE pose when grip is (re-)engaged.

        SWGR is absolute and re-anchors at the *startup* EE pose, so pressing grip
        after the arm has drifted commands an instant jump: measured on this robot a
        0.2 m target step moves a joint 44 deg in one IK cycle even with the smooth
        cost. ``engage_ramp_time <= 0`` restores unfiltered upstream behaviour.
        """
        if self.engage_ramp_time <= 0.0:
            return target
        link = self.manipulator_config[name]["link_name"]
        active = bool(self.active[name])
        if active and not self._engaged.get(name, False):
            self._ramp_from[name] = self.kinematics.frame(link).copy()
            self._ramp_left[name] = self.engage_ramp_time
        self._engaged[name] = active
        remaining = self._ramp_left.get(name, 0.0)
        if not active or remaining <= 0.0:
            return target
        alpha = 1.0 - remaining / self.engage_ramp_time
        self._ramp_left[name] = max(0.0, remaining - self.dt)
        return blend_se3(self._ramp_from[name], target, min(1.0, max(0.0, alpha)))

    # ------------------------------------------------------------ visualization
    def _init_viz(self):
        mesh_dir = os.path.dirname(self.robot_urdf_path)
        visual_model = pin.buildGeomFromUrdf(
            self.arm_ik.model, self.robot_urdf_path, pin.GeometryType.VISUAL, mesh_dir
        )
        self.viz = MeshcatVisualizer(self.arm_ik.model, None, visual_model)
        self.viz.initViewer(loadModel=True)
        if self.open_browser:
            webbrowser.open(self.viz.viewer.url())
        for side, frame in (("left", self.left_frame), ("right", self.right_frame)):
            self.viz.viewer[f"target/{frame}"].set_object(
                mgeom.Sphere(0.012),
                mgeom.MeshLambertMaterial(color=TARGET_COLORS[side], opacity=0.85),
            )
        self.viz.display(self.kinematics.state.q)
        print(f"[casadi] meshcat: {self.viz.viewer.url()}")

    def _update_viz(self):
        self.viz.display(self.kinematics.state.q)
        for frame, target in self._last_targets.items():
            self.viz.viewer[f"target/{frame}"].set_transform(target)

    def _send_command(self):
        self._update_viz()

    # ------------------------------------------------------------------- loop
    def run(self):
        # The URDF/pinocchio/casadi/meshcat object graph is fully built by now and never
        # changes, so move it out of the GC's scan set. Measured: a full collection drops
        # from ~1 ms to ~0.001 ms, which removes a periodic hitch from the hot loop.
        gc.collect()
        gc.freeze()
        status_t0 = time.perf_counter()
        deadline = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                # Absolute deadlines, not a sleep relative to this cycle's start: a
                # relative sleep accumulates its own overshoot into the period, and
                # time.time() is not monotonic (an NTP step shifts the loop rate).
                deadline += self.dt
                self._update_ik()
                self._send_command()
                if time.perf_counter() - status_t0 >= 5.0:
                    status_t0 = time.perf_counter()
                    print(f"[casadi] {self._tracking_report()}")
                lag = deadline - time.perf_counter()
                if lag > 0.0:
                    time.sleep(lag)
                else:
                    deadline = time.perf_counter()  # fell behind: resync, do not burst
            except KeyboardInterrupt:
                print("\nTeleoperation stopped.")
                self._stop_event.set()

    def _tracking_report(self) -> str:
        parts = []
        for frame, target in self._last_targets.items():
            name = self._frame_to_manipulator[frame]
            err = np.linalg.norm(self.kinematics.frame(frame)[:3, 3] - target[:3, 3])
            parts.append(f"{frame} pos_err={err * 1000:5.1f}mm engaged={self._engaged.get(name, False)}")
        if self._solve_ms:
            # mean and max over the status window, not the last cycle: a single sample
            # hides the GC/scheduler spike that max is here to catch.
            a = np.asarray(self._solve_ms)
            parts.append(f"ik mean={a.mean():5.2f}ms max={a.max():5.2f}ms n={len(a)}")
            self._solve_ms.clear()
        return " | ".join(parts) if parts else "idle"
