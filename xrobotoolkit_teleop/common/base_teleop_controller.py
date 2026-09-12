import abc
import threading
import time
from typing import Any, Dict

import meshcat.transformations as tf
import numpy as np

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import swgr_ee_position


# XRoboToolkit body tracking joint indices (SMPL-like, 24 joints).
# Only the upper-limb keypoints are used; SWGR end-effector retargeting consumes
# shoulder + wrist, and the elbow is captured but deliberately unused.
BODY_JOINT_INDEX = {
    "left": {"shoulder": 16, "elbow": 18, "wrist": 20},
    "right": {"shoulder": 17, "elbow": 19, "wrist": 21},
}


class BaseTeleopController(abc.ABC):
    """XR input -> SWGR retargeting -> one 4x4 world target per manipulator.

    Backend-agnostic. A subclass supplies three things: the kinematics handle
    ``self.kinematics`` (``state.q``, ``update_kinematics()``, ``frame(name)``), the
    per-manipulator target objects in ``self.effector_task``, and ``_solve_ik``.
    See ``simulation/casadi_teleop_controller.py``.
    """

    def __init__(
        self,
        robot_urdf_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        R_headset_world: np.ndarray,
        q_init: np.ndarray,
        dt: float,
    ):
        self.robot_urdf_path = robot_urdf_path
        self.manipulator_config = manipulator_config
        self.R_headset_world = R_headset_world
        self.q_init = q_init
        self.dt = dt
        self.xr_client = XrClient()

        # Engagement latch: None means grip is released, so the next press re-announces.
        self.ref_ee_xyz = {name: None for name in manipulator_config}
        self.effector_task = {}
        self.effector_control_mode = {
            name: config.get("control_mode", "pose") for name, config in manipulator_config.items()
        }
        self.active = {}

        # Shoulder-Wrist Geometric Retargeting (SWGR): the only retargeting path.
        missing = [name for name, config in manipulator_config.items() if "swgr" not in config]
        if missing:
            raise ValueError(f"manipulator_config entries without an 'swgr' block: {missing}")
        self.swgr_scale = {}  # robot_reach / human_reach per end effector
        self.swgr_rot_offset = {}  # constant controller->tool rotation offset per end effector
        self.swgr_elbow = {}  # retargeted elbow position, collected but not used by IK
        self._body_sanity_checked = False
        self._body_warn_t0 = 0.0  # throttles the "body tracking unavailable" warning
        for name, config in manipulator_config.items():
            swgr = config["swgr"]
            self.swgr_scale[name] = swgr["robot_reach"] / swgr["human_reach"]
            rx, ry, rz = np.radians(swgr.get("ee_rot_offset", [0.0, 0.0, 0.0]))
            self.swgr_rot_offset[name] = tf.euler_matrix(rx, ry, rz, "sxyz")[:3, :3]
            print(
                f"SWGR enabled for {name}: shoulder={swgr['shoulder_link']} "
                f"scale={self.swgr_scale[name]:.4f} "
                f"(robot_reach={swgr['robot_reach']} m / human_reach={swgr['human_reach']} m)"
            )
        self.use_swgr = True

        self._stop_event = threading.Event()

        self._robot_setup()
        self._solver_setup()

    def _read_body_upper_limbs(self):
        """Reads operator shoulder/elbow/wrist positions from XR body tracking.

        Returns:
            Dict mapping side ("left"/"right") to {"shoulder", "elbow", "wrist"} positions
            in the VR world frame, or None when body tracking is unavailable.
        """
        body = self.xr_client.get_body_tracking_data()
        if body is None:
            return None

        poses = np.asarray(body["poses"], dtype=float)
        limbs = {side: {k: poses[i, :3] for k, i in idx.items()} for side, idx in BODY_JOINT_INDEX.items()}

        # One-shot sanity check: guards against a wrong frame convention or unit assumption.
        if not self._body_sanity_checked:
            self._body_sanity_checked = True
            for side, limb in limbs.items():
                print(
                    f"[SWGR] {side} |shoulder->wrist|={np.linalg.norm(limb['wrist'] - limb['shoulder']):.3f} m "
                    f"(must be <= human_reach)"
                )
            print(
                f"[SWGR] shoulder width={np.linalg.norm(limbs['right']['shoulder'] - limbs['left']['shoulder']):.3f} m "
                f"(expect ~0.35 m; if not, the body frame needs an extra transform)"
            )

        return limbs

    def _update_swgr_target(self, src_name, config, limbs):
        """Sets the end-effector task target by absolute shoulder-wrist geometric retargeting.

        Position comes from the operator's shoulder->wrist vector scaled by robot_reach /
        human_reach and anchored at the robot shoulder. Orientation comes from the
        controller pose mapped absolutely into the robot world frame. Neither uses a
        reference captured at engagement, so there is no incremental drift.
        """
        swgr = config["swgr"]
        side = "left" if config["pose_source"].startswith("left") else "right"
        limb = limbs[side]

        shoulder_robot = np.array(self._get_link_pose(swgr["shoulder_link"])[0], dtype=float)
        target_xyz = swgr_ee_position(
            shoulder_robot,
            limb["shoulder"],
            limb["wrist"],
            self.swgr_scale[src_name],
            self.R_headset_world,
        )

        xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
        controller_quat = [xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]]  # (w, x, y, z)
        R_target = self.R_headset_world @ tf.quaternion_matrix(controller_quat)[:3, :3] @ self.swgr_rot_offset[src_name]

        # Elbow is retargeted with the same geometry for logging only; SWGR deliberately
        # does not feed it to the IK, so the shoulder-wrist relation stays the sole driver.
        self.swgr_elbow[src_name] = swgr_ee_position(
            shoulder_robot,
            limb["shoulder"],
            limb["elbow"],
            self.swgr_scale[src_name],
            self.R_headset_world,
        )

        if self.effector_control_mode[src_name] == "position":
            self.effector_task[src_name].target_world = target_xyz
        else:
            target_pose = np.eye(4)
            target_pose[:3, :3] = R_target
            target_pose[:3, 3] = target_xyz
            self.effector_task[src_name].T_world_frame = target_pose

    def _update_ik(self):
        """Core per-cycle block: read the robot state, retarget from XR, solve IK."""
        self._update_robot_state()
        self.kinematics.update_kinematics()

        # When body tracking drops out, hold the last target rather than freezing mid-loop.
        body_limbs = self._read_body_upper_limbs()

        for src_name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            self.active[src_name] = xr_grip_val > 0.9

            if self.active[src_name]:
                if self.ref_ee_xyz[src_name] is None:
                    print(f"{src_name} is activated.")
                    self.ref_ee_xyz[src_name] = self._get_link_pose(config["link_name"])[0]

                if body_limbs is None:
                    # Without body data the arm silently holds its last target; say so loudly.
                    if time.time() - self._body_warn_t0 > 10.0:
                        self._body_warn_t0 = time.time()
                        print(
                            f"[SWGR] body tracking unavailable, {src_name} holds its last target. "
                            "Check: PICO headset connected, Full Body Tracking mode enabled in the "
                            "Unity app, at least two Pico Swift trackers connected and calibrated. "
                            "Probe with dependencies/XRoboToolkit-PC-Service-Pybind/examples/"
                            "example_body_tracking.py"
                        )
                else:
                    self._update_swgr_target(src_name, config, body_limbs)
            elif self.ref_ee_xyz[src_name] is not None:
                print(f"{src_name} is deactivated.")
                self.ref_ee_xyz[src_name] = None

        try:
            self._solve_ik()
        except RuntimeError as e:
            print(f"IK solver failed: {e}")

    # ---------------------------------------------------------
    # --- Abstract Methods (to be implemented by subclasses) ---
    # ---------------------------------------------------------

    @abc.abstractmethod
    def _robot_setup(self):
        """Initializes the specific backend (connects to robot, starts sim, etc.)."""
        raise NotImplementedError

    @abc.abstractmethod
    def _solver_setup(self):
        """Builds the IK solver and ``self.effector_task``.

        Must also leave a ``self.kinematics`` exposing ``state.q``,
        ``update_kinematics()`` and ``frame(name)``, because ``_update_ik`` reads all
        three.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _solve_ik(self):
        """Solves one IK step against the current task targets."""
        raise NotImplementedError

    @abc.abstractmethod
    def _update_robot_state(self):
        """Reads the current joint states from the robot/sim into self.kinematics.state.q."""
        raise NotImplementedError

    @abc.abstractmethod
    def _send_command(self):
        """Sends the calculated target joint positions from self.kinematics.state.q out."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_link_pose(self, link_name):
        """Gets the current world pose for a given link name."""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        """
        The main entry point. Subclasses must implement this to define their
        execution model (single-threaded or multi-threaded).
        """
        raise NotImplementedError
