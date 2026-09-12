import abc
import threading
import time
import webbrowser
from typing import Any, Dict

import meshcat.transformations as tf
import numpy as np

from xrobotoolkit_teleop.common.data_logger import DataLogger
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import (
    apply_delta_pose,
    quat_diff_as_angle_axis,
    swgr_ee_position,
)
from xrobotoolkit_teleop.utils.parallel_gripper_utils import (
    calc_parallel_gripper_position,
)


# XRoboToolkit body tracking joint indices (SMPL-like, 24 joints).
# Only the upper-limb keypoints are used; SWGR end-effector retargeting consumes
# shoulder + wrist, and the elbow is captured but deliberately unused.
BODY_JOINT_INDEX = {
    "left": {"shoulder": 16, "elbow": 18, "wrist": 20},
    "right": {"shoulder": 17, "elbow": 19, "wrist": 21},
}

class BaseTeleopController(abc.ABC):
    def __init__(
        self,
        robot_urdf_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        floating_base: bool,
        R_headset_world: np.ndarray,
        scale_factor: float,
        q_init: np.ndarray,
        dt: float,
        enable_log_data: bool = False,
        log_dir: str = "logs",
        log_freq: float = 50,
    ):
        self.robot_urdf_path = robot_urdf_path
        self.manipulator_config = manipulator_config
        self.floating_base = floating_base
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.q_init = q_init
        self.dt = dt
        self.xr_client = XrClient()

        self.enable_log_data = enable_log_data
        self.log_dir = log_dir
        self.log_freq = log_freq
        if enable_log_data:
            self.data_logger = DataLogger(log_dir=log_dir)

        # Initial poses
        self.ref_ee_xyz = {name: None for name in manipulator_config.keys()}
        self.ref_ee_quat = {name: None for name in manipulator_config.keys()}
        self.ref_controller_xyz = {name: None for name in manipulator_config.keys()}
        self.ref_controller_quat = {name: None for name in manipulator_config.keys()}
        self.effector_task = {}
        self.effector_control_mode = {}  # Store control mode for each end effector
        self.active = {}
        self.gripper_pos_target = {}

        # Shoulder-Wrist Geometric Retargeting (SWGR) support
        self.swgr_scale = {}  # robot_reach / human_reach per end effector
        self.swgr_rot_offset = {}  # constant controller->tool rotation offset per end effector
        self.swgr_elbow = {}  # retargeted elbow position, collected but not used by IK
        self._body_sanity_checked = False
        self._body_warn_t0 = 0.0  # throttles the "body tracking unavailable" warning
        for name, config in self.manipulator_config.items():
            if "swgr" not in config:
                continue
            swgr = config["swgr"]
            self.swgr_scale[name] = swgr["robot_reach"] / swgr["human_reach"]
            rx, ry, rz = np.radians(swgr.get("ee_rot_offset", [0.0, 0.0, 0.0]))
            self.swgr_rot_offset[name] = tf.euler_matrix(rx, ry, rz, "sxyz")[:3, :3]
            print(
                f"SWGR enabled for {name}: shoulder={swgr['shoulder_link']} "
                f"scale={self.swgr_scale[name]:.4f} "
                f"(robot_reach={swgr['robot_reach']} m / human_reach={swgr['human_reach']} m)"
            )
        self.use_swgr = bool(self.swgr_scale)

        # Motion tracker support
        self.motion_tracker_task = {}
        # Cached so _update_motion_tracker_tasks can skip the SDK round-trip entirely
        # when no manipulator uses a tracker (e.g. the JAKA K1 SWGR config).
        self.use_motion_tracker = any("motion_tracker" in c for c in manipulator_config.values())
        self.ref_tracker_xyz = {}  # Store initial tracker positions
        self.ref_robot_xyz = {}  # Store initial robot end-effector positions
        for name, config in self.manipulator_config.items():
            if "gripper_config" in config:
                gripper_config = config["gripper_config"]
                self.gripper_pos_target[name] = {
                    joint_name: joint_pos
                    for joint_name, joint_pos in zip(gripper_config["joint_names"], gripper_config["open_pos"])
                }

        self._stop_event = threading.Event()

        self._robot_setup()
        self._solver_setup()

    def _process_xr_pose(self, xr_pose, src_name):
        """Process the current XR controller pose."""
        # Get position and orientation
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = [
            xr_pose[6],  # w
            xr_pose[3],  # x
            xr_pose[4],  # y
            xr_pose[5],  # z
        ]

        controller_xyz = self.R_headset_world @ controller_xyz

        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat),
            tf.quaternion_conjugate(R_quat),
        )

        if self.ref_controller_xyz[src_name] is None:
            self.ref_controller_xyz[src_name] = controller_xyz
            self.ref_controller_quat[src_name] = controller_quat

            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])
        else:
            delta_xyz = (controller_xyz - self.ref_controller_xyz[src_name]) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(self.ref_controller_quat[src_name], controller_quat)

        return delta_xyz, delta_rot

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

    def _solver_setup(self):
        """Build the IK solver, ``self.effector_task`` and the kinematics handle.

        Defaults to placo. Override to plug in a different solver (see
        ``simulation/casadi_teleop_controller.py``); the override must still leave a
        ``self.placo_robot`` exposing ``state.q`` and ``update_kinematics()``, because
        ``_update_ik`` reads both.
        """
        self._placo_setup()

    def _solve_ik(self):
        """Solve one IK step against the current task targets. Override together with ``_solver_setup``."""
        self.solver.solve(True)

    def _placo_setup(self):
        """Set up the placo inverse kinematics solver."""
        import placo

        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        print("Joint names in the Placo model:")
        for joint_name in self.placo_robot.model.names:
            print(f"  {joint_name}")

        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = self.dt
        # self.solver.add_kinetic_energy_regularization_task(1e-6)

        # Set initial configuration
        if self.q_init is not None:
            if self.floating_base:
                self.placo_robot.state.q = self.q_init.copy()
            else:
                self.solver.mask_fbase(True)
                self.placo_robot.state.q[7:] = self.q_init.copy()
        else:
            if not self.floating_base:
                self.solver.mask_fbase(True)
            self.placo_robot.state.q[:7] = np.array([0, 0, 0, 0, 0, 0, 1])  # Identity quaternion for base

        self.placo_robot.update_kinematics()

        # Set up end effector tasks
        for name, config in self.manipulator_config.items():
            # Get control mode (default to "pose" for backward compatibility)
            control_mode = config.get("control_mode", "pose")
            self.effector_control_mode[name] = control_mode
            
            ee_xyz, ee_quat = self._get_link_pose(config["link_name"])
            
            if control_mode == "position":
                # Position-only control
                self.effector_task[name] = self.solver.add_position_task(config["link_name"], ee_xyz)
                print(f"Created position task for {name} -> {config['link_name']}")
            else:
                # Full pose control (default)
                ee_target = tf.quaternion_matrix(ee_quat)
                ee_target[:3, 3] = ee_xyz
                self.effector_task[name] = self.solver.add_frame_task(config["link_name"], ee_target)
                print(f"Created pose task for {name} -> {config['link_name']}")
            
            self.effector_task[name].configure(name, "soft", 1.0)
            manipulability = self.solver.add_manipulability_task(config["link_name"], "both", 1.0)
            manipulability.configure("manipulability", "soft", 1e-2)

            # Set up motion tracker tasks if configured (position only)
            if "motion_tracker" in config:
                tracker_config = config["motion_tracker"]
                link_target = tracker_config["link_target"]

                # Get current position of the target link
                target_xyz, _ = self._get_link_pose(link_target)

                # Create position task for motion tracker target (xyz only)
                tracker_task_name = f"{name}_tracker"
                self.motion_tracker_task[name] = self.solver.add_position_task(link_target, target_xyz)
                self.motion_tracker_task[name].configure(tracker_task_name, "soft", 1.0)

                print(f"Motion tracker position task created for {name} -> {link_target}")

        self.placo_robot.update_kinematics()

    def _update_ik(self):
        """
        This is the core IK logic block. It reads from XR, updates Placo tasks,
        and solves the kinematics.
        """
        self._update_robot_state()
        self.placo_robot.update_kinematics()

        body_limbs = self._read_body_upper_limbs() if self.use_swgr else None

        for src_name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            self.active[src_name] = xr_grip_val > 0.9

            if self.active[src_name]:
                if self.ref_ee_xyz[src_name] is None:
                    print(f"{src_name} is activated.")
                    self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name] = self._get_link_pose(config["link_name"])

                # SWGR: absolute retargeting. When body tracking drops out, hold the last
                # target rather than falling back to a different mapping mid-motion.
                if "swgr" in config:
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
                    continue

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot = self._process_xr_pose(xr_pose, src_name)
                
                if self.effector_control_mode[src_name] == "position":
                    # Position-only control: only apply position delta
                    target_xyz = self.ref_ee_xyz[src_name] + delta_xyz
                    self.effector_task[src_name].target_world = target_xyz
                else:
                    # Full pose control: apply both position and orientation deltas
                    target_xyz, target_quat = apply_delta_pose(
                        self.ref_ee_xyz[src_name],
                        self.ref_ee_quat[src_name],
                        delta_xyz,
                        delta_rot,
                    )
                    target_pose = tf.quaternion_matrix(target_quat)
                    target_pose[:3, 3] = target_xyz
                    self.effector_task[src_name].T_world_frame = target_pose
            else:
                if self.ref_ee_xyz[src_name] is not None:
                    print(f"{src_name} is deactivated.")
                    self.ref_ee_xyz[src_name] = None
                    self.ref_controller_xyz[src_name] = None

        # Process motion tracker data
        self._update_motion_tracker_tasks()

        try:
            self._solve_ik()
        except RuntimeError as e:
            print(f"IK solver failed: {e}")

    def _update_motion_tracker_tasks(self):
        """Process motion tracker data and update corresponding Placo tasks."""
        if not self.use_motion_tracker:
            return  # no manipulator is tracker-driven; skip the per-frame SDK call
        motion_tracker_data = self.xr_client.get_motion_tracker_data()

        for src_name, config in self.manipulator_config.items():
            # Skip if no motion tracker configured for this end effector
            if "motion_tracker" not in config:
                continue

            # Skip if main controller is not active
            if not self.active.get(src_name, False):
                # Reset motion tracker references when controller is inactive
                if src_name in self.ref_tracker_xyz:
                    del self.ref_tracker_xyz[src_name]
                    del self.ref_robot_xyz[src_name]
                continue

            tracker_config = config["motion_tracker"]
            serial = tracker_config["serial"]

            # Skip if this tracker is not available
            if serial not in motion_tracker_data:
                continue

            # Get motion tracker pose
            tracker_pose = motion_tracker_data[serial]["pose"]
            tracker_xyz = self.R_headset_world @ np.array(tracker_pose[:3])

            # Initialize reference positions on first detection
            if src_name not in self.ref_tracker_xyz:
                self.ref_tracker_xyz[src_name] = tracker_xyz.copy()
                # Get current robot end-effector position as baseline
                robot_xyz, _ = self._get_link_pose(config["motion_tracker"]["link_target"])
                self.ref_robot_xyz[src_name] = robot_xyz.copy()
                continue

            # Calculate movement delta from tracker's initial position
            tracker_delta = tracker_xyz - self.ref_tracker_xyz[src_name]

            # Apply scaled tracker movement to robot's initial position
            final_target_xyz = self.ref_robot_xyz[src_name] + tracker_delta * self.scale_factor

            # Update motion tracker task target position
            if src_name in self.motion_tracker_task:
                self.motion_tracker_task[src_name].target_world = final_target_xyz

    def _init_placo_viz(self):
        from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz

        self.placo_vis = robot_viz(self.placo_robot)
        webbrowser.open(self.placo_vis.viewer.url())
        self.placo_vis.display(self.placo_robot.state.q)
        for name, config in self.manipulator_config.items():
            robot_frame_viz(self.placo_robot, config["link_name"])
            
            # Show appropriate visualization based on control mode
            if self.effector_control_mode[name] == "position":
                # Create a frame matrix for position-only visualization
                target_frame = np.eye(4)
                target_frame[:3, 3] = self.effector_task[name].target_world
                frame_viz(f"vis_target_{name}", target_frame)
            else:
                # Full pose visualization
                frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

            # Visualize motion tracker target if configured
            if "motion_tracker" in config and name in self.motion_tracker_task:
                link_target = config["motion_tracker"]["link_target"]
                robot_frame_viz(self.placo_robot, link_target)
                # Create a frame matrix for visualization
                tracker_frame = np.eye(4)
                tracker_frame[:3, 3] = self.motion_tracker_task[name].target_world
                frame_viz(f"vis_tracker_{name}", tracker_frame)

    def _update_placo_viz(self):
        from placo_utils.visualization import frame_viz, robot_frame_viz

        self.placo_vis.display(self.placo_robot.state.q)
        for name, config in self.manipulator_config.items():
            robot_frame_viz(self.placo_robot, config["link_name"])
            
            # Show appropriate visualization based on control mode
            if self.effector_control_mode[name] == "position":
                # Create a frame matrix for position-only visualization
                target_frame = np.eye(4)
                target_frame[:3, 3] = self.effector_task[name].target_world
                frame_viz(f"vis_target_{name}", target_frame)
            else:
                # Full pose visualization
                frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

            # Update motion tracker target visualization if configured
            if "motion_tracker" in config and name in self.motion_tracker_task:
                link_target = config["motion_tracker"]["link_target"]
                robot_frame_viz(self.placo_robot, link_target)
                # Create a frame matrix for visualization
                tracker_frame = np.eye(4)
                tracker_frame[:3, 3] = self.motion_tracker_task[name].target_world
                frame_viz(f"vis_tracker_{name}", tracker_frame)

    def sync_end_effector_poses_to_placo_tasks(self):
        """
        Syncs the current end effector link poses to their corresponding placo tasks.
        This is useful for initializing or resetting task targets to current robot state.
        """
        for name, config in self.manipulator_config.items():
            # Get current link pose
            ee_xyz, ee_quat = self._get_link_pose(config["link_name"])
            
            # Update the corresponding placo task
            if self.effector_control_mode[name] == "position":
                # Position-only control: update target position
                self.effector_task[name].target_world = ee_xyz
            else:
                # Full pose control: update target pose
                ee_target = tf.quaternion_matrix(ee_quat)
                ee_target[:3, 3] = ee_xyz
                self.effector_task[name].T_world_frame = ee_target
            
            print(f"Synced {name} end effector pose to placo task: {config['link_name']}")

    def _update_gripper_target(self):
        for gripper_name in self.manipulator_config.keys():
            if "gripper_config" not in self.manipulator_config[gripper_name]:
                continue

            gripper_config = self.manipulator_config[gripper_name]["gripper_config"]
            gripper_config = self.manipulator_config[gripper_name]["gripper_config"]
            gripper_type = gripper_config["type"]
            if gripper_type == "parallel":
                trigger_value = self.xr_client.get_key_value_by_name(gripper_config["gripper_trigger"])
                for joint_name, open_pos, close_pos in zip(
                    gripper_config["joint_names"],
                    gripper_config["open_pos"],
                    gripper_config["close_pos"],
                ):
                    # Calculate the target position based on the trigger value
                    gripper_pos = calc_parallel_gripper_position(open_pos, close_pos, trigger_value)
                    self.gripper_pos_target[gripper_name][joint_name] = gripper_pos
                    self.gripper_pos_target[gripper_name][joint_name] = gripper_pos
            else:
                # TODO: add dexterous hand support
                raise ValueError(f"Unsupported gripper type: {gripper_type}")

    def _log_data(self):
        """
        Logs the current state of the robot, including joint positions, end effector poses,
        and any other relevant data
        """
        if self.enable_log_data:
            raise NotImplementedError

    # ---------------------------------------------------------
    # --- Abstract Methods (to be implemented by subclasses) ---
    # ---------------------------------------------------------

    @abc.abstractmethod
    def _robot_setup(self):
        """Initializes the specific backend (connects to robot, starts sim, etc.)."""
        raise NotImplementedError

    @abc.abstractmethod
    def _update_robot_state(self):
        """Reads the current joint states from the robot/sim and updates self.placo_robot.state.q."""
        raise NotImplementedError

    @abc.abstractmethod
    def _send_command(self):
        """Sends the calculated target joint positions from self.placo_robot.state.q to the robot/sim."""
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
