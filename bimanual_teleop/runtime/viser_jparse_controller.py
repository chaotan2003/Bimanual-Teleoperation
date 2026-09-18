from __future__ import annotations

import os
import threading
import time
import types
import webbrowser

import meshcat.transformations as tf
import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

from bimanual_teleop.core.teleop_controller import BaseTeleopController
from bimanual_teleop.filters.pose import PoseFilter, quaternion_slerp
from bimanual_teleop.ik.jparse import stacked_jparse_step
from bimanual_teleop.kinematics_backend import Robot
from bimanual_teleop.retargeting.swgr import R_HEADSET_TO_WORLD

DEFAULT_FREQUENCY = 60.0
DEFAULT_ENGAGE_RAMP_TIME = 0.08
DEFAULT_GAMMA = 0.1
DEFAULT_NULLSPACE_GAIN = 0.5
DEFAULT_MAX_JOINT_VELOCITY = 6.0
DEFAULT_POSITION_GAIN = 8.0
DEFAULT_ORIENTATION_GAIN = 2.0
DEFAULT_POSITION_FILTER_MIN_CUTOFF = 12.0
DEFAULT_POSITION_FILTER_BETA = 20.0
DEFAULT_ORIENTATION_FILTER_ALPHA = 0.98
DEFAULT_ELBOW_GAIN = 0.0
DEFAULT_ELBOW_DEADBAND = 0.04
DEFAULT_ELBOW_WEIGHT_MAX = 1.0
DEFAULT_ELBOW_FILTER_ALPHA = 0.35
DEFAULT_GRIP_ON_THRESHOLD = 0.9
DEFAULT_GRIP_OFF_THRESHOLD = 0.75
DEFAULT_OUTPUT_FREQUENCY = 200.0
DEFAULT_ENABLE_OUTPUT_INTERPOLATION = False
DEFAULT_VISUALIZATION_FREQUENCY = 30.0


def matrix_to_wxyz(T: np.ndarray) -> np.ndarray:
    return np.asarray(tf.quaternion_from_matrix(T), dtype=float)


def wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    return tf.quaternion_matrix(np.asarray(q, dtype=float))[:3, :3]


def blend_se3(T_from: np.ndarray, T_to: np.ndarray, alpha: float) -> np.ndarray:
    T = np.eye(4)
    alpha = min(1.0, max(0.0, float(alpha)))
    T[:3, 3] = (1.0 - alpha) * T_from[:3, 3] + alpha * T_to[:3, 3]
    T[:3, :3] = wxyz_to_matrix(quaternion_slerp(matrix_to_wxyz(T_from), matrix_to_wxyz(T_to), alpha))
    return T


def server_url(server) -> str | None:
    get_url = getattr(server, "get_url", None)
    return get_url() if callable(get_url) else None


class _Task:
    def __init__(self, T_world_frame: np.ndarray):
        self.T_world_frame = np.array(T_world_frame, dtype=float)
        self.target_world = self.T_world_frame[:3, 3].copy()


class _Kinematics:
    def __init__(self, controller: "ViserJparseController"):
        self._controller = controller
        self.state = types.SimpleNamespace(q=controller.q_cmd.copy())
        self._frames = {}
        self._last_q = None
        self.update_kinematics()

    def update_kinematics(self) -> None:
        if self._last_q is not None and np.array_equal(self.state.q, self._last_q):
            return
        poses = np.asarray(self._controller.robot.forward_kinematics(self.state.q))
        self._frames = {}
        for name, idx in self._controller.link_indices.items():
            T = tf.quaternion_matrix(poses[idx, :4])
            T[:3, 3] = poses[idx, 4:]
            self._frames[name] = T
        self._last_q = self.state.q.copy()

    def frame(self, name: str) -> np.ndarray:
        return self._frames[name]


class ViserJparseController(BaseTeleopController):
    def __init__(
        self,
        robot_urdf_path: str,
        manipulator_config: dict,
        q_init,
        q_init_joint_names: list[str] | None = None,
        left_frame: str = "lt",
        right_frame: str = "rt",
        frequency: float = DEFAULT_FREQUENCY,
        engage_ramp_time: float = DEFAULT_ENGAGE_RAMP_TIME,
        gamma: float = DEFAULT_GAMMA,
        nullspace_gain: float = DEFAULT_NULLSPACE_GAIN,
        max_joint_velocity: float = DEFAULT_MAX_JOINT_VELOCITY,
        position_gain: float = DEFAULT_POSITION_GAIN,
        orientation_gain: float = DEFAULT_ORIENTATION_GAIN,
        position_filter_min_cutoff: float = DEFAULT_POSITION_FILTER_MIN_CUTOFF,
        position_filter_beta: float = DEFAULT_POSITION_FILTER_BETA,
        orientation_filter_alpha: float = DEFAULT_ORIENTATION_FILTER_ALPHA,
        elbow_gain: float = DEFAULT_ELBOW_GAIN,
        elbow_deadband: float = DEFAULT_ELBOW_DEADBAND,
        elbow_weight_max: float = DEFAULT_ELBOW_WEIGHT_MAX,
        elbow_filter_alpha: float = DEFAULT_ELBOW_FILTER_ALPHA,
        grip_on_threshold: float = DEFAULT_GRIP_ON_THRESHOLD,
        grip_off_threshold: float = DEFAULT_GRIP_OFF_THRESHOLD,
        output_frequency: float = DEFAULT_OUTPUT_FREQUENCY,
        enable_output_interpolation: bool = DEFAULT_ENABLE_OUTPUT_INTERPOLATION,
        visualization_frequency: float = DEFAULT_VISUALIZATION_FREQUENCY,
        R_headset_world=R_HEADSET_TO_WORLD,
        open_browser: bool = True,
    ):
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.frequency = float(frequency)
        self.engage_ramp_time = float(engage_ramp_time)
        self.gamma = float(gamma)
        self.nullspace_gain = float(nullspace_gain)
        self.max_joint_velocity = float(max_joint_velocity)
        self.position_gain = float(position_gain)
        self.orientation_gain = float(orientation_gain)
        self.position_filter_min_cutoff = float(position_filter_min_cutoff)
        self.position_filter_beta = float(position_filter_beta)
        self.orientation_filter_alpha = float(orientation_filter_alpha)
        self.elbow_gain = float(elbow_gain)
        self.elbow_deadband = float(elbow_deadband)
        self.elbow_weight_max = float(elbow_weight_max)
        self.elbow_filter_alpha = float(elbow_filter_alpha)
        self.grip_on_threshold = float(grip_on_threshold)
        self.grip_off_threshold = float(grip_off_threshold)
        self.output_frequency = float(output_frequency)
        self.enable_output_interpolation = bool(enable_output_interpolation)
        self.visualization_frequency = float(visualization_frequency)
        self.open_browser = open_browser
        self.q_cmd = np.asarray(q_init, dtype=float).ravel().copy()
        self.q_output = self.q_cmd.copy()
        self.q_init_joint_names = q_init_joint_names
        self._frame_to_manipulator = {config["link_name"]: name for name, config in manipulator_config.items()}
        self._engaged = {}
        self._ramp_from = {}
        self._ramp_left = {}
        self._last_targets = {}
        self._last_info = {}
        self._solve_ms = []
        self._loop_ms = []
        self._loop_hz = 0.0
        self._output_hz = 0.0
        self._loop_overruns = 0
        self._output_lock = threading.Lock()
        self._output_stop = threading.Event()
        self._output_thread = None
        now = time.perf_counter()
        self._output_from = self.q_cmd.copy()
        self._output_to = self.q_cmd.copy()
        self._output_t0 = now
        self._output_t1 = now
        self._output_frames = 0
        self._next_viz_t = now
        super().__init__(robot_urdf_path, manipulator_config, R_headset_world, self.q_cmd, 1.0 / self.frequency)
        self._init_viz()
        if self.enable_output_interpolation:
            self._start_output_thread()

    def _robot_setup(self):
        pass

    def _solver_setup(self):
        base_path = os.path.dirname(self.robot_urdf_path)

        def filename_handler(fname: str) -> str:
            return yourdfpy.filename_handler_magic(fname, dir=base_path)

        self.urdf = yourdfpy.URDF.load(self.robot_urdf_path, filename_handler=filename_handler)
        self.robot = Robot.from_urdf(self.urdf)
        if self.q_cmd.shape != (self.robot.joints.num_actuated_joints,):
            raise ValueError(f"q_init has {self.q_cmd.size} values, model has {self.robot.joints.num_actuated_joints}")
        if self.q_init_joint_names is not None:
            source = dict(zip(self.q_init_joint_names, self.q_cmd, strict=True))
            self.q_cmd = np.array([source[name] for name in self.robot.joints.actuated_names], dtype=float)
            self.q_init = self.q_cmd.copy()
            self.q_output = self.q_cmd.copy()
            self._output_from = self.q_cmd.copy()
            self._output_to = self.q_cmd.copy()
        self.link_indices = {name: i for i, name in enumerate(self.robot.links.names)}
        for frame in (self.left_frame, self.right_frame):
            if frame not in self.link_indices:
                raise ValueError(f"frame {frame!r} not found in URDF links")
        self.target_link_indices = [self.link_indices[self.left_frame], self.link_indices[self.right_frame]]
        self.kinematics = _Kinematics(self)
        self.pose_filters = {
            name: PoseFilter(
                self.position_filter_min_cutoff,
                self.position_filter_beta,
                self.orientation_filter_alpha,
            )
            for name in self.manipulator_config
        }
        for name, config in self.manipulator_config.items():
            self.effector_control_mode[name] = config.get("control_mode", "pose")
            self.effector_task[name] = _Task(self.kinematics.frame(config["link_name"]))
            print(f"Created J-PARSE task for {name} -> {config['link_name']}")

    def _update_robot_state(self):
        pass

    def _get_link_pose(self, link_name):
        T = self.kinematics.frame(link_name)
        return T[:3, 3].copy(), tf.quaternion_from_matrix(T)

    def _task_target(self, name: str) -> np.ndarray:
        task = self.effector_task[name]
        if self.effector_control_mode[name] == "position":
            T = self.kinematics.frame(self.manipulator_config[name]["link_name"]).copy()
            T[:3, 3] = task.target_world
            return T
        return np.asarray(task.T_world_frame, dtype=float)

    def _ramp_target(self, name: str, target: np.ndarray) -> np.ndarray:
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
        return blend_se3(self._ramp_from[name], target, alpha)

    def _solve_ik(self):
        target_positions = []
        target_wxyzs = []
        elbow_constraints = []
        now = time.perf_counter()
        for frame in (self.left_frame, self.right_frame):
            name = self._frame_to_manipulator[frame]
            target = self._ramp_target(name, self._task_target(name))
            if self.active.get(name, False):
                target = self.pose_filters[name].next(now, target, matrix_to_wxyz, wxyz_to_matrix)
            else:
                self.pose_filters[name].reset()
            self._last_targets[frame] = target
            target_positions.append(target[:3, 3])
            target_wxyzs.append(matrix_to_wxyz(target))
            direction = self.swgr_elbow_direction.get(name)
            weight = self.swgr_elbow_weight.get(name, 0.0)
            swgr = self.manipulator_config[name]["swgr"]
            elbow_link = swgr.get("elbow_link")
            if self.active.get(name, False) and direction is not None and elbow_link and weight > 0.0:
                elbow_constraints.append(
                    (
                        self.link_indices[swgr["shoulder_link"]],
                        self.link_indices[elbow_link],
                        self.link_indices[frame],
                        direction,
                        weight,
                    )
                )

        t_solve = time.perf_counter()
        q, info = stacked_jparse_step(
            self.robot,
            self.kinematics.state.q,
            self.target_link_indices,
            np.asarray(target_positions),
            np.asarray(target_wxyzs),
            gamma=self.gamma,
            position_gain=self.position_gain,
            orientation_gain=self.orientation_gain,
            nullspace_gain=self.nullspace_gain,
            max_joint_velocity=self.max_joint_velocity,
            dt=self.dt,
            home_cfg=self.q_init,
            elbow_constraints=elbow_constraints,
            elbow_gain=self.elbow_gain,
        )
        self._solve_ms.append((time.perf_counter() - t_solve) * 1e3)
        self._last_info = info
        self.q_cmd = q
        self.kinematics.state.q = q
        self.kinematics.update_kinematics()

    def _init_viz(self):
        self.server = viser.ViserServer()
        self.server.scene.add_grid("/ground", width=2, height=2)
        self.urdf_vis = ViserUrdf(self.server, self.urdf, root_node_name="/base")
        self.target_handles = {}
        for frame in (self.left_frame, self.right_frame):
            T = self.kinematics.frame(frame)
            self.target_handles[frame] = self.server.scene.add_transform_controls(
                f"/target/{frame}",
                scale=0.16,
                position=tuple(T[:3, 3]),
                wxyz=tuple(matrix_to_wxyz(T)),
            )
        self.status_error = self.server.gui.add_number("Max position error (mm)", 0.0, disabled=True)
        self.status_cond = self.server.gui.add_number("Inverse condition", 0.0, disabled=True)
        self.status_ms = self.server.gui.add_number("IK max (ms)", 0.0, disabled=True)
        self.urdf_vis.update_cfg(self.kinematics.state.q)
        url = server_url(self.server)
        if self.open_browser and url is not None:
            webbrowser.open(url)
        print(f"[jparse] viser: {url or 'server started'}")

    def _update_viz(self, update_robot: bool = True):
        if update_robot:
            self.urdf_vis.update_cfg(self.q_output)
        for frame, target in self._last_targets.items():
            handle = self.target_handles[frame]
            handle.position = tuple(target[:3, 3])
            handle.wxyz = tuple(matrix_to_wxyz(target))
        if self._last_info:
            self.status_error.value = round(self._last_info["position_error"] * 1000.0, 3)
            self.status_cond.value = round(self._last_info["inverse_condition_number"], 4)
        if self._solve_ms:
            self.status_ms.value = round(max(self._solve_ms), 3)

    def _output_at_locked(self, now: float) -> np.ndarray:
        if self._output_t1 <= self._output_t0:
            return self._output_to.copy()
        alpha = np.clip((now - self._output_t0) / (self._output_t1 - self._output_t0), 0.0, 1.0)
        return (1.0 - alpha) * self._output_from + alpha * self._output_to

    def _set_output_target(self, q: np.ndarray):
        q = np.asarray(q, dtype=float).ravel().copy()
        if not self.enable_output_interpolation:
            self.q_output = q
            return
        now = time.perf_counter()
        with self._output_lock:
            self._output_from = self._output_at_locked(now)
            self._output_to = q
            self._output_t0 = now
            self._output_t1 = now + self.dt

    def _start_output_thread(self):
        if self.output_frequency <= 0.0:
            raise ValueError("output_frequency must be > 0")
        self._output_thread = threading.Thread(target=self._output_loop, daemon=True)
        self._output_thread.start()

    def _stop_output_thread(self):
        self._output_stop.set()
        if self._output_thread is not None:
            self._output_thread.join(timeout=1.0)

    def _output_loop(self):
        dt = 1.0 / self.output_frequency
        deadline = time.perf_counter()
        while not self._output_stop.is_set():
            deadline += dt
            now = time.perf_counter()
            with self._output_lock:
                self.q_output = self._output_at_locked(now)
                q = self.q_output.copy()
                self._output_frames += 1
            self.urdf_vis.update_cfg(q)
            lag = deadline - time.perf_counter()
            if lag > 0.0:
                self._output_stop.wait(lag)
            else:
                deadline = time.perf_counter()

    def _send_command(self):
        self._set_output_target(self.q_cmd)
        if self.visualization_frequency <= 0.0:
            return
        now = time.perf_counter()
        if now < self._next_viz_t:
            return
        self._next_viz_t = now + 1.0 / self.visualization_frequency
        self._update_viz(update_robot=not self.enable_output_interpolation)

    def run(self):
        status_t0 = time.perf_counter()
        status_frames = 0
        with self._output_lock:
            status_output_frames = self._output_frames
        deadline = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                cycle_t0 = time.perf_counter()
                deadline += self.dt
                self._update_ik()
                self._send_command()
                self._loop_ms.append((time.perf_counter() - cycle_t0) * 1e3)
                status_frames += 1
                now = time.perf_counter()
                if now - status_t0 >= 5.0:
                    elapsed = now - status_t0
                    self._loop_hz = status_frames / elapsed
                    with self._output_lock:
                        output_frames = self._output_frames
                    self._output_hz = (output_frames - status_output_frames) / elapsed
                    status_t0 = time.perf_counter()
                    status_frames = 0
                    status_output_frames = output_frames
                    print(f"[jparse] {self._tracking_report()}")
                lag = deadline - time.perf_counter()
                if lag > 0.0:
                    time.sleep(lag)
                else:
                    self._loop_overruns += 1
                    deadline = time.perf_counter()
        except KeyboardInterrupt:
            print("\nTeleoperation stopped.")
            self._stop_event.set()
        finally:
            self._stop_output_thread()

    def _tracking_report(self) -> str:
        parts = []
        for frame, target in self._last_targets.items():
            err = np.linalg.norm(self.kinematics.frame(frame)[:3, 3] - target[:3, 3])
            parts.append(f"{frame} pos_err={err * 1000:5.1f}mm")
        if self._last_info:
            parts.append(f"cond={self._last_info['inverse_condition_number']:.4f}")
            parts.append(
                f"aeac n={self._last_info.get('elbow_count', 0)} "
                f"w={self._last_info.get('elbow_weight', 0.0):.3f} "
                f"cost={self._last_info.get('elbow_cost', 0.0):.3f}"
            )
            parts.append(
                f"dq={self._last_info.get('max_joint_vel', 0.0):.2f} "
                f"scale={self._last_info.get('velocity_scale', 1.0):.2f}"
            )
        if self._solve_ms:
            a = np.asarray(self._solve_ms)
            parts.append(f"ik mean={a.mean():5.2f}ms max={a.max():5.2f}ms n={len(a)}")
            self._solve_ms.clear()
        if self._loop_ms:
            a = np.asarray(self._loop_ms)
            parts.append(
                f"loop={self._loop_hz:5.1f}Hz mean={a.mean():5.2f}ms max={a.max():5.2f}ms over={self._loop_overruns}"
            )
            self._loop_ms.clear()
            self._loop_overruns = 0
        if self.enable_output_interpolation:
            parts.append(f"out={self._output_hz:5.1f}Hz")
        return " | ".join(parts) if parts else "idle"
