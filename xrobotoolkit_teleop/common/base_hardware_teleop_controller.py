import threading
import time
from abc import ABC, abstractmethod
from typing import Dict

import cv2
import meshcat.transformations as tf
import numpy as np

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.hardware.interface.base_camera import BaseCameraInterface


class HardwareTeleopController(BaseTeleopController, ABC):
    """
    An abstract base class for hardware teleoperation controllers that consolidates
    common logic for threading, logging, and visualization.
    """

    def __init__(
        self,
        robot_urdf_path: str,
        manipulator_config: dict,
        R_headset_world: np.ndarray,
        floating_base: bool,
        scale_factor: float,
        visualize_placo: bool,
        control_rate_hz: int,
        enable_log_data: bool,
        log_dir: str,
        log_freq: float,
        enable_camera: bool,
        camera_fps: int,
        **kwargs,
    ):
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            floating_base=floating_base,
            R_headset_world=R_headset_world,
            scale_factor=scale_factor,
            q_init=kwargs.get("q_init"),
            dt=1.0 / control_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
        )

        self._start_time = 0
        self.control_rate_hz = control_rate_hz
        self.log_freq = log_freq
        self.visualize_placo = visualize_placo
        self.enable_camera = enable_camera
        self.camera_interface: BaseCameraInterface = None
        self.camera_fps = camera_fps

        if self.visualize_placo:
            self._init_placo_viz()

        self._prev_b_button_state = False
        self._is_logging = False

    @abstractmethod
    def _robot_setup():
        """Initializes hardware-specific interfaces (e.g., CAN, ROS)."""
        pass

    @abstractmethod
    def _initialize_camera():
        """Initializes the specific camera interface."""
        pass

    @abstractmethod
    def _update_robot_state():
        """Reads the current robot state from hardware and updates Placo."""
        pass

    @abstractmethod
    def _send_command():
        """Sends motor commands to the hardware."""
        pass

    @abstractmethod
    def _get_robot_state_for_logging() -> Dict:
        """Returns a dictionary of robot-specific data for logging."""
        pass

    @abstractmethod
    def _shutdown_robot():
        """Performs graceful shutdown of the robot hardware."""
        pass

    @abstractmethod
    def _get_camera_frame_for_logging(self) -> Dict:
        """Returns a dictionary of camera frames for logging with camera names as keys."""
        pass

    def _get_link_pose(self, link_name: str):
        """Gets the current world pose for a given link name from Placo."""
        T_world_link = self.placo_robot.get_T_world_frame(link_name)
        pos = T_world_link[:3, 3]
        quat = tf.quaternion_from_matrix(T_world_link)
        return pos, quat

    def _log_data(self):
        """Logs the current state of the robot and camera."""
        if not self.enable_log_data:
            return

        timestamp = time.time() - self._start_time
        data_entry = {"timestamp": timestamp}
        data_entry.update(self._get_robot_state_for_logging())

        if self.enable_camera and self.camera_interface:
            frames = self._get_camera_frame_for_logging()
            if frames:
                data_entry["image"] = frames

        self.data_logger.add_entry(data_entry)

    def _pre_ik_update(self):
        """Hook for subclasses to run logic before the main IK update."""
        pass

    def _ik_thread(self, stop_event: threading.Event):
        """Dedicated thread for running the IK solver."""
        while not stop_event.is_set():
            start_time = time.time()
            self._update_robot_state()
            self._update_gripper_target()
            self._pre_ik_update()
            self._update_ik()
            if self.visualize_placo:
                self._update_placo_viz()
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        print("IK loop has stopped.")

    def _control_thread(self, stop_event: threading.Event):
        """Dedicated thread for sending commands to hardware."""
        while not stop_event.is_set():
            start_time = time.time()
            self._send_command()
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._shutdown_robot()
        print("Control loop has stopped.")

    def _data_logging_thread(self, stop_event: threading.Event):
        """Dedicated thread for data logging."""
        while not stop_event.is_set():
            start_time = time.time()
            self._check_logging_button()
            if self._is_logging:
                self._log_data()
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.log_freq) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        print("Data logging thread has stopped.")

    def _check_logging_button(self):
        """Checks for the 'B' button press to toggle data logging."""
        b_button_state = self.xr_client.get_button_state_by_name("B")
        right_axis_click = self.xr_client.get_button_state_by_name("right_axis_click")

        if b_button_state and not self._prev_b_button_state:
            self._is_logging = not self._is_logging
            if self._is_logging:
                print("--- Started data logging ---")
            else:
                print("--- Stopped data logging. Saving data... ---")
                self.data_logger.save()
                self.data_logger.reset()

        if right_axis_click and self._is_logging:
            print("--- Stopped data logging. Discarding data... ---")
            self.data_logger.reset()
            self._is_logging = False

        self._prev_b_button_state = b_button_state

    def _camera_thread(self, stop_event: threading.Event):
        """Dedicated thread for managing the camera lifecycle and streaming."""
        if not self.camera_interface:
            return

        print("Camera thread started.")
        window_name = "Hardware Cameras"
        window_created = False

        try:
            while not stop_event.is_set():
                self.camera_interface.update_frames()
                if self._is_logging:
                    if not window_created:
                        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                        window_created = True

                    frames_dict = self.camera_interface.get_frames()
                    if not frames_dict:
                        time.sleep(1.0 / self.camera_fps)
                        continue

                    all_camera_rows = []
                    sorted_camera_names = sorted(frames_dict.keys())

                    for name in sorted_camera_names:
                        frames = frames_dict[name]
                        images_in_row = []

                        color_image = frames.get("color")
                        if color_image is not None:
                            if len(color_image.shape) == 2:
                                color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2BGR)
                            images_in_row.append(color_image)

                        depth_image = frames.get("depth")
                        if depth_image is not None:
                            depth_colormap = cv2.applyColorMap(
                                cv2.convertScaleAbs(depth_image, alpha=0.03),
                                cv2.COLORMAP_JET,
                            )
                            images_in_row.append(depth_colormap)

                        if images_in_row:
                            all_camera_rows.append(np.hstack(images_in_row))

                    if all_camera_rows:
                        max_width = max(row.shape[1] for row in all_camera_rows)
                        padded_rows = [
                            (
                                np.hstack(
                                    [
                                        row,
                                        np.zeros(
                                            (row.shape[0], max_width - row.shape[1], 3),
                                            dtype=np.uint8,
                                        ),
                                    ]
                                )
                                if row.shape[1] < max_width
                                else row
                            )
                            for row in all_camera_rows
                        ]
                        if padded_rows:
                            combined_image = np.vstack(padded_rows)
                            cv2.imshow(
                                window_name,
                                cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR),
                            )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    if window_created:
                        cv2.destroyWindow(window_name)
                        window_created = False
                    time.sleep(1.0 / self.camera_fps)
        finally:
            if self.camera_interface:
                self.camera_interface.stop()
            if window_created:
                cv2.destroyAllWindows()
            print("Camera thread has stopped.")

    def _should_keep_running(self) -> bool:
        """Returns True if the main loop should continue running."""
        return not self._stop_event.is_set()

    def run(self):
        """Main entry point that starts all threads."""
        self._robot_setup()
        self._initialize_camera()

        self._start_time = time.time()
        self._stop_event = threading.Event()
        threads = []

        core_threads = {
            "_ik_thread": self._ik_thread,
            "_control_thread": self._control_thread,
        }
        for name, target in core_threads.items():
            thread = threading.Thread(name=name, target=target, args=(self._stop_event,))
            threads.append(thread)

        if self.enable_log_data:
            log_thread = threading.Thread(
                name="_data_logging_thread",
                target=self._data_logging_thread,
                args=(self._stop_event,),
            )
            threads.append(log_thread)
        if self.enable_camera and self.camera_interface:
            camera_thread = threading.Thread(
                name="_camera_thread",
                target=self._camera_thread,
                args=(self._stop_event,),
            )
            threads.append(camera_thread)

        for t in threads:
            t.daemon = True
            t.start()

        print("Teleoperation running. Press Ctrl+C to exit.")
        try:
            while self._should_keep_running():
                all_threads_alive = all(t.is_alive() for t in threads)
                if not all_threads_alive:
                    print("A thread has died. Shutting down.")
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
        finally:
            print("Shutting down...")
            self._stop_event.set()
            for t in threads:
                t.join(timeout=2.0)
            print("All threads have been shut down.")
