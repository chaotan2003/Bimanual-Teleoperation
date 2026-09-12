import os
from typing import Dict

import numpy as np
import rospy

from xrobotoolkit_teleop.common.base_hardware_teleop_controller import (
    HardwareTeleopController,
)
from xrobotoolkit_teleop.hardware.interface.galaxea import (
    A1XController,
    R1LiteChassisController,
    R1LiteTorsoController,
)
from xrobotoolkit_teleop.hardware.interface.ros_camera import RosCameraInterface
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

# Default paths and configurations for R1 Lite dual arm
DEFAULT_DUAL_A1X_URDF_PATH = os.path.join(ASSET_PATH, "galaxea/A1X/dual_a1x.urdf")
DEFAULT_SCALE_FACTOR = 1.0
CONTROLLER_DEADZONE = 0.1

# R1 Lite always has both arms - no single arm configuration needed
DEFAULT_MANIPULATOR_CONFIG = {
    "right_arm": {
        "link_name": "right_gripper_link",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": [
                "right_gripper_finger_joint1",
            ],
            "open_pos": [
                -2.9,
            ],
            "close_pos": [
                0.0,
            ],
        },
    },
    "left_arm": {
        "link_name": "left_gripper_link",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "left_trigger",
            "joint_names": [
                "left_gripper_finger_joint1",
            ],
            "open_pos": [
                -2.9,
            ],
            "close_pos": [
                0.0,
            ],
        },
    },
}


class GalaxeaR1LiteTeleopController(HardwareTeleopController):
    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_DUAL_A1X_URDF_PATH,
        manipulator_config: dict = DEFAULT_MANIPULATOR_CONFIG,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        chassis_velocity_scale: list[float] = [0.75, 0.75, 1.0],
        visualize_placo: bool = False,
        control_rate_hz: int = 100,
        enable_log_data: bool = True,
        log_dir: str = "logs/galaxea_r1_lite",
        log_freq: float = 50,
        enable_camera: bool = True,
        camera_fps: int = 30,
    ):
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            R_headset_world=R_headset_world,
            floating_base=False,
            scale_factor=scale_factor,
            visualize_placo=visualize_placo,
            control_rate_hz=control_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
            enable_camera=enable_camera,
            camera_fps=camera_fps,
        )
        self.chassis_velocity_scale = chassis_velocity_scale

    def _placo_setup(self):
        super()._placo_setup()
        # R1 Lite always has both left and right arms
        self.placo_arm_joint_slice = {}
        for arm_name in ["left_arm", "right_arm"]:
            config = self.manipulator_config[arm_name]
            ee_link_name = config["link_name"]
            arm_prefix = ee_link_name.replace("gripper_link", "")
            arm_joint_names = [f"{arm_prefix}arm_joint{i}" for i in range(1, 7)]
            self.placo_arm_joint_slice[arm_name] = slice(
                self.placo_robot.get_joint_offset(arm_joint_names[0]),
                self.placo_robot.get_joint_offset(arm_joint_names[-1]) + 1,
            )

    def _robot_setup(self):
        rospy.init_node("galaxea_r1_lite_teleop_controller", anonymous=True)

        # setup arm controllers
        self.arm_controllers: Dict[str, A1XController] = {}
        for arm_name in ["left_arm", "right_arm"]:
            arm_prefix = arm_name.replace("_arm", "")
            controller = A1XController(
                arm_control_topic=f"/motion_control/control_arm_{arm_prefix}",
                gripper_control_topic=f"/motion_control/control_gripper_{arm_prefix}",
                arm_state_topic=f"/hdas/feedback_arm_{arm_prefix}",
                rate_hz=1.0 / self.dt,
                gripper_position_control=False,
            )
            self.arm_controllers[arm_name] = controller

        print("Waiting for initial joint states from both R1 Lite arms...")
        all_controllers_ready = False
        while not rospy.is_shutdown() and not all_controllers_ready:
            all_controllers_ready = all(
                controller.timestamp > 0 for controller in self.arm_controllers.values()
            )
            rospy.sleep(0.1)
        print("Both arm controllers received initial state.")

        # Setup chassis controller
        self.chassis_controller = R1LiteChassisController(
            chassis_state_topic="/hdas/feedback_chassis",
            chassis_control_topic="/motion_target/target_speed_chassis",
            rate_hz=1.0 / self.dt,
        )

        print("Waiting for initial chassis state...")
        while not rospy.is_shutdown() and self.chassis_controller.timestamp == -1:
            rospy.sleep(0.1)
        print("Chassis controller received initial state.")

        self.torso_controller = R1LiteTorsoController(
            torso_state_topic="/hdas/feedback_torso",
            torso_control_topic="/motion_target/target_speed_torso",
            rate_hz=1.0 / self.dt,
        )

        print("Waiting for initial torso state...")
        while not rospy.is_shutdown() and self.torso_controller.timestamp == -1:
            rospy.sleep(0.1)
        print("Torso controller received initial state.")

    def _initialize_camera(self):
        if self.enable_camera:
            print("Initializing camera...")
            try:
                camera_topics = {
                    "left": {
                        "color": "/hdas/camera_wrist_left/color/image_raw/compressed",
                        # "depth": "/hdas/camera_wrist_left/aligned_depth_to_color/image_raw",
                    },
                    "right": {
                        "color": "/hdas/camera_wrist_right/color/image_raw/compressed",
                        # "depth": "/hdas/camera_wrist_right/aligned_depth_to_color/image_raw",
                    },
                    "head_left": {
                        "color": "/hdas/camera_head/left_raw/image_raw_color/compressed",
                    },
                    "head_right": {
                        "color": "/hdas/camera_head/right_raw/image_raw_color/compressed",
                    },
                }
                self.camera_interface = RosCameraInterface(
                    camera_topics=camera_topics, width=424, height=240, enable_depth=False
                )
                self.camera_interface.start()
                print("Camera initialized successfully.")
            except Exception as e:
                print(f"Error initializing camera: {e}")
                self.enable_camera = False

    def _update_robot_state(self):
        """Reads current joint states from both arm controllers and updates Placo."""
        for arm_name, controller in self.arm_controllers.items():
            self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]] = controller.qpos

    def _send_command(self):
        """Sends the solved joint targets to both arm controllers."""
        for arm_name, controller in self.arm_controllers.items():
            if self.active.get(arm_name, False):
                controller.q_des = self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]].copy()

            controller.q_des_gripper = [
                self.gripper_pos_target[arm_name][gripper_joint]
                for gripper_joint in self.gripper_pos_target[arm_name].keys()
            ]

            controller.publish_arm_control()
            controller.publish_gripper_control()

        self.chassis_controller.publish_chassis_control()
        self.torso_controller.publish_torso_control()

    def _pre_ik_update(self):
        """Updates the chassis and torso velocity commands based on joystick input."""
        self._update_joystick_velocity_command()
        self._update_torso_velocity_command()

    def _update_joystick_velocity_command(self):
        """Updates the chassis velocity commands based on joystick input."""
        left_axis = self.xr_client.get_joystick_state("left")
        right_axis = self.xr_client.get_joystick_state("right")

        vx = left_axis[1] * self.chassis_velocity_scale[0]
        vy = -left_axis[0] * self.chassis_velocity_scale[1]
        omega = -right_axis[0] * self.chassis_velocity_scale[2]

        self.chassis_controller.set_velocity_command(vx, vy, omega)

    def _update_torso_velocity_command(self):
        buttonY = self.xr_client.get_button_state_by_name("Y")
        buttonX = self.xr_client.get_button_state_by_name("X")

        vz = 2.5 if buttonY else -2.5 if buttonX else 0.0
        self.torso_controller.set_velocity_command(vz)

    def _get_robot_state_for_logging(self) -> Dict:
        """Returns a dictionary of robot-specific data for logging."""
        return {
            "qpos": {arm: controller.qpos for arm, controller in self.arm_controllers.items()},
            "qvel": {arm: controller.qvel for arm, controller in self.arm_controllers.items()},
            "qpos_des": {arm: controller.q_des for arm, controller in self.arm_controllers.items()},
            "gripper_qpos": {
                arm: controller.qpos_gripper for arm, controller in self.arm_controllers.items()
            },
            "gripper_qpos_des": {
                arm: controller.q_des_gripper for arm, controller in self.arm_controllers.items()
            },
            "chassis_velocity_cmd": self.chassis_controller.get_velocity_command(),
        }

    def _get_camera_frame_for_logging(self) -> Dict:
        """Returns a dictionary of camera frames for logging with camera names as keys."""
        if not self.camera_interface:
            return {}

        # Use compressed frames for logging to reduce file size
        if self.camera_interface.enable_compression:
            frames = self.camera_interface.get_compressed_frames()
        else:
            # Fallback to regular frames for compatibility
            frames = self.camera_interface.get_frames()

        # ROS camera interface already uses camera names as keys, so no mapping needed
        return frames if frames else {}

    def _should_keep_running(self) -> bool:
        """Returns True if the main loop should continue running."""
        return super()._should_keep_running() and not rospy.is_shutdown()

    def _shutdown_robot(self):
        """Performs graceful shutdown of the robot hardware."""
        for arm_controller in self.arm_controllers.values():
            arm_controller.stop()
        print("Arm controllers stopped.")
        self.torso_controller.stop_torso()
        print("Torso stopped.")
        self.chassis_controller.stop_chassis()
        print("Chassis stopped.")
