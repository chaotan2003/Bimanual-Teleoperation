import numpy as np
import rtde_control
import rtde_receive

from xrobotoolkit_teleop.hardware.interface.robotiq_gripper import RobotiqGripper

LEFT_ROBOT_IP = "192.168.50.55"
RIGHT_ROBOT_IP = "192.168.50.195"

SERVO_TIME = 0.017
LOOKAHEAD_TIME = 0.1
SERVO_GAIN = 300.0
MAX_VELOCITY = 0.5
MAX_ACCELERATION = 1.0

GRIPPER_FORCE = 128
GRIPPER_SPEED = 255
CONTROLLER_DEADZONE = 0.1

LEFT_INITIAL_JOINT_DEG = np.array([165.26, -47.50, 118.93, -38.96, 87.51, 149.56])
RIGHT_INITIAL_JOINT_DEG = np.array([-166.53,-136.17,-106.02,210.01,-86.87,-128.40])


class URController:
    def __init__(
        self,
        robot_ip: str,
        initial_joint_positions: np.ndarray,
        max_velocity: float = MAX_VELOCITY,
        max_acceleration: float = MAX_ACCELERATION,
        servo_time: float = SERVO_TIME,
        lookahead_time: float = LOOKAHEAD_TIME,
        servo_gain: float = SERVO_GAIN,
        gripper_force: float = GRIPPER_FORCE,
        gripper_speed: float = GRIPPER_SPEED,
    ):
        self.robot_ip = robot_ip
        self.initial_joint_positions = initial_joint_positions
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.servo_time = servo_time
        self.lookahead_time = lookahead_time
        self.servo_gain = servo_gain
        self.gripper_force = gripper_force
        self.gripper_speed = gripper_speed

        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        print(f"Connected to UR robot at {robot_ip}")

        self.gripper = RobotiqGripper()
        self.gripper.connect(robot_ip, 63352)
        print("Gripper connected.")

    def reset(self):
        print(f"Moving to initial joint positions: {self.initial_joint_positions}")
        self.rtde_c.moveJ(self.initial_joint_positions)
        print("Reached initial position.")
        self.gripper.activate()
        print("Gripper activated.")

    def servo_joints(self, joint_positions: np.ndarray):
        t_start = self.rtde_c.initPeriod()
        self.rtde_c.servoJ(
            joint_positions,
            self.max_velocity,
            self.max_acceleration,
            self.servo_time,
            self.lookahead_time,
            self.servo_gain,
        )
        self.rtde_c.waitPeriod(t_start)

    def open_gripper(self):
        self.gripper.move_and_wait_for_pos(
            self.gripper.get_open_position(),
            self.gripper_speed,
            self.gripper_force,
        )

    def close_gripper(self):
        self.gripper.move_and_wait_for_pos(
            self.gripper.get_closed_position(),
            self.gripper_speed,
            self.gripper_force,
        )

    def get_current_joint_positions(self) -> np.ndarray:
        return np.array(self.rtde_r.getActualQ())

    def get_current_tcp_pose(self) -> np.ndarray:
        return np.array(self.rtde_r.getActualTCPPose())

    def close(self):
        self.rtde_c.servoStop()
        self.rtde_c.stopScript()
        self.gripper.disconnect()
        print("UR controller closed and gripper disconnected.")

    def __del__(self):
        """Ensures resources are released when the object is deleted."""
        self.close()
