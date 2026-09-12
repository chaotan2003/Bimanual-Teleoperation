import os
from typing import List, Optional, Tuple, Union

import arx_r5_python.arx_r5_python as arx
import meshcat.transformations as tf
import numpy as np

from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


class ARXR5Interface:
    """
    Base class for a single robot arm.

    Args:
        config (Dict[str, sAny]): Configuration dictionary for the robot arm

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for the robot arm
        num_joints (int): Number of joints in the arm
    """

    def __init__(
        self,
        can_port: str = "can0",
        dt: float = 0.01,
    ):
        self.dt = dt

        urdf_path = os.path.join(ASSET_PATH, "arx/R5a/R5a.urdf")
        self.arm = arx.InterfacesPy(urdf_path, can_port, 0)
        self.arm.arx_x(500, 2000, 10)

    def get_joint_names(self) -> List[str]:
        """
        Get the names of all joints in the arm.

        Returns:
            List[str]: List of joint names. Shape: (num_joints,)
        """
        return NotImplementedError

    def go_home(self) -> bool:
        """
        Move the robot arm to a pre-defined home pose.

        Returns:
            bool: True if the action was successful, False otherwise
        """
        self.arm.set_arm_status(1)
        return True

    def gravity_compensation(self) -> bool:
        self.arm.set_arm_status(3)
        return True

    def protect_mode(self) -> bool:
        self.arm.set_arm_status(2)
        return True

    def set_joint_positions(
        self,
        positions: Union[float, List[float], np.ndarray],
        **kwargs,  # Shape: (num_joints,)
    ) -> bool:
        """
        Move the arm to the given joint position(s).

        Args:
            positions: Desired joint position(s). Shape: (6)
            **kwargs: Additional arguments

        """
        self.arm.set_joint_positions(positions)
        self.arm.set_arm_status(5)

    def set_ee_pose(
        self,
        pos: Optional[Union[List[float], np.ndarray]] = None,  # Shape: (3,)
        quat: Optional[Union[List[float], np.ndarray]] = None,  # Shape: (4,)
        **kwargs,
    ) -> bool:
        """
        Move the end effector to the given pose.

        Args:
            pos: Desired position [x, y, z]. Shape: (3,)
            ori: Desired orientation (quaternion).
                 Shape: (4,) (w, x, y, z)
            **kwargs: Additional arguments

        """

        self.arm.set_ee_pose([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])
        self.arm.set_arm_status(4)

    def set_ee_pose_xyzrpy(
        self,
        xyzrpy: Optional[Union[List[float], np.ndarray]] = None,
        **kwargs,  # Shape: (6,)
    ) -> bool:
        """
        Move the end effector to the given pose.

        Args:
            xyzrpy: Desired position [x, y, z, rol, pitch, yaw]. Shape: (6,)
            **kwargs: Additional arguments

        """
        quat = tf.quaternion_from_euler(xyzrpy[3], xyzrpy[4], xyzrpy[5])

        self.arm.set_ee_pose([xyzrpy[0], xyzrpy[1], xyzrpy[2], quat[0], quat[1], quat[2], quat[3]])
        self.arm.set_arm_status(4)

    def set_catch_pos(self, pos: float):
        self.arm.set_catch(pos)
        self.arm.set_arm_status(5)

    def get_joint_positions(self, joint_names: Optional[Union[str, List[str]]] = None) -> Union[float, List[float]]:
        """
        Get the current joint position(s) of the arm.

        Args:
            joint_names: Name(s) of the joint(s) to get positions for. Shape: (num_joints,) or single string. If None,
                            return positions for all joints.

        """
        return self.arm.get_joint_positions()

    def get_joint_velocities(self, joint_names: Optional[Union[str, List[str]]] = None) -> Union[float, List[float]]:
        """
        Get the current joint velocity(ies) of the arm.

        Args:
            joint_names: Name(s) of the joint(s) to get velocities for. Shape: (num_joints,) or single string. If None,
                            return velocities for all joints.

        """
        return self.arm.get_joint_velocities()

    def get_joint_currents(self, joint_names: Optional[Union[str, List[str]]] = None) -> Union[float, List[float]]:
        return self.arm.get_joint_currents()

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the current end effector pose of the arm.

        Returns:
            End effector pose as (position, quaternion)
            Shapes: position (3,), quaternion (4,) [w, x, y, z]
        """
        xyzwxyz = self.arm.get_ee_pose()

        return xyzwxyz

    def get_ee_pose_xyzrpy(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        xyzwxyz = self.arm.get_ee_pose()

        array = np.array([xyzwxyz[3], xyzwxyz[4], xyzwxyz[5], xyzwxyz[6]])

        roll, pitch, yaw = tf.quaternion_from_euler(array)

        xyzrpy = np.array([xyzwxyz[0], xyzwxyz[1], xyzwxyz[2], roll, pitch, yaw])

        return xyzrpy

    def __del__(self):
        print("ARXR5Interface is being deleted")
