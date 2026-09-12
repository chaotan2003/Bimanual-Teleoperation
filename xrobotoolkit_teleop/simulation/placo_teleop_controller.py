import time
from typing import Any, Dict

import meshcat.transformations as tf

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)


class PlacoTeleopController(BaseTeleopController):
    """
    Placo teleoperation controller for a robot using inverse kinematics.

    This controller allows teleoperation of a robot by mapping XR controller poses
    to end effector movements using the Placo IK solver.
    """

    def __init__(
        self,
        robot_urdf_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        floating_base=False,
        R_headset_world=R_HEADSET_TO_WORLD,
        scale_factor=1.0,
        q_init=None,
        dt=0.01,
    ):
        super().__init__(
            robot_urdf_path,
            manipulator_config,
            floating_base,
            R_headset_world,
            scale_factor,
            q_init,
            dt,
        )
        self._init_placo_viz()

    def _robot_setup(self):
        pass

    def _send_command(self):
        self._update_placo_viz()

    def _update_robot_state(self):
        pass

    def _get_link_pose(self, link_name):
        link_xyz = self.placo_robot.get_T_world_frame(link_name)[:3, 3]
        link_quat = tf.quaternion_from_matrix(self.placo_robot.get_T_world_frame(link_name))
        return link_xyz, link_quat

    def run(self):
        """
        Run the main teleoperation loop.
        This method is inherited from BaseTeleopController and implements the teleoperation logic.
        """
        while not self._stop_event.is_set():
            try:
                start_time = time.time()
                self._update_ik()
                self._send_command()
                end_time = time.time()
                time.sleep(max(0, self.dt - (end_time - start_time)))
            except KeyboardInterrupt:
                print("\nTeleoperation stopped.")
                self._stop_event.set()
