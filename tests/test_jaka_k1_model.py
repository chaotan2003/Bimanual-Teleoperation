"""Model-level checks for the JAKA K1: joints, tool frames and the q_init the IK starts from."""

import unittest

import numpy as np
import yourdfpy

from bimanual_teleop.kinematics_backend import Robot
from bimanual_teleop.robots.jaka_k1 import (
    JAKA_K1_MANIPULATOR_CONFIG,
    JAKA_K1_Q_INIT,
    JAKA_K1_URDF_PATH,
)

EXPECTED_JOINTS = [f"l-j{i}" for i in range(1, 8)] + [f"r-j{i}" for i in range(1, 8)]


class JakaK1ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.urdf = yourdfpy.URDF.load(JAKA_K1_URDF_PATH)
        cls.robot = Robot.from_urdf(cls.urdf)
        cls.frame_names = set(cls.robot.links.names)

    def test_model_has_expected_joints_and_frames(self):
        self.assertEqual(list(self.robot.joints.actuated_names), EXPECTED_JOINTS)
        # Fixed base: no floating-base quaternion in q, so q_init indexes joints directly.
        self.assertEqual(self.robot.joints.num_actuated_joints, 14)
        self.assertEqual(JAKA_K1_MANIPULATOR_CONFIG["left_hand"]["link_name"], "lt")
        self.assertEqual(JAKA_K1_MANIPULATOR_CONFIG["right_hand"]["link_name"], "rt")
        for frame in ("l2", "r2", "lt", "rt"):  # SWGR shoulder anchors and tools
            self.assertIn(frame, self.frame_names, frame)

    def test_q_init_is_inside_joint_limits(self):
        """The IK warm-starts from q_init, and IPOPT will not iterate from outside the bounds."""
        self.assertEqual(JAKA_K1_Q_INIT.shape, (14,))
        lower, upper = self.robot.joints.lower_limits, self.robot.joints.upper_limits
        self.assertTrue(np.all(JAKA_K1_Q_INIT >= lower))
        self.assertTrue(np.all(JAKA_K1_Q_INIT <= upper))

    def test_zero_q_init_would_be_infeasible(self):
        """q=0 is not a valid init for this robot: the K1 elbow limits exclude it.

        Upstream xr_teleoperate defaults its IK warm start to np.zeros(nq), which silently
        breaks here -- this pins the reason JAKA_K1_Q_INIT exists.
        """
        lower, upper = self.robot.joints.lower_limits, self.robot.joints.upper_limits
        infeasible = np.where(~((lower <= 0.0) & (0.0 <= upper)))[0]
        self.assertEqual([self.robot.joints.actuated_names[i] for i in infeasible], ["l-j4", "r-j4"])

    def test_forward_kinematics_is_finite_at_q_init(self):
        poses = np.asarray(self.robot.forward_kinematics(JAKA_K1_Q_INIT))
        link_indices = {name: i for i, name in enumerate(self.robot.links.names)}
        for name, config in JAKA_K1_MANIPULATOR_CONFIG.items():
            self.assertTrue(np.isfinite(poses[link_indices[config["link_name"]]]).all(), name)


if __name__ == "__main__":
    unittest.main()
