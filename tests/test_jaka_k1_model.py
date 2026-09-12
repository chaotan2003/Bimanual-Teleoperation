"""Model-level checks for the JAKA K1: joints, tool frames and the q_init the IK starts from."""

import unittest

import numpy as np
import pinocchio as pin

from scripts.simulation.teleop_jaka_k1_casadi import (
    JAKA_K1_MANIPULATOR_CONFIG,
    JAKA_K1_Q_INIT,
    JAKA_K1_URDF_PATH,
)

EXPECTED_JOINTS = [f"l-j{i}" for i in range(1, 8)] + [f"r-j{i}" for i in range(1, 8)]


class JakaK1ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = pin.buildModelFromUrdf(JAKA_K1_URDF_PATH)
        cls.data = cls.model.createData()
        cls.frame_names = {f.name for f in cls.model.frames}

    def test_model_has_expected_joints_and_frames(self):
        self.assertEqual(list(self.model.names[1:]), EXPECTED_JOINTS)
        # Fixed base: no floating-base quaternion in q, so q_init indexes joints directly.
        self.assertEqual(self.model.nq, 14)
        self.assertEqual(self.model.nv, 14)
        self.assertEqual(JAKA_K1_MANIPULATOR_CONFIG["left_hand"]["link_name"], "lt")
        self.assertEqual(JAKA_K1_MANIPULATOR_CONFIG["right_hand"]["link_name"], "rt")
        for frame in ("l2", "r2", "lt", "rt"):  # SWGR shoulder anchors and tools
            self.assertIn(frame, self.frame_names, frame)

    def test_q_init_is_inside_joint_limits(self):
        """The IK warm-starts from q_init, and IPOPT will not iterate from outside the bounds."""
        self.assertEqual(JAKA_K1_Q_INIT.shape, (14,))
        lower, upper = self.model.lowerPositionLimit, self.model.upperPositionLimit
        self.assertTrue(np.all(JAKA_K1_Q_INIT >= lower))
        self.assertTrue(np.all(JAKA_K1_Q_INIT <= upper))

    def test_zero_q_init_would_be_infeasible(self):
        """q=0 is not a valid init for this robot: the K1 elbow limits exclude it.

        Upstream xr_teleoperate defaults its IK warm start to np.zeros(nq), which silently
        breaks here -- this pins the reason JAKA_K1_Q_INIT exists.
        """
        lower, upper = self.model.lowerPositionLimit, self.model.upperPositionLimit
        infeasible = np.where(~((lower <= 0.0) & (0.0 <= upper)))[0]
        names = list(self.model.names)[1:]
        self.assertEqual([names[i] for i in infeasible], ["l-j4", "r-j4"])

    def test_forward_kinematics_is_finite_at_q_init(self):
        pin.framesForwardKinematics(self.model, self.data, JAKA_K1_Q_INIT)
        for name, config in JAKA_K1_MANIPULATOR_CONFIG.items():
            T = self.data.oMf[self.model.getFrameId(config["link_name"])].homogeneous
            self.assertTrue(np.isfinite(T).all(), name)


if __name__ == "__main__":
    unittest.main()
