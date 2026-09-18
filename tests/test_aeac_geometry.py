import unittest

import numpy as np

from bimanual_teleop.retargeting.swgr import adaptive_elbow_weight, elbow_axis_direction


class AeacGeometryTest(unittest.TestCase):
    def test_list_inputs_are_supported(self):
        direction, radius = elbow_axis_direction([0, 0, 0], [0.5, 0.2, 0], [1, 0, 0])

        np.testing.assert_allclose(direction, [0.0, 1.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(radius, 0.2)

    def test_straight_arm_disables_constraint(self):
        direction, radius = elbow_axis_direction([0, 0, 0], [0.5, 0, 0], [1, 0, 0])

        self.assertIsNone(direction)
        self.assertEqual(adaptive_elbow_weight(radius, 0.58), 0.0)

    def test_weight_ramps_after_deadband(self):
        self.assertEqual(adaptive_elbow_weight(0.02, 0.58, deadband=0.04), 0.0)
        self.assertGreater(adaptive_elbow_weight(0.2, 0.58, deadband=0.04), 0.0)
        self.assertEqual(adaptive_elbow_weight(1.0, 0.58, deadband=0.04), 1.0)


if __name__ == "__main__":
    unittest.main()
