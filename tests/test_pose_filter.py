import unittest

import numpy as np

from bimanual_teleop.filters.pose import LPRotationFilter, OneEuroFilter, quaternion_slerp


class OneEuroFilterTest(unittest.TestCase):
    def test_first_sample_passes_through(self):
        f = OneEuroFilter()
        np.testing.assert_allclose(f.next(1.0, np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])

    def test_step_is_smoothed(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        f.next(0.0, np.array([0.0]))
        y = f.next(0.02, np.array([1.0]))
        self.assertGreater(float(y[0]), 0.0)
        self.assertLess(float(y[0]), 1.0)

    def test_shape_change_is_rejected(self):
        f = OneEuroFilter()
        f.next(0.0, np.zeros(3))
        with self.assertRaises(ValueError):
            f.next(0.1, np.zeros(2))


class RotationFilterTest(unittest.TestCase):
    def test_slerp_takes_shortest_path(self):
        q = quaternion_slerp(np.array([1.0, 0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0, 0.0]), 0.5)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_low_pass_returns_unit_quaternion(self):
        f = LPRotationFilter(alpha=0.5)
        f.next(np.array([1.0, 0.0, 0.0, 0.0]))
        q = f.next(np.array([0.0, 1.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
