"""Checks for the CasADi/IPOPT IK port and the simulation controller built on it.

Run with the casadi environment (see README): ``python -m unittest discover -s tests``.
"""

import unittest
from unittest import mock

import casadi as ca
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from scripts.simulation.teleop_jaka_k1_casadi import (
    JAKA_K1_MANIPULATOR_CONFIG,
    JAKA_K1_Q_INIT,
    JAKA_K1_URDF_PATH,
)
from xrobotoolkit_teleop.common import base_teleop_controller as base_mod
from xrobotoolkit_teleop.common.base_teleop_controller import BODY_JOINT_INDEX
from xrobotoolkit_teleop.simulation.casadi_arm_ik import (
    FILTER_WEIGHTS,
    SMOOTH_COST_WEIGHT,
    DualArmCasadiIK,
    build_casadi_model,
)
from xrobotoolkit_teleop.simulation.casadi_teleop_controller import CasadiTeleopController, blend_se3
from xrobotoolkit_teleop.utils.weighted_moving_filter import WeightedMovingFilter

LEFT, RIGHT = "lt", "rt"


def _shift(T, dxyz):
    """Copy of a 4x4 pose translated by dxyz in the world frame, orientation untouched."""
    out = np.array(T, dtype=float)
    out[:3, 3] += dxyz
    return out


class CasadiModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmodel, cls.model = build_casadi_model(JAKA_K1_URDF_PATH)

    def test_symbolic_fk_matches_numeric_fk(self):
        """The hand-built casadi model must be kinematically identical to pinocchio's.

        conda-forge's casadi bindings do not expose buildModelFromUrdf, so the symbolic
        model is reconstructed joint by joint; a wrong placement would show up here.
        """
        cdata = self.cmodel.createData()
        data = self.model.createData()
        q = ca.SX.sym("q", self.model.nq)
        cpin.framesForwardKinematics(self.cmodel, cdata, q)
        fk = ca.Function(
            "fk", [q], [cdata.oMf[self.cmodel.getFrameId(LEFT)].homogeneous,
                        cdata.oMf[self.cmodel.getFrameId(RIGHT)].homogeneous]
        )
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(20):
            qq = self.model.lowerPositionLimit + (
                self.model.upperPositionLimit - self.model.lowerPositionLimit
            ) * rng.random(self.model.nq)
            pin.framesForwardKinematics(self.model, data, qq)
            for sym, frame in zip(fk(qq), (LEFT, RIGHT)):
                ref = data.oMf[self.model.getFrameId(frame)].homogeneous
                worst = max(worst, float(np.abs(np.array(sym) - ref).max()))
        self.assertLess(worst, 1e-12, f"symbolic FK deviates by {worst:.3e}")

    def test_casadi_model_covers_every_joint_and_frame(self):
        self.assertEqual(self.cmodel.nq, self.model.nq)
        self.assertEqual(list(self.cmodel.names), list(self.model.names))
        self.assertEqual({f.name for f in self.cmodel.frames}, {f.name for f in self.model.frames})


class ParameterPackingTest(unittest.TestCase):
    """casadi flattens matrices column-major; numpy's .ravel() is row-major by default."""

    @classmethod
    def setUpClass(cls):
        cls.ik = DualArmCasadiIK(JAKA_K1_URDF_PATH, LEFT, RIGHT, JAKA_K1_Q_INIT)

    def test_pack_params_round_trips_through_the_nlp_parameter(self):
        q_last = self.ik.q_init + 0.1
        left = self.ik.frame_pose(q_last, LEFT)
        right = self.ik.frame_pose(q_last, RIGHT)
        packed = DualArmCasadiIK.pack_params(q_last, left, right)

        sym_q = ca.SX.sym("q_last", self.ik.nq)
        sym_l = ca.SX.sym("left", 4, 4)
        sym_r = ca.SX.sym("right", 4, 4)
        unpack = ca.Function("unpack", [ca.vertcat(sym_q, sym_l[:], sym_r[:])], [sym_q, sym_l, sym_r])
        got_q, got_l, got_r = unpack(packed)

        np.testing.assert_allclose(np.array(got_q).ravel(), q_last, atol=1e-12)
        np.testing.assert_allclose(np.array(got_l), left, atol=1e-12)
        np.testing.assert_allclose(np.array(got_r), right, atol=1e-12)

    def test_row_major_packing_would_be_detected(self):
        """Guard the guard: the check above must actually fail on a scrambled target."""
        q_last = self.ik.q_init
        left = self.ik.frame_pose(q_last, LEFT)
        right = self.ik.frame_pose(q_last, RIGHT)
        scrambled = np.concatenate([q_last, left.ravel(), right.ravel()])  # row-major == wrong

        sym_q = ca.SX.sym("q_last", self.ik.nq)
        sym_l = ca.SX.sym("left", 4, 4)
        sym_r = ca.SX.sym("right", 4, 4)
        unpack = ca.Function("unpack", [ca.vertcat(sym_q, sym_l[:], sym_r[:])], [sym_q, sym_l, sym_r])
        _, got_l, _ = unpack(scrambled)
        self.assertGreater(float(np.abs(np.array(got_l) - left).max()), 1e-6)


class IkSolveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ik = DualArmCasadiIK(JAKA_K1_URDF_PATH, LEFT, RIGHT, JAKA_K1_Q_INIT)
        cls.home_left = cls.ik.frame_pose(JAKA_K1_Q_INIT, LEFT)
        cls.home_right = cls.ik.frame_pose(JAKA_K1_Q_INIT, RIGHT)

    def _solve_from_home(self, left_target, right_target, smooth_cost_weight=None):
        ik = self.ik if smooth_cost_weight is None else DualArmCasadiIK(
            JAKA_K1_URDF_PATH, LEFT, RIGHT, JAKA_K1_Q_INIT, smooth_cost_weight=smooth_cost_weight
        )
        ik.last_q = JAKA_K1_Q_INIT.copy()
        ik.filter = WeightedMovingFilter(FILTER_WEIGHTS, ik.nq)
        q, torque = ik.solve_ik(left_target, right_target, q_meas=JAKA_K1_Q_INIT)
        self.assertIsNone(torque)  # simulation port: no rnea output
        return ik, q

    def test_zero_q_init_is_rejected(self):
        """Upstream defaults its warm start to zeros, which is out of bounds for the K1 elbow."""
        with self.assertRaises(ValueError) as ctx:
            DualArmCasadiIK(JAKA_K1_URDF_PATH, LEFT, RIGHT, np.zeros(14))
        self.assertIn("l-j4", str(ctx.exception))

    def test_wrong_q_init_length_is_rejected(self):
        with self.assertRaises(ValueError):
            DualArmCasadiIK(JAKA_K1_URDF_PATH, LEFT, RIGHT, np.zeros(12))

    def test_reaches_a_reachable_target(self):
        target = _shift(self.home_left, [0.03, 0.0, 0.0])
        _, q = self._solve_from_home(target, self.home_right)
        residual = np.linalg.norm(self.ik.frame_pose(q, LEFT)[:3, 3] - target[:3, 3])
        self.assertLess(residual, 0.01, f"residual {residual * 1000:.1f} mm")
        self.assertTrue(np.all(q >= self.ik.model.lowerPositionLimit - 1e-9))
        self.assertTrue(np.all(q <= self.ik.model.upperPositionLimit + 1e-9))

    def test_holds_still_when_the_target_is_the_current_pose(self):
        _, q = self._solve_from_home(self.home_left, self.home_right)
        self.assertLess(float(np.degrees(np.abs(q - JAKA_K1_Q_INIT).max())), 0.05)

    def test_a_small_target_step_moves_the_joints_less_than_a_degree(self):
        """A 5 mm target step costs <1 deg of joint motion, whatever the loop rate is.

        The placo path closed the same step inside one dt, which turned position error into
        a velocity command (~26000 deg/s peak for a 10 mm step). Here the per-cycle joint
        move is bounded by the smooth cost, so joint rate follows target rate instead.
        """
        _, q = self._solve_from_home(_shift(self.home_left, [0.005, 0.0, 0.0]), self.home_right)
        self.assertLess(float(np.degrees(np.abs(q - JAKA_K1_Q_INIT).max())), 1.0)

    def test_a_large_target_step_is_not_closed_in_one_cycle(self):
        """Where the smooth cost really bites: a 200 mm step leaves >10 mm of residual."""
        target = _shift(self.home_left, [0.2, 0.0, 0.0])
        _, q = self._solve_from_home(target, self.home_right)
        residual = np.linalg.norm(self.ik.frame_pose(q, LEFT)[:3, 3] - target[:3, 3])
        self.assertGreater(residual, 0.01, f"residual {residual * 1000:.1f} mm")
        self.assertLess(float(np.degrees(np.abs(q - JAKA_K1_Q_INIT).max())), 50.0)

    def test_higher_smooth_cost_moves_less_per_cycle(self):
        step = [0.02, 0.0, 0.0]
        _, q_low = self._solve_from_home(_shift(self.home_left, step), self.home_right)
        _, q_high = self._solve_from_home(
            _shift(self.home_left, step), self.home_right, smooth_cost_weight=100 * SMOOTH_COST_WEIGHT
        )
        self.assertLess(
            float(np.abs(q_high - JAKA_K1_Q_INIT).max()), float(np.abs(q_low - JAKA_K1_Q_INIT).max())
        )

    def test_tracking_a_moving_target_stays_rate_bounded(self):
        ik = DualArmCasadiIK(JAKA_K1_URDF_PATH, LEFT, RIGHT, JAKA_K1_Q_INIT)
        q = JAKA_K1_Q_INIT.copy()
        worst_rate, worst_error = 0.0, 0.0
        for i in range(60):
            q_prev = q
            target = _shift(self.home_left, [0.02 * np.sin(i * 0.2), 0.01 * np.cos(i * 0.15), 0.0])
            q, _ = ik.solve_ik(target, self.home_right, q_meas=q)
            pose = ik.frame_pose(q, LEFT)
            worst_rate = max(worst_rate, float(np.abs(q - q_prev).max()))
            worst_error = max(worst_error, float(np.linalg.norm(pose[:3, 3] - target[:3, 3])))
        # ~4 mm/cycle of target motion: the EE stays within 2 cm and no joint moves more
        # than ~2 deg per cycle, i.e. <=60 deg/s at the 30 Hz default.
        self.assertLess(worst_error, 0.02, f"EE tracking error {worst_error * 1000:.1f} mm")
        self.assertLess(float(np.degrees(worst_rate)), 2.0, f"{np.degrees(worst_rate):.2f} deg/cycle")
        self.assertTrue(np.all(q >= ik.model.lowerPositionLimit - 1e-9))
        self.assertTrue(np.all(q <= ik.model.upperPositionLimit + 1e-9))


class WeightedMovingFilterTest(unittest.TestCase):
    def test_matches_a_manual_weighted_average(self):
        f = WeightedMovingFilter([0.4, 0.3, 0.2, 0.1], 1)
        for value in (1.0, 2.0, 3.0, 4.0):
            f.add_data(np.array([value]))
        w = np.array([0.4, 0.3, 0.2, 0.1])
        expected = (w @ np.array([[4.0], [3.0], [2.0], [1.0]])) / w.sum()
        np.testing.assert_allclose(f.get_filtered_data(), expected.ravel(), atol=1e-12)

    def test_repeated_samples_are_not_skipped(self):
        f = WeightedMovingFilter([0.5, 0.5], 1)
        f.add_data(np.array([0.0]))
        f.add_data(np.array([0.0]))
        f.add_data(np.array([1.0]))
        np.testing.assert_allclose(f.get_filtered_data(), [0.5], atol=1e-12)

    def test_rejects_the_wrong_size(self):
        f = WeightedMovingFilter([0.5, 0.5], 14)
        with self.assertRaises(ValueError):
            f.add_data(np.zeros(13))


class BlendSe3Test(unittest.TestCase):
    def setUp(self):
        self.T0 = np.eye(4)
        self.T0[:3, 3] = [0.1, 0.2, 0.3]
        self.T1 = np.eye(4)
        self.T1[:3, :3] = pin.exp3(np.ascontiguousarray(np.array([0.3, -0.2, 0.1])))
        self.T1[:3, 3] = [0.4, 0.0, 0.1]

    def test_endpoints_and_midpoint(self):
        np.testing.assert_allclose(blend_se3(self.T0, self.T1, 0.0), self.T0, atol=1e-12)
        np.testing.assert_allclose(blend_se3(self.T0, self.T1, 1.0), self.T1, atol=1e-12)
        mid = blend_se3(self.T0, self.T1, 0.5)
        np.testing.assert_allclose(mid[:3, 3], 0.5 * (self.T0[:3, 3] + self.T1[:3, 3]), atol=1e-12)

    def test_rotation_stays_a_rotation(self):
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            R = blend_se3(self.T0, self.T1, alpha)[:3, :3]
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)


class _StubXrClient:
    """Synthetic XR frames, so the retargeting path runs without a headset or server.

    ``hands`` is filled in by ``ControllerOfflineTest._calibrate`` so the retargeted target
    lands on the startup EE pose; the offsets then produce a known-size target step.
    """

    SHOULDERS = {"left": np.array([0.0, -0.18, 0.0]), "right": np.array([0.0, 0.18, 0.0])}

    def __init__(self):
        self.grip = 0.0
        self.hands = {"left": self.SHOULDERS["left"] + np.array([0.4, 0.0, -0.3]),
                      "right": self.SHOULDERS["right"] + np.array([0.4, 0.0, -0.3])}
        self.wrist_offset = np.zeros(3)
        self.motion = np.zeros(3)

    def get_body_tracking_data(self):
        poses = np.zeros((24, 7))
        poses[:, 6] = 1.0
        for side, idx in BODY_JOINT_INDEX.items():
            shoulder = self.SHOULDERS[side]
            wrist = self.hands[side] + self.wrist_offset + self.motion
            poses[idx["shoulder"], :3] = shoulder
            poses[idx["elbow"], :3] = shoulder + 0.5 * (wrist - shoulder)
            poses[idx["wrist"], :3] = wrist
        return {"poses": poses}

    def get_key_value_by_name(self, name):
        return self.grip if name.endswith("grip") else 0.0

    def get_motion_tracker_data(self):
        return None

    def get_pose_by_name(self, name):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # identity orientation


class ControllerOfflineTest(unittest.TestCase):
    """Drives CasadiTeleopController._update_ik without a headset, solver or browser."""

    def _build(self, engage_ramp_time):
        with mock.patch.object(base_mod, "XrClient", _StubXrClient), mock.patch.object(
            CasadiTeleopController, "_init_viz", lambda self: None
        ), mock.patch.object(CasadiTeleopController, "_update_viz", lambda self: None):
            ctrl = CasadiTeleopController(
                robot_urdf_path=JAKA_K1_URDF_PATH,
                manipulator_config=JAKA_K1_MANIPULATOR_CONFIG,
                q_init=JAKA_K1_Q_INIT,
                left_frame=LEFT,
                right_frame=RIGHT,
                engage_ramp_time=engage_ramp_time,
            )
        self._calibrate(ctrl, ctrl.xr_client)
        return ctrl

    def _calibrate(self, ctrl, stub):
        """Put the stub's hands where the startup EE pose already is, so grip engages calmly.

        Inverts swgr_ee_position: wrist = shoulder + R_world_vr^T (ee - shoulder_robot) / scale.
        Without this the synthetic hand lands hundreds of mm from the tool frame and the
        first engagement becomes a full-speed acquisition instead of a test of the ramp.
        """
        for side, name in (("left", "left_hand"), ("right", "right_hand")):
            config = ctrl.manipulator_config[name]
            shoulder_robot = np.array(ctrl._get_link_pose(config["swgr"]["shoulder_link"])[0])
            home = ctrl.placo_robot.frame(config["link_name"])[:3, 3]
            stub.hands[side] = stub.SHOULDERS[side] + ctrl.R_headset_world.T @ (
                (home - shoulder_robot) / ctrl.swgr_scale[name]
            )

    def _peak_rate(self, ctrl, stub, cycles=12):
        """Peak single-cycle joint move (deg) after the hand is moved while grip is released."""
        stub.grip = 1.0
        for _ in range(10):  # engaged, settled on the calibrated target
            ctrl._update_ik()
        stub.grip = 0.0
        for _ in range(3):  # released: target holds, ref_tracker_xyz is NOT reset
            ctrl._update_ik()
        stub.wrist_offset = np.array([0.06, 0.0, 0.0])  # operator moved while released
        stub.grip = 1.0
        peak = 0.0
        for _ in range(cycles):
            q_before = ctrl.placo_robot.state.q.copy()
            ctrl._update_ik()
            peak = max(peak, float(np.degrees(np.abs(ctrl.placo_robot.state.q - q_before).max())))
        return peak

    def test_engage_ramp_caps_the_reattach_jump(self):
        """Re-pressing grip after moving the hand commands a step; the ramp spreads it out."""
        bare = self._build(0.0)
        unfiltered = self._peak_rate(bare, bare.xr_client)
        ctrl = self._build(0.3)
        ramped = self._peak_rate(ctrl, ctrl.xr_client)
        self.assertGreater(unfiltered, 2.0, f"expected a real jump, got {unfiltered:.2f} deg/cycle")
        self.assertLess(ramped, 0.6 * unfiltered, f"ramp {ramped:.2f} vs unfiltered {unfiltered:.2f} deg/cycle")

    def test_ramp_still_reaches_the_target(self):
        ctrl = self._build(0.3)
        stub = ctrl.xr_client
        stub.grip = 1.0
        for _ in range(5):
            ctrl._update_ik()
        stub.wrist_offset = np.array([0.06, 0.0, 0.0])  # ~80 mm of tool travel after scaling
        for _ in range(90):  # 3 s at 30 Hz, well past the 0.3 s ramp
            ctrl._update_ik()
        for name, config in ctrl.manipulator_config.items():
            target = ctrl._task_target(name)
            actual = ctrl.placo_robot.frame(config["link_name"])
            self.assertLess(
                float(np.linalg.norm(actual[:3, 3] - target[:3, 3])), 0.02, f"{name} did not converge"
            )

    def test_targets_are_held_while_disengaged(self):
        ctrl = self._build(0.3)
        stub = ctrl.xr_client
        stub.grip = 1.0
        for _ in range(5):
            ctrl._update_ik()
        held = {n: np.array(t.T_world_frame) for n, t in ctrl.effector_task.items()}
        stub.grip = 0.0
        stub.wrist_offset = np.array([0.08, 0.0, 0.0])
        for _ in range(5):
            ctrl._update_ik()
        for name, task in ctrl.effector_task.items():
            np.testing.assert_allclose(task.T_world_frame, held[name], atol=1e-12, err_msg=name)


if __name__ == "__main__":
    unittest.main()
