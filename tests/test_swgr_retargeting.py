"""Checks for Shoulder-Wrist Geometric Retargeting (SWGR).

SWGR maps the operator's shoulder->wrist vector onto the robot with a single scale
factor robot_reach / human_reach, anchored at the robot shoulder. These tests pin the
properties that make the method work: full extension lands on the robot's max reach,
the mapping is proportional, and the VR world origin cancels (so no clutch re-anchoring
and no incremental drift). The last test ties the configured robot_reach back to the
actual JAKA K1 model so a wrong constant cannot slip through.
"""

import unittest

import meshcat.transformations as tf
import numpy as np
import yourdfpy

from bimanual_teleop.kinematics_backend import Robot
from bimanual_teleop.robots.jaka_k1 import (
    HUMAN_REACH,
    JAKA_K1_MANIPULATOR_CONFIG,
    JAKA_K1_Q_INIT,
    JAKA_K1_REACH,
    JAKA_K1_URDF_PATH,
)
from bimanual_teleop.core.teleop_controller import BaseTeleopController
from bimanual_teleop.retargeting.swgr import (
    R_HEADSET_TO_WORLD,
    adaptive_elbow_weight,
    elbow_axis_direction,
    swgr_ee_position,
)


def _fk(q):
    """Frame name -> 4x4 world transform for the JAKA K1 at joint configuration q."""
    robot = Robot.from_urdf(yourdfpy.URDF.load(JAKA_K1_URDF_PATH))
    poses = np.asarray(robot.forward_kinematics(np.asarray(q, dtype=float)))
    frames = {}
    for name, pose in zip(robot.links.names, poses, strict=True):
        T = tf.quaternion_matrix(pose[:4])
        T[:3, 3] = pose[4:]
        frames[name] = T
    return frames

SHOULDER_ROBOT = np.array([0.0, 0.2225, 0.217])  # JAKA K1 left shoulder (l2) in world
SCALE = JAKA_K1_REACH / HUMAN_REACH


class SwgrRetargetingTest(unittest.TestCase):
    def retarget(self, shoulder_human, wrist_human):
        return swgr_ee_position(
            SHOULDER_ROBOT, shoulder_human, wrist_human, SCALE, R_HEADSET_TO_WORLD
        )

    def test_full_extension_lands_on_robot_reach(self):
        """An operator arm stretched to human_reach maps exactly onto robot_reach."""
        shoulder_human = np.array([0.0, 0.0, 1.4])
        wrist_human = shoulder_human + np.array([HUMAN_REACH, 0.0, 0.0])

        target = self.retarget(shoulder_human, wrist_human)

        self.assertAlmostEqual(float(np.linalg.norm(target - SHOULDER_ROBOT)), JAKA_K1_REACH, places=6)

    def test_bent_arm_scales_proportionally(self):
        """Half the human shoulder->wrist span gives half the robot span."""
        shoulder_human = np.array([0.1, -0.3, 1.2])
        bent = np.array([0.0, 0.6, -0.8])  # a non-axis-aligned, elbow-bent span
        wrist_human = shoulder_human + bent * (HUMAN_REACH / 2) / np.linalg.norm(bent)

        target = self.retarget(shoulder_human, wrist_human)

        self.assertAlmostEqual(
            float(np.linalg.norm(target - SHOULDER_ROBOT)), JAKA_K1_REACH / 2, places=6
        )

    def test_direction_is_preserved(self):
        """The retargeted direction is the VR direction rotated into the robot world."""
        shoulder_human = np.array([0.2, 0.1, 1.3])
        offset = np.array([0.31, -0.42, 0.19])
        wrist_human = shoulder_human + offset

        target = self.retarget(shoulder_human, wrist_human)
        expected_dir = R_HEADSET_TO_WORLD @ offset
        expected_dir /= np.linalg.norm(expected_dir)

        actual = target - SHOULDER_ROBOT
        np.testing.assert_allclose(actual / np.linalg.norm(actual), expected_dir, atol=1e-9)

    def test_vr_world_origin_cancels(self):
        """Only the shoulder->wrist vector matters, so the VR origin offset is irrelevant.

        This is why SWGR needs no re-anchoring on engagement and does not drift.
        """
        shoulder_human = np.array([0.4, -0.7, 1.55])
        wrist_human = shoulder_human + np.array([0.2, 0.35, -0.1])
        drift = np.array([3.7, -2.1, 0.8])

        np.testing.assert_allclose(
            self.retarget(shoulder_human, wrist_human),
            self.retarget(shoulder_human + drift, wrist_human + drift),
            atol=1e-12,
        )

    def test_swgr_scale_is_reach_ratio(self):
        """The single SWGR scale factor is robot_reach / human_reach, not a tuned gain."""
        self.assertAlmostEqual(SCALE, 0.768 / 0.58, places=6)
        for name, config in JAKA_K1_MANIPULATOR_CONFIG.items():
            swgr = config["swgr"]
            self.assertAlmostEqual(swgr["robot_reach"] / swgr["human_reach"], SCALE, places=9, msg=name)

    def test_elbow_direction_turns_off_when_arm_is_straight(self):
        shoulder = np.array([0.0, 0.0, 0.0])
        elbow = np.array([0.2, 0.0, 0.0])
        wrist = np.array([0.4, 0.0, 0.0])

        direction, radius = elbow_axis_direction(shoulder, elbow, wrist)

        self.assertIsNone(direction)
        self.assertEqual(radius, 0.0)
        self.assertEqual(adaptive_elbow_weight(radius, HUMAN_REACH), 0.0)

    def test_elbow_direction_is_perpendicular_to_shoulder_wrist_axis(self):
        shoulder = np.array([0.0, 0.0, 0.0])
        wrist = np.array([1.0, 0.0, 0.0])
        elbow = np.array([0.5, 0.2, 0.0])

        direction, radius = elbow_axis_direction(shoulder, elbow, wrist)

        np.testing.assert_allclose(direction, [0.0, 1.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(radius, 0.2)
        self.assertGreater(adaptive_elbow_weight(radius, HUMAN_REACH), 0.0)


class JakaK1ReachTest(unittest.TestCase):
    def test_configured_reach_matches_model(self):
        """robot_reach must equal the real shoulder->tool span of the loaded JAKA K1."""
        frames = _fk(np.zeros(14))  # fully stretched chain, zero offsets between joints

        for name, config in JAKA_K1_MANIPULATOR_CONFIG.items():
            swgr = config["swgr"]
            shoulder = frames[swgr["shoulder_link"]][:3, 3]
            tool = frames[config["link_name"]][:3, 3]
            self.assertAlmostEqual(
                float(np.linalg.norm(tool - shoulder)), swgr["robot_reach"], places=3, msg=name
            )


class EeRotOffsetTest(unittest.TestCase):
    """ee_rot_offset is the controller->tool alignment, calibrated against JAKA_K1_Q_INIT."""

    def test_neutral_controller_lands_on_q_init_tool_frame(self):
        frames = _fk(JAKA_K1_Q_INIT)

        for name, config in JAKA_K1_MANIPULATOR_CONFIG.items():
            rx, ry, rz = np.radians(config["swgr"]["ee_rot_offset"])
            # same composition as _update_swgr_target, with the controller at identity
            R_target = R_HEADSET_TO_WORLD @ tf.euler_matrix(rx, ry, rz, "sxyz")[:3, :3]
            R_tool = frames[config["link_name"]][:3, :3]

            angle = np.degrees(np.arccos(np.clip((np.trace(R_tool.T @ R_target) - 1) / 2, -1.0, 1.0)))
            self.assertLess(angle, 0.5, msg=f"{name} off by {angle:.3f} deg")


class _StubXrClient:
    """Stands in for XrClient so the controller path is testable without a headset."""

    def __init__(self, pose):
        self.pose = pose

    def get_pose_by_name(self, name):
        return self.pose


class _BareController(BaseTeleopController):
    """Concrete shell around BaseTeleopController; __init__ is never called."""

    def _robot_setup(self):
        pass

    def _solver_setup(self):
        pass

    def _solve_ik(self):
        pass

    def _update_robot_state(self):
        pass

    def _send_command(self):
        pass

    def _get_link_pose(self, link_name):
        raise NotImplementedError

    def run(self):
        pass


class SwgrControllerPathTest(unittest.TestCase):
    """Drives BaseTeleopController._update_swgr_target directly, with stubs for robot and XR."""

    LEFT_CONFIG = JAKA_K1_MANIPULATOR_CONFIG["left_hand"]
    REF_XYZ = np.array([0.30, 0.70, 0.40])
    LIMBS = {
        "left": {
            "shoulder": np.array([0.05, 0.18, 1.35]),
            "elbow": np.array([0.20, 0.10, 1.05]),
            "wrist": np.array([0.35, -0.10, 1.00]),
        }
    }

    def build(self, controller_pose=(0, 0, 0, 0, 0, 0, 1)):
        class _Task:
            T_world_frame = np.eye(4)

        ctrl = _BareController.__new__(_BareController)
        ctrl.R_headset_world = R_HEADSET_TO_WORLD
        ctrl.swgr_scale = {"left_hand": SCALE}
        ctrl.swgr_rot_offset = {"left_hand": np.eye(3)}
        ctrl.swgr_elbow = {}
        ctrl.swgr_elbow_direction = {}
        ctrl.swgr_elbow_weight = {}
        ctrl._swgr_elbow_direction_filtered = {}
        ctrl._swgr_elbow_weight_filtered = {}
        ctrl._grip_active = {"left_hand": False}
        ctrl.ref_ee_xyz = {"left_hand": self.REF_XYZ.copy()}
        ctrl.effector_control_mode = {"left_hand": "pose"}
        ctrl.effector_task = {"left_hand": _Task()}
        ctrl.xr_client = _StubXrClient(controller_pose)
        ctrl._get_link_pose = lambda link: (SHOULDER_ROBOT.copy(), np.array([1.0, 0, 0, 0]))
        return ctrl

    def absolute_target(self):
        limb = self.LIMBS["left"]
        return swgr_ee_position(
            SHOULDER_ROBOT, limb["shoulder"], limb["wrist"], SCALE, R_HEADSET_TO_WORLD
        )

    def test_target_is_absolute(self):
        """Every call maps straight to the retargeted pose; nothing blends from ref_ee_*."""
        ctrl = self.build()
        ctrl._update_swgr_target("left_hand", self.LEFT_CONFIG, self.LIMBS)

        np.testing.assert_allclose(ctrl.effector_task["left_hand"].T_world_frame[:3, 3],
                                  self.absolute_target(), atol=1e-9)
        # identity controller quaternion -> orientation is just the VR->world rotation
        np.testing.assert_allclose(ctrl.effector_task["left_hand"].T_world_frame[:3, :3],
                                  R_HEADSET_TO_WORLD, atol=1e-9)

    def test_elbow_is_collected_but_scaled_like_the_arm(self):
        ctrl = self.build()
        ctrl._update_swgr_target("left_hand", self.LEFT_CONFIG, self.LIMBS)

        limb = self.LIMBS["left"]
        expected = swgr_ee_position(SHOULDER_ROBOT, limb["shoulder"], limb["elbow"], SCALE, R_HEADSET_TO_WORLD)
        np.testing.assert_allclose(ctrl.swgr_elbow["left_hand"], expected, atol=1e-12)

    def test_elbow_direction_and_weight_are_collected_for_aeac(self):
        ctrl = self.build()
        ctrl._update_swgr_target("left_hand", self.LEFT_CONFIG, self.LIMBS)

        limb = self.LIMBS["left"]
        n_h, r_h = elbow_axis_direction(limb["shoulder"], limb["elbow"], limb["wrist"])
        np.testing.assert_allclose(ctrl.swgr_elbow_direction["left_hand"], R_HEADSET_TO_WORLD @ n_h, atol=1e-12)
        self.assertAlmostEqual(ctrl.swgr_elbow_weight["left_hand"], adaptive_elbow_weight(r_h, HUMAN_REACH))

    def test_elbow_direction_filter_smooths_after_first_sample(self):
        ctrl = self.build()
        ctrl.elbow_filter_alpha = 0.5
        ctrl._set_elbow_constraint("left_hand", np.array([1.0, 0.0, 0.0]), 1.0)
        ctrl._set_elbow_constraint("left_hand", np.array([0.0, 1.0, 0.0]), 0.2)

        expected = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        np.testing.assert_allclose(ctrl.swgr_elbow_direction["left_hand"], expected, atol=1e-12)
        self.assertAlmostEqual(ctrl.swgr_elbow_weight["left_hand"], 0.6)

    def test_grip_activation_has_hysteresis(self):
        ctrl = self.build()
        self.assertFalse(ctrl._grip_is_active("left_hand", 0.85))
        self.assertTrue(ctrl._grip_is_active("left_hand", 0.95))
        self.assertTrue(ctrl._grip_is_active("left_hand", 0.80))
        self.assertFalse(ctrl._grip_is_active("left_hand", 0.70))


if __name__ == "__main__":
    unittest.main()
