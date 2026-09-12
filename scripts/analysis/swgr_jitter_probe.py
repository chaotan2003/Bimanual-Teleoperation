"""Offline probe: why does JAKA K1 / SWGR teleop jerk (跳变)?

Replays a *smooth* intended end-effector trajectory through the real SWGR + Placo
pipeline (no headset, no meshcat) and measures what comes out in joint space. The
intended motion is back-mapped into fake human shoulder/wrist samples that are
band-limited to the tracker rate and carry tracker-like noise, so anything violent
in the output is produced by the pipeline itself, not by the operator.

Loop order mirrors BaseTeleopController._update_ik exactly:
update_kinematics() -> read trackers -> set task target -> solve(True).

Run:  python scripts/analysis/swgr_jitter_probe.py
"""

import os

import meshcat.transformations as tf
import numpy as np
import placo

from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD, swgr_ee_position
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

URDF = os.path.join(ASSET_PATH, "robot_model/urdf/jaka_k1.urdf")
DT = 0.01  # IK loop period, as in PlacoTeleopController
TRACKER_HZ = 30.0  # Pico Swift body-tracking publish rate (assumed)
SCALE = 0.768 / 0.58  # robot_reach / human_reach, as in teleop_jaka_k1_placo.py
Q_INIT = np.array([1.57, -1.57, -1.57, -1.57, 0.0, 0.0, 0.0, -1.57, -1.57, 1.57, -1.57, 0.0, 0.0, 0.0])
T_END = 6.0
NOISE_POS = 0.002  # 2 mm gaussian tracker noise
NOISE_ROT_DEG = 1.0  # hand tremor on the controller quaternion
SHOULDER_HUMAN = np.array([0.0, 0.0, 1.4])  # VR-world operator shoulder
EE_VMAX = 1.0  # m/s, slew-rate clamp used by the "slew" variant


def intended_ee(t, ee0, offset=0.0):
    """The smooth motion the operator actually asks for (~0.3 m/s peak).

    offset shifts the whole path: the operator's hand is not exactly at the nominal
    rest pose that Q_INIT encodes, which is what happens at every grip engagement.
    """
    return ee0 + np.array([0.15 * np.sin(2 * np.pi * 0.3 * t), 0.0, 0.08 * np.sin(2 * np.pi * 0.5 * t) + offset])


def human_wrist(ee_target, shoulder_robot):
    """Inverse of swgr_ee_position: robot-world EE target -> VR-world wrist sample."""
    return SHOULDER_HUMAN + R_HEADSET_TO_WORLD.T @ ((ee_target - shoulder_robot) / SCALE)


class Build:
    """Solver configuration, mirroring _placo_setup + teleop_jaka_k1_placo.py."""

    def __init__(self, variant):
        self.variant = variant
        self.robot = placo.RobotWrapper(URDF)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.dt = DT
        self.solver.mask_fbase(True)
        self.robot.state.q[7:] = Q_INIT
        self.robot.update_kinematics()

        self.R0 = self.robot.get_T_world_frame("lt")[:3, :3].copy()
        self.ee0 = self.robot.get_T_world_frame("lt")[:3, 3].copy()

        self.task = self.solver.add_frame_task("lt", self.robot.get_T_world_frame("lt"))
        self.task.configure("left_hand", "soft", 1.0)
        mani = self.solver.add_manipulability_task("lt", "both", 1.0)
        mani.configure("manipulability", "soft", 1e-2)
        # The real controller tasks both arms; pin the right one where it starts so the
        # probe measures the left arm's response instead of an unconstrained nullspace.
        rt = self.solver.add_frame_task("rt", self.robot.get_T_world_frame("rt"))
        rt.configure("right_hand", "soft", 1.0)
        self.joints = self.solver.add_joints_task()
        self.joints.set_joints({j: v for j, v in zip(self.robot.joint_names(), Q_INIT)})
        self.joints.configure("joints_regularization", "soft", 1e-1 if "strong_posture" in variant else 1e-4)

        if "joint_limits" in variant:
            self.solver.enable_joint_limits(True)
        if "velocity_limits" in variant:
            # The URDF declares velocity="0" for every joint, so the limits must be set
            # explicitly or enable_velocity_limits() freezes the robot completely.
            for j in self.robot.joint_names():
                self.robot.set_velocity_limit(j, 2.0)
            self.solver.enable_velocity_limits(True)
        for w in ("kin1e-6", "kin1e-3", "kin1e-2"):
            if w in variant:
                self.solver.add_kinetic_energy_regularization_task(float(w[3:]))
        # Controller quaternion (VR frame) that reproduces R0 through R_headset_world @ R_ctrl.
        T_ctrl0 = np.eye(4)
        T_ctrl0[:3, :3] = R_HEADSET_TO_WORLD.T @ self.R0
        self.q_ctrl0 = tf.quaternion_from_matrix(T_ctrl0)


def run(variant, seed=0, engage_offset=0.0, trace=False):
    b = Build(variant)
    rng = np.random.default_rng(seed)
    shoulder_robot = b.robot.get_T_world_frame("l2")[:3, 3].copy()
    ema = None
    alpha = 0.2  # ~8 Hz cutoff at a 100 Hz loop
    every = max(1, int(round((1.0 / TRACKER_HZ) / DT)))
    wrist = q_ctrl = None
    prev_target = b.ee0.copy()

    qs, ees, want = [], [], []
    n = int(T_END / DT)
    for k in range(n):
        t = k * DT
        b.robot.update_kinematics()  # what _update_ik does before reading poses
        if k % every == 0:  # trackers publish at TRACKER_HZ; the SDK returns the latest sample
            ee_want = intended_ee(t, b.ee0, engage_offset)
            wrist = human_wrist(ee_want, shoulder_robot) + rng.normal(0, NOISE_POS, 3)
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            rot = tf.quaternion_about_axis(np.radians(rng.normal(0, NOISE_ROT_DEG)), axis)
            q_ctrl = tf.quaternion_multiply(rot, b.q_ctrl0)

        xyz = swgr_ee_position(shoulder_robot, SHOULDER_HUMAN, wrist, SCALE, R_HEADSET_TO_WORLD)
        if "ema" in variant:
            xyz = xyz if ema is None else alpha * xyz + (1 - alpha) * ema
            ema = xyz
        if "slew" in variant:
            d = xyz - prev_target
            step = np.linalg.norm(d)
            if step > EE_VMAX * DT:
                xyz = prev_target + d * (EE_VMAX * DT / step)
        prev_target = xyz

        T = np.eye(4)
        T[:3, :3] = R_HEADSET_TO_WORLD @ tf.quaternion_matrix(q_ctrl)[:3, :3]
        T[:3, 3] = xyz
        b.task.T_world_frame = T

        try:
            b.solver.solve(True)
        except RuntimeError as e:
            if trace:
                print("    solve failed:", e)
        b.robot.update_kinematics()
        qs.append(b.robot.state.q[7:].copy())
        ees.append(b.robot.get_T_world_frame("lt")[:3, 3].copy())
        want.append(intended_ee(t, b.ee0, engage_offset))
        if trace and k < 12:
            print(
                f"    t={t:5.3f} target={np.round(xyz,3)} ee={np.round(ees[-1],3)} "
                f"max|dq|={np.degrees(np.abs(b.robot.state.q[7:] - (qs[-2] if len(qs) > 1 else qs[-1])).max()):7.2f} deg"
            )

    qs = np.array(qs)
    ees = np.array(ees)
    want = np.array(want)
    dq = np.diff(qs, axis=0)
    d2q = np.diff(dq, axis=0)
    ee_vel = np.linalg.norm(np.diff(ees, axis=0), axis=1) / DT
    return {
        "peak_joint_vel_deg/s": np.degrees(np.abs(dq).max()) / DT,
        "p99_joint_vel_deg/s": np.degrees(np.percentile(np.abs(dq), 99)) / DT,
        "peak_ee_vel_m/s": ee_vel.max(),
        "jerk_rms_deg/s2": np.degrees(np.sqrt((d2q**2).mean())) / DT**2,
        "ee_rms_err_mm": 1e3 * np.sqrt(((ees - want) ** 2).sum(1).mean()),
        "ee_max_err_mm": 1e3 * np.linalg.norm(ees - want, axis=1).max(),
        "elbow_wander_deg": float(np.degrees(qs[:, [2, 4, 6]].std(0).sum())),
    }


VARIANTS = [
    "baseline",
    "kin1e-6",  # what dual_arm_ur_controller.py uses (hardware path)
    "kin1e-3",
    "kin1e-2",
    "ema",
    "slew",
    "velocity_limits",
    "strong_posture",
    "ema+velocity_limits",
    "slew+ema+velocity_limits",
    "slew+ema+velocity_limits+joint_limits+strong_posture",
]

if __name__ == "__main__":
    for engage in (0.0, 0.10):
        print(f"\n{'='*78}\nSWGR engage offset = {engage*100:.0f} cm (hand not at the nominal rest pose)\n{'='*78}")
        keys = None
        for v in VARIANTS:
            m = run(v, engage_offset=engage)
            keys = keys or list(m)
            print(f"\n### {v}")
            for k in keys:
                print(f"    {k:26s} {m[k]:.4g}")
