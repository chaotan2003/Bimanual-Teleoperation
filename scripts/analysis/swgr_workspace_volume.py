"""E1: does the SWGR scale factor actually buy reachable workspace?

Pure geometry + IK. No VR headset, no body tracking, no operator needed.

For each candidate scale s, sample the operator-commandable EE target cloud
    t = p_shoulder + s * v ,  |v| in [V_MIN, HUMAN_REACH]
run the real placo IK on it (same task setup as BaseTeleopController._placo_setup),
and report how much of that cloud the robot can actually reach. The headline number is
relative usable volume = s^3 x IK-feasible fraction: scaling up buys commandable volume
but eventually pushes targets outside the robot's reachable set, so the curve has a knee.

Usage:
    PYTHONPATH=. python scripts/analysis/swgr_workspace_volume.py [--arm right] [--n 200]
"""

import argparse
import os
import sys

import numpy as np
import placo

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.normpath(os.path.join(HERE, "..", "..", "assets", "robot_model", "urdf", "jaka_k1.urdf"))
HUMAN_REACH = 0.58
ROBOT_REACH = 0.768  # l2/r2 -> lt/rt fully extended, from the URDF
Q_INIT = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0] * 2)
SWGR_S = ROBOT_REACH / HUMAN_REACH

# ponytail: isotropic ball proxy for the human shoulder->wrist workspace. A real arm
# reaches a forward sector, so absolute volumes are conservative; the s-vs-s comparison
# is what this script is for. Swap in a sector mask before quoting absolute numbers.
V_MIN = 0.15


def sample_arm_vectors(n, rng):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = (rng.uniform(V_MIN**3, HUMAN_REACH**3, size=n) ** (1 / 3))[:, None]
    return d * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="right", choices=["left", "right"])
    ap.add_argument("--n", type=int, default=150, help="IK samples per scale factor")
    ap.add_argument("--steps", type=int, default=100, help="IK ramp steps per target")
    ap.add_argument("--scales", default="0.80,1.00,1.15,1.3241,1.50,1.70")
    args = ap.parse_args()

    prefix = args.arm[0]
    shoulder, ee = f"{prefix}2", f"{prefix}t"
    scales = [float(x) for x in args.scales.split(",")]
    rng = np.random.default_rng(0)

    robot = placo.RobotWrapper(URDF)
    solver = placo.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.mask_fbase(True)  # the URDF loads as floating base; without this the base translates
    robot.state.q[7:] = Q_INIT
    robot.update_kinematics()

    # Same task stack as BaseTeleopController._placo_setup + teleop_jaka_k1_placo.main.
    task = solver.add_position_task(ee, robot.get_T_world_frame(ee)[:3, 3])
    task.configure("ee", "soft", 1.0)
    solver.add_manipulability_task(ee, "both", 1.0).configure("manip", "soft", 1e-2)
    jt = solver.add_joints_task()
    jt.set_joints(dict(zip(robot.joint_names(), Q_INIT)))
    jt.configure("joints_regularization", "soft", 1e-4)

    def reach(target):
        """Ramp the target from the current EE pose, like the live controller does per frame.

        A single big jump diverges; a ramp converges. ponytail: warm start from the previous
        solution (as teleoperation actually does), so 'feasible' is order-dependent. The
        procedure is identical across s, so the s-vs-s comparison stays fair.
        """
        p0 = robot.get_T_world_frame(ee)[:3, 3]
        for a in np.linspace(0.0, 1.0, args.steps):
            task.target_world = p0 + (target - p0) * a
            solver.solve(True)
            robot.update_kinematics()
        return float(np.linalg.norm(robot.get_T_world_frame(ee)[:3, 3] - target))

    p_sh = robot.get_T_world_frame(shoulder)[:3, 3]
    print(f"arm={args.arm} shoulder={shoulder}@{np.round(p_sh,3).tolist()} ee={ee} n={args.n}/scale")
    print(f"{'s':>7} {'cmd radius m':>12} {'IK ok %':>8} {'mean err m':>11} {'rel. usable vol':>15}")

    rows = []
    for s in scales:
        errs = np.array([reach(p_sh + s * vi) for vi in sample_arm_vectors(args.n, rng)])
        ok = float((errs < 0.01).mean())
        # Usable workspace = commandable volume (grows as s^3) x fraction the robot can reach.
        vol = s**3 * ok
        rows.append((s, ok, errs.mean(), vol))
        print(f"{s:7.4f} {s*HUMAN_REACH:12.3f} {ok*100:8.1f} {errs.mean():11.4f} {vol:15.3f}")

    best = max(rows, key=lambda r: r[3])
    at_swgr = next((r for r in rows if abs(r[0] - SWGR_S) < 1e-3), None)
    at_one = next((r for r in rows if abs(r[0] - 1.0) < 1e-3), None)
    print(f"\nvolume-optimal scale : s={best[0]:.4f} (rel. usable vol {best[3]:.3f})")
    if at_swgr:
        print(f"SWGR configured      : s={SWGR_S:.4f} (rel. usable vol {at_swgr[3]:.3f}, {at_swgr[1]*100:.1f}% feasible)")
    if at_one and at_swgr:
        print(f"gain over s=1 baseline (B2): {at_swgr[3]/at_one[3]:.2f}x usable workspace")
    assert best[1] > 0, "no IK solution converged at any scale; check URDF/joint setup"


if __name__ == "__main__":
    main()
