import os

# Single-thread the BLAS *before numpy is imported*: the IK is a 14-variable NLP, and
# multi-threaded OpenBLAS/MKL spends more on thread sync than on math -- measured
# p50 24 ms / p95 107 ms per IPOPT solve (~40% of 30 Hz frames over budget, visible as
# stutter) vs p50 0.9 ms / p95 1.0 ms single-threaded. See casadi_arm_ik.py docstring.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import tyro

from xrobotoolkit_teleop.simulation.casadi_arm_ik import SMOOTH_COST_WEIGHT
from xrobotoolkit_teleop.simulation.casadi_teleop_controller import (
    DEFAULT_ENGAGE_RAMP_TIME,
    DEFAULT_FREQUENCY,
    CasadiTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


JAKA_K1_URDF_PATH = os.path.join(ASSET_PATH, "robot_model/urdf/jaka_k1.urdf")
# Pinocchio joint order is the left arm (l-j1..l-j7) then the right arm (r-j1..r-j7).
# Both tools start where the operator's hands rest (0.44 m forward, 0.33 m below the
# shoulder), so engaging grip needs no blend-in: the absolute SWGR target already matches.
# NOTE: this is also the IK warm start and the posture-regularization centre, and it must
# stay inside the URDF joint limits -- q=0 does not, because the K1 elbow (l-j4/r-j4)
# limits are [-2.531, -0.052].
JAKA_K1_Q_INIT = np.array(
    [1.57, -1.57, -1.57, -1.57, 0.0, 0.0, 0.0]  # left  l-j1..l-j7
    + [-1.57, -1.57, 1.57, -1.57, 0.0, 0.0, 0.0]  # right r-j1..r-j7
)

# Operator shoulder->wrist span at full extension, in meters.
HUMAN_REACH = 0.58
# JAKA K1 shoulder (l2/r2) -> tool (lt/rt) span at full extension, in meters.
JAKA_K1_REACH = 0.768

JAKA_K1_MANIPULATOR_CONFIG = {
    "right_hand": {
        "link_name": "rt",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "swgr": {
            "shoulder_link": "r2",
            "human_reach": HUMAN_REACH,
            "robot_reach": JAKA_K1_REACH,
            # Constant controller->tool rotation offset in degrees, applied as
            # Rz(z) @ Ry(y) @ Rx(x) (meshcat "sxyz"). Derived so that a controller held at
            # its neutral VR pose reproduces the rt tool frame at JAKA_K1_Q_INIT.
            "ee_rot_offset": [-180.0, 0.0, 90.0],
        },
    },
    "left_hand": {
        "link_name": "lt",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "swgr": {
            "shoulder_link": "l2",
            "human_reach": HUMAN_REACH,
            "robot_reach": JAKA_K1_REACH,
            "ee_rot_offset": [-180.0, 0.0, -90.0],  # mirror of right_hand, same derivation
        },
    },
}


def main(
    robot_urdf_path: str = JAKA_K1_URDF_PATH,
    scale_factor: float = 1.0,  # unused by SWGR arms, which derive their own scale
    smooth_cost_weight: float = SMOOTH_COST_WEIGHT,
    frequency: float = DEFAULT_FREQUENCY,
    engage_ramp_time: float = DEFAULT_ENGAGE_RAMP_TIME,
):
    """Run JAKA K1 dual-arm teleoperation with the xr_teleoperate CasADi/IPOPT optimizer.

    End-effector poses are retargeted with Shoulder-Wrist Geometric Retargeting (SWGR),
    unchanged from the placo script: the operator's shoulder->wrist vector is scaled by
    robot_reach / human_reach and anchored at the robot shoulder, so the mapping is
    absolute and needs no re-anchoring. Requires XR body tracking (Pico Swift trackers)
    plus a controller for orientation.

    Downstream of the retargeted target the IK is the one from unitreerobotics/
    xr_teleoperate: IPOPT over a pinocchio-casadi model, trading EE pose error against
    movement from the previous solution, then a 4-tap weighted moving average.
    ``smooth_cost_weight`` (upstream 0.1) is the knob that trades tracking accuracy for
    smoothness -- raise it for calmer motion, lower it for tighter tracking.

    The placo script's 1e-4 joints task toward Q_INIT is the ``reg_cost_weight`` term
    here (upstream weight 0.02, centred on Q_INIT).
    """
    controller = CasadiTeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=JAKA_K1_MANIPULATOR_CONFIG,
        q_init=JAKA_K1_Q_INIT,
        left_frame="lt",
        right_frame="rt",
        scale_factor=scale_factor,
        smooth_cost_weight=smooth_cost_weight,
        frequency=frequency,
        engage_ramp_time=engage_ramp_time,
    )
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
