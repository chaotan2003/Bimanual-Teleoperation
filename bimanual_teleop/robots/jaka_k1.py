import os

import numpy as np

from bimanual_teleop.robots.paths import ASSET_PATH


JAKA_K1_URDF_PATH = os.path.join(ASSET_PATH, "robot_model/urdf/jaka_k1.urdf")

# The kinematics backend joint order is left arm (l-j1..l-j7), then right arm (r-j1..r-j7).
# This is the IK warm start and posture-regularization center; q=0 violates the K1
# elbow limits, so keep this calibrated inside the URDF limits.
JAKA_K1_Q_INIT = np.array(
    [1.57, -1.57, -1.57, -1.57, 0.0, 0.0, 0.0]
    + [-1.57, -1.57, 1.57, -1.57, 0.0, 0.0, 0.0]
)
JAKA_K1_Q_INIT_JOINT_NAMES = [f"l-j{i}" for i in range(1, 8)] + [f"r-j{i}" for i in range(1, 8)]

HUMAN_REACH = 0.58
JAKA_K1_REACH = 0.768

JAKA_K1_MANIPULATOR_CONFIG = {
    "right_hand": {
        "link_name": "rt",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "swgr": {
            "shoulder_link": "r2",
            "elbow_link": "r4",
            "human_reach": HUMAN_REACH,
            "robot_reach": JAKA_K1_REACH,
            "ee_rot_offset": [-180.0, 0.0, 90.0],
        },
    },
    "left_hand": {
        "link_name": "lt",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "swgr": {
            "shoulder_link": "l2",
            "elbow_link": "l4",
            "human_reach": HUMAN_REACH,
            "robot_reach": JAKA_K1_REACH,
            "ee_rot_offset": [-180.0, 0.0, -90.0],
        },
    },
}
