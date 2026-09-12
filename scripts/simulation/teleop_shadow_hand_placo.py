import os
import webbrowser

import numpy as np
import placo
import tyro
from dex_retargeting.constants import HandType, RetargetingType, RobotName
from placo_utils.visualization import (
    robot_viz,
)

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.dex_hand_utils import (
    DexHandTracker,
    pico_hand_state_to_mediapipe,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def main(
    urdf_path: str = os.path.join(ASSET_PATH, "shadow_hand/shadow_hand_right.urdf"),
    hand_type: str = "right",
):
    """
    Main function to run the Shadow hand teleoperation with Placo.
    """
    if hand_type not in ["left", "right"]:
        raise ValueError("hand_type must be 'left' or 'right'")

    robot = placo.RobotWrapper(urdf_path)
    viz = robot_viz(robot, "Shadow Hand")
    webbrowser.open(viz.viewer.url())

    xr_client = XrClient()

    dextracker_hand_type = HandType.left if hand_type == "left" else HandType.right
    dextracker = DexHandTracker(
        robot_name=RobotName.shadow,
        urdf_path=urdf_path,
        hand_type=dextracker_hand_type,
        retargeting_type=RetargetingType.vector,
    )
    while True:
        viz.display(robot.state.q)
        hand_state = xr_client.get_hand_tracking_state(hand_type)
        if hand_state is None:
            continue
        hand_state = np.array(hand_state)
        if np.all(hand_state == 0):
            print("all zero, ignore")
            continue
        mediapipe_hand_state = pico_hand_state_to_mediapipe(hand_state)
        qpos = dextracker.retarget(mediapipe_hand_state)
        robot.state.q[7:] = qpos


if __name__ == "__main__":
    tyro.cli(main)
