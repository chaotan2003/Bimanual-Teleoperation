import os

import tyro
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import (
    MujocoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def main(
    xml_path: str = os.path.join(ASSET_PATH, "galaxea/A1X/scene.xml"),
    robot_urdf_path: str = os.path.join(ASSET_PATH, "galaxea/A1X/dual_a1x_fixed_gripper.urdf"),
    scale_factor: float = 1.5,
    visualize_placo: bool = True,
):
    """
    Main function to run the dual A1X teleoperation in MuJoCo.
    """
    config = {
        "right_hand": {
            "link_name": "right_arm_link6",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "vis_target": "right_target",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "right_trigger",
                "joint_names": [
                    "right_gripper_finger_joint1",
                ],
                "open_pos": [
                    0.05,
                ],
                "close_pos": [
                    0.0,
                ],
            },
        },
        "left_hand": {
            "link_name": "left_arm_link6",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "vis_target": "left_target",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "left_trigger",
                "joint_names": [
                    "left_gripper_finger_joint1",
                ],
                "open_pos": [
                    0.05,
                ],
                "close_pos": [
                    0.0,
                ],
            },
        },
    }

    # Create and initialize the teleoperation controller
    controller = MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
    )

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
