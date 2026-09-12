import os

import tyro

from xrobotoolkit_teleop.simulation.placo_teleop_controller import (
    PlacoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def main(
    robot_urdf_path: str = os.path.join(ASSET_PATH, "unitree/g1/g1_dual_arm.urdf"),
    scale_factor: float = 1,
):
    """
    Main function to run the Unitree G1 dual arm teleoperation with Placo visualization.
    """
    # Define dual arm configuration for Unitree G1
    config = {
        "left_arm": {
            "link_name": "left_rubber_hand",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "motion_tracker": {
                "serial": "PC2310BLH9020707B",
                "link_target": "left_elbow_link",
            },
        },
        "right_arm": {
            "link_name": "right_rubber_hand",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "motion_tracker": {
                "serial": "PC2310BLH9020740B",
                "link_target": "right_elbow_link",
            },
        },
    }

    # Create and initialize the teleoperation controller
    controller = PlacoTeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
    )

    # Add joint regularization task to keep arms in natural position
    joints_task = controller.solver.add_joints_task()

    # Define default joint positions for natural arm pose
    default_joints = {
        # Left arm default positions (slightly bent, natural pose)
        "left_shoulder_pitch_joint": 0.3,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 1.0,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        # Right arm default positions (mirrored)
        "right_shoulder_pitch_joint": 0.3,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 1.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }

    joints_task.set_joints(default_joints)
    joints_task.configure("joints_regularization", "soft", 1e-4)

    print("Starting Unitree G1 dual arm teleoperation...")
    print("Control mapping:")
    print("  - Left controller -> Left arm (left_rubber_hand)")
    print("  - Right controller -> Right arm (right_rubber_hand)")
    print("  - Hold grip buttons to activate arm control")

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
