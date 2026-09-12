import argparse
import webbrowser

import placo
from placo_utils.visualization import robot_viz


def main():
    parser = argparse.ArgumentParser(description="Visualize a URDF model using Placo and Meshcat.")
    parser.add_argument(
        "--urdf_path",
        type=str,
        help="Path to the URDF file to visualize.",
    )
    args = parser.parse_args()
    robot = placo.RobotWrapper(args.urdf_path)
    print(f"robot state: {robot.state.q}")
    print(f"robot state shape: {robot.state.q.shape}")
    # robot.state.q[:7] = np.array([0, 0, 0, 1, 0, 0, 0])
    # robot.state.q[7:] = 0.5 * np.ones(robot.state.q[7:].shape)  # Set initial joint positions
    robot.update_kinematics()
    viz = robot_viz(robot)
    webbrowser.open(viz.viewer.url())

    while True:
        robot.update_kinematics()
        viz.display(robot.state.q)


if __name__ == "__main__":
    main()
