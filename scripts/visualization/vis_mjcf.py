import argparse

import mujoco
from mujoco import viewer as mj_viewer


def main():
    parser = argparse.ArgumentParser(description="Visualize a mjcf using MuJoCo")
    parser.add_argument(
        "--xml_path",
        type=str,
        help="Path to the xml file to visualize.",
    )
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)
    # mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    viewer = mj_viewer.launch_passive(model, data)

    while True:
        mujoco.mj_step(model, data)
        viewer.sync()
    viewer.close()


if __name__ == "__main__":
    main()
