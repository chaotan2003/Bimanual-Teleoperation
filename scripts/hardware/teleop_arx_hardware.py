import tyro
from xrobotoolkit_teleop.hardware.arx_r5_teleop_controller import (
    DEFAULT_ARX_R5_MANIPULATOR_CONFIG,
    DEFAULT_ARX_R5_URDF_PATH,
    ARXR5TeleopController,
)


def main(
    robot_urdf_path: str = DEFAULT_ARX_R5_URDF_PATH,
    scale_factor: float = 1.5,
    enable_camera: bool = True,
    enable_log_data: bool = True,
    visualize_placo: bool = False,
    control_rate_hz: int = 50,
    log_dir: str = "logs/arx_r5",
    enable_camera_compression: bool = True,
    camera_jpg_quality: int = 85,
):
    """
    Main function to run the ARX R5 teleoperation.
    """
    controller = ARXR5TeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=DEFAULT_ARX_R5_MANIPULATOR_CONFIG,
        scale_factor=scale_factor,
        enable_camera=enable_camera,
        enable_log_data=enable_log_data,
        can_ports={"right_arm": "can3"},
        visualize_placo=visualize_placo,
        control_rate_hz=control_rate_hz,
        log_dir=log_dir,
        enable_camera_compression=enable_camera_compression,
        camera_jpg_quality=camera_jpg_quality,
    )
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
