import tyro

from bimanual_teleop.robots.jaka_k1 import (
    JAKA_K1_MANIPULATOR_CONFIG,
    JAKA_K1_Q_INIT,
    JAKA_K1_Q_INIT_JOINT_NAMES,
    JAKA_K1_URDF_PATH,
)
from bimanual_teleop.runtime.viser_jparse_controller import (
    DEFAULT_ELBOW_DEADBAND,
    DEFAULT_ELBOW_FILTER_ALPHA,
    DEFAULT_ELBOW_GAIN,
    DEFAULT_ELBOW_WEIGHT_MAX,
    DEFAULT_ENABLE_OUTPUT_INTERPOLATION,
    DEFAULT_ENGAGE_RAMP_TIME,
    DEFAULT_FREQUENCY,
    DEFAULT_GAMMA,
    DEFAULT_GRIP_OFF_THRESHOLD,
    DEFAULT_GRIP_ON_THRESHOLD,
    DEFAULT_MAX_JOINT_VELOCITY,
    DEFAULT_NULLSPACE_GAIN,
    DEFAULT_ORIENTATION_FILTER_ALPHA,
    DEFAULT_ORIENTATION_GAIN,
    DEFAULT_OUTPUT_FREQUENCY,
    DEFAULT_POSITION_FILTER_BETA,
    DEFAULT_POSITION_FILTER_MIN_CUTOFF,
    DEFAULT_POSITION_GAIN,
    DEFAULT_VISUALIZATION_FREQUENCY,
    ViserJparseController,
)


def main(
    robot_urdf_path: str = JAKA_K1_URDF_PATH,
    frequency: float = DEFAULT_FREQUENCY,
    engage_ramp_time: float = DEFAULT_ENGAGE_RAMP_TIME,
    gamma: float = DEFAULT_GAMMA,
    nullspace_gain: float = DEFAULT_NULLSPACE_GAIN,
    max_joint_velocity: float = DEFAULT_MAX_JOINT_VELOCITY,
    position_gain: float = DEFAULT_POSITION_GAIN,
    orientation_gain: float = DEFAULT_ORIENTATION_GAIN,
    position_filter_min_cutoff: float = DEFAULT_POSITION_FILTER_MIN_CUTOFF,
    position_filter_beta: float = DEFAULT_POSITION_FILTER_BETA,
    orientation_filter_alpha: float = DEFAULT_ORIENTATION_FILTER_ALPHA,
    elbow_gain: float = DEFAULT_ELBOW_GAIN,
    elbow_deadband: float = DEFAULT_ELBOW_DEADBAND,
    elbow_weight_max: float = DEFAULT_ELBOW_WEIGHT_MAX,
    elbow_filter_alpha: float = DEFAULT_ELBOW_FILTER_ALPHA,
    grip_on_threshold: float = DEFAULT_GRIP_ON_THRESHOLD,
    grip_off_threshold: float = DEFAULT_GRIP_OFF_THRESHOLD,
    output_frequency: float = DEFAULT_OUTPUT_FREQUENCY,
    enable_output_interpolation: bool = DEFAULT_ENABLE_OUTPUT_INTERPOLATION,
    visualization_frequency: float = DEFAULT_VISUALIZATION_FREQUENCY,
):
    """Run JAKA K1 dual-arm teleoperation with SWGR, AEAC, J-PARSE IK and Viser."""
    controller = ViserJparseController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=JAKA_K1_MANIPULATOR_CONFIG,
        q_init=JAKA_K1_Q_INIT,
        q_init_joint_names=JAKA_K1_Q_INIT_JOINT_NAMES,
        left_frame="lt",
        right_frame="rt",
        frequency=frequency,
        engage_ramp_time=engage_ramp_time,
        gamma=gamma,
        nullspace_gain=nullspace_gain,
        max_joint_velocity=max_joint_velocity,
        position_gain=position_gain,
        orientation_gain=orientation_gain,
        position_filter_min_cutoff=position_filter_min_cutoff,
        position_filter_beta=position_filter_beta,
        orientation_filter_alpha=orientation_filter_alpha,
        elbow_gain=elbow_gain,
        elbow_deadband=elbow_deadband,
        elbow_weight_max=elbow_weight_max,
        elbow_filter_alpha=elbow_filter_alpha,
        grip_on_threshold=grip_on_threshold,
        grip_off_threshold=grip_off_threshold,
        output_frequency=output_frequency,
        enable_output_interpolation=enable_output_interpolation,
        visualization_frequency=visualization_frequency,
    )
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
