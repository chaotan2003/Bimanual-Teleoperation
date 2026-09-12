import numpy as np
import meshcat.transformations as tf

R_HEADSET_TO_WORLD = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


def is_valid_quaternion(quat, tol=1e-6):
    if not isinstance(quat, (list, tuple, np.ndarray)) or len(quat) != 4:
        return False

    if not np.all(np.isfinite(quat)):
        return False

    magnitude = np.sqrt(np.sum(np.square(quat)))
    return abs(magnitude - 1.0) <= tol


def quaternion_to_angle_axis(quat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Converts a quaternion to an angle-axis representation.

    Args:
        quat: Quaternion in (w, x, y, z) format.
        eps: Tolerance for checking if the angle is close to zero.

    Returns:
        Angle-axis vector [ax*angle, ay*angle, az*angle].
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    if q[0] < 0.0:
        q = -q

    w = q[0]
    vec_part = q[1:]

    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    if angle < eps:
        return np.zeros(3, dtype=np.float64)

    sin_half_angle = np.sin(angle / 2.0)
    if sin_half_angle < eps:
        return np.zeros(3, dtype=np.float64)

    axis = vec_part / sin_half_angle
    return axis * angle


def quat_diff_as_angle_axis(q1: np.ndarray, q2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Calculates the rotation from q1 to q2 as an angle-axis vector.

    This computes DeltaQ such that q2 = DeltaQ * q1.
    The result is the angle-axis representation of DeltaQ.

    Args:
        q1: Source quaternion (w, x, y, z).
        q2: Target quaternion (w, x, y, z).
        eps: Tolerance for small angle calculations.

    Returns:
        Angle-axis vector [ax*angle, ay*angle, az*angle] representing DeltaQ.
    """
    if not (is_valid_quaternion(q1, tol=eps) and is_valid_quaternion(q2, tol=eps)):
        print("Warning: Invalid quaternion input to calculate_rotation_error_as_angle_axis.")

    q1_inv = tf.quaternion_inverse(q1)
    delta_q = tf.quaternion_multiply(q2, q1_inv)

    return quaternion_to_angle_axis(delta_q, eps)


def swgr_ee_position(
    shoulder_robot: np.ndarray,
    shoulder_human: np.ndarray,
    wrist_human: np.ndarray,
    scale: float,
    R_world_vr: np.ndarray,
) -> np.ndarray:
    """Shoulder-Wrist Geometric Retargeting (SWGR) end-effector position.

    Retargets the whole shoulder->wrist geometry with a single scale factor instead of
    mapping upper arm and forearm separately. The elbow is not used. Only the
    shoulder->wrist *vector* enters the map, so the VR world origin cancels out and the
    result needs no clutch re-anchoring.

    Args:
        shoulder_robot: Robot shoulder anchor in the robot world frame. Shape (3,).
        shoulder_human: Operator shoulder in the VR world frame. Shape (3,).
        wrist_human: Operator wrist in the VR world frame. Shape (3,).
        scale: robot_reach / human_reach, the single SWGR scale factor.
        R_world_vr: Rotation from the VR world frame to the robot world frame. Shape (3, 3).

    Returns:
        Target end-effector position in the robot world frame. Shape (3,).
    """
    return shoulder_robot + scale * (R_world_vr @ (wrist_human - shoulder_human))


def apply_delta_pose(
    source_pos: np.ndarray,
    source_rot: np.ndarray,
    delta_pos: np.ndarray,
    delta_rot: np.ndarray,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies delta pose transformation on source pose using NumPy.

    Args:
        source_pos: Position of source frame. Shape is (3,).
        source_rot: Quaternion orientation of source frame in (w, x, y, z). Shape is (4,).
        delta_pos: Cartesian position displacement. Shape is (3,).
        delta_rot: Orientation displacement in angle-axis format. Shape is (3,).
        eps: The tolerance to consider orientation displacement as zero. Defaults to 1.0e-6.

    Returns:
        A tuple containing the displaced position and orientation frames.
        target_pos: Shape is (3,).
        target_rot: Shape is (4,).
    """
    if not (
        isinstance(source_pos, np.ndarray)
        and source_pos.shape == (3,)
        and isinstance(source_rot, np.ndarray)
        and source_rot.shape == (4,)
        and isinstance(delta_pos, np.ndarray)
        and delta_pos.shape == (3,)
        and isinstance(delta_rot, np.ndarray)
        and delta_rot.shape == (3,)
    ):
        raise ValueError(
            "Inputs must be 1D NumPy arrays with shapes: "
            "source_pos (3,), source_rot (4,), delta_pos (3,), delta_rot (3,)."
        )

    # Calculate target position
    target_pos = source_pos + delta_pos

    # Calculate target rotation
    angle = np.linalg.norm(delta_rot)
    rot_delta_quat: np.ndarray
    if angle > eps:
        axis = delta_rot / angle
        rot_delta_quat = tf.quaternion_about_axis(angle, axis)
    else:
        rot_delta_quat = np.array([1.0, 0.0, 0.0, 0.0])

    target_rot = tf.quaternion_multiply(rot_delta_quat, source_rot)

    return target_pos, target_rot
