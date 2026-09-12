import numpy as np

R_HEADSET_TO_WORLD = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


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
