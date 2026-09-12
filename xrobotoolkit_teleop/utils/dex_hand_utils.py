from pathlib import Path
from typing import Optional

import numpy as np
from dex_retargeting.constants import (
    OPERATOR2MANO,
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig

mediapipe_to_pico = {
    0: 1,  # WRIST -> Wrist
    1: 2,  # THUMB_CMC -> Thumb_metacarpal
    2: 3,  # THUMB_MCP -> Thumb_proximal
    3: 4,  # THUMB_IP  -> Thumb_distal
    4: 5,  # THUMB_TIP -> Thumb_tip
    5: 7,  # INDEX_FINGER_MCP -> Index_proximal
    6: 8,  # INDEX_FINGER_PIP -> Index_intermediate
    7: 9,  # INDEX_FINGER_DIP -> Index_distal
    8: 10,  # INDEX_FINGER_TIP -> Index_tip
    9: 12,  # MIDDLE_FINGER_MCP -> Middle_proximal
    10: 13,  # MIDDLE_FINGER_PIP -> Middle_intermediate
    11: 14,  # MIDDLE_FINGER_DIP -> Middle_distal
    12: 15,  # MIDDLE_FINGER_TIP -> Middle_tip
    13: 17,  # RING_FINGER_MCP -> Ring_proximal
    14: 18,  # RING_FINGER_PIP -> Ring_intermediate
    15: 19,  # RING_FINGER_DIP -> Ring_distal
    16: 20,  # RING_FINGER_TIP -> Ring_tip
    17: 22,  # PINKY_MCP -> Little_proximal
    18: 23,  # PINKY_PIP -> Little_intermediate
    19: 24,  # PINKY_DIP -> Little_distal
    20: 25,  # PINKY_TIP -> Little_tip
}

pico_to_mediapipe = {
    1: 0,  # Wrist
    2: 1,  # Thumb_metacarpal -> THUMB_CMC
    3: 2,  # Thumb_proximal   -> THUMB_MCP
    4: 3,  # Thumb_distal     -> THUMB_IP
    5: 4,  # Thumb_tip        -> THUMB_TIP
    7: 5,  # Index_proximal   -> INDEX_FINGER_MCP
    8: 6,  # Index_intermediate -> INDEX_FINGER_PIP
    9: 7,  # Index_distal     -> INDEX_FINGER_DIP
    10: 8,  # Index_tip        -> INDEX_FINGER_TIP
    12: 9,  # Middle_proximal  -> MIDDLE_FINGER_MCP
    13: 10,  # Middle_intermediate -> MIDDLE_FINGER_PIP
    14: 11,  # Middle_distal    -> MIDDLE_FINGER_DIP
    15: 12,  # Middle_tip       -> MIDDLE_FINGER_TIP
    17: 13,  # Ring_proximal    -> RING_FINGER_MCP
    18: 14,  # Ring_intermediate -> RING_FINGER_PIP
    19: 15,  # Ring_distal      -> RING_FINGER_DIP
    20: 16,  # Ring_tip         -> RING_FINGER_TIP
    22: 17,  # Little_proximal  -> PINKY_MCP
    23: 18,  # Little_intermediate -> PINKY_PIP
    24: 19,  # Little_distal    -> PINKY_DIP
    25: 20,  # Little_tip       -> PINKY_TIP
}

"""
PICO Index,MediaPipe Index,Joint Name,Description,OpenXR Enum
0,,Palm,The central point in the palm,XR_HAND_JOINT_PALM_EXT
1,0,Wrist,The joint of the wrist,XR_HAND_JOINT_WRIST_EXT
2,1,Thumb_metacarpal,The metacarpal joint of the thumb,XR_HAND_JOINT_THUMB_METACARPAL_EXT
3,2,Thumb_proximal,The proximal joint of the thumb,XR_HAND_JOINT_THUMB_PROXIMAL_EXT
4,3,Thumb_distal,The distal joint of the thumb,XR_HAND_JOINT_THUMB_DISTAL_EXT
5,4,Thumb_tip,The fingertip of the thumb,XR_HAND_JOINT_THUMB_TIP_EXT
6,,Index_metacarpal,The metacarpal joint of the index finger,XR_HAND_JOINT_INDEX_METACARPAL_EXT
7,5,Index_proximal,The proximal joint of the index finger,XR_HAND_JOINT_INDEX_PROXIMAL_EXT
8,6,Index_intermediate,The intermediate joint of the index finger,XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT
9,7,Index_distal,The distal joint of the index finger,XR_HAND_JOINT_INDEX_DISTAL_EXT
10,8,Index_tip,The fingertip of the index finger,XR_HAND_JOINT_INDEX_TIP_EXT
11,,Middle_metacarpal,The metacarpal joint of the middle finger,XR_HAND_JOINT_MIDDLE_METACARPAL_EXT
12,9,Middle_proximal,The proximal joint of the middle finger,XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT
13,10,Middle_intermediate,The intermediate joint of the middle finger,XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT
14,11,Middle_distal,The distal joint of the middle finger,XR_HAND_JOINT_MIDDLE_DISTAL_EXT
15,12,Middle_tip,The fingertip of the middle finger,XR_HAND_JOINT_MIDDLE_TIP_EXT
16,,Ring_metacarpal,The metacarpal joint of the ring finger,XR_HAND_JOINT_RING_METACARPAL_EXT
17,13,Ring_proximal,The proximal joint of the ring finger,XR_HAND_JOINT_RING_PROXIMAL_EXT
18,14,Ring_intermediate,The intermediate joint of the ring finger,XR_HAND_JOINT_RING_INTERMEDIATE_EXT
19,15,Ring_distal,The distal joint of the ring finger,XR_HAND_JOINT_RING_DISTAL_EXT
20,16,Ring_tip,The fingertip of the ring finger,XR_HAND_JOINT_RING_TIP_EXT
21,,Little_metacarpal,The metacarpal joint of the little finger,XR_HAND_JOINT_LITTLE_METACARPAL_EXT
22,17,Little_proximal,The proximal joint of the little finger,XR_HAND_JOINT_LITTLE_PROXIMAL_EXT
23,18,Little_intermediate,The intermediate joint of the little finger,XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT
24,19,Little_distal,The distal joint of the little finger,XR_HAND_JOINT_LITTLE_DISTAL_EXT
25,20,Little_tip,The fingertip of the little finger,XR_HAND_JOINT_LITTLE_TIP_EXT
"""
# Create the data as a list of dictionaries
data = [
    {
        "PICO Index": 0,
        "MediaPipe Index": "",
        "Joint Name": "Palm",
        "Description": "The central point in the palm",
        "OpenXR Enum": "XR_HAND_JOINT_PALM_EXT",
    },
    {
        "PICO Index": 1,
        "MediaPipe Index": 0,
        "Joint Name": "Wrist",
        "Description": "The joint of the wrist",
        "OpenXR Enum": "XR_HAND_JOINT_WRIST_EXT",
    },
    {
        "PICO Index": 2,
        "MediaPipe Index": 1,
        "Joint Name": "Thumb_metacarpal",
        "Description": "The metacarpal joint of the thumb",
        "OpenXR Enum": "XR_HAND_JOINT_THUMB_METACARPAL_EXT",
    },
    {
        "PICO Index": 3,
        "MediaPipe Index": 2,
        "Joint Name": "Thumb_proximal",
        "Description": "The proximal joint of the thumb",
        "OpenXR Enum": "XR_HAND_JOINT_THUMB_PROXIMAL_EXT",
    },
    {
        "PICO Index": 4,
        "MediaPipe Index": 3,
        "Joint Name": "Thumb_distal",
        "Description": "The distal joint of the thumb",
        "OpenXR Enum": "XR_HAND_JOINT_THUMB_DISTAL_EXT",
    },
    {
        "PICO Index": 5,
        "MediaPipe Index": 4,
        "Joint Name": "Thumb_tip",
        "Description": "The fingertip of the thumb",
        "OpenXR Enum": "XR_HAND_JOINT_THUMB_TIP_EXT",
    },
    {
        "PICO Index": 6,
        "MediaPipe Index": "",
        "Joint Name": "Index_metacarpal",
        "Description": "The metacarpal joint of the index finger",
        "OpenXR Enum": "XR_HAND_JOINT_INDEX_METACARPAL_EXT",
    },
    {
        "PICO Index": 7,
        "MediaPipe Index": 5,
        "Joint Name": "Index_proximal",
        "Description": "The proximal joint of the index finger",
        "OpenXR Enum": "XR_HAND_JOINT_INDEX_PROXIMAL_EXT",
    },
    {
        "PICO Index": 8,
        "MediaPipe Index": 6,
        "Joint Name": "Index_intermediate",
        "Description": "The intermediate joint of the index finger",
        "OpenXR Enum": "XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT",
    },
    {
        "PICO Index": 9,
        "MediaPipe Index": 7,
        "Joint Name": "Index_distal",
        "Description": "The distal joint of the index finger",
        "OpenXR Enum": "XR_HAND_JOINT_INDEX_DISTAL_EXT",
    },
    {
        "PICO Index": 10,
        "MediaPipe Index": 8,
        "Joint Name": "Index_tip",
        "Description": "The fingertip of the index finger",
        "OpenXR Enum": "XR_HAND_JOINT_INDEX_TIP_EXT",
    },
    {
        "PICO Index": 11,
        "MediaPipe Index": "",
        "Joint Name": "Middle_metacarpal",
        "Description": "The metacarpal joint of the middle finger",
        "OpenXR Enum": "XR_HAND_JOINT_MIDDLE_METACARPAL_EXT",
    },
    {
        "PICO Index": 12,
        "MediaPipe Index": 9,
        "Joint Name": "Middle_proximal",
        "Description": "The proximal joint of the middle finger",
        "OpenXR Enum": "XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT",
    },
    {
        "PICO Index": 13,
        "MediaPipe Index": 10,
        "Joint Name": "Middle_intermediate",
        "Description": "The intermediate joint of the middle finger",
        "OpenXR Enum": "XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT",
    },
    {
        "PICO Index": 14,
        "MediaPipe Index": 11,
        "Joint Name": "Middle_distal",
        "Description": "The distal joint of the middle finger",
        "OpenXR Enum": "XR_HAND_JOINT_MIDDLE_DISTAL_EXT",
    },
    {
        "PICO Index": 15,
        "MediaPipe Index": 12,
        "Joint Name": "Middle_tip",
        "Description": "The fingertip of the middle finger",
        "OpenXR Enum": "XR_HAND_JOINT_MIDDLE_TIP_EXT",
    },
    {
        "PICO Index": 16,
        "MediaPipe Index": "",
        "Joint Name": "Ring_metacarpal",
        "Description": "The metacarpal joint of the ring finger",
        "OpenXR Enum": "XR_HAND_JOINT_RING_METACARPAL_EXT",
    },
    {
        "PICO Index": 17,
        "MediaPipe Index": 13,
        "Joint Name": "Ring_proximal",
        "Description": "The proximal joint of the ring finger",
        "OpenXR Enum": "XR_HAND_JOINT_RING_PROXIMAL_EXT",
    },
    {
        "PICO Index": 18,
        "MediaPipe Index": 14,
        "Joint Name": "Ring_intermediate",
        "Description": "The intermediate joint of the ring finger",
        "OpenXR Enum": "XR_HAND_JOINT_RING_INTERMEDIATE_EXT",
    },
    {
        "PICO Index": 19,
        "MediaPipe Index": 15,
        "Joint Name": "Ring_distal",
        "Description": "The distal joint of the ring finger",
        "OpenXR Enum": "XR_HAND_JOINT_RING_DISTAL_EXT",
    },
    {
        "PICO Index": 20,
        "MediaPipe Index": 16,
        "Joint Name": "Ring_tip",
        "Description": "The fingertip of the ring finger",
        "OpenXR Enum": "XR_HAND_JOINT_RING_TIP_EXT",
    },
    {
        "PICO Index": 21,
        "MediaPipe Index": "",
        "Joint Name": "Little_metacarpal",
        "Description": "The metacarpal joint of the little finger",
        "OpenXR Enum": "XR_HAND_JOINT_LITTLE_METACARPAL_EXT",
    },
    {
        "PICO Index": 22,
        "MediaPipe Index": 17,
        "Joint Name": "Little_proximal",
        "Description": "The proximal joint of the little finger",
        "OpenXR Enum": "XR_HAND_JOINT_LITTLE_PROXIMAL_EXT",
    },
    {
        "PICO Index": 23,
        "MediaPipe Index": 18,
        "Joint Name": "Little_intermediate",
        "Description": "The intermediate joint of the little finger",
        "OpenXR Enum": "XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT",
    },
    {
        "PICO Index": 24,
        "MediaPipe Index": 19,
        "Joint Name": "Little_distal",
        "Description": "The distal joint of the little finger",
        "OpenXR Enum": "XR_HAND_JOINT_LITTLE_DISTAL_EXT",
    },
    {
        "PICO Index": 25,
        "MediaPipe Index": 20,
        "Joint Name": "Little_tip",
        "Description": "The fingertip of the little finger",
        "OpenXR Enum": "XR_HAND_JOINT_LITTLE_TIP_EXT",
    },
]


def pico_hand_state_to_mediapipe(hand_state: np.ndarray) -> np.ndarray:
    """
    Converts a hand state from PICO format to MediaPipe format.

    Args:
        hand_state: A numpy array of shape (27, 7) representing the PICO hand state.
                    Each row is [x, y, z, qx, qy, qz, qw] for each joint.

    Returns:
        A numpy array of shape (21, 3) representing the MediaPipe hand joint positions.
    """
    mediapipe_state = np.zeros((21, 3), dtype=float)
    for pico_idx, mediapipe_idx in pico_to_mediapipe.items():
        mediapipe_state[mediapipe_idx] = hand_state[pico_idx, :3]
    return mediapipe_state - mediapipe_state[0:1, :]  # Center at wrist

def calc_hand_position(open_pos, close_pos, percentage):
    if not (0.0 <= percentage <= 1.0):
        raise ValueError("Percentage must be between 0.0 and 1.0.")
    position = open_pos.copy()
    for i in range(len(open_pos) - 1):
        position[i] = open_pos[i] + (close_pos[i] - open_pos[i]) * percentage
    return position

def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Compute the 3D coordinate frame (orientation only) from detected 3d key points
    :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
    :return: the coordinate frame of wrist in MANO convention
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram–Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / (np.linalg.norm(x) + 1e-6)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame


class DexHandTracker:
    def __init__(
        self,
        robot_name: RobotName,
        urdf_path: str,
        retargeting_type: RetargetingType,
        hand_type: HandType,
    ):
        self.robot_name = robot_name
        self.retargeting_type = retargeting_type
        self.hand_type = hand_type
        self.urdf_path = urdf_path

        self.config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
        self.OPERATOR2MANO = OPERATOR2MANO[hand_type]

        # Set the default URDF directory for the retargeting library
        robot_dir = Path(urdf_path).parent.parent
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        # Build the retargeting module from the specified config
        self.retargeting = RetargetingConfig.load_from_file(self.config_path).build()

    def retarget(self, hand_pos: np.ndarray, wrist_rot: np.ndarray = None) -> Optional[np.ndarray]:
        """
        Retargets human hand joint positions to robot hand joint positions (qpos).

        Args:
            hand_pos: A numpy array of shape (21, 7) representing the 3D positions
                      of the 21 MediaPipe hand joints. It is recommended to center
                      the positions at the wrist (joint 0).
            wrist_rot: A numpy array representing the rotation matrix of the wrist
                       in the form of a 3x3 rotation matrix. This is used to align
                       the hand pose with the robot's coordinate system.

        Returns:
            A numpy array representing the target qpos for the robot hand,
            or None if retargeting fails.
        """
        if hand_pos is None or hand_pos.shape != (21, 3):
            return None

        # # Estimate the wrist orientation frame from the hand points
        if wrist_rot is None:
            # If wrist rotation is not provided, estimate it from the hand points
            wrist_rot = estimate_frame_from_hand_points(hand_pos)
        # mediapipe_wrist_rot = self.detector.estimate_frame_from_hand_points(hand_pos)

        # Transform the hand points into the MANO frame, which is the reference for retargeting
        transformed_pos = hand_pos @ wrist_rot @ self.OPERATOR2MANO

        # Prepare the reference values for the optimizer based on retargeting type
        indices = self.retargeting.optimizer.target_link_human_indices
        if self.retargeting_type == RetargetingType.position:
            ref_value = transformed_pos[indices, :]
        elif self.retargeting_type == RetargetingType.vector:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = transformed_pos[task_indices, :] - transformed_pos[origin_indices, :]
        else:
            raise NotImplementedError(f"Retargeting type {self.retargeting_type} is not supported.")

        try:
            # Perform the retargeting to get the robot's joint positions
            qpos = self.retargeting.retarget(ref_value)
            return qpos
        except (RuntimeWarning, RuntimeError) as e:
            # Catch potential numerical issues in the optimizer and return None
            print(f"Warning: Retargeting failed with an error: {e}")
            return None
