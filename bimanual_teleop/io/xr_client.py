import numpy as np


class XrClient:
    """Client for the XrClient SDK to interact with XR devices."""

    def __init__(self):
        """Initializes the XrClient and the SDK."""
        import xrobotoolkit_sdk as xrt

        self._xrt = xrt
        xrt.init()
        print("XRoboToolkit SDK initialized.")

    def get_pose_by_name(self, name: str) -> np.ndarray:
        """Returns the pose of the specified device by name.
        Valid names: "left_controller", "right_controller", "headset".
        Pose is [x, y, z, qx, qy, qz, qw]."""
        xrt = self._xrt
        if name == "left_controller":
            return xrt.get_left_controller_pose()
        elif name == "right_controller":
            return xrt.get_right_controller_pose()
        elif name == "headset":
            return xrt.get_headset_pose()
        else:
            raise ValueError(
                f"Invalid name: {name}. Valid names are: 'left_controller', 'right_controller', 'headset'."
            )

    def get_key_value_by_name(self, name: str) -> float:
        """Returns the trigger/grip value by name (float).
        Valid names: "left_trigger", "right_trigger", "left_grip", "right_grip".
        """
        xrt = self._xrt
        if name == "left_trigger":
            return xrt.get_left_trigger()
        elif name == "right_trigger":
            return xrt.get_right_trigger()
        elif name == "left_grip":
            return xrt.get_left_grip()
        elif name == "right_grip":
            return xrt.get_right_grip()
        else:
            raise ValueError(
                f"Invalid name: {name}. Valid names are: 'left_trigger', 'right_trigger', 'left_grip', 'right_grip'."
            )

    def get_body_tracking_data(self) -> dict | None:
        """Returns body joint poses, or None if unavailable.

        Returns:
            Dict with key 'poses': (24, 7) array [x,y,z,qx,qy,qz,qw] for each joint.

        Only poses are fetched. Velocity/acceleration are also published by the SDK, but
        the sole caller (BaseTeleopController._read_body_upper_limbs) reads 'poses' only,
        and each xrt.get_* is a pybind11 round-trip to the PICO PC Service on the teleop
        hot path. Add them back here if a caller ever needs them.
        """
        xrt = self._xrt
        if not xrt.is_body_data_available():
            return None

        return {"poses": xrt.get_body_joints_pose()}

    def close(self):
        self._xrt.close()
