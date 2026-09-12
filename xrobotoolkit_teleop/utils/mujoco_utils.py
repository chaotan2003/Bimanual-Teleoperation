import mujoco
import numpy as np
import pinocchio as pin
import placo


def calc_mujoco_qpos_from_placo_q(
    mujoco_model: mujoco.MjModel,
    placo_robot: placo.RobotWrapper,
    placo_q: np.ndarray,
    floating_base: bool = False,
) -> np.ndarray:
    """
    Convert Placo joint configuration to MuJoCo qpos.

    Args:
        mujoco_model: The MuJoCo model.
        placo_robot: The Placo robot wrapper.
        placo_q: The joint configuration in Placo format.

    Returns:
        np.ndarray: The corresponding MuJoCo qpos.
    """
    mujoco_qpos = calc_mujoco_qpos_from_pin_q(mujoco_model, placo_robot.model, placo_q, floating_base)

    return mujoco_qpos


def calc_mujoco_qpos_from_pin_q(
    mujoco_model: mujoco.MjModel,
    pin_model: pin.Model,
    pin_q: np.ndarray,
    floating_base: bool = False,
) -> np.ndarray:
    """
    Convert Pinocchio joint configuration to MuJoCo qpos.

    Args:
        mujoco_model: The MuJoCo model.
        pin_robot: pinocchio robot model.
        pin_q: The joint configuration in Pinocchio format.

    Returns:
        np.ndarray: The corresponding MuJoCo qpos.
    """
    mujoco_qpos = np.zeros(mujoco_model.nq)
    if floating_base:
        # Check if the root joint is free floating
        if pin_model.joints[1].shortname() != "JointModelFreeFlyer":
            raise ValueError("Expected free floating root joint but found different joint type")
        mujoco_qpos[:3] = pin_q[:3]
        mujoco_qpos[3:7] = mujoco_quat_from_pin_quat(pin_q[3:7])

    pin_joint_names = [name for name in pin_model.names if name != "root_joint" and name != "universe"]

    # Start index for actuated joints in pin_q, depends on floating_base
    if (not floating_base) and pin_model.joints[1].shortname() == "JointModelFreeFlyer":
        pin_q_offset = 7
    else:
        pin_q_offset = 0

    for i, pin_joint_name in enumerate(pin_joint_names):
        # Placo q for actuated joints starts after the floating base if it exists
        pin_joint_value = pin_q[pin_q_offset + i]

        success = set_mujoco_joint_pos_by_name(
            mujoco_model,
            mujoco_qpos,
            pin_joint_name,
            pin_joint_value,
        )
        if not success:
            raise ValueError(f"Joint '{pin_joint_name}' not found in MuJoCo model.")

    return mujoco_qpos


def calc_placo_q_from_mujoco_qpos(
    mujoco_model: mujoco.MjModel,
    placo_robot: placo.RobotWrapper,
    mujoco_qpos: np.ndarray,
    floating_base: bool = False,
) -> np.ndarray:
    """
    Convert MuJoCo qpos to Placo joint configuration.

    Args:
        mujoco_model: The MuJoCo model.
        mujoco_qpos: The joint configuration in MuJoCo format.

    Returns:
        np.ndarray: The corresponding Placo joint configuration.
    """
    placo_q = calc_pin_q_from_mujoco_qpos(
        mujoco_model,
        placo_robot.model,
        mujoco_qpos,
        floating_base=floating_base,
    )

    return placo_q


def calc_pin_q_from_mujoco_qpos(
    mujoco_model: mujoco.MjModel,
    pin_model: pin.Model,
    mujoco_qpos: np.ndarray,
    floating_base: bool = False,
) -> np.ndarray:
    """
    Convert MuJoCo qpos to Placo joint configuration.

    Args:
        mujoco_model: The MuJoCo model.
        mujoco_qpos: The joint configuration in MuJoCo format.

    Returns:
        np.ndarray: The corresponding Placo joint configuration.
    """
    pin_q = np.zeros(pin_model.nq)

    if floating_base:
        # Check if the root joint is free floating
        if pin_model.joints[1].shortname() != "JointModelFreeFlyer":
            raise ValueError("Expected free floating root joint but found different joint type")
        pin_q[:3] = mujoco_qpos[:3]  # Position (x, y, z)
        pin_q[3:7] = pin_quat_from_mujoco_quat(mujoco_qpos[3:7])
    else:
        pin_q[:7] = np.array([0, 0, 0, 0, 0, 0, 1])

    pin_joint_names = [name for name in pin_model.names if name != "root_joint" and name != "universe"]
    # Start index for actuated joints in pin_q, depends on floating_base
    if (not floating_base) and pin_model.joints[1].shortname() == "JointModelFreeFlyer":
        pin_q_offset = 7
    else:
        pin_q_offset = 0
    for i, joint_name in enumerate(pin_joint_names):
        mujoco_joint_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if mujoco_joint_id != -1:
            qpos_addr = mujoco_model.jnt_qposadr[mujoco_joint_id]
            if qpos_addr < len(mujoco_qpos):
                pin_q[i + pin_q_offset] = mujoco_qpos[qpos_addr]

    return pin_q


def set_mujoco_joint_pos_by_name(
    mujoco_model: mujoco.MjModel,
    qpos: np.ndarray,
    joint_name: str,
    joint_pos: float,
) -> None:
    """
    Set the position of a MuJoCo joint by its name.

    Args:
        mujoco_model: The MuJoCo model.
        mujoco_data: The MuJoCo data.
        joint_name: The name of the joint to set.
        joint_pos: The desired position for the joint.
    """
    joint_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id == -1:
        return False  # Joint not found

    qpos_addr = mujoco_model.jnt_qposadr[joint_id]
    qpos[qpos_addr] = joint_pos
    return True  # Joint position set successfully


def calc_mujoco_ctrl_from_qpos(mujoco_model: mujoco.MjModel, mujoco_qpos: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo qpos to control signals.

    Args:
        mujoco_model: The MuJoCo model.
        mujoco_qpos: The joint configuration in MuJoCo format.

    Returns:
        np.ndarray: The corresponding control signals.
    """
    mujoco_ctrl = np.zeros(mujoco_model.nu)
    for i in range(mujoco_model.nu):
        # Get the ID of the transmission target (e.g., joint ID) for actuator i
        target_id = mujoco_model.actuator_trnid[i, 0]

        # Assuming the actuator targets a joint (common case for robot arms)
        # and that the joint is 1-DOF.
        qpos_addr = mujoco_model.jnt_qposadr[target_id]

        # Assign the corresponding qpos value to the control signal
        mujoco_ctrl[i] = mujoco_qpos[qpos_addr]

    return mujoco_ctrl


def pin_quat_from_mujoco_quat(mujoco_quat: np.ndarray) -> np.ndarray:
    """
    Convert a MuJoCo quaternion to a Placo quaternion.

    Args:
        mujoco_quat: The quaternion in MuJoCo format (w, x, y, z).

    Returns:
        np.ndarray: The corresponding Placo quaternion (x, y, z, w).
    """
    if len(mujoco_quat) != 4:
        raise ValueError("MuJoCo quaternion must have 4 elements (w, x, y, z).")

    return np.array([mujoco_quat[1], mujoco_quat[2], mujoco_quat[3], mujoco_quat[0]])


def mujoco_quat_from_pin_quat(pin_quat: np.ndarray) -> np.ndarray:
    """
    Convert a Placo quaternion to a MuJoCo quaternion.

    Args:
        pin_quat: The quaternion in Placo format (x, y, z, w).

    Returns:
        np.ndarray: The corresponding MuJoCo quaternion (w, x, y, z).
    """
    if len(pin_quat) != 4:
        raise ValueError("Placo quaternion must have 4 elements (x, y, z, w).")

    return np.array([pin_quat[3], pin_quat[0], pin_quat[1], pin_quat[2]])
