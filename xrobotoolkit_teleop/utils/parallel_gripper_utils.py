DEFAULT_OPEN_POS = 0.0
DEFAULT_CLOSE_POS = 1.0


def calc_parallel_gripper_position(open_pos: float, close_pos: float, percentage: float) -> float:
    if not (0.0 <= percentage <= 1.0):
        raise ValueError("Percentage must be between 0.0 and 1.0.")
    return open_pos + (close_pos - open_pos) * percentage
