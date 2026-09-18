import math

import numpy as np


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).ravel()
    if q.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    return q / np.linalg.norm(q)


def smoothing_factor(t_e: float, cutoff: np.ndarray | float) -> np.ndarray | float:
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.t_prev = None
        self.x_prev = None
        self.dx_prev = None

    def reset(self) -> None:
        self.t_prev = None
        self.x_prev = None
        self.dx_prev = None

    def next(self, t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.t_prev is None:
            self.t_prev = float(t)
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()
        if x.shape != self.x_prev.shape:
            raise ValueError("Unexpected data shape")

        t_e = max(float(t) - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / t_e
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.t_prev = float(t)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat.copy()


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quat(q0 + alpha * (q1 - q0))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - alpha) * theta) / sin_theta * q0
        + math.sin(alpha * theta) / sin_theta * q1
    )


class LPRotationFilter:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.y = None

    def reset(self) -> None:
        self.y = None

    def next(self, q_wxyz: np.ndarray) -> np.ndarray:
        q_wxyz = _normalize_quat(q_wxyz)
        if self.y is None:
            self.y = q_wxyz
        else:
            self.y = quaternion_slerp(self.y, q_wxyz, self.alpha)
        return self.y.copy()


class PoseFilter:
    def __init__(self, position_min_cutoff: float, position_beta: float, orientation_alpha: float):
        self.position = OneEuroFilter(position_min_cutoff, position_beta)
        self.orientation = LPRotationFilter(orientation_alpha)

    def reset(self) -> None:
        self.position.reset()
        self.orientation.reset()

    def next(self, t: float, T: np.ndarray, matrix_to_wxyz, wxyz_to_matrix) -> np.ndarray:
        out = np.asarray(T, dtype=float).copy()
        out[:3, 3] = self.position.next(t, out[:3, 3])
        out[:3, :3] = wxyz_to_matrix(self.orientation.next(matrix_to_wxyz(out)))
        return out
