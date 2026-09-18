from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from jax.typing import ArrayLike


def _elbow_axis_direction(poses: jnp.ndarray, shoulder_idx: int, elbow_idx: int, wrist_idx: int) -> jnp.ndarray:
    shoulder = poses[shoulder_idx, 4:]
    elbow = poses[elbow_idx, 4:]
    wrist = poses[wrist_idx, 4:]
    axis = wrist - shoulder
    d = axis / jnp.maximum(jnp.linalg.norm(axis), 1e-9)
    v = elbow - (shoulder + d * jnp.dot(d, elbow - shoulder))
    return v / jnp.maximum(jnp.linalg.norm(v), 1e-9)


def compute_jacobian(
    robot: Any,
    cfg: ArrayLike,
    target_link_index: int,
    anchor_pose: ArrayLike | None = None,
) -> jnp.ndarray:
    cfg = jnp.asarray(cfg)
    if anchor_pose is None:
        anchor_pose = robot.forward_kinematics(cfg)[target_link_index]
    r_anchor_inv = jaxlie.SE3(anchor_pose).rotation().inverse()

    def pose_components(q):
        pose = jaxlie.SE3(robot.forward_kinematics(q)[target_link_index])
        return jnp.concatenate([pose.translation(), (pose.rotation() @ r_anchor_inv).log()])

    return jax.jacfwd(pose_components)(cfg)


def jparse_pseudoinverse(
    jacobian: ArrayLike,
    gamma: float,
    singular_gains: ArrayLike,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    J = jnp.asarray(jacobian)
    n = J.shape[1]
    U, S, Vt = jnp.linalg.svd(J, full_matrices=True)
    k = S.shape[0]
    sigma_max = jnp.maximum(jnp.max(S), 1e-9)
    threshold = jnp.maximum(gamma * sigma_max, 1e-9)
    non_singular = S > threshold
    S_safety = jnp.where(non_singular, S, threshold)
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    V_k = Vt_k.T

    phi = jnp.where(non_singular, 0.0, S / threshold)
    inv_regular = jnp.where(non_singular, 1.0 / S_safety, 0.0)
    J_parse = (V_k * inv_regular[None, :]) @ U_k.T
    J_parse = J_parse + (V_k * (phi / S_safety)[None, :]) @ (U_k.T * jnp.asarray(singular_gains)[None, :])
    return J_parse, jnp.eye(n) - V_k @ Vt_k, S


def stacked_jparse_step(
    robot: Any,
    cfg: ArrayLike,
    target_link_indices: list[int],
    target_positions: np.ndarray,
    target_wxyzs: np.ndarray,
    *,
    gamma: float = 0.1,
    position_gain: float = 5.0,
    orientation_gain: float = 1.0,
    singular_direction_gain_position: float = 1.0,
    singular_direction_gain_angular: float = 1.0,
    nullspace_gain: float = 0.5,
    max_joint_velocity: float = 3.14,
    dt: float = 0.02,
    home_cfg: ArrayLike | None = None,
    elbow_constraints: list[tuple[int, int, int, np.ndarray, float]] | None = None,
    elbow_gain: float = 0.5,
) -> tuple[np.ndarray, dict]:
    cfg = jnp.asarray(cfg)
    poses = robot.forward_kinematics(cfg)
    rows = []
    desired = []
    gains = []
    pos_errors = []
    ori_errors = []

    for link_idx, target_position, target_wxyz in zip(
        target_link_indices, target_positions, target_wxyzs, strict=True
    ):
        pose = jaxlie.SE3(poses[link_idx])
        pos_error = jnp.asarray(target_position) - pose.translation()

        target_q = jnp.asarray(target_wxyz)
        target_q = target_q / jnp.linalg.norm(target_q)
        current_q = pose.rotation().wxyz
        target_q = jnp.where(jnp.dot(target_q, current_q) < 0.0, -target_q, target_q)
        omega_error = (jaxlie.SO3(target_q) @ pose.rotation().inverse()).log()
        omega_norm = jnp.linalg.norm(omega_error)
        omega_error = jnp.where(omega_norm > 1.0, omega_error / jnp.maximum(omega_norm, 1e-9), omega_error)

        rows.append(compute_jacobian(robot, cfg, link_idx, poses[link_idx]))
        desired.append(jnp.concatenate([position_gain * pos_error, orientation_gain * omega_error]))
        gains.extend([singular_direction_gain_position] * 3 + [singular_direction_gain_angular] * 3)
        pos_errors.append(jnp.linalg.norm(pos_error))
        ori_errors.append(jnp.linalg.norm(omega_error))

    J = jnp.vstack(rows)
    v_des = jnp.concatenate(desired)
    J_inv, N, singular_values = jparse_pseudoinverse(J, gamma, jnp.asarray(gains))
    dq = J_inv @ v_des

    dq_null = jnp.zeros_like(cfg)
    if nullspace_gain > 0.0:
        home = (robot.joints.lower_limits + robot.joints.upper_limits) / 2.0
        if home_cfg is not None:
            home = jnp.asarray(home_cfg)
        dq_null = dq_null - nullspace_gain * (cfg - home)

    elbow_costs = []
    if elbow_gain > 0.0 and elbow_constraints:
        for shoulder_idx, elbow_idx, wrist_idx, target_direction, weight in elbow_constraints:
            target = jnp.asarray(target_direction)
            target = target / jnp.maximum(jnp.linalg.norm(target), 1e-9)

            def elbow_cost(q):
                n_r = _elbow_axis_direction(robot.forward_kinematics(q), shoulder_idx, elbow_idx, wrist_idx)
                return float(weight) * jnp.sum((n_r - target) ** 2)

            cost = elbow_cost(cfg)
            elbow_costs.append(cost)
            dq_null = dq_null - elbow_gain * jax.grad(elbow_cost)(cfg)

    if nullspace_gain > 0.0 or elbow_costs:
        dq = dq + N @ dq_null

    max_raw = jnp.max(jnp.abs(dq))
    velocity_scale = jnp.minimum(1.0, max_joint_velocity / jnp.maximum(max_raw, 1e-9))
    dq = dq * velocity_scale
    new_cfg = jnp.clip(cfg + dq * dt, robot.joints.lower_limits, robot.joints.upper_limits)
    info = {
        "position_error": float(jnp.max(jnp.asarray(pos_errors))),
        "orientation_error": float(jnp.max(jnp.asarray(ori_errors))),
        "elbow_cost": float(jnp.max(jnp.asarray(elbow_costs))) if elbow_costs else 0.0,
        "elbow_count": len(elbow_constraints or []),
        "elbow_weight": max((float(c[-1]) for c in (elbow_constraints or [])), default=0.0),
        "max_joint_vel": float(max_raw),
        "velocity_scale": float(velocity_scale),
        "inverse_condition_number": float(jnp.min(singular_values) / jnp.maximum(jnp.max(singular_values), 1e-9)),
    }
    return np.asarray(new_cfg), info
