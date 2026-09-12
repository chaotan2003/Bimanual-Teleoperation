"""CasADi + IPOPT inverse kinematics for fixed-base dual arms.

Ported from unitreerobotics/xr_teleoperate ``teleop/robot_control/robot_arm_ik.py``
(``R1_A7_ArmIK``). The objective, its weights, the IPOPT options and the
weighted-moving post-filter are the upstream ones.

PERFORMANCE: solve time is dominated by BLAS thread synchronization, not math. The NLP
is tiny (14 variables), so a multi-threaded OpenBLAS/MKL makes each IPOPT call cost
p50 24 ms / p95 107 ms -- over the 33 ms frame budget on ~40% of cycles, which is the
stutter this port was tuned against. With ``OMP_NUM_THREADS=1`` (+ MKL/OPENBLAS
variants) the same solve is p50 0.9 ms / p95 1.0 ms. The entry script
``scripts/simulation/teleop_jaka_k1_casadi.py`` sets those before importing numpy;
any new entry point using this module must do the same.

Upstream builds the NLP with ``casadi.Opti`` over ``casadi.MX`` symbols. The
conda-forge ``pinocchio`` casadi bindings (3.1.0 / 4.1.0, both checked) only accept
``casadi.SX``, and CasADi's ``Opti`` cannot hold SX variables, so the *same* NLP is
expressed with ``casadi.nlpsol`` -- the formulation upstream itself keeps as its
legacy alternative in ``R1_A5_ArmIK_Legacy``. Remaining deviations:

  * the casadi model is reconstructed joint-by-joint from the numeric pinocchio
    model, because ``pinocchio.casadi.buildModelFromUrdf`` is not exposed in the
    conda-forge bindings. Bodies are attached with zero inertia: the IK is
    kinematics-only, so inertia never enters the objective.
    Verified against numeric pinocchio FK: max abs deviation 3.3e-16 over 30
    random configurations.
  * no floating base is appended. ``appendModel(..., "root_joint")`` is how upstream
    reaches ``nq = 6 + <arm dofs>``; this robot is fixed base, so ``nq`` stays
    ``<arm dofs>`` and ``q_meas`` / targets are not offset by 6.
  * ``logging_mp`` -> stdlib ``logging``; no ``pickle`` model cache (measured
    cold build: 0.12 s total, NLP included).
  * no torque output: ``rnea`` is only meaningful against a real robot, and this
    port targets simulation. ``solve_ik`` returns ``(q, None)`` so the upstream
    call signature is unchanged.
  * the posture regularization keeps upstream's weight (0.02) but is centred on
    ``q_ref`` (default ``q_init``) instead of zero. Upstream regularizes toward
    zero because zero is a sane mid-posture for their arms; it is not merely a bad
    centre for this robot but an infeasible one (the JAKA K1 elbow limits exclude
    j4 = 0), so toward-zero would park both elbows on a joint limit.
  * casadi flattens a 4x4 parameter column-major; ``pack_params`` matches that.
    Filling it row-major scrambles the target rotation, which makes ``log3`` take
    an ``acos`` outside [-1, 1] and IPOPT abort with ``Invalid_Number_Detected``
    at iteration 0 -- a silent failure, so the packing is unit-tested.
"""

import logging
import time
from typing import Dict, Optional, Sequence, Tuple

import casadi as ca
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from xrobotoolkit_teleop.utils.weighted_moving_filter import DEFAULT_WEIGHTS, WeightedMovingFilter

logger = logging.getLogger(__name__)

# Upstream objective weights (robot_arm_ik.py: set_translational_cost /
# set_rotational_cost / set_regularization_cost / set_smooth_cost).
TRANS_COST_WEIGHT = 50.0
ROT_COST_WEIGHT = 1.0
REGULARIZATION_COST_WEIGHT = 0.02
SMOOTH_COST_WEIGHT = 0.1

# Upstream IPOPT options (robot_arm_ik.py). ``detect_simple_bounds`` and
# ``calc_lam_p`` are casadi.Opti internals with no nlpsol equivalent: variable
# bounds are handed to IPOPT natively via lbx/ubx instead.
# Do NOT add ``ipopt.hessian_approximation: limited-memory``: measured ~200 ms per
# iteration on this build (~2 s per solve). Do NOT lower max_iter/tolerances to buy
# speed -- the real cost is BLAS threading (see module docstring); with it fixed,
# solves take ~1 ms and converge in 2-3 iterations.
IPOPT_OPTIONS = {
    "ipopt.max_iter": 30,
    "ipopt.tol": 1e-4,
    "ipopt.acceptable_tol": 5e-4,
    "ipopt.acceptable_iter": 5,
    "ipopt.warm_start_init_point": "yes",
    "ipopt.print_level": 0,
    "print_time": False,
    "expand": True,
}

FILTER_WEIGHTS = DEFAULT_WEIGHTS

_REVOLUTE_JOINTS = {
    "JointModelRX": cpin.JointModelRX,
    "JointModelRY": cpin.JointModelRY,
    "JointModelRZ": cpin.JointModelRZ,
}


def build_casadi_model(urdf_path: str) -> Tuple["cpin.Model", pin.Model]:
    """Build the symbolic (casadi) pinocchio model and its numeric twin from a URDF."""
    numeric = pin.buildModelFromUrdf(urdf_path)
    model = cpin.Model()
    model.name = numeric.name

    joint_ids = {0: 0}  # numeric joint id -> casadi joint id ("universe" maps to itself)
    for i in range(1, numeric.njoints):
        kind = numeric.joints[i].shortname()
        if kind not in _REVOLUTE_JOINTS:
            raise NotImplementedError(f"joint {numeric.names[i]!r} is {kind}; only revolute joints are supported")
        joint_id = model.addJoint(
            joint_ids[numeric.parents[i]],
            _REVOLUTE_JOINTS[kind](),
            cpin.SE3(numeric.jointPlacements[i]),
            numeric.names[i],
        )
        model.appendBodyToJoint(joint_id, cpin.Inertia.Zero(), cpin.SE3.Identity())
        model.addJointFrame(joint_id)
        joint_ids[i] = joint_id

    known = set(numeric.names)  # joint frames were already added by addJointFrame
    for frame in numeric.frames:
        if frame.name in known:
            continue
        known.add(frame.name)
        model.addFrame(
            cpin.Frame(
                frame.name,
                joint_ids[frame.parentJoint],
                cpin.SE3(frame.placement),
                getattr(cpin.FrameType, str(frame.type).split(".")[-1]),
            )
        )
    return model, numeric


class DualArmCasadiIK:
    """Two-arm 6-DoF IK solved by IPOPT, warm-started from the previous solution."""

    def __init__(
        self,
        urdf_path: str,
        left_frame: str,
        right_frame: str,
        q_init: Sequence[float],
        smooth_cost_weight: float = SMOOTH_COST_WEIGHT,
        trans_cost_weight: float = TRANS_COST_WEIGHT,
        rot_cost_weight: float = ROT_COST_WEIGHT,
        reg_cost_weight: float = REGULARIZATION_COST_WEIGHT,
        q_ref: Optional[Sequence[float]] = None,
        filter_weights: Sequence[float] = FILTER_WEIGHTS,
        ipopt_options: Optional[Dict] = None,
    ):
        self.cmodel, self.model = build_casadi_model(urdf_path)
        self.cdata = self.cmodel.createData()
        self.data = self.model.createData()
        self.nq = self.cmodel.nq
        self.left_id = self.cmodel.getFrameId(left_frame)
        self.right_id = self.cmodel.getFrameId(right_frame)
        self.left_frame = left_frame
        self.right_frame = right_frame

        self.q_init = np.asarray(q_init, dtype=float).ravel()
        if self.q_init.shape[0] != self.nq:
            raise ValueError(f"q_init has {self.q_init.shape[0]} values, model has nq={self.nq}")

        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()
        # A warm start outside the bounds makes IPOPT abort before iterating, and the
        # JAKA K1 elbow limits exclude zero, so a zero q_init is a real trap here.
        if not np.all((self.lower <= self.q_init) & (self.q_init <= self.upper)):
            bad = np.where(~((self.lower <= self.q_init) & (self.q_init <= self.upper)))[0]
            names = list(self.model.names)[1:]
            raise ValueError(
                "q_init violates joint limits: "
                + ", ".join(f"{names[i]}={self.q_init[i]:.3f} not in "
                            f"[{self.lower[i]:.3f}, {self.upper[i]:.3f}]" for i in bad)
            )

        self.smooth_cost_weight = float(smooth_cost_weight)
        self.q_ref = self.q_init.copy() if q_ref is None else np.asarray(q_ref, dtype=float).ravel()
        if self.q_ref.shape[0] != self.nq:
            raise ValueError(f"q_ref has {self.q_ref.shape[0]} values, model has nq={self.nq}")
        self.filter = WeightedMovingFilter(filter_weights, self.nq)
        self.last_q = self.q_init.copy()

        start = time.perf_counter()
        self.solver = self._build_nlp(
            trans_cost_weight, rot_cost_weight, reg_cost_weight, self.smooth_cost_weight,
            ipopt_options or IPOPT_OPTIONS,
        )
        logger.info("CasADi IK built in %.2f s (nq=%d, frames=%s/%s)",
                    time.perf_counter() - start, self.nq, left_frame, right_frame)

    # ------------------------------------------------------------------ NLP
    def _build_nlp(self, trans_w: float, rot_w: float, reg_w: float, smooth_w: float, opts: Dict):
        q = ca.SX.sym("q", self.nq)
        q_last = ca.SX.sym("q_last", self.nq)
        left_target = ca.SX.sym("left_target", 4, 4)
        right_target = ca.SX.sym("right_target", 4, 4)

        cpin.framesForwardKinematics(self.cmodel, self.cdata, q)
        left = self.cdata.oMf[self.left_id]
        right = self.cdata.oMf[self.right_id]

        cost = (
            trans_w * ca.sumsqr(left.translation - left_target[0:3, 3])
            + trans_w * ca.sumsqr(right.translation - right_target[0:3, 3])
            + rot_w * ca.sumsqr(cpin.log3(left.rotation @ left_target[0:3, 0:3].T))
            + rot_w * ca.sumsqr(cpin.log3(right.rotation @ right_target[0:3, 0:3].T))
            + reg_w * ca.sumsqr(q - self.q_ref)
            + smooth_w * ca.sumsqr(q - q_last)
        )
        nlp = {
            "x": q,
            "f": cost,
            "p": ca.vertcat(q_last, left_target[:], right_target[:]),
        }
        return ca.nlpsol("dual_arm_ik", "ipopt", nlp, opts)

    @staticmethod
    def pack_params(q_last: np.ndarray, left_target: np.ndarray, right_target: np.ndarray) -> np.ndarray:
        """Flatten the parameter vector exactly the way ``ca.vertcat(M[:])`` expects.

        casadi is column-major, so 4x4 targets must be raveled with ``order='F'``.
        """
        return np.concatenate(
            [
                np.asarray(q_last, dtype=float).ravel(),
                np.asarray(left_target, dtype=float).ravel(order="F"),
                np.asarray(right_target, dtype=float).ravel(order="F"),
            ]
        )

    # ---------------------------------------------------------------- solve
    def solve_ik(
        self,
        left_target: np.ndarray,
        right_target: np.ndarray,
        q_meas: Optional[np.ndarray] = None,
        dq_meas: Optional[np.ndarray] = None,  # noqa: ARG002 - upstream signature
    ) -> Tuple[np.ndarray, None]:
        """Solve for both arms at once. ``q_meas`` is the warm start (motor state upstream)."""
        q_start = self.last_q if q_meas is None else np.asarray(q_meas, dtype=float).ravel()
        params = self.pack_params(q_start, left_target, right_target)
        try:
            result = self.solver(x0=q_start, p=params, lbx=self.lower, ubx=self.upper)
            stats = self.solver.stats()
            if not stats.get("success", False):
                logger.warning("IK did not converge (%s, iter=%s); holding last solution",
                               stats.get("return_status"), stats.get("iter_count"))
                q_sol = q_start
            else:
                q_sol = np.array(result["x"]).ravel()
        except Exception:
            logger.exception("IK solve failed; holding last solution")
            q_sol = q_start

        self.filter.add_data(q_sol)
        filtered = self.filter.get_filtered_data()
        self.last_q = filtered.copy()
        return filtered, None

    # ------------------------------------------------------------- forward kinematics
    def forward_kinematics(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        """Frame name -> 4x4 world transform, for every frame in the model."""
        pin.framesForwardKinematics(self.model, self.data, np.asarray(q, dtype=float).ravel())
        return {f.name: self.data.oMf[i].homogeneous.copy() for i, f in enumerate(self.model.frames)}

    def frame_pose(self, q: np.ndarray, frame_name: str) -> np.ndarray:
        pin.framesForwardKinematics(self.model, self.data, np.asarray(q, dtype=float).ravel())
        return self.data.oMf[self.model.getFrameId(frame_name)].homogeneous.copy()
