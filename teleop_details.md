# Detailed descriptions on the python teleoperation examples

## XR client 
- [`common/xr_client.py`](xrobotoolkit_teleop/common/xr_client.py)
- connects to XR device using the python binding of [xrobotoolkit-sdk](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind)

## CasADi / IPOPT (xr_teleoperate optimizer)

The JAKA K1 arm IK uses the optimizer from
[unitreerobotics/xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate)
(`teleop/robot_control/robot_arm_ik.py`), ported to
[`casadi_arm_ik.py`](xrobotoolkit_teleop/simulation/casadi_arm_ik.py) and driven by
[`CasadiTeleopController`](xrobotoolkit_teleop/simulation/casadi_teleop_controller.py)
(meshcat visualization, no physics backend):

- IPOPT solves one NLP for both arms at ~30 Hz: weighted EE position error (50.0) plus
  rotation error via `log3` (1.0) plus posture regularization toward `q_ref` (0.02) plus
  movement from the previous solution (0.1, the `--smooth-cost-weight` knob), with the
  URDF joint limits as bounds and the previous solution as warm start.
- The solution passes through a 4-tap weighted moving average
  ([`weighted_moving_filter.py`](xrobotoolkit_teleop/utils/weighted_moving_filter.py),
  upstream weights `[0.4, 0.3, 0.2, 0.1]`).
- Deviations from upstream (SX-only conda-forge casadi bindings so the NLP uses
  `casadi.nlpsol`, fixed base, no model cache, regularization centred on `q_init`
  because `q=0` violates the K1 elbow limits) are listed in the `casadi_arm_ik.py`
  module docstring.

This stack needs its own conda environment: pinocchio's casadi bindings are only shipped
by conda-forge.

```bash
conda create -n bimanual-casadi -c conda-forge python=3.10 casadi=3.6.7 pinocchio=3.1.0 -y
conda activate bimanual-casadi
pip install meshcat tyro
cd dependencies/XRoboToolkit-PC-Service-Pybind && bash setup_ubuntu.sh && cd ../..
pip install -e .
```

Run the offline checks with:

```bash
python -m unittest tests.test_casadi_ik tests.test_jaka_k1_model tests.test_swgr_retargeting
```

## Shoulder-Wrist Geometric Retargeting (SWGR)

SWGR is an absolute retargeting mode for the end-effector pose. XR body tracking supplies
the operator's shoulder, elbow and wrist positions, but only the shoulder and the wrist
enter the end-effector map: unlike two-segment retargeting, which scales the upper arm and
the forearm separately and reassembles them, SWGR retargets the whole shoulder→wrist
geometry with a single scale factor derived from the two maximum arm spans.

```
v         = wrist_vr - shoulder_vr                        # operator shoulder→wrist vector
s         = robot_reach / human_reach                     # JAKA K1: 0.768 / 0.58 = 1.3241
ee_target = shoulder_robot + s * (R_headset_world @ v)    # anchored at the robot shoulder
R_target  = R_headset_world @ R_controller @ R_ee_offset  # orientation, absolute
```

Because only the shoulder→wrist *vector* is used, the VR world origin cancels out. The
mapping therefore needs no reference pose captured at engagement, does not accumulate
drift, and stays correct if the operator's torso moves. `control_trigger` remains an
enable switch: releasing it freezes the last target, and a body-tracking dropout freezes
it too (no fallback to a different mapping mid-motion). On engagement the target is
blended in over `engage_ramp_time` (0.3 s, linear geodesic SE(3) blend) from the pose the
arm actually held, so the robot does not snap to the absolute target.

The elbow is read and retargeted with the same geometry into `controller.swgr_elbow[name]`
for logging, but is deliberately not fed to the IK.

Requirements: XR body tracking (Pico Swift trackers) for position, plus a controller for
orientation. Enabled per manipulator, currently wired up for the JAKA K1 dual arm:

```bash
conda activate bimanual-casadi
python scripts/simulation/teleop_jaka_k1_casadi.py
```

On the first body-tracking frame the controller prints `|shoulder→wrist|` per side (must be
≤ `human_reach`) and the shoulder width (≈ 0.35 m). If those are off, the body-tracking
frame convention differs from the controller frame and needs an extra transform.

## Manipulator config

A teleoperation task is one entry per end effector in the `manipulator_config` dict. Only
`.urdf` is required; there is no physics backend, so no `.xml`.

| Key | Meaning |
|---|---|
| `link_name` | End-effector link as named in the URDF. Also the frame the IK tracks. |
| `pose_source` | XR pose source read by `XrClient`, e.g. `left_controller`. Supplies orientation. |
| `control_trigger` | Key that enables the arm, e.g. `left_grip`. Release freezes the last target. |
| `control_mode` | Optional. `"pose"` (default, full 6DOF) or `"position"` (3DOF, orientation held). |
| `swgr` | **Required.** Switches the manipulator to absolute Shoulder-Wrist Geometric Retargeting. |

`swgr` sub-keys:

| Key | Meaning |
|---|---|
| `shoulder_link` | Robot link used as the shoulder anchor (`l2` on JAKA K1). |
| `human_reach` | Operator shoulder→wrist span at full extension, metres. |
| `robot_reach` | `shoulder_link`→`link_name` span at full extension, metres. Asserted against forward kinematics by `tests/test_swgr_retargeting.py::JakaK1ReachTest`. |
| `ee_rot_offset` | Optional constant controller→tool rotation offset in degrees `[rx, ry, rz]`, static XYZ. Default `[0, 0, 0]`. |

The JAKA K1 config in
[`teleop_jaka_k1_casadi.py`](scripts/simulation/teleop_jaka_k1_casadi.py) is the worked
example:

```python
JAKA_K1_MANIPULATOR_CONFIG = {
    "right_hand": {
        "link_name": "rt",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "swgr": {
            "shoulder_link": "r2",
            "human_reach": HUMAN_REACH,
            "robot_reach": JAKA_K1_REACH,
            "ee_rot_offset": [-180.0, 0.0, 90.0],
        },
    },
    "left_hand": {
        "link_name": "lt",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "swgr": {
            "shoulder_link": "l2",
            "human_reach": HUMAN_REACH,
            "robot_reach": JAKA_K1_REACH,
            "ee_rot_offset": [-180.0, 0.0, -90.0],  # mirror of right_hand
        },
    },
}
```

`ee_rot_offset` is calibrated against `JAKA_K1_Q_INIT`: with a neutral controller
quaternion the composed target rotation must land on the tool frame the init pose already
holds. `tests/test_swgr_retargeting.py::EeRotOffsetTest` asserts exactly that, so changing
`JAKA_K1_Q_INIT` without re-deriving the offsets fails the suite.

There is no gripper, motion-tracker or head config: the JAKA K1 model has no gripper
joints, the IK is end-effector only, and there is no head to drive.
