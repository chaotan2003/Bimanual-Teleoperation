# Bimanual Teleoperation

[中文说明](README.zh-CN.md)

PICO teleoperation of a dual-arm JAKA K1. The runtime path is intentionally small:
XR body/controller input -> SWGR target retargeting -> AEAC elbow preference ->
stacked J-PARSE velocity IK -> Viser visualization. There is no physics backend and no
hardware output path.

## Overview

```
PICO headset + controllers + Swift trackers
        |  xrobotoolkit_sdk
        v
     XrClient
        |
        v
  SWGR target pose per arm + grip engagement
        |
        v
  One Euro / Slerp pose filters
        |
        v
  ViserJparseController -> J-PARSE IK -> Viser
```

`BaseTeleopController` owns XR reads, body-tracking sanity checks, SWGR mapping and
engagement latching. `ViserJparseController` owns target filtering, AEAC/J-PARSE IK,
output interpolation and visualization.

## Installation

Tested on Ubuntu 22.04 and 24.04 with Python 3.10 or 3.11.

```bash
conda deactivate
conda create -n bimanual-teleop python=3.10 -y
conda activate bimanual-teleop
bash setup_conda.sh --install
```

The installer builds the XRoboToolkit Python binding under `dependencies/` and installs
this project in editable mode. `xrobotoolkit_sdk` is only used to acquire
PICO/XRoboToolkit input data; the kinematic model comes from an external git
dependency.

Verify:

```bash
python -m pip check
python -c "import viser, yourdfpy, xrobotoolkit_sdk, bimanual_teleop; print('ok')"
python -m unittest discover tests
```

## Usage

Start XRoboToolkit PC Service, connect the headset, enable full body tracking, then run:

```bash
conda activate bimanual-teleop
python scripts/simulation/teleop_jaka_k1.py
```

Both arms use SWGR: the operator shoulder->wrist vector is scaled by the robot/human
reach ratio (`0.768 / 0.58`) and anchored at the robot shoulder. Controller orientation
sets the tool orientation. Hold left/right grip to engage each arm; releasing grip holds
the last target.

Important flags:

| Flag | Default | Effect |
|---|---:|---|
| `--frequency` | `60.0` | Control loop rate in Hz. |
| `--engage-ramp-time` | `0.08` | Blend-in time when grip is pressed. |
| `--gamma` | `0.1` | J-PARSE singularity threshold. |
| `--max-joint-velocity` | `6.0` | Joint velocity clamp in rad/s. |
| `--position-filter-min-cutoff` | `12.0` | One Euro position smoothing baseline. |
| `--position-filter-beta` | `20.0` | Extra cutoff when targets move quickly. |
| `--orientation-filter-alpha` | `0.98` | Slerp orientation smoothing. |
| `--elbow-gain` | `0.0` | AEAC elbow-direction nullspace gain; use `0.5` to enable. |
| `--elbow-deadband` | `0.04` | Human elbow offset below this is ignored. |
| `--elbow-weight-max` | `1.0` | Maximum adaptive AEAC weight. |
| `--elbow-filter-alpha` | `0.35` | Low-pass factor for AEAC elbow direction/weight. |
| `--grip-on-threshold` | `0.9` | Engage grip above this value. |
| `--grip-off-threshold` | `0.75` | Release grip below this value to avoid analog input chatter. |
| `--enable-output-interpolation` | `False` | Interpolate displayed joint output between IK updates. |
| `--output-frequency` | `200.0` | Interpolation update rate when enabled. |
| `--visualization-frequency` | `30.0` | Viser refresh rate; control still runs at `--frequency`. |

For latency checks, keep `--elbow-gain 0`. If the Viser UI stalls, try
`--visualization-frequency 20`. Enable AEAC later with `--elbow-gain 0.5` only after
the primary end-effector tracking loop is fast enough.

## Repository Layout

```
scripts/simulation/teleop_jaka_k1.py   runtime entrypoint
bimanual_teleop/core/                  backend-agnostic teleop controller
bimanual_teleop/io/                    XR SDK wrapper
bimanual_teleop/retargeting/           SWGR + AEAC geometry
bimanual_teleop/ik/                    J-PARSE velocity IK
bimanual_teleop/runtime/               Viser runtime controller
bimanual_teleop/robots/                JAKA K1 config and asset paths
assets/robot_model/                    JAKA K1 URDF and meshes
tests/                                 offline geometry/filter/model tests
```

## License

MIT - see [LICENSE](LICENSE).
