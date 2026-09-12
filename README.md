# Bimanual-Teleoperation

PICO teleoperation of a dual-arm JAKA K1, written in Python. End-effector targets come
from Shoulder-Wrist Geometric Retargeting (SWGR); the inverse kinematics is a single
CasADi/IPOPT nonlinear program that solves both arms at once. Visualization is meshcat —
there is no physics backend and no hardware path.

## Overview

```
PICO headset + controllers + Swift trackers
        │  xrobotoolkit_sdk
        ▼
   XrClient ──────────────┐
        │                 │
        ▼                 ▼
  SWGR retargeting   grip / trigger state
        │
        ▼  one 4x4 world target per arm
  DualArmCasadiIK  (IPOPT over a pinocchio-casadi model, 14 variables)
        │
        ▼  4-tap weighted moving average
  CasadiTeleopController ── meshcat
```

`BaseTeleopController` owns everything up to the target: XR reads, body-tracking sanity
checks, SWGR mapping, engagement latching. `CasadiTeleopController` owns everything
downstream: the IK solver, the ramp-in on grip press, and the 120 Hz render loop.

## Installation

Tested on Ubuntu 22.04 and 24.04. Conda is required, not optional: the IK solver uses
the SX-only `pinocchio.casadi` bindings that ship in the conda-forge `pinocchio` build,
and the PyPI sdist does not provide them.

### 1. Install the XR service

Download and install [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service).
The service must be running and the XR device connected before starting a demo.

### 2. Clone the repository

```bash
git clone https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python.git
cd Bimanual-Teleoperation
```

### 3. Create an isolated Python environment

Do not reuse an environment that contains another robotics or Pinocchio project. Create a
Python 3.10 or 3.11 environment specifically for this repository:

```bash
conda deactivate
conda create -n bimanual-casadi python=3.10 -y
conda activate bimanual-casadi
which python   # must end with envs/bimanual-casadi/bin/python
```

### 4. Install the project

```bash
bash setup_conda.sh --install
```

The installer pulls `pinocchio` and `casadi` from conda-forge, builds the XRoboToolkit
Python binding under `dependencies/`, and installs this project in editable mode.

`setup_conda.sh --conda <name>` is the destructive variant: it removes an existing
environment of that name before recreating it, so the explicit `conda create` above is
safer when the environment already holds useful work.

### 5. Verify the installation

```bash
python -m pip check
python -c "import casadi, pinocchio, tyro, xrobotoolkit_sdk, xrobotoolkit_teleop; print('ok')"
python -m unittest tests.test_casadi_ik tests.test_jaka_k1_model tests.test_swgr_retargeting
```

The import check matters even when `pip check` succeeds: package metadata cannot detect a
native-library ABI mismatch. All 33 tests run offline (no headset) in under a second.
`tests/` has no `__init__.py`, so `unittest discover` will not find them — name the three
modules explicitly as above.

## Usage

Start XRoboToolkit PC Service, connect the headset, then:

```bash
cd ~/Bimanual-Teleoperation
conda activate bimanual-casadi
python scripts/simulation/teleop_jaka_k1_casadi.py
```

This initializes [`CasadiTeleopController`](xrobotoolkit_teleop/simulation/casadi_teleop_controller.py)
against the JAKA K1 URDF and opens meshcat in a browser. `Ctrl+C` stops it.

Both arms use SWGR, which derives its own scale from the human/robot reach ratio
(`0.768 / 0.58`), so there is no global scale knob. The model has no gripper joints: only
the grip buttons and the controller poses are used.

Flags worth knowing:

| Flag | Default | Effect |
|---|---|---|
| `--smooth-cost-weight` | `0.1` | Trades tracking accuracy for motion smoothness. Raise for calmer motion, lower for tighter tracking. |
| `--frequency` | `120.0` | Control loop rate in Hz. Upstream `xr_teleoperate` runs ~30 Hz; 120 Hz costs p95 3.13 ms of compute against an 8.33 ms budget. |
| `--engage-ramp-time` | `0.3` | Seconds spent blending the target in from the current EE pose when grip is pressed. `0` restores the unfiltered upstream jump. |

The script sets single-threaded BLAS environment variables before importing `numpy`;
multi-threaded BLAS makes each small IPOPT solve slower, not faster.

### Other scripts

```bash
# Per-phase latency probe (XR SDK reads, IK, meshcat) — needs the headset connected and
# both grips held; its docstring carries the offline stub-client reference numbers
python scripts/simulation/probe_jaka_latency.py --seconds 20

# The IK module on its own (20 tests: NLP build, warm start, bounds, smoothness)
python -m tests.test_casadi_ik
```

## Teleoperation Guide

### Tracking

The operator needs **full body tracking** — at least two Pico Swift trackers, calibrated,
with Full Body Tracking enabled in the Unity app. SWGR reads shoulder and wrist from the
body stream; orientation comes from the controller.

Without body tracking the arms hold their last target and the console prints a throttled
warning every 10 s naming what to check.

SWGR is *absolute*: it maps the shoulder→wrist vector scaled by `robot_reach /
human_reach`, anchored at the robot shoulder. Only that vector enters the map, so the VR
world origin cancels and the mapping needs no clutch or re-anchoring.

### Controller buttons

| Input | Action |
|---|---|
| **Left / right grip** | Hold to engage that arm; release to hold position. Re-engaging ramps the target in over `--engage-ramp-time` instead of jumping. |

Trigger, joystick, A/B and head-tracking inputs are not used: the JAKA K1 model has no
gripper joints, no mobile base, and no head.

## Dependencies

- [`xrobotoolkit_sdk`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind) — Python binding for XRoboToolkit PC Service, MIT License
- [`casadi`](https://github.com/casadi/casadi) — symbolic NLP modelling and the IPOPT interface, LGPL
- [`pinocchio`](https://github.com/stack-of-tasks/pinocchio) — rigid-body dynamics and forward kinematics (conda-forge build, provides `pinocchio.casadi`), BSD-2
- [`meshcat`](https://github.com/meshcat-dev/meshcat) — browser-based 3D visualization
- [`tyro`](https://github.com/brentyi/tyro) — CLI argument parsing

Upstream IK reference: [unitreerobotics/xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate).

## Repository layout

```
scripts/simulation/          entry points (teleop + latency probe)
tests/                       offline tests: IK, K1 model, SWGR retargeting
xrobotoolkit_teleop/
  common/                    BaseTeleopController, XrClient
  simulation/                DualArmCasadiIK, CasadiTeleopController
  utils/                     SWGR geometry, output filter, asset paths
assets/robot_model/          jaka_k1.urdf + meshes
```

For the retargeting math, the optimizer formulation and the measured numbers behind the
defaults, see [`teleop_details.md`](teleop_details.md).

## License

MIT — see [LICENSE](LICENSE).
