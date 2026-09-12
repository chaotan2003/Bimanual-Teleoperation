# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Project Overview

*   **Purpose:** XR (PICO) teleoperation of a dual-arm JAKA K1 in meshcat visualization. End-effector targets come from Shoulder-Wrist Geometric Retargeting (SWGR); inverse kinematics is a single CasADi/IPOPT NLP solving both arms at once.
*   **Tech Stack:** Python 3.10, CasADi 3.6 + IPOPT, Pinocchio 3.1 (conda-forge build, for the SX-only `pinocchio.casadi` bindings), meshcat, tyro, XRoboToolkit SDK.
*   **No physics backend and no hardware path.** The "robot" is the kinematic model: its state is whatever the IK returned. MuJoCo, Placo and all robot-hardware controllers were removed.
*   **Architecture:** Two-layer. `BaseTeleopController` owns XR reads → SWGR → one 4x4 world target per manipulator. `CasadiTeleopController` owns the IK solver, the engage ramp and the render loop. Single-threaded.

---

## 2. Build, Test, and Run Commands

*   **Environment setup (conda is required, not optional):**
    ```sh
    conda create -n bimanual-casadi python=3.10 -y
    conda activate bimanual-casadi
    ./setup_conda.sh --install   # conda-forge pinocchio+casadi, XR SDK binding, editable install
    ```
    `setup_conda.sh --conda <name>` is the destructive variant (removes an existing env first).

*   **Run the teleop demo:**
    ```sh
    python scripts/simulation/teleop_jaka_k1_casadi.py
    python scripts/simulation/teleop_jaka_k1_casadi.py --help   # tyro flags, no headset needed
    ```

*   **Latency probe (stub XR client, no headset):**
    ```sh
    python scripts/simulation/probe_jaka_latency.py
    ```

*   **Tests (all offline, ~0.4 s total):**
    ```sh
    python -m unittest tests.test_casadi_ik tests.test_jaka_k1_model tests.test_swgr_retargeting
    ```

*   **Code formatting:**
    ```sh
    black .   # line length 120
    ```

---

## 3. Dependency Management

*   **Primary tools:** Conda + pip/uv, driven by `setup_conda.sh`.
*   **Key dependencies:** `casadi`, `pinocchio` (**conda-forge only** — the PyPI sdist lacks the casadi bindings), `numpy`, `meshcat`, `tyro`, XRoboToolkit SDK (built from `dependencies/XRoboToolkit-PC-Service-Pybind`).
*   **`pinocchio` is deliberately absent from `pyproject.toml`** — a comment there explains why. Do not "fix" that by adding it.
*   **Configuration files:** `pyproject.toml`, `setup_conda.sh`.

---

## 4. Coding Style and Conventions

*   **Code style:** PEP 8, formatted with `black` (line length 120).
*   **Naming:** `snake_case` variables/functions, `PascalCase` classes.
*   **Key pattern:** the base class defines the contract, one subclass implements it.
    *   `BaseTeleopController._update_ik()` reads XR, retargets via SWGR, writes `self.effector_task[name].T_world_frame` (or `.target_world` in `"position"` mode), then calls `self._solve_ik()`.
    *   A subclass must provide `self.kinematics` exposing `state.q`, `update_kinematics()` and `frame(name)`, plus `_robot_setup`, `_solver_setup`, `_solve_ik`, `_update_robot_state`, `_send_command`, `_get_link_pose`, `run`.

### Control flow

1.  XR device → `XrClient` (body tracking for shoulder/wrist, controller pose for orientation, grip for engagement)
2.  `BaseTeleopController._update_swgr_target` → absolute 4x4 world target per arm
3.  `CasadiTeleopController._ramp_target` → geodesic SE(3) blend-in on grip press
4.  `DualArmCasadiIK.solve_ik` → IPOPT over 14 variables (l-j1..l-j7, r-j1..r-j7), warm-started
5.  `WeightedMovingFilter` → 4-tap moving average `(0.4, 0.3, 0.2, 0.1)`
6.  meshcat display, on an absolute-deadline 120 Hz loop

---

## 5. Key Files and Directories

*   `pyproject.toml`: metadata + the 4 pip-installable dependencies.
*   `setup_conda.sh`: the only installation script.
*   `assets/robot_model/`: `jaka_k1.urdf` + meshes. The only asset tree left.
*   `scripts/simulation/teleop_jaka_k1_casadi.py`: entry point; holds `JAKA_K1_Q_INIT`, `HUMAN_REACH`, `JAKA_K1_REACH`, `JAKA_K1_MANIPULATOR_CONFIG`.
*   `scripts/simulation/probe_jaka_latency.py`: per-phase latency probe with a stub XR client.
*   `xrobotoolkit_teleop/common/base_teleop_controller.py`: XR → SWGR → task targets.
*   `xrobotoolkit_teleop/common/xr_client.py`: XRoboToolkit SDK wrapper.
*   `xrobotoolkit_teleop/simulation/casadi_arm_ik.py`: `DualArmCasadiIK` — NLP construction, IPOPT options, param packing.
*   `xrobotoolkit_teleop/simulation/casadi_teleop_controller.py`: `CasadiTeleopController`, `_Task`, `_Kinematics`, `blend_se3`.
*   `xrobotoolkit_teleop/utils/geometry.py`: `R_HEADSET_TO_WORLD`, `swgr_ee_position`.
*   `xrobotoolkit_teleop/utils/weighted_moving_filter.py`: output FIR.
*   `xrobotoolkit_teleop/utils/path_utils.py`: `ASSET_PATH`.
*   `tests/`: three offline test modules. `test_swgr_retargeting.py` instantiates `BaseTeleopController` via `__new__` (never `__init__`), so it must stub every abstract method.
*   `teleop_details.md`: the retargeting math, optimizer formulation and measured numbers.

---

## 6. Development Guidelines

*   **Invariants that must not regress** (each has a test):
    *   `DualArmCasadiIK.pack_params` flattens 4x4 targets **column-major** (`order='F'`).
    *   `q_init` must lie inside the joint limits — `j4=0` is excluded on the K1, so all-zeros is invalid.
    *   On IK failure, hold the previous `q` and warn; never raise into the control loop.
    *   `teleop_jaka_k1_casadi.py` sets single-threaded BLAS env vars **before** importing `numpy`.
    *   SWGR is absolute: no reference pose captured at engagement, so no drift and no clutch.
    *   `ee_rot_offset` is calibrated against `JAKA_K1_Q_INIT`; changing one without the other fails `EeRotOffsetTest`.
*   **The NLP uses `casadi.nlpsol`, not `Opti`/`MX`**, because the conda-forge Pinocchio casadi bindings only accept `SX`.
*   **Real-time loop:** `run()` uses absolute deadlines (`deadline += dt`), resyncs instead of bursting when it falls behind, and calls `gc.freeze()` after the model graph is built.
*   **Asset paths:** always absolute, via `path_utils`.
*   **Adding a robot:** supply a URDF, a `manipulator_config` with an `swgr` block per arm, and a `q_init` inside the limits. `tests/test_swgr_retargeting.py::JakaK1ReachTest` shows how to assert `robot_reach` against forward kinematics; `EeRotOffsetTest` shows how to pin `ee_rot_offset` to the init pose.

---

## 7. Change-Speed Policy (overrides the global skills)

These rules win over any always-on skill (`test-driven-development`, `doubt-driven-development`, `code-review-and-quality`, `constraint-driven-development`) for this repo:

*   **Small change = no new test.** If a change touches ≤ 2 files and does not add a new algorithm, do NOT write a test and do NOT run the suite unless the user asks. Just make the edit and state what it changes.
*   **When a test IS needed, run one test only.** `python -m unittest tests.test_casadi_ik -k <name>` (or the matching single file). The suite is three modules / 33 tests and runs in under a second, but a single test is still the fastest feedback loop.
*   **No subagent review passes** on edits. No fresh-context adversarial review, no multi-axis self-review, no CONSTRAINTS.md, unless explicitly requested.
*   **Skip the knowledge-graph MCP for known paths.** If the user names a file, `read` it directly. `search_graph` / `trace_path` / `check_index_coverage` are for "where is X" and "who calls X" questions only, not a pre-flight checklist on every edit.
*   **Verification = one cheap smoke check.** For pure-Python edits: `python -c "import xrobotoolkit_teleop..."` or a syntax check. Hardware/sim behavior is verified on the robot, not in CI (this repo has none).
