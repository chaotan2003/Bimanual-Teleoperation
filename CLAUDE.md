# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Project Overview

*   **Purpose:** XR-based teleoperation framework for controlling robots through VR/AR devices in both simulation and hardware environments.
*   **Tech Stack:** Python 3.10+, MuJoCo (physics simulation), Placo (inverse kinematics), XRoboToolkit SDK (VR/AR interface), various robot hardware APIs.
*   **Architecture:** Controller-based design with base classes extended for specific simulation and hardware environments. Multi-threaded execution for real-time control, data logging, and camera streaming.

---

## 2. Build, Test, and Run Commands

*   **Environment Setup (Recommended):**
    ```sh
    # Setup conda environment (Ubuntu 22.04 tested)
    ./setup_conda.sh --conda xr-robotics
    conda activate xr-robotics
    ./setup_conda.sh --install
    ```

*   **Alternative System Installation:**
    ```sh
    # System-wide installation (if not using conda)
    ./setup.sh
    ```

*   **Common Simulation Scripts:**
    ```sh
    # Dual UR5e arms in MuJoCo
    python scripts/simulation/teleop_dual_ur5e_mujoco.py
    
    # ARX X7S with Placo visualization
    python scripts/simulation/teleop_x7s_placo.py
    
    # Shadow Hand dexterous manipulation
    python scripts/simulation/teleop_shadow_hand_mujoco.py
    
    # Flexiv Rizon4s robot
    python scripts/simulation/teleop_flexiv_rizon4s_mujoco.py
    ```

*   **Hardware Scripts:**
    ```sh
    # Dual UR5e with reset option
    python scripts/hardware/teleop_dual_ur5e_hardware.py --reset
    
    # ARX R5 robotic arm
    python scripts/hardware/teleop_arx_hardware.py
    
    # Galaxea R1 Lite humanoid
    python scripts/hardware/teleop_r1lite_hardware.py
    ```

*   **Code Formatting:**
    ```sh
    # Format all Python code (line length: 120)
    black .
    ```

---

## 3. Dependency Management

*   **Primary Tools:** Conda (recommended), pip, custom shell scripts
*   **Key Dependencies:** 
    - XRoboToolkit SDK (proprietary VR/AR interface)
    - placo==0.9.4 (inverse kinematics solver)
    - mujoco (physics simulation)
    - ur_rtde (Universal Robots interface)
    - dynamixel-sdk (servo control)
*   **Configuration Files:** `pyproject.toml`, `setup_conda.sh`, `setup.sh`

---

## 4. Coding Style and Conventions

*   **Code Style:** PEP 8
*   **Formatter:** `black` (line length: 120)
*   **Naming Conventions:** `snake_case` for variables and functions, `PascalCase` for classes.
*   **Key Architectural Patterns:** Controller-based architecture with inheritance hierarchy. `BaseTeleopController` provides common interface, extended by `HardwareTeleopController` for hardware-specific features (threading, logging, cameras).

### Core Architecture

The project uses a layered architecture with clear separation of concerns:

*   **Base Layer (`xrobotoolkit_teleop/common/`):**
    - `BaseTeleopController`: Abstract base class defining teleoperation interface
    - `HardwareTeleopController`: Adds hardware-specific features (threading, logging, camera support)

*   **Interface Layer (`xrobotoolkit_teleop/hardware/interface/`):**
    - Low-level wrappers for hardware communication (robots, grippers, cameras)
    - Each hardware type has its own interface class (e.g., `RobotiqGripperInterface`, `URRobotInterface`)

*   **Controller Layer:**
    - **Simulation**: `MujocoTeleopController`, `PlacoTeleopController`
    - **Hardware**: Robot-specific controllers (e.g., `ARXTeleopController`, `DualArmURController`)

*   **Control Flow:**
    1. XR device captures user input → `XrClient`
    2. Pose processing and IK target updates → `BaseTeleopController`
    3. Inverse kinematics solving → Placo solver
    4. Command execution → Hardware/simulation interfaces
    5. Concurrent: Data logging, camera streaming, visualization

---

## 5. Key Files and Directories

*   `pyproject.toml`: Project metadata and Python package dependencies.
*   `setup_conda.sh`, `setup.sh`: Installation scripts for conda/system environments.
*   `assets/`: Contains URDF models, meshes, and robot configurations.
*   `scripts/simulation/`: High-level teleoperation scripts for MuJoCo and Placo environments.
*   `scripts/hardware/`: High-level teleoperation scripts for physical robots.
*   `xrobotoolkit_teleop/`: Core Python package containing all teleoperation logic.
*   `xrobotoolkit_teleop/common/base_teleop_controller.py`: Abstract base class for all controllers.
*   `xrobotoolkit_teleop/common/base_hardware_teleop_controller.py`: Hardware base class with threading and logging.
*   `xrobotoolkit_teleop/hardware/interface/`: Low-level hardware communication wrappers.
*   `xrobotoolkit_teleop/simulation/`: Controllers for MuJoCo and Placo environments.
*   `dependencies/`: External dependencies cloned during setup (ARX SDK, XRoboToolkit).

---

## 6. Development Guidelines

*   **Adding New Robots:** Extend appropriate base controller class, implement required abstract methods, create interface wrapper in `hardware/interface/` if needed.
*   **Threading Model:** Hardware controllers use separate threads for IK computation, robot control, data logging, and camera streaming.
*   **Configuration:** Robot-specific settings defined as dictionaries in controller classes (joint limits, IK constraints, camera parameters).
*   **Asset Paths:** Always use absolute paths for URDF loading, leveraging `path_utils` module for cross-platform compatibility.
*   **Error Handling:** Hardware controllers implement safety checks and manipulability constraints to prevent dangerous robot movements.

---

## 7. Change-Speed Policy (overrides the global skills)

These rules win over any always-on skill (`test-driven-development`, `doubt-driven-development`, `code-review-and-quality`, `constraint-driven-development`) for this repo:

*   **Small change = no new test.** If a change touches ≤ 2 files and does not add a new algorithm, do NOT write a test and do NOT run the suite unless the user asks. Just make the edit and state what it changes.
*   **When a test IS needed, run one test only.** `python -m unittest tests.test_casadi_ik -k <name>` (or the matching single file). Never run all of `tests/` — those load pinocchio/CasADi/MuJoCo models and are minutes, not seconds.
*   **No subagent review passes** on edits. No fresh-context adversarial review, no multi-axis self-review, no CONSTRAINTS.md, unless explicitly requested.
*   **Skip the knowledge-graph MCP for known paths.** If the user names a file, `read` it directly. `search_graph` / `trace_path` / `check_index_coverage` are for "where is X" and "who calls X" questions only, not a pre-flight checklist on every edit.
*   **Verification = one cheap smoke check.** For pure-Python edits: `python -c "import xrobotoolkit_teleop..."` or a syntax check. Hardware/sim behavior is verified on the robot, not in CI (this repo has none).
